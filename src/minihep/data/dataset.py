"""Week 2: turn events on disk into the dicts of tensors the model eats.

This is a small copy of ``hepattn.experiments.trackml.data.TrackMLDataset``. Keep that
file open next to this one: the method names, the cuts and the output keys are the same.

One call to ``dataset[i]`` returns ``(inputs, targets)``, two flat dicts of tensors that
all carry a leading batch dimension of 1:

    inputs["hit_valid"]           bool    (1, N)       every real hit is True
    inputs["hit_x"], ...          float32 (1, N)       one entry per requested field
    targets["hit_valid"]          bool    (1, N)
    targets["particle_valid"]     bool    (1, P)       P = event_max_num_particles, padded with False
    targets["particle_hit_valid"] bool    (1, P, N)    [p, n] is True if hit n belongs to particle p
    targets["hit_on_valid_particle"], targets["hit_is_first"]   bool (1, N)
    targets["particle_pt"], ...   float32 (1, P)       NaN in padded slots
    targets["sample_id"]          int64   (1,)

Because every item already has its batch dimension, the DataLoader is built with
``batch_size=None`` (no collation), exactly like hepattn: events have different numbers
of hits, so they cannot simply be stacked. ``pad_collate`` at the bottom shows the
alternative: pad to the longest event and mark the padding with ``hit_valid = False``.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from minihep.data.prep import read_event_hdf5

# The toy detector writes millimetres. Networks like inputs of order 1, so convert to metres.
# (hepattn: HIT_COORDINATE_SCALE = 0.01 for TrackML.)
HIT_COORDINATE_SCALE = 0.001
PARTICLE_PAD_ID = -999  # never equal to a real particle id, nor to the noise id 0


class ToyTrackingDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict[str, list[str]],
        targets: dict[str, list[str]],
        num_events: int = -1,
        hit_volume_ids: list[int] | None = None,
        particle_min_pt: float = 1.0,
        particle_max_abs_eta: float = 2.5,
        particle_min_num_hits: int = 3,
        event_max_num_particles: int = 50,
        hit_eval_path: str | None = None,
        hit_filter_threshold: float = 0.5,
    ):
        """Args mirror TrackMLDataset.

        Args:
            dirpath: a directory of ``event*-hits.parquet`` / ``event*-parts.parquet`` files,
                or a single ``.h5`` file written by ``prep.parquet_dir_to_hdf5``.
            inputs: e.g. ``{"hit": ["x", "y", "z", "r", "eta", "phi"]}``.
            targets: e.g. ``{"hit": ["on_valid_particle", "is_first"], "particle": ["pt", "eta", "phi"]}``.
            num_events: how many events to use, -1 for all.
            hit_volume_ids: keep only hits in these detector volumes (None keeps all).
            particle_min_pt, particle_max_abs_eta, particle_min_num_hits: a particle is
                "reconstructable" (a valid target) only if it passes all three.
            event_max_num_particles: pad (or truncate) the particle axis to this length.
            hit_eval_path: optional HDF5 file of hit-filter predictions (week 7); hits whose
                predicted probability is below ``hit_filter_threshold`` are dropped.
        """
        super().__init__()
        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.hit_volume_ids = hit_volume_ids
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_hits = particle_min_num_hits
        self.event_max_num_particles = event_max_num_particles
        self.hit_eval_path = hit_eval_path
        self.hit_filter_threshold = hit_filter_threshold

        self.is_hdf5 = self.dirpath.suffix == ".h5"
        if self.is_hdf5:
            with h5py.File(self.dirpath, "r") as f:
                names = sorted(f.keys())
        else:
            names = sorted(p.name.removesuffix("-parts.parquet") for p in self.dirpath.glob("event*-parts.parquet"))
        if not names:
            raise FileNotFoundError(f"No events found in {self.dirpath}")
        if num_events > len(names):
            raise ValueError(f"Requested {num_events} events, but only {len(names)} are available in {self.dirpath}")
        self.event_names = names if num_events < 0 else names[:num_events]
        self.sample_ids = [int(name.removeprefix("event")) for name in self.event_names]

        # Opened lazily, see _h5_file(). Never open an HDF5 file in __init__ and keep it:
        # DataLoader workers are forked copies of this object, and an h5py file handle
        # shared across processes is not safe to read from.
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.event_names)

    def _h5_file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dirpath, "r")
        return self._h5

    def read_raw_event(self, idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read one event exactly as stored on disk (provided)."""
        name = self.event_names[idx]
        if self.is_hdf5:
            return read_event_hdf5(self._h5_file(), name)
        hits = pd.read_parquet(self.dirpath / f"{name}-hits.parquet")
        particles = pd.read_parquet(self.dirpath / f"{name}-parts.parquet")
        return hits, particles

    def load_event(self, idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read an event, add derived columns, apply the cuts. Returns ``(hits, particles)``.

        Follow ``TrackMLDataset.load_event`` step by step:

        1. ``read_raw_event``; if ``hit_volume_ids`` is set keep only hits in those volumes
           (``Series.isin``).
        2. Scale x, y, z by HIT_COORDINATE_SCALE (mm -> m).
        3. Add hit columns: ``r = sqrt(x² + y²)``, ``s = sqrt(x² + y² + z²)``,
           ``theta = arccos(z / s)``, ``phi = arctan2(y, x)``, ``eta = -log(tan(theta / 2))``.
        4. Add particle columns: ``pt``, ``p``, ``eta = arctanh(pz / p)``, ``phi = arctan2(py, px)``,
           ``qopt = q / pt``.
        5. ``hits = self._apply_hit_filter(hits, idx)`` (a no-op until week 7).
        6. Particle cuts: ``pt > particle_min_pt``, ``|eta| < particle_max_abs_eta`` and at least
           ``particle_min_num_hits`` hits *among the hits that are left*
           (``hits["particle_id"].value_counts()``). Never count noise (particle_id 0).
        7. ``hits["on_valid_particle"]``: True if the hit's particle survived the cuts.
        8. ``hits["is_first"]``: True for the innermost (smallest r) hit of each surviving
           particle. ``groupby("particle_id")["r"].idxmin()`` gives the row labels.

        Return copies, not views, so later edits cannot touch cached data
        (``reset_index(drop=True)`` after filtering keeps row labels 0..N-1).
        """
        # TODO(week02): implement the eight steps above
        raise NotImplementedError("week02 exercise (dataset.py)")

    def _apply_hit_filter(self, hits: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Week 7: drop the hits that the hit filter scored below ``hit_filter_threshold``.

        The filter's predictions live in the HDF5 file written by
        ``minihep.lightning.callbacks.PredictionWriter`` at
        ``/{event_name}/preds/final/hit_filter/hit_on_valid_particle_prob`` with shape (1, N),
        in the same hit order as ``hits`` (both come from the same, unshuffled event).
        If ``hit_eval_path`` is None, return ``hits`` unchanged. hepattn equivalent: the
        ``if self.hit_eval_path:`` block in TrackMLDataset.load_event.
        """
        if self.hit_eval_path is None:
            return hits
        # TODO(week07): open hit_eval_path, read the probabilities for this event, keep hits >= threshold
        raise NotImplementedError("week07 exercise (dataset.py)")

    def __getitem__(self, idx: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Build the ``(inputs, targets)`` dicts described in the module docstring.

        Steps (mirroring TrackMLDataset.__getitem__):

        1. ``hits, particles = self.load_event(idx)``.
        2. For each ``input_name, fields`` in ``self.inputs``: ``inputs[f"{input_name}_valid"]`` all True,
           and one float32 tensor per field. Every tensor gets a leading batch dim of 1
           (``.unsqueeze(0)``). Also copy the valid mask into ``targets``.
        3. Truncate particles to ``event_max_num_particles`` if there are more.
        4. ``targets["particle_valid"]``: True for real particles, False for padding.
        5. ``targets["particle_hit_valid"]``: pad the particle ids with PARTICLE_PAD_ID, then
           compare ``particle_ids[:, None] == hit_particle_ids[None, :]`` (broadcasting!).
        6. For each field in ``self.targets.get("hit", [])``: ``targets[f"hit_{field}"]`` (bool).
        7. For each field in ``self.targets.get("particle", [])``: a float32 tensor of length
           ``event_max_num_particles`` filled with NaN, real values at the front. The NaNs
           are a tripwire: if a loss ever reads a padded slot, it becomes NaN and you notice.
        8. ``targets["sample_id"] = torch.tensor([sample_id])``.
        """
        # TODO(week02): implement the eight steps above
        raise NotImplementedError("week02 exercise (dataset.py)")


def pad_collate(batch: list[tuple[dict[str, Tensor], dict[str, Tensor]]], input_name: str = "hit") -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Stack several events into one batch by padding the hit axis.

    Each element of ``batch`` is a ``(inputs, targets)`` pair from ``ToyTrackingDataset``
    (every tensor has a leading dim of 1). Return one ``(inputs, targets)`` pair where:

      * tensors whose *last* axis is the hit axis (``{input_name}_*`` keys, and
        ``particle_hit_valid``) are right-padded to the longest event: with False for bool
        tensors and 0 for float tensors, then concatenated along dim 0;
      * every other tensor (``particle_valid``, ``particle_pt``, ``sample_id``) already has the
        same shape in every event and is just concatenated along dim 0.

    After padding, ``inputs[f"{input_name}_valid"]`` is False exactly on the padded slots. This
    is the ``kv_mask`` that attention layers receive (week 4), and the thing
    flash-attention's ``varlen`` interface lets you avoid (week 5).

    Hint: ``torch.nn.functional.pad(t, (0, n_pad))`` pads the last dimension on the right.
    """
    # TODO(week02): find the longest hit axis, then pad (if hit-axis tensor) and torch.cat every key
    raise NotImplementedError("week02 exercise (dataset.py)")
