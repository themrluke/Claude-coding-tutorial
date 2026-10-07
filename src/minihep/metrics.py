"""Week 6: tracking performance metrics.

Two families, matching what you see in hepattn:

* **After Hungarian matching** (``matched_efficiency``), the numbers logged during training,
  e.g. ``val/p0.5_eff`` in Comet. Copied from TrackMLTracker.log_custom_metrics in
  experiments/trackml/run_tracking.py.
* **Without matching** (``double_majority``), the physics evaluation: TrackML-style
  "double majority" matching between reconstructed tracks and truth particles, as in
  experiments/trackml/eval/. These are the numbers in the paper's tables.

Definitions, for one predicted track and one truth particle:
  shared = hits on both, n_track = hits on the track, n_particle = hits of the particle
  track purity  = shared / n_track      ("most of the track's hits are the particle's")
  hit efficiency = shared / n_particle  ("most of the particle's hits are on the track")
"""

import torch
from torch import Tensor


def matched_efficiency(
    pred_valid: Tensor, pred_masks: Tensor, true_valid: Tensor, true_masks: Tensor, working_point: float = 0.5
) -> dict[str, Tensor]:
    """Metrics on *matched* outputs: prediction i is compared with target i.

    Shapes: pred_valid/true_valid (B, Q) bool, pred_masks/true_masks (B, Q, N) bool.

    - Zero the masks of invalid slots: ``pred_masks & pred_valid[..., None]`` and likewise for truth.
    - ``hit_tp = (pred & true).sum(-1)``, ``hit_p = pred.sum(-1)``, ``hit_t = true.sum(-1)``.
    - A particle is *efficient* if both slots are valid and ``hit_tp / hit_t >= working_point``.
    - A track is *pure* if both slots are valid and ``hit_tp / hit_p >= working_point``.
    - ``eff = efficient.sum(-1) / true_valid.sum(-1)``, ``pur = pure.sum(-1) / pred_valid.sum(-1)``,
      each averaged over the batch with ``nanmean`` (an event with no predicted tracks gives 0/0).
    Return ``{"eff": ..., "pur": ...}`` as 0-dim tensors.
    """
    # TODO(week06): follow log_custom_metrics; beware 0/0 for empty slots (it is fine, the both-valid mask removes them)
    raise NotImplementedError("week06 exercise (metrics.py)")


def double_majority(pred_valid: Tensor, pred_masks: Tensor, true_valid: Tensor, true_masks: Tensor) -> dict[str, float]:
    """Matching-free evaluation for ONE event (no batch dim): pred (Q,), (Q, N); true (T,), (T, N).

    For every valid predicted track with at least one hit:
      - find the truth particle sharing the most hits with it (``shared = pred @ true.T`` as ints),
      - the track is **matched** to that particle if ``shared / n_track > 0.5`` AND
        ``shared / n_particle > 0.5`` (strictly more than half: the "double majority").
    Then:
      - ``eff``: fraction of valid particles that at least one track is matched to
      - ``fake_rate``: fraction of valid tracks matched to no particle
      - ``duplicate_rate``: fraction of valid tracks that are matched, but to a particle that
        another, earlier-indexed track is also matched to
    Return Python floats. If there are no valid tracks, fake_rate and duplicate_rate are 0.
    """
    # TODO(week06): shared-hit matrix, best particle per track, the two majority conditions, then count
    raise NotImplementedError("week06 exercise (metrics.py)")
