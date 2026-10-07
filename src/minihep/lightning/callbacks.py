"""Week 7 (and 8): Lightning callbacks.

Mirror of src/hepattn/callbacks/. A callback is an object with hook methods that the
Trainer calls at fixed moments: ``setup``, ``on_train_start``, ``on_validation_epoch_end``,
``on_test_batch_end``, ``teardown``... It lets you bolt behaviour onto training (saving
predictions, compiling the model, timing, logging extra files) without touching the model.
"""

import socket
from pathlib import Path

import h5py
import lightning
import numpy as np
import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    """Save every epoch's checkpoint as ``{run_dir}/ckpts/epoch=007-val_loss=0.12345.ckpt`` (provided).

    hepattn keeps all of them (``save_top_k=-1``) and puts the val loss in the file name, so
    ``minihep.cli.get_best_epoch`` can pick the best one later by parsing names.
    """

    def __init__(self, monitor: str = "val/loss", **kwargs):
        filename = "epoch={epoch:03d}-" + monitor.replace("/", "_") + "={" + monitor + ":.5f}"
        super().__init__(save_top_k=-1, monitor=monitor, filename=filename, auto_insert_metric_name=False, **kwargs)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage == "fit" and not trainer.fast_dev_run:
            self.dirpath = str(Path(trainer.default_root_dir) / "ckpts")


class SaveMetadata(Callback):
    """Write ``metadata.yaml`` next to the config at the start of training (provided). hepattn: SaveConfig."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero or trainer.fast_dev_run:
            return
        meta = {
            "num_train": len(trainer.datamodule.train_dataloader().dataset),
            "trainable_params": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            "torch_version": str(torch.__version__),
            "lightning_version": str(lightning.__version__),
            "cuda_available": torch.cuda.is_available(),
            "hostname": socket.gethostname(),
            "log_url": getattr(getattr(trainer.logger, "experiment", None), "url", None) if trainer.logger else None,
        }
        out = Path(trainer.default_root_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """bf16 has no numpy equivalent, so upcast it first (provided; hepattn: tensor_to_numpy)."""
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


class PredictionWriter(Callback):
    def __init__(self, write_targets: bool = True, write_outputs: bool = False, output_path: str | None = None):
        """Write test-set predictions to HDF5, one group per event:

            /{event_name}/preds/final/{task_name}/{pred_key}       e.g. /event000950001/preds/final/hit_filter/hit_on_valid_particle_prob
            /{event_name}/outputs/final/{task_name}/{output_key}   (if write_outputs)
            /{event_name}/targets/{target_key}                     (if write_targets)
            attribute sample_id on the event group

        Every array keeps its batch dim of 1, like hepattn's files (readers index ``[0]``).

        Output path: ``output_path`` if given, else ``{checkpoint dir}/{checkpoint stem}__{test dir name}.h5``,
        so the filter's predictions for train/val/test land next to the checkpoint that made them,
        with different names (cf. your "Hit filter training" vault note: hepattn writes ``__test.h5``
        every time and you had to rename the files per split).
        """
        super().__init__()
        self.write_targets = write_targets
        self.write_outputs = write_outputs
        self.output_path = output_path
        self.file: h5py.File | None = None

    def resolve_path(self, trainer: Trainer) -> Path:
        """Provided. ``trainer.ckpt_path`` is set by Lightning when testing from a checkpoint."""
        if self.output_path is not None:
            return Path(self.output_path)
        split = Path(trainer.datamodule.test_dir).stem
        if trainer.ckpt_path:
            ckpt = Path(trainer.ckpt_path)
            return ckpt.parent / f"{ckpt.stem}__{split}.h5"
        return Path(trainer.default_root_dir) / f"predictions__{split}.h5"

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # TODO(week07): resolve the path, make its parent directory, open the file for writing
        raise NotImplementedError("week07 exercise (callbacks.py)")

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """``outputs`` here is whatever ``test_step`` returned: ``(model_outputs, preds, targets)``.

        Name the group after the event: ``trainer.datamodule.test_dataset.event_names[batch_idx]``
        (the test loader is not shuffled, so batch_idx is the event index). Write
        ``preds["final"]`` (and optionally ``model_outputs["final"]``) task by task, and the targets.
        Use ``group.create_dataset(key, data=to_numpy(tensor), compression="lzf")``.
        """
        # TODO(week07): create the event group, set the sample_id attribute, write the requested dicts
        raise NotImplementedError("week07 exercise (callbacks.py)")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # TODO(week07): close the file (and print where it is)
        raise NotImplementedError("week07 exercise (callbacks.py)")


class Compile(Callback):
    def __init__(self, dynamic: bool = True, mode: str | None = None):
        """Week 8: ``torch.compile`` the encoder and decoder when training (or testing) starts.

        Why in a callback and not in ``__init__``? Lightning runs a "sanity check" (two validation
        batches) before training. Compiling after it means the first graph is traced in training mode,
        avoiding an immediate recompile (see the comment in hepattn/callbacks/compile.py).
        """
        super().__init__()
        self.dynamic = dynamic
        self.mode = mode
        self.compiled = False

    def _compile(self, pl_module: LightningModule) -> None:
        """Call ``submodule.compile(dynamic=..., mode=...)`` (in place) on ``pl_module.model.encoder`` and
        ``pl_module.model.decoder`` if they exist and are not None. Only once."""
        # TODO(week08): guard with self.compiled, compile the two submodules in place
        raise NotImplementedError("week08 exercise (callbacks.py)")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._compile(pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._compile(pl_module)
