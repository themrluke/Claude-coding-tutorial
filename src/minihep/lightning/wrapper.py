"""Week 7: wrap a model in a LightningModule.

Mirror of src/hepattn/models/wrapper.py (ModelWrapper) and the experiment subclasses in
experiments/trackml/run_tracking.py (TrackMLTracker) and run_filtering.py (TrackMLFilter).

Lightning owns the loop you wrote in week 3. You provide the pieces it calls:

    training_step(batch, batch_idx) -> loss          Lightning does zero_grad/backward/clip/step
    validation_step(batch, batch_idx)                 run under no_grad + eval() automatically
    test_step(batch, batch_idx) -> whatever callbacks need (the PredictionWriter reads it)
    configure_optimizers() -> optimiser (+ scheduler)

``self.log(name, value)`` sends a number to every logger (CSV, Comet...), averaged over the
epoch for validation. Names follow hepattn: ``f"{stage}/{layer}_{task}_{loss}"``,
``f"{stage}/{layer}_loss"``, ``f"{stage}/loss"``.
"""

from typing import Literal

import torch
from lightning import LightningModule
from torch import Tensor, nn

from minihep.metrics import matched_efficiency

try:
    from lion_pytorch import Lion
except ImportError:  # pragma: no cover
    Lion = None


class ModelWrapper(LightningModule):
    def __init__(self, name: str, model: nn.Module, lrs_config: dict, optimizer: Literal["AdamW", "Lion"] = "AdamW"):
        """``lrs_config`` has the keys initial, max, end, pct_start, weight_decay and optionally skip_scheduler.

        These type hints are not decoration: LightningCLI reads them to know what the YAML may contain
        and to check it. ``optimizer: Literal["AdamW", "Lion"]`` means a typo like ``Adam`` is rejected
        when the config is parsed, before any training starts.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.name = name
        self.model = model
        self.lrs_config = lrs_config
        self.optimizer = optimizer

    def forward(self, inputs: dict[str, Tensor]) -> dict:
        return self.model(inputs)

    def aggregate_losses(self, losses: dict[str, dict[str, dict[str, Tensor]]], stage: str, batch_size: int = 1) -> Tensor:
        """Sum every loss into one scalar and log the pieces.

        For every (layer, task, loss_name, value): log ``f"{stage}/{layer}_{task}_{loss_name}"``. Also log
        each layer's sum as ``f"{stage}/{layer}_loss"`` and the grand total as ``f"{stage}/loss"``.
        Pass ``batch_size=batch_size`` to every ``self.log`` (Lightning cannot infer it from our dicts).
        Return the total.
        """
        # TODO(week07): nested loops, self.log each piece, accumulate layer and total sums
        raise NotImplementedError("week07 exercise (wrapper.py)")

    def log_metrics(self, preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        """Log ``task.metrics`` for the final layer as ``f"{stage}/final_{task.name}_{metric}"``, then call
        ``self.log_custom_metrics(preds, targets, stage)`` (a hook subclasses override). Provided."""
        for task in self.model.tasks:
            if task.name in preds["final"]:
                for k, v in task.metrics(preds["final"][task.name], targets).items():
                    self.log(f"{stage}/final_{task.name}_{k}", v, batch_size=1)
        self.log_custom_metrics(preds, targets, stage)

    def log_custom_metrics(self, preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        """Override in subclasses for experiment-specific metrics."""

    def training_step(self, batch: tuple[dict, dict], batch_idx: int) -> Tensor:
        """forward -> model.loss -> aggregate_losses(stage="train"). Every ``trainer.log_every_n_steps``
        batches also predict and log_metrics (prediction costs time, so not every step).
        Return the total loss: Lightning calls backward on it."""
        # TODO(week07): unpack, forward, loss, aggregate, occasional metrics, return
        raise NotImplementedError("week07 exercise (wrapper.py)")

    def validation_step(self, batch: tuple[dict, dict], batch_idx: int) -> Tensor:
        """Same as training_step with stage "val", and always log metrics."""
        # TODO(week07): forward, loss, aggregate, metrics
        raise NotImplementedError("week07 exercise (wrapper.py)")

    def test_step(self, batch: tuple[dict, dict], batch_idx: int) -> tuple[dict, dict, dict]:
        """Return ``(outputs, preds, targets)`` for the PredictionWriter. Call ``model.loss`` first even
        though we do not need the loss: it runs the matching and permutes the outputs, so the written
        predictions line up with the targets (hepattn does the same)."""
        # TODO(week07): forward, loss (for the matching), predict, return the three dicts
        raise NotImplementedError("week07 exercise (wrapper.py)")

    def configure_optimizers(self):
        """AdamW or Lion, with OneCycleLR stepped every batch (``{"scheduler": sch, "interval": "step"}``),
        exactly as hepattn's ModelWrapper.configure_optimizers. The total number of steps is
        ``self.trainer.estimated_stepping_batches``. If ``lrs_config.get("skip_scheduler")`` return just the optimiser.
        """
        # TODO(week07): pick the optimiser class, build it, build OneCycleLR like week 3, return ([opt], [sch_dict])
        raise NotImplementedError("week07 exercise (wrapper.py)")


class Filter(ModelWrapper):
    """Hit filter: the generic task metrics are all we log (provided)."""


class Tracker(ModelWrapper):
    def log_custom_metrics(self, preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        """Log ``f"{stage}/p{wp}_eff"`` and ``f"{stage}/p{wp}_pur"`` for wp in (0.5, 0.75, 1.0) using
        ``matched_efficiency`` on the final layer, plus ``f"{stage}/num_tracks"`` (mean number of predicted
        valid tracks). Task names "track_valid" / "track_hit_valid" as in week 6.
        (hepattn: TrackMLTracker.log_custom_metrics.)
        """
        # TODO(week07): pull the final predictions, loop over working points, self.log
        raise NotImplementedError("week07 exercise (wrapper.py)")
