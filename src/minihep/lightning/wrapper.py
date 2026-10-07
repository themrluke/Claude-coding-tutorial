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
        # >>> week07: nested loops, self.log each piece, accumulate layer and total sums
        total = torch.zeros((), device=self.device)
        for layer_name, layer_losses in losses.items():
            layer_total = torch.zeros((), device=self.device)
            for task_name, task_losses in layer_losses.items():
                for loss_name, value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", value, batch_size=batch_size)
                    layer_total = layer_total + value
            self.log(f"{stage}/{layer_name}_loss", layer_total, batch_size=batch_size)
            total = total + layer_total
        self.log(f"{stage}/loss", total, batch_size=batch_size, prog_bar=True)
        return total
        # <<< week07

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
        # >>> week07: unpack, forward, loss, aggregate, occasional metrics, return
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        total = self.aggregate_losses(losses, stage="train")
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log_metrics(self.model.predict(outputs), targets, "train")
        return total
        # <<< week07

    def validation_step(self, batch: tuple[dict, dict], batch_idx: int) -> Tensor:
        """Same as training_step with stage "val", and always log metrics."""
        # >>> week07: forward, loss, aggregate, metrics
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        total = self.aggregate_losses(losses, stage="val")
        self.log_metrics(self.model.predict(outputs), targets, "val")
        return total
        # <<< week07

    def test_step(self, batch: tuple[dict, dict], batch_idx: int) -> tuple[dict, dict, dict]:
        """Return ``(outputs, preds, targets)`` for the PredictionWriter. Call ``model.loss`` first even
        though we do not need the loss: it runs the matching and permutes the outputs, so the written
        predictions line up with the targets (hepattn does the same)."""
        # >>> week07: forward, loss (for the matching), predict, return the three dicts
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, _ = self.model.loss(outputs, targets)
        return outputs, self.model.predict(outputs), targets
        # <<< week07

    def configure_optimizers(self):
        """AdamW or Lion, with OneCycleLR stepped every batch (``{"scheduler": sch, "interval": "step"}``),
        exactly as hepattn's ModelWrapper.configure_optimizers. The total number of steps is
        ``self.trainer.estimated_stepping_batches``. If ``lrs_config.get("skip_scheduler")`` return just the optimiser.
        """
        # >>> week07: pick the optimiser class, build it, build OneCycleLR like week 3, return ([opt], [sch_dict])
        cfg = self.lrs_config
        opt_cls = {"adamw": torch.optim.AdamW, "lion": Lion}[self.optimizer.lower()]
        opt = opt_cls(self.model.parameters(), lr=cfg["initial"], weight_decay=cfg["weight_decay"])
        if cfg.get("skip_scheduler"):
            return opt
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg["max"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=cfg["max"] / cfg["initial"],
            final_div_factor=cfg["initial"] / cfg["end"],
            pct_start=float(cfg["pct_start"]),
        )
        return [opt], [{"scheduler": sch, "interval": "step"}]
        # <<< week07


class Filter(ModelWrapper):
    """Hit filter: the generic task metrics are all we log (provided)."""


class Tracker(ModelWrapper):
    def log_custom_metrics(self, preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        """Log ``f"{stage}/p{wp}_eff"`` and ``f"{stage}/p{wp}_pur"`` for wp in (0.5, 0.75, 1.0) using
        ``matched_efficiency`` on the final layer, plus ``f"{stage}/num_tracks"`` (mean number of predicted
        valid tracks). Task names "track_valid" / "track_hit_valid" as in week 6.
        (hepattn: TrackMLTracker.log_custom_metrics.)
        """
        # >>> week07: pull the final predictions, loop over working points, self.log
        final = preds["final"]
        pred_valid = final["track_valid"]["track_valid"]
        pred_masks = final["track_hit_valid"]["track_hit_valid"]
        for wp in (0.5, 0.75, 1.0):
            m = matched_efficiency(pred_valid, pred_masks, targets["particle_valid"], targets["particle_hit_valid"], wp)
            self.log(f"{stage}/p{wp}_eff", m["eff"], batch_size=1)
            self.log(f"{stage}/p{wp}_pur", m["pur"], batch_size=1)
        self.log(f"{stage}/num_tracks", pred_valid.sum(-1).float().mean(), batch_size=1)
        # <<< week07
