"""Week 3, exercise 2: a training loop written by hand.

Lightning (week 7) writes this loop for you. Write it once yourself so that you know
exactly what Lightning is doing when it calls your ``training_step``.

The loop for one step is always the same five lines:

    outputs = model(inputs)                         # forward
    loss = ...                                      # scalar
    optimizer.zero_grad()                           # clear old gradients
    loss.backward()                                 # fill p.grad for every parameter
    torch.nn.utils.clip_grad_norm_(params, max)     # optional, hepattn uses gradient_clip_val: 0.1
    optimizer.step(); scheduler.step()              # update weights, then the learning rate

hepattn's optimiser/scheduler setup is ``ModelWrapper.configure_optimizers`` in
src/hepattn/models/wrapper.py: AdamW (or Lion) + OneCycleLR stepped every batch.
"""

import itertools
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


def to_device(batch: dict[str, Tensor], device: str | torch.device) -> dict[str, Tensor]:
    """Move every tensor in a flat dict to ``device`` (provided)."""
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class LRConfig:
    """Same keys as hepattn's ``lrs_config``."""

    initial: float = 1e-4
    max: float = 1e-3
    end: float = 1e-5
    pct_start: float = 0.1
    weight_decay: float = 1e-5


def make_optimizer_and_scheduler(
    model: nn.Module, total_steps: int, cfg: LRConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW + OneCycleLR, translated exactly as hepattn does it.

    OneCycleLR is parameterised by *ratios*, so hepattn converts its three learning rates:
        max_lr = cfg.max
        div_factor = cfg.max / cfg.initial        (initial lr = max_lr / div_factor)
        final_div_factor = cfg.initial / cfg.end  (final lr = initial / final_div_factor)
    """
    # >>> week03: torch.optim.AdamW(model.parameters(), lr=cfg.initial, weight_decay=...), then OneCycleLR
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.initial, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.max,
        total_steps=total_steps,
        div_factor=cfg.max / cfg.initial,
        final_div_factor=cfg.initial / cfg.end,
        pct_start=cfg.pct_start,
    )
    return opt, sch
    # <<< week03


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)


def train(
    model: nn.Module,
    dataset: Dataset,
    num_steps: int,
    cfg: LRConfig | None = None,
    grad_clip: float | None = 1.0,
    device: str = "cpu",
) -> History:
    """Train ``model`` for exactly ``num_steps`` optimiser steps and return the history.

    The model follows hepattn's interface: ``outputs = model(inputs)`` and
    ``_, _, losses = model.loss(outputs, targets)`` where ``losses`` is a nested dict
    ``{layer_name: {task_name: {loss_name: scalar}}}``. The total loss is the sum of every
    scalar in it (see ``ModelWrapper.aggregate_losses``).

    Use ``DataLoader(dataset, batch_size=None, shuffle=True)`` and cycle through it as many
    times as needed (``itertools.cycle`` re-uses the first epoch's order, which is fine here).
    Put the model in train mode. Record, at every step: the loss (``loss.item()``), the
    learning rate *used for that step* (``scheduler.get_last_lr()[0]`` before stepping the
    scheduler) and the gradient norm (the return value of ``clip_grad_norm_``; if
    ``grad_clip`` is None, call it with ``max_norm=math.inf`` to just measure).
    """
    cfg = cfg or LRConfig()
    model.to(device)
    # >>> week03: optimiser + scheduler, loader, then the five-line loop num_steps times
    model.train()
    opt, sch = make_optimizer_and_scheduler(model, num_steps, cfg)
    loader = itertools.cycle(DataLoader(dataset, batch_size=None, shuffle=True))
    history = History()
    for _ in range(num_steps):
        inputs, targets = next(loader)
        inputs, targets = to_device(inputs, device), to_device(targets, device)
        outputs = model(inputs)
        _, _, losses = model.loss(outputs, targets)
        loss = sum(value for layer in losses.values() for task in layer.values() for value in task.values())
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip is not None else math.inf)
        history.lr.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
        history.loss.append(loss.item())
        history.grad_norm.append(float(grad_norm))
    return history
    # <<< week03


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, task_name: str, device: str = "cpu") -> dict[str, float]:
    """Average ``task.metrics`` over every event in ``dataset``.

    Put the model in eval mode, run forward + predict on each event, call
    ``task.metrics(preds["final"][task_name], targets)`` for the task whose ``name == task_name``,
    and return the mean of each metric as a Python float. Restore train mode at the end.
    """
    # >>> week03: eval(), loop, predict, collect metrics, average, train()
    model.eval()
    task = next(t for t in model.tasks if t.name == task_name)
    totals: dict[str, float] = {}
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        inputs, targets = to_device(inputs, device), to_device(targets, device)
        preds = model.predict(model(inputs))
        for k, v in task.metrics(preds["final"][task_name], targets).items():
            totals[k] = totals.get(k, 0.0) + float(v)
    model.train()
    return {k: v / len(dataset) for k, v in totals.items()}
    # <<< week03
