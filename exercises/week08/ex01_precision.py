"""Week 8, exercise 1: mixed precision, and why your hit filter died at epoch 3.

Three 16-bit-ish formats matter:

    format   exponent bits  mantissa bits  max          machine eps
    float32  8              23             3.4e38       1.2e-7
    float16  5              10             65504        9.8e-4
    bfloat16 8              7              3.4e38       7.8e-3

fp16 has precision but almost no *range*: anything above 65504 becomes inf. bf16 has fp32's
range but very little precision. "Mixed precision" (``trainer.precision: 16-mixed`` or
``bf16-mixed``) runs matmuls in 16 bits under ``torch.autocast`` while keeping the weights,
the optimiser state and numerically delicate ops (losses, reductions, softmax on CUDA) in fp32.

fp16 training also needs a **GradScaler**. Small gradients underflow to 0 in fp16, so the loss
is multiplied by a large scale S before backward and the gradients divided by S afterwards. If
any gradient is inf/NaN the step is *skipped* and S is halved; after ``growth_interval``
clean steps in a row S is doubled. You will reproduce, in about 20 lines, the failure from
your 30 Sep 2026 filter run: six corrupt events (charge_frac up to 717k, inf in fp16) arrived
roughly every 1400 steps, more often than the 2000-step growth interval, so the scale only
ever went down: 4096 -> 256 -> 16 -> 0, and then everything was NaN.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


def overflows(values: Tensor, dtype: torch.dtype) -> Tensor:
    """Bool tensor: True where ``values`` (float32/64) become inf or NaN when cast to ``dtype``."""
    # TODO(week08): cast, then ~isfinite
    raise NotImplementedError("week08 exercise (ex01_precision.py)")


def relative_rounding_error(values: Tensor, dtype: torch.dtype) -> Tensor:
    """``|x - float(cast(x))| / |x|`` element-wise, computed in float64 (round-trip through ``dtype``)."""
    # TODO(week08): round-trip, compare in float64
    raise NotImplementedError("week08 exercise (ex01_precision.py)")


def accumulate(n: int, increment: float, dtype: torch.dtype) -> float:
    """Add ``increment`` to a running total ``n`` times, keeping the total in ``dtype``; return it as a float.

    Run it with n=10000, increment=1e-4 in float16: the true answer is 1.0, but once the total is
    big enough, total + increment rounds back to total ("swamping"). This is why sums, means and
    losses are kept in float32 even under autocast.
    """
    # TODO(week08): a Python loop with a 0-dim tensor of the given dtype
    raise NotImplementedError("week08 exercise (ex01_precision.py)")


def autocast_dtypes(device: str = "cpu") -> dict[str, torch.dtype]:
    """Run three things under ``torch.autocast(device, dtype=torch.bfloat16)`` and report the output dtypes:

    - "linear": ``nn.Linear(8, 8)`` on a float32 input        -> bf16 (matmuls are autocast's main target)
    - "bce": ``F.binary_cross_entropy_with_logits`` on that output -> float32 (autocast keeps losses in fp32)
    - "cost": the same linear inside ``torch.autocast(device, enabled=False)`` with ``.float()`` inputs -> float32
      (hepattn wraps its matching costs like this, see models/loss.py)
    Build the layer and input on ``device``.
    """
    # TODO(week08): three ops, record .dtype of each
    raise NotImplementedError("week08 exercise (ex01_precision.py)")


@dataclass
class MiniGradScaler:
    """The dynamic loss-scaling rule of ``torch.amp.GradScaler``, without the tensors.

    Rules for ``update(found_inf)``, called once per optimiser step:
      - found_inf: the optimiser step is skipped; ``scale *= backoff_factor``; the clean-step counter resets to 0.
      - otherwise: the step happens; the counter goes up by one; when it reaches ``growth_interval``,
        ``scale *= growth_factor`` and the counter resets to 0.
    ``update`` returns True if the optimiser step was taken. ``history`` records the scale *after* each update.
    """

    scale: float = 2.0**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    clean_steps: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, found_inf: bool) -> bool:
        # TODO(week08): implement the two rules, append to history, return whether the step was taken
        raise NotImplementedError("week08 exercise (ex01_precision.py)")


def simulate_corrupt_events(num_steps: int, bad_every: int, growth_interval: int = 2000, init_scale: float = 4096.0) -> MiniGradScaler:
    """Run a MiniGradScaler for ``num_steps`` steps where every ``bad_every``-th step (steps bad_every,
    2*bad_every, ...; counting from 1) has an inf gradient. Return the scaler (look at ``.history``)."""
    # TODO(week08): one update per step
    raise NotImplementedError("week08 exercise (ex01_precision.py)")
