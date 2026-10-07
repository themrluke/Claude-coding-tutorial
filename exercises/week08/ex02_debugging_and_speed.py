"""Week 8, exercise 2: finding NaNs, timing code properly, and graph breaks.

* NaN hunting: when a 3-hour run's loss turns NaN you want the *first* module that produced
  a non-finite value, not the loss. Forward hooks let you watch every module without editing
  the model. (Also: ``torch.autograd.set_detect_anomaly(True)`` / ``--trainer.detect_anomaly true``
  for the backward pass; slow, use it only while debugging.)
* Timing on a GPU: CUDA calls return immediately and run asynchronously. Without
  ``torch.cuda.synchronize()`` you time how long it took to *queue* the work. Always warm up
  first (the first calls include compilation, cuDNN autotuning and memory allocation).
* torch.compile traces Python into a graph. Anything it cannot trace (``.item()``, ``print``,
  data-dependent Python control flow) causes a *graph break*: the function is split into
  several graphs with slow Python in between. ``torch._dynamo.explain`` counts them.
"""

import statistics
import time
from collections.abc import Callable

import torch
from torch import Tensor, nn


def check_inputs_finite(batch: dict[str, Tensor], max_abs: float | None = None) -> list[str]:
    """Return the sorted keys of every floating-point tensor in ``batch`` that contains a NaN or inf,
    or (if ``max_abs`` is given) any value with ``|x| > max_abs``. Non-float tensors are ignored.

    Run this on suspicious batches. With ``max_abs=65504`` it would have flagged the corrupt TrackML
    events (charge_frac 121k-717k) before they were cast to fp16 by ``.half()`` in the data loader.
    """
    # TODO(week08): loop over items, skip non-floating dtypes, test isfinite and the magnitude
    raise NotImplementedError("week08 exercise (ex02_debugging_and_speed.py)")


def first_nonfinite_module(model: nn.Module, *args, **kwargs) -> str | None:
    """Run ``model(*args, **kwargs)`` with a forward hook on every submodule and return the qualified
    name (from ``model.named_modules()``) of the first module, in execution order, whose output
    contains a non-finite value. Return None if everything is finite.

    - ``module.register_forward_hook(fn)`` calls ``fn(module, inputs, output)`` after forward.
      Outputs may be a Tensor or a tuple/dict of them: check every tensor inside.
    - Keep the handles and ``handle.remove()`` them all at the end (use try/finally), or the hooks
      stay attached to the model forever.
    - Skip the root module (name ""), so a container is not reported before its children.
    """
    # TODO(week08): register one hook per named submodule that records names of non-finite outputs in order
    raise NotImplementedError("week08 exercise (ex02_debugging_and_speed.py)")


def time_fn(fn: Callable, *args, warmup: int = 3, iters: int = 10, device: str = "cpu") -> float:
    """Median wall time of ``fn(*args)`` in milliseconds.

    Call it ``warmup`` times untimed, then ``iters`` timed calls. On CUDA, ``torch.cuda.synchronize()``
    before starting and before stopping each timer. (hepattn: utils/cuda_timer.py and the
    InferenceTimer callback use CUDA events, which is the same idea.)
    """
    # TODO(week08): warm up, then time each call with perf_counter around synchronize() calls
    raise NotImplementedError("week08 exercise (ex02_debugging_and_speed.py)")


def count_graph_breaks(fn: Callable, *args) -> int:
    """Number of graph breaks torch.compile would hit on ``fn(*args)``.

    ``explanation = torch._dynamo.explain(fn)(*args)`` describes the tracing; count its
    ``break_reasons`` (each has a ``.reason`` string worth printing). In torch 2.9 a break at the very
    end of a function, like ``.item()`` feeding an ``if``, is listed in ``break_reasons`` but not counted
    in ``graph_break_count``. Call ``torch._dynamo.reset()`` first so earlier compilations do not interfere.
    """
    # TODO(week08): reset, explain, return the number of break reasons
    raise NotImplementedError("week08 exercise (ex02_debugging_and_speed.py)")


def peak_memory_mb(fn: Callable, *args) -> float:
    """Peak CUDA memory (MB) allocated while running ``fn(*args)`` once, above what was allocated before.

    ``torch.cuda.reset_peak_memory_stats()``, note ``memory_allocated()``, run, synchronize,
    ``max_memory_allocated()``. GPU only. (hepattn: callbacks/memory_stats.py.)
    """
    # TODO(week08): reset stats, baseline, run, peak minus baseline
    raise NotImplementedError("week08 exercise (ex02_debugging_and_speed.py)")
