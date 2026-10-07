"""Week 1, exercise 4: context managers, decorators and properties.

Where these show up in hepattn:

    with torch.autocast(device_type="cuda", enabled=False):   models/loss.py (keep costs in float32)
    with h5py.File(path, "r") as f:                            experiments/trackml/data.py
    with contextlib.suppress(Exception):                       models/matcher.py
    @torch.no_grad()                                           Matcher.forward
    @atexit.register                                           matcher._close_pools
    @property                                                  MaskFormer.input_names
    datetime.now().strftime("%Y%m%d-T%H%M%S")                  utils/cli.py (run directory names)
"""

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime


class Timer:
    """(a) A class-based context manager that measures wall time.

        with Timer() as t:
            do_work()
        print(t.elapsed)   # seconds, as a float

    ``elapsed`` should be None before the block finishes. Use ``time.perf_counter``.
    The timer must still record ``elapsed`` if the block raises, and must not
    swallow the exception (``__exit__`` returns False/None).
    """

    def __init__(self) -> None:
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        # TODO(week01): start the clock and return self
        raise NotImplementedError("week01 exercise (ex04_context_and_decorators.py)")

    def __exit__(self, exc_type, exc, tb) -> bool:
        # TODO(week01): store the elapsed time; return False so exceptions propagate
        raise NotImplementedError("week01 exercise (ex04_context_and_decorators.py)")


@contextmanager
def timer() -> Iterator[dict]:
    """(b) The same thing written as a generator with ``@contextmanager``.

    Yield a dict; after the block, it must contain ``"elapsed"``. Use try/finally so the
    time is recorded even if the block raises.
    """
    # TODO(week01): record the start, yield a dict, fill in "elapsed" in a finally block
    raise NotImplementedError("week01 exercise (ex04_context_and_decorators.py)")


def log_calls(fn: Callable) -> Callable:
    """(c) A decorator that records every call's arguments.

    The wrapped function must:
      - behave exactly like ``fn`` (same return value),
      - keep ``fn``'s ``__name__`` and docstring (use ``functools.wraps``),
      - have an attribute ``calls``: a list of ``(args, kwargs)`` tuples, one per call.
    """

    # TODO(week01): write an inner wrapper with functools.wraps, attach a calls list
    raise NotImplementedError("week01 exercise (ex04_context_and_decorators.py)")


RUN_TIMESTAMP_FORMAT = "%Y%m%d-T%H%M%S"


class RunDir:
    """(d) Name run directories the way hepattn's CLI does: ``{name}_{timestamp}``.

    - ``RunDir("TRK-Pix0.6", datetime(2026, 10, 2, 12, 31, 10)).dirname`` is the
      *property* ``"TRK-Pix0.6_20261002-T123110"``.
    - ``RunDir.from_dirname("TRK-Pix0.6_20261002-T123110")`` is a *classmethod* that
      parses it back. Names can contain underscores (``my_run_20261002-T123110``):
      split on the *last* underscore. Raise ValueError if the suffix is not a timestamp
      (``datetime.strptime`` already does).
    """

    def __init__(self, name: str, timestamp: datetime):
        self.name = name
        self.timestamp = timestamp

    @property
    def dirname(self) -> str:
        # TODO(week01): format name and timestamp with RUN_TIMESTAMP_FORMAT
        raise NotImplementedError("week01 exercise (ex04_context_and_decorators.py)")

    @classmethod
    def from_dirname(cls, dirname: str) -> "RunDir":
        # TODO(week01): split on the last underscore and parse the timestamp with datetime.strptime
        raise NotImplementedError("week01 exercise (ex04_context_and_decorators.py)")
