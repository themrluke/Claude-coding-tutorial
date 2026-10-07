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
        # >>> week01: start the clock and return self
        self._start = time.perf_counter()
        return self
        # <<< week01

    def __exit__(self, exc_type, exc, tb) -> bool:
        # >>> week01: store the elapsed time; return False so exceptions propagate
        self.elapsed = time.perf_counter() - self._start
        return False
        # <<< week01


@contextmanager
def timer() -> Iterator[dict]:
    """(b) The same thing written as a generator with ``@contextmanager``.

    Yield a dict; after the block, it must contain ``"elapsed"``. Use try/finally so the
    time is recorded even if the block raises.
    """
    # >>> week01: record the start, yield a dict, fill in "elapsed" in a finally block
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
    # <<< week01


def log_calls(fn: Callable) -> Callable:
    """(c) A decorator that records every call's arguments.

    The wrapped function must:
      - behave exactly like ``fn`` (same return value),
      - keep ``fn``'s ``__name__`` and docstring (use ``functools.wraps``),
      - have an attribute ``calls``: a list of ``(args, kwargs)`` tuples, one per call.
    """

    # >>> week01: write an inner wrapper with functools.wraps, attach a calls list
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        wrapper.calls.append((args, kwargs))
        return fn(*args, **kwargs)

    wrapper.calls = []
    return wrapper
    # <<< week01


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
        # >>> week01: format name and timestamp with RUN_TIMESTAMP_FORMAT
        return f"{self.name}_{self.timestamp.strftime(RUN_TIMESTAMP_FORMAT)}"
        # <<< week01

    @classmethod
    def from_dirname(cls, dirname: str) -> "RunDir":
        # >>> week01: split on the last underscore and parse the timestamp with datetime.strptime
        name, _, stamp = dirname.rpartition("_")
        return cls(name, datetime.strptime(stamp, RUN_TIMESTAMP_FORMAT))
        # <<< week01
