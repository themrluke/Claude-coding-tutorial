"""Week 1, exercise 2: closures.

A closure is a function that remembers variables from the scope it was created
in. FlexAttention is built on them: you write a small ``mask_mod(b, h, q_idx, kv_idx)``
function, and anything else it needs (the window size, the sequence length, a
table of ranks) is captured from the enclosing function. Read these first:

    src/hepattn/flex/sliding_window.py     sliding_window_mask / sliding_window_mask_wrapped
    src/hepattn/flex/per_head_window.py    per_head_window_mask_mod (captures a tensor of ranks)
    src/hepattn/models/encoder.py          the comment "Written in place, not rebound" in forward()

The last one is the subtle bit, and exercise (b) below reproduces it.
"""

from collections.abc import Callable


def make_window_mask(window_size: int) -> Callable[[int, int], bool]:
    """(a) Return ``mask(q_idx, kv_idx)`` that is True when ``|q_idx - kv_idx| <= window_size // 2``.

    Same rule as hepattn's ``sliding_window_mask``.
    """
    # >>> week01: define an inner function that uses window_size, then return it
    half = window_size // 2

    def mask(q_idx: int, kv_idx: int) -> bool:
        return abs(q_idx - kv_idx) <= half

    return mask
    # <<< week01


def make_wrapped_window_mask(window_size: int, seq_len: list[int]) -> Callable[[int, int], bool]:
    """(b) Like (a), but the window wraps around the end of the sequence (phi is periodic!).

    ``seq_len`` is a one-element list holding the current sequence length. The
    returned function must read ``seq_len[0]`` *every time it is called*, so the
    caller can change the length later without rebuilding the mask:

        n = [10]
        mask = make_wrapped_window_mask(4, n)
        mask(0, 9)   # True: 0 and 9 are neighbours on a ring of 10
        n[0] = 100
        mask(0, 9)   # False: on a ring of 100 they are 9 apart

    hepattn does exactly this with a one-element *tensor* ``self.seq_len`` and writes
    ``self.seq_len[0] = seq_len`` instead of ``self.seq_len = torch.tensor(...)``.
    Rebinding the name would leave the cached closure looking at the old tensor.

    Rule: True if the plain distance or the distance around the ring is <= window_size // 2.
    """
    # >>> week01: compute the direct distance and the wrap-around distance using seq_len[0]
    half = window_size // 2

    def mask(q_idx: int, kv_idx: int) -> bool:
        d = abs(q_idx - kv_idx)
        return min(d, seq_len[0] - d) <= half

    return mask
    # <<< week01


def make_counter() -> Callable[[], int]:
    """(c) Return a function that returns 1, 2, 3, ... on successive calls.

    Each counter made by ``make_counter`` must count independently. You'll need
    ``nonlocal``.
    """
    # >>> week01: keep a count in the enclosing scope and update it with nonlocal
    count = 0

    def counter() -> int:
        nonlocal count
        count += 1
        return count

    return counter
    # <<< week01


def make_multipliers(n: int) -> list[Callable[[float], float]]:
    """(d) Return ``[f_0, ..., f_{n-1}]`` where ``f_i(x) == i * x``.

    The tempting one-liner ``[lambda x: i * x for i in range(n)]`` is wrong: every
    lambda looks up ``i`` when it is *called*, after the loop has finished, so they all
    multiply by ``n - 1``. This is called late binding. Fix it.
    """
    # >>> week01: bind the current value of i at creation time (default argument or a factory function)
    return [lambda x, i=i: i * x for i in range(n)]
    # <<< week01
