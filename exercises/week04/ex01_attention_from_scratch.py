"""Week 4, exercise 1: attention with nothing but matrix multiplies.

    Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d)) V

Shapes: q (..., Lq, d), k (..., Lk, d), v (..., Lk, dv) -> (..., Lq, dv). The leading
"..." is any batch/head dims; ``@`` broadcasts over them.

Masks follow the hepattn (and PyTorch SDPA) convention: **True means "take part"**.
A False entry at [i, j] means query i may not look at key j.
"""

import math

import torch
from torch import Tensor


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """softmax without overflow: subtract the max along ``dim`` before exponentiating.

    exp(1000) is inf in float32; exp(1000 - 1000) is 1. The result is mathematically
    identical because the max cancels between numerator and denominator.
    Do not call torch.softmax.
    """
    # TODO(week04): subtract amax(dim, keepdim=True), exp, divide by the sum
    raise NotImplementedError("week04 exercise (ex01_attention_from_scratch.py)")


def naive_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None, scale: float | None = None) -> tuple[Tensor, Tensor]:
    """Return ``(output, weights)`` where weights is the (..., Lq, Lk) softmax matrix.

    - scale defaults to 1 / sqrt(d).
    - Where ``mask`` is False, set the score to -inf *before* the softmax, so its weight is
      exactly 0. ``mask`` broadcasts against the scores (e.g. (B, 1, Lq, Lk) over heads).
    - Use your stable_softmax.
    """
    # TODO(week04): scores = q @ kᵀ * scale, masked_fill(~mask, -inf), softmax, @ v
    raise NotImplementedError("week04 exercise (ex01_attention_from_scratch.py)")


def fully_masked_rows(mask: Tensor) -> Tensor:
    """Bool tensor (..., Lq): True for every query row in which *every* key is masked out.

    Such a row has all scores -inf, softmax computes 0/0, and the output is NaN. NaNs then
    spread through every later layer. This is exactly why hepattn's MaskFormer decoder has
    ``unmask_all_false``.
    """
    # TODO(week04): a row is fully masked when no entry along the last dim is True
    raise NotImplementedError("week04 exercise (ex01_attention_from_scratch.py)")


def unmask_all_false(mask: Tensor) -> Tensor:
    """Return a copy of ``mask`` where every fully-masked row is replaced by all True.

    hepattn (models/decoder.py):
        attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)
    A query that has been told to look at nothing is allowed to look at everything instead.
    """
    # TODO(week04): torch.where with the fully-masked rows (keepdim so they broadcast)
    raise NotImplementedError("week04 exercise (ex01_attention_from_scratch.py)")


def attention_is_permutation_equivariant(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """A property worth seeing with your own eyes. Return ``(a, b)`` with:

        a = naive_attention(q[perm_q], k[perm_kv], v[perm_kv])[0]
        b = naive_attention(q, k, v)[0][perm_q]

    for random permutations perm_q of the queries and perm_kv of the keys/values (permute the
    second-to-last axis; draw the permutations with torch.randperm). The test checks a == b:
    shuffling keys changes nothing, shuffling queries shuffles the output the same way.
    Attention sees a *set*. Anything order-dependent (positions, windows) must be added on purpose.
    """
    # TODO(week04): draw two permutations, index the -2 axis, compare
    raise NotImplementedError("week04 exercise (ex01_attention_from_scratch.py)")
