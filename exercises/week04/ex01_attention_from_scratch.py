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
    # >>> week04: subtract amax(dim, keepdim=True), exp, divide by the sum
    z = x - x.amax(dim=dim, keepdim=True)
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)
    # <<< week04


def naive_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None, scale: float | None = None) -> tuple[Tensor, Tensor]:
    """Return ``(output, weights)`` where weights is the (..., Lq, Lk) softmax matrix.

    - scale defaults to 1 / sqrt(d).
    - Where ``mask`` is False, set the score to -inf *before* the softmax, so its weight is
      exactly 0. ``mask`` broadcasts against the scores (e.g. (B, 1, Lq, Lk) over heads).
    - Use your stable_softmax.
    """
    # >>> week04: scores = q @ kᵀ * scale, masked_fill(~mask, -inf), softmax, @ v
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = stable_softmax(scores, dim=-1)
    return weights @ v, weights
    # <<< week04


def fully_masked_rows(mask: Tensor) -> Tensor:
    """Bool tensor (..., Lq): True for every query row in which *every* key is masked out.

    Such a row has all scores -inf, softmax computes 0/0, and the output is NaN. NaNs then
    spread through every later layer. This is exactly why hepattn's MaskFormer decoder has
    ``unmask_all_false``.
    """
    # >>> week04: a row is fully masked when no entry along the last dim is True
    return ~mask.any(dim=-1)
    # <<< week04


def unmask_all_false(mask: Tensor) -> Tensor:
    """Return a copy of ``mask`` where every fully-masked row is replaced by all True.

    hepattn (models/decoder.py):
        attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)
    A query that has been told to look at nothing is allowed to look at everything instead.
    """
    # >>> week04: torch.where with the fully-masked rows (keepdim so they broadcast)
    return torch.where(fully_masked_rows(mask).unsqueeze(-1), True, mask)
    # <<< week04


def attention_is_permutation_equivariant(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """A property worth seeing with your own eyes. Return ``(a, b)`` with:

        a = naive_attention(q[perm_q], k[perm_kv], v[perm_kv])[0]
        b = naive_attention(q, k, v)[0][perm_q]

    for random permutations perm_q of the queries and perm_kv of the keys/values (permute the
    second-to-last axis; draw the permutations with torch.randperm). The test checks a == b:
    shuffling keys changes nothing, shuffling queries shuffles the output the same way.
    Attention sees a *set*. Anything order-dependent (positions, windows) must be added on purpose.
    """
    # >>> week04: draw two permutations, index the -2 axis, compare
    perm_q = torch.randperm(q.shape[-2])
    perm_kv = torch.randperm(k.shape[-2])
    a = naive_attention(q[..., perm_q, :], k[..., perm_kv, :], v[..., perm_kv, :])[0]
    b = naive_attention(q, k, v)[0][..., perm_q, :]
    return a, b
    # <<< week04
