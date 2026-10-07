"""Week 5, exercise 1: how FlashAttention computes exact attention without the N x N matrix.

The trick is the *online softmax*. Process the keys in blocks. For each query keep three
running quantities:

    m   the largest score seen so far
    l   sum of exp(score - m) over the keys seen so far
    acc sum of exp(score - m) * v over the keys seen so far

When a new block arrives with a bigger max m_new, everything accumulated so far was
computed relative to the old m, so rescale it by exp(m - m_new) before adding the new
block's contribution. At the end, output = acc / l and logsumexp = m + log(l).

Memory: one (Lq, block) tile of scores at a time instead of (Lq, Lk). On a GPU the tile
lives in fast on-chip SRAM: that is FlashAttention. The *same* bookkeeping lets you merge
attention results computed separately over different key sets (exercise c), which is how
hepattn merges its OR-amplified orderings (``or_merge_lse`` in models/attention.py).
"""

import math

import torch
from torch import Tensor


def logsumexp_blocks(scores: Tensor, block_size: int) -> Tensor:
    """(a) ``torch.logsumexp(scores, dim=-1)`` but visiting the last axis ``block_size`` columns at a time.

    Keep a running max ``m`` and running sum ``l`` (start m = -inf, l = 0). For each block:
    ``m_new = max(m, block.amax(-1))``; ``l = l * exp(m - m_new) + exp(block - m_new).sum(-1)``; ``m = m_new``.
    Return ``m + log(l)``. Careful: exp(-inf - (-inf)) is NaN; on the first block l is 0 anyway,
    so you can replace the NaN rescale factor by 0 (``torch.nan_to_num``) or special-case it.
    """
    # TODO(week05): loop over column blocks, update m and l, return m + log(l)
    raise NotImplementedError("week05 exercise (ex01_online_softmax.py)")


def tiled_attention(q: Tensor, k: Tensor, v: Tensor, block_size: int = 64) -> tuple[Tensor, Tensor]:
    """(b) FlashAttention's forward pass in plain PyTorch. Returns ``(out, lse)``.

    q (..., Lq, d), k (..., Lk, d), v (..., Lk, dv). out (..., Lq, dv), lse (..., Lq).
    Never build the full (Lq, Lk) score matrix: loop over key blocks, compute that block's
    scores ``q @ k_blockᵀ / sqrt(d)``, and update m, l and acc as described at the top.
    """
    # TODO(week05): initialise m, l, acc; for each key block rescale and accumulate; finish with acc / l
    raise NotImplementedError("week05 exercise (ex01_online_softmax.py)")


def merge_two(out_a: Tensor, lse_a: Tensor, out_b: Tensor, lse_b: Tensor) -> tuple[Tensor, Tensor]:
    """(c) Combine attention computed over two *disjoint* key sets A and B into attention over A ∪ B.

    Each part's output is a weighted average normalised by its own sum ``exp(lse)``. The full
    softmax normalises by ``exp(lse_a) + exp(lse_b)``, so

        lse = logaddexp(lse_a, lse_b)
        out = out_a * exp(lse_a - lse) + out_b * exp(lse_b - lse)

    This is exactly right, not an approximation. It is how ring attention and flash-decoding
    split work across GPUs or key chunks.
    """
    # TODO(week05): torch.logaddexp, then the two weights (unsqueeze to broadcast over the feature dim)
    raise NotImplementedError("week05 exercise (ex01_online_softmax.py)")


def or_merge(outs: Tensor, lses: Tensor) -> Tensor:
    """(d) hepattn's ``or_merge_lse``: merge C attention results with softmax-over-lse weights.

    outs (C, ..., Lq, dv) and lses (C, ..., Lq) -> (..., Lq, dv), weights ``softmax(lses, dim=0)``.

    For disjoint key sets this equals ``merge_two`` applied repeatedly. In OR amplification the C
    orderings' windows *overlap* (a key near the query is in several windows), so the result is
    not ordinary attention over the union: keys found by several orderings get counted several
    times. It is a deliberate, cheap, differentiable approximation. The test shows both facts.
    """
    # TODO(week05): weights = softmax over the C axis, then a weighted sum over C
    raise NotImplementedError("week05 exercise (ex01_online_softmax.py)")
