"""Week 5: FlexAttention mask functions.

Mirror of src/hepattn/flex/sliding_window.py and flex/local_ca.py.

A ``mask_mod(b, h, q_idx, kv_idx) -> bool tensor`` describes *which* (query, key) pairs
are allowed. FlexAttention never calls it once per pair at run time. Instead:

  * ``create_block_mask`` evaluates it (vectorised with vmap) on a coarse grid of
    128 x 128 blocks and records which blocks are completely empty, completely full or
    partial. Empty blocks are skipped entirely by the kernel: that is where the speed-up
    comes from.
  * ``torch.compile`` turns the mask_mod into code inside the fused kernel, which is applied
    element by element only in the partial blocks.

So the mask_mod must be written with tensor operations (``&``, ``|``, ``abs``, indexing a
captured tensor), never with Python ``if``/``and`` on the indices, which are tensors.
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature


def sliding_window_mask(window_size: int) -> _mask_mod_signature:
    """Allow pairs with ``|q_idx - kv_idx| <= window_size // 2``. (Same as hepattn.)"""

    # TODO(week05): return a closure mask_mod(b, h, q_idx, kv_idx); combine two comparisons with &
    raise NotImplementedError("week05 exercise (masks.py)")


def sliding_window_mask_wrapped(window_size: int, q_len: Tensor) -> _mask_mod_signature:
    """Like sliding_window_mask, but the sequence is a ring of length ``q_len[0]`` (phi wraps at ±pi).

    ``q_len`` is a 1-element tensor read at call time (week 1's closure exercise, now for real).
    Allowed if the direct distance OR the distance the other way round the ring is <= window_size // 2.
    """

    # TODO(week05): diagonal band | the two wrap-around corners (q - kv + L) and (kv - q + L)
    raise NotImplementedError("week05 exercise (masks.py)")


def strided_window_mask(window_size: int, q_len: int, kv_len: int) -> _mask_mod_signature:
    """Local *cross*-attention (LCA) for the decoder: Lq queries spread evenly along Lk sorted keys.

    Query i sits at key position ``round(i * kv_len / q_len)`` and sees keys within
    ``window_size // 2`` of that centre. This is hepattn's ``sliding_window_mask_strided``
    (flex/local_ca.py) and what the paper calls LSCA. No wrapping here.
    """

    # TODO(week05): compute the stride as a float, the centre with torch.round, compare |kv - centre|
    raise NotImplementedError("week05 exercise (masks.py)")


def mask_mod_to_dense(mask_mod: _mask_mod_signature, q_len: int, kv_len: int, device: str = "cpu") -> Tensor:
    """Evaluate a mask_mod on every pair: a (q_len, kv_len) bool tensor (provided, for tests and plots)."""
    q_idx = torch.arange(q_len, device=device).view(-1, 1)
    kv_idx = torch.arange(kv_len, device=device).view(1, -1)
    zero = torch.zeros((), dtype=torch.long, device=device)
    return mask_mod(zero, zero, q_idx, kv_idx).expand(q_len, kv_len)


def block_sparsity(block_mask: BlockMask) -> float:
    """Fraction of blocks the kernel can skip (provided). 0 = dense, close to 1 = very sparse."""
    return block_mask.sparsity() / 100.0
