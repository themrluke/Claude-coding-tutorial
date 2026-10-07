"""Week 5, exercise 2: "varlen" packing, how flash-attention handles a padded batch without padding.

Events have different numbers of hits. Padding a batch to the longest event wastes compute
on fake tokens and needs a kv_mask. The alternative (``flash_attn_varlen_func``) packs
every real token of every event into one long (total_tokens, ...) tensor and passes
``cu_seqlens``, the cumulative sequence lengths, so the kernel knows where each event
starts and stops:

    lengths    = [3, 5, 2]
    cu_seqlens = [0, 3, 8, 10]    (int32, length B + 1)
    event b occupies rows cu_seqlens[b] : cu_seqlens[b + 1] of the packed tensor

hepattn: utils/bert_padding.py (unpad_input / pad_input, adapted from flash-attn) and
``unpad_for_flash_varlen`` / ``repad_from_flash_varlen`` in models/attention.py. The
encoder unpads once before the first layer and repads after the last.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def unpad(x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    """Pack the valid tokens of x (B, S, ...) into (total, ...).

    Returns:
        x_packed: (total, ...) the valid tokens, event 0's first, in order.
        indices: (total,) int64 positions of those tokens in the flattened (B * S) axis.
        cu_seqlens: (B + 1,) int32, starting with 0.
        max_seqlen: Python int, the longest event.

    Hints: ``valid.flatten().nonzero().flatten()``; ``F.pad(lengths.cumsum(0), (1, 0))``.
    """
    # TODO(week05): lengths per row, flat indices of valid tokens, gather, cumulative lengths
    raise NotImplementedError("week05 exercise (ex02_varlen.py)")


def pad(x_packed: Tensor, indices: Tensor, batch_size: int, seq_len: int) -> Tensor:
    """Inverse of ``unpad``: scatter the packed tokens back into a zero-filled (B, S, ...) tensor."""
    # TODO(week05): allocate (B * S, ...) zeros, assign at indices, unflatten
    raise NotImplementedError("week05 exercise (ex02_varlen.py)")


def varlen_self_attention(q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Reference varlen attention on packed (total, H, Dh) tensors, one event at a time.

    For each event b, slice rows ``cu_seqlens[b]:cu_seqlens[b+1]``, move heads first
    ((S, H, Dh) -> (H, S, Dh)), run ``F.scaled_dot_product_attention`` and write the result back.
    flash_attn_varlen_func does this for all events in one kernel launch; the result is the same.
    """
    # TODO(week05): loop over consecutive pairs of cu_seqlens, transpose, SDPA, transpose back
    raise NotImplementedError("week05 exercise (ex02_varlen.py)")
