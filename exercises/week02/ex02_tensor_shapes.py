"""Week 2, exercise 2: the tensor moves hepattn makes over and over.

Shapes are given in the docstrings with letters:
    B batch, N hits (keys), Q queries / objects, D embedding dim, H heads, Dh = D // H.

Every function here is 1-4 lines. Each is lifted from a real place in hepattn, which
is named in the docstring. Read that line after you pass the test.
"""

import torch
from torch import Tensor


def build_target_masks(object_ids: Tensor, hit_object_ids: Tensor) -> Tensor:
    """(B, Q) object ids and (B, N) per-hit object ids -> (B, Q, N) bool mask.

    ``mask[b, q, n]`` is True when hit n belongs to object q. Padded objects have id -999
    and must never match anything.  (hepattn: utils/masks.py build_target_masks, and the
    ``particle_hit_valid`` line in experiments/trackml/data.py)
    """
    # >>> week02: unsqueeze so the two id tensors broadcast against each other, then ==
    return object_ids.unsqueeze(-1) == hit_object_ids.unsqueeze(-2)
    # <<< week02


def sort_tokens(x: Tensor, sort_value: Tensor) -> tuple[Tensor, Tensor]:
    """Sort tokens (B, N, D) by a per-token value (B, N), e.g. phi. Return ``(x_sorted, sort_idx)``.

    hepattn: Encoder.forward, ``torch.gather(x, dim=-2, index=x_sort_idx.unsqueeze(-1).expand_as(x))``.
    ``gather`` needs the index to have the same number of dims as x, hence unsqueeze + expand.
    """
    # >>> week02: argsort the values, then gather along the token axis
    sort_idx = torch.argsort(sort_value, dim=-1)
    return torch.gather(x, dim=-2, index=sort_idx.unsqueeze(-1).expand_as(x)), sort_idx
    # <<< week02


def unsort_tokens(x_sorted: Tensor, sort_idx: Tensor) -> Tensor:
    """Undo ``sort_tokens``: return x in the original token order."""
    # >>> week02: argsort the sort index (inverse permutation), gather again
    unsort_idx = torch.argsort(sort_idx, dim=-1)
    return torch.gather(x_sorted, dim=-2, index=unsort_idx.unsqueeze(-1).expand_as(x_sorted))
    # <<< week02


def mask_logits(queries: Tensor, hits: Tensor, hit_valid: Tensor | None = None) -> Tensor:
    """Dot every query (B, Q, D) with every hit (B, N, D) -> logits (B, Q, N).

    If ``hit_valid`` (B, N) is given, set the logits of padded hits to the most negative
    representable value of the dtype (``torch.finfo(dtype).min``), so sigmoid gives 0.
    Do not modify the inputs. (hepattn: ObjectHitMaskTask.forward in models/task.py)
    """
    # >>> week02: einsum "bqd,bnd->bqn", then masked_fill where ~hit_valid (unsqueezed to (B, 1, N))
    logits = torch.einsum("bqd,bnd->bqn", queries, hits)
    if hit_valid is not None:
        logits = logits.masked_fill(~hit_valid.unsqueeze(-2), torch.finfo(logits.dtype).min)
    return logits
    # <<< week02


def separate_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, S, D) -> (B, H, S, Dh). hepattn: Attention.separate_heads (non-flash branch)."""
    # >>> week02: unflatten the last dim into (H, Dh), then swap the S and H axes
    return x.unflatten(-1, (num_heads, -1)).transpose(-3, -2)
    # <<< week02


def recombine_heads(x: Tensor) -> Tensor:
    """(B, H, S, Dh) -> (B, S, D). Inverse of ``separate_heads``."""
    # >>> week02: swap back, then flatten the last two dims
    return x.transpose(-3, -2).flatten(-2)
    # <<< week02


def permute_objects(x: Tensor, idx: Tensor) -> Tensor:
    """Reorder the object axis of x (B, Q, ...) with a different permutation per batch element.

    ``idx`` is (B, Q): output[b, i] = x[b, idx[b, i]]. This is how MaskFormer applies the
    Hungarian matching result: ``output_tensor[batch_idxs_expanded, pred_idxs]``
    (models/maskformer.py, _match_and_permute_outputs). Use advanced indexing with a
    (B, 1) batch index that broadcasts against idx.
    """
    # >>> week02: batch index of shape (B, 1) from arange, then x[batch_idx, idx]
    batch_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
    return x[batch_idx, idx]
    # <<< week02


def expand_is_a_view(x: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """Return ``(expanded, repeated)``: x of shape (1, S, D) broadcast to (n, S, D) two ways.

    ``expand`` returns a *view* (no copy, stride 0 on the new axis); ``repeat`` copies.
    The test checks which one shares memory with x. hepattn uses ``expand`` for the learned
    queries (``self.initial_queries.expand(batch_size, -1, -1)``) so batching costs no memory.
    """
    # >>> week02: x.expand(n, -1, -1) and x.repeat(n, 1, 1)
    return x.expand(n, -1, -1), x.repeat(n, 1, 1)
    # <<< week02
