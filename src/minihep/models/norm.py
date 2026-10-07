"""Week 4: normalisation layers.

Mirror of src/hepattn/models/norm.py. A norm layer rescales each token's feature
vector so its size stays roughly constant through a deep network.

* LayerNorm: subtract the mean of the features, divide by their std, then a learned
  per-feature scale (and shift).
* RMSNorm: skip the mean subtraction; divide by the root-mean-square. Cheaper, and
  works as well in practice. ``x / rms(x)`` equals ``F.normalize(x) * sqrt(dim)``.
* DyT (Dynamic Tanh, arXiv:2503.10622): no statistics at all, ``tanh(alpha * x)`` with a
  learned alpha, then a per-feature scale and shift.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """``y = x / sqrt(mean(x², over the last dim) + eps) * weight``, weight initialised to ones."""
        super().__init__()
        # >>> week04: store eps and a learnable weight of ones
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # <<< week04

    def forward(self, x: Tensor) -> Tensor:
        # >>> week04: compute the rms over the last dim with keepdim=True (use torch.rsqrt)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        # <<< week04


class DyT(nn.Module):
    def __init__(self, dim: int, alpha_init_value: float = 0.5):
        """``y = tanh(alpha * x) * weight + bias``. alpha is one learnable scalar; weight ones, bias zeros."""
        super().__init__()
        # >>> week04: three parameters: alpha (shape (1,)), weight, bias
        self.alpha = nn.Parameter(torch.full((1,), alpha_init_value))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        # <<< week04

    def forward(self, x: Tensor) -> Tensor:
        # >>> week04: tanh, scale, shift
        return torch.tanh(self.alpha * x) * self.weight + self.bias
        # <<< week04


class SimpleRMSNorm(nn.Module):
    """RMSNorm with no learnable weight (provided). Used for qkv-norm."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale


NORM_TYPES: dict[str, type[nn.Module]] = {
    "LayerNorm": nn.LayerNorm,
    "RMSNorm": RMSNorm,
    "SimpleRMSNorm": SimpleRMSNorm,
    "DyT": DyT,
}


def get_hybrid_norm_config(norm: str | None, depth: int, hybrid_norm: bool, qkv_norm: bool) -> tuple[str | None, bool, bool]:
    """HybridNorm (arXiv:2503.04598), provided; identical to hepattn.

    With ``hybrid_norm``, every layer after the first drops the pre-norm in front of attention
    (it normalises q, k and v inside attention instead) and puts the dense block's norm *after*
    the residual add (post-norm). Layer 0 keeps the usual pre-norm.

    Returns ``(attn_norm, dense_post_norm, qkv_norm)``.
    """
    qkv_norm = qkv_norm or hybrid_norm
    is_hybrid_subsequent = hybrid_norm and depth > 0
    attn_norm = None if is_hybrid_subsequent else norm
    return attn_norm, is_hybrid_subsequent, qkv_norm
