"""Multi-head attention with swappable kernels (week 4: torch; week 5: flex and flash).

Mirror of src/hepattn/models/attention.py. Read the real file after each week: you
will recognise ``merge_masks``, ``separate_heads``, ``recombine_heads``, ``set_backend``
and the per-backend branches in ``forward``.

Shapes: B batch, Lq query tokens, Lk key tokens, D = dim, H heads, Dh = D // H.
Mask convention: True means "take part in attention".
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from minihep.models.norm import NORM_TYPES

try:
    from flash_attn import flash_attn_func
except ImportError:  # flash-attn is optional: `pixi run -e fa2 ...`
    flash_attn_func = None

ATTN_TYPES = ("torch", "flex", "flash")
FLASH_ATTN_TYPES = ("flash",)  # backends that want (B, S, H, Dh) instead of (B, H, S, Dh)
ATTN_MASK_ATTN_TYPES = ("torch", "flex")  # backends that accept an attention mask
WINDOW_ATTN_TYPES = ("flash",)  # backends that take the window as an argument


def merge_masks(
    q_mask: Tensor | None, kv_mask: Tensor | None, attn_mask: Tensor | None, q_len: int, kv_len: int, batch_size: int, device
) -> Tensor | None:
    """Combine padding masks and an attention mask into one (B, Lq, Lk) bool mask, or None.

    - ``q_mask`` (B, Lq) and ``kv_mask`` (B, Lk): False marks padding. If at least one is given,
      the pair mask is ``q_mask[:, :, None] & kv_mask[:, None, :]`` (a missing one counts as all True).
    - ``attn_mask`` (B, Lq, Lk): ANDed in if given.
    - If nothing is given, return None (SDPA is fastest with no mask at all).

    hepattn: merge_masks at the top of models/attention.py.
    """
    # TODO(week04): build the pair mask from the padding masks if any, then AND in attn_mask
    raise NotImplementedError("week04 exercise (attention.py)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        window_size: int | None = None,
        qkv_norm: bool = False,
        norm: str | None = None,
    ) -> None:
        """Multi-head attention.

        Parameters (the layout matters: week 4's test copies these weights into
        ``nn.MultiheadAttention`` and expects identical outputs):
            in_proj_weight: (3 * dim, dim) holding W_q, W_k, W_v stacked in that order.
            in_proj_bias:   (3 * dim,) or None if not bias.
            out_proj:       nn.Linear(dim, dim, bias=bias).
        If ``qkv_norm``: ``q_norm``, ``k_norm``, ``v_norm`` = NORM_TYPES[norm](dim), applied after projection.

        Initialise in_proj_weight with xavier_uniform_ and in_proj_bias with zeros.
        Finish with ``self.set_backend(attn_type, window_size)``.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide dim={dim}")
        if qkv_norm and norm is None:
            raise ValueError("norm must be given when qkv_norm=True")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_norm = qkv_norm
        # TODO(week04): create in_proj_weight, in_proj_bias, out_proj (and the qkv norms), initialise
        raise NotImplementedError("week04 exercise (attention.py)")
        self.set_backend(attn_type, window_size)

    def set_backend(self, attn_type: str, window_size: int | None = None) -> None:
        """Choose the attention kernel. Can be called again later, e.g. to evaluate a flex-trained model with SDPA."""
        if attn_type not in ATTN_TYPES:
            raise ValueError(f"Invalid attention type: {attn_type}. Choose from {ATTN_TYPES}")
        if window_size is not None and attn_type not in WINDOW_ATTN_TYPES:
            raise ValueError(f"window_size can only be passed to {WINDOW_ATTN_TYPES}; flex and torch take a window through the mask")
        self.attn_type = attn_type
        self.window_size = window_size
        self.flex_eager = self.flex_compiled = None
        self.flash_window = (-1, -1)
        if attn_type == "flex":
            self._setup_flex()
        elif attn_type == "flash":
            self._setup_flash()

    def _setup_flex(self) -> None:
        """Week 5: keep two versions of flex_attention, ``self.flex_eager`` (the plain function) and
        ``self.flex_compiled = torch.compile(flex_attention, dynamic=True)``.

        Compiling is what makes flex fast and memory-light; uncompiled it builds the full score matrix.
        Inductor cannot compile flex for CPU training, so the eager one is kept for CPU tensors.
        """
        # TODO(week05): store the eager and the compiled function
        raise NotImplementedError("week05 exercise (attention.py)")

    def _setup_flash(self) -> None:
        """Week 5: raise ImportError if flash-attn is missing; store ``self.flash_window``.

        flash-attn takes the window as ``(left, right)`` token counts; ``(-1, -1)`` means no window.
        A full window of ``window_size`` is ``(window_size // 2, window_size // 2)``.
        """
        # TODO(week05): check flash_attn_func is not None, then compute the (left, right) window
        raise NotImplementedError("week05 exercise (attention.py)")

    def separate_heads(self, x: Tensor) -> Tensor:
        """(B, S, D) -> (B, H, S, Dh), or (B, S, H, Dh) for flash backends."""
        # TODO(week04): unflatten the last dim into (H, Dh); transpose S and H unless the backend is a flash one
        raise NotImplementedError("week04 exercise (attention.py)")

    def recombine_heads(self, x: Tensor) -> Tensor:
        """Inverse of separate_heads, back to (B, S, D)."""
        # TODO(week04): undo the transpose (if any) and flatten the last two dims
        raise NotImplementedError("week04 exercise (attention.py)")

    def project_qkv(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply W_q, W_k, W_v (slices of the packed weight) and the optional qkv norms. Shapes stay (B, S, D).

        Use ``self.in_proj_weight.chunk(3)`` (and the bias likewise) and ``F.linear``.
        hepattn calls ``F._in_projection_packed``, which does the same thing with a fast path for
        self-attention (one matmul, then chunk the result).
        """
        # TODO(week04): chunk weights and biases into three, F.linear each input, then norms
        raise NotImplementedError("week04 exercise (attention.py)")

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        attn_mask: Tensor | BlockMask | None = None,
    ) -> Tensor:
        """Self-attention if k and v are None, cross-attention otherwise (v defaults to k).

        Steps:
          1. Resolve k and v. Validate: only torch takes kv/q padding masks; only torch and
             flex take an attn_mask (raise ValueError otherwise).
          2. Project, separate heads.
          3. Backend:
             - torch: ``merge_masks``; if the merged mask is 3D, unsqueeze a head axis at dim 1;
               ``F.scaled_dot_product_attention(q, k, v, attn_mask=mask)``.
             - flex (week 5): see ``_flex``.
             - flash (week 5): see ``_flash``.
          4. Recombine heads, out_proj.
        """
        if k is None and v is None:
            k = v = q
        elif v is None:
            v = k
        # TODO(week04): validation, projection, the torch branch, recombine, out_proj
        raise NotImplementedError("week04 exercise (attention.py)")

    def _flex(self, q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask | None) -> Tensor:
        """Week 5. FlexAttention takes (B, H, S, Dh) and an optional BlockMask.

        - A dense bool tensor is not accepted (raise ValueError); flex wants a BlockMask.
        - Use the compiled function on CUDA tensors and the eager one otherwise
          (hepattn: ``attn_fn = self.attn if q.is_cuda else self.attn_uncompiled``). Uncompiled
          flex is correct but materialises the full score matrix, so it is only for CPU tests.
        """
        # TODO(week05): check the mask type, pick compiled vs eager, call it with block_mask=
        raise NotImplementedError("week05 exercise (attention.py)")

    def _flash(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Week 5. FlashAttention-2 takes (B, S, H, Dh) in fp16/bf16 on CUDA, and a (left, right) window."""
        # TODO(week05): call flash_attn_func with window_size=self.flash_window
        raise NotImplementedError("week05 exercise (attention.py)")
