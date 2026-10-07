"""The transformer encoder (week 4) with windowed attention (week 5).

Mirror of src/hepattn/models/encoder.py. One encoder layer is

    x = x + attention(norm(x))      # tokens exchange information
    x = x + dense(norm(x))          # each token is processed on its own

wrapped by ``Residual``, which also handles LayerScale, DropPath and post-norm.
"""

from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from minihep.flex.masks import sliding_window_mask, sliding_window_mask_wrapped
from minihep.models.attention import FLASH_ATTN_TYPES, Attention
from minihep.models.dense import Dense
from minihep.models.norm import NORM_TYPES, get_hybrid_norm_config


class LayerScale(nn.Module):
    """Multiply by a learnable per-feature gamma, initialised small (arXiv:2103.17239).

    Starting each residual branch near zero makes a deep network start out close to the
    identity, which trains more stably.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        # TODO(week04): one parameter of shape (dim,) filled with init_value
        raise NotImplementedError("week04 exercise (encoder.py)")

    def forward(self, x: Tensor) -> Tensor:
        # TODO(week04): scale
        raise NotImplementedError("week04 exercise (encoder.py)")


class DropPath(nn.Module):
    """Stochastic depth (arXiv:1603.09382): in training, zero a whole sample's residual branch with probability p.

    Surviving samples are divided by (1 - p) so the expected value is unchanged. In eval mode,
    or with p == 0, it is the identity. One random draw per *sample* (shape (B, 1, 1, ...)), not per element.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        # TODO(week04): identity if not training or p == 0; else a (B, 1, ..., 1) keep mask, divide by keep prob
        raise NotImplementedError("week04 exercise (encoder.py)")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, dim: int, norm: str | None, post_norm: bool = False, layer_scale: float | None = None, drop_path: float = 0.0):
        """``x + drop(scale(fn(norm(x))))``, or with ``post_norm``: ``x = norm(x); x + drop(scale(fn(x)))``.

        Note hepattn's post_norm normalises the *input* of the residual block (which is the output
        of the previous add), so the residual stream itself gets normalised. Unused parts are
        ``nn.Identity()``. Keyword arguments to forward are passed through to ``fn``
        (that is how masks reach the attention inside).
        """
        super().__init__()
        if post_norm and norm is None:
            raise ValueError("post_norm=True needs a norm")
        # TODO(week04): store fn and post_norm; norm, layer scale and drop path as modules (Identity when unused)
        raise NotImplementedError("week04 exercise (encoder.py)")

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # TODO(week04): the two orderings described above
        raise NotImplementedError("week04 exercise (encoder.py)")


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 0,
        norm: str = "LayerNorm",
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        hybrid_norm: bool = False,
        qkv_norm: bool = False,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ):
        """Self-attention residual block followed by a dense residual block.

        Use ``get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)`` to decide the attention
        pre-norm, whether the dense block is post-norm, and whether attention uses qkv-norm.
        ``partial(Residual, dim=dim, layer_scale=..., drop_path=...)`` saves repetition (hepattn does this).
        Attributes must be called ``self.attn`` and ``self.dense`` (tests reach in through ``.fn``).
        """
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}
        # TODO(week04): hybrid-norm config, then two Residual blocks
        raise NotImplementedError("week04 exercise (encoder.py)")

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        attn_kwargs: dict | None = None,
        **layer_kwargs,
    ):
        """A stack of EncoderLayers. ``layer_kwargs`` go to every EncoderLayer (norm, layer_scale, hybrid_norm...).

        ``attn_type`` is copied into (a copy of) ``attn_kwargs`` for every layer. ``window_size`` is passed to the
        attention only for flash backends (flex/torch get it through a mask built in forward).
        Layer i gets ``depth=i``.
        """
        super().__init__()
        if window_wrap and not window_size:
            raise ValueError("window_wrap needs a window_size")
        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = attn_type
        self.window_size = window_size
        self.window_wrap = window_wrap
        # TODO(week04): copy attn_kwargs, set attn_type and window_size, build an nn.ModuleList of layers
        raise NotImplementedError("week04 exercise (encoder.py)")

    def build_attn_mask(self, seq_len: int, device) -> Tensor | object | None:
        """Week 5: the sliding-window mask for this sequence length, or None.

        - No window, or a flash backend (the kernel applies the window itself): None.
        - torch backend: a dense (1, 1, L, L) bool mask, ``create_mask(mask_mod, 1, 1, L, L, device=...)``.
        - flex backend: a BlockMask, ``create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=...)``.
        mask_mod is ``sliding_window_mask(window_size)``, or ``sliding_window_mask_wrapped(window_size,
        torch.tensor([L]))`` when ``window_wrap``. (hepattn caches the mask_mod and writes the length into
        the tensor in place; building it fresh each time is simpler and fine for us.)
        """
        # TODO(week05): choose the mask_mod, then build a dense mask or a BlockMask depending on attn_type
        raise NotImplementedError("week05 exercise (encoder.py)")

    def forward(self, x: Tensor, x_sort_value: Tensor | None = None, kv_mask: Tensor | None = None) -> Tensor:
        """Encode (B, N, D) tokens.

        Week 4:
          1. If ``x_sort_value`` (B, N) is given, sort the tokens (and kv_mask) by it, run the
             layers, then unsort the output so the caller gets tokens back in their original order.
          2. Run each layer, passing ``kv_mask=kv_mask`` (and ``attn_mask`` from step 3).
        Week 5:
          3. ``attn_mask = self.build_attn_mask(N, x.device)``.
          4. Flash with ``window_wrap``: the kernel cannot wrap, so wrap the *data* instead: prepend
             the last ``window_size // 2`` tokens and append the first ``window_size // 2``, run the
             layers, then cut them off again. (Exactly what hepattn does.)
        """
        # TODO(week04): sort (gather), layers, unsort (gather with the inverse permutation)
        raise NotImplementedError("week04 exercise (encoder.py)")

    def set_backend(self, attn_type: str) -> None:
        """Switch every layer's attention kernel (provided). hepattn: Encoder.set_backend / change_attn_backends."""
        self.attn_type = attn_type
        for layer in self.layers:
            layer.attn.fn.set_backend(attn_type, self.window_size if attn_type in FLASH_ATTN_TYPES else None)
