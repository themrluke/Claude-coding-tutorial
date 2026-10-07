"""Week 6: the MaskFormer decoder.

Mirror of src/hepattn/models/decoder.py (MaskFormerDecoder, MaskFormerDecoderLayer).

The decoder holds Q *queries*, one per potential track. Each layer:

  1. runs every task on the current query/hit embeddings (intermediate outputs, each with
     its own loss: "deep supervision"),
  2. asks the tasks for an attention mask: query q may only attend to the hits its current
     mask prediction claims (mask attention, the key idea of Mask2Former),
  3. updates the queries: cross-attention to the hits (masked), a dense block, and
     self-attention between queries (so two queries can "agree" not to chase the same track),
  4. optionally updates the hits by cross-attending to the queries (bidirectional).
"""

import torch
from torch import Tensor, nn

from minihep.models.attention import Attention
from minihep.models.dense import Dense
from minihep.models.encoder import Residual


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self, dim: int, norm: str = "LayerNorm", dense_kwargs: dict | None = None, attn_kwargs: dict | None = None, bidirectional_ca: bool = False
    ):
        """Residual blocks (all pre-norm with ``norm``):

        - ``q_ca``: Attention, queries attend to hits (cross-attention)
        - ``q_dense``: Dense on the queries
        - ``q_sa``: Attention, queries attend to queries (self-attention)
        - if ``bidirectional_ca``: ``kv_ca`` (hits attend to queries) and ``kv_dense``
        """
        super().__init__()
        dense_kwargs = dense_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        self.bidirectional_ca = bidirectional_ca
        # TODO(week06): build the Residual-wrapped blocks
        raise NotImplementedError("week06 exercise (decoder.py)")

    def forward(self, q: Tensor, kv: Tensor, attn_mask: Tensor | None = None, kv_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """q (B, Q, D), kv (B, N, D), attn_mask (B, Q, N) bool or None, kv_mask (B, N) bool or None.

        - ``q = q_ca(q, k=kv, v=kv, attn_mask=attn_mask, kv_mask=kv_mask)``, then q_dense, then ``q_sa(q)``.
        - If bidirectional: ``kv = kv_ca(kv, k=q, v=q, attn_mask=<the mask transposed to (B, N, Q)>)``,
          then kv_dense. A hit that no query claims would have a fully-masked row, so un-mask those
          rows (hits with no owner attend to every query).
        Return ``(q, kv)``.
        """
        # TODO(week06): query update, then the optional hit update
        raise NotImplementedError("week06 exercise (decoder.py)")


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        dim: int,
        num_layers: int,
        layer_kwargs: dict | None = None,
        mask_attention: bool = True,
        unmask_all_false: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.mask_attention = mask_attention
        self.unmask_all_false = unmask_all_false
        self.layers = nn.ModuleList([MaskFormerDecoderLayer(dim, **(layer_kwargs or {})) for _ in range(num_layers)])
        # One learned starting vector per query. hepattn: self.initial_queries (or dynamic queries,
        # chosen from the encoded hits by a first-hit classifier: decoder.initialize_dynamic_queries).
        self.initial_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.tasks: nn.ModuleList | None = None  # set by MaskFormer, as in hepattn

    def build_attn_mask(self, x: dict[str, Tensor], layer_outputs: dict[str, dict[str, Tensor]], input_names: list[str]) -> Tensor | None:
        """Combine the tasks' attention masks into one (B, Q, N_total) mask over the merged key sequence.

        - Each task's ``attn_mask(outputs)`` returns ``{input_name: (B, Q, N_input) bool}``. If several
          tasks give a mask for the same input, OR them.
        - Place each input's mask into the key positions that input occupies: those where
          ``x[f"key_is_{name}"]`` is True (start from an all-False (B, Q, N_total) tensor).
        - Return None if no task produced a mask or mask_attention is off.
        - If ``unmask_all_false``: rows with no True entry become all True (week 4).
        - ``.detach()`` the result: a mask is a hard decision, gradients cannot flow through it.
        """
        if not self.mask_attention:
            return None
        # TODO(week06): OR per input, scatter into the merged key axis, unmask empty rows, detach
        raise NotImplementedError("week06 exercise (decoder.py)")

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict[str, dict[str, Tensor]]]]:
        """Run the decoder. Returns ``(x, outputs)`` with ``outputs[f"layer_{i}"][task.name]`` for every layer.

        1. ``x["query_embed"] = self.initial_queries.expand(B, -1, -1)`` (a view: week 2).
        2. For each layer i:
             - run every task on x into ``outputs[f"layer_{i}"]``
             - ``attn_mask = self.build_attn_mask(x, outputs[f"layer_{i}"], input_names)``
             - ``x["query_embed"], x["key_embed"] = layer(x["query_embed"], x["key_embed"], attn_mask, x.get("key_valid"))``
             - unmerge: ``x[f"{name}_embed"] = x["key_embed"][:, x[f"key_is_{name}"]]`` for each input name
        The *final* task outputs are computed by MaskFormer after the last layer.
        """
        # TODO(week06): the loop described above
        raise NotImplementedError("week06 exercise (decoder.py)")
