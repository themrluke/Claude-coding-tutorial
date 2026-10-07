"""Tasks: the heads that turn embeddings into outputs, predictions, losses and costs.

Mirror of src/hepattn/models/task.py (that file is 2,300 lines; we write three tasks).

Every task has the same life cycle, driven by the model:

    outputs = task(x)                     forward: embeddings -> raw logits (differentiable)
    preds   = task.predict(outputs)       logits -> probabilities / booleans (no gradient)
    losses  = task.loss(outputs, targets) -> {"loss_name": scalar}
    costs   = task.cost(outputs, targets) -> {"cost_name": (B, num_pred, num_target)}   (week 6)
    masks   = task.attn_mask(outputs)     -> {"hit": (B, num_queries, N) bool}         (week 6)

Week 3 writes HitFilterTask; week 6 adds ObjectValidTask and HitMaskTask.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from minihep.models.dense import Dense
from minihep.models.loss import COST_FNS, focal_loss, hit_bce_loss, mask_dice_loss, mask_focal_loss, object_bce_loss


class Task(nn.Module, ABC):
    """Base class, provided. Same contract as hepattn.models.task.Task."""

    def __init__(self, name: str, has_intermediate_loss: bool = False, permute_loss: bool = True):
        super().__init__()
        self.name = name
        self.has_intermediate_loss = has_intermediate_loss
        self.permute_loss = permute_loss
        self.outputs: list[str] = []  # names of the output tensors that matching should permute

    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]: ...

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}


class HitFilterTask(Task):
    def __init__(self, name: str, input_object: str, target_field: str, dim: int, threshold: float = 0.5, loss_fn: str = "bce"):
        """Classify every hit: is it on a reconstructable particle? (hepattn: HitFilterTask)

        - ``forward``: ``self.net = Dense(dim, 1)`` applied to ``x[f"{input_object}_embed"]``,
          squeezed to (B, N), returned as ``{f"{input_object}_logit": ...}``.
        - ``predict``: ``{f"{input_object}_{target_field}_prob": sigmoid,
          f"{input_object}_{target_field}": prob >= threshold}``.
        - ``loss``: target is ``targets[f"{input_object}_{target_field}"]``, padding mask is
          ``targets[f"{input_object}_valid"]``. ``loss_fn`` "bce" -> hit_bce_loss, "focal" ->
          focal_loss. Return ``{f"{input_object}_{loss_fn}": value}``.
        - ``metrics``: recall (true positives / all positives) and precision (true positives /
          all predicted positives), over valid hits only, as 0-dim tensors keyed "recall" and
          "precision".
        """
        super().__init__(name=name, permute_loss=False)
        if loss_fn not in {"bce", "focal"}:
            raise ValueError(f"Unknown loss_fn {loss_fn!r}, expected 'bce' or 'focal'")
        self.input_object = input_object
        self.target_field = target_field
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week03: run the net on the embeddings and squeeze the last dim
        return {f"{self.input_object}_logit": self.net(x[f"{self.input_object}_embed"]).squeeze(-1)}
        # <<< week03

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week03: sigmoid, then threshold
        probs = outputs[f"{self.input_object}_logit"].detach().sigmoid()
        return {
            f"{self.input_object}_{self.target_field}_prob": probs,
            f"{self.input_object}_{self.target_field}": probs >= self.threshold,
        }
        # <<< week03

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week03: pick the target and the valid mask, call the chosen loss function
        logits = outputs[f"{self.input_object}_logit"]
        target = targets[f"{self.input_object}_{self.target_field}"]
        valid = targets.get(f"{self.input_object}_valid")
        if self.loss_fn == "bce":
            value = hit_bce_loss(logits, target, valid=valid)
        else:
            value = focal_loss(logits, target, valid=valid)
        return {f"{self.input_object}_{self.loss_fn}": value}
        # <<< week03

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week03: count true positives over valid hits; guard the divisions with clamp_min(1)
        key = f"{self.input_object}_{self.target_field}"
        pred, true = preds[key], targets[key].bool()
        valid = targets.get(f"{self.input_object}_valid", torch.ones_like(true))
        pred, true = pred & valid, true & valid
        tp = (pred & true).sum()
        return {
            "recall": tp / true.sum().clamp_min(1),
            "precision": tp / pred.sum().clamp_min(1),
        }
        # <<< week03


# ----------------------------------------------------------------------------- week 6


class ObjectValidTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        dim: int,
        loss_weight: float = 1.0,
        cost_weight: float = 1.0,
        null_weight: float = 1.0,
    ):
        """Does query q correspond to a real object? (hepattn: ObjectClassificationTask with num_classes=1)

        - forward: ``self.net = Dense(dim, 1)`` on ``x[f"{input_object}_embed"]`` (the queries), squeezed to
          (B, Q), returned as ``{f"{output_object}_logit": ...}``. Set ``self.outputs`` to that key's name in
          ``__init__`` so the model knows which tensors to permute after matching.
        - predict: ``{f"{output_object}_valid_prob": sigmoid, f"{output_object}_valid": prob >= 0.5}``.
        - cost: ``{"object_bce": cost_weight * object_bce_cost(logits, targets[f"{target_object}_valid"].float())}``
          computed on detached float32 logits.
        - loss: ``{"object_bce": loss_weight * object_bce_loss(logits, targets[f"{target_object}_valid"], null_weight)}``.
        """
        super().__init__(name=name, has_intermediate_loss=True)
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.null_weight = null_weight
        # >>> week06: the classification net and self.outputs
        self.net = Dense(dim, 1)
        self.outputs = [f"{output_object}_logit"]
        # <<< week06

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: one logit per query
        return {f"{self.output_object}_logit": self.net(x[f"{self.input_object}_embed"]).squeeze(-1)}
        # <<< week06

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: probability and boolean prediction
        prob = outputs[f"{self.output_object}_logit"].detach().sigmoid()
        return {f"{self.output_object}_valid_prob": prob, f"{self.output_object}_valid": prob >= 0.5}
        # <<< week06

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: detached float32 logits against the target validity
        logits = outputs[f"{self.output_object}_logit"].detach().float()
        return {"object_bce": self.cost_weight * COST_FNS["object_bce"](logits, targets[f"{self.target_object}_valid"].float())}
        # <<< week06

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: weighted object BCE
        logits = outputs[f"{self.output_object}_logit"]
        return {"object_bce": self.loss_weight * object_bce_loss(logits, targets[f"{self.target_object}_valid"], self.null_weight)}
        # <<< week06


class HitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        dim: int,
        losses: dict[str, float],
        costs: dict[str, float],
        mask_attention_threshold: float = 0.5,
        pred_threshold: float = 0.5,
    ):
        """Which hits belong to query q? (hepattn: ObjectHitMaskTask)

        Output key: ``f"{output_object}_{input_constituent}_logit"``, e.g. ``track_hit_logit`` (B, Q, N).
        Target key: ``f"{target_object}_{input_constituent}_valid"``, e.g. ``particle_hit_valid`` (B, T, N).
        Hit padding: ``f"{input_constituent}_valid"`` (B, N), in ``x`` for forward and in ``targets`` for loss/cost.

        - forward: ``mask_tokens = self.object_net(query_embeds)`` (``Dense(dim, dim)``), then
          ``einsum("bqd,bnd->bqn", mask_tokens, hit_embeds)``. Fill padded hits with
          ``torch.finfo(dtype).min`` (out of place: ``masked_fill``).
        - attn_mask: ``{input_constituent: sigmoid(logits.detach()) >= mask_attention_threshold}``.
          This is the *mask attention* of MaskFormer: in the next decoder layer, query q may only
          look at the hits it currently thinks it owns.
        - predict: ``{f"{output_object}_{input_constituent}_valid_prob": ..., f"..._valid": prob >= pred_threshold}``.
        - cost: for each ``name, weight`` in ``costs`` (keys of COST_FNS, e.g. "mask_dice", "mask_focal"):
          ``weight * COST_FNS[name](logits.detach().float(), targets.float(), input_valid)``.
        - loss: "mask_dice" -> mask_dice_loss, "mask_focal" -> mask_focal_loss, each times its weight,
          using ``targets[f"{target_object}_valid"]`` as object_valid.
        """
        super().__init__(name=name, has_intermediate_loss=True)
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.mask_attention_threshold = mask_attention_threshold
        self.pred_threshold = pred_threshold
        self.logit_key = f"{output_object}_{input_constituent}_logit"
        self.target_key = f"{target_object}_{input_constituent}_valid"
        self.outputs = [self.logit_key]
        self.object_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: mask tokens, einsum with the hit embeddings, mask padded hits
        mask_tokens = self.object_net(x[f"{self.input_object}_embed"])
        logits = torch.einsum("bqd,bnd->bqn", mask_tokens, x[f"{self.input_constituent}_embed"])
        valid = x.get(f"{self.input_constituent}_valid")
        if valid is not None:
            logits = logits.masked_fill(~valid.unsqueeze(-2), torch.finfo(logits.dtype).min)
        return {self.logit_key: logits}
        # <<< week06

    def attn_mask(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: threshold the detached probabilities
        return {self.input_constituent: outputs[self.logit_key].detach().sigmoid() >= self.mask_attention_threshold}
        # <<< week06

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: probabilities and booleans
        prob = outputs[self.logit_key].detach().sigmoid()
        stem = f"{self.output_object}_{self.input_constituent}_valid"
        return {f"{stem}_prob": prob, stem: prob >= self.pred_threshold}
        # <<< week06

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: weighted costs from COST_FNS on detached float32 logits
        logits = outputs[self.logit_key].detach().float()
        target = targets[self.target_key].float()
        valid = targets.get(f"{self.input_constituent}_valid")
        return {name: weight * COST_FNS[name](logits, target, valid) for name, weight in self.costs.items()}
        # <<< week06

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # >>> week06: dice and/or focal on the real objects
        logits = outputs[self.logit_key]
        target = targets[self.target_key]
        object_valid = targets[f"{self.target_object}_valid"]
        valid = targets.get(f"{self.input_constituent}_valid")
        fns = {"mask_dice": mask_dice_loss, "mask_focal": mask_focal_loss}
        return {name: weight * fns[name](logits, target, object_valid, valid) for name, weight in self.losses.items()}
        # <<< week06
