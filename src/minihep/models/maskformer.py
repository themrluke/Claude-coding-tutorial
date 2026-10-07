"""Week 6: the full MaskFormer: embed -> encode -> decode -> tasks, plus matching and losses.

Mirror of src/hepattn/models/maskformer.py. Read its ``loss`` docstring: our ``loss`` does
the same four steps (costs, match, permute, losses) for every decoder layer.
"""

import torch
from torch import Tensor, nn

from minihep.models.decoder import MaskFormerDecoder
from minihep.models.matcher import Matcher, permute_outputs


class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
        decoder: MaskFormerDecoder,
        tasks: nn.ModuleList,
        matcher: Matcher | None = None,
        target_object: str = "particle",
        input_sort_field: str | None = None,
    ):
        super().__init__()
        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.tasks = tasks
        self.tasks = tasks
        self.matcher = matcher or Matcher()
        self.target_object = target_object
        self.input_sort_field = input_sort_field
        if any("_" in name for name in self.input_names):
            raise ValueError("Input names cannot contain underscores (they are used to build dict keys)")

    @property
    def input_names(self) -> list[str]:
        return [net.input_name for net in self.input_nets]

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, dict[str, dict[str, Tensor]]]:
        """Returns ``{"layer_0": {...}, ..., "final": {task_name: task_outputs}}``.

        1. Embed each input type and merge into ``key_embed`` / ``key_valid`` / ``key_is_{name}``
           exactly like HitFilter.forward (week 3). If every key is valid set ``x["key_valid"] = None``
           (no mask is cheaper; hepattn does the same).
        2. Encoder (if any), with the merged ``{name}_{input_sort_field}`` values as the sort value.
        3. Unmerge into ``{name}_embed``.
        4. ``x, outputs = self.decoder(x, self.input_names)``.
        5. ``outputs["final"] = {task.name: task(x) for task in self.tasks}``.
        """
        # TODO(week06): embed + merge, encode, unmerge, decode, final task outputs
        raise NotImplementedError("week06 exercise (maskformer.py)")

    def predict(self, outputs: dict) -> dict:
        return {
            layer: {task.name: task.predict(layer_outputs[task.name]) for task in self.tasks if task.name in layer_outputs}
            for layer, layer_outputs in outputs.items()
        }

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """For every layer (each ``layer_i`` and ``final``):

        1. **cost**: sum every task's cost matrices into one (B, Q, T) tensor (tasks with no cost
           return {}).
        2. **match**: ``pred_idx = self.matcher(cost, targets[f"{target_object}_valid"])``.
        3. **permute**: for every task with ``permute_loss``, replace each tensor named in
           ``task.outputs`` by ``permute_outputs(tensor, pred_idx)``. Do this *in place* in
           ``outputs`` so predictions made later line up with the targets too.
        4. **loss**: ``losses[layer][task.name] = task.loss(layer_outputs[task.name], targets)``.

        Return ``(outputs, targets, losses)`` (the same signature as hepattn).
        """
        losses: dict[str, dict[str, dict[str, Tensor]]] = {}
        # TODO(week06): the four steps per layer
        raise NotImplementedError("week06 exercise (maskformer.py)")
        return outputs, targets, losses
