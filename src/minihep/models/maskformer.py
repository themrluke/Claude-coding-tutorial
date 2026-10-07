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
        # >>> week06: embed + merge, encode, unmerge, decode, final task outputs
        x: dict[str, Tensor] = {}
        for net in self.input_nets:
            x[f"{net.input_name}_embed"] = net(inputs)
            x[f"{net.input_name}_valid"] = inputs[f"{net.input_name}_valid"]
        device = x[f"{self.input_names[0]}_valid"].device
        for name in self.input_names:
            x[f"key_is_{name}"] = torch.cat(
                [torch.full((inputs[f"{other}_valid"].shape[-1],), other == name, dtype=torch.bool, device=device) for other in self.input_names]
            )
        x["key_embed"] = torch.cat([x[f"{name}_embed"] for name in self.input_names], dim=-2)
        key_valid = torch.cat([x[f"{name}_valid"] for name in self.input_names], dim=-1)
        x["key_valid"] = None if key_valid.all() else key_valid

        if self.encoder is not None:
            sort_value = None
            if self.input_sort_field is not None:
                sort_value = torch.cat([inputs[f"{name}_{self.input_sort_field}"] for name in self.input_names], dim=-1)
            x["key_embed"] = self.encoder(x["key_embed"], sort_value, kv_mask=x["key_valid"])
        for name in self.input_names:
            x[f"{name}_embed"] = x["key_embed"][:, x[f"key_is_{name}"]]

        x, outputs = self.decoder(x, self.input_names)
        outputs["final"] = {task.name: task(x) for task in self.tasks}
        return outputs
        # <<< week06

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
        # >>> week06: the four steps per layer
        target_valid = targets[f"{self.target_object}_valid"]
        for layer_name, layer_outputs in outputs.items():
            cost = None
            for task in self.tasks:
                if task.name not in layer_outputs:
                    continue
                for c in task.cost(layer_outputs[task.name], targets).values():
                    cost = c if cost is None else cost + c
            if cost is not None:
                pred_idx = self.matcher(cost, target_valid)
                for task in self.tasks:
                    if task.name not in layer_outputs or not task.permute_loss:
                        continue
                    for key in task.outputs:
                        layer_outputs[task.name][key] = permute_outputs(layer_outputs[task.name][key], pred_idx)
            losses[layer_name] = {task.name: task.loss(layer_outputs[task.name], targets) for task in self.tasks if task.name in layer_outputs}
        # <<< week06
        return outputs, targets, losses
