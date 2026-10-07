"""Week 3 (and week 4): the hit-filter model.

Mirror of src/hepattn/models/hitfilter.py. In week 3 you train it with ``encoder=None``
(each hit is classified on its own features: an MLP). In week 4 you plug in your
transformer Encoder and each hit can look at the others. Compare the two!
"""

import torch
from torch import Tensor, nn


class HitFilter(nn.Module):
    def __init__(self, input_nets: nn.ModuleList, tasks: nn.ModuleList, encoder: nn.Module | None = None, input_sort_field: str | None = None):
        super().__init__()
        self.input_nets = input_nets
        self.encoder = encoder
        self.tasks = tasks
        self.input_sort_field = input_sort_field

    @property
    def input_names(self) -> list[str]:
        return [net.input_name for net in self.input_nets]

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, dict[str, dict[str, Tensor]]]:
        """Embed, merge, encode, unmerge, run the tasks. Returns ``{"final": {task.name: task_outputs}}``.

        1. For each input net: ``x[f"{name}_embed"] = net(inputs)``, ``x[f"{name}_valid"] = inputs[f"{name}_valid"]``.
        2. Merge all input types along the token axis into ``x["key_embed"]`` and ``x["key_valid"]``,
           and record which tokens came from which input in ``x[f"key_is_{name}"]`` (1D bool,
           length = total tokens). With a single input type this is trivial, but write the general
           version: hepattn feeds pixel and strip hits as separate input types.
        3. If there is an encoder: if ``input_sort_field`` is set, also merge
           ``inputs[f"{name}_{input_sort_field}"]`` into a (B, N) tensor and pass it as the second
           argument; then ``x["key_embed"] = self.encoder(x["key_embed"], sort_value)``.
           (Week 3 has no encoder, so this step is skipped.)
        4. Unmerge: ``x[f"{name}_embed"] = x["key_embed"][:, x[f"key_is_{name}"]]``.
        5. ``{"final": {task.name: task(x) for task in self.tasks}}``.
        """
        # TODO(week03): implement the five steps (merge with torch.cat on dim -2 for embeddings, -1 for masks)
        raise NotImplementedError("week03 exercise (hitfilter.py)")

    def predict(self, outputs: dict) -> dict:
        return {"final": {task.name: task.predict(outputs["final"][task.name]) for task in self.tasks}}

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        losses = {"final": {task.name: task.loss(outputs["final"][task.name], targets) for task in self.tasks}}
        return outputs, targets, losses
