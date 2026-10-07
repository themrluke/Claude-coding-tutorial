"""Week 3: turn a dict of per-hit features into one embedding per hit.

Mirror of src/hepattn/models/input.py (InputNet) and utils/tensor_utils.concat_tensors.
"""

import torch
from torch import Tensor, nn


def concat_tensors(tensors: list[Tensor]) -> Tensor:
    """Concatenate along the last dim; any 2D (B, N) tensor is first made (B, N, 1). Provided."""
    return torch.cat([t.unsqueeze(-1) if t.ndim == 2 else t for t in tensors], dim=-1)


class InputNet(nn.Module):
    def __init__(self, input_name: str, net: nn.Module, fields: list[str], posenc: nn.Module | None = None):
        """Embed one input type.

        Args:
            input_name: e.g. "hit". The features read are ``inputs[f"{input_name}_{field}"]``.
            net: maps the stacked features (B, N, len(fields)) to (B, N, dim), usually a Dense.
            fields: which features to stack, in this order.
            posenc: optional module whose output (B, N, dim) is *added* to the embedding.
        """
        super().__init__()
        self.input_name = input_name
        self.net = net
        self.fields = fields
        self.posenc = posenc

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        # TODO(week03): stack the fields with concat_tensors, run net, add posenc(inputs) if there is one
        raise NotImplementedError("week03 exercise (input.py)")
