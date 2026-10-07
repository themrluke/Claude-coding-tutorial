"""Week 3: the feed-forward network used everywhere in hepattn.

Mirror of src/hepattn/models/dense.py and models/activation.py. Read both after you
finish: the real version is almost line-for-line what you write here.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """Gated activation: split the input in half along the last dim into (a, b), return ``a * silu(b)``.

    Because it halves the width, the Linear layer *before* a SwiGLU must output twice the
    hidden size. That is what ``gate`` is for in Dense below. (Shazeer 2020, "GLU Variants
    Improve Transformer".)
    """

    def forward(self, x: Tensor) -> Tensor:
        # >>> week03: torch.chunk into two halves on the last dim, multiply the first by silu of the second
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * F.silu(x2)
        # <<< week03


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: nn.Module | str | None = None,
        final_activation: nn.Module | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """A multi-layer perceptron: Linear -> activation (-> dropout) -> ... -> Linear (-> final_activation).

        Args:
            input_size: Input features.
            output_size: Output features; defaults to input_size.
            hidden_layers: Sizes of the hidden layers. Default: one layer of input_size * hidden_dim_scale.
            hidden_dim_scale: See above.
            activation: An nn.Module, the string "SwiGLU", or None for nn.SiLU().
            final_activation: Optional module applied after the last Linear.
            dropout: If > 0, add nn.Dropout after each hidden activation.
            bias: Bias in every Linear.

        Build ``self.net = nn.Sequential(...)``. With SwiGLU, each *hidden* Linear outputs
        twice its nominal size (the activation halves it back). Store ``self.input_size`` and
        ``self.output_size``: tasks check ``net.output_size`` (see ObjectClassificationTask).
        """
        super().__init__()
        # >>> week03: resolve defaults, then build the list of layers and wrap it in nn.Sequential
        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]
        if activation is None:
            activation = nn.SiLU()
        if activation == "SwiGLU":
            activation = SwiGLU()
        gate = isinstance(activation, SwiGLU)

        self.input_size = input_size
        self.output_size = output_size

        layers: list[nn.Module] = []
        sizes = [input_size, *hidden_layers]
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.extend((nn.Linear(in_dim, out_dim * 2 if gate else out_dim, bias=bias), activation))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-1], output_size, bias=bias))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)
        # <<< week03

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
