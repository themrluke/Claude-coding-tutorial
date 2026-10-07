"""Week 3: sinusoidal positional encodings for continuous coordinates.

Mirror of src/hepattn/models/posenc.py.

A transformer treats its input as a *set*: it has no idea which token is where. In
hepattn the "position" of a hit is physical (r, eta, phi), and the encoder is told
about it by adding a vector of sines and cosines at many frequencies to each hit's
embedding. Low frequencies tell the model coarse position; high frequencies give
fine detail.

phi is an angle: phi = -pi and phi = +pi are the same place. ``pos_enc_symmetric``
first maps phi to (sin phi, cos phi), which is periodic, so the encoding is too.
"""

import torch
from torch import Tensor, nn


def get_omegas(alpha: float, dim: int, base: float, **kwargs) -> tuple[Tensor, Tensor]:
    """Frequencies, provided. ``dim // 2`` values spaced logarithmically, times ``alpha``.

    For odd ``dim`` the second set has one extra frequency so the total width is ``dim``.
    """
    omega_1 = alpha * torch.logspace(0, 2 / dim - 1, dim // 2, base, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / dim - 1, dim // 2 + 1, base, **kwargs)
    return omega_1, omega_2


def pos_enc(xs: Tensor, dim: int, alpha: float = 1000, base: float = 100) -> Tensor:
    """Encode a tensor of scalars (...,) as (..., dim): ``[sin(x * omega_1), cos(x * omega_2)]``.

    Steps: ``xs.unsqueeze(-1)`` so it broadcasts against the frequencies, compute the two
    halves, ``torch.cat`` on the last dim. Build the omegas on xs's device and dtype.
    """
    # >>> week03: unsqueeze, get_omegas(alpha, dim, base, device=..., dtype=...), sin/cos, cat
    xs = xs.unsqueeze(-1)
    omega_1, omega_2 = get_omegas(alpha, dim, base, device=xs.device, dtype=xs.dtype)
    return torch.cat(((xs * omega_1).sin(), (xs * omega_2).cos()), dim=-1)
    # <<< week03


def pos_enc_symmetric(xs: Tensor, dim: int, alpha: float = 1000, base: float = 100) -> Tensor:
    """Periodic version for angles: ``[sin(sin(x) * omega_1), sin(cos(x) * omega_2)]``.

    Check for yourself that ``pos_enc_symmetric(x) == pos_enc_symmetric(x + 2 pi)``.
    """
    # >>> week03: same as pos_enc but feed sin(x) and cos(x) into the two halves, both through sin
    xs = xs.unsqueeze(-1)
    omega_1, omega_2 = get_omegas(alpha, dim, base, device=xs.device, dtype=xs.dtype)
    return torch.cat(((xs.sin() * omega_1).sin(), (xs.cos() * omega_2).sin()), dim=-1)
    # <<< week03


class PositionEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha: float = 1000, base: float = 100):
        """Encode several coordinates of one input type and concatenate them into a ``dim``-wide vector.

        Each field gets ``dim // len(fields)`` channels; if that does not divide evenly the
        remaining channels are zeros. Fields listed in ``sym_fields`` use pos_enc_symmetric.
        """
        super().__init__()
        self.input_name = input_name
        self.fields = fields
        self.sym_fields = sym_fields or []
        self.dim = dim
        self.alpha = alpha
        self.base = base
        self.per_input_dim = dim // len(fields)
        self.remainder_dim = dim % len(fields)

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """``inputs[f"{input_name}_{field}"]`` is (B, N). Return (B, N, dim)."""
        # >>> week03: one encoding per field (symmetric or not), zero padding for the remainder, cat on the last dim
        encodings = []
        for field in self.fields:
            fn = pos_enc_symmetric if field in self.sym_fields else pos_enc
            encodings.append(fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha, self.base))
        if self.remainder_dim:
            encodings.append(torch.zeros_like(encodings[0])[..., : self.remainder_dim])
        return torch.cat(encodings, dim=-1)
        # <<< week03
