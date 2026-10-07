"""Week 3, exercise 1: autograd and nn.Module mechanics.

The questions you should be able to answer after this file:
  * What does ``loss.backward()`` actually fill in?
  * What is the difference between an ``nn.Parameter``, a buffer and a plain tensor attribute?
  * Why does ``model.eval()`` matter, and why is it *not* the same as ``torch.no_grad()``?
  * What is in a checkpoint's ``state_dict``?

hepattn examples: ``register_buffer("loss_weights", ...)`` in ObjectClassificationTask,
``E2LSHOrderingGrid`` stores frozen random projections as buffers, ``Matcher.forward`` is
decorated with ``@torch.no_grad()``, costs are always ``.detach()``-ed.
"""

import torch
from torch import Tensor, nn


def gradient_of_polynomial(x: Tensor) -> Tensor:
    """Return d/dx of ``sum(x**3 - 2 x)`` at x, computed with autograd (not by hand).

    Do not modify the caller's tensor: work on ``x.detach().clone().requires_grad_(True)``.
    """
    # TODO(week03): make a leaf tensor that requires grad, build the scalar, call backward, return .grad
    raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")


class Standardiser(nn.Module):
    """Learnable scale and shift applied after a fixed, data-derived standardisation.

    ``forward(x) = weight * (x - mean) / std + bias``

    - ``weight`` (ones) and ``bias`` (zeros), both of shape (num_features,): **parameters**
      (they train).
    - ``mean`` and ``std``, shape (num_features,): **buffers** (``self.register_buffer``). They
      are saved in the state_dict and move with ``.to(device)``, but the optimiser never
      touches them. Initialise to zeros and ones.
    - ``fit(x)``: set mean/std from a (num_samples, num_features) tensor, in place
      (``self.mean.copy_(...)``), without recording gradients.
    """

    def __init__(self, num_features: int):
        super().__init__()
        # TODO(week03): two nn.Parameters and two buffers
        raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")

    @torch.no_grad()
    def fit(self, x: Tensor) -> None:
        # TODO(week03): copy_ the column mean and std (clamp the std away from zero) into the buffers
        raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")

    def forward(self, x: Tensor) -> Tensor:
        # TODO(week03): standardise, then scale and shift
        raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Return ``{"trainable": ..., "frozen": ..., "buffers": ...}``: numbers of *elements*, not tensors.

    trainable: parameters with requires_grad, frozen: parameters without, buffers: all buffers.
    (hepattn's SaveConfig callback logs ``sum(p.numel() for p in model.parameters() if p.requires_grad)``.)
    """
    # TODO(week03): iterate module.parameters() and module.buffers(), sum numel()
    raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")


def freeze(module: nn.Module) -> None:
    """Stop every parameter of ``module`` from training (set requires_grad False)."""
    # TODO(week03): loop over parameters and call requires_grad_(False)
    raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")


def sgd_step(module: nn.Module, lr: float) -> None:
    """One step of plain gradient descent by hand: ``p -= lr * p.grad`` for every parameter with a grad.

    Do the update inside ``torch.no_grad()`` (otherwise autograd would try to track the update
    itself) and then zero the gradients (``p.grad = None``), which is what
    ``optimizer.zero_grad()`` does.
    """
    # TODO(week03): update in place under no_grad, then reset .grad
    raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")


def dropout_outputs(p: float, x: Tensor) -> tuple[Tensor, Tensor]:
    """Run the same ``nn.Dropout(p)`` on x in train mode and then in eval mode; return both outputs.

    The test checks that eval mode is the identity and train mode zeroes some entries and
    rescales the rest by 1 / (1 - p). Note ``torch.no_grad()`` would *not* switch dropout off:
    only ``.eval()`` does.
    """
    # TODO(week03): build the layer, .train() then call, .eval() then call
    raise NotImplementedError("week03 exercise (ex01_autograd_and_modules.py)")
