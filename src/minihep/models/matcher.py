"""Week 6: Hungarian matching between predicted objects and target objects.

Mirror of src/hepattn/models/matcher.py (the real one adds solver auto-selection,
a C++ solver ``lap1015`` and thread/process pools to match many events in parallel).

The model predicts Q objects in no particular order; the event has T <= Q real targets.
Before any loss can be computed we must decide which prediction is "supposed to be" which
target. Hungarian matching picks the one-to-one assignment with the smallest total cost
(the linear assignment problem), using ``scipy.optimize.linear_sum_assignment``.

Convention (same as hepattn): the matcher returns ``pred_idx`` of shape (B, Q) such that
``outputs[b, pred_idx[b]]`` puts the prediction matched to target t at position t. Positions
t >= (number of real targets) hold the unmatched predictions, in increasing order. After that
permutation, "prediction i" and "target i" can be compared directly and ``particle_valid``
says which positions are real.
"""

import numpy as np
import scipy.optimize
import torch
from torch import Tensor, nn


def match_one(cost: np.ndarray, num_valid_targets: int) -> np.ndarray:
    """Solve one event. ``cost`` is (Q, T) as a numpy float array. Return a length-Q int permutation.

    1. Keep only the real targets and put targets on rows: ``cost[:, :num_valid_targets].T``
       (shape (T_valid, Q); hepattn transposes the same way).
    2. ``rows, cols = linear_sum_assignment(that)``: ``cols[t]`` is the prediction for target t
       (rows come back sorted, 0..T_valid-1).
    3. Append every prediction not in ``cols``, in increasing order, so the result is a full
       permutation of 0..Q-1.
    """
    # >>> week06: transpose the valid part, solve, append the unused predictions
    _, cols = scipy.optimize.linear_sum_assignment(cost[:, :num_valid_targets].T)
    unused = np.ones(cost.shape[0], dtype=bool)
    unused[cols] = False
    return np.concatenate([cols, np.flatnonzero(unused)]).astype(np.int64)
    # <<< week06


class Matcher(nn.Module):
    """Batched wrapper, provided apart from forward."""

    @torch.no_grad()
    def forward(self, costs: Tensor, target_valid: Tensor) -> Tensor:
        """``costs`` (B, Q, T) and ``target_valid`` (B, T) -> ``pred_idx`` (B, Q) int64 on costs' device.

        Move the costs to CPU float32 numpy (``.detach().float().cpu().numpy()``): the solver is not
        differentiable and runs on the CPU. NaN or inf costs make scipy raise, so check with
        ``np.isfinite`` and raise a ValueError with a helpful message instead.
        Real targets come first in ``target_valid`` (the dataset pads at the end), so the
        number of real targets is ``target_valid[b].sum()``.
        """
        # >>> week06: to numpy, finite check, match_one per batch element, stack, back to a tensor
        costs_np = costs.detach().float().cpu().numpy()
        if not np.isfinite(costs_np).all():
            raise ValueError("Matching costs contain NaN or inf: check the model outputs and the padding masks")
        lengths = target_valid.sum(-1).tolist()
        idx = np.stack([match_one(costs_np[b], int(lengths[b])) for b in range(costs_np.shape[0])])
        return torch.from_numpy(idx).to(costs.device)
        # <<< week06


def permute_outputs(x: Tensor, pred_idx: Tensor) -> Tensor:
    """Apply the matching to any (B, Q, ...) tensor. Provided (you wrote it in week 2)."""
    batch_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
    return x[batch_idx, pred_idx]
