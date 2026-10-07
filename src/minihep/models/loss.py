"""Losses (week 3) and matching costs (week 6).

Mirror of src/hepattn/models/loss.py. Two families of functions live here:

* **losses** compare an output with its *matched* target and return one scalar to
  backpropagate.
* **costs** (week 6) compare *every* prediction with *every* target and return a
  (B, num_pred, num_target) matrix. The Hungarian matcher reads it to decide which
  prediction goes with which target. Costs are computed without gradients.

Always work with **logits** (raw scores) rather than probabilities: the
``*_with_logits`` functions combine the sigmoid and the log in one numerically stable
step. ``log(sigmoid(x))`` computed naively becomes ``log(0) = -inf`` for x < -100 or so.
"""

import functools

import torch
import torch.nn.functional as F
from torch import Tensor

# ----------------------------------------------------------------------------- week 3


def hit_bce_loss(logits: Tensor, targets: Tensor, valid: Tensor | None = None, balance: bool = True) -> Tensor:
    """Binary cross-entropy for per-hit classification (the hit filter).

    Args:
        logits: (B, N) raw scores.
        targets: (B, N) bool or float, 1 for "keep this hit".
        valid: optional (B, N) bool; padded hits (False) are left out of the loss entirely.
        balance: if True, weight the positive class by ``1 / mean(targets)`` like hepattn's
            HitFilterTask (``pos_weight = 1 / target.float().mean()``), so a rare positive
            class still matters. Guard against a batch with no positives (use weight 1).

    Use ``F.binary_cross_entropy_with_logits(..., pos_weight=...)``. Return a scalar.
    """
    # TODO(week03): drop padded hits with boolean indexing, compute pos_weight, call F.binary_cross_entropy_with_logits
    raise NotImplementedError("week03 exercise (loss.py)")


def focal_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0, valid: Tensor | None = None) -> Tensor:
    """Focal loss (Lin et al. 2017): BCE scaled by ``(1 - p_t) ** gamma``.

    ``p_t`` is the predicted probability of the *true* class: ``p`` where the target is 1,
    ``1 - p`` where it is 0. Confidently-correct examples (p_t near 1) are scaled towards
    zero, so training concentrates on the hard ones. gamma = 0 gives plain BCE.

    Args:
        logits, targets: same shape, any shape.
        valid: optional bool mask of the same shape; only valid elements count.
    Return the mean over valid elements.
    """
    # TODO(week03): unreduced BCE with logits, p = sigmoid, p_t, scale, mean (over valid elements)
    raise NotImplementedError("week03 exercise (loss.py)")


# ----------------------------------------------------------------------------- week 6
#
# Shapes: B batch, Q predicted objects (queries), T target objects, N hits.
# Every *cost* returns (B, Q, T): entry [b, q, t] is how bad it would be to pair prediction q
# with target t. Costs are computed for all pairs at once with einsum, never with loops.
# Every *loss* returns a scalar computed on already-matched pairs (prediction i <-> target i).


def object_bce_cost(logits: Tensor, targets: Tensor) -> Tensor:
    """Classification cost: (B, Q) logits, (B, T) target validity (float) -> (B, Q, T).

    hepattn's approximation of BCE: ``-p * t - (1 - p) * (1 - t)`` with p = sigmoid(logit),
    broadcast so p varies along Q and t along T.
    """
    # TODO(week06): sigmoid, unsqueeze p to (B, Q, 1) and t to (B, 1, T), combine
    raise NotImplementedError("week06 exercise (loss.py)")


def mask_bce_cost(logits: Tensor, targets: Tensor, input_valid: Tensor | None = None) -> Tensor:
    """Mask BCE cost: (B, Q, N) logits, (B, T, N) target masks -> (B, Q, T), summed over hits.

    For one pair the BCE over hits is ``sum_n [t_n * pos_n + (1 - t_n) * neg_n]`` where
    ``pos = BCE(logit, 1)`` and ``neg = BCE(logit, 0)`` (``F.binary_cross_entropy_with_logits``
    with ``reduction="none"``). Written for all pairs:
        ``einsum("bqn,btn->bqt", pos, t) + einsum("bqn,btn->bqt", neg, 1 - t)``
    Zero ``pos`` and ``neg`` on padded hits first (multiply by ``input_valid.unsqueeze(1)``).
    """
    # TODO(week06): pos and neg per (query, hit), mask padding, two einsums
    raise NotImplementedError("week06 exercise (loss.py)")


def mask_focal_cost(logits: Tensor, targets: Tensor, input_valid: Tensor | None = None, gamma: float = 2.0) -> Tensor:
    """Like mask_bce_cost but each term carries the focal factor:
    ``pos *= (1 - p) ** gamma`` and ``neg *= p ** gamma``. (hepattn: mask_focal_cost.)
    """
    # TODO(week06): focal-weighted pos and neg, mask padding, two einsums
    raise NotImplementedError("week06 exercise (loss.py)")


def mask_dice_cost(logits: Tensor, targets: Tensor, input_valid: Tensor | None = None) -> Tensor:
    """Soft Dice cost ``1 - (2 |P ∩ T| + 1) / (|P| + |T| + 1)`` for every pair, with P = sigmoid(logits).

    |P ∩ T| = ``einsum("bqn,btn->bqt", p, t)``, |P| = p.sum(-1) as (B, Q, 1), |T| = t.sum(-1) as (B, 1, T).
    The +1 smoothing keeps empty masks finite. Zero p on padded hits first.
    """
    # TODO(week06): probabilities, mask padding, numerator via einsum, denominator via broadcasting
    raise NotImplementedError("week06 exercise (loss.py)")


def object_bce_loss(logits: Tensor, targets: Tensor, null_weight: float = 1.0, query_valid: Tensor | None = None) -> Tensor:
    """Matched classification loss: (B, Q) logits vs (B, Q) target validity (after matching).

    Weight the "no object" slots (target 0) by ``null_weight`` and real ones by 1, like hepattn's
    ``sample_weight = target + null_weight * (1 - target)``. If ``query_valid`` is given, padded
    queries get weight 0.
    """
    # TODO(week06): build the sample weight, F.binary_cross_entropy_with_logits(..., weight=...)
    raise NotImplementedError("week06 exercise (loss.py)")


def mask_dice_loss(logits: Tensor, targets: Tensor, object_valid: Tensor, input_valid: Tensor | None = None) -> Tensor:
    """Matched Dice loss, averaged over *real* objects only: (B, Q, N) vs (B, Q, N), object_valid (B, Q).

    Select the real objects with boolean indexing (``logits[object_valid]`` gives (num_real, N)),
    zero the padded hits, then mean of ``1 - (2 sum(p t) + 1) / (sum(p) + sum(t) + 1)``.
    """
    # TODO(week06): index valid objects, mask padded hits (expand input_valid to (B, Q, N) first), dice per object, mean
    raise NotImplementedError("week06 exercise (loss.py)")


def mask_focal_loss(logits: Tensor, targets: Tensor, object_valid: Tensor, input_valid: Tensor | None = None, gamma: float = 2.0) -> Tensor:
    """Matched focal loss for masks, each real object weighted equally.

    Per element: ``bce * (1 - p_t) ** gamma`` (week 3). Zero the padded hits, sum over hits and
    divide by the number of valid hits (``input_valid.sum(-1)``), then average over real objects.
    (hepattn: mask_focal_loss with input_pad_mask.)
    """
    # TODO(week06): focal per element on (B, Q, N), mask hits, normalise per object, select real objects, mean
    raise NotImplementedError("week06 exercise (loss.py)")


def in_float32(cost_fn):
    """Run a cost function in float32 even under ``torch.autocast`` (provided).

    ``.float()`` on the inputs is not enough: autocast casts einsum's inputs back down to bf16/fp16,
    and bf16's 8-bit mantissa can create ties or flip the matching. hepattn computes its costs inside
    ``torch.autocast(device_type="cuda", enabled=False)`` for the same reason (week 8, ``autocast_dtypes``).
    """

    @functools.wraps(cost_fn)
    def wrapper(*args: Tensor | None, **kwargs) -> Tensor:
        device_type = next(a for a in args if isinstance(a, Tensor)).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            args = tuple(a.float() if isinstance(a, Tensor) and a.is_floating_point() else a for a in args)
            return cost_fn(*args, **kwargs)

    return wrapper


COST_FNS = {
    "object_bce": in_float32(object_bce_cost),
    "mask_bce": in_float32(mask_bce_cost),
    "mask_focal": in_float32(mask_focal_cost),
    "mask_dice": in_float32(mask_dice_cost),
}
