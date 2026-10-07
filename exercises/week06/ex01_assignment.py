"""Week 6, exercise 1: the assignment problem, three ways.

Given a cost matrix C (n_targets x n_preds, n_targets <= n_preds), choose a distinct
prediction for every target to minimise the total cost.

  (a) brute force over every permutation: correct, O(n!), useless beyond n ~ 8
  (b) greedy: repeatedly take the globally cheapest remaining pair: fast, often wrong
  (c) scipy.optimize.linear_sum_assignment (Hungarian / Jonker-Volgenant): exact, O(n³)

A MaskFormer on a TrackML event solves a ~2600 x 3900 problem for every decoder layer at
every training step. The memory note "Hungarian matching overhead" in your vault is
about exactly this cost: ~15 s per event with scipy, ~1 s with lap1015.
"""

import itertools

import numpy as np
import scipy.optimize


def brute_force_assignment(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(cols, total)``: ``cols[t]`` is the prediction for target t; ``total`` the minimal cost.

    Try every ordered choice of n_targets distinct predictions (``itertools.permutations(range(n_preds), n_targets)``).
    """
    # TODO(week06): loop over permutations, keep the cheapest
    raise NotImplementedError("week06 exercise (ex01_assignment.py)")


def greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Repeatedly pick the smallest entry whose row (target) and column (prediction) are both unused."""
    # TODO(week06): sort all entries once (np.argsort on cost.ravel()), walk through them
    raise NotImplementedError("week06 exercise (ex01_assignment.py)")


def scipy_assignment(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Use ``scipy.optimize.linear_sum_assignment``. Same return convention as the others."""
    # TODO(week06): rows come back sorted, so the column array is already indexed by target
    raise NotImplementedError("week06 exercise (ex01_assignment.py)")


def greedy_counterexample() -> np.ndarray:
    """Return a 2x2 cost matrix on which greedy is strictly worse than optimal.

    Think: greedy grabs the single cheapest pair even if that forces a terrible second pair.
    """
    # TODO(week06): one tiny entry whose row/column partner is expensive
    raise NotImplementedError("week06 exercise (ex01_assignment.py)")
