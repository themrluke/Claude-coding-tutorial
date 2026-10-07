"""Week 2, exercise 1: NumPy drills on detector-shaped data.

No Python loops over hits or particles in this file: every function should be a few
lines of array operations. That is the habit that makes PyTorch code readable later.
"""

import numpy as np


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    """Map any angle into [-pi, pi). Works on arrays of any shape."""
    # TODO(week02): shift by pi, take the remainder modulo 2 pi, shift back
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed smallest angle from b to a, in [-pi, pi). ``delta_phi(3.1, -3.1)`` is about -0.083."""
    # TODO(week02): wrap the plain difference
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def eta_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pseudorapidity of a point seen from the origin: eta = -log(tan(theta / 2)), theta = polar angle."""
    # TODO(week02): theta = arctan2(r, z) with r = hypot(x, y)
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of a (N, D) and every row of b (M, D). Returns (N, M).

    Use broadcasting: ``a[:, None, :] - b[None, :, :]`` has shape (N, M, D).
    """
    # TODO(week02): broadcast, square, sum over the last axis, sqrt
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Return ``inv`` such that ``x[perm][inv] == x`` for any array x.

    This is how hepattn's encoder undoes its sort by phi (``torch.argsort(x_sort_idx)``).
    Do it two ways in your head: ``np.argsort(perm)``, or a scatter ``inv[perm] = arange(n)``.
    """
    # TODO(week02): scatter arange into the permuted positions (or argsort)
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def ranks(values: np.ndarray) -> np.ndarray:
    """Position each element would take if ``values`` were sorted: ``ranks([30, 10, 20]) == [2, 0, 1]``.

    This "argsort of argsort" trick is all over src/hepattn/models/ordering.py.
    Use a stable sort so ties keep their original order.
    """
    # TODO(week02): argsort twice (kind="stable")
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def innermost_hit_mask(particle_id: np.ndarray, r: np.ndarray) -> np.ndarray:
    """True for the smallest-r hit of each particle, False elsewhere and for noise (particle_id == 0).

    Without pandas! Hint: ``np.lexsort((r, particle_id))`` sorts by particle id, then by r
    within a particle; the first element of each particle's run is its innermost hit.
    A run starts wherever the sorted particle id differs from the previous one.
    """
    # TODO(week02): lexsort, find run starts, scatter True back to the original positions, drop noise
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")


def hits_per_particle(particle_id: np.ndarray, valid_ids: np.ndarray) -> np.ndarray:
    """For each id in ``valid_ids``, count how many entries of ``particle_id`` equal it.

    Returns an int array with the same length as ``valid_ids``. Hint: ``np.unique(...,
    return_counts=True)`` then ``np.searchsorted``, or a broadcast comparison and ``sum``.
    """
    # TODO(week02): compare (len(valid_ids), 1) against (1, num_hits) and sum over hits
    raise NotImplementedError("week02 exercise (ex01_numpy.py)")
