"""Week 2, exercise 1: NumPy drills on detector-shaped data.

No Python loops over hits or particles in this file: every function should be a few
lines of array operations. That is the habit that makes PyTorch code readable later.
"""

import numpy as np


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    """Map any angle into [-pi, pi). Works on arrays of any shape."""
    # >>> week02: shift by pi, take the remainder modulo 2 pi, shift back
    return (phi + np.pi) % (2 * np.pi) - np.pi
    # <<< week02


def delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed smallest angle from b to a, in [-pi, pi). ``delta_phi(3.1, -3.1)`` is about -0.083."""
    # >>> week02: wrap the plain difference
    return wrap_phi(a - b)
    # <<< week02


def eta_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pseudorapidity of a point seen from the origin: eta = -log(tan(theta / 2)), theta = polar angle."""
    # >>> week02: theta = arctan2(r, z) with r = hypot(x, y)
    theta = np.arctan2(np.hypot(x, y), z)
    return -np.log(np.tan(theta / 2))
    # <<< week02


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of a (N, D) and every row of b (M, D). Returns (N, M).

    Use broadcasting: ``a[:, None, :] - b[None, :, :]`` has shape (N, M, D).
    """
    # >>> week02: broadcast, square, sum over the last axis, sqrt
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    # <<< week02


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Return ``inv`` such that ``x[perm][inv] == x`` for any array x.

    This is how hepattn's encoder undoes its sort by phi (``torch.argsort(x_sort_idx)``).
    Do it two ways in your head: ``np.argsort(perm)``, or a scatter ``inv[perm] = arange(n)``.
    """
    # >>> week02: scatter arange into the permuted positions (or argsort)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv
    # <<< week02


def ranks(values: np.ndarray) -> np.ndarray:
    """Position each element would take if ``values`` were sorted: ``ranks([30, 10, 20]) == [2, 0, 1]``.

    This "argsort of argsort" trick is all over src/hepattn/models/ordering.py.
    Use a stable sort so ties keep their original order.
    """
    # >>> week02: argsort twice (kind="stable")
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")
    # <<< week02


def innermost_hit_mask(particle_id: np.ndarray, r: np.ndarray) -> np.ndarray:
    """True for the smallest-r hit of each particle, False elsewhere and for noise (particle_id == 0).

    Without pandas! Hint: ``np.lexsort((r, particle_id))`` sorts by particle id, then by r
    within a particle; the first element of each particle's run is its innermost hit.
    A run starts wherever the sorted particle id differs from the previous one.
    """
    # >>> week02: lexsort, find run starts, scatter True back to the original positions, drop noise
    order = np.lexsort((r, particle_id))
    sorted_ids = particle_id[order]
    is_start = np.ones(len(order), dtype=bool)
    is_start[1:] = sorted_ids[1:] != sorted_ids[:-1]
    mask = np.zeros(len(order), dtype=bool)
    mask[order[is_start]] = True
    return mask & (particle_id != 0)
    # <<< week02


def hits_per_particle(particle_id: np.ndarray, valid_ids: np.ndarray) -> np.ndarray:
    """For each id in ``valid_ids``, count how many entries of ``particle_id`` equal it.

    Returns an int array with the same length as ``valid_ids``. Hint: ``np.unique(...,
    return_counts=True)`` then ``np.searchsorted``, or a broadcast comparison and ``sum``.
    """
    # >>> week02: compare (len(valid_ids), 1) against (1, num_hits) and sum over hits
    return (valid_ids[:, None] == particle_id[None, :]).sum(axis=1)
    # <<< week02
