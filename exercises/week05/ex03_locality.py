"""Week 5, exercise 3: why hepattn sorts hits by phi before windowed attention.

A sliding window lets each token attend only to its ``window_size`` nearest neighbours *in
the sequence*. That only helps if hits that belong together (same particle) are near each
other in the sequence. Real events come in arbitrary order, so hepattn sorts them first
(``input_sort_field: phi``), and its research branches try smarter orderings
(models/ordering.py, the HEPT E2LSH ordering with OR amplification).

The figure of merit you compute here is *pair recall*: of all pairs of hits from the same
particle, what fraction end up within ``window_size // 2`` of each other in the ordering?
With full attention it is 1. Your notebook for this week plots it against window size for
random order, phi order and eta-phi cell order.
"""

import numpy as np


def positions_from_sort_value(sort_value: np.ndarray) -> np.ndarray:
    """Position of each hit in the sequence after sorting by ``sort_value`` (the rank). Stable sort."""
    # TODO(week05): argsort of argsort (you wrote this in week 2)
    raise NotImplementedError("week05 exercise (ex03_locality.py)")


def pair_recall(particle_id: np.ndarray, positions: np.ndarray, window_size: int, seq_len: int | None = None) -> float:
    """Fraction of same-particle hit pairs (i < j, particle_id != 0) with ``|pos_i - pos_j| <= window_size // 2``.

    If ``seq_len`` is given the window wraps around (ring distance ``min(d, seq_len - d)``),
    matching ``window_wrap: true`` in hepattn configs. Return 1.0 if there are no pairs.
    Vectorise per particle: for one particle with hit positions p, the pairwise distances are
    ``abs(p[:, None] - p[None, :])``; take the upper triangle (``np.triu_indices``).
    """
    # TODO(week05): loop over unique non-zero particle ids, count pairs and pairs within the window
    raise NotImplementedError("week05 exercise (ex03_locality.py)")


def cell_sort_value(eta: np.ndarray, phi: np.ndarray, num_phi_bins: int, eta_range: float = 4.0) -> np.ndarray:
    """A crude 2D ordering: chop phi into ``num_phi_bins`` slices, order slice by slice, by eta inside a slice.

    Return one float per hit: ``phi_bin * (2 * eta_range + 1) + (eta + eta_range)``, with
    ``phi_bin = floor((phi + pi) / (2 pi) * num_phi_bins)`` clipped to [0, num_phi_bins - 1].
    This is the idea behind models/ordering.py (which uses equal-occupancy quantile bins and a
    random projection instead of plain eta inside each cell).
    """
    # TODO(week05): integer phi bin, then combine with shifted eta so bins never overlap
    raise NotImplementedError("week05 exercise (ex03_locality.py)")
