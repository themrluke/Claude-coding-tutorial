"""Event displays (provided)."""

import matplotlib.pyplot as plt
import numpy as np


def _colours(labels: np.ndarray) -> np.ndarray:
    """One colour per label; label 0 (noise / unassigned) is light grey."""
    cmap = plt.get_cmap("tab20")
    unique = {lab: i for i, lab in enumerate(sorted(set(labels.tolist()) - {0}))}
    colours = np.array([cmap(unique[lab] % 20) if lab != 0 else (0.8, 0.8, 0.8, 1.0) for lab in labels.tolist()])
    return colours


def event_display(x, y, z, labels, title: str = "", axes=None):
    """Two panels: transverse plane (x, y) and longitudinal (z, r). Points coloured by ``labels``."""
    x, y, z, labels = (np.asarray(a) for a in (x, y, z, labels))
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colours = _colours(labels)
    r = np.hypot(x, y)
    for ax, (u, v, xl, yl) in zip(axes, ((x, y, "x [m]", "y [m]"), (z, r, "z [m]", "r [m]"))):
        ax.scatter(u, v, c=colours, s=6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].set_aspect("equal")
    axes[0].set_title(title)
    return axes
