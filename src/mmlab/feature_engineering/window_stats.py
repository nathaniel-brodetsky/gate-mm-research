"""Statistics of a price path over one trailing window: the baselines that topological features must beat.

Every function takes the observed mid of one window, one value per second, oldest first (any unit, e.g.
ticks), and uses nothing outside it. Changes are not demeaned: over seconds to minutes the drift is
part of what a maker must see.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float64]


def realized_vol(m: F) -> float:
    """Square root of the sum of squared 1-step changes."""
    d = np.diff(m)
    return float(np.sqrt(np.sum(d * d)))


def price_range(m: F) -> float:
    return float(np.max(m) - np.min(m))


def variance_ratio(m: F, k: int) -> float:
    """Mean square of the overlapping k-step changes over k times that of the 1-step changes.

    Below 1 when moves revert within k steps, above 1 when they continue; 1 when the window has no
    change or is too short.
    """
    d = np.diff(m)
    v1 = float(np.mean(d * d)) if len(d) else 0.0
    if v1 == 0.0 or len(m) <= k:
        return 1.0
    dk = m[k:] - m[:-k]
    return float(np.mean(dk * dk) / (k * v1))


def block_autocorr(m: F, k: int) -> float:
    """First-order autocorrelation of the non-overlapping k-step changes (uncentred); 0 if undefined."""
    d = np.diff(m[::k])
    den = float(np.sum(d * d))
    if len(d) < 3 or den == 0.0:
        return 0.0
    return float(np.sum(d[1:] * d[:-1]) / den)


def efficiency_ratio(m: F) -> float:
    """Net change over the path length: 1 for a straight move, near 0 for back-and-forth; 0 if no move."""
    path = float(np.sum(np.abs(np.diff(m))))
    return abs(float(m[-1] - m[0])) / path if path > 0 else 0.0


def max_share(m: F) -> float:
    """Share of the largest squared 1-step change in their sum (a jump measure); 0 if no move."""
    d2 = np.diff(m) ** 2
    s = float(np.sum(d2))
    return float(np.max(d2)) / s if s > 0 else 0.0


def change_share(m: F) -> float:
    """Share of steps in which the price changed."""
    d = np.diff(m)
    return float(np.mean(d != 0)) if len(d) else 0.0
