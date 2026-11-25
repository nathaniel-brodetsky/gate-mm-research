"""Persistence diagrams of price windows and their summaries (GUDHI).

- `swing_bars`: H0 of the sublevel filtration of a path and of its negative. Each finite bar pairs a
  local extremum with the level at which its basin merges into an older one: its length is the size of
  a swing that reversed, which is what a maker quoting both sides can earn. For a path the lengths sum
  to its total variation minus its range, so only their distribution (not their sum) is new.
- `delay_embedding` and `rips_bars`: the sliding-window (Takens) embedding of a series and the H0 and
  H1 bars of its Vietoris-Rips filtration (loops appear when the series cycles).

Summaries of a set of bars: total and largest length, count above a length, persistence entropy and
the L1 norm of the persistence landscape (the sum over bars of length^2 / 4).
"""

from __future__ import annotations

import gudhi
import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float64]


def _finite(bars: F) -> F:
    b = np.asarray(bars, dtype=np.float64).reshape(-1, 2)
    return b[np.isfinite(b[:, 1])]


def sublevel_bars(y: F) -> F:
    """Finite H0 bars (birth, death) of the sublevel filtration of the path y (values on vertices)."""
    cc = gudhi.CubicalComplex(vertices=np.asarray(y, dtype=np.float64))
    cc.compute_persistence()
    return _finite(cc.persistence_intervals_in_dimension(0))


def swing_bars(y: F) -> F:
    """Bars of the sublevel filtrations of y and of -y together: the swings down and up."""
    return np.concatenate([sublevel_bars(y), sublevel_bars(-np.asarray(y, dtype=np.float64))])


def delay_embedding(y: F, dim: int = 3, delay: int = 1) -> F:
    """Points (y[i], y[i + delay], ..., y[i + (dim - 1) delay])."""
    n = len(y) - (dim - 1) * delay
    if n <= 0:
        return np.empty((0, dim))
    return np.stack([y[j * delay : j * delay + n] for j in range(dim)], axis=1).astype(np.float64)


def rips_bars(points: F) -> tuple[F, F]:
    """Finite H0 and H1 bars of the Vietoris-Rips filtration of the points (Euclidean distances)."""
    if len(points) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    st = gudhi.RipsComplex(points=points).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    return _finite(st.persistence_intervals_in_dimension(0)), _finite(
        st.persistence_intervals_in_dimension(1)
    )


def lengths(bars: F) -> F:
    return np.asarray(bars[:, 1] - bars[:, 0], dtype=np.float64) if len(bars) else np.empty(0)


def total(bars: F) -> float:
    return float(np.sum(lengths(bars)))


def longest(bars: F) -> float:
    ell = lengths(bars)
    return float(np.max(ell)) if len(ell) else 0.0


def count_at_least(bars: F, h: float) -> int:
    return int(np.sum(lengths(bars) >= h))


def landscape_l1(bars: F) -> float:
    """L1 norm of the persistence landscape: every bar adds a tent of area length^2 / 4."""
    ell = lengths(bars)
    return float(np.sum(ell * ell) / 4.0)


def entropy(bars: F) -> float:
    """Shannon entropy of the bar lengths as shares of their total; 0 for no or zero-length bars."""
    ell = lengths(bars)
    ell = ell[ell > 0]
    s = float(np.sum(ell))
    if s == 0.0:
        return 0.0
    p = ell / s
    return float(-np.sum(p * np.log(p)))
