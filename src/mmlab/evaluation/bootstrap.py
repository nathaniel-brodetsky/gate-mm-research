"""Stationary block bootstrap (Politis & Romano, 1994) for ratio estimators over days."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stationary_indices(n: int, resamples: int, mean_block: float, seed: int) -> NDArray[np.int64]:
    """(resamples, n) index matrix; blocks have geometric length with the given mean and wrap around."""
    if n < 1 or resamples < 1 or mean_block < 1:
        raise ValueError("n, resamples and mean_block must be >= 1")
    rng = np.random.default_rng(seed)
    idx = np.empty((resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, resamples)
    p_new = 1.0 / mean_block
    for j in range(1, n):
        new = rng.random(resamples) < p_new
        idx[:, j] = np.where(new, rng.integers(0, n, resamples), (idx[:, j - 1] + 1) % n)
    return idx


def ratio_resamples(
    num: NDArray[np.float64], den: NDArray[np.float64], idx: NDArray[np.int64]
) -> NDArray[np.float64]:
    """sum(num[i]) / sum(den[i]) for every resample row i of idx; nan where the denominator is 0."""
    n_sum = num[idx].sum(axis=1)
    d_sum = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: NDArray[np.float64] = np.where(d_sum != 0, n_sum / d_sum, np.nan)
    return out
