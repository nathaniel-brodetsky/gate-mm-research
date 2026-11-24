"""As-of activity signals on a one-second grid (stage 3, step 3).

Grid point g is an integer UTC second. It holds only what was known before second g began: the mid of
the last valid top of book with avail_us < g * 1e6 (the archive makes a 100 ms batch available 100 ms
after its label), and the trades with ts_us < g * 1e6. A trade at t is matched to grid point
floor((t - latency) / 1e6), so every signal it sees is at least `latency` old.

Signals:
- trailing_range: max / min - 1 of the known mid over grid points g - w + 1 .. g;
- trades_before: number of trades in [g - w, g) seconds;
- daily_baseline: for each grid point, the mean over the previous `days` UTC days (not the current
  one) of each day's mean of a signal; a signal divided by its baseline says how unusual it is.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.microstructure.conditional import book_table

DAY_S = 86_400


def mid_before(tob: pl.DataFrame, grid_s: NDArray[np.int64]) -> NDArray[np.float64]:
    """Mid of the last valid book with avail_us < g * 1e6 for every grid second g; NaN before the first."""
    book = book_table(tob).filter(pl.col("ok"))
    avail = book["avail_us"].to_numpy()
    mid = ((book["bid"] + book["ask"]) / 2).to_numpy()
    idx = np.searchsorted(avail, grid_s * 1_000_000, side="left") - 1
    out = np.full(grid_s.size, np.nan)
    has = idx >= 0
    out[has] = mid[idx[has]]
    return out


def trailing_range(mid: NDArray[np.float64], w: int) -> NDArray[np.float64]:
    """max / min - 1 of the mid over the last w grid points (NaN points ignored; NaN if none)."""
    s = pl.Series(mid).fill_nan(None)
    hi = s.rolling_max(w, min_samples=1).to_numpy()
    lo = s.rolling_min(w, min_samples=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(hi / lo - 1.0, dtype=np.float64)


def trades_before(ts_us: NDArray[np.int64], grid_s: NDArray[np.int64], w: int) -> NDArray[np.int64]:
    """Number of trades with (g - w) * 1e6 <= ts_us < g * 1e6; ts_us sorted ascending."""
    hi = np.searchsorted(ts_us, grid_s * 1_000_000, side="left")
    lo = np.searchsorted(ts_us, (grid_s - w) * 1_000_000, side="left")
    return np.asarray(hi - lo, dtype=np.int64)


def daily_baseline(values: NDArray[np.float64], days: int = 7, min_days: int = 3) -> NDArray[np.float64]:
    """Per grid point: mean of the daily means of the previous `days` days; NaN with fewer than min_days.

    `values` covers whole UTC days from midnight (length a multiple of 86,400); NaN points are ignored.
    """
    if values.size % DAY_S:
        raise ValueError("the grid must cover whole days")
    per_day = values.reshape(-1, DAY_S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # days without any value give NaN
        means = np.nanmean(per_day, axis=1)
    base = np.full(means.size, np.nan)
    for d in range(means.size):
        prev = means[max(0, d - days) : d]
        prev = prev[np.isfinite(prev)]
        if prev.size >= min_days:
            base[d] = float(prev.mean())
    return np.repeat(base, DAY_S)


def grid_index(ts_us: NDArray[np.int64], g0: int, latency_us: int) -> NDArray[np.int64]:
    """Grid position of the information a trade at ts_us could use: floor((t - latency) / 1e6) - g0."""
    return np.asarray(np.floor_divide(ts_us - latency_us, 1_000_000) - g0, dtype=np.int64)
