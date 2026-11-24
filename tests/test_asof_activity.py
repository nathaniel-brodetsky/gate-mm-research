"""As-of signals: nothing known before it was available; windows and baselines as documented."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mmlab.feature_engineering import asof_activity as aa
from test_h003 import S, T, tob

G0 = T // S  # 2026-08-01 00:00 UTC in seconds


def test_mid_before_uses_only_available_valid_books() -> None:
    book = tob([(0, 99.0, 101.0, 1.0, 1.0), (10, 101.0, 103.0, 1.0, 1.0)])
    bad = tob([(20, 1.0, 2.0, 1.0, 1.0)], valid=False)
    grid = G0 + np.array([0, 1, 10, 11, 21, 30])
    mid = aa.mid_before(pl.concat([book, bad]), grid)
    assert np.isnan(mid[0])  # the first batch is available at 0.1 s, after second 0 began
    assert mid[1] == 100.0
    assert mid[2] == 100.0  # the batch labelled 10 s is available at 10.1 s
    assert mid[3] == 102.0
    assert mid[4] == 102.0  # an invalid book is skipped
    assert mid[5] == 102.0


def test_mid_before_is_strict_at_the_second() -> None:
    book = tob([(0, 99.0, 101.0, 1.0, 1.0), (9.9, 101.0, 103.0, 1.0, 1.0)])  # available at exactly 10 s
    mid = aa.mid_before(book, G0 + np.array([10, 11]))
    assert mid.tolist() == [100.0, 102.0]


def test_mid_before_ignores_later_rows() -> None:
    early = tob([(0, 99.0, 101.0, 1.0, 1.0), (50, 101.0, 103.0, 1.0, 1.0)])
    late = pl.concat([early, tob([(100, 200.0, 202.0, 1.0, 1.0)])])
    grid = G0 + np.arange(0, 101)
    assert np.array_equal(aa.mid_before(early, grid), aa.mid_before(late, grid), equal_nan=True)


def test_trailing_range_against_brute_force() -> None:
    rng = np.random.default_rng(3)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    mid[:7] = np.nan
    for w in (1, 5, 60):
        got = aa.trailing_range(mid, w)
        for g in range(mid.size):
            win = mid[max(0, g - w + 1) : g + 1]
            win = win[np.isfinite(win)]
            want = win.max() / win.min() - 1 if win.size else np.nan
            assert got[g] == pytest.approx(want, nan_ok=True)


def test_trades_before_counts_the_half_open_window() -> None:
    ts = ((G0 + np.array([0.0, 0.5, 1.0, 59.999, 60.0, 61.0])) * 1e6).astype(np.int64)
    grid = G0 + np.array([0, 1, 60, 61, 62, 121])
    assert aa.trades_before(ts, grid, 60).tolist() == [0, 2, 4, 3, 3, 1]  # [g - 60, g) seconds
    assert aa.trades_before(ts, grid, 1).tolist() == [0, 2, 1, 1, 1, 0]


def test_daily_baseline() -> None:
    day = aa.DAY_S
    vals = np.concatenate([np.full(day, float(d + 1)) for d in range(6)])
    vals[3 * day : 4 * day] = np.nan  # a day without any value is skipped
    base = aa.daily_baseline(vals, days=7, min_days=3)
    assert np.isnan(base[: 3 * day]).all()  # fewer than 3 previous days
    assert base[3 * day] == pytest.approx((1 + 2 + 3) / 3)  # its own (missing) day does not count
    assert base[4 * day] == pytest.approx((1 + 2 + 3) / 3)  # day 3 has no value
    assert base[5 * day + 17] == pytest.approx((1 + 2 + 3 + 5) / 4)
    assert base[6 * day - 1] == base[5 * day]  # constant within a day
    short = aa.daily_baseline(vals, days=2, min_days=1)
    assert short[5 * day] == pytest.approx(5.0)  # day 3 NaN, day 4 = 5
    with pytest.raises(ValueError, match="whole days"):
        aa.daily_baseline(np.zeros(10))


def test_grid_index_subtracts_the_latency() -> None:
    ts = ((G0 + np.array([10.0, 10.999, 11.0, 0.5])) * 1e6).astype(np.int64)
    assert aa.grid_index(ts, G0, 1_000_000).tolist() == [9, 9, 10, -1]
    assert aa.grid_index(ts, G0, 0).tolist() == [10, 10, 11, 0]
