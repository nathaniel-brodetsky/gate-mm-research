from __future__ import annotations

import math

import numpy as np
import pytest

from mmlab.feature_engineering import window_stats as ws


def a(*x: float) -> np.ndarray:
    return np.array(x, dtype=np.float64)


def test_realized_vol_and_range() -> None:
    assert ws.realized_vol(a(0, 1, 1, 3)) == pytest.approx(math.sqrt(5))
    assert ws.price_range(a(2, -1, 4, 0)) == 5


def test_variance_ratio() -> None:
    assert ws.variance_ratio(a(0, 1, 0, 1, 0, 1, 0), 2) == 0.0  # every move reverses
    assert ws.variance_ratio(a(0, 1, 2, 3, 4, 5, 6), 2) == pytest.approx(2.0)  # every move continues
    assert ws.variance_ratio(a(0, 1, 2, 3, 4, 5, 6), 3) == pytest.approx(3.0)
    assert ws.variance_ratio(a(3, 3, 3, 3), 2) == 1.0
    assert ws.variance_ratio(a(0, 1, 2), 3) == 1.0
    assert ws.variance_ratio(a(0, 2, 2, 0), 2) == pytest.approx((4 + 4) / 2 / (2 * 8 / 3))


def test_block_autocorr() -> None:
    assert ws.block_autocorr(a(0, 1, 2, 3, 4), 1) == pytest.approx(3 / 4)
    assert ws.block_autocorr(a(0, 1, 0, 1, 0), 1) == pytest.approx(-3 / 4)
    assert ws.block_autocorr(a(0, 9, 1, 9, 2, 9, 3, 9, 4), 2) == pytest.approx(3 / 4)  # steps of 2: 0,1,2,3,4
    assert ws.block_autocorr(a(5, 5, 5, 5, 5), 1) == 0.0
    assert ws.block_autocorr(a(0, 1, 2), 1) == 0.0  # two changes: undefined


def test_efficiency_share_and_changes() -> None:
    assert ws.efficiency_ratio(a(0, 1, 2, 3)) == 1.0
    assert ws.efficiency_ratio(a(3, 2, 1, 0)) == 1.0
    assert ws.efficiency_ratio(a(0, 1, 0)) == 0.0
    assert ws.efficiency_ratio(a(0, 2, 1)) == pytest.approx(1 / 3)
    assert ws.efficiency_ratio(a(4, 4)) == 0.0
    assert ws.max_share(a(0, 0, 3, 4)) == pytest.approx(0.9)
    assert ws.max_share(a(1, 1)) == 0.0
    assert ws.change_share(a(0, 0, 1, 1, 2)) == 0.5
    assert ws.change_share(a(7)) == 0.0
