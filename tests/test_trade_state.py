"""As-of trade state: sides line up with states(); features use only what came before t."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mmlab.evaluation import h003
from mmlab.feature_engineering import trade_state as ts_
from test_h003 import S, T, tob, trades


def _us(seconds: list[float]) -> np.ndarray:
    return np.array([T + round(s * S) for s in seconds], dtype=np.int64)


def test_taker_sides_line_up_with_states() -> None:
    book = pl.concat([tob([(-5, 1.0, 2.0, 1.0, 1.0)], valid=False), tob([(0, 99.0, 101.0, 1.0, 1.0)])])
    tr = trades([(-1, 101.0, 1), (1, 101.0, 1), (2, 99.0, -1), (2, 101.0, 1), (3, 99.0, -1)])
    st = h003.states(tr, book)
    assert st.height == 4  # the trade before the first valid book is dropped
    assert ts_.taker_sides(tr, book, st).tolist() == [1.0, -1.0, 1.0, -1.0]
    with pytest.raises(ValueError, match="line up"):
        ts_.taker_sides(tr, book, st.head(3))
    with pytest.raises(ValueError, match="line up"):
        ts_.taker_sides(tr, book, st.with_columns(pl.col("v") * 2))


def test_signed_imbalance() -> None:
    book = pl.concat(
        [
            tob([(0, 99.0, 101.0, 3.0, 1.0), (10, 99.0, 101.0, 1.0, 4.0)]),
            tob([(20, 99.0, 101.0, 9.0, 1.0)], valid=False),
        ]
    )
    t = _us([0.05, 0.1, 5, 10.1, 10.2, 30])
    d = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    got = ts_.signed_imbalance(book, t, d)
    assert np.isnan(got[0])
    assert np.isnan(got[1])  # available at exactly 0.1 s: not before t
    assert got[2] == pytest.approx(-0.5)  # sell into a bid-heavy book: (3 - 1) / 4, signed -1
    assert got[3] == pytest.approx(0.5)
    assert got[4] == pytest.approx(-0.6)  # (1 - 4) / 5
    assert got[5] == pytest.approx(0.6)  # the invalid book is skipped


def test_signed_flow() -> None:
    all_ts = _us([0, 30, 70])
    side = np.array([1.0, -1.0, 1.0])
    v = np.array([2.0, 1.0, 1.0])
    t = _us([60, 60, 30, 0, 130, 71])
    d = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    got = ts_.signed_flow(all_ts, side, v, t, d, 60)
    assert got[0] == pytest.approx(1 / 3)  # [0, 60): buy 2, sell 1
    assert got[1] == pytest.approx(-1 / 3)
    assert got[2] == pytest.approx(1.0)  # the trade at exactly 30 s is not before t
    assert np.isnan(got[3])
    assert got[4] == pytest.approx(1.0)  # [70, 130): the trade at 70 s is in, at the lower edge
    assert got[5] == pytest.approx(0.0)  # [11, 71): sell 1, buy 1


def test_since_last_trade() -> None:
    all_ts = _us([0, 30])
    got = ts_.since_last_trade(all_ts, _us([0, 0.5, 30, 45]))
    assert got[0] == np.inf
    assert got[1:].tolist() == pytest.approx([0.5, 30.0, 15.0])
