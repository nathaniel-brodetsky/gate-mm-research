"""Markouts against hand-computed values; availability time and validity are enforced."""

from __future__ import annotations

import polars as pl
import pytest

from mmlab.microstructure.markouts import daily_sums, in_period, markouts, mid_table

S = 1_000_000  # one second in microseconds
T = 1_785_542_400 * S  # 2026-08-01 00:00 UTC


def tob(rows: list[tuple[float, float, float, bool]]) -> pl.DataFrame:
    """rows: (label seconds after T, bid, ask, valid); quantities 1, availability label + 100 ms."""
    return pl.DataFrame(
        {
            "label_us": [T + round(x[0] * S) for x in rows],
            "avail_us": [T + round(x[0] * S) + 100_000 for x in rows],
            "bid": [x[1] for x in rows],
            "bid_qty": [1.0] * len(rows),
            "ask": [x[2] for x in rows],
            "ask_qty": [1.0] * len(rows),
            "valid": [x[3] for x in rows],
        }
    )


def trades(rows: list[tuple[float, float, int]]) -> pl.DataFrame:
    """rows: (seconds after T, price, taker +1/-1), quantity 2."""
    return pl.DataFrame(
        {
            "ts_us": [T + round(x[0] * S) for x in rows],
            "trade_id": list(range(len(rows))),
            "price": [x[1] for x in rows],
            "qty": [2.0] * len(rows),
            "taker": [x[2] for x in rows],
        },
        schema_overrides={"taker": pl.Int8},
    )


BOOK = tob([(0, 99.5, 100.5, True), (29.9, 100.5, 101.5, True)])  # mid 100, from 30 s mid 101


def test_values_for_buy_and_sell() -> None:
    mk = markouts(trades([(10, 100.5, 1), (10, 99.5, -1)]), BOOK, horizons_s=(1, 60))
    buy, sell = mk.row(0, named=True), mk.row(1, named=True)
    assert buy["es"] == pytest.approx(0.005)
    assert (buy["rs_1s"], buy["pi_1s"]) == (pytest.approx(0.005), pytest.approx(0.0))
    assert (buy["rs_60s"], buy["pi_60s"]) == (pytest.approx(-0.005), pytest.approx(0.01))
    assert sell["es"] == pytest.approx(0.005)
    assert sell["pi_60s"] == pytest.approx(-0.01)
    assert buy["v"] == pytest.approx(201.0)
    assert buy["es"] == pytest.approx(buy["rs_60s"] + buy["pi_60s"])


def test_no_lookahead_state_is_used_only_after_availability() -> None:
    book = tob([(0, 99.5, 100.5, True), (9.95, 199.5, 200.5, True)])  # available at 10.05 s
    mk = markouts(trades([(10, 100.5, 1)]), book, horizons_s=(0.1,))
    row = mk.row(0, named=True)
    assert row["es"] == pytest.approx(0.005)  # mid 100, not 200
    assert row["pi_0p1s"] == pytest.approx(1.0)  # at 10.1 s the jump is known


def test_invalid_rows_and_crossed_books_drop_observations() -> None:
    book = tob(
        [(0, 99.5, 100.5, True), (20, 99.5, 100.5, False), (40, 100.5, 101.5, True), (50, 101, 100, True)]
    )
    mk = markouts(trades([(10, 100.5, 1), (25, 100.5, 1)]), book, horizons_s=(5, 20, 45))
    assert mk.height == 1  # the trade at 25 s has no valid mid
    row = mk.row(0, named=True)
    assert row["pi_5s"] == pytest.approx(0.0)
    assert row["pi_20s"] is None  # 30 s falls after the invalid row
    assert row["pi_45s"] is None  # 55 s: crossed book
    assert mid_table(book)["ok"].to_list() == [True, False, True, False]


def test_microprice_variant() -> None:
    book = tob([(0, 99.0, 101.0, True)]).with_columns(pl.lit(3.0).alias("bid_qty"))
    mk = markouts(trades([(10, 101.0, 1)]), book, horizons_s=(1,), price="micro")
    micro = (99 * 1 + 101 * 3) / 4  # 100.5
    assert mk.row(0, named=True)["es"] == pytest.approx((101 - micro) / micro)
    with pytest.raises(ValueError, match="price"):
        markouts(trades([(10, 101.0, 1)]), book, price="last")


def test_period_filter_and_daily_sums() -> None:
    mk = markouts(trades([(10, 100.5, 1), (86_400 + 10, 100.5, 1)]), BOOK, horizons_s=(1,))
    assert in_period(mk, "2026-08-01", "2026-08-02").height == 1
    num, den = daily_sums(mk, "rs_1s")
    assert den == [pytest.approx(201.0), pytest.approx(201.0)]
    assert num[0] == pytest.approx(201.0 * 0.005)


def test_last_row_wins_for_equal_availability() -> None:
    book = tob([(0, 99.5, 100.5, True), (5, 0, 0, False), (5, 109.5, 110.5, True)])
    mk = markouts(trades([(10, 110.5, 1)]), book, horizons_s=(1,))
    assert mk.row(0, named=True)["es"] == pytest.approx(0.5 / 110)
