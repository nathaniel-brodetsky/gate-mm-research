"""Book state before a trade and its realized spread, against hand-computed values."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl
import pytest

from mmlab.evaluation import h001_h002 as hh
from mmlab.microstructure import conditional as cd
from mmlab.microstructure.markouts import in_period
from test_h001_h002 import impact_pair

S = 1_000_000
T = 1_785_542_400 * S  # 2026-08-01 00:00 UTC


def tob(rows: list[tuple[float, float, float, float, float, bool]]) -> pl.DataFrame:
    """rows: (label seconds after T, bid, ask, bid_qty, ask_qty, valid); available 100 ms after the label."""
    return pl.DataFrame(
        {
            "label_us": [T + round(r[0] * S) for r in rows],
            "avail_us": [T + round(r[0] * S) + 100_000 for r in rows],
            "bid": [r[1] for r in rows],
            "bid_qty": [r[3] for r in rows],
            "ask": [r[2] for r in rows],
            "ask_qty": [r[4] for r in rows],
            "valid": [r[5] for r in rows],
        }
    )


def trades(rows: list[tuple[float, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_us": [T + round(r[0] * S) for r in rows],
            "trade_id": list(range(len(rows))),
            "price": [r[1] for r in rows],
            "qty": [1.0] * len(rows),
            "taker": [r[2] for r in rows],
        },
        schema_overrides={"taker": pl.Int8},
    )


BOOK = tob(
    [
        (0, 99.0, 101.0, 3.0, 1.0, True),  # mid 100, q 100 bp; ask side thin
        (29.9, 99.9, 100.1, 2.0, 2.0, True),  # mid 100, q 10 bp
        (89.9, 101.9, 102.1, 1.0, 1.0, True),  # mid 102
        (149.9, 101.9, 102.1, 1.0, 1.0, False),  # invalid from 150 s
    ]
)
TRADES = trades(
    [
        (10, 101.0, 1),  # buy into the thin ask
        (10, 99.0, -1),  # sell into the thick bid
        (40, 100.1, 1),
        (100, 102.1, 1),  # book at 160 s invalid: no RS(60 s)
        (155, 102.1, 1),  # book at t invalid: dropped
    ]
)


def test_states_against_hand_values() -> None:
    st = cd.trade_states(TRADES, BOOK)
    assert st.height == 4
    buy, sell, mid_spread, late = st.rows(named=True)
    assert buy["q"] == pytest.approx(0.01)
    assert (buy["rho"], sell["rho"]) == (pytest.approx(0.25), pytest.approx(0.75))
    assert (buy["es"], buy["rs_60s"]) == (pytest.approx(0.01), pytest.approx(0.01))  # mid at 70 s: 100
    assert (sell["es"], sell["rs_60s"]) == (pytest.approx(0.01), pytest.approx(0.01))
    assert mid_spread["q"] == pytest.approx(0.001)
    assert mid_spread["rho"] == pytest.approx(0.5)
    assert mid_spread["rs_60s"] == pytest.approx((100.1 - 102.0) / 100)  # mid at 100 s: 102
    assert late["rs_60s"] is None
    assert buy["v"] == pytest.approx(101.0)
    assert str(buy["day"]) == "2026-08-01"


def test_last_row_per_availability_wins_and_locked_books_are_unusable() -> None:
    book = tob(
        [
            (0, 99.0, 101.0, 1.0, 1.0, True),
            (0, 98.0, 102.0, 1.0, 1.0, True),  # same availability time: this one counts
            (19.9, 100.0, 100.0, 1.0, 1.0, True),  # locked: bid = ask
        ]
    )
    st = cd.trade_states(trades([(10, 101.0, 1), (25, 101.0, 1)]), book)
    assert st.height == 1
    assert st["q"][0] == pytest.approx(0.02)


def test_state_is_used_only_after_availability() -> None:
    book = tob([(0, 99.0, 101.0, 1.0, 1.0, True), (9.95, 99.5, 100.5, 1.0, 1.0, True)])  # known from 10.05 s
    st = cd.trade_states(trades([(10, 101.0, 1)]), book)
    assert st["q"][0] == pytest.approx(0.01)  # the book of 0 s, not the one labelled 9.95 s


def test_buckets_include_the_lower_edge_despite_rounding() -> None:
    cells = cd.cells(cd.trade_states(TRADES, BOOK), "X_USDT", "development", 61)
    q = {c.bucket: c for c in cells if c.variable == "q"}
    assert q["[10, 20) bp"].trades == 1  # 100.1 - 99.9 is 0.19999999999999574 in floating point
    assert q["[80, inf) bp"].trades == 2
    late = q["[5, 10) bp"]  # the trade at 100 s: q = 0.2 / 204, no RS(60 s)
    assert (late.trades, late.rs60, late.theta) == (1, None, None)
    assert late.es == pytest.approx((102.1 - 102.0) / 102.0)
    assert q["[80, inf) bp"].theta == pytest.approx(0.01 - cd.MAKER_FEE)
    assert q["[80, inf) bp"].usdt_per_day == pytest.approx((101.0 + 99.0) / 61)
    assert q["[0, 5) bp"].es is None
    rho = {c.bucket: c.trades for c in cells if c.variable == "rho"}
    assert rho == {"[0, 0.2)": 0, "[0.2, 0.4)": 1, "[0.4, 0.6)": 2, "[0.6, 0.8)": 1, "[0.8, 1)": 0}
    assert sum(c.trades for c in cells if c.variable == "q") == 4


def test_agrees_with_the_h001_markouts() -> None:
    tob_i, tr_i = impact_pair()
    st = in_period(cd.trade_states(tr_i, tob_i), "2026-08-01", "2026-10-01")
    r = hh.evaluate_pair("I", tob_i, tr_i, "2026-08-01", "2026-10-01")
    rs = st.filter(pl.col("rs_60s").is_not_null())
    vw = rs.select((pl.col("v") * pl.col("rs_60s")).sum() / pl.col("v").sum()).item()
    assert vw == pytest.approx(r.rs["60s"])
    assert st.select((pl.col("v") * pl.col("es")).sum() / pl.col("v").sum()).item() == pytest.approx(r.es)
    assert st.height == r.trades


def test_pooled_counts_only_pairs_with_enough_trades() -> None:
    def cell(pair: str, trades: int, days: int, theta: float | None, bucket: str = "[20, 40) bp") -> cd.Cell:
        return cd.Cell("development", pair, "q", bucket, trades, days, 100.0, None, None, theta)

    rows = [
        cell("A", 50, 10, 0.002),
        cell("B", 50, 10, -0.001),
        cell("C", 50, 10, 0.004),
        cell("D", 29, 10, 0.5),  # too few trades
        cell("E", 50, 4, 0.5),  # too few days
        cell("F", 30, 5, 0.0),  # exactly at both minimums: counted; theta 0 is not positive
        cell("G", 30, 5, 0.003),
        cell("A", 50, 10, -0.003, "[5, 10) bp"),
    ]
    out = cd.pooled(rows)
    assert [p.bucket for p in out] == ["[5, 10) bp", "[20, 40) bp"]  # ordered by the lower edge
    p = out[1]
    assert (p.pairs, p.trades, p.pairs_theta_pos) == (5, 289, 3)
    assert p.theta_median == pytest.approx(0.002)
    assert p.usdt_per_day == pytest.approx(700.0)
    text = cd.report(out)
    assert "development  q    [20, 40) bp          5      289         700      20.0       3" in text


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = tmp_path / "ds"
    for sub in ("tob", "trades"):
        (ds / sub).mkdir(parents=True)
    BOOK.write_parquet(ds / "tob" / "X_USDT.parquet")
    TRADES.write_parquet(ds / "trades" / "X_USDT.parquet")
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["X_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    out = tmp_path / "r" / "x.csv"
    assert cd.main([str(ds), "--universe", str(uni), "--out", str(out)]) == 0
    rows = out.read_text().splitlines()
    assert rows[0] == "period,pair,variable,bucket,trades,days,usdt_per_day,es,rs60,theta"
    with out.open() as fh:
        table = list(csv.DictReader(fh))
    key = ("development", "q", "[80, inf) bp")
    top = next(r for r in table if (r["period"], r["variable"], r["bucket"]) == key)
    assert float(top["usdt_per_day"]) == pytest.approx(200.0 / 61)  # August and September: 61 calendar days
    assert len(rows) == 1 + 3 * (6 + 5)  # three periods, six q and five rho buckets
    printed = capsys.readouterr().out
    assert "theta med" in printed
    firsts = [ln.split()[0] for ln in printed.splitlines()[2:]]
    assert firsts == ["development"] * 11 + ["validation"] * 11 + ["all"] * 11
    assert b"\r" not in out.read_bytes()
    assert cd.main([str(ds), "--universe", str(uni), "--out", str(out), "--periods", "holdout"]) == 2
    (ds / "tob" / "X_USDT.parquet").unlink()
    assert cd.main([str(ds), "--universe", str(uni), "--out", str(out)]) == 2


def test_periods_stay_before_the_time_holdout() -> None:
    assert all(end <= cd.TIME_HOLDOUT_START for _, end in cd.PERIODS.values())
    assert cd.TIME_HOLDOUT_START == "2026-11-01"
    assert cd.MAKER_FEE == hh.MAKER_FEE
