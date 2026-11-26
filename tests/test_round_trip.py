"""Round trips: every way a piece can be closed, with its P&L by hand; summary and a small run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mmlab.evaluation.h003 import MAKER_FEE
from mmlab.execution_simulator import queue as qu
from mmlab.execution_simulator import round_trip as rt
from mmlab.execution_simulator.fill_explore import TAKER_FEE
from test_queue import BASE, S, book, trades
from test_quote_pause import _dataset


def _trip(b: qu.Book, t: qu.Trades, exit_life_s: float = 10.0) -> dict[str, object]:
    entries = qu.place(b, np.array([S]), 1, 100.5, 300_000, 100 * S, 1, 0.5)  # sell at 100.5, alone
    fills = qu.simulate(b, t, entries)
    frame = rt.round_trips(b, t, entries, fills, 300_000, int(exit_life_s * S), 1, 0.5)
    assert frame.height == 1
    return frame.row(0, named=True)


def test_close_price() -> None:
    b = book([*BASE, (5.0, 99.0, 101.0, 5.0, 4.0, False)])
    t = np.array([2 * S, 6 * S])
    assert rt.close_price(b, 1, t)[0] == 101.0  # a short is closed at the ask
    assert rt.close_price(b, -1, t)[0] == 99.0  # a long at the bid
    assert np.isnan(rt.close_price(b, 1, t)[1])  # no valid book


def test_passive_exit() -> None:
    # entry filled at 2 s by a buy through 100.5; exit bid at 99.5 filled at 5 s by a sell through it
    r = _trip(book(BASE), trades([(2.0, 101.0, 1.0, 1), (5.0, 99.0, 2.0, -1)]))
    assert r["passive_share"] == 1.0
    assert r["had_exit_order"] is True
    assert r["hold_s"] == pytest.approx(3.0)
    assert r["gross"] == pytest.approx(100.5 - 99.5)
    assert r["net"] == pytest.approx(1.0 - MAKER_FEE * (100.5 + 99.5))
    assert (r["maker_value"], r["taker_value"]) == (pytest.approx(200.0), 0.0)


def test_crossed_at_expiry() -> None:
    r = _trip(book(BASE), trades([(2.0, 101.0, 1.0, 1)]))
    assert r["passive_share"] == 0.0
    assert r["hold_s"] == pytest.approx(10.4)  # the exit joined at 2.4 s and lived 10 s
    assert r["gross"] == pytest.approx(100.5 - 101.0)
    assert r["net"] == pytest.approx(-0.5 - MAKER_FEE * 100.5 - TAKER_FEE * 101.0)
    assert (r["maker_value"], r["taker_value"]) == (pytest.approx(100.5), pytest.approx(101.0))


def test_partial_passive_then_crossed() -> None:
    r = _trip(book(BASE), trades([(2.0, 101.0, 1.0, 1), (5.0, 99.0, 0.4, -1)]))
    assert r["passive_share"] == pytest.approx(0.4)
    assert r["gross"] == pytest.approx(0.4 * 1.0 + 0.6 * -0.5)
    fees = MAKER_FEE * 100.5 + MAKER_FEE * 99.5 * 0.4 + TAKER_FEE * 101.0 * 0.6
    assert r["net"] == pytest.approx(0.1 - fees)
    assert r["maker_value"] == pytest.approx(100.5 + 99.5 * 0.4)
    assert r["taker_value"] == pytest.approx(101.0 * 0.6)


def test_crossed_at_once_without_an_exit_order_and_unpriced() -> None:
    b = book([*BASE, (1.9, 99.0, 101.0, 5.0, 4.0, False), (2.2, 99.0, 101.0, 5.0, 4.0, True)])
    r = _trip(b, trades([(2.0, 101.0, 1.0, 1)]))
    assert r["had_exit_order"] is False  # the book at the fill was invalid
    assert r["hold_s"] == pytest.approx(0.3)
    assert r["net"] == pytest.approx(-0.5 - MAKER_FEE * 100.5 - TAKER_FEE * 101.0)
    dark = book([*BASE, (1.9, 99.0, 101.0, 5.0, 4.0, False)])
    r = _trip(dark, trades([(2.0, 101.0, 1.0, 1)]))
    assert r["net"] is None  # nothing valid to cross at


def test_summarize_pair() -> None:
    day = 86_400 * S
    trips = pl.DataFrame(
        {
            "entry": [0, 1, 1, 2],
            "side": [1, -1, -1, 1],
            "ts_us": [10 * S, 20 * S, day + 5 * S, day + 9 * S],
            "value": [100.0, 50.0, 50.0, 100.0],
            "passive_share": [1.0, 0.0, 0.5, 1.0],
            "had_exit_order": [True, True, True, True],
            "hold_s": [3.0, 10.0, 7.0, 1.0],
            "gross": [1.0, -0.5, 0.2, None],
            "net": [0.8, -0.7, 0.05, None],
        }
    )
    out = rt.summarize_pair(trips, np.array([True, False, True]), n_days=2)
    a, w = out["all"], out["q80"]
    assert (a["round_trips"], a["unpriced"], a["days"]) == (3, 1, 2)
    assert a["entry_value"] == 200.0
    assert a["passive_share"] == pytest.approx((100 + 25) / 200)
    assert a["median_hold_s"] == 7.0
    assert a["net_bp"] == pytest.approx(0.15 / 200 * 1e4)
    assert a["gross_bp"] == pytest.approx(0.7 / 200 * 1e4)
    assert a["net_usdt_per_day"] == pytest.approx(0.075)
    assert (w["round_trips"], w["unpriced"], w["net_usdt_per_day"]) == (1, 1, pytest.approx(0.4))
    empty = rt.summarize_pair(trips.head(0), np.array([True]), n_days=2)["all"]
    assert (empty["round_trips"], empty["net_bp"], empty["median_hold_s"]) == (0, None, None)


def test_explore_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(tmp_path)
    pair, res = rt.explore_pair(ds, "A_USDT", "2026-08-01", "2026-08-08", 600, 300, 300, 300, 10.0, 0)
    assert pair == "A_USDT"
    assert res is not None
    # at the best price (no improving): the sell decided at 15:00 is filled in the burst; its exit bid
    # at 99 meets no seller, so it is crossed at the ask 101 when it expires
    a = res["all"]
    assert a["round_trips"] == 1
    assert a["passive_share"] == 0.0
    assert a["gross_bp"] == pytest.approx((101 - 101) / 101 * 1e4)
    assert a["net_bp"] == pytest.approx(-(MAKER_FEE + TAKER_FEE) * 1e4)
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "THIN_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    out = tmp_path / "r" / "trips.json"
    argv = [
        str(ds),
        "--universe",
        str(uni),
        "--end",
        "2026-08-08",
        "--every-s",
        "600",
        "--improve-ticks",
        "0",
    ]
    assert rt.main([*argv, "--workers", "1", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["too_thin"] == ["THIN_USDT"]
    assert (doc["horizon_s"], doc["exit_life_s"], doc["improve_ticks"]) == (60, 300, 0)
    line = capsys.readouterr().out
    assert line.startswith("entry   60 s +0 tick, exit   300 s, 10 USDT, pessimistic | all: pairs 0")
    assert "USDT/day" in line
    assert rt.main([*argv, "--end", "2026-11-02", "--out", str(out)]) == 2


def test_two_entries_on_both_sides_with_their_own_sizes() -> None:
    b = book(BASE)
    entries = sorted(
        qu.place(b, np.array([S]), 1, 201.0, 300_000, 100 * S, 1, 0.5)  # sell 2 at 100.5
        + qu.place(b, np.array([S]), -1, 99.5, 300_000, 100 * S, 1, 0.5),  # buy 1 at 99.5
        key=lambda o: (o.act_us, o.side),
    )
    t = trades([(2.0, 101.0, 5.0, 1), (2.5, 99.0, 5.0, -1), (3.0, 101.0, 0.5, 1)])
    fills = qu.simulate(b, t, entries)
    frame = rt.round_trips(b, t, entries, fills, 300_000, 10 * S, 1, 0.5)
    short, long_ = frame.row(0, named=True), frame.row(1, named=True)
    # the short of 2 is closed in full by its exit bid at 99.5, hit by the sell at 2.5 s
    assert (short["side"], short["value"], short["passive_share"]) == (1, 201.0, 1.0)
    assert short["net"] == pytest.approx(2 * 1.0 - MAKER_FEE * (201.0 + 199.0))
    # the long of 1, bought at 2.5 s: its exit ask at 100.5 gets 0.5 at 3 s, the rest is sold at the bid
    assert (long_["side"], long_["passive_share"]) == (-1, 0.5)
    gross = -((99.5 - 100.5) * 0.5 + (99.5 - 99.0) * 0.5)
    assert long_["gross"] == pytest.approx(gross)
    fees = MAKER_FEE * 99.5 + MAKER_FEE * 100.5 * 0.5 + TAKER_FEE * 99.0 * 0.5
    assert long_["net"] == pytest.approx(gross - fees)


def test_taker_fee_on_the_crossed_part_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "TAKER_FEE", 0.0025)  # VIP 0 charges makers and takers alike: tell them apart
    r = _trip(book(BASE), trades([(2.0, 101.0, 1.0, 1), (5.0, 99.0, 0.4, -1)]))
    fees = MAKER_FEE * 100.5 + MAKER_FEE * 99.5 * 0.4 + 0.0025 * 101.0 * 0.6
    assert r["net"] == pytest.approx(0.1 - fees)
