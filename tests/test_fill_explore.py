"""Exploration of queue fills: decision grid, fill markouts by hand, per-subset summary, a small run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mmlab.evaluation.h003 import MAKER_FEE
from mmlab.execution_simulator import fill_explore as fe
from mmlab.execution_simulator import queue as qu
from mmlab.feature_engineering.asof_activity import epoch_s
from test_queue import S, book
from test_quote_pause import _dataset


def test_decision_times_leave_the_full_horizon() -> None:
    t = fe.decision_times("2026-08-01", "2026-08-02", 60, 300)
    g0 = epoch_s("2026-08-01") * S
    assert t[0] == g0
    assert t.size == 1435
    assert t[-1] == g0 + 86_040 * S


def _orders() -> list[qu.Order]:
    return [qu.Order(1, 101.0, 1.0, 0, 0, 10**12, 0.0), qu.Order(-1, 99.0, 1.0, 0, 0, 10**12, 0.0)]


def test_fill_markouts_by_hand() -> None:
    b = book(
        [(0.0, 99.0, 101.0, 1, 1, True), (70.0, 101.0, 103.0, 1, 1, True), (85.0, 101.0, 103.0, 1, 1, False)]
    )
    fills = [qu.Fill(0, 15 * S, 1.0), qu.Fill(1, 15 * S, 0.5), qu.Fill(0, 30 * S, 1.0)]
    fm = fe.fill_markouts(b, _orders(), fills)
    assert fm["value"].to_list() == [101.0, 49.5, 101.0]
    assert fm["rs"].to_list()[:2] == pytest.approx([(101 - 102) / 100, (102 - 99) / 100])
    assert fm["rs_exit"].to_list()[:2] == pytest.approx([(101 - 103) / 100, (101 - 99) / 100])
    assert fm["rs"][2] is None  # the book 60 s later (at 90 s) is invalid
    assert fe.fill_markouts(b, _orders(), []).height == 0


def test_summarize_orders_by_hand() -> None:
    orders = [qu.Order(1, 100.0, 0.1, d * S, 0, 0, 0.0) for d in (0, 0, 86_400, 86_400)]
    fm = pl.DataFrame(
        {
            "order": [0, 0, 2],
            "side": [1, 1, 1],
            "ts_us": [10 * S, 20 * S, (86_400 + 30) * S],
            "value": [4.0, 6.0, 10.0],
            "rs": [0.01, 0.02, None],
            "rs_exit": [-0.01, -0.02, -0.05],
        }
    )
    wide = np.array([True, False, True, False])
    out = fe.summarize_orders(orders, wide, fm, {"all": -0.001, "q80": 0.003})
    a, w = out["all"], out["q80"]
    assert (a["orders"], a["orders_filled"], w["orders"], w["orders_filled"]) == (4, 2, 2, 2)
    assert a["filled_share_of_size"] == pytest.approx(20 / 40)
    assert a["median_wait_s"] == pytest.approx((10 + 30) / 2)
    assert a["fills"] == 2  # the fill without a markout is left out
    assert a["fill_days"] == 1
    assert a["theta_fill"] == pytest.approx((4 * 0.01 + 6 * 0.02) / 10 - MAKER_FEE)
    assert a["theta_exit"] == pytest.approx((-0.04 - 0.12 - 0.5) / 20 - MAKER_FEE - fe.TAKER_FEE)
    assert (a["theta_trades"], w["theta_trades"]) == (-0.001, 0.003)
    none = fe.summarize_orders(orders, np.zeros(4, dtype=bool), fm, {"all": None, "q80": None})["q80"]
    assert none == {
        "orders": 0,
        "orders_filled": 0,
        "filled_share_of_size": None,
        "median_wait_s": None,
        "fills": 0,
        "fill_days": 0,
        "theta_fill": None,
        "theta_exit": None,
        "theta_trades": None,
    }


def test_explore_pair_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(tmp_path)
    pair, res = fe.explore_pair(ds, "A_USDT", "2026-08-01", "2026-08-08", 600, 300, 300, 10.0)
    assert pair == "A_USDT"
    assert res is not None
    a = res["all"]
    # 1,008 decisions on two sides, less: both at the very start (nothing known yet), the sells at the two
    # jumps and the buys at the two drops (post-only)
    assert a["orders"] == 2 * 1008 - 6
    # hourly buys of one unit meet exactly the one unit ahead; only the burst of 15:00 fills the sell
    # decided at 15:00:00: the first buy clears the queue, the next one fills it 4 s after the decision
    assert (a["orders_filled"], a["fills"], a["median_wait_s"]) == (1, 1, 4.0)
    assert a["theta_fill"] == pytest.approx(0.01 - MAKER_FEE)
    assert a["theta_exit"] == pytest.approx(0.0 - MAKER_FEE - fe.TAKER_FEE)  # buys back at 101
    assert res["q80"]["orders"] == a["orders"] - 22  # decided while the book stood at 149 / 151 (67 bp)
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "THIN_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    out = tmp_path / "r" / "fills.json"
    argv = [str(ds), "--universe", str(uni), "--end", "2026-08-08", "--every-s", "600", "--workers", "1"]
    assert fe.main([*argv, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["too_thin"] == ["THIN_USDT"]
    assert (doc["every_s"], doc["horizon_s"], doc["latency_ms"], doc["size_usdt"]) == (600, 300, 300, 10.0)
    assert doc["queue"] == "pessimistic"
    text = capsys.readouterr().out
    assert "exploration, no inference" in text
    assert "fills - trades" in text
    assert fe.main([*argv, "--end", "2026-11-02", "--out", str(out)]) == 2
    assert fe.main([*argv, "--queue", "optimistic", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["queue"] == "optimistic"
    assert "optimistic queue" in capsys.readouterr().out
    line = fe.compact(doc)
    assert line.startswith("optimistic    300 ms   300 s    10 USDT | all: filled ")
    assert "| q80: filled" in line
    assert "fills      n/a" in line  # one fill in the small dataset: below 30 fills on 5 days
    assert fe.main([*argv, "--improve-ticks", "1", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["improve_ticks"] == 1
    assert doc["pairs"]["A_USDT"]["tick"] == pytest.approx(2.0)  # 99 / 101 / 149 / 151: gaps of 2
    assert "1 tick(s) inside the spread" in capsys.readouterr().out
    assert fe.compact(doc).split("|")[0].rstrip().endswith("10 USDT +1 tick")
