"""Exploration by as-of state: buckets, tables, a run on a small dataset and the summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mmlab.microstructure import state_explore as se
from test_h003 import S, T
from test_quote_pause import JUMP, _dataset


def test_labels_and_buckets() -> None:
    assert se.labels(se.SIGNED) == ["[-1,-0.6)", "[-0.6,-0.2)", "[-0.2,0.2)", "[0.2,0.6)", "[0.6,1]", "none"]
    assert se.labels((0.0, 1.0, math.inf))[-2:] == ["[1,inf)", "none"]
    v = np.array([-1.0, -0.6, 0.19999999999, 0.2, 1.0, np.nan, 1.5, -1.2])
    assert se.bucket_index(v, se.SIGNED).tolist() == [0, 1, 3, 3, 4, 5, 5, 5]
    inf_edges = se.EDGES["since_last_trade_s"]
    assert se.bucket_index(np.array([0.0, 0.5, 1e9, np.inf]), inf_edges).tolist() == [0, 0, 4, 5]


def test_bucket_table_by_hand() -> None:
    v = np.array([1.0, 2.0, 1.0, 4.0])
    x = np.array([0.01, -0.02, 0.03, 0.0])
    day = np.array([0, 0, 1, 2])
    t = se.bucket_table(v, x, day, {"imbalance": np.array([-0.9, -0.7, 0.5, np.nan])})["imbalance"]
    assert t["[-1,-0.6)"] == {"trades": 2, "active_days": 1, "volume": 3.0, "theta": pytest.approx(-0.01)}
    assert t["[0.2,0.6)"]["theta"] == pytest.approx(0.03)
    assert t["none"]["volume"] == 4.0
    assert t["[0.6,1]"] == {"trades": 0, "active_days": 0, "volume": 0.0, "theta": None}


def test_explore_pair_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(tmp_path)
    pair, res = se.explore_pair(ds, "A_USDT", "2026-08-01", "2026-08-08", 1_000_000)
    assert pair == "A_USDT"
    assert res is not None
    for subset in ("all", "q80"):
        for name, table in res[subset]["features"].items():
            assert sum(c["volume"] for c in table.values()) == pytest.approx(res[subset]["volume"]), name
            assert sum(c["trades"] for c in table.values()) == res[subset]["trades"]
    assert res["q80"]["trades"] < res["all"]["trades"]  # at the jump the half-spread is 67 bp
    rr = res["all"]["features"]["range_ratio_300s"]
    assert rr["[5,inf)"]["trades"] == 1  # D; at B and C the jump is not yet 1 s old on the second grid
    assert rr["none"]["trades"] > 0  # the first three days have no baseline
    since = res["all"]["features"]["since_last_trade_s"]
    assert since["[600,inf)"]["trades"] > since["[0,1)"]["trades"]
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "THIN_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    out = tmp_path / "r" / "state.json"
    argv = [str(ds), "--universe", str(uni), "--end", "2026-08-08", "--workers", "1", "--out", str(out)]
    assert se.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["too_thin"] == ["THIN_USDT"]
    assert doc["edges"]["since_last_trade_s"][-1] == "inf"
    text = capsys.readouterr().out
    assert "exploration, no inference" in text
    assert "range_ratio_300s" in text
    assert "too thin: THIN_USDT" in text
    assert se.main([str(ds), "--universe", str(uni), "--end", "2026-12-01", "--out", str(out)]) == 2


def test_range_ratio_sees_the_jump_only_after_the_latency(tmp_path: Path) -> None:
    tob = pl.read_parquet(_dataset(tmp_path) / "tob" / "A_USDT.parquet")
    secs = (JUMP - 0.1, JUMP + 0.5, JUMP + 1.2, JUMP + 2.5, 2 * 86_400.0)
    ts = np.array([T + round(s * S) for s in secs], dtype=np.int64)
    r = se.range_ratio(tob, ts, "2026-08-01", "2026-08-08", 1_000_000)
    assert r[:3].tolist() == [0.0, 0.0, 0.0]  # the jump (available at +0.1 s) is used from grid second +1
    # baseline of day 5: the day-1 jump gives a range of 0.5 on 2 x 299 grid seconds, over 5 days
    assert r[3] == pytest.approx(0.5 / (0.5 * 598 / 86_400 / 5))
    assert np.isnan(r[4])  # day 2 has only two previous days


def test_features_are_known_before_the_trade() -> None:
    from mmlab.evaluation import h003
    from mmlab.feature_engineering.trade_state import taker_sides
    from test_h003 import tob, trades

    book = tob([(0, 99.0, 101.0, 1.0, 3.0), (129.5, 99.0, 101.0, 3.0, 1.0)])
    tr = trades([(1, 99.0, -1), (100, 101.0, 1), (130, 101.0, 1)])
    st = h003.states(tr, book)
    st = st.with_columns(pl.Series("d", taker_sides(tr, book, st)))
    f = se.features(tr, book, st, "2026-08-01", "2026-08-02", 1_000_000)
    last = {k: float(v[-1]) for k, v in f.items()}  # the buy at 130 s, seen from 129 s
    assert last["flow_60s"] == pytest.approx(1.0)  # [69, 129): the buy at 100 s
    assert last["flow_300s"] == pytest.approx((101 - 99) / 200)  # the sell at 1 s too, in USDT
    assert last["since_last_trade_s"] == pytest.approx(29.0)
    assert last["imbalance"] == pytest.approx(-0.5)  # the book of 129.6 s is not known at 129 s
    assert np.isnan(last["range_ratio_300s"])  # day 0 has no baseline
    now = se.features(tr, book, st, "2026-08-01", "2026-08-02", 0)
    assert float(now["imbalance"][-1]) == pytest.approx(0.5)
    assert float(now["since_last_trade_s"][-1]) == pytest.approx(30.0)


def test_summary_counts_only_buckets_with_enough_trades() -> None:
    def cell(trades: int, theta: float) -> dict[str, float | int | None]:
        return {"trades": trades, "active_days": 10, "volume": 1.0, "theta": theta}

    def pair(theta_big: float, theta_small: float) -> dict[str, object]:
        feats = {name: {lab: cell(0, 0.0) for lab in se.labels(edges)} for name, edges in se.EDGES.items()}
        feats["imbalance"]["[-1,-0.6)"] = cell(100, theta_big)
        feats["imbalance"]["[0.6,1]"] = cell(5, theta_small)
        return {s: {"trades": 105, "volume": 2.0, "features": feats} for s in ("all", "q80")}

    res = {
        "period": ["x", "y"],
        "latency_ms": 1000,
        "pairs": {"P": pair(0.001, 0.05), "Q": pair(0.003, 0.05)},
    }
    text = se.summarize(res)
    assert "[-1,-0.6)    pairs  2  volume share 0.500  theta bp   +20.00" in text
    assert "[0.6,1]      pairs  0  volume share 0.500  theta bp      n/a" in text
