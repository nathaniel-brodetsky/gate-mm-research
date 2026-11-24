"""Quote pauses: flags use only information known `latency` before a trade; evaluation by hand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mmlab.evaluation.h003 import MAKER_FEE
from mmlab.microstructure import quote_pause as qp
from test_h003 import S, T, tob, trades

DAY, HOUR = 86_400, 3_600
JUMP = 5 * DAY + 12 * HOUR  # day 5, 12:00: the mid jumps from 100 to 150 for an hour
EARLY = 1 * DAY + 12 * HOUR  # day 1, 12:00: the same jump, but too early for a baseline


def _book() -> pl.DataFrame:
    rows = [(d * DAY, 99.0, 101.0, 1.0, 1.0) for d in range(7)]
    for t0 in (EARLY, JUMP):
        rows += [(t0, 149.0, 151.0, 1.0, 1.0), (t0 + HOUR, 99.0, 101.0, 1.0, 1.0)]
    return tob(sorted(rows))


def _trades() -> pl.DataFrame:
    rows: list[tuple[float, float, int]] = [
        (d * DAY + h * HOUR, 101.0, 1) for d in range(7) for h in range(1, 11)
    ]
    rows += [
        (JUMP - 0.1, 101.0, 1),  # A: the maker sells just before the jump and loses 49%
        (JUMP + 0.5, 151.0, 1),  # B: after the jump, but within the latency
        (JUMP + 1.2, 151.0, 1),  # C: the jump is known 1 s before (from 12:00:00.1)
        (JUMP + 30, 151.0, 1),  # D
        (EARLY + 30, 151.0, 1),  # E: day 1 has no baseline
    ]
    rows += [(5 * DAY + 15 * HOUR + s, 101.0, 1) for s in (0, 2, 4, 6, 8)]  # a burst of five trades
    return trades(sorted(rows))


def _ts(seconds: list[float]) -> np.ndarray:
    return np.array([T + round(s * S) for s in seconds], dtype=np.int64)


def test_pause_flags_see_only_the_past() -> None:
    tr = _trades()
    ts = _ts(
        [
            JUMP - 0.1,
            JUMP + 0.5,
            JUMP + 1.2,
            JUMP + 30,
            EARLY + 30,
            5 * DAY + 15 * HOUR + 6,
            5 * DAY + 15 * HOUR + 8,
        ]
    )
    trade_ts = np.sort(tr["ts_us"].to_numpy())
    f = qp.pause_flags(_book(), trade_ts, ts, "2026-08-01", "2026-08-08", 1_000_000)
    assert set(f) == set(qp.RULES)
    assert f["range/60s/x2"].tolist() == [False, False, False, True, False, False, False]
    assert f["range/1800s/x10"].tolist() == [False, False, False, True, False, False, False]
    # D follows A, B and C within 60 s; the burst trades follow 3 and 4 others
    assert f["count/60s/x2"].tolist() == [False, False, False, True, False, True, True]
    assert f["count/60s/x3"].tolist() == [False] * 6 + [True]
    assert not f["count/60s/x5"].any()
    fast = qp.pause_flags(_book(), trade_ts, ts, "2026-08-01", "2026-08-08", 0)
    assert fast["range/60s/x2"].tolist()[:4] == [False, False, True, True]  # C is paused without latency


def test_evaluate_by_hand() -> None:
    v = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    x = np.array([0.01, 0.01, -0.02, 0.03, -0.05])
    day = np.array([0, 0, 1, 1, 2])
    flags = {"r": np.array([False, False, True, False, False]), "none": np.zeros(5, dtype=bool)}
    out = qp.evaluate(v, x, day, flags)
    theta = (0.01 + 0.01 - 0.04 + 0.03 - 0.05) / 6
    assert out["theta"] == pytest.approx(theta)
    assert out["worst_day"] == 2  # day 2: -0.05 - theta = -0.0433 per unit, below day 1
    assert out["active_days"] == 3
    assert out["theta_without_3_worst_days"] is None  # only three days
    r = out["rules"]["r"]
    assert r["kept_volume"] == pytest.approx(4 / 6)
    assert r["theta_kept"] == pytest.approx((0.01 + 0.01 + 0.03 - 0.05) / 4)
    assert r["theta_paused"] == pytest.approx(-0.02)
    assert r["delta"] == pytest.approx(r["theta_kept"] - theta)
    assert r["worst_day_paused"] == 0.0
    assert out["rules"]["none"]["theta_paused"] is None
    assert out["rules"]["none"]["delta"] == pytest.approx(0.0)
    four = qp.evaluate(np.ones(4), np.array([0.01, -0.03, -0.02, -0.01]), np.arange(4), {})
    assert four["theta_without_3_worst_days"] == pytest.approx(0.01)
    all_paused = qp.evaluate(v, x, day, {"all": np.ones(5, dtype=bool)})["rules"]["all"]
    assert all_paused["theta_kept"] is None
    assert all_paused["delta"] is None
    assert all_paused["worst_day_paused"] == 1.0


def _dataset(root: Path) -> Path:
    ds = root / "ds"
    for sub in ("tob", "trades"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    _book().write_parquet(ds / "tob" / "A_USDT.parquet")
    _trades().write_parquet(ds / "trades" / "A_USDT.parquet")
    _book().write_parquet(ds / "tob" / "THIN_USDT.parquet")
    _trades().head(5).write_parquet(ds / "trades" / "THIN_USDT.parquet")
    return ds


def test_explore_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(tmp_path)
    res = qp.explore(ds, ["A_USDT", "THIN_USDT"], "2026-08-01", "2026-08-08", 1000)
    assert res["too_thin"] == ["THIN_USDT"]
    a = res["pairs"]["A_USDT"]
    assert a["worst_day"] == 5
    assert a["rules"]["range/60s/x2"]["worst_day_paused"] > 0  # D is paused, A is not
    assert a["rules"]["range/60s/x2"]["theta_paused"] == pytest.approx(1 / 150 - MAKER_FEE)
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    out = tmp_path / "r" / "pause.json"
    argv = [str(ds), "--universe", str(uni), "--end", "2026-08-08", "--workers", "1", "--out", str(out)]
    assert qp.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["latency_ms"] == 1000
    assert list(doc["pairs"]) == ["A_USDT"]
    text = capsys.readouterr().out
    assert "exploration, no inference" in text
    assert "range/60s/x2" in text
    assert qp.main([str(ds), "--universe", str(uni), "--end", "2026-11-02", "--out", str(out)]) == 2


def test_summary_orders_rules_by_median_change() -> None:
    rule = {"kept_volume": 1.0, "theta_kept": 0.0, "theta_paused": None, "worst_day_paused": 0.0}
    pairs = {
        p: {
            "theta": 0.0,
            "theta_without_3_worst_days": 0.001,
            "rules": {"a": {**rule, "delta": d1}, "b": {**rule, "delta": d2}},
        }
        for p, d1, d2 in (("P", 0.0001, 0.0005), ("Q", -0.0001, 0.0003), ("R", 0.0002, 0.0004))
    }
    text = qp.summarize({"period": ["x", "y"], "latency_ms": 1000, "rules": ["a", "b"], "pairs": pairs})
    lines = text.splitlines()
    assert lines[3].split()[0] == "b"  # median +4 bp before +1 bp
    assert "+4.00" in lines[3]
    assert "2/3" in lines[4]
    assert "gain without the 3 worst days (ex post) +10.00 bp" in text
