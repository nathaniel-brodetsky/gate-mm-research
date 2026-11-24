"""Diagnostic of the heaviest calendar days: shares, dates, shifts, data checks and the summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmlab.evaluation import lab, tails
from mmlab.stochastic_models import markout_panel as mp
from test_h003 import trades


def _pool(num_c: list[float], den: list[float], block: float = 1.0) -> mp.DayPool:
    return mp.DayPool(np.array(num_c), np.array(den), np.array([int(d > 0) * 5 for d in den]), block)


def test_top_share_and_normal_reference() -> None:
    assert tails.top_share(np.array([1.0, -2.0, 3.0, 0.5, 0.5])) == pytest.approx(6 / 7)
    assert tails.top_share(np.array([1.0, -2.0, 3.0, 0.5, 0.5]), k=1) == pytest.approx(3 / 7)
    assert tails.top_share(np.zeros(4)) == 0.0
    ref92, ref30 = tails.normal_top_share(92), tails.normal_top_share(30)
    assert 0.08 < ref92 < 0.12
    assert ref30 > ref92
    assert tails.normal_top_share(3) == 1.0
    assert tails.normal_top_share(4) < 1.0
    assert tails.normal_top_share(92) == ref92  # fixed seed


def test_heavy_days_dates_and_shifts() -> None:
    pool = _pool([0.5, -3.0, 0.0, 1.0, 2.0, -0.5], [1.0, 2.0, 0.0, 1.0, 1.0, 1.0])
    h = tails.heavy_days(pool, "2026-08-01")
    assert [d["date"] for d in h["worst"]] == ["2026-08-02", "2026-08-06"]  # only days with a loss
    assert [d["date"] for d in h["best"]] == ["2026-08-05", "2026-08-04", "2026-08-01"]
    w = h["worst"][0]
    assert w["shift_bp"] == pytest.approx(3.0 / (6.0 - 2.0) * 1e4)
    assert w["volume_share"] == pytest.approx(2 / 6)
    assert w["trades"] == 5
    assert h["best"][0]["shift_bp"] == pytest.approx(-2.0 / 5.0 * 1e4)
    one = tails.heavy_days(_pool([-1.0], [1.0]), "2026-08-01")
    assert one["worst"][0]["shift_bp"] is None  # nothing left without the day
    assert one["best"] == []
    gap = tails.heavy_days(_pool([2.0, 0.0, -2.0], [1.0, 0.0, 1.0]), "2026-08-01")  # 08-02: no trades
    assert [d["date"] for d in gap["best"]] == ["2026-08-01"]
    assert [d["date"] for d in gap["worst"]] == ["2026-08-03"]


def test_shape_of_lopsided_and_flat_days() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 92)
    x[10] = -40.0
    x -= x.mean()
    s = tails.shape(_pool(list(x), [1.0] * 92))
    assert s["active_days"] == 92
    assert s["skew"] < -5
    assert s["excess_kurtosis"] > 30
    assert s["top3_share"] > 2 * s["top3_share_normal"]
    two = tails.shape(_pool([1.0, -1.0, 1.0, -1.0], [1.0] * 4))
    assert two["skew"] == pytest.approx(0.0)
    assert two["excess_kurtosis"] == pytest.approx(-2.0)  # two-point distribution: kurtosis 1, excess -2
    sym = tails.shape(_pool([1.0, -1.0, 0.0, 9.0], [1.0, 1.0, 0.0, 0.0]))  # inactive days are left out
    assert sym["active_days"] == 2
    assert sym["skew"] == 0.0
    assert sym["top3_share"] == 1.0


def _write_ds(root: Path) -> Path:
    ds = root / "ds"
    for sub in ("trades", "hours"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    rows = [
        (60, 105.0, 1),
        (120, 110.0, 1),
        (150, 100.0, 1),
        (180, 103.0, -1),
        (86_400 + 60, 100.0, 1),
        (86_400 + 120, 101.0, 1),
    ]
    trades(rows).write_parquet(ds / "trades" / "A_USDT.parquet")
    (ds / "hours" / "A_USDT.csv").write_text(
        "hour,status,chained,updates\n"
        "2026-08-01T00,clean,true,5\n2026-08-01T01,issues,true,5\n2026-08-01T02,missing,,0\n"
        "2026-08-02T00,clean,false,5\n2026-08-02T01,clean,true,5\n"
    )
    return ds


def test_day_quality(tmp_path: Path) -> None:
    ds = _write_ds(tmp_path)
    q = tails.day_quality(ds, "A_USDT", ["2026-08-01", "2026-08-02", "2026-08-09"])
    d1, d2, d9 = (q["days"][d] for d in ("2026-08-01", "2026-08-02", "2026-08-09"))
    assert d1["range"] == pytest.approx(0.10)
    assert d1["ret"] == pytest.approx(103 / 105 - 1)  # last over first, not over the low
    assert d1["problem_hours"] == 2
    assert d2["range"] == pytest.approx(0.01)
    assert d2["problem_hours"] == 1  # does not chain to the previous hour
    assert d9 == {"range": None, "ret": None, "problem_hours": 0}
    assert q["median_range"] == pytest.approx(0.055)


def _setup(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs: dict[str, Any] = {}
    cmp: dict[str, Any] = {}
    rng = np.random.default_rng(7)
    for i in range(6):
        x = rng.normal(0, 1, 92)
        x[20] -= 5.0 * i  # pair i has a loss day on 2026-08-21 that grows with i
        x -= x.mean()
        path = tmp_path / f"P{i}.npz"
        lab.save_pool(path, _pool(list(x), [1.0] * 92, block=1.0 + i))
        pairs[f"P{i}_USDT"] = {
            "trades": 500,
            "active_days": 92,
            "pool": str(path),
            "theta": -0.001,
            "block": 1.0 + i,
        }
        for n in (30, 61):
            cmp[f"compare/all/{n}d/size/P{i}_USDT"] = {
                "reps": 400,
                "reject_at_alpha": {"student_pw": {"rate": 0.05 + 0.02 * i}, "percentile": {"rate": 0.3}},
            }
    pairs["THIN_USDT"] = {"trades": 3, "active_days": 1, "pool": "absent.npz", "theta": 0.0, "block": 1.0}
    cal = {
        "period": ["2026-08-01", "2026-11-01"],
        "n_days": 92,
        "kind": "days",
        "subsets": {"all": {"pairs": pairs}},
    }
    return cal, cmp


def test_tails_across_pairs(tmp_path: Path) -> None:
    cal, cmp = _setup(tmp_path)
    res = tails.tails(cal, cmp)
    sub = res["subsets"]["all"]
    assert set(sub["pairs"]) == {f"P{i}_USDT" for i in range(6)}  # thin pairs are not tested on their own
    p5 = sub["pairs"]["P5_USDT"]
    assert p5["worst"][0]["date"] == "2026-08-21"
    assert p5["size"] == pytest.approx(
        {
            "30d/student_pw": 0.15,
            "30d/percentile": 0.3,
            "61d/student_pw": 0.15,
            "61d/percentile": 0.3,
        }
    )
    assert sub["spearman_size30"]["block"]["rho"] == pytest.approx(1.0)
    assert sub["spearman_size30"]["top3_share"]["rho"] > 0.8
    assert sub["spearman_size30"]["skew"]["rho"] < -0.8
    assert "2026-08-21" in sub["shared_worst_dates"]
    assert sub["shared_worst_dates"]["2026-08-21"] >= 4
    assert all(n >= tails.MIN_SHARED for n in sub["shared_worst_dates"].values())
    assert "median_range" not in p5
    with pytest.raises(ValueError, match="day pools"):
        tails.tails({**cal, "kind": "trades"}, cmp)


def test_size_missing_and_spearman_guards() -> None:
    assert tails._size({}, "all", 30, "X", "student_pw") is None
    assert tails._size({"compare/all/30d/size/X": {"reps": 0}}, "all", 30, "X", "student_pw") is None
    assert tails._spearman([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]) is None  # fewer than 5 pairs
    assert tails._spearman([0.1] * 6, [1, 2, 3, 4, 5, 6]) is None  # constant
    c = tails._spearman([0.1, 0.2, 0.3, 0.4, 0.5], [5, 4, 3, 2, 1])
    assert c is not None
    assert c["rho"] == pytest.approx(-1.0)
    assert c["pairs"] == 5


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cal, cmp = _setup(tmp_path)
    ds = _write_ds(tmp_path)
    for i in range(6):
        (ds / "trades" / f"P{i}_USDT.parquet").write_bytes((ds / "trades" / "A_USDT.parquet").read_bytes())
    pools = tmp_path / "daypools.json"
    pools.write_text(json.dumps(cal))
    comp = tmp_path / "compare.json"
    comp.write_text(json.dumps({"results": cmp}))
    out = tmp_path / "r" / "tails.json"
    assert tails.main([str(pools), "--compare", str(comp), "--dataset", str(ds), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    p5 = doc["subsets"]["all"]["pairs"]["P5_USDT"]
    assert p5["median_range"] == pytest.approx(0.055)
    assert p5["worst"][0]["range"] is None  # no trades in the toy dataset on 2026-08-21
    text = capsys.readouterr().out
    lines = text.splitlines()
    first = next(i for i, s in enumerate(lines) if s.lstrip().startswith("P"))
    assert lines[first].split()[0] == "P5_USDT"  # sorted by size, largest first
    assert "2026-08-21" in text
    assert "spearman(size30, block) +1.00" in text


def test_shared_dates_count_all_worst_days(tmp_path: Path) -> None:
    pairs = {}
    for i in range(3):
        x = np.zeros(10)
        x[i] = -5.0  # the worst day of each pair is its own
        x[7] = -2.0  # the second worst day is the same for all
        x[9] = 7.0
        path = tmp_path / f"S{i}.npz"
        lab.save_pool(path, _pool(list(x), [1.0] * 10))
        pairs[f"S{i}_USDT"] = {"trades": 50, "active_days": 10, "pool": str(path), "theta": 0.0, "block": 1.0}
    cal = {"period": ["2026-08-01", "2026-08-11"], "kind": "days", "subsets": {"all": {"pairs": pairs}}}
    sub = tails.tails(cal, {})["subsets"]["all"]
    assert sub["shared_worst_dates"] == {"2026-08-08": 3}
    assert sub["spearman_size30"]["block"] is None  # no sizes without a comparison
