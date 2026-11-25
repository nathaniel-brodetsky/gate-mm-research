"""Lead-lag exploration: reference moves known only after the latency, skip effects, screen, a small run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from mmlab.microstructure import leadlag_explore as ll
from test_h003 import S, T
from test_quote_pause import JUMP, _dataset

G0 = T // S
UP = math.log(1.01) * 1e4  # a 1% move in bp


def _logp(n: int, jump_at: int) -> np.ndarray:
    """log of price_before: 100 before grid second jump_at, 101 from it on."""
    return np.where(np.arange(n) >= jump_at, math.log(101.0), math.log(100.0))


def test_reference_features_by_hand() -> None:
    logp = _logp(200, 51)  # a trade at 50.x s moved the reference to 101
    secs = [52.3, 51.5, 52.3, 10.2, 190.0, 0.5]
    ts = np.array([T + round(s * S) for s in secs], dtype=np.int64)
    d = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    f = ll.reference_features(ts, d, G0, logp, 1_000_000)
    assert set(f) == set(ll.FEATURES)
    assert f["ref_1s"][0] == pytest.approx(UP)  # known at grid 51, compared with grid 50
    assert f["ref_30s"][0] == pytest.approx(UP)
    assert math.isnan(f["ref_60s"][0])  # grid -9: before the reference starts
    assert f["ref_1s"][1] == 0.0  # at 51.5 the information of 50.x is not 1 s old yet
    assert f["ref_1s"][2] == pytest.approx(-UP)  # a sell: the reference moved against the taker
    assert f["ref_1s"][3] == 0.0
    assert f[ll.EX_POST][3] == pytest.approx(UP)  # ex post: from grid 10 to grid 71
    assert f[ll.EX_POST][0] == 0.0
    assert math.isnan(f[ll.EX_POST][4])  # beyond the grid
    assert math.isnan(f["ref_5s"][5])  # before the grid
    sell = ll.reference_features(ts[3:4], -d[3:4], G0, logp, 1_000_000)
    assert sell[ll.EX_POST][0] == pytest.approx(-UP)
    fast = ll.reference_features(ts[1:2], d[1:2], G0, logp, 0)
    assert fast["ref_1s"][0] == pytest.approx(UP)  # without latency 51.5 sees grid 51


def test_skip_effect_by_hand() -> None:
    v = np.array([1.0, 1.0, 2.0])
    x = np.array([0.01, -0.03, 0.02])
    e = ll.skip_effect(v, x, np.array([5.0, np.nan, 1.0]))
    assert e["kept_share"] == pytest.approx(0.75)
    assert e["theta_kept"] == pytest.approx(0.01 / 3)
    assert e["gain"] == pytest.approx(0.01 / 3 - 0.005)
    assert ll.skip_effect(v, x, np.array([3.0, 3.0, 3.0]))["gain"] == pytest.approx(0.0)  # at 3 bp: kept
    none = ll.skip_effect(v, x, np.array([4.0, 4.0, 4.0]))
    assert none == {"kept_share": 0.0, "theta_kept": None, "gain": None}
    assert ll.skip_effect(np.zeros(2), np.zeros(2), np.zeros(2))["gain"] is None


def _reference(tmp_path: Path) -> Path:
    n = 9 * 86_400
    start = G0 - 100
    jump = G0 + JUMP  # a reference trade at JUMP - 0.5 s: price_before is 101 from grid JUMP on
    sec = np.arange(start, start + n, dtype=np.int64)
    path = tmp_path / "ref" / "BTC_USDT.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"second": sec, "price_before": np.where(sec >= jump, 101.0, 100.0)}).write_parquet(path)
    return path


def test_explore_pair_sees_the_jump_only_after_the_latency(tmp_path: Path) -> None:
    ds, ref = _dataset(tmp_path), _reference(tmp_path)
    pair, res = ll.explore_pair(ds, "A_USDT", "2026-08-01", "2026-08-08", 1_000_000, str(ref))
    assert pair == "A_USDT"
    assert res is not None
    f = res["all"]["features"]
    top = "[10,inf)"
    assert f["ref_1s"][top]["trades"] == 1  # C (+1.2 s); B (+0.5 s) is within the latency
    assert f["ref_10s"][top]["trades"] == 1
    assert f["ref_30s"][top]["trades"] == 2  # C and D (+30 s)
    assert f[ll.EX_POST][top]["trades"] == 1  # A, the maker's sale just before the jump
    for name, table in f.items():
        assert sum(c["trades"] for c in table.values()) == res["all"]["trades"], name
    skip = res["all"]["skip"]
    assert skip[ll.EX_POST]["gain"] > 0.001  # not quoting A would have saved its 49% loss
    assert skip["ref_1s"]["kept_share"] < 1.0
    assert res["q80"]["trades"] < res["all"]["trades"]
    assert res["all"]["theta"] == pytest.approx(
        sum(c["theta"] * c["volume"] for c in f["ref_1s"].values() if c["theta"] is not None)
        / res["all"]["volume"]
    )


def _pair(gains: dict[str, float], kept: float = 0.6) -> dict[str, Any]:
    skip = {f: {"kept_share": kept, "theta_kept": 0.0, "gain": gains.get(f, 0.0)} for f in ll.FEATURES}
    sub = {"trades": 100, "volume": 1.0, "theta": -0.001, "skip": skip}
    return {"all": sub, "q80": {**sub, "skip": {f: {**s, "gain": None} for f, s in skip.items()}}}


def test_screen_and_the_rule() -> None:
    pairs = {
        "P": _pair({ll.EX_POST: 5e-4, "ref_10s": 4e-4}),
        "Q": _pair({ll.EX_POST: 2e-4, "ref_10s": 3e-4}),
        "R": _pair({ll.EX_POST: 4e-4, "ref_10s": 1e-4}),
    }
    sc = ll.screen(pairs)
    assert sc["all"]["theta_bp"] == pytest.approx(-10.0)
    assert sc["all"]["skip"]["ref_10s"]["gain_bp"] == pytest.approx(3.0)
    assert sc["all"]["skip"]["ref_10s"]["improved"] == 3
    assert sc["q80"]["skip"]["ref_10s"]["pairs"] == 0
    d = sc["decision"]
    assert d["ceiling_gain_bp"] == pytest.approx(4.0)
    assert d["closed"] is False
    assert d["candidates"] == ["all ref_10s"]
    low_kept = {p: _pair({ll.EX_POST: 5e-4, "ref_10s": 9e-4}, kept=0.4) for p in "PQR"}
    assert ll.screen(low_kept)["decision"]["candidates"] == []
    closed = ll.screen({p: _pair({ll.EX_POST: 2.9e-4}) for p in "PQR"})["decision"]
    assert closed["closed"] is True
    assert ll.screen({p: _pair({ll.EX_POST: 3e-4}) for p in "PQR"})["decision"]["closed"] is False  # 3 bp
    assert ll.screen({})["decision"]["closed"] is True


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds, ref = _dataset(tmp_path), _reference(tmp_path)
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "THIN_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    out = tmp_path / "r" / "leadlag.json"
    argv = [str(ds), "--reference", str(ref), "--universe", str(uni), "--end", "2026-08-08", "--workers", "1"]
    assert ll.main([*argv, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["too_thin"] == ["THIN_USDT"]
    assert doc["reference"] == "BTC_USDT.parquet"
    assert "decision" in doc["screen"]
    text = capsys.readouterr().out
    assert "exploration, no inference" in text
    assert "decision: ceiling (ex post)" in text
    assert "skip > 3 bp" in text
    short = ll.brief(doc).splitlines()
    assert len(short) == 1 + 2 * (1 + len(ll.FEATURES)) + 1
    assert short[-1].startswith("decision: ceiling (ex post)")
    assert short[2].split()[0] == "ref_1s"
    assert ll.main([*argv[:-2], "--end", "2026-12-01", "--out", str(out)]) == 2


def test_the_reference_must_be_contiguous(tmp_path: Path) -> None:
    path = tmp_path / "gap.parquet"
    pl.DataFrame({"second": [1, 2, 4], "price_before": [1.0, 1.0, 1.0]}).write_parquet(path)
    with pytest.raises(ValueError, match="contiguous"):
        ll.load_reference(str(path))
