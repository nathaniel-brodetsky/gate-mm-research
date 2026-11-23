"""H-001 / H-002 evaluation on synthetic pairs with known answers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from mmlab.evaluation import h001_h002 as hh

S = 1_000_000
T = 1_785_542_400 * S  # 2026-08-01 00:00 UTC
DAY = 86_400 * S


def _tob(rows: list[tuple[int, float]]) -> pl.DataFrame:
    """rows: (label_us, mid); spread 2 bp around the mid, quantity 1, always valid."""
    return pl.DataFrame(
        {
            "label_us": [r[0] for r in rows],
            "avail_us": [r[0] + 100_000 for r in rows],
            "bid": [r[1] * (1 - 1e-4) for r in rows],
            "bid_qty": [1.0] * len(rows),
            "ask": [r[1] * (1 + 1e-4) for r in rows],
            "ask_qty": [1.0] * len(rows),
            "valid": [True] * len(rows),
        }
    )


def _trades(rows: list[tuple[int, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_us": [r[0] for r in rows],
            "trade_id": list(range(len(rows))),
            "price": [r[1] for r in rows],
            "qty": [1.0] * len(rows),
            "taker": [r[2] for r in rows],
        },
        schema_overrides={"taker": pl.Int8},
    )


def _times(days: int = 20, per_day: int = 4) -> list[int]:
    return [T + d * DAY + (k * 3 + 1) * 3_600 * S for d in range(days) for k in range(per_day)]


def spread_pair(half_spread: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Constant mid 100; buys at 100 (1 + hs), sells at 100 (1 - hs): RS(h) = hs for every h."""
    tob = _tob([(T - S, 100.0)])
    tr = [(t, 100 * (1 + half_spread * d), d) for i, t in enumerate(_times()) for d in [1 if i % 2 else -1]]
    return tob, _trades(tr)


def impact_pair() -> tuple[pl.DataFrame, pl.DataFrame]:
    """After each trade the mid moves 0.2% in its direction within 0.5 s and 1% in total by 100 s."""
    mid, rows, tr = 100.0, [(T - S, 100.0)], []
    for i, t in enumerate(_times()):
        d = 1 if i % 2 == 0 else -1
        tr.append((t, mid, d))
        rows += [(t + 400_000, mid * (1 + 0.002 * d)), (t + 99_900_000, mid * (1 + 0.01 * d))]
        mid *= 1 + 0.01 * d
    return _tob(rows), _trades(tr)


def test_h001_detects_spread_above_fee() -> None:
    good = hh.evaluate_pair("G", *spread_pair(0.01), "2026-08-01", "2026-10-01")
    assert good.h001_theta == pytest.approx(0.01 - hh.MAKER_FEE)
    assert good.h001_p == pytest.approx(1 / (hh.RESAMPLES + 1))  # no resample at or below zero
    assert good.h001_ci is not None
    assert good.h001_ci[0] > 0
    flat = hh.evaluate_pair("F", *spread_pair(0.0), "2026-08-01", "2026-10-01")
    assert flat.h001_theta == pytest.approx(-hh.MAKER_FEE)
    assert flat.h001_p > 0.99
    assert good.h002_s is None  # no impact: S undefined
    assert good.h001_theta_micro == pytest.approx(good.h001_theta)


def test_h002_share_of_impact() -> None:
    r = hh.evaluate_pair("I", *impact_pair(), "2026-08-01", "2026-10-01")
    assert r.impact["1s"] == pytest.approx(0.002, rel=1e-6)
    assert r.impact["300s"] == pytest.approx(0.01, rel=1e-6)
    assert r.h002_s == pytest.approx(0.2, rel=1e-6)
    assert r.h002_ci is not None
    assert r.h002_ci[0] == pytest.approx(0.2, rel=1e-6)
    assert r.h002_ci[1] == pytest.approx(0.2, rel=1e-6)


def test_too_few_trades_cannot_reject() -> None:
    tob, tr = spread_pair(0.01)
    r = hh.evaluate_pair("X", tob, tr.head(10), "2026-08-01", "2026-10-01")
    assert (r.h001_p, r.h001_theta) == (1.0, None)
    assert "too few" in r.note


def test_decision_rules() -> None:
    def pr(name: str, theta: float, p: float, s: float | None) -> hh.PairResult:
        return hh.PairResult(name, 100, 20, h001_theta=theta, h001_p=p, h002_s=s)

    strong = [pr(f"A{i}", 0.002, 0.0001, 0.2) for i in range(12)]
    weak = [pr(f"B{i}", -0.001, 0.9, 0.3) for i in range(12)]
    d = hh.decide(strong + weak, registered_pairs=24)
    assert d["H-001"]["rejections_after_bh"] == 12
    assert d["H-001"]["median_theta"] == pytest.approx(0.0005)  # (-0.001 + 0.002) / 2
    assert d["H-001"]["criterion_met"] is True
    assert d["H-002"]["outcome"] == "accept"
    d2 = hh.decide(strong[:11] + weak + [pr("C", 0.002, 0.5, 0.9)], registered_pairs=24)
    assert d2["H-001"]["rejections_after_bh"] == 11
    assert d2["H-001"]["criterion_met"] is False
    d3 = hh.decide([pr(f"D{i}", 0.001, 0.5, 0.8) for i in range(5)], registered_pairs=24)
    assert d3["H-002"]["outcome"] == "reject"


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = tmp_path / "ds"
    for sub in ("tob", "trades"):
        (ds / sub).mkdir(parents=True)
    pairs = {"GOOD_USDT": spread_pair(0.01), "IMP_USDT": impact_pair(), "SPYX_USDT": spread_pair(0.0)}
    for name, (tob, tr) in pairs.items():
        tob.write_parquet(ds / "tob" / f"{name}.parquet")
        tr.write_parquet(ds / "trades" / f"{name}.parquet")
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": list(pairs)}, "holdout": {"0": ["HOLD_USDT"]}}))
    out = tmp_path / "r.json"
    argv = [str(ds), "--period", "development", "--universe", str(uni), "--out", str(out)]
    assert hh.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["pairs"] == list(pairs)
    assert doc["decision"]["H-001"]["pairs_rejected"] == ["GOOD_USDT"]
    assert hh.main([*argv, "--exclude", "SPYX_USDT"]) == 0
    assert json.loads(out.read_text())["pairs"] == ["GOOD_USDT", "IMP_USDT"]
    (ds / "tob" / "IMP_USDT.parquet").unlink()
    assert hh.main(argv) == 2
    assert "bp = basis points" in capsys.readouterr().out


def test_registered_parameters_are_unchanged() -> None:
    assert (hh.MAKER_FEE, hh.RESAMPLES, hh.MEAN_BLOCK_DAYS, hh.BH_Q) == (0.0010, 10_000, 3.0, 0.10)
    assert (hh.SEED_H001, hh.SEED_H002) == (20261123, 20261124)
    assert hh.PERIODS == {
        "development": ("2026-08-01", "2026-10-01"),
        "validation": ("2026-10-01", "2026-11-01"),
    }


def test_decision_edge_cases() -> None:
    def pr(name: str, theta: float, p: float, s: float | None = None) -> hh.PairResult:
        return hh.PairResult(name, 100, 20, h001_theta=theta, h001_p=p, h002_s=s)

    # 11 rejections among 22 evaluated pairs: still short of 12 of the 24 registered
    d = hh.decide(
        [pr(f"A{i}", 0.002, 0.0001) for i in range(11)] + [pr(f"B{i}", 0.001, 0.9) for i in range(11)], 24
    )
    assert d["H-001"]["criterion_met"] is False
    # 12 rejections but a negative cross-pair median
    d = hh.decide(
        [pr(f"A{i}", 0.001, 0.0001) for i in range(12)] + [pr(f"B{i}", -0.002, 0.9) for i in range(12)], 24
    )
    assert d["H-001"]["median_theta"] < 0
    assert d["H-001"]["criterion_met"] is False
    # S spread around 0.5: interval covers 0.5
    d = hh.decide([pr(f"S{i}", 0.0, 1.0, s) for i, s in enumerate((0.1, 0.9, 0.2, 0.8, 0.5, 0.45, 0.55))], 24)
    assert d["H-002"]["outcome"] == "undecided"
