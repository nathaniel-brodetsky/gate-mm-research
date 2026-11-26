"""Perpetuals exploration: measures by hand on a constant book, the delisting run-up, screen, CLI."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest

from mmlab.evaluation.h003 import states
from mmlab.microstructure import perp_explore as pe

S = 1_000_000
T25 = 1_754_006_400 * S  # 2025-08-01 00:00 UTC
DAY = 86_400


def _tob(rows: list[tuple[float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label_us": [T25 + round(r[0] * S) for r in rows],
            "avail_us": [T25 + round(r[0] * S) + 100_000 for r in rows],
            "bid": [r[1] for r in rows],
            "bid_qty": [1.0] * len(rows),
            "ask": [r[2] for r in rows],
            "ask_qty": [1.0] * len(rows),
            "valid": [True] * len(rows),
        }
    )


def _trades(rows: Sequence[tuple[float, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_us": [T25 + round(r[0] * S) for r in rows],
            "trade_id": list(range(len(rows))),
            "price": [r[1] for r in rows],
            "qty": [2.0] * len(rows),
            "taker": [r[2] for r in rows],
        },
        schema_overrides={"taker": pl.Int8},
    )


def _constant(days: int = 10, per_day: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Book 98 / 102 all along (q = 200 bp); buys at 102 and sells at 98: RS = 200 bp at every horizon."""
    tob = _tob([(d * DAY, 98.0, 102.0) for d in range(days + 1)])
    rows = [
        (d * DAY + 3600 + 60 * k, 102.0 if k % 2 == 0 else 98.0, 1 if k % 2 == 0 else -1)
        for d in range(days)
        for k in range(per_day)
    ]
    return tob, _trades(rows)


def test_block_by_hand() -> None:
    tob, tr = _constant()
    b = pe.block(states(tr, tob))
    assert b["trades"] == 60
    assert b["days"] == 10
    assert b["counted"] is True
    assert b["q_median_bp"] == pytest.approx(200.0)
    assert b["rs"]["60s"] == pytest.approx(0.02)
    assert b["rs_exit_60s"] == pytest.approx(0.0)  # buy at 102, the ask is still 102
    assert b["theta"]["maker_0.02%"] == pytest.approx(0.02 - 0.0002)
    assert b["theta"]["maker_-0.01%"] == pytest.approx(0.02 + 0.0001)
    assert b["by_q"]["[80,inf)"]["trades"] == 60
    assert b["by_q"]["[80,inf)"]["volume_share"] == pytest.approx(1.0)
    assert b["by_q"]["[0,5)"] == {"trades": 0, "volume_share": 0.0, "rs_60s": None}
    assert pe.block(states(tr.head(0), tob))["counted"] is False
    tob4, tr4 = _constant(days=4, per_day=15)
    four = pe.block(states(tr4, tob4))
    assert (four["trades"], four["days"], four["counted"]) == (60, 4, False)  # enough trades, too few days


def test_runup_start() -> None:
    assert pe.runup_start("") is None
    assert pe.runup_start("2025-08-15") == T25  # 14 days before
    assert pe.runup_start("2025-08-16") == T25 + DAY * S


def _dataset(root: Path) -> Path:
    ds = root / "ds"
    for sub in ("tob", "trades"):
        (ds / sub).mkdir(parents=True)
    tob, tr = _constant()
    for name in ("A_USDT", "B_USDT"):
        tob.write_parquet(ds / "tob" / f"{name}.parquet")
        tr.write_parquet(ds / "trades" / f"{name}.parquet")
    return ds


def test_explore_contract_drops_the_runup(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    _, r = pe.explore_contract(ds, "A_USDT", "2025-07-01", "2025-11-01", "2025-08-18")  # cut at 08-04
    assert r is not None
    assert r["all"]["trades"] == 60
    assert r["without_runup"]["trades"] == 18  # days 0-2 stay
    _, kept = pe.explore_contract(ds, "A_USDT", "2025-07-01", "2025-11-01", "")
    assert kept is not None
    assert kept["without_runup"] == kept["all"]
    assert pe.explore_contract(ds, "NONE_USDT", "2025-07-01", "2025-11-01", "") == ("NONE_USDT", None)
    _, window = pe.explore_contract(ds, "A_USDT", "2025-08-03", "2025-08-05", "")
    assert window is not None
    assert window["all"]["trades"] == 12


def test_screen_counts_and_strata() -> None:
    def blk(rs60: float, counted: bool = True) -> dict[str, object]:
        return {
            "counted": counted,
            "q_median_bp": 10.0,
            "rs": {"10s": rs60, "60s": rs60, "300s": rs60},
            "rs_exit_60s": rs60 - 0.001,
            "theta": {k: rs60 - f for k, f in pe.FEES.items()},
        }

    contracts = {
        "A": {"all": blk(0.0010), "without_runup": blk(0.0010)},
        "B": {"all": blk(0.0001), "without_runup": blk(0.0003)},
        "C": {"all": blk(-0.0005), "without_runup": blk(-0.0005, counted=False)},
        "D": {"all": blk(0.5, counted=False), "without_runup": blk(0.5, counted=False)},
    }
    sc = pe.screen(contracts, {"A": 0, "B": 0, "C": 1, "D": 1})
    a = sc["all"]["all"]
    assert a["contracts"] == 3
    assert a["rs_60s_bp"] == pytest.approx(1.0)
    assert a["theta_maker_0.02%_bp"] == pytest.approx(-1.0)
    assert a["theta_maker_-0.01%_bp"] == pytest.approx(2.0)
    assert a["positive_after_0.02%"] == 1
    assert sc["all"]["stratum 0"]["contracts"] == 2
    assert sc["all"]["stratum 1"]["contracts"] == 1
    w = sc["without_runup"]
    assert w["all"]["contracts"] == 2
    assert w["all"]["rs_60s_bp"] == pytest.approx(6.5)
    assert "stratum 1" not in w


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(tmp_path)
    uni = tmp_path / "u.json"
    uni.write_text(
        json.dumps({"research": {"0": ["A_USDT", "MISSING_USDT"], "1": ["B_USDT"]}, "holdout": {}})
    )
    cand = tmp_path / "c.csv"
    with cand.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "available_to"])
        w.writerows([["A_USDT", "2025-08-20"], ["B_USDT", ""], ["MISSING_USDT", ""]])
    out = tmp_path / "r" / "m.json"
    argv = [str(ds), "--universe", str(uni), "--candidates", str(cand), "--workers", "1", "--out", str(out)]
    assert pe.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["missing"] == ["MISSING_USDT"]
    assert set(doc["contracts"]) == {"A_USDT", "B_USDT"}
    assert doc["screen"]["all"]["all"]["contracts"] == 2
    assert doc["screen"]["without_runup"]["all"]["contracts"] == 2
    text = capsys.readouterr().out
    assert "exploration, no inference" in text
    assert "without the last 14 days before a delisting" in text
    assert "q [80,inf)" in text
    assert "not built: MISSING_USDT" in text
    assert pe.main([*argv[:-2], "--end", "2025-11-02", "--out", str(out)]) == 2
