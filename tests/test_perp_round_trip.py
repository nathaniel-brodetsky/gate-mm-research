"""Round trips on perpetuals: the fee scenarios by hand, a small run, the screen and the CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from mmlab.execution_simulator import perp_round_trip as pr
from test_perp_explore import T25, _dataset

DAY_US = 86_400_000_000


def _frame(rows: list[tuple[int, float, float, float, float, float, float]]) -> pl.DataFrame:
    """rows: (day, value, gross, maker_value, taker_value, passive_share, hold_s)."""
    return pl.DataFrame(
        {
            "ts_us": [T25 + d * DAY_US for d, *_ in rows],
            "value": [r[1] for r in rows],
            "gross": [r[2] for r in rows],
            "maker_value": [r[3] for r in rows],
            "taker_value": [r[4] for r in rows],
            "passive_share": [r[5] for r in rows],
            "hold_s": [r[6] for r in rows],
        }
    )


def test_summarize_by_hand() -> None:
    f = _frame([(0, 100.0, 1.0, 199.0, 0.0, 1.0, 10.0), (1, 100.0, -0.5, 100.0, 100.5, 0.0, 300.0)])
    s = pr.summarize(f)
    assert (s["round_trips"], s["days"], s["counted"]) == (2, 2, False)
    assert s["gross_bp"] == pytest.approx(0.5 / 200 * 1e4)
    assert s["passive_share"] == pytest.approx(0.5)
    assert s["median_hold_s"] == pytest.approx(155.0)
    vip0 = 0.5 - 0.0002 * 299.0 - 0.0005 * 100.5
    fields = 0.5 + 0.0001 * 299.0 - 0.00075 * 100.5
    assert s["net_bp"]["vip0"] == pytest.approx(vip0 / 200 * 1e4)
    assert s["net_bp"]["contract_fields"] == pytest.approx(fields / 200 * 1e4)
    unpriced = pl.concat([f, f.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("gross"))])
    assert pr.summarize(unpriced)["unpriced"] == 1
    empty = pr.summarize(f.head(0))
    assert (empty["round_trips"], empty["gross_bp"], empty["net_bp"]["vip0"]) == (0, None, None)
    many = _frame([(d % 5, 10.0, 0.01, 20.0, 0.0, 1.0, 5.0) for d in range(30)])
    assert pr.summarize(many)["counted"] is True


def test_a_small_run(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)  # book 98 / 102 all along; buys at 102, sells at 98
    trips = pr.contract_trips(ds, "A_USDT", "2025-08-01", "2025-08-11", improve=0)
    assert trips is not None
    assert trips.height > 0
    big = ds / "trades" / "A_USDT.parquet"
    pl.read_parquet(big).with_columns(pl.lit(5.0).alias("qty")).write_parquet(big)  # room for 2 contracts
    one = pr.contract_trips(ds, "A_USDT", "2025-08-01", "2025-08-11", improve=0)
    assert one is not None
    assert one["value"].to_numpy().max() == pytest.approx(102.0)  # orders of one contract
    two = pr.contract_trips(ds, "A_USDT", "2025-08-01", "2025-08-11", improve=0, size=2.0)
    assert two is not None
    assert two["value"].to_numpy().max() == pytest.approx(204.0)
    s = pr.summarize(trips)
    assert s["gross_bp"] is not None
    assert s["gross_bp"] > 0  # sold at 102 and bought back at 98, or the reverse
    assert s["net_bp"]["vip0"] < s["gross_bp"]
    assert pr.contract_trips(ds, "NONE_USDT", "2025-08-01", "2025-08-11") is None
    _, r = pr.explore_contract(ds, "A_USDT", "2025-08-01", "2025-08-11", "2025-08-18", "pessimistic", 0)
    assert r is not None
    assert r["without_runup"]["round_trips"] < r["all"]["round_trips"]


def _s(gross: float, vip0: float, fields: float, counted: bool = True) -> dict[str, Any]:
    return {
        "counted": counted,
        "gross_bp": gross,
        "passive_share": 0.5,
        "median_hold_s": 60.0,
        "net_bp": {"vip0": vip0, "contract_fields": fields},
    }


def test_screen() -> None:
    contracts = {
        "A": {"all": _s(5.0, 1.0, 2.0), "without_runup": _s(5.0, 1.0, 2.0)},
        "B": {"all": _s(1.0, -3.0, -1.0), "without_runup": _s(1.0, -3.0, -1.0, counted=False)},
        "C": {"all": _s(9.0, 9.0, 9.0, counted=False), "without_runup": _s(9.0, 9.0, 9.0, counted=False)},
    }
    sc = pr.screen(contracts, {"A": 0, "B": 1, "C": 1})
    a = sc["all"]["all"]
    assert a["contracts"] == 2
    assert a["gross_bp"] == pytest.approx(3.0)
    assert a["net_vip0_bp"] == pytest.approx(-1.0)
    assert a["net_contract_fields_bp"] == pytest.approx(0.5)
    assert (a["positive_vip0"], a["positive_contract_fields"]) == (1, 1)
    assert sc["without_runup"]["all"]["contracts"] == 1
    assert "stratum 1" not in sc["without_runup"]


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(tmp_path)
    uni = tmp_path / "u.json"
    uni.write_text(
        json.dumps({"research": {"0": ["A_USDT", "MISSING_USDT"], "3": ["B_USDT"]}, "holdout": {}})
    )
    cand = tmp_path / "c.csv"
    with cand.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "available_to"])
        w.writerows([["A_USDT", "2025-08-18"], ["B_USDT", ""]])
    out = tmp_path / "r" / "rt.json"
    argv = [
        str(ds),
        "--universe",
        str(uni),
        "--candidates",
        str(cand),
        "--start",
        "2025-08-01",
        "--end",
        "2025-08-11",
        "--improve-ticks",
        "0",
        "--workers",
        "1",
        "--out",
        str(out),
    ]
    assert pr.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["missing"] == ["MISSING_USDT"]
    assert doc["scenarios"]["contract_fields"] == [-0.0001, 0.00075]
    assert doc["size_contracts"] == 1.0
    text = capsys.readouterr().out
    assert "exploration, no inference" in text
    assert "not built: MISSING_USDT" in text
    assert pr.main([*argv[:-2], "--end", "2025-11-02", "--out", str(out)]) == 2


def test_own_fills_by_hand() -> None:
    from mmlab.execution_simulator.queue import Fill, Order
    from test_queue import S, book

    # mids: 100 from 0 s, 100.5 from 1.5 s, 101 from 10 s, 102 from 100 s
    b = book(
        [
            (0.0, 99.0, 101.0, 5.0, 5.0, True),
            (1.5, 99.5, 101.5, 5.0, 5.0, True),
            (10.0, 100.0, 102.0, 5.0, 5.0, True),
            (100.0, 101.0, 103.0, 5.0, 5.0, True),
        ]
    )
    orders = [Order(1, 101.0, 2.0, 0, 0, 0, 0.0), Order(-1, 99.0, 1.0, 0, 0, 0, 0.0)]
    f = pr.own_fills(b, orders, [Fill(0, 1 * S, 2.0), Fill(1, 2 * S, 1.0)])  # a sell at 1 s, a buy at 2 s
    assert f["value"].to_list() == [202.0, 99.0]
    assert f["rs_60s"].to_list() == pytest.approx([0.0, 2.0 / 100.5])
    assert f["rs_300s"].to_list() == pytest.approx([-1.0 / 100.0, 3.0 / 100.5])
    s = pr.summarize_fills(f)
    assert s["fills"] == 2
    assert s["rs_60s_bp"] == pytest.approx(99.0 * 2.0 / 100.5 / 301.0 * 1e4)
    assert s["rs_300s_bp"] == pytest.approx((-202.0 / 100.0 + 99.0 * 3.0 / 100.5) / 301.0 * 1e4)
    assert s["net_bp"]["vip0"] == pytest.approx(s["rs_300s_bp"] - 2.0)
    assert s["net_bp"]["contract_fields"] == pytest.approx(s["rs_300s_bp"] + 1.0)
    empty = pr.summarize_fills(f.head(0))
    assert (empty["fills"], empty["rs_60s_bp"], empty["net_bp"]["vip0"]) == (0, None, None)


def test_screen_fills_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def f(rs300: float, counted: bool = True) -> dict[str, Any]:
        return {
            "counted": counted,
            "rs_60s_bp": rs300 + 1,
            "rs_300s_bp": rs300,
            "net_bp": {"vip0": rs300 - 2, "contract_fields": rs300 + 1},
        }

    sc = pr.screen_fills(
        {
            "A": {"all": f(-4.0), "without_runup": f(-4.0)},
            "B": {"all": f(3.0), "without_runup": f(3.0, counted=False)},
        },
        {"A": 0, "B": 1},
    )
    a = sc["all"]["all"]
    assert (a["contracts"], a["rs_300s_bp"], a["net_vip0_bp"]) == (
        2,
        pytest.approx(-0.5),
        pytest.approx(-2.5),
    )
    assert (a["positive_vip0"], a["positive_contract_fields"]) == (1, 1)
    assert sc["without_runup"]["all"]["contracts"] == 1
    ds = _dataset(tmp_path)
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT"]}, "holdout": {}}))
    cand = tmp_path / "c.csv"
    cand.write_text("name,available_to\nA_USDT,\n")
    out = tmp_path / "f.json"
    argv = [
        str(ds),
        "--universe",
        str(uni),
        "--candidates",
        str(cand),
        "--start",
        "2025-08-01",
        "--end",
        "2025-08-11",
        "--improve-ticks",
        "0",
        "--measure",
        "fills",
        "--workers",
        "1",
        "--out",
        str(out),
    ]
    assert pr.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["measure"] == "fills"
    r = doc["contracts"]["A_USDT"]["all"]
    assert r["fills"] > 0
    assert r["rs_60s_bp"] > 0  # constant book: a sell at 102 is 200 bp above the mid
    assert "own entry fills" in capsys.readouterr().out
