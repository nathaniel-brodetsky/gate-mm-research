"""Activity of candidate perpetuals: month bounds, per-day counts by hand, missing files, the CLI."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from mmlab.data_ingestion import perp_activity as pa
from mmlab.data_ingestion.gate_archive import archive_ref
from mmlab.data_ingestion.gate_download import cache_path

JUNE = 1_748_736_000  # 2025-06-01 00:00 UTC


def test_month_bounds() -> None:
    a, b = pa.month_bounds("2025-06")
    assert a == JUNE * 1_000_000
    assert (b - a) // pa.DAY_US == 30
    a, b = pa.month_bounds("2025-12")
    assert (b - a) // pa.DAY_US == 31


def _file(root: Path, name: str, lines: list[str]) -> None:
    path = cache_path(root, archive_ref("futures_usdt", "trades", name, "202506"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))


LINES = [
    f"{JUNE - 5}.0,1,1.0,10",  # May: outside the month
    f"{JUNE + 10}.5,2,2.0,3",
    f"{JUNE + 20}.0,3,2.5,-4",  # a sell: notional counts the size, not its sign
    f"{JUNE + 86_400 * 3 + 7}.0,4,3.0,1",
    f"{JUNE + 86_400 * 30}.0,5,9.0,1",  # July 1: outside
]


def test_day_stats_by_hand(tmp_path: Path) -> None:
    _file(tmp_path, "A_USDT", LINES)
    path = cache_path(tmp_path, archive_ref("futures_usdt", "trades", "A_USDT", "202506"))
    count, notional = pa.day_stats(path, *pa.month_bounds("2025-06"))
    assert len(count) == 30
    assert count[:5] == [2, 0, 0, 1, 0]
    assert sum(count) == 3
    assert notional[0] == pytest.approx(2.0 * 3 + 2.5 * 4)
    assert notional[3] == pytest.approx(3.0)


def test_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        f"{JUNE + 86_400 * d + 60}.0,{d},2.0,5" for d in range(30) for _ in range(d % 3)
    ]  # 0,1,2,0,1,2..
    _file(tmp_path, "B_USDT", lines)
    row = {"name": "B_USDT", "listed_on_gate_now": "True", "gate_quanto_multiplier": "10"}
    a = pa.activity(row, "2025-06", tmp_path, "http://127.0.0.1:9", 0)
    assert a["status"] == "ok"
    assert a["days"] == 30
    assert a["active_days"] == 20
    assert a["trades"] == 30
    assert a["median_daily_trades"] == 1
    assert a["median_daily_contract_notional"] == 10.0  # one trade of 5 contracts at 2
    assert a["median_daily_quote_volume"] == 100.0
    gone = pa.activity({**row, "gate_quanto_multiplier": ""}, "2025-06", tmp_path, "http://127.0.0.1:9", 0)
    assert gone["median_daily_quote_volume"] == ""
    monkeypatch.setattr(pa, "fetch", lambda *a, **k: None)
    missing = pa.activity({**row, "name": "C_USDT"}, "2025-06", tmp_path, "http://127.0.0.1:9", 0)
    assert missing == {
        "name": "C_USDT",
        "listed_on_gate_now": "True",
        "gate_quanto_multiplier": "10",
        "status": "missing",
    }


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw"
    _file(raw, "A_USDT", LINES)
    _file(raw, "FULL_USDT", [f"{JUNE + 86_400 * d + 1}.0,{d},1.0,1" for d in range(28)])  # exactly 28 days
    cand = tmp_path / "candidates.csv"
    with cand.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["name", "listed_on_gate_now", "gate_quanto_multiplier", "trades_status"]
        )
        w.writeheader()
        w.writerow(
            {
                "name": "A_USDT",
                "listed_on_gate_now": "False",
                "gate_quanto_multiplier": "",
                "trades_status": "200",
            }
        )
        w.writerow(
            {
                "name": "FULL_USDT",
                "listed_on_gate_now": "True",
                "gate_quanto_multiplier": "1",
                "trades_status": "200",
            }
        )
        w.writerow(
            {
                "name": "NOFILE_USDT",
                "listed_on_gate_now": "True",
                "gate_quanto_multiplier": "1",
                "trades_status": "404",
            }
        )
    out = tmp_path / "act.csv"
    argv = [str(cand), str(out), "--raw-dir", str(raw), "--base", "http://127.0.0.1:9", "--workers", "1"]
    assert pa.main(argv) == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["name"] for r in rows] == ["A_USDT", "FULL_USDT"]
    assert rows[1]["active_days"] == "28"
    assert rows[0]["median_daily_quote_volume"] == ""
    assert out.read_bytes().count(b"\r") == 0
    assert "2 candidates with a trades file, 2 read, 1 active on >= 28 days" in capsys.readouterr().out
