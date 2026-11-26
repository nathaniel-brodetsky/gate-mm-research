"""Candidate perpetuals: paging, the window filter, the cross-check, the archive sizes and the CLI."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mmlab.data_ingestion import perp_candidates as pc
from mmlab.data_ingestion.gate_archive import ArchiveRef, ProbeResult

TARDIS: dict[str, Any] = {
    "id": "gate-io-futures",
    "availableSymbols": [
        {"id": "OLD_USDT", "type": "perpetual", "availableSince": "2021-01-01T00:00:00.000Z"},
        {
            "id": "GONE_USDT",
            "type": "perpetual",
            "availableSince": "2024-03-01T00:00:00.000Z",
            "availableTo": "2025-10-02T00:00:00.000Z",
        },  # delisted inside the window: kept
        {
            "id": "EARLY_USDT",
            "type": "perpetual",
            "availableSince": "2023-01-01T00:00:00.000Z",
            "availableTo": "2025-07-01T00:00:00.000Z",
        },  # gone when the window starts
        {"id": "NEW_USDT", "type": "perpetual", "availableSince": "2025-06-01T00:00:00.000Z"},  # too late
        {"id": "BTC_USD", "type": "perpetual", "availableSince": "2020-07-01T00:00:00.000Z"},  # not USDT
        {"id": "BTC_USDT_20250926", "type": "future", "availableSince": "2025-06-01T00:00:00.000Z"},
        {"id": "NODATE_USDT", "type": "perpetual"},
        {"id": "FUT_USDT", "type": "future", "availableSince": "2021-01-01T00:00:00.000Z"},  # not perpetual
    ],
}
GATE = [
    {"name": "OLD_USDT", "launch_time": 1609459200, "quanto_multiplier": "10", "in_delisting": False},
    {"name": "NEW_USDT", "launch_time": 1748736000, "quanto_multiplier": "1", "in_delisting": False},
    {"name": "HIDDEN_USDT", "launch_time": 1700000000, "quanto_multiplier": "1", "in_delisting": True},
    {"name": "LATE_USDT", "launch_time": 1780000000, "quanto_multiplier": "1", "in_delisting": False},
]


def test_candidates() -> None:
    rows = pc.candidates(TARDIS, GATE, date(2025, 7, 1), date(2025, 6, 1))
    assert [r["name"] for r in rows] == ["GONE_USDT", "OLD_USDT"]
    gone, old = rows
    assert gone["available_to"] == "2025-10-02"
    assert gone["listed_on_gate_now"] is False
    assert gone["gate_quanto_multiplier"] == ""
    assert old == {
        "name": "OLD_USDT",
        "available_since": "2021-01-01",
        "available_to": "",
        "listed_on_gate_now": True,
        "gate_launch_time": "2021-01-01",
        "gate_quanto_multiplier": "10",
        "gate_in_delisting": False,
    }
    assert pc.gate_only(TARDIS, GATE, date(2025, 6, 1)) == ["HIDDEN_USDT"]


def test_gate_contracts_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "PAGE", 2)
    names = [f"C{i}_USDT" for i in range(5)]
    calls: list[str] = []

    def paged(url: str) -> Any:
        calls.append(url)
        off = int(url.rsplit("offset=", 1)[1])
        return [{"name": n} for n in names[off : off + 2]]

    assert [c["name"] for c in pc.gate_contracts(paged)] == names
    assert len(calls) == 3  # 2 + 2 + 1
    calls.clear()

    def ignores_paging(url: str) -> Any:
        calls.append(url)
        return [{"name": n} for n in names]

    assert len(pc.gate_contracts(ignores_paging)) == 5
    assert len(calls) == 2  # the second page adds nothing new


def _probe(ref: ArchiveRef) -> ProbeResult:
    size = 1234 if ref.market == "OLD_USDT" else -1
    return ProbeResult(
        **asdict(ref),
        status=200 if size > 0 else 404,
        size_bytes=size,
        last_modified="",
        method="HEAD",
        error="",
    )


def test_add_trades_sizes() -> None:
    rows = pc.candidates(TARDIS, GATE, date(2025, 7, 1), date(2025, 6, 1))
    seen: list[str] = []

    def probe(ref: ArchiveRef) -> ProbeResult:
        seen.append(ref.url)
        return _probe(ref)

    pc.add_trades_sizes(rows, "2025-06", probe, base="http://x", sleep_s=0)
    assert seen == [
        "http://x/futures_usdt/trades/202506/GONE_USDT-202506.csv.gz",
        "http://x/futures_usdt/trades/202506/OLD_USDT-202506.csv.gz",
    ]
    assert [(r["trades_status"], r["trades_bytes"]) for r in rows] == [(404, -1), (200, 1234)]


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(pc, "probe", lambda ref, timeout: _probe(ref))

    def fetch(url: str) -> Any:
        return TARDIS if url == pc.TARDIS_URL else GATE

    out = tmp_path / "res" / "candidates.csv"
    assert pc.main([str(tmp_path / "raw"), str(out), "--sleep", "0"], fetch) == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["name"] for r in rows] == ["GONE_USDT", "OLD_USDT"]
    assert rows[1]["trades_bytes"] == "1234"
    assert out.read_bytes().count(b"\r") == 0
    assert len(list((tmp_path / "raw").glob("tardis_gate-io-futures_*.json"))) == 1
    assert len(list((tmp_path / "raw").glob("gate_futures_usdt_contracts_*.json"))) == 1
    text = capsys.readouterr().out
    assert "candidates 2" in text
    assert "1 of them no longer listed" in text
    assert "1 with a trades file for 2025-06" in text
    assert "unknown to Tardis: 1 (HIDDEN_USDT)" in text
