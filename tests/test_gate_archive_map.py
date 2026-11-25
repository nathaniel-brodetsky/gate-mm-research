"""Archive map: calendar hours, resumable scan with a fake probe, monthly summary."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mmlab.data_ingestion import gate_archive_map as am
from mmlab.data_ingestion.gate_archive import ArchiveRef, ProbeResult


def test_month_hours_walks_real_calendar_days() -> None:
    grid = list(am.month_hours("2024-02", "2024-03", [13, 1]))
    assert len(grid) == (29 + 31) * 2  # 2024 is a leap year
    assert grid[0] == datetime(2024, 2, 1, 1, tzinfo=UTC)
    assert grid[1] == datetime(2024, 2, 1, 13, tzinfo=UTC)
    assert grid[-1] == datetime(2024, 3, 31, 13, tzinfo=UTC)
    assert len(list(am.month_hours("2025-12", "2025-12", [0]))) == 31  # across a year end
    refs = list(am.refs("futures_usdt", ["A_USDT", "B_USDT"], grid[:3]))
    assert [r.period for r in refs] == ["2024020101", "2024020113", "2024020201"] * 2
    assert refs[0].url.endswith("/futures_usdt/orderbooks/202402/A_USDT-2024020101.csv.gz")


def _fake(sizes: dict[str, int], calls: list[str]):  # type: ignore[no-untyped-def]
    def probe(ref: ArchiveRef) -> ProbeResult:
        calls.append(ref.period)
        size = sizes.get(ref.period, -1)
        status = 0 if ref.period == "NETERR" else (200 if size >= 0 else 404)
        return ProbeResult(ref.biz, ref.type, ref.market, ref.period, ref.url, status, size, "", "HEAD", "")

    return probe


def test_scan_appends_and_resumes(tmp_path: Path) -> None:
    grid = list(am.month_hours("2024-02", "2024-02", [1]))[:4]
    todo = list(am.refs("futures_usdt", ["A_USDT"], grid))
    out = tmp_path / "map" / "a.csv"
    calls: list[str] = []
    sizes = {"2024020101": 1000, "2024020201": 50}
    assert am.scan(todo[:2], out, _fake(sizes, calls), sleep_s=0) == 2
    assert am.scan(todo, out, _fake(sizes, calls), sleep_s=0) == 2  # the first two are skipped
    assert calls == ["2024020101", "2024020201", "2024020301", "2024020401"]
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["status"] for r in rows] == ["200", "200", "404", "404"]
    assert am.scan(todo, out, _fake(sizes, calls), sleep_s=0) == 0


def _row(market: str, period: str, status: int, size: int) -> dict[str, str]:
    url = f"u/{market}/{period}"
    return {"market": market, "period": period, "status": str(status), "size_bytes": str(size), "url": url}


def test_summarize_months_and_thin_files() -> None:
    rows = [
        _row("A_USDT", "2024020101", 200, 1000),
        _row("A_USDT", "2024020113", 200, 900),
        _row("A_USDT", "2024020201", 200, 50),  # thin: a snapshot without updates
        _row("A_USDT", "2024020213", 404, -1),
        _row("A_USDT", "2024030101", 0, -1),
        _row("A_USDT", "2024030113", 206, 1100),
        _row("B_USDT", "2024020101", 404, -1),
    ]
    table = am.summarize(rows)
    feb, mar, b = table
    assert feb == {
        "market": "A_USDT",
        "month": "2024-02",
        "hours": 4,
        "with_file": 3,
        "thin": 1,
        "network_errors": 0,
        "median_mb": 0.001,  # 900 bytes, rounded to 3 decimals of MB
    }
    assert (mar["hours"], mar["with_file"], mar["thin"], mar["network_errors"]) == (2, 1, 0, 1)
    assert (b["market"], b["with_file"], b["median_mb"]) == ("B_USDT", 0, 0.0)
    assert am.summarize(rows, thin_share=0.01)[0]["thin"] == 0  # 50 > 1% of the median 950
    text = am.show(table)
    assert "A_USDT         2024-02     4     3     1   50.0" in text


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(am, "probe", lambda r, timeout: _fake({"2024020101": 10}, calls)(r))
    out = tmp_path / "m.csv"
    argv = ["scan", "--biz", "futures_usdt", "--markets", "A_USDT", "--start", "2024-02", "--end", "2024-02"]
    assert am.main([*argv, "--hours", "1", "--sleep", "0", "--out", str(out)]) == 0
    assert "probed 29 new of 29 files" in capsys.readouterr().out
    table = tmp_path / "t.csv"
    assert am.main(["summary", str(out), "--out", str(table)]) == 0
    assert "A_USDT         2024-02    29     1" in capsys.readouterr().out
    assert table.read_text().splitlines()[0] == "market,month,hours,with_file,thin,network_errors,median_mb"


def test_scan_writes_each_row_at_once(tmp_path: Path) -> None:
    out = tmp_path / "a.csv"
    todo = list(am.refs("futures_usdt", ["A_USDT"], list(am.month_hours("2024-02", "2024-02", [1]))[:3]))
    seen: list[int] = []

    def probe(ref: ArchiveRef) -> ProbeResult:
        seen.append(len(out.read_text().splitlines()))  # what a crash at this moment would leave
        return ProbeResult(ref.biz, ref.type, ref.market, ref.period, ref.url, 200, 10, "", "HEAD", "")

    am.scan(todo, out, probe, sleep_s=0)
    assert seen == [1, 2, 3]  # the header, then one more row before every new probe


def test_scan_after_a_crash_before_the_header(tmp_path: Path) -> None:
    out = tmp_path / "a.csv"
    out.write_text("")
    todo = list(am.refs("futures_usdt", ["A_USDT"], list(am.month_hours("2024-02", "2024-02", [1]))[:2]))
    assert am.scan(todo, out, _fake({}, []), sleep_s=0) == 2
    with out.open(newline="") as fh:
        assert [r["status"] for r in csv.DictReader(fh)] == ["404", "404"]


def test_network_errors_are_asked_again_and_the_last_answer_counts(tmp_path: Path) -> None:
    out = tmp_path / "a.csv"
    todo = list(am.refs("futures_usdt", ["A_USDT"], list(am.month_hours("2024-02", "2024-02", [1]))[:2]))
    calls: list[str] = []

    def down(ref: ArchiveRef) -> ProbeResult:
        calls.append(ref.period)
        return ProbeResult(ref.biz, ref.type, ref.market, ref.period, ref.url, 0, -1, "", "HEAD", "timeout")

    assert am.scan(todo, out, down, sleep_s=0) == 2
    assert am.scan(todo, out, _fake({"2024020101": 10}, calls), sleep_s=0) == 2  # both asked again
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    feb = am.summarize(rows)[0]
    assert (feb["hours"], feb["with_file"], feb["network_errors"]) == (2, 1, 0)
