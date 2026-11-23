"""Dataset builder against a local mock archive with a truncated hour and a missing hour."""

from __future__ import annotations

import csv
import gzip
import json
import threading
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from _gate_synth import SynthMarket
from mmlab.market_data import build_dataset as bd

T_OCT = 1_792_022_400  # 2026-10-15 00:00 UTC
FILES: dict[str, bytes] = {}
HITS: Counter[str] = Counter()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        HITS[self.path] += 1
        if "BAD_USDT" in self.path:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = FILES.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Length", str(len(body or b"")))
        self.end_headers()
        self.wfile.write(body or b"")


def _gz(rows: list[str]) -> bytes:
    return gzip.compress(("\n".join(rows) + "\n").encode())


@pytest.fixture(scope="module")
def archive() -> Iterator[tuple[str, dict[str, Any]]]:
    m = SynthMarket("spot", seed=31)
    info: dict[str, Any] = {"last_ts": {}}
    for h in range(6):
        rows = m.hour(T_OCT + 3600 * h, steps=300)
        if h == 2:
            rows = rows[: len(rows) - 80]  # outage: the end of hour 2 is lost
        if h != 4:  # hour 4 is never published
            FILES[f"/spot/orderbooks/202610/SYN_USDT-20261015{h:02d}.csv.gz"] = _gz(rows)
        info["last_ts"][h] = rows[-1].split(",")[0]
    FILES["/spot/deals/202610/SYN_USDT-202610.csv.gz"] = _gz(m.trade_rows())
    info["trade_rows"] = m.trade_rows()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", info
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def universe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"research": {"0": ["SYN_USDT"]}, "holdout": {"0": ["HOLD_USDT"]}}))
    return p


def _argv(universe: Path, out: Path, base: str, *extra: str) -> list[str]:
    return [
        str(universe),
        str(out),
        "--start",
        "2026-10-15T00",
        "--end",
        "2026-10-15T06",
        "--workers",
        "1",
        "--raw-dir",
        str(out.parent / "raw"),
        "--base",
        base,
        *extra,
    ]


def test_build_marks_gaps_and_writes_parquet(
    archive: tuple[str, dict[str, Any]], universe: Path, tmp_path: Path
) -> None:
    base, info = archive
    out = tmp_path / "ds"
    assert bd.main(_argv(universe, out, base)) == 0

    hours = list(csv.DictReader((out / "hours" / "SYN_USDT.csv").open()))
    assert [h["status"] for h in hours] == ["clean", "clean", "clean", "clean", "missing", "clean"]
    assert [h["chained"] for h in hours] == ["", "true", "true", "false", "", ""]

    tob = pl.read_parquet(out / "tob" / "SYN_USDT.parquet")
    assert tob["avail_us"].is_sorted()
    assert (tob["avail_us"] - tob["label_us"] == 100_000).all()
    sentinels = tob.filter(pl.col("bid").is_null() & ~pl.col("valid"))
    expected = sorted(round(float(info["last_ts"][h]) * 1e6) for h in (2, 3))
    assert (
        sentinels["label_us"].to_list() == expected
    )  # end of hour 2 (outage) and of hour 3 (hour 4 missing)
    assert tob.filter(pl.col("valid")).height > 100

    trades = pl.read_parquet(out / "trades" / "SYN_USDT.parquet")
    end_us = (T_OCT + 6 * 3600) * 1_000_000
    rows = [r.split(",") for r in info["trade_rows"]]
    inside = [r for r in rows if round(float(r[0]) * 1e6) < end_us]
    assert trades.height == len(inside)
    assert (trades["taker"] == 1).sum() == sum(r[4] == "1" for r in inside)  # archive code 1 = taker buy

    summary = list(csv.DictReader((out / "summary.csv").open()))
    assert (summary[0]["pair"], summary[0]["missing"], summary[0]["unchained"], summary[0]["sentinels"]) == (
        "SYN_USDT",
        "1",
        "1",
        "2",
    )


def test_resume_skips_built_pairs(
    archive: tuple[str, dict[str, Any]], universe: Path, tmp_path: Path
) -> None:
    base, _ = archive
    out = tmp_path / "ds"
    assert bd.main(_argv(universe, out, base)) == 0
    before = sum(HITS.values())
    assert bd.main(_argv(universe, out, base)) == 0
    assert sum(HITS.values()) == before


def test_delete_raw_removes_hourly_files(
    archive: tuple[str, dict[str, Any]], universe: Path, tmp_path: Path
) -> None:
    base, _ = archive
    out = tmp_path / "ds"
    assert bd.main(_argv(universe, out, base, "--delete-raw")) == 0
    raw = tmp_path / "raw"
    assert not list(raw.rglob("orderbooks/**/*.csv.gz"))
    assert list(raw.rglob("deals/**/*.csv.gz"))


def test_guards(archive: tuple[str, dict[str, Any]], universe: Path, tmp_path: Path) -> None:
    base, _ = archive
    out = tmp_path / "ds"
    late = [str(universe), str(out), "--start", "2026-10-31", "--end", "2026-11-02", "--base", base]
    assert bd.main(late) == 2
    assert bd.main([*_argv(universe, out, base), "--pairs", "HOLD_USDT"]) == 2
    assert not out.exists()


def test_failed_pair_is_reported_and_others_are_kept(
    archive: tuple[str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base, _ = archive
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["SYN_USDT", "BAD_USDT"]}, "holdout": {}}))
    out = tmp_path / "ds"
    argv = [*_argv(uni, out, base), "--workers", "2", "--retries", "1", "--backoff", "0"]
    assert bd.main(argv) == 1
    text = capsys.readouterr().out
    assert "FAILED BAD_USDT: RuntimeError: HTTPError: HTTP Error 503" in text
    assert (out / "tob" / "SYN_USDT.parquet").exists()
    assert not (out / "tob" / "BAD_USDT.parquet").exists()
    assert [r["pair"] for r in csv.DictReader((out / "summary.csv").open())] == ["SYN_USDT"]
