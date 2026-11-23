"""Downloader and coverage scan against a local HTTP server with synthetic archives."""

from __future__ import annotations

import gzip
import threading
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from _gate_synth import T0, SynthMarket
from mmlab.data_ingestion.gate_archive import archive_ref
from mmlab.data_ingestion.gate_download import DownloadError, cache_path, fetch
from mmlab.orderbook import gate_coverage as gc

FILES: dict[str, bytes] = {}
HITS: Counter[str] = Counter()
FLAKY: dict[str, int] = {}  # path -> number of first requests answered by dropping the connection


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        HITS[self.path] += 1
        if HITS[self.path] <= FLAKY.get(self.path, 0):
            self.close_connection = True  # no status line at all: the client sees a dropped connection
            return
        if "FORBID" in self.path:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = FILES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _gz(rows: list[str]) -> bytes:
    return gzip.compress(("\n".join(rows) + "\n").encode())


def _key(biz: str, market: str, t: datetime) -> str:
    return f"/{biz}/orderbooks/{t:%Y%m}/{market}-{t:%Y%m%d%H}.csv.gz"


def test_fetch_caches_and_handles_404_and_corruption(
    base: str, opener: urllib.request.OpenerDirector, tmp_path: Path
) -> None:
    FILES["/spot/deals/202301/AAA_USDT-202301.csv.gz"] = gzip.compress(b"1,2,3,4,1\n")
    FILES["/spot/deals/202302/AAA_USDT-202302.csv.gz"] = b"not gzip at all"
    ok = archive_ref("spot", "deals", "AAA_USDT", "202301", base=base)
    p1 = fetch(ok, tmp_path, opener)
    p2 = fetch(ok, tmp_path, opener)
    assert p1 == p2 == cache_path(tmp_path, ok)
    assert p1 is not None
    assert p1.read_bytes() == FILES["/spot/deals/202301/AAA_USDT-202301.csv.gz"]
    assert HITS["/spot/deals/202301/AAA_USDT-202301.csv.gz"] == 1  # second call served from cache
    assert fetch(archive_ref("spot", "deals", "ZZZ_USDT", "202301", base=base), tmp_path, opener) is None
    bad = archive_ref("spot", "deals", "AAA_USDT", "202302", base=base)
    with pytest.raises(DownloadError):
        fetch(bad, tmp_path, opener, backoff_s=0)
    assert HITS["/spot/deals/202302/AAA_USDT-202302.csv.gz"] == 6  # first try + 5 retries
    assert not cache_path(tmp_path, bad).exists()
    assert not list(cache_path(tmp_path, bad).parent.glob("*.part"))


def test_scan_finds_truncated_hour_missing_hour_and_resync(
    base: str, opener: urllib.request.OpenerDirector, tmp_path: Path
) -> None:
    m = SynthMarket("futures", seed=11)
    t = [datetime.fromtimestamp(T0 + 3600 * k, UTC) for k in range(6)]
    FILES[_key("futures_usdt", "SYN_USDT", t[0])] = _gz(m.hour(T0, steps=400))
    FILES[_key("futures_usdt", "SYN_USDT", t[1])] = _gz(m.hour(T0 + 3600, steps=400))
    rows2 = m.hour(T0 + 7200, steps=400)
    FILES[_key("futures_usdt", "SYN_USDT", t[2])] = _gz(rows2[: len(rows2) - 120])  # outage: tail lost
    FILES[_key("futures_usdt", "SYN_USDT", t[3])] = _gz(m.hour(T0 + 10800, steps=400))
    m.hour(T0 + 14400, steps=400)  # hour 4 never published
    FILES[_key("futures_usdt", "SYN_USDT", t[5])] = _gz(m.hour(T0 + 18000, steps=400))

    rows = gc.scan(
        "futures_usdt", "SYN_USDT", t[0], datetime.fromtimestamp(T0 + 6 * 3600, UTC), tmp_path, base, opener
    )
    assert [r.status for r in rows] == ["clean", "clean", "clean", "clean", "missing", "clean"]
    assert [r.chain for r in rows] == ["n/a", "ok", "ok", "broken", "n/a", "n/a"]
    assert rows[3].id_gap > 1
    assert rows[3].level_mismatches > 0
    assert rows[1].id_gap == 0
    assert rows[2].last_row_s < 3600 < rows[1].last_row_s


def test_cli_summary(
    base: str, opener: urllib.request.OpenerDirector, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = SynthMarket("spot", seed=5)
    t0 = datetime.fromtimestamp(T0, UTC)
    FILES[_key("spot", "CLI_USDT", t0)] = _gz(m.hour(T0, steps=200))
    out = tmp_path / "cov.csv"
    argv = [
        "--biz",
        "spot",
        "--market",
        "CLI_USDT",
        "--start",
        "2026-11-15T12",
        "--end",
        "2026-11-15T14",
        "--data-dir",
        str(tmp_path / "d"),
        "--base",
        base,
        "--sleep",
        "0",
        "--out",
        str(out),
    ]
    assert gc.main(argv, opener) == 0
    text = capsys.readouterr().out
    assert "hours 2, missing 1, with issues 0, chain breaks 0" in text
    assert out.read_text().splitlines()[0].startswith("hour,status,")


def test_parse_hour() -> None:
    assert gc.parse_hour("2026-10-15") == datetime(2026, 10, 15, tzinfo=UTC)
    assert gc.parse_hour("2026-10-15T13") == datetime(2026, 10, 15, 13, tzinfo=UTC)
    with pytest.raises(ValueError, match="bad hour"):
        gc.parse_hour("2026/10/15")


def test_scan_reports_every_issue_counter(
    base: str, opener: urllib.request.OpenerDirector, tmp_path: Path
) -> None:
    t = datetime.fromtimestamp(T0, UTC)
    rows = [
        f"{T0},2,set,1,5,10,0",
        f"{T0}.2,2,make,1,1,11,1",
        f"{T0}.1,2,make,1,1,12,1",
        f"{T0}.05,2,make,1,1,13,1",
        f"{T0}.3,2,set,2,1,14,1",
    ]
    FILES[_key("spot", "ISS_USDT", t)] = _gz(rows)
    (r,) = gc.scan("spot", "ISS_USDT", t, datetime.fromtimestamp(T0 + 3600, UTC), tmp_path, base, opener)
    assert r.status == "issues"
    counters = (r.non_monotonic_ts, r.late_set_rows, r.id_breaks, r.crossed_batches, r.negative_levels)
    assert counters == (2, 1, 0, 0, 0)


def test_fetch_retries_transient_failures(
    base: str, opener: urllib.request.OpenerDirector, tmp_path: Path
) -> None:
    path = "/spot/deals/202303/FLK_USDT-202303.csv.gz"
    FILES[path] = gzip.compress(b"1,2,3,4,1\n")
    FLAKY[path] = 2
    ref = archive_ref("spot", "deals", "FLK_USDT", "202303", base=base)
    assert fetch(ref, tmp_path, opener, backoff_s=0) == cache_path(tmp_path, ref)
    assert HITS[path] == 3

    dead = "/spot/deals/202304/FLK_USDT-202304.csv.gz"
    FILES[dead] = FILES[path]
    FLAKY[dead] = 100
    ref2 = archive_ref("spot", "deals", "FLK_USDT", "202304", base=base)
    with pytest.raises(OSError):  # noqa: PT011
        fetch(ref2, tmp_path, opener, retries=2, backoff_s=0)
    assert HITS[dead] == 3
    assert not list(cache_path(tmp_path, ref2).parent.glob("*.part"))


def test_fetch_does_not_retry_client_errors(
    base: str, opener: urllib.request.OpenerDirector, tmp_path: Path
) -> None:
    ref = archive_ref("spot", "deals", "FORBID_USDT", "202305", base=base)
    with pytest.raises(urllib.error.HTTPError):
        fetch(ref, tmp_path, opener, backoff_s=0)
    assert HITS["/spot/deals/202305/FORBID_USDT-202305.csv.gz"] == 1
