"""Offline tests for the archive probe: URL scheme and probing against a local HTTP server."""

from __future__ import annotations

import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from mmlab.data_ingestion import gate_archive as ga

LAST_MODIFIED = "Mon, 02 Jan 2023 03:04:05 GMT"
EXISTING = {
    "/spot/deals/202301/BTC_USDT-202301.csv.gz": 1234,
    "/spot/orderbooks/202301/BTC_USDT-2023011505.csv.gz": 777,
    "/nohead/spot/deals/202301/BTC_USDT-202301.csv.gz": 55,
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_HEAD(self) -> None:
        if self.path.startswith("/nohead/"):
            self.send_response(405)
            self.end_headers()
            return
        size = EXISTING.get(self.path)
        self.send_response(200 if size else 404)
        if size:
            self.send_header("Content-Length", str(size))
            self.send_header("Last-Modified", LAST_MODIFIED)
        self.end_headers()

    def do_GET(self) -> None:
        size = EXISTING.get(self.path)
        if not size:
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("Range") != "bytes=0-0":
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            self.wfile.write(b"x" * size)
            return
        self.send_response(206)
        self.send_header("Content-Range", f"bytes 0-0/{size}")
        self.send_header("Content-Length", "1")
        self.end_headers()
        self.wfile.write(b"x")


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass env proxies


def test_url_scheme_matches_official_examples() -> None:
    ref = ga.archive_ref("futures_btc", "orderbooks_slice", "BTC_USD", "202304", 25, 5)
    assert (
        ref.url == "https://download.gatedata.org/futures_btc/orderbooks_slice/202304/BTC_USD-2023042505.gz"
    )
    assert ref.period == "2023042505"
    ref = ga.archive_ref("spot", "deals", "ADA_BTC", "201801")
    assert ref.url == "https://download.gatedata.org/spot/deals/201801/ADA_BTC-201801.csv.gz"
    ref = ga.archive_ref("spot", "orderbooks", "BTC_USDT", "202301", 15, 5)
    assert ref.url.endswith("/spot/orderbooks/202301/BTC_USDT-2023011505.csv.gz")
    ref = ga.archive_ref("spot", "deals", "ADA_BTC", "201801", base="http://mirror.example/")
    assert ref.url == "http://mirror.example/spot/deals/201801/ADA_BTC-201801.csv.gz"


@pytest.mark.parametrize(
    "args",
    [
        ("perp", "deals", "A_B", "202301", 1, 0),
        ("spot", "deals", "A_B", "202313", 1, 0),
        ("spot", "deals", "A_B", "2023-01", 1, 0),
        ("spot", "orderbooks", "A_B", "202302", 30, 0),
        ("spot", "orderbooks", "A_B", "202301", 1, 24),
        ("spot", "deals", "btc_usdt", "202301", 1, 0),
        ("spot", "deals", "../X_Y", "202301", 1, 0),
        ("spot", "../deals", "A_B", "202301", 1, 0),
    ],
)
def test_invalid_inputs_rejected(args: tuple[str, str, str, str, int, int]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ga.archive_ref(*args)


def test_expand_monthly_ignores_days_and_hours() -> None:
    refs = list(ga.expand("spot", ["deals", "orderbooks"], ["A_B"], ["202301"], [1, 2], [0, 12]))
    assert [r.period for r in refs] == ["202301", "2023010100", "2023010112", "2023010200", "2023010212"]


def test_probe_found_missing_and_head_fallback(server: str, opener: urllib.request.OpenerDirector) -> None:
    refs = list(
        ga.expand(
            "spot", ["deals", "orderbooks"], ["BTC_USDT", "ZZZ_USDT"], ["202301"], [15], [5, 6], base=server
        )
    )
    got = {(r.market, r.type, r.period): r for r in (ga.probe(x, opener, 5) for x in refs)}
    hit = got[("BTC_USDT", "deals", "202301")]
    assert (hit.status, hit.size_bytes, hit.last_modified, hit.available) == (200, 1234, LAST_MODIFIED, True)
    assert got[("BTC_USDT", "orderbooks", "2023011505")].size_bytes == 777
    miss = got[("BTC_USDT", "orderbooks", "2023011506")]
    assert (miss.status, miss.size_bytes, miss.available) == (404, -1, False)
    assert got[("ZZZ_USDT", "deals", "202301")].status == 404

    ref = ga.archive_ref("spot", "deals", "BTC_USDT", "202301", base=server + "/nohead")
    r = ga.probe(ref, opener, 5)
    assert (r.status, r.size_bytes, r.method, r.error) == (206, 55, "GET", "")


def test_network_error_reported_not_raised(opener: urllib.request.OpenerDirector) -> None:
    ref = ga.archive_ref("spot", "deals", "BTC_USDT", "202301", base="http://127.0.0.1:1")
    r = ga.probe(ref, opener, 2)
    assert r.status == 0
    assert not r.available
    assert r.error


def test_cli_writes_csv(server: str, opener: urllib.request.OpenerDirector, tmp_path: Path) -> None:
    out = tmp_path / "probe.csv"
    argv = [
        "--base",
        server,
        "--biz",
        "spot",
        "--markets",
        "BTC_USDT",
        "--types",
        "deals,orderbooks",
        "--months",
        "2023-01",
        "--days",
        "15",
        "--hours",
        "5,6",
        "--sleep",
        "0",
        "--out",
        str(out),
    ]
    assert ga.main(argv, opener) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split(",")[:6] == ["biz", "type", "market", "period", "url", "status"]
    assert len(lines) == 1 + 3
    assert sum(",200," in line for line in lines) == 2


def test_cli_fails_when_every_request_fails(opener: urllib.request.OpenerDirector, tmp_path: Path) -> None:
    argv = [
        "--base",
        "http://127.0.0.1:1",
        "--biz",
        "spot",
        "--markets",
        "BTC_USDT",
        "--types",
        "deals",
        "--months",
        "2023-01",
        "--sleep",
        "0",
        "--timeout",
        "2",
        "--out",
        str(tmp_path / "p.csv"),
    ]
    assert ga.main(argv, opener) == 1
