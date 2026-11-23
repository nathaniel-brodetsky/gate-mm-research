"""Venue comparison against two local mock venues.

The mock EU venue has its own books, rejects requests without a Timestamp header and wants a key on
/spot/time, like Gate EU on 2026-11-23.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mmlab.data_ingestion import gate_venues as gv

VENUES: dict[str, dict[str, Any]] = {
    "/g": {
        "pairs": [{"id": "BTC_USDT", "quote": "USDT", "trade_status": "tradable"},
                  {"id": "BTC_USDC", "quote": "USDC", "trade_status": "tradable"}],
        "tickers": [{"currency_pair": "BTC_USDT", "quote_volume": "1000000000"},
                    {"currency_pair": "BTC_USDC", "quote_volume": "1382250"}],
        "book": {"id": 1, "asks": [["62986.5", "0.007948"]], "bids": [["62971.04", "0.003948"]]},
    },
    "/e": {
        "pairs": [{"id": "BTC_USDC", "quote": "USDC", "trade_status": "tradable"},
                  {"id": "USDC_EUR", "quote": "EUR", "trade_status": "tradable"},
                  {"id": "OLD_EUR", "quote": "EUR", "trade_status": "untradable"}],
        "tickers": [{"currency_pair": "BTC_USDC", "quote_volume": "21211"},
                    {"currency_pair": "USDC_EUR", "quote_volume": "24832"}],
        "book": {"id": 9, "asks": [["63058.66", "0.017584"]], "bids": [["62990.07", "0.097081"]]},
    },
}  # fmt: skip
SEEN_TIMESTAMP: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, code: int, body: object) -> None:
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        venue, rest = VENUES[self.path[:2]], self.path[2:]
        if self.path.startswith("/e"):
            ts = self.headers.get("Timestamp", "")
            SEEN_TIMESTAMP.append(ts)
            if not ts.isdigit():
                return self._send(400, {"label": "MISSING_REQUIRED_HEADER", "message": "Timestamp"})
            if rest.startswith("/api/v4/spot/time"):
                return self._send(400, {"label": "MISSING_REQUIRED_HEADER", "message": "KEY"})
        for key, prefix in (("pairs", "currency_pairs"), ("tickers", "tickers"), ("book", "order_book")):
            if rest.startswith(f"/api/v4/spot/{prefix}"):
                return self._send(200, venue[key])
        if rest.startswith("/api/v4/spot/time"):
            return self._send(200, {"server_time": 1})
        return self._send(404, {})


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def test_separate_books_are_reported(base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = gv.main([str(tmp_path), "--global-url", f"{base}/g/api/v4", "--eu-url", f"{base}/e/api/v4"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "pairs listed 3, tradable 2; by quote: USDC 1, EUR 1" in text
    assert "GET /spot/time: RuntimeError" in text  # EU needs a key there, the survey goes on
    assert "pairs tradable on both venues: 1" in text
    assert "BTC_USDC: identical top-5 False" in text
    assert (tmp_path / "book_eu_BTC_USDC.json").exists()
    assert SEEN_TIMESTAMP
    assert all(ts.isdigit() for ts in SEEN_TIMESTAMP)


def test_unreachable_venue_fails_cleanly(base: str, tmp_path: Path) -> None:
    rc = gv.main([str(tmp_path), "--global-url", f"{base}/g/api/v4", "--eu-url", "http://127.0.0.1:1/api/v4"])
    assert rc == 1
