"""Venue history against a local mock venue."""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterator
from decimal import Decimal as D
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from mmlab.data_ingestion import gate_rest_history as rh
from mmlab.data_ingestion.gate_venues import Venue

DAY = 86_400
T0 = 1_793_491_200  # 2026-11-01 00:00 UTC


def _candles(vols: list[str], closes: list[str]) -> list[list[str]]:
    rows = [
        [str(T0 + i * DAY), v, c, c, c, c, "1", "true"]
        for i, (v, c) in enumerate(zip(vols, closes, strict=True))
    ]
    rows.append([str(T0 + len(vols) * DAY), "999999", "1", "1", "1", "1", "1", "false"])  # current day, open
    return rows


CANDLES = {
    "BTC_USDC": _candles(["100", "300", "200"], ["60000", "61000", "62000"]),
    "USDC_EUR": _candles(["50", "50", "0"], ["0.9", "0.9", "0.9"]),
}
PAIRS = [
    {"id": "BTC_USDC", "trade_status": "tradable"},
    {"id": "USDC_EUR", "trade_status": "tradable"},
    {"id": "DEAD_EUR", "trade_status": "tradable"},  # candles fail with 400
    {"id": "OLD_EUR", "trade_status": "untradable"},
]
TICKERS = [
    {"currency_pair": "BTC_USDC", "quote_volume": "250"},
    {"currency_pair": "USDC_EUR", "quote_volume": "40"},
]


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
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path.endswith("/spot/currency_pairs"):
            return self._send(200, PAIRS)
        if u.path.endswith("/spot/tickers"):
            return self._send(200, TICKERS)
        if u.path.endswith("/spot/candlesticks"):
            rows = CANDLES.get(q["currency_pair"])
            return self._send(200, rows) if rows else self._send(400, {"label": "INVALID_CURRENCY_PAIR"})
        if u.path.endswith("/spot/trades"):
            # like Gate on 2026-11-23: windows older than 30 days are rejected
            if int(q["from"]) < int(q["now"]) - 30 * DAY:
                msg = "for public trades older than 30 days, download from historical data instead"
                return self._send(400, {"label": "INVALID_PARAM_VALUE", "message": msg})
            return self._send(200, [{"id": "1"}] if q["currency_pair"] == "BTC_USDC" else [])
        return self._send(404, {})


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/api/v4"
    srv.shutdown()
    srv.server_close()


def test_parse_candles_drops_open_and_malformed_rows() -> None:
    bars = rh.parse_candles("X_EUR", [*CANDLES["BTC_USDC"], ["1", "2"]])
    assert [(b.day, b.quote_volume) for b in bars] == [
        ("2026-11-01", D(100)),
        ("2026-11-02", D(300)),
        ("2026-11-03", D(200)),
    ]


def test_aggregation_and_buckets() -> None:
    bars = rh.parse_candles("BTC_USDC", CANDLES["BTC_USDC"]) + rh.parse_candles(
        "USDC_EUR", CANDLES["USDC_EUR"]
    )
    daily = rh.venue_daily(bars)
    assert daily["2026-11-02"] == {"USDC": D(300), "EUR": D(50)}
    stats = rh.pair_stats(bars)
    assert stats["BTC_USDC"] == (D(200), D(300), 3, 3)
    assert stats["USDC_EUR"] == (D(50), D(50), 2, 3)
    assert [rh.bucket(D(x)) for x in ("0", "5", "100", "999", "5000", "50000", "100000")] == [
        "0",
        "<100",
        "100-1k",
        "100-1k",
        "1k-10k",
        "10k-100k",
        ">=100k",
    ]


def test_main_end_to_end(
    base: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_probe = rh.probe_trade_depth

    def probe(v: Venue, pair: str, now_s: int) -> dict[int, str]:
        # pass "now" to the mock so it can decide what is "recent"
        orig_get = v.get
        monkeypatch.setattr(v, "get", lambda path: orig_get(f"{path}&now={now_s}"))
        return real_probe(v, pair, now_s)

    monkeypatch.setattr(rh, "probe_trade_depth", probe)
    assert rh.main([str(tmp_path), "--base", base, "--sleep", "0", "--probe-pairs", "1"]) == 0
    text = capsys.readouterr().out
    assert "3 pairs, 6 closed daily candles, failed pairs 1" in text
    assert "2026-11 ( 3 d)  USDC            600 [        200]  EUR            100 [         50]" in text
    assert "BTC_USDC         1d: data  7d: data  30d: data  90d: error: HTTP 400" in text
    assert "for public trades older than 30 days" in text
    rows = list(csv.reader((tmp_path / "eu_venue_daily.csv").open()))
    assert rows[0] == ["day", "USDC", "EUR"]
    assert rows[2] == ["2026-11-02", "300", "50"]


def test_main_fails_without_pairs(tmp_path: Path) -> None:
    assert rh.main([str(tmp_path), "--base", "http://127.0.0.1:1/api/v4"]) == 1
