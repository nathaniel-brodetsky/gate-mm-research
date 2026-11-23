"""Compare Gate venues on public spot market data: global (api.gateio.ws) vs Gate EU (api.gateeu.com).

Read-only, no credentials. Per venue: network round trip (minimum TCP connect time), listed pairs by
quote currency, 24 h volume by quote currency, most active pairs. For pairs tradable on both venues,
order books are fetched back to back and compared; identical books would mean a shared matching engine.

Gate EU rejects requests without a `Timestamp` header (HTTP 400) even on public endpoints, and
`/spot/time` requires an API key there; pairs, tickers and order books do not.

Usage:
  python -m mmlab.data_ingestion.gate_venues OUT_DIR
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

GLOBAL_URL = "https://api.gateio.ws/api/v4"
EU_URL = "https://api.gateeu.com/api/v4"
_HEADERS = {"User-Agent": "mmlab-venue-check/0.1", "Accept": "application/json"}


class Venue:
    def __init__(self, name: str, base: str) -> None:
        self.name, self.base = name, base.rstrip("/")
        u = urlsplit(self.base)
        self._host, self._prefix = u.hostname or "", u.path
        self._port = u.port or (443 if u.scheme == "https" else 80)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self._conn = cls(u.netloc, timeout=30)

    def tcp_rtt_ms(self, n: int = 5) -> float:
        """Minimum TCP connect time: one network round trip, free of HTTP and TLS effects."""
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            with socket.create_connection((self._host, self._port), timeout=10):
                out.append((time.perf_counter() - t0) * 1000)
        return min(out)

    def get(self, path: str) -> tuple[Any, float]:
        """GET JSON; raises RuntimeError on a non-200 answer (connection is reset for the next call)."""
        headers = {**_HEADERS, "Timestamp": str(int(time.time()))}  # protocol header, not a credential
        t0 = time.perf_counter()
        try:
            self._conn.request("GET", self._prefix + path, headers=headers)
            r = self._conn.getresponse()
            body = r.read()
        except (OSError, http.client.HTTPException):
            self._conn.close()
            raise
        ms = (time.perf_counter() - t0) * 1000
        if r.status != 200:
            self._conn.close()
            raise RuntimeError(f"{self.name} {path}: HTTP {r.status} {body[:160]!r}")
        return json.loads(body), ms


def _dec(x: object) -> Decimal:
    try:
        return Decimal(str(x))
    except InvalidOperation:
        return Decimal(0)


def _top(book: dict[str, Any]) -> str:
    bid = book["bids"][0] if book.get("bids") else ["-", "-"]
    ask = book["asks"][0] if book.get("asks") else ["-", "-"]
    return f"{bid[1]}@{bid[0]} / {ask[1]}@{ask[0]}"


def survey(v: Venue, out: Path) -> dict[str, dict[str, Any]] | None:
    """Print one venue's summary; return tradable pair ids and tickers, or None if unavailable."""
    print(f"\n=== {v.name}: {v.base}")
    try:
        print(f"  network RTT (min TCP connect of 5): {v.tcp_rtt_ms():.1f} ms")
    except OSError as e:
        print(f"  network RTT: {type(e).__name__}: {e}")
    got: dict[str, Any] = {}
    for name, path in (
        ("time", "/spot/time"),
        ("pairs", "/spot/currency_pairs"),
        ("tickers", "/spot/tickers"),
    ):
        try:
            got[name], ms = v.get(path)
            print(f"  GET {path}: OK in {ms:.0f} ms")
        except (RuntimeError, OSError, http.client.HTTPException, ValueError) as e:
            print(f"  GET {path}: {type(e).__name__}: {str(e)[:160]}")
    if "pairs" not in got or "tickers" not in got:
        return None
    pairs: list[dict[str, Any]] = got["pairs"]
    tickers: list[dict[str, Any]] = got["tickers"]
    (out / f"{v.name}_currency_pairs.json").write_text(json.dumps(pairs))
    (out / f"{v.name}_tickers.json").write_text(json.dumps(tickers))
    tradable = [x for x in pairs if x.get("trade_status") == "tradable"]
    quotes = Counter(str(x.get("quote")) for x in tradable).most_common(8)
    print(
        f"  pairs listed {len(pairs)}, tradable {len(tradable)}; by quote: "
        + ", ".join(f"{q} {n}" for q, n in quotes)
    )
    vol: defaultdict[str, Decimal] = defaultdict(Decimal)
    for t in tickers:
        vol[str(t["currency_pair"]).rsplit("_", 1)[-1]] += _dec(t.get("quote_volume"))
    by_vol = sorted(vol.items(), key=lambda kv: -kv[1])[:6]
    print("  24h quote volume by quote currency: " + ", ".join(f"{q} {x:,.0f}" for q, x in by_vol))
    top = sorted(tickers, key=lambda t: -_dec(t.get("quote_volume")))[:10]
    print(
        "  top 10 by 24h quote volume: "
        + ", ".join(f"{t['currency_pair']} {_dec(t.get('quote_volume')):,.0f}" for t in top)
    )
    return {
        "pairs": {str(x["id"]): x for x in tradable},
        "tickers": {str(t["currency_pair"]): t for t in tickers},
    }


def compare_books(
    g: Venue, e: Venue, gd: dict[str, dict[str, Any]], ed: dict[str, dict[str, Any]], out: Path
) -> int:
    """Compare top-5 books of the most active common pairs; return how many were identical."""
    common = gd["pairs"].keys() & ed["pairs"].keys()
    print(f"\n=== pairs tradable on both venues: {len(common)}")
    identical = 0
    for sym in sorted(common, key=lambda s: -_dec(ed["tickers"].get(s, {}).get("quote_volume")))[:6]:
        try:
            bg, _ = g.get(f"/spot/order_book?currency_pair={sym}&limit=5&with_id=true")
            be, _ = e.get(f"/spot/order_book?currency_pair={sym}&limit=5&with_id=true")
        except (RuntimeError, OSError, http.client.HTTPException, ValueError) as ex:
            print(f"  {sym}: {type(ex).__name__}: {ex}")
            continue
        (out / f"book_global_{sym}.json").write_text(json.dumps(bg))
        (out / f"book_eu_{sym}.json").write_text(json.dumps(be))
        same = bg.get("asks") == be.get("asks") and bg.get("bids") == be.get("bids")
        identical += same
        vg = _dec(gd["tickers"].get(sym, {}).get("quote_volume"))
        ve = _dec(ed["tickers"].get(sym, {}).get("quote_volume"))
        print(
            f"  {sym}: identical top-5 {same}; 24h vol global {vg:,.0f} eu {ve:,.0f}\n"
            f"    global {_top(bg)}  (id {bg.get('id')})\n    eu     {_top(be)}  (id {be.get('id')})"
        )
    return identical


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare Gate global and Gate EU public spot market data.")
    p.add_argument("out", help="directory for raw JSON responses")
    p.add_argument("--global-url", default=GLOBAL_URL)
    p.add_argument("--eu-url", default=EU_URL)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g, e = Venue("global", a.global_url), Venue("eu", a.eu_url)
    gd, ed = survey(g, out), survey(e, out)
    if gd is None or ed is None:
        print("\nvenue comparison skipped: a venue did not return pairs and tickers")
        return 1
    compare_books(g, e, gd, ed, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
