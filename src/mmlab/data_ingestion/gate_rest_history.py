"""Venue-wide trading history from public REST: daily candles for every tradable pair, plus a probe of
how far back the public trades endpoint reaches. Read-only, no credentials.

Candlestick rows (Gate API v4 docs): [ts, quote volume, close, high, low, open, base volume, closed].
Only closed daily candles are used.

Usage:
  python -m mmlab.data_ingestion.gate_rest_history OUT_DIR                 # Gate EU, 365 days
  python -m mmlab.data_ingestion.gate_rest_history OUT_DIR --venue global --max-pairs 300
"""

from __future__ import annotations

import argparse
import csv
import http.client
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from mmlab.data_ingestion.gate_venues import EU_URL, GLOBAL_URL, Venue

DAY_S = 86_400
TRADE_PROBE_DAYS = (1, 7, 30, 90, 180, 365)
_ERRORS = (RuntimeError, OSError, http.client.HTTPException, ValueError)


@dataclass(frozen=True, slots=True)
class DailyBar:
    pair: str
    day: str  # YYYY-MM-DD, UTC
    quote_volume: Decimal
    close: Decimal


def _dec(x: object) -> Decimal:
    try:
        return Decimal(str(x))
    except InvalidOperation:
        return Decimal(0)


def parse_candles(pair: str, rows: Iterable[Sequence[Any]]) -> list[DailyBar]:
    out = []
    for r in rows:
        if len(r) < 8 or str(r[7]).lower() != "true":
            continue  # malformed or the still-open current day
        day = datetime.fromtimestamp(int(r[0]), UTC).strftime("%Y-%m-%d")
        out.append(DailyBar(pair, day, _dec(r[1]), _dec(r[2])))
    return out


def daily_bars(v: Venue, pairs: Sequence[str], days: int, sleep: float) -> tuple[list[DailyBar], list[str]]:
    bars: list[DailyBar] = []
    failed: list[str] = []
    for i, pair in enumerate(pairs):
        if i and sleep:
            time.sleep(sleep)
        try:
            rows, _ = v.get(
                f"/spot/candlesticks?currency_pair={pair}&interval=1d&limit={min(days + 1, 1000)}"
            )
            bars += parse_candles(pair, rows)
        except _ERRORS:
            failed.append(pair)
    return bars, failed


def venue_daily(bars: Iterable[DailyBar]) -> dict[str, dict[str, Decimal]]:
    """day -> quote currency -> total quote volume."""
    out: defaultdict[str, defaultdict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for b in bars:
        out[b.day][b.pair.rsplit("_", 1)[-1]] += b.quote_volume
    return {d: dict(q) for d, q in out.items()}


def pair_stats(bars: Iterable[DailyBar]) -> dict[str, tuple[Decimal, Decimal, int, int]]:
    """pair -> (median daily volume, max daily volume, days with volume, days observed)."""
    by: defaultdict[str, list[Decimal]] = defaultdict(list)
    for b in bars:
        by[b.pair].append(b.quote_volume)
    return {p: (statistics.median(v), max(v), sum(x > 0 for x in v), len(v)) for p, v in by.items()}


def probe_trade_depth(v: Venue, pair: str, now_s: int) -> dict[int, str]:
    """For each look-back, whether /spot/trades returns trades in that one-day window."""
    out: dict[int, str] = {}
    for d in TRADE_PROBE_DAYS:
        start = now_s - d * DAY_S
        try:
            rows, _ = v.get(f"/spot/trades?currency_pair={pair}&limit=1&from={start}&to={start + DAY_S}")
            out[d] = "data" if rows else "empty"
        except _ERRORS as e:
            msg = str(e)
            out[d] = "error: " + (msg[msg.find("HTTP") :] if "HTTP" in msg else msg)[:200]
    return out


BUCKETS = (
    (Decimal(0), "0"),
    (Decimal(100), "<100"),
    (Decimal(1_000), "100-1k"),
    (Decimal(10_000), "1k-10k"),
    (Decimal(100_000), "10k-100k"),
)


def bucket(x: Decimal) -> str:
    if x <= 0:
        return "0"
    label = ">=100k"
    for upper, name in BUCKETS[1:]:
        if x < upper:
            label = name
            break
    return label


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Venue-wide daily volume history from Gate public REST.")
    p.add_argument("out")
    p.add_argument("--venue", choices=("eu", "global"), default="eu")
    p.add_argument("--base", default=None, help="override base URL")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--max-pairs", type=int, default=0, help="only the N most active pairs by 24 h volume")
    p.add_argument("--probe-pairs", type=int, default=3, help="pairs used for the trades depth probe")
    p.add_argument("--sleep", type=float, default=0.05)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Venue(a.venue, a.base or (EU_URL if a.venue == "eu" else GLOBAL_URL))

    try:
        pairs_raw, _ = v.get("/spot/currency_pairs")
        tickers, _ = v.get("/spot/tickers")
    except _ERRORS as e:
        print(f"{a.venue}: cannot list pairs: {e}")
        return 1
    vol24 = {str(t["currency_pair"]): _dec(t.get("quote_volume")) for t in tickers}
    pairs = sorted(
        (str(x["id"]) for x in pairs_raw if x.get("trade_status") == "tradable"),
        key=lambda s: -vol24.get(s, 0),
    )
    if a.max_pairs:
        pairs = pairs[: a.max_pairs]
    bars, failed = daily_bars(v, pairs, a.days, a.sleep)
    print(f"{a.venue}: {len(pairs)} pairs, {len(bars)} closed daily candles, failed pairs {len(failed)}")

    with open(out / f"{a.venue}_daily_bars.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "day", "quote_volume", "close"])
        w.writerows((b.pair, b.day, b.quote_volume, b.close) for b in bars)
    daily = venue_daily(bars)
    quotes = sorted(
        {q for d in daily.values() for q in d}, key=lambda q: -sum(d.get(q, 0) for d in daily.values())
    )
    with open(out / f"{a.venue}_venue_daily.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", *quotes])
        w.writerows([d, *(daily[d].get(q, 0) for q in quotes)] for d in sorted(daily))

    shown = quotes[:3]
    print("\nvenue volume by month (sum of daily quote volume; median day in brackets):")
    months: defaultdict[str, list[dict[str, Decimal]]] = defaultdict(list)
    for d in sorted(daily):
        months[d[:7]].append(daily[d])
    for m, ds in sorted(months.items()):
        cells = []
        for q in shown:
            vals = [x.get(q, Decimal(0)) for x in ds]
            cells.append(f"{q} {sum(vals):>14,.0f} [{statistics.median(vals):>11,.0f}]")
        print(f"  {m} ({len(ds):2d} d)  " + "  ".join(cells))

    stats = pair_stats(bars)
    last = sorted(daily)[-30:]
    recent = [b for b in bars if b.day in set(last)]
    rstats = pair_stats(recent)
    hist: defaultdict[str, int] = defaultdict(int)
    for med, _, _, _ in rstats.values():
        hist[bucket(med)] += 1
    order = ["0", "<100", "100-1k", "1k-10k", "10k-100k", ">=100k"]
    print(
        "\npairs by median daily quote volume, last 30 days: " + ", ".join(f"{k}: {hist[k]}" for k in order)
    )
    print("top 15 pairs by median daily volume, last 30 days (median / max / active days):")
    for pair, (med, mx, act, n) in sorted(rstats.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"  {pair:16} {med:>14,.0f} {mx:>14,.0f}  {act}/{n}")
    with open(out / f"{a.venue}_pair_stats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "median_all", "max_all", "active_days_all", "days_all", "median_30d"])
        w.writerows([pp, *stats[pp], rstats.get(pp, (Decimal(0),))[0]] for pp in sorted(stats))

    now = int(time.time())
    print("\npublic trades endpoint, data in a one-day window N days back:")
    for pair in pairs[: a.probe_pairs]:
        res = probe_trade_depth(v, pair, now)
        print(f"  {pair:16} " + "  ".join(f"{d}d: {r}" for d, r in res.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
