"""Month-by-month map of Gate's hourly archive files, from file headers only (stage 4, step 1).

Stage 1 found the futures order book archive of October to November 2026 mostly missing or holding only
the hourly snapshot. Before any futures study, this maps earlier months cheaply: for a sample of hours
of every calendar day it asks the archive for each hourly file's size without downloading it
(`gate_archive.probe`: HEAD, or a 1-byte ranged GET). Per market and month it reports how many sampled
hours have a file, and how many of those are "thin": smaller than `thin_share` of the market's median
file size over the whole scan, which is what an hour with a snapshot and no updates looks like.
Sizes say nothing about mid-hour gaps or broken chains; months that look complete here still need
`mmlab.orderbook.gate_coverage` before use.

The scan appends to its CSV as it goes and skips files already answered in it, so an interrupted scan
resumes; files that failed at network level are asked again, and the summary keeps the last answer.

Usage:
    python -m mmlab.data_ingestion.gate_archive_map scan --biz futures_usdt --markets BTC_USDT,GPS_USDT \\
        --start 2024-01 --end 2026-11 --hours 1,7,13,19 --out data/interim/archive_map/futures.csv
    python -m mmlab.data_ingestion.gate_archive_map summary data/interim/archive_map/futures.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mmlab.data_ingestion.gate_archive import BASE_URL, BIZ, ArchiveRef, ProbeResult, archive_ref, probe

THIN_SHARE = 0.10


def month_hours(start: str, end: str, hours: Sequence[int]) -> Iterator[datetime]:
    """The given UTC hours of every calendar day from month `start` to month `end`, both included."""
    t = datetime.strptime(start, "%Y-%m").replace(tzinfo=UTC)
    stop = datetime.strptime(end, "%Y-%m").replace(tzinfo=UTC)
    stop = (stop.replace(day=28) + timedelta(days=4)).replace(day=1)  # first day after `end`
    while t < stop:
        for h in sorted(hours):
            yield t.replace(hour=h)
        t += timedelta(days=1)


def refs(
    biz: str, markets: Sequence[str], grid: Sequence[datetime], base: str = BASE_URL
) -> Iterator[ArchiveRef]:
    for market in markets:
        for t in grid:
            yield archive_ref(biz, "orderbooks", market, t.strftime("%Y%m"), t.day, t.hour, base)


def scan(
    todo: Sequence[ArchiveRef],
    out: Path,
    probe_fn: Callable[[ArchiveRef], ProbeResult],
    sleep_s: float = 0.25,
) -> int:
    """Probe every ref not yet in `out` and append the results; returns how many were probed now."""
    names = [f.name for f in fields(ProbeResult)]
    done: set[str] = set()
    new = not out.exists() or out.stat().st_size == 0  # an empty file (a crash before the header) is new
    if not new:
        with out.open(newline="") as fh:
            done = {r["url"] for r in csv.DictReader(fh) if r["status"] != "0"}
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w" if new else "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        if new:
            w.writeheader()
            fh.flush()
        for ref in todo:
            if ref.url in done:
                continue
            if n:
                time.sleep(sleep_s)
            w.writerow(asdict(probe_fn(ref)))
            fh.flush()
            n += 1
    return n


def summarize(rows: Sequence[dict[str, str]], thin_share: float = THIN_SHARE) -> list[dict[str, object]]:
    """Per market and month: sampled hours, with a file, thin files, network errors, median size in MB."""
    last = {r["url"]: r for r in rows}  # a file asked again after a network error: its last answer
    by_market: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in last.values():
        by_market[r["market"]].append(r)
    out: list[dict[str, object]] = []
    for market in sorted(by_market):
        rs = by_market[market]
        sizes = [
            int(r["size_bytes"]) for r in rs if r["status"] in ("200", "206") and int(r["size_bytes"]) > 0
        ]
        median = statistics.median(sizes) if sizes else 0.0
        months: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in rs:
            months[r["period"][:6]].append(r)
        for ym in sorted(months):
            ms = months[ym]
            have = [r for r in ms if r["status"] in ("200", "206")]
            msizes = [int(r["size_bytes"]) for r in have if int(r["size_bytes"]) > 0]
            out.append(
                {
                    "market": market,
                    "month": f"{ym[:4]}-{ym[4:]}",
                    "hours": len(ms),
                    "with_file": len(have),
                    "thin": sum(0 <= int(r["size_bytes"]) < thin_share * median for r in have),
                    "network_errors": sum(r["status"] == "0" for r in ms),
                    "median_mb": round(statistics.median(msizes) / 1e6, 3) if msizes else 0.0,
                }
            )
    return out


def show(table: Sequence[dict[str, object]]) -> str:
    lines = [
        f"{'market':14} {'month':7} {'hours':>5} {'file':>5} {'thin':>5} {'full %':>6} "
        f"{'neterr':>6} {'MB':>8}"
    ]
    for r in table:
        hours = int(str(r["hours"]))
        full = int(str(r["with_file"])) - int(str(r["thin"]))
        lines.append(
            f"{r['market']!s:14} {r['month']!s:7} {hours:5d} {r['with_file']!s:>5} {r['thin']!s:>5} "
            f"{100 * full / hours if hours else 0:6.1f} {r['network_errors']!s:>6} {r['median_mb']!s:>8}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Month-by-month map of Gate's hourly archive, headers only.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--biz", required=True, choices=BIZ)
    s.add_argument("--markets", required=True, help="comma-separated")
    s.add_argument("--start", required=True, help="first month, YYYY-MM")
    s.add_argument("--end", required=True, help="last month, YYYY-MM (included)")
    s.add_argument("--hours", default="1,7,13,19", help="UTC hours sampled each day")
    s.add_argument("--base", default=BASE_URL)
    s.add_argument("--sleep", type=float, default=0.25)
    s.add_argument("--timeout", type=float, default=20.0)
    s.add_argument("--out", required=True)
    m = sub.add_parser("summary")
    m.add_argument("csv")
    m.add_argument("--thin-share", type=float, default=THIN_SHARE)
    m.add_argument("--out", help="CSV for the table")
    a = p.parse_args(argv)
    if a.cmd == "scan":
        grid = list(month_hours(a.start, a.end, [int(h) for h in a.hours.split(",")]))
        todo = list(refs(a.biz, [x.strip() for x in a.markets.split(",") if x.strip()], grid, a.base))
        n = scan(todo, Path(a.out), lambda r: probe(r, timeout=a.timeout), a.sleep)
        print(f"probed {n} new of {len(todo)} files; results in {a.out}")
        return 0
    with Path(a.csv).open(newline="") as fh:
        table = summarize(list(csv.DictReader(fh)), a.thin_share)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(a.out).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]) if table else ["market"], lineterminator="\n")
            w.writeheader()
            w.writerows(table)
    print(show(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
