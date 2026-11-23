"""Hour-by-hour coverage of Gate `orderbooks` archives: download, replay, chain consecutive hours.

A chain break (next snapshot id != last covered id, or replayed book != next snapshot) means updates
are missing from the archive; the next hourly snapshot resynchronises the book.

Usage:
  python -m mmlab.orderbook.gate_coverage --biz futures_usdt --market GPS_USDT \\
      --start 2026-10-15T00 --end 2026-10-16T00 --out coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mmlab.data_ingestion.gate_archive import BASE_URL, BIZ, archive_ref
from mmlab.data_ingestion.gate_download import fetch
from mmlab.data_ingestion.gate_orderbook_file import Kind, iter_orderbook_file
from mmlab.orderbook.gate_replay import ReplayReport, check_chain, replay
from mmlab.orderbook.l2 import L2Book


@dataclass(frozen=True, slots=True)
class HourCoverage:
    hour: str  # YYYY-MM-DDTHH, UTC
    status: str  # missing | clean | issues
    snapshot_levels: int
    updates: int
    last_row_s: float  # seconds from hour start to the last row; -1 if missing
    id_breaks: int
    crossed_batches: int
    negative_levels: int
    non_monotonic_ts: int
    late_set_rows: int
    chain: str  # n/a | ok | broken  (relative to the previous hour)
    id_gap: int  # this snapshot id - previous last covered id; 0 when chained
    level_mismatches: int


def parse_hour(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    raise ValueError(f"bad hour: {s!r} (use YYYY-MM-DD or YYYY-MM-DDTHH)")


def hours(start: datetime, end: datetime) -> Iterator[datetime]:
    t = start
    while t < end:
        yield t
        t += timedelta(hours=1)


def kind_of(biz: str) -> Kind:
    return "spot" if biz == "spot" else "futures"


def scan(
    biz: str,
    market: str,
    start: datetime,
    end: datetime,
    root: Path,
    base: str | None = None,
    opener: urllib.request.OpenerDirector | None = None,
    sleep: float = 0.0,
) -> list[HourCoverage]:
    kind = kind_of(biz)
    out: list[HourCoverage] = []
    prev: tuple[L2Book, ReplayReport] | None = None
    snaps: list[L2Book] = []
    for i, t in enumerate(hours(start, end)):
        if i and sleep:
            time.sleep(sleep)
        ym = t.strftime("%Y%m")
        ref = archive_ref(biz, "orderbooks", market, ym, t.day, t.hour, base or BASE_URL)
        path = fetch(ref, root, opener)
        label = t.strftime("%Y-%m-%dT%H")
        if path is None:
            out.append(HourCoverage(label, "missing", 0, 0, -1.0, 0, 0, 0, 0, 0, "n/a", 0, 0))
            prev = None
            continue
        snaps.clear()
        book, rep = replay(iter_orderbook_file(path, kind), on_snapshot=lambda b: snaps.append(b.copy()))
        chain, gap, mism = "n/a", 0, 0
        if prev is not None:
            ch = check_chain(prev[0], prev[1], snaps[0], rep)
            chain = "ok" if ch.ok else "broken"
            gap, mism = rep.snapshot_id - prev[1].last_covered_id, ch.level_mismatches
        out.append(
            HourCoverage(
                label,
                "clean" if rep.clean else "issues",
                rep.snapshot_rows,
                rep.updates,
                round((rep.last_ts_us - rep.snapshot_ts_us) / 1e6, 1),
                rep.id_breaks,
                rep.crossed_batches,
                rep.negative_levels,
                rep.non_monotonic_ts,
                rep.late_set_rows,
                chain,
                gap,
                mism,
            )
        )
        prev = (book, rep)
    return out


def main(argv: Sequence[str] | None = None, opener: urllib.request.OpenerDirector | None = None) -> int:
    p = argparse.ArgumentParser(description="Hourly coverage of Gate orderbooks archives.")
    p.add_argument("--biz", required=True, choices=BIZ)
    p.add_argument("--market", required=True)
    p.add_argument("--start", required=True, help="UTC, YYYY-MM-DD or YYYY-MM-DDTHH (inclusive)")
    p.add_argument("--end", required=True, help="UTC, YYYY-MM-DD or YYYY-MM-DDTHH (exclusive)")
    p.add_argument("--data-dir", default="data/raw/gate")
    p.add_argument("--base", default=None, help="archive base URL (default: official)")
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--out", default=None, help="CSV path")
    a = p.parse_args(argv)

    rows = scan(
        a.biz, a.market, parse_hour(a.start), parse_hour(a.end), Path(a.data_dir), a.base, opener, a.sleep
    )
    for r in rows:
        print(
            f"{r.hour}  {r.status:7} levels {r.snapshot_levels:6} updates {r.updates:7} "
            f"last {r.last_row_s:7.1f}s  chain {r.chain:6} id_gap {r.id_gap:7} mism {r.level_mismatches}"
            + (
                f"  [id_breaks {r.id_breaks} crossed {r.crossed_batches} negative {r.negative_levels} "
                f"non_monotonic {r.non_monotonic_ts} late_set {r.late_set_rows}]"
                if r.status == "issues"
                else ""
            )
        )
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(HourCoverage)])
            w.writeheader()
            w.writerows(asdict(r) for r in rows)
    n = len(rows)
    missing = sum(r.status == "missing" for r in rows)
    issues = sum(r.status == "issues" for r in rows)
    broken = sum(r.chain == "broken" for r in rows)
    lost = sum(max(r.id_gap, 0) for r in rows if r.chain == "broken")
    print(
        f"SUMMARY {a.biz} {a.market}: hours {n}, missing {missing}, with issues {issues}, "
        f"chain breaks {broken}, missing update ids {lost}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
