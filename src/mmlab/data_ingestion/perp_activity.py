"""Activity of candidate perpetuals in the selection month, from Gate's monthly trades files (stage 4, 2b-2).

For every candidate of `perp_candidates` with a trades file in the month, the file is downloaded into
the archive cache and read: trades per calendar day (days without a trade count as 0), the median over
the month's days, active days, and the median daily notional in contracts x price. Sizes are in
contracts; a quote volume needs the contract's multiplier, which Gate's API gives only for contracts
still listed, so it is reported where known and left empty otherwise. The trade count needs no
multiplier and is the measure that a universe rule can use for every candidate alike.

Usage:
  python -m mmlab.data_ingestion.perp_activity results/perp_universe/candidates.csv \\
      results/perp_universe/activity_2025-06.csv --month 2025-06 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from mmlab.data_ingestion.gate_archive import BASE_URL, archive_ref
from mmlab.data_ingestion.gate_download import fetch
from mmlab.data_ingestion.gate_trades_file import iter_trades_file

DAY_US: Final = 86_400_000_000
FIELDS: Final = (
    "name",
    "listed_on_gate_now",
    "status",
    "days",
    "active_days",
    "trades",
    "median_daily_trades",
    "median_daily_contract_notional",
    "gate_quanto_multiplier",
    "median_daily_quote_volume",
)


def month_bounds(month: str) -> tuple[int, int]:
    """[start, end) of a YYYY-MM month in microseconds since the epoch."""
    y, m = (int(x) for x in month.split("-"))
    a = datetime(y, m, 1, tzinfo=UTC)
    b = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=UTC)
    return int(a.timestamp()) * 1_000_000, int(b.timestamp()) * 1_000_000


def day_stats(path: Path, start_us: int, end_us: int) -> tuple[list[int], list[float]]:
    """Trades and notional (contracts x price) per calendar day of [start, end)."""
    n = (end_us - start_us) // DAY_US
    count, notional = [0] * n, [0.0] * n
    for t in iter_trades_file(path, "futures", start_us, end_us):
        d = (t.ts_us - start_us) // DAY_US
        count[d] += 1
        notional[d] += float(t.qty * t.price)
    return count, notional


def activity(row: dict[str, str], month: str, raw_root: Path, base: str, retries: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        k: row.get(k, "") for k in ("name", "listed_on_gate_now", "gate_quanto_multiplier")
    }
    start_us, end_us = month_bounds(month)
    ref = archive_ref("futures_usdt", "trades", row["name"], month.replace("-", ""), base=base)
    path = fetch(ref, raw_root, retries=retries)
    if path is None:
        return out | {"status": "missing"}
    count, notional = day_stats(path, start_us, end_us)
    med_notional = statistics.median(notional)
    mult = row.get("gate_quanto_multiplier", "")
    out |= {
        "status": "ok",
        "days": len(count),
        "active_days": sum(c > 0 for c in count),
        "trades": sum(count),
        "median_daily_trades": statistics.median(count),
        "median_daily_contract_notional": round(med_notional, 6),
        "median_daily_quote_volume": round(med_notional * float(mult), 2) if mult else "",
    }
    return out


def _run(args: tuple[dict[str, str], str, Path, str, int]) -> dict[str, Any]:
    return activity(*args)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Activity of candidate perpetuals in the selection month.")
    p.add_argument("candidates")
    p.add_argument("out")
    p.add_argument("--month", default="2025-06")
    p.add_argument("--raw-dir", default="data/raw/gate")
    p.add_argument("--base", default=BASE_URL)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    with open(a.candidates, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["trades_status"] in ("200", "206")]
    t0 = time.monotonic()
    args = [(r, a.month, Path(a.raw_dir), a.base, a.retries) for r in rows]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            done = list(ex.map(_run, args, chunksize=1))
    else:
        done = [_run(x) for x in args]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS), lineterminator="\n")
        w.writeheader()
        w.writerows(done)
    ok = [d for d in done if d["status"] == "ok"]
    full = [d for d in ok if d["active_days"] >= 28]
    print(
        f"{a.month}: {len(rows)} candidates with a trades file, {len(ok)} read, {len(full)} active on >= 28 "
        f"days; {time.monotonic() - t0:.0f} s -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
