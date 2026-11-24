"""Build the research dataset of a universe: top of book and trades per research pair, as Parquet.

For every research pair (holdout pairs only with --holdout-pairs and an H-003 test result that is
accepted or rejected, see mmlab.hypothesis_registry.holdout) and every hour of the period, the hourly spot
`orderbooks` file is downloaded into the cache, replayed, reduced to top of book with availability time
(`mmlab.orderbook.top_of_book`) and chained to the previous hour. Monthly `deals` files give the trades.

Validity: rows are invalid after an anomaly inside an hour. When an hour is missing or does not chain to
the previous one, an invalid sentinel row is added right after the last known update of the previous
hour, so an as-of lookup never carries a stale book across a gap.

Holdout guard: periods reaching 2026-11-01 or later are refused unless --open-time-holdout is given
(H-001, H-002, D-001).

Usage:
  python -m mmlab.market_data.build_dataset configs/universe_U001.json data/processed/U001 \\
      --start 2026-08-01 --end 2026-11-01 --workers 6 [--pairs A_USDT,B_USDT] [--delete-raw]
Writes per pair tob/{pair}.parquet, trades/{pair}.parquet, hours/{pair}.csv, and summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from mmlab.data_ingestion.gate_archive import BASE_URL, archive_ref
from mmlab.data_ingestion.gate_download import fetch
from mmlab.data_ingestion.gate_orderbook_file import iter_orderbook_file
from mmlab.data_ingestion.gate_trades_file import Trade, check_sequence, iter_trades_file
from mmlab.hypothesis_registry.holdout import holdout_unlocked
from mmlab.orderbook.gate_coverage import hours, parse_hour
from mmlab.orderbook.gate_replay import ReplayReport, check_chain
from mmlab.orderbook.l2 import L2Book
from mmlab.orderbook.top_of_book import BATCH_US, TopOfBook, top_of_book

TIME_HOLDOUT_START = datetime(2026, 11, 1, tzinfo=UTC)
TOB_SCHEMA = {
    "label_us": pl.Int64,
    "avail_us": pl.Int64,
    "bid": pl.Float64,
    "bid_qty": pl.Float64,
    "ask": pl.Float64,
    "ask_qty": pl.Float64,
    "valid": pl.Boolean,
}
TRADE_SCHEMA = {
    "ts_us": pl.Int64,
    "trade_id": pl.Int64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "taker": pl.Int8,
}


@dataclass(frozen=True)
class PairSummary:
    pair: str
    hours: int
    missing: int
    with_issues: int
    unchained: int
    tob_rows: int
    sentinels: int
    trades: int
    trade_id_gaps: int
    raw_mb: float
    seconds: float


def _f(x: object) -> float | None:
    return None if x is None else float(str(x))


def _tob_frame(rows: Sequence[TopOfBook]) -> pl.DataFrame:
    return pl.DataFrame(
        [(r.label_us, r.avail_us, _f(r.bid), _f(r.bid_qty), _f(r.ask), _f(r.ask_qty), r.valid) for r in rows],
        schema=TOB_SCHEMA,
        orient="row",
    )


def _collect(trades: Iterable[Trade], cols: dict[str, list[Any]]) -> Iterator[Trade]:
    """Pass trades through while appending plain values to columns (no Trade objects are kept)."""
    for x in trades:
        cols["ts_us"].append(x.ts_us)
        cols["trade_id"].append(x.trade_id)
        cols["price"].append(float(x.price))
        cols["qty"].append(float(x.qty))
        cols["taker"].append(1 if x.taker == "buy" else -1)
        yield x


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def build_pair(
    pair: str,
    start: datetime,
    end: datetime,
    raw_root: Path,
    out: Path,
    base: str,
    delete_raw: bool,
    retries: int = 5,
    backoff_s: float = 2.0,
) -> PairSummary:
    t0 = time.monotonic()
    frames: list[pl.DataFrame] = []  # one small frame per hour keeps memory flat
    hour_rows: list[dict[str, Any]] = []
    prev: tuple[L2Book, ReplayReport] | None = None
    last_rep: ReplayReport | None = None
    snaps: list[L2Book] = []
    raw = 0
    sentinels = 0

    def sentinel(rep: ReplayReport) -> None:
        nonlocal sentinels
        frames.append(
            _tob_frame([TopOfBook(rep.last_ts_us, rep.last_ts_us + BATCH_US, None, None, None, None, False)])
        )
        sentinels += 1

    for t in hours(start, end):
        ref = archive_ref("spot", "orderbooks", pair, t.strftime("%Y%m"), t.day, t.hour, base)
        path = fetch(ref, raw_root, retries=retries, backoff_s=backoff_s)
        label = t.strftime("%Y-%m-%dT%H")
        if path is None:
            if last_rep is not None and prev is not None:
                sentinel(last_rep)
            prev = None
            hour_rows.append({"hour": label, "status": "missing", "chained": "", "updates": 0})
            continue
        raw += path.stat().st_size
        snaps.clear()
        tob, rep, book = top_of_book(
            iter_orderbook_file(path, "spot"), on_snapshot=lambda b: snaps.append(b.copy())
        )
        chained = prev is not None and check_chain(prev[0], prev[1], snaps[0], rep).ok
        if prev is not None and not chained:
            sentinel(prev[1])
        frames.append(_tob_frame(tob))
        hour_rows.append(
            {
                "hour": label,
                "status": "clean" if rep.clean else "issues",
                "chained": "" if prev is None else str(chained).lower(),
                "updates": rep.updates,
                "last_row_s": round((rep.last_ts_us - rep.snapshot_ts_us) / 1e6, 1),
                "id_breaks": rep.id_breaks,
                "crossed_batches": rep.crossed_batches,
                "negative_levels": rep.negative_levels,
                "non_monotonic_ts": rep.non_monotonic_ts,
                "late_set_rows": rep.late_set_rows,
                "tob_rows": len(tob),
            }
        )
        prev, last_rep = (book, rep), rep
        if delete_raw:
            path.unlink()

    tob_df = (pl.concat(frames) if frames else _tob_frame([])).sort("avail_us", maintain_order=True)
    frames.clear()

    start_us, end_us = int(start.timestamp() * 1e6), int(end.timestamp() * 1e6)
    cols: dict[str, list[Any]] = {k: [] for k in TRADE_SCHEMA}
    paths = []
    for ym in sorted({t.strftime("%Y%m") for t in hours(start, end)}):
        tpath = fetch(
            archive_ref("spot", "deals", pair, ym, base=base), raw_root, retries=retries, backoff_s=backoff_s
        )
        if tpath is not None:
            raw += tpath.stat().st_size
            paths.append(tpath)
    stream = (x for tp in paths for x in iter_trades_file(tp, "spot", start_us, end_us))
    seq = check_sequence(_collect(stream, cols))
    trades_df = pl.DataFrame(cols, schema=TRADE_SCHEMA)

    for sub in ("tob", "trades", "hours"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    _write_parquet(tob_df, out / "tob" / f"{pair}.parquet")
    _write_parquet(trades_df, out / "trades" / f"{pair}.parquet")
    fields = list(hour_rows[0]) if hour_rows else ["hour"]
    for h in hour_rows:
        fields += [k for k in h if k not in fields]
    tmp = out / "hours" / f"{pair}.csv.tmp"
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(hour_rows)
    os.replace(tmp, out / "hours" / f"{pair}.csv")
    return PairSummary(
        pair,
        len(hour_rows),
        sum(h["status"] == "missing" for h in hour_rows),
        sum(h["status"] == "issues" for h in hour_rows),
        sum(h.get("chained") == "false" for h in hour_rows),
        tob_df.height,
        sentinels,
        trades_df.height,
        seq.id_gaps,
        round(raw / 1e6, 1),
        round(time.monotonic() - t0, 1),
    )


def _build_pair_safe(*args: Any) -> PairSummary:
    """build_pair for worker processes: exceptions come back as picklable RuntimeErrors (HTTPError is not)."""
    try:
        return build_pair(*args)
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}") from None


def research_pairs(universe: dict[str, Any]) -> list[str]:
    return [p for s in sorted(universe["research"], key=int) for p in universe["research"][s]]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build top of book and trades Parquet files for research pairs.")
    p.add_argument("universe")
    p.add_argument("out")
    p.add_argument("--start", default="2026-08-01")
    p.add_argument("--end", default="2026-11-01", help="exclusive")
    p.add_argument("--pairs", default="", help="comma-separated subset of the research pairs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--raw-dir", default="data/raw/gate")
    p.add_argument("--base", default=BASE_URL)
    p.add_argument("--delete-raw", action="store_true", help="delete hourly order book files once processed")
    p.add_argument("--force", action="store_true", help="rebuild pairs that already have output")
    p.add_argument("--retries", type=int, default=5, help="download retries on transient errors")
    p.add_argument("--backoff", type=float, default=2.0, help="first retry delay, seconds (doubles)")
    p.add_argument("--open-time-holdout", action="store_true")
    p.add_argument(
        "--holdout-pairs", action="store_true", help="build the U-001 holdout pairs (needs --decision)"
    )
    p.add_argument(
        "--decision", help="H-003 test result, accepted or rejected, that unlocks the holdout pairs"
    )
    a = p.parse_args(argv)

    start, end = parse_hour(a.start), parse_hour(a.end)
    if end > TIME_HOLDOUT_START and not a.open_time_holdout:
        print(f"refused: period ends after {TIME_HOLDOUT_START:%Y-%m-%d}, the untouched time holdout")
        return 2
    uni = json.loads(Path(a.universe).read_text())
    if a.holdout_pairs:
        err = holdout_unlocked(a.decision)
        if err:
            print(f"refused: {err}")
            return 2
        allowed, kind = [x for s in sorted(uni["holdout"], key=int) for x in uni["holdout"][s]], "holdout"
    else:
        allowed, kind = research_pairs(uni), "research"
    pairs = [x.strip() for x in a.pairs.split(",") if x.strip()] or allowed
    outside = [x for x in pairs if x not in allowed]
    if outside:
        print(f"refused: not {kind} pairs of this universe: {outside}")
        return 2
    out = Path(a.out)
    todo = [
        x
        for x in pairs
        if a.force
        or not all(
            (out / d / f"{x}.{e}").exists()
            for d, e in (("tob", "parquet"), ("trades", "parquet"), ("hours", "csv"))
        )
    ]
    print(
        f"{len(pairs)} pairs, {len(pairs) - len(todo)} already built, "
        f"building {len(todo)} with {a.workers} workers"
    )
    results: list[PairSummary] = []
    failed: list[str] = []
    # spawn, not fork: polars' thread pool does not survive fork (worker processes can deadlock).
    # One pair per worker process, so memory is returned to the system after every pair.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = {
            ex.submit(
                _build_pair_safe,
                x,
                start,
                end,
                Path(a.raw_dir),
                out,
                a.base,
                a.delete_raw,
                a.retries,
                a.backoff,
            ): x
            for x in todo
        }
        for fut in as_completed(futs):
            pair = futs[fut]
            try:
                s = fut.result()
            except BrokenProcessPool as e:
                failed.append(pair)
                print(
                    f"FAILED {pair}: {e} (a worker was killed, often out of memory; rerun to resume, "
                    f"with fewer --workers)",
                    flush=True,
                )
                continue
            except Exception as e:  # report and continue with the other pairs
                failed.append(pair)
                print(f"FAILED {pair}: {type(e).__name__}: {e}", flush=True)
                continue
            results.append(s)
            print(
                f"{s.pair:16} hours {s.hours} missing {s.missing} issues {s.with_issues} "
                f"unchained {s.unchained} | "
                f"tob {s.tob_rows} sentinels {s.sentinels} | trades {s.trades} id gaps {s.trade_id_gaps} | "
                f"raw {s.raw_mb} MB, {s.seconds} s",
                flush=True,
            )
    if results:
        out.mkdir(parents=True, exist_ok=True)
        summary = out / "summary.csv"
        new = not summary.exists()
        with summary.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[*asdict(results[0]), "built_at"])
            if new:
                w.writeheader()
            stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerows({**asdict(r), "built_at": stamp} for r in sorted(results, key=lambda r: r.pair))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
