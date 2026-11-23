"""Replay and integrity checks for Gate hourly `orderbooks` archive files.

Usage:
  python -m mmlab.orderbook.gate_replay --kind spot FILE_H00 FILE_H01 ...
Files are processed in the given order; consecutive files are checked as a chain:
the next snapshot id must equal the last covered update id, and the replayed book must
equal the next snapshot level by level.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain

from mmlab.data_ingestion.gate_orderbook_file import BookEvent, Kind, iter_orderbook_file
from mmlab.orderbook.l2 import BookError, L2Book

HOUR_US = 3_600_000_000


class FormatError(ValueError):
    """The file does not have the expected structure."""


@dataclass(slots=True)
class ReplayReport:
    snapshot_rows: int = 0
    snapshot_id: int = -1
    snapshot_ts_us: int = -1
    updates: int = 0
    batches: int = 0
    id_breaks: int = 0
    first_break: tuple[int, int] | None = None  # (got, expected)
    non_monotonic_ts: int = 0
    crossed_batches: int = 0
    negative_levels: int = 0
    late_set_rows: int = 0
    last_covered_id: int = -1
    last_ts_us: int = -1
    rows_after_hour_end: int = 0

    @property
    def clean(self) -> bool:
        return not (
            self.id_breaks
            or self.non_monotonic_ts
            or self.crossed_batches
            or self.negative_levels
            or self.late_set_rows
        )


@dataclass(frozen=True, slots=True)
class ChainReport:
    id_match: bool
    level_mismatches: int

    @property
    def ok(self) -> bool:
        return self.id_match and self.level_mismatches == 0


def replay(
    events: Iterable[BookEvent],
    on_snapshot: Callable[[L2Book], None] | None = None,
    on_batch: Callable[[int, L2Book, ReplayReport], None] | None = None,
) -> tuple[L2Book, ReplayReport]:
    """Build the snapshot, then apply delta updates in 100 ms batches (rows sharing a timestamp).

    `on_batch(ts, book, report)` runs after each batch; the report holds the counters so far.
    """
    book, rep = L2Book(), ReplayReport()
    it = iter(events)
    first_update: BookEvent | None = None
    for ev in it:
        if ev.action != "set":
            first_update = ev
            break
        if rep.snapshot_rows == 0:
            rep.snapshot_id, rep.snapshot_ts_us = ev.update_id, ev.ts_us
        elif (ev.update_id, ev.ts_us) != (rep.snapshot_id, rep.snapshot_ts_us):
            raise FormatError("snapshot rows disagree on id or timestamp")
        book.set_level(ev.side, ev.price, ev.qty)
        rep.snapshot_rows += 1
    if rep.snapshot_rows == 0:
        raise FormatError("file does not start with a snapshot")
    if on_snapshot is not None:
        on_snapshot(book)

    rep.last_covered_id, rep.last_ts_us = rep.snapshot_id, rep.snapshot_ts_us
    hour_end = rep.snapshot_ts_us + HOUR_US
    expected = rep.snapshot_id + 1
    batch_ts: int | None = None

    def close_batch(ts: int) -> None:
        rep.batches += 1
        if book.is_crossed():
            rep.crossed_batches += 1
        if on_batch is not None:
            on_batch(ts, book, rep)

    updates = chain([first_update], it) if first_update is not None else iter(())
    for ev in updates:
        if batch_ts is not None and ev.ts_us != batch_ts:
            close_batch(batch_ts)
        batch_ts = ev.ts_us
        rep.updates += 1
        if ev.ts_us < rep.last_ts_us:
            rep.non_monotonic_ts += 1
        if ev.ts_us >= hour_end:
            rep.rows_after_hour_end += 1
        if ev.update_id != expected or ev.merged < 1:
            rep.id_breaks += 1
            if rep.first_break is None:
                rep.first_break = (ev.update_id, expected)
        expected = ev.update_id + max(ev.merged, 1)
        rep.last_covered_id = expected - 1
        rep.last_ts_us = max(rep.last_ts_us, ev.ts_us)
        try:
            if ev.action == "set":
                rep.late_set_rows += 1
                book.set_level(ev.side, ev.price, ev.qty)
            elif ev.action == "make":
                book.apply_delta(ev.side, ev.price, ev.qty)
            else:
                book.apply_delta(ev.side, ev.price, -ev.qty)
        except BookError:
            rep.negative_levels += 1
    if batch_ts is not None:
        close_batch(batch_ts)
    return book, rep


def check_chain(
    prev_book: L2Book, prev: ReplayReport, next_snapshot: L2Book, nxt: ReplayReport
) -> ChainReport:
    return ChainReport(nxt.snapshot_id == prev.last_covered_id, prev_book.mismatches(next_snapshot))


def _fmt_ts(us: int) -> str:
    return datetime.fromtimestamp(us / 1e6, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay Gate hourly orderbooks files and check integrity.")
    p.add_argument("--kind", required=True, choices=("spot", "futures"))
    p.add_argument("files", nargs="+")
    a = p.parse_args(argv)
    kind: Kind = a.kind

    ok = True
    prev: tuple[L2Book, ReplayReport] | None = None
    snaps: list[L2Book] = []
    for path in a.files:
        snaps.clear()
        book, rep = replay(iter_orderbook_file(path, kind), on_snapshot=lambda b: snaps.append(b.copy()))
        ok &= rep.clean
        bb, ba = book.best_bid(), book.best_ask()
        print(
            f"{path}\n"
            f"  snapshot {rep.snapshot_rows} levels @ {_fmt_ts(rep.snapshot_ts_us)} id {rep.snapshot_id}; "
            f"updates {rep.updates} in {rep.batches} batches; last {_fmt_ts(rep.last_ts_us)} "
            f"(rows after hour end {rep.rows_after_hour_end}); last covered id {rep.last_covered_id}\n"
            f"  id breaks {rep.id_breaks} first {rep.first_break}; non-monotonic ts {rep.non_monotonic_ts}; "
            f"crossed batches {rep.crossed_batches}; negative levels {rep.negative_levels}; "
            f"late set rows {rep.late_set_rows}\n"
            f"  end: bid/ask levels {book.depth('bid')}/{book.depth('ask')}, best {bb} / {ba}; "
            f"{'CLEAN' if rep.clean else 'ISSUES'}"
        )
        if prev is not None:
            ch = check_chain(prev[0], prev[1], snaps[0], rep)
            ok &= ch.ok
            print(
                f"  chain from previous file: snapshot id == last covered id: {ch.id_match}; "
                f"level mismatches {ch.level_mismatches}; {'OK' if ch.ok else 'BROKEN'}"
            )
        prev = (book, rep)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
