"""Top of book per 100 ms batch from a Gate hourly `orderbooks` file, with availability time.

A batch labelled t holds changes made in [t, t + 100 ms), so its state may be used only from t + 100 ms
(stage 1 audit). A row is emitted only when the top of book or the validity flag changes. Rows are
marked invalid from the first anomaly in the file onwards (id break, negative level, crossed book,
time going backwards, snapshot row among updates), because the book cannot be trusted until the next
hourly snapshot. The snapshot itself is not emitted: its exact cut-off time is not known.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from mmlab.data_ingestion.gate_orderbook_file import BookEvent
from mmlab.orderbook.gate_replay import ReplayReport, replay
from mmlab.orderbook.l2 import L2Book

BATCH_US = 100_000


@dataclass(frozen=True, slots=True)
class TopOfBook:
    label_us: int
    avail_us: int  # label + 100 ms: earliest time this state is known
    bid: Decimal | None
    bid_qty: Decimal | None
    ask: Decimal | None
    ask_qty: Decimal | None
    valid: bool


def _anomalies(rep: ReplayReport) -> int:
    return (
        rep.id_breaks + rep.negative_levels + rep.crossed_batches + rep.non_monotonic_ts + rep.late_set_rows
    )


def top_of_book(events: Iterable[BookEvent]) -> tuple[list[TopOfBook], ReplayReport]:
    rows: list[TopOfBook] = []
    last: tuple[object, ...] | None = None

    def on_batch(ts: int, book: L2Book, rep: ReplayReport) -> None:
        nonlocal last
        bb, ba = book.best_bid(), book.best_ask()
        state = (
            bb,
            book.qty("bid", bb) if bb is not None else None,
            ba,
            book.qty("ask", ba) if ba is not None else None,
            _anomalies(rep) == 0,
        )
        if state != last:
            rows.append(TopOfBook(ts, ts + BATCH_US, *state))
            last = state

    _, rep = replay(events, on_batch=on_batch)
    return rows, rep
