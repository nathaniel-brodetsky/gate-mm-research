"""Match archive trades to `take` rows of the hourly order book.

Each trade is matched to take quantity on the side it hit (taker buy -> ask, taker sell -> bid), at the
same price, within +-1 s, nearest batch first, consuming that quantity. Take rows of one level in one
100 ms batch are pooled. The offset histogram
(book batch timestamp minus trade timestamp, in 100 ms buckets) tells how batch timestamps relate to
event time; the unmatched share of take volume is cancellations.

Usage:
  python -m mmlab.orderbook.gate_reconcile --kind futures --trades TRADES.csv.gz BOOK_H12 BOOK_H13 ...
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from mmlab.data_ingestion.gate_orderbook_file import BookEvent, Kind, iter_orderbook_file
from mmlab.data_ingestion.gate_trades_file import Trade, iter_trades_file
from mmlab.orderbook.l2 import Side

WINDOW_US = 1_000_000
BUCKET_US = 100_000
EDGE_US = 1_000_000  # trades this close to the file's time range edges are skipped


@dataclass(slots=True)
class _TakeRow:
    ts_us: int
    left: Decimal


@dataclass(slots=True)
class ReconcileReport:
    trades: int = 0
    trade_qty: Decimal = Decimal(0)
    matched: int = 0
    matched_qty: Decimal = Decimal(0)
    take_rows: int = 0
    take_qty: Decimal = Decimal(0)
    offsets: Counter[int] = field(default_factory=Counter)  # bucket -> matched trades

    def merge(self, o: ReconcileReport) -> None:
        self.trades += o.trades
        self.trade_qty += o.trade_qty
        self.matched += o.matched
        self.matched_qty += o.matched_qty
        self.take_rows += o.take_rows
        self.take_qty += o.take_qty
        self.offsets.update(o.offsets)


def reconcile(book_events: Iterable[BookEvent], trades: Iterable[Trade]) -> ReconcileReport:
    """Trades must cover at least the book file's time range; only trades inside it are used."""
    rep = ReconcileReport()
    takes: dict[tuple[Side, Decimal], list[_TakeRow]] = defaultdict(list)
    t_min: int | None = None
    t_max = 0
    for ev in book_events:
        t_min = ev.ts_us if t_min is None else t_min
        t_max = max(t_max, ev.ts_us)
        if ev.action == "take":
            rows = takes[(ev.side, ev.price)]
            if rows and rows[-1].ts_us == ev.ts_us:  # same level, same batch: one pool of quantity
                rows[-1].left += ev.qty
            else:
                rows.append(_TakeRow(ev.ts_us, ev.qty))
            rep.take_rows += 1
            rep.take_qty += ev.qty
    if t_min is None:
        return rep
    lo, hi = t_min + EDGE_US, t_max - EDGE_US
    for tr in trades:
        if not lo <= tr.ts_us < hi:
            continue
        rep.trades += 1
        rep.trade_qty += tr.qty
        side: Side = "ask" if tr.taker == "buy" else "bid"
        best: _TakeRow | None = None
        best_off = 0
        for row in takes.get((side, tr.price), ()):
            off = row.ts_us - tr.ts_us
            if abs(off) <= WINDOW_US and row.left >= tr.qty and (best is None or abs(off) < abs(best_off)):
                best, best_off = row, off
        if best is not None:
            best.left -= tr.qty
            rep.matched += 1
            rep.matched_qty += tr.qty
            rep.offsets[best_off // BUCKET_US] += 1
    return rep


def describe(rep: ReconcileReport) -> str:
    def pct(a: Decimal | int, b: Decimal | int) -> str:
        return f"{100 * Decimal(a) / Decimal(b):.1f}%" if b else "n/a"

    hist = "  ".join(f"[{k * 100:+d},{(k + 1) * 100:+d})ms:{v}" for k, v in sorted(rep.offsets.items()))
    return (
        f"trades {rep.trades}, matched {rep.matched} ({pct(rep.matched, rep.trades)}), "
        f"matched qty {pct(rep.matched_qty, rep.trade_qty)}; take rows {rep.take_rows}, "
        f"take qty explained by trades {pct(rep.matched_qty, rep.take_qty)}\n"
        f"  offset = book batch ts - trade ts: {hist or 'none'}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Match Gate archive trades to order book take rows.")
    p.add_argument("--kind", required=True, choices=("spot", "futures"))
    p.add_argument("--trades", required=True, help="monthly deals/trades file covering the book hours")
    p.add_argument("books", nargs="+", help="hourly orderbooks files")
    a = p.parse_args(argv)
    kind: Kind = a.kind
    total = ReconcileReport()
    for path in a.books:
        events = list(iter_orderbook_file(path, kind))
        if not events:
            continue
        lo, hi = events[0].ts_us, max(e.ts_us for e in events) + 1
        rep = reconcile(events, iter_trades_file(a.trades, kind, lo, hi))
        print(f"{Path(path).name}: {describe(rep)}")
        total.merge(rep)
    print(f"TOTAL: {describe(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
