"""Synthetic Gate hourly `orderbooks` files with known ground truth (delta semantics)."""

from __future__ import annotations

import gzip
import random
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from mmlab.data_ingestion.gate_orderbook_file import Kind
from mmlab.orderbook.l2 import Side

TICK = Decimal("0.00000001")
T0 = 1794744000  # 2026-11-15 12:00:00 UTC


@dataclass
class SynthMarket:
    kind: Kind
    seed: int = 7
    next_id: int = 2_943_578_795
    trade_prob: float = 0.6  # share of take rows caused by trades; the rest are cancellations
    batch_ts: str = "end"  # "end": batch ts closes its 100 ms window; "start": opens it
    next_trade_id: int = 4_137_792
    trades: list[tuple[int, int, Decimal, Decimal, str]] = field(default_factory=list)
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        mid = 3550
        for i in range(1, 40):
            self.bids[(mid - i) * TICK] = Decimal(self.rng.randint(1, 10**6)) / 10
            self.asks[(mid + i) * TICK] = Decimal(self.rng.randint(1, 10**6)) / 10

    def _row(self, ts: str, side: Side, action: str, px: Decimal, qty: Decimal, uid: int, merged: int) -> str:
        if self.kind == "spot":
            q = f"{qty:.16f}" if action == "set" else str(qty)
            return f"{ts},{1 if side == 'ask' else 2},{action},{px:f},{q},{uid},{merged}"
        signed = qty if side == "bid" else -qty
        return f"{ts},{action},{px:f},{signed},{uid},{merged}"

    def hour(self, t0: int, steps: int = 3000, tail_steps: int = 3) -> list[str]:
        """Snapshot at t0 (id = last covered id), then `steps` updates, plus a tail past the hour end."""
        snap_id = self.next_id
        rows = [self._row(str(t0), "bid", "set", p, q, snap_id, 0) for p, q in sorted(self.bids.items())]
        rows += [self._row(str(t0), "ask", "set", p, q, snap_id, 0) for p, q in sorted(self.asks.items())]
        uid = snap_id + 1
        tenths = sorted(self.rng.sample(range(1, 36000), steps)) + [36000 + k for k in range(tail_steps)]
        for k in tenths:
            ts = f"{t0 + k // 10}.{k % 10}"
            ts_us = (t0 * 10 + k) * 100_000
            for _ in range(self.rng.choice((1, 1, 2))):  # 1-2 rows per 100 ms batch
                merged = self.rng.choice((1, 1, 1, 2, 4))
                rows.append(self._update(ts, ts_us, uid, merged))
                uid += merged
        self.next_id = uid - 1
        return rows

    def _trade(self, batch_us: int, px: Decimal, qty: Decimal, taker: str) -> None:
        lag = self.rng.randint(1, 99_999)
        self.trades.append(
            (
                batch_us - lag if self.batch_ts == "end" else batch_us + lag - 1,
                self.next_trade_id,
                px,
                qty,
                taker,
            )
        )
        self.next_trade_id += 1

    def trade_rows(self) -> list[str]:
        out = []
        for ts_us, tid, px, qty, taker in sorted(self.trades):
            ts = f"{ts_us // 1_000_000}.{ts_us % 1_000_000:06d}"
            if self.kind == "spot":
                out.append(f"{ts},{tid},{px:f},{qty},{1 if taker == 'buy' else 2}")  # 1 = taker buy
            else:
                out.append(f"{ts},{tid},{px:f},{qty if taker == 'buy' else -qty}")
        return out

    def _update(self, ts: str, ts_us: int, uid: int, merged: int) -> str:
        side: Side = self.rng.choice(("bid", "ask"))
        book = self.bids if side == "bid" else self.asks
        other = self.asks if side == "bid" else self.bids
        if book and self.rng.random() < 0.5:  # take: trade or cancel, never below zero
            px = self.rng.choice(sorted(book)[-5:] if side == "bid" else sorted(book)[:5])
            qty = min(book[px], Decimal(self.rng.randint(1, 10**6)) / 10)
            book[px] -= qty
            if book[px] == 0:
                del book[px]
            if self.rng.random() < self.trade_prob:
                taker = "buy" if side == "ask" else "sell"
                if qty > 1 and self.rng.random() < 0.3:  # one take row, two prints
                    self._trade(ts_us, px, qty / 2, taker)
                    self._trade(ts_us, px, qty - qty / 2, taker)
                else:
                    self._trade(ts_us, px, qty, taker)
            return self._row(ts, side, "take", px, qty, uid, merged)
        best_other = (min(other) if side == "bid" else max(other)) if other else None
        offset = self.rng.randint(1, 6) * TICK
        px = (best_other - offset if side == "bid" else best_other + offset) if best_other else 3550 * TICK
        qty = Decimal(self.rng.randint(1, 10**6)) / 10
        book[px] = book.get(px, Decimal(0)) + qty
        return self._row(ts, side, "make", px, qty, uid, merged)


def write_gz(path: Path, rows: list[str]) -> Path:
    with gzip.open(path, "wt", newline="") as fh:
        fh.write("\n".join(rows) + "\n")
    return path
