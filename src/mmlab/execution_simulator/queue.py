"""Fills of virtual resting orders at the top of the book, from Gate's archive (stage 3, step 6a).

The dataset has, per pair, the top of book after every 100 ms batch in which it changed (state time =
availability time) and every trade with its exact price and taker side. A virtual order never trades
on a price touch; it is filled only by archived trades, after the displayed queue ahead of it:

- Placement: at a decision time t the order copies the best ask (to sell) or best bid (to buy) of the
  last valid book with avail_us <= t. It reaches the exchange `latency` later and joins the queue at
  the end of the 100 ms batch it arrives in (activation); the book in force then is the last state
  with avail_us <= activation (a state is written only when the top changes). Everything displayed at
  its price in that state is ahead of it; trades before activation do not count. If that state is
  invalid, or its opposite side has reached the order's price (post-only would reject it), the order
  is not placed. If the level has disappeared in that state the order is alone at its price; if a
  better price has appeared, the queue ahead is the displayed size seen at the decision.
- Queue: a trade at the order's price first consumes the queue ahead, then fills the order; a trade at
  a worse price for the order's side (a taker sweeping through the level) fills what is left. Later
  valid states cap the queue ahead at the displayed size when the order's price is still the best
  (cancellations ahead are assumed only as far as the display forces), and empty it when the level is
  gone. The order's own presence changes nothing in the archive.
- The order lives `horizon` after activation; whatever is unfilled then is cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.microstructure.conditional import book_table

REL_EPS = 1e-12  # prices are equal if they differ by less than this, relative
BATCH_US = 100_000


@dataclass(frozen=True)
class Book:
    avail: NDArray[np.int64]
    bid: NDArray[np.float64]
    ask: NDArray[np.float64]
    bid_qty: NDArray[np.float64]
    ask_qty: NDArray[np.float64]
    ok: NDArray[np.bool_]

    @classmethod
    def from_tob(cls, tob: pl.DataFrame) -> Book:
        b = book_table(tob)
        return cls(
            b["avail_us"].to_numpy(),
            b["bid"].fill_null(np.nan).to_numpy(),
            b["ask"].fill_null(np.nan).to_numpy(),
            b["bid_qty"].fill_null(np.nan).to_numpy(),
            b["ask_qty"].fill_null(np.nan).to_numpy(),
            b["ok"].to_numpy(),
        )


@dataclass(frozen=True)
class Trades:
    ts: NDArray[np.int64]
    price: NDArray[np.float64]
    qty: NDArray[np.float64]
    taker: NDArray[np.int64]  # +1 buy (hits asks), -1 sell (hits bids)

    @classmethod
    def from_frame(cls, trades: pl.DataFrame) -> Trades:
        t = trades.sort("ts_us", maintain_order=True)
        return cls(
            t["ts_us"].to_numpy(),
            t["price"].to_numpy(),
            t["qty"].to_numpy(),
            t["taker"].cast(pl.Int64).to_numpy(),
        )


@dataclass(frozen=True)
class Order:
    side: int  # +1 sells at the ask (filled by taker buys), -1 buys at the bid
    price: float
    size: float  # base units
    decided_us: int
    act_us: int  # joins the queue: end of the batch it arrives in
    expire_us: int
    ahead: float  # queue ahead at activation, base units


@dataclass(frozen=True)
class Fill:
    order: int
    ts: int
    qty: float


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= REL_EPS * abs(b)


def place(
    book: Book, decide_us: NDArray[np.int64], side: int, size_quote: float, latency_us: int, horizon_us: int
) -> list[Order]:
    """Orders on one side at the given decision times (sorted); skipped where they cannot be placed."""
    known = np.searchsorted(book.avail, decide_us, side="right") - 1
    act_us = ((decide_us + latency_us) // BATCH_US + 1) * BATCH_US
    act = np.searchsorted(book.avail, act_us, side="right") - 1
    px = book.ask if side == 1 else book.bid
    qty = book.ask_qty if side == 1 else book.bid_qty
    opp = book.bid if side == 1 else book.ask
    out = []
    for t, ta, i, j in zip(decide_us.tolist(), act_us.tolist(), known.tolist(), act.tolist(), strict=True):
        if i < 0 or not book.ok[i] or not book.ok[j]:
            continue
        p = float(px[i])
        if (
            not np.isfinite(p)
            or not np.isfinite(opp[j])
            or (side == 1 and opp[j] >= p)
            or (side == -1 and opp[j] <= p)
        ):
            continue  # no price, or post-only would reject it
        best = float(px[j])
        if np.isfinite(best) and _same(best, p):
            ahead = float(qty[j])
        elif np.isfinite(best) and (best - p) * side < 0:
            ahead = float(qty[i])  # a better price stands in front; our level was last seen at the decision
        else:
            ahead = 0.0  # the level is gone: the order is alone at its price
        out.append(Order(side, p, size_quote / p, t, ta, ta + horizon_us, ahead))
    return out


def simulate(book: Book, trades: Trades, orders: list[Order]) -> list[Fill]:
    """One chronological pass over book states and trades; orders sorted by activation time.

    At equal times a book state comes first, then orders joining, then trades.
    """
    fills: list[Fill] = []
    active: list[list[float | int]] = []  # [order index, ahead, remaining]
    nxt = 0
    j = k = 0
    nb, nt = book.avail.size, trades.ts.size
    while j < nb or k < nt:
        is_book = k >= nt or (j < nb and book.avail[j] <= trades.ts[k])  # a state before trades at its time
        now = int(book.avail[j]) if is_book else int(trades.ts[k])
        while nxt < len(orders) and (orders[nxt].act_us < now or (not is_book and orders[nxt].act_us == now)):
            active.append([nxt, orders[nxt].ahead, orders[nxt].size])
            nxt += 1
        active = [a for a in active if orders[int(a[0])].expire_us > now]
        if is_book:
            if book.ok[j]:
                for a in active:
                    o = orders[int(a[0])]
                    best = book.ask[j] if o.side == 1 else book.bid[j]
                    if np.isfinite(best) and _same(float(best), o.price):
                        shown = book.ask_qty[j] if o.side == 1 else book.bid_qty[j]
                        a[1] = min(float(a[1]), float(shown))
                    elif not np.isfinite(best) or (best - o.price) * o.side > 0:
                        a[1] = 0.0  # the level is gone
            j += 1
            if not active and nxt >= len(orders):
                break
            continue
        p, x, s = float(trades.price[k]), float(trades.qty[k]), int(trades.taker[k])
        for a in active:
            o = orders[int(a[0])]
            if o.side != s:
                continue
            if _same(p, o.price):
                take = min(float(a[1]), x)
                a[1] = float(a[1]) - take
                got = min(x - take, float(a[2]))
            elif (p - o.price) * o.side > 0:
                got = float(a[2])  # swept through the order's level
            else:
                continue
            if got > 0:
                fills.append(Fill(int(a[0]), int(trades.ts[k]), got))
                a[2] = float(a[2]) - got
        active = [a for a in active if float(a[2]) > 0]
        k += 1
    return fills
