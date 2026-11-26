"""Exploration: round trips of a maker that steps inside the spread (stage 3, step 6d).

Entries are the orders of `fill_explore` (every `every` seconds on both sides, `improve` ticks inside
the spread, cancelled after `horizon`). Every piece of an entry fill is then closed on its own:
- passively first: an exit order on the other side, decided at the fill, `improve` ticks inside the
  spread (or at the best price with 0), for the piece's base size, kept for `exit_life`;
- whatever is still open then, or a piece whose exit order cannot be placed, is closed across the
  spread at the best opposite price known at that moment (taker).
Pieces are independent: no netting of inventory between them. Net P&L of a piece in USDT is
side * (entry price - closing price) * size minus the maker fee on the entry and on the passive part
of the exit and the taker fee on the aggressive part.

Per pair, for all entries and for entries decided at half-spreads of at least 80 bp: round trips, the
share of size closed passively, the median holding time, the net and gross P&L per unit of entry value
(volume-weighted over pieces) and the net P&L per day in USDT. Exploration only.

Usage:
    python -m mmlab.execution_simulator.round_trip data/processed/U001 \
        --out results/h004_explore/round_trips.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.evaluation.h003 import MAKER_FEE, MIN_DAYS, MIN_TRADES, THRESHOLD_BP
from mmlab.execution_simulator.fill_explore import TAKER_FEE, _asof, decision_times
from mmlab.execution_simulator.queue import Book, Fill, Order, Trades, place, simulate, tick_size
from mmlab.feature_engineering.asof_activity import epoch_s
from mmlab.microstructure.quote_pause import EXPLORATION_END

EVERY_S = 60
HORIZON_S = 60
EXIT_LIFE_S = 300
LATENCY_MS = 300
SIZE_USDT = 10.0
IMPROVE = 1


def close_price(book: Book, side: int, t_us: NDArray[np.int64]) -> NDArray[np.float64]:
    """Price that closes, as a taker at t, a position opened by an entry on `side` (NaN without a book).

    An entry that sold (+1) is closed by buying at the best ask; one that bought (-1), by selling at the
    best bid.
    """
    i = _asof(book, t_us)
    px = book.ask if side == 1 else book.bid
    return np.where(i >= 0, px[np.clip(i, 0, None)], np.nan)


def round_trips(
    book: Book,
    trades: Trades,
    entries: list[Order],
    fills: list[Fill],
    latency_us: int,
    exit_life_us: int,
    improve: int,
    tick: float,
    optimistic: bool = False,
) -> pl.DataFrame:
    """One row per entry fill piece with how and at what price it was closed, and its P&L in USDT.

    `net` uses the spot fees; `maker_value` and `taker_value` (the value traded as maker and as taker)
    give the net under any other fees: gross - maker fee * maker_value - taker fee * taker_value.
    """
    n = len(fills)
    ts = np.array([f.ts for f in fills], dtype=np.int64)
    qty = np.array([f.qty for f in fills])
    side = np.array([entries[f.order].side for f in fills], dtype=np.int64)
    pe = np.array([entries[f.order].price for f in fills])
    exits: list[Order] = []
    for s in (1, -1):
        k = np.flatnonzero(side == s)
        exits += place(book, ts[k], -s, 0.0, latency_us, exit_life_us, improve, tick, qty[k], k)
    exits.sort(key=lambda o: (o.act_us, o.decided_us, o.tag))
    passive = np.zeros(n)
    last = ts.copy()
    for f in simulate(book, trades, exits, optimistic):
        piece = exits[f.order].tag
        passive[piece] += f.qty
        last[piece] = f.ts
    has_exit = np.zeros(n, dtype=bool)
    expire = ts + latency_us  # no exit order: crossed as soon as the fill is known and sent
    px_exit = np.full(n, np.nan)
    for o in exits:
        has_exit[o.tag] = True
        expire[o.tag] = o.expire_us
        px_exit[o.tag] = o.price
    passive = np.minimum(passive, qty)
    aggr = qty - passive
    done = aggr <= 1e-12 * qty
    px_aggr = np.full(n, np.nan)
    for s in (1, -1):
        k = np.flatnonzero(side == s)
        px_aggr[k] = close_price(book, s, expire[k])
    px_pass = np.nan_to_num(px_exit)  # passive size is 0 where there was no exit order
    aggr_part = np.where(done, 0.0, (pe - px_aggr) * aggr)  # NaN if a crossing was needed but unpriced
    gross = side * ((pe - px_pass) * passive + aggr_part)
    maker_value = pe * qty + px_pass * passive  # traded as maker: the entry and the passive exit
    taker_value = np.where(done, 0.0, px_aggr * aggr)  # crossed at the end
    t_close = np.where(done, last, expire)
    return pl.DataFrame(
        {
            "entry": np.array([f.order for f in fills], dtype=np.int64),
            "side": side,
            "ts_us": ts,
            "value": pe * qty,
            "passive_share": passive / qty,
            "had_exit_order": has_exit,
            "hold_s": (t_close - ts) / 1e6,
            "gross": gross,
            "net": gross - MAKER_FEE * maker_value - TAKER_FEE * taker_value,
            "maker_value": maker_value,
            "taker_value": taker_value,
        }
    ).fill_nan(None)


def summarize_pair(trips: pl.DataFrame, wide_entry: NDArray[np.bool_], n_days: int) -> dict[str, Any]:
    """Per subset: round trips, passive share, holding time, net and gross bp, net USDT per day."""
    out: dict[str, Any] = {}
    w = trips.with_columns(pl.Series("wide", wide_entry[trips["entry"].to_numpy()], dtype=pl.Boolean))
    for subset, frame in (("all", w), ("q80", w.filter(pl.col("wide")))):
        ok = frame.drop_nulls(["net"])
        value = float(ok["value"].sum())
        net = float(ok["net"].sum())
        gross = float(ok["gross"].sum())
        days = (
            int(ok.select(pl.from_epoch("ts_us", time_unit="us").dt.date().n_unique()).item())
            if ok.height
            else 0
        )
        out[subset] = {
            "round_trips": int(ok.height),
            "unpriced": int(frame.height - ok.height),
            "days": days,
            "entry_value": value,
            "passive_share": float((ok["passive_share"] * ok["value"]).sum()) / value if value > 0 else None,
            "median_hold_s": float(np.median(ok["hold_s"].to_numpy())) if ok.height else None,
            "net_bp": net / value * 1e4 if value > 0 else None,
            "gross_bp": gross / value * 1e4 if value > 0 else None,
            "net_usdt_per_day": net / n_days,
        }
    return out


def explore_pair(
    ds: Path,
    pair: str,
    start: str,
    end: str,
    every_s: int,
    horizon_s: int,
    exit_life_s: int,
    latency_ms: int,
    size: float,
    improve: int,
    queue: str = "pessimistic",
) -> tuple[str, dict[str, Any] | None]:
    trades_df = pl.read_parquet(ds / "trades" / f"{pair}.parquet")
    book = Book.from_tob(pl.read_parquet(ds / "tob" / f"{pair}.parquet"))
    tr = Trades.from_frame(trades_df)
    if tr.ts.size < MIN_TRADES:
        return pair, None
    tick = tick_size(book) if improve else 0.0
    lat, hor = latency_ms * 1000, horizon_s * 1_000_000
    t = decision_times(start, end, every_s, horizon_s)
    entries = sorted(
        place(book, t, 1, size, lat, hor, improve, tick) + place(book, t, -1, size, lat, hor, improve, tick),
        key=lambda o: (o.act_us, o.decided_us, o.side),
    )
    optimistic = queue == "optimistic"
    fills = simulate(book, tr, entries, optimistic)
    trips = round_trips(book, tr, entries, fills, lat, exit_life_s * 1_000_000, improve, tick, optimistic)
    known = _asof(book, np.array([o.decided_us for o in entries], dtype=np.int64))
    k = np.clip(known, 0, None)
    q = (book.ask[k] - book.bid[k]) / (book.ask[k] + book.bid[k])
    wide = np.round(q * 1e4, 6) >= THRESHOLD_BP
    n_days = (epoch_s(end) - epoch_s(start)) // 86_400
    res = summarize_pair(trips, wide, n_days)
    res["tick"] = tick
    return pair, res


def explore(
    ds: Path,
    pairs: Sequence[str],
    start: str,
    end: str,
    every_s: int = EVERY_S,
    horizon_s: int = HORIZON_S,
    exit_life_s: int = EXIT_LIFE_S,
    latency_ms: int = LATENCY_MS,
    size: float = SIZE_USDT,
    improve: int = IMPROVE,
    queue: str = "pessimistic",
    workers: int = 1,
) -> dict[str, Any]:
    args = [
        (ds, p, start, end, every_s, horizon_s, exit_life_s, latency_ms, size, improve, queue) for p in pairs
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(explore_pair, *zip(*args, strict=True)))
    else:
        done = [explore_pair(*a) for a in args]
    return {
        "period": [start, end],
        "every_s": every_s,
        "horizon_s": horizon_s,
        "exit_life_s": exit_life_s,
        "latency_ms": latency_ms,
        "size_usdt": size,
        "improve_ticks": improve,
        "queue": queue,
        "pairs": {p: r for p, r in done if r is not None},
        "too_thin": [p for p, r in done if r is None],
    }


def _m(xs: Sequence[float], fmt: str) -> str:
    return format(statistics.median(xs), fmt) if xs else "n/a"


def compact(res: Mapping[str, Any]) -> str:
    """One line per run: settings; per subset the median net and gross bp, passive share, holding time,
    pairs with a positive net, and the net USDT per day summed over pairs."""
    head = (
        f"entry {res['horizon_s']:4d} s +{res['improve_ticks']} tick, exit {res['exit_life_s']:5d} s, "
        f"{res['size_usdt']:g} USDT, {res['queue']}"
    )
    parts = []
    for subset in ("all", "q80"):
        es = [e[subset] for e in res["pairs"].values()]
        counted = [e for e in es if e["round_trips"] >= MIN_TRADES and e["days"] >= MIN_DAYS]
        net = [e["net_bp"] for e in counted if e["net_bp"] is not None]
        gross = [e["gross_bp"] for e in counted if e["gross_bp"] is not None]
        parts.append(
            f"{subset}: pairs {len(counted)} net {_m(net, '+.1f')} gross {_m(gross, '+.1f')} bp, passive "
            f"{_m([e['passive_share'] for e in counted if e['passive_share'] is not None], '.2f')}, hold "
            f"{_m([e['median_hold_s'] for e in counted if e['median_hold_s'] is not None], '.0f')} s, "
            f"positive {sum(x > 0 for x in net)}/{len(net)}, "
            f"{sum(e['net_usdt_per_day'] for e in es):+.2f} USDT/day"
        )
    return head + " | " + " | ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Round trips of a maker stepping inside the spread.")
    p.add_argument("dataset")
    p.add_argument("--universe", default="configs/universe_U001.json")
    p.add_argument("--start", default="2026-08-01")
    p.add_argument("--end", default=EXPLORATION_END)
    p.add_argument("--every-s", type=int, default=EVERY_S)
    p.add_argument("--horizon-s", type=int, default=HORIZON_S, help="lifetime of entry orders")
    p.add_argument("--exit-life-s", type=int, default=EXIT_LIFE_S, help="lifetime of passive exit orders")
    p.add_argument("--latency-ms", type=int, default=LATENCY_MS)
    p.add_argument("--size-usdt", type=float, default=SIZE_USDT)
    p.add_argument("--improve-ticks", type=int, default=IMPROVE)
    p.add_argument("--queue", choices=["pessimistic", "optimistic"], default="pessimistic")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.end > EXPLORATION_END:
        print(f"refused: exploration stays before {EXPLORATION_END}")
        return 2
    uni = json.loads(Path(a.universe).read_text())
    pairs = [x for s in sorted(uni["research"], key=int) for x in uni["research"][s]]
    res = explore(
        Path(a.dataset),
        pairs,
        a.start,
        a.end,
        a.every_s,
        a.horizon_s,
        a.exit_life_s,
        a.latency_ms,
        a.size_usdt,
        a.improve_ticks,
        a.queue,
        a.workers,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(compact(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
