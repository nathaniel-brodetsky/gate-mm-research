"""Exploration: what a maker who joins the back of the queue actually gets (stage 3, step 6a).

Every `every` seconds of the exploration period, a virtual order of `size` USDT is placed at the best
bid and one at the best ask (`mmlab.execution_simulator.queue`: queue position from the archive, fills
only from archived trades, never on a price touch, `latency` to reach the exchange, cancelled after
`horizon`). Each order stands alone: no inventory, no interaction between the virtual orders.

Per pair, for all orders and for orders placed at a quoted half-spread of at least 80 bp (H-003's
condition, at the decision): how many orders get any fill, the share of the ordered size filled, the
median wait for the first fill, and for the fills (volume-weighted by fill value):
- theta_fill: RS(60 s) against the mid known at the fill and 60 s later, minus the maker fee;
- theta_exit: the same against the price that closes the position 60 s later by crossing the spread
  (best ask after a sell, best bid after a buy), minus the maker and the taker fee;
- theta_trades: the average trade's theta in the same subset (H-001 / H-003 measure), for comparison:
  theta_fill - theta_trades is the extra adverse selection of fills at the back of the queue.

Exploration only.

Usage:
    python -m mmlab.execution_simulator.fill_explore data/processed/U001 --out results/h004_explore/fills.json
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

from mmlab.evaluation.h003 import MAKER_FEE, MIN_DAYS, MIN_TRADES, THRESHOLD_BP, eligible, states
from mmlab.execution_simulator.queue import Book, Fill, Order, Trades, place, simulate
from mmlab.feature_engineering.asof_activity import epoch_s
from mmlab.microstructure.markouts import in_period
from mmlab.microstructure.quote_pause import EXPLORATION_END

TAKER_FEE = 0.0010  # global spot VIP 0 taker, as the maker fee (Gate fee page, read 2026-11-24)
EVERY_S = 60
HORIZON_S = 300
LATENCY_MS = 300
SIZE_USDT = 10.0
MARKOUT_S = 60


def decision_times(start: str, end: str, every_s: int, horizon_s: int) -> NDArray[np.int64]:
    """Decision times every `every_s` from start, leaving each order its full horizon before end."""
    g0, g1 = epoch_s(start), epoch_s(end)
    return np.arange(g0, g1 - horizon_s, every_s, dtype=np.int64) * 1_000_000


def _asof(book: Book, t: NDArray[np.int64]) -> NDArray[np.int64]:
    """Index of the last book state with avail <= t, -1 if none or not valid."""
    i = np.searchsorted(book.avail, t, side="right") - 1
    good = (i >= 0) & book.ok[np.clip(i, 0, None)]
    return np.where(good, i, -1)


def fill_markouts(book: Book, orders: list[Order], fills: list[Fill]) -> pl.DataFrame:
    """One row per fill: order, side, time, value, wide (decision half-spread >= 80 bp), rs, rs_exit."""
    if not fills:
        return pl.DataFrame(
            schema={
                "order": pl.Int64,
                "side": pl.Int64,
                "ts_us": pl.Int64,
                "value": pl.Float64,
                "rs": pl.Float64,
                "rs_exit": pl.Float64,
            }
        )
    oid = np.array([f.order for f in fills], dtype=np.int64)
    ts = np.array([f.ts for f in fills], dtype=np.int64)
    qty = np.array([f.qty for f in fills])
    side = np.array([orders[i].side for i in oid], dtype=np.int64)
    price = np.array([orders[i].price for i in oid])
    i0, i1 = _asof(book, ts), _asof(book, ts + MARKOUT_S * 1_000_000)
    mid = (book.bid + book.ask) / 2
    m0 = np.where(i0 >= 0, mid[np.clip(i0, 0, None)], np.nan)
    m1 = np.where(i1 >= 0, mid[np.clip(i1, 0, None)], np.nan)
    exit_px = np.where(side == 1, book.ask[np.clip(i1, 0, None)], book.bid[np.clip(i1, 0, None)])
    exit_px = np.where(i1 >= 0, exit_px, np.nan)
    return pl.DataFrame(
        {
            "order": oid,
            "side": side,
            "ts_us": ts,
            "value": qty * price,
            "rs": side * (price - m1) / m0,
            "rs_exit": side * (price - exit_px) / m0,
        }
    ).fill_nan(None)


def _vw(value: NDArray[np.float64], x: NDArray[np.float64]) -> float | None:
    s = float(value.sum())
    return float((value * x).sum() / s) if s > 0 else None


def summarize_orders(
    orders: list[Order], wide: NDArray[np.bool_], fm: pl.DataFrame, trades_theta: Mapping[str, float | None]
) -> dict[str, Any]:
    """Per subset ("all", "q80"): orders, fill rate and size, wait, theta of fills and of trades."""
    out: dict[str, Any] = {}
    first = fm.group_by("order").agg(pl.col("ts_us").min().alias("t1"), pl.col("value").sum().alias("val"))
    t1 = dict(zip(first["order"].to_list(), first["t1"].to_list(), strict=True))
    val = dict(zip(first["order"].to_list(), first["val"].to_list(), strict=True))
    for subset, sel in (("all", np.ones(len(orders), dtype=bool)), ("q80", wide)):
        ids = np.flatnonzero(sel)
        filled = [i for i in ids.tolist() if i in t1]
        waits = [(t1[i] - orders[i].decided_us) / 1e6 for i in filled]
        ordered = sum(orders[i].size * orders[i].price for i in ids.tolist())
        f = fm.filter(pl.col("order").is_in(ids.tolist()))
        ok = f.drop_nulls(["rs"])
        okx = f.drop_nulls(["rs_exit"])
        days = (
            int(ok.select(pl.from_epoch("ts_us", time_unit="us").dt.date().n_unique()).item())
            if ok.height
            else 0
        )
        rs = _vw(ok["value"].to_numpy(), ok["rs"].to_numpy())
        rx = _vw(okx["value"].to_numpy(), okx["rs_exit"].to_numpy())
        out[subset] = {
            "orders": int(ids.size),
            "orders_filled": len(filled),
            "filled_share_of_size": float(sum(val[i] for i in filled) / ordered) if ordered > 0 else None,
            "median_wait_s": float(statistics.median(waits)) if waits else None,
            "fills": int(ok.height),
            "fill_days": days,
            "theta_fill": None if rs is None else rs - MAKER_FEE,
            "theta_exit": None if rx is None else rx - MAKER_FEE - TAKER_FEE,
            "theta_trades": trades_theta[subset],
        }
    return out


def explore_pair(
    ds: Path, pair: str, start: str, end: str, every_s: int, horizon_s: int, latency_ms: int, size: float
) -> tuple[str, dict[str, Any] | None]:
    trades = pl.read_parquet(ds / "trades" / f"{pair}.parquet")
    tob = pl.read_parquet(ds / "tob" / f"{pair}.parquet")
    st = in_period(states(trades, tob), start, end).filter(pl.col("rs_60s").is_not_null())
    if st.height < MIN_TRADES or st["day"].n_unique() < MIN_DAYS:
        return pair, None
    wide_trades = eligible(st, THRESHOLD_BP)
    theta_trades = {
        "all": _vw(st["v"].to_numpy(), st["rs_60s"].to_numpy() - MAKER_FEE),
        "q80": _vw(wide_trades["v"].to_numpy(), wide_trades["rs_60s"].to_numpy() - MAKER_FEE),
    }
    book = Book.from_tob(tob)
    tr = Trades.from_frame(trades)
    t = decision_times(start, end, every_s, horizon_s)
    orders = sorted(
        place(book, t, 1, size, latency_ms * 1000, horizon_s * 1_000_000)
        + place(book, t, -1, size, latency_ms * 1000, horizon_s * 1_000_000),
        key=lambda o: (o.act_us, o.decided_us, o.side),
    )
    fills = simulate(book, tr, orders)
    known = _asof(book, np.array([o.decided_us for o in orders], dtype=np.int64))
    k = np.clip(known, 0, None)
    q = (book.ask[k] - book.bid[k]) / (book.ask[k] + book.bid[k])
    wide = np.round(q * 1e4, 6) >= THRESHOLD_BP
    res = summarize_orders(orders, wide, fill_markouts(book, orders, fills), theta_trades)
    return pair, res


def explore(
    ds: Path,
    pairs: Sequence[str],
    start: str,
    end: str,
    every_s: int = EVERY_S,
    horizon_s: int = HORIZON_S,
    latency_ms: int = LATENCY_MS,
    size: float = SIZE_USDT,
    workers: int = 1,
) -> dict[str, Any]:
    args = [(ds, p, start, end, every_s, horizon_s, latency_ms, size) for p in pairs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(explore_pair, *zip(*args, strict=True)))
    else:
        done = [explore_pair(*a) for a in args]
    return {
        "period": [start, end],
        "every_s": every_s,
        "horizon_s": horizon_s,
        "latency_ms": latency_ms,
        "size_usdt": size,
        "pairs": {p: r for p, r in done if r is not None},
        "too_thin": [p for p, r in done if r is None],
    }


def _med(xs: Sequence[float]) -> str:
    return f"{statistics.median(xs) * 1e4:+8.2f}" if xs else "     n/a"


def summarize(res: Mapping[str, Any]) -> str:
    pairs = res["pairs"]
    lines = [
        f"virtual orders at the back of the queue, {res['period'][0]} .. {res['period'][1]}: every "
        f"{res['every_s']} s on both sides, {res['size_usdt']:g} USDT, latency {res['latency_ms']} ms, "
        f"cancelled after {res['horizon_s']} s; {len(pairs)} pairs; exploration, no inference",
        f"theta in bp, median over pairs with >= {MIN_TRADES} fills on >= {MIN_DAYS} days; "
        "fill rate and wait: median over all pairs",
    ]
    for subset in ("all", "q80"):
        es = [e[subset] for e in pairs.values() if e[subset]["orders"] > 0]
        counted = [e for e in es if e["fills"] >= MIN_TRADES and e["fill_days"] >= MIN_DAYS]
        rate = [e["orders_filled"] / e["orders"] for e in es]
        waits = [e["median_wait_s"] for e in es if e["median_wait_s"] is not None]
        fill = [e["theta_fill"] for e in counted if e["theta_fill"] is not None]
        ex = [e["theta_exit"] for e in counted if e["theta_exit"] is not None]
        trd = [e["theta_trades"] for e in counted if e["theta_trades"] is not None]
        gap = [
            e["theta_fill"] - e["theta_trades"]
            for e in counted
            if e["theta_fill"] is not None and e["theta_trades"] is not None
        ]
        lines.append(
            f"{subset}: pairs {len(es)} (counted {len(counted)}); orders filled "
            f"{statistics.median(rate) if rate else float('nan'):.3f}; median wait "
            f"{statistics.median(waits) if waits else float('nan'):.0f} s"
        )
        lines.append(
            f"  theta fills {_med(fill)}  exit across the spread {_med(ex)}  trades {_med(trd)}  "
            f"fills - trades {_med(gap)}"
        )
    if res.get("too_thin"):
        lines.append(f"too thin: {', '.join(res['too_thin'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fills of virtual orders at the back of the queue.")
    p.add_argument("dataset")
    p.add_argument("--universe", default="configs/universe_U001.json")
    p.add_argument("--start", default="2026-08-01")
    p.add_argument("--end", default=EXPLORATION_END)
    p.add_argument("--every-s", type=int, default=EVERY_S)
    p.add_argument("--horizon-s", type=int, default=HORIZON_S)
    p.add_argument("--latency-ms", type=int, default=LATENCY_MS)
    p.add_argument("--size-usdt", type=float, default=SIZE_USDT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.end > EXPLORATION_END:
        print(f"refused: exploration stays before {EXPLORATION_END}")
        return 2
    uni = json.loads(Path(a.universe).read_text())
    pairs = [x for s in sorted(uni["research"], key=int) for x in uni["research"][s]]
    res = explore(
        Path(a.dataset), pairs, a.start, a.end, a.every_s, a.horizon_s, a.latency_ms, a.size_usdt, a.workers
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(summarize(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
