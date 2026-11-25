"""Exploration: realized spread by the as-of state of the market before the trade (stage 3, step 4).

For every trade of the exploration period and every research pair, features known 1 s before the
trade (`mmlab.feature_engineering.trade_state`, `asof_activity`): the signed top-of-book imbalance,
the signed taker flow over the last 60 and 300 s, the time since the last trade, and the mid's range
over the last 300 s relative to its 7-day mean. Signed features are positive when the state pointed
the taker's way. Trades are cut into fixed buckets of each feature; per pair and bucket, theta is the
volume-weighted RS(60 s) minus the maker fee, for all trades and for trades at quoted half-spreads of
at least 80 bp (the subset of H-003). Per bucket the summary gives the median over pairs that have at
least 30 trades on 5 days in it.

Exploration only: it picks candidates for the H-004 batch, which D-002 then requires to be calibrated
in the lab and tested on December.

Usage:
    python -m mmlab.microstructure.state_explore data/processed/U001 --out results/h004_explore/state.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.evaluation.h003 import MAKER_FEE, MIN_DAYS, MIN_TRADES, THRESHOLD_BP, eligible, states
from mmlab.feature_engineering.asof_activity import (
    DAY_S,
    daily_baseline,
    epoch_s,
    grid_index,
    mid_before,
    trailing_range,
)
from mmlab.feature_engineering.trade_state import signed_flow, signed_imbalance, since_last_trade, taker_sides
from mmlab.microstructure.markouts import in_period
from mmlab.microstructure.quote_pause import EXPLORATION_END

SIGNED = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)
EDGES: dict[str, tuple[float, ...]] = {
    "imbalance": SIGNED,
    "flow_60s": SIGNED,
    "flow_300s": SIGNED,
    "since_last_trade_s": (0.0, 1.0, 10.0, 60.0, 600.0, math.inf),
    "range_ratio_300s": (0.0, 0.5, 1.0, 2.0, 5.0, math.inf),
}
NONE = "none"
LATENCY_MS = 1000
RANGE_FLOOR = 1e-4


def labels(edges: Sequence[float]) -> list[str]:
    out = [f"[{a:g},{b:g})" for a, b in itertools.pairwise(edges)]
    if math.isfinite(edges[-1]):
        out[-1] = out[-1][:-1] + "]"  # a finite top edge belongs to the last bucket
    return [*out, NONE]


def bucket_index(values: NDArray[np.float64], edges: Sequence[float]) -> NDArray[np.int64]:
    """Bucket of each value (rounded to 9 decimals); len(edges) - 1 means none (NaN or outside)."""
    e = np.asarray(edges)
    v = np.round(values, 9)
    idx = np.searchsorted(e, v, side="right") - 1
    if math.isfinite(e[-1]):
        idx = np.where(v == e[-1], e.size - 2, idx)
    ok = np.isfinite(v) & (idx >= 0) & (idx <= e.size - 2)
    return np.asarray(np.where(ok, idx, e.size - 1), dtype=np.int64)


def range_ratio(
    tob: pl.DataFrame, ts_us: NDArray[np.int64], start: str, end: str, latency_us: int
) -> NDArray[np.float64]:
    """Mid range over the last 300 s known `latency` before each trade, over its 7-day mean (floor 1 bp)."""
    g0 = epoch_s(start)
    n = (epoch_s(end) - g0) // DAY_S * DAY_S
    grid = g0 + np.arange(n, dtype=np.int64)
    r = trailing_range(mid_before(tob, grid), 300)
    base = daily_baseline(r)
    gi = grid_index(ts_us, g0, latency_us)
    at = np.clip(gi, 0, n - 1)
    with np.errstate(invalid="ignore"):
        out = r[at] / np.maximum(base[at], RANGE_FLOOR)
    return np.asarray(np.where((gi >= 0) & (gi < n), out, np.nan), dtype=np.float64)


def features(
    trades: pl.DataFrame, tob: pl.DataFrame, st: pl.DataFrame, start: str, end: str, latency_us: int
) -> dict[str, NDArray[np.float64]]:
    """Features of the rows of st (states with a column d), each known `latency` before the trade."""
    ts = st["ts_us"].to_numpy()
    t = ts - latency_us
    d = st["d"].to_numpy()
    tr = trades.sort("ts_us", maintain_order=True)
    all_ts = tr["ts_us"].to_numpy()
    all_side = tr["taker"].cast(pl.Float64).to_numpy()
    all_v = (tr["price"] * tr["qty"]).to_numpy()
    return {
        "imbalance": signed_imbalance(tob, t, d),
        "flow_60s": signed_flow(all_ts, all_side, all_v, t, d, 60),
        "flow_300s": signed_flow(all_ts, all_side, all_v, t, d, 300),
        "since_last_trade_s": since_last_trade(all_ts, t),
        "range_ratio_300s": range_ratio(tob, ts, start, end, latency_us),
    }


def bucket_table(
    v: NDArray[np.float64],
    x: NDArray[np.float64],
    day: NDArray[np.int64],
    feats: Mapping[str, NDArray[np.float64]],
    edges: Mapping[str, Sequence[float]] = EDGES,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Per feature and bucket: trades, active days, volume and theta (None without volume)."""
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for name, values in feats.items():
        idx = bucket_index(values, edges[name])
        out[name] = {}
        for k, lab in enumerate(labels(edges[name])):
            m = idx == k
            vol = float(v[m].sum())
            out[name][lab] = {
                "trades": int(m.sum()),
                "active_days": int(np.unique(day[m]).size),
                "volume": vol,
                "theta": float((v[m] * x[m]).sum() / vol) if vol > 0 else None,
            }
    return out


def explore_pair(
    ds: Path, pair: str, start: str, end: str, latency_us: int
) -> tuple[str, dict[str, Any] | None]:
    trades = pl.read_parquet(ds / "trades" / f"{pair}.parquet")
    tob = pl.read_parquet(ds / "tob" / f"{pair}.parquet")
    full = states(trades, tob)
    full = full.with_columns(pl.Series("d", taker_sides(trades, tob, full)))
    st = in_period(full, start, end).filter(pl.col("rs_60s").is_not_null()).with_row_index("_r")
    if st.height < MIN_TRADES or st["day"].n_unique() < MIN_DAYS:
        return pair, None
    d0 = date.fromisoformat(start)
    day = np.array([(x - d0).days for x in st["day"].to_list()], dtype=np.int64)
    v = st["v"].to_numpy()
    x = st["rs_60s"].to_numpy() - MAKER_FEE
    feats = features(trades, tob, st, start, end, latency_us)
    wide = np.zeros(st.height, dtype=bool)
    wide[eligible(st, THRESHOLD_BP)["_r"].to_numpy()] = True
    out: dict[str, Any] = {}
    for subset, m in (("all", np.ones(st.height, dtype=bool)), ("q80", wide)):
        out[subset] = {
            "trades": int(m.sum()),
            "volume": float(v[m].sum()),
            "features": bucket_table(v[m], x[m], day[m], {k: f[m] for k, f in feats.items()}),
        }
    return pair, out


def explore(
    ds: Path, pairs: Sequence[str], start: str, end: str, latency_ms: int = LATENCY_MS, workers: int = 1
) -> dict[str, Any]:
    args = [(ds, p, start, end, latency_ms * 1000) for p in pairs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(explore_pair, *zip(*args, strict=True)))
    else:
        done = [explore_pair(*a) for a in args]
    return {
        "period": [start, end],
        "latency_ms": latency_ms,
        "edges": {k: [str(e) for e in v] for k, v in EDGES.items()},
        "pairs": {p: r for p, r in done if r is not None},
        "too_thin": [p for p, r in done if r is None],
    }


def summarize(res: Mapping[str, Any]) -> str:
    pairs = res["pairs"]
    lines = [
        f"realized spread by the state {res['latency_ms']} ms before the trade, {res['period'][0]} .. "
        f"{res['period'][1]}; {len(pairs)} pairs; median over pairs with >= {MIN_TRADES} trades on "
        f">= {MIN_DAYS} days in the bucket; exploration, no inference"
    ]
    for subset in ("all", "q80"):
        lines.append(f"{subset}:")
        for name, edges in EDGES.items():
            lines.append(f"  {name}")
            for lab in labels(edges):
                cells = [e[subset]["features"][name][lab] for e in pairs.values()]
                shares = [
                    c["volume"] / e[subset]["volume"]
                    for c, e in zip(cells, pairs.values(), strict=True)
                    if e[subset]["volume"] > 0
                ]
                th = [
                    c["theta"]
                    for c in cells
                    if c["theta"] is not None and c["trades"] >= MIN_TRADES and c["active_days"] >= MIN_DAYS
                ]
                med = f"{statistics.median(th) * 1e4:+8.2f}" if th else "     n/a"
                share = f"{statistics.median(shares):.3f}" if shares else "n/a"
                lines.append(f"    {lab:12} pairs {len(th):2d}  volume share {share:>5}  theta bp {med}")
    if res.get("too_thin"):
        lines.append(f"too thin: {', '.join(res['too_thin'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Realized spread by the as-of state before the trade.")
    p.add_argument("dataset")
    p.add_argument("--universe", default="configs/universe_U001.json")
    p.add_argument("--start", default="2026-08-01")
    p.add_argument("--end", default=EXPLORATION_END)
    p.add_argument("--latency-ms", type=int, default=LATENCY_MS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.end > EXPLORATION_END:
        print(f"refused: exploration stays before {EXPLORATION_END}")
        return 2
    uni = json.loads(Path(a.universe).read_text())
    pairs = [x for s in sorted(uni["research"], key=int) for x in uni["research"][s]]
    res = explore(Path(a.dataset), pairs, a.start, a.end, a.latency_ms, a.workers)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(summarize(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
