"""Exploration: pausing quotes when a pair's prices or trading turn unusual (stage 3, step 3).

Step 2e found that a maker's losses on these pairs come in rare days of extreme prices of a single
token. This exploration asks, on the exploration period only, whether rules that see only as-of
information stop quoting on such days early enough to matter. It decides nothing: candidates go to a
batch registered under D-002 and tested on December.

A rule pauses quotes at time t when a signal over the last W seconds, known `latency` before t
(`mmlab.feature_engineering.asof_activity`), is more than k times its baseline, the mean over the
previous 7 days:
- range: max / min - 1 of the mid (the baseline is floored at 1 bp);
- count: number of trades (the baseline is floored at 1 trade).
A paused maker takes no part in the trades during the pause; on the other trades it earns what the
liquidity that traded earned (volume-weighted RS(60 s) minus the maker fee, as in H-001). Days with
fewer than 3 previous days have no baseline and are never paused.

Per pair and rule: share of volume kept, theta of all, kept and paused trades, the change of theta,
and the share of the pair's worst day (largest volume-weighted loss against its theta) that was
paused. For scale, the theta without the pair's three worst days (an ex-post bound no rule can reach).

Usage:
    python -m mmlab.microstructure.quote_pause data/processed/U001 --out results/h004_explore/pause.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.evaluation.h003 import MAKER_FEE, MIN_DAYS, MIN_TRADES, states
from mmlab.feature_engineering.asof_activity import (
    DAY_S,
    daily_baseline,
    grid_index,
    mid_before,
    trades_before,
    trailing_range,
)
from mmlab.microstructure.markouts import in_period

SIGNALS = ("range", "count")
WINDOWS_S = (60, 300, 1800)
MULTIPLES = (2.0, 3.0, 5.0, 10.0)
FLOORS = {"range": 1e-4, "count": 1.0}
LATENCY_MS = 1000
EXPLORATION_END = "2026-11-01"


def rule_name(signal: str, w: int, k: float) -> str:
    return f"{signal}/{w}s/x{k:g}"


RULES = tuple(rule_name(s, w, k) for s in SIGNALS for w in WINDOWS_S for k in MULTIPLES)


def _epoch_s(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def pause_flags(
    tob: pl.DataFrame,
    trade_ts: NDArray[np.int64],
    ts_us: NDArray[np.int64],
    start: str,
    end: str,
    latency_us: int,
) -> dict[str, NDArray[np.bool_]]:
    """For every rule, whether each trade at ts_us falls into a pause. trade_ts: all trades, sorted."""
    g0 = _epoch_s(start)
    n = (_epoch_s(end) - g0) // DAY_S * DAY_S
    grid = g0 + np.arange(n, dtype=np.int64)
    gi = grid_index(ts_us, g0, latency_us)
    known = (gi >= 0) & (gi < n)
    at = np.clip(gi, 0, n - 1)
    mid = mid_before(tob, grid)
    flags: dict[str, NDArray[np.bool_]] = {}
    for w in WINDOWS_S:
        signal = {
            "range": trailing_range(mid, w),
            "count": trades_before(trade_ts, grid, w).astype(np.float64),
        }
        for name in SIGNALS:
            values = signal[name]
            base = daily_baseline(values)
            v_at, b_at = values[at], base[at]
            with np.errstate(invalid="ignore"):
                level = np.maximum(b_at, FLOORS[name])
                for k in MULTIPLES:
                    flags[rule_name(name, w, k)] = known & np.isfinite(b_at) & (v_at > k * level)
    return flags


def _theta(v: NDArray[np.float64], x: NDArray[np.float64]) -> float | None:
    s = float(v.sum())
    return float((v * x).sum() / s) if s > 0 else None


def evaluate(
    v: NDArray[np.float64],
    x: NDArray[np.float64],
    day: NDArray[np.int64],
    flags: Mapping[str, NDArray[np.bool_]],
) -> dict[str, Any]:
    """Theta of all, kept and paused trades per rule; share of the worst day paused; ex-post bound."""
    theta = _theta(v, x)
    if theta is None:
        raise ValueError("no volume")
    days = np.unique(day)
    loss = np.array([float((v[day == d] * (x[day == d] - theta)).sum()) for d in days])
    worst = int(days[int(np.argmin(loss))])
    top3 = days[np.argsort(loss, kind="stable")[:3]]
    out: dict[str, Any] = {
        "trades": int(v.size),
        "active_days": int(days.size),
        "theta": theta,
        "worst_day": worst,
        "theta_without_3_worst_days": _theta(v[~np.isin(day, top3)], x[~np.isin(day, top3)]),
        "rules": {},
    }
    on_worst = day == worst
    for rule, f in flags.items():
        kept = _theta(v[~f], x[~f])
        out["rules"][rule] = {
            "kept_volume": float(v[~f].sum() / v.sum()),
            "theta_kept": kept,
            "theta_paused": _theta(v[f], x[f]),
            "delta": None if kept is None else kept - theta,
            "worst_day_paused": float(v[f & on_worst].sum() / v[on_worst].sum()),
        }
    return out


def explore_pair(
    ds: Path, pair: str, start: str, end: str, latency_us: int
) -> tuple[str, dict[str, Any] | None]:
    trades = pl.read_parquet(ds / "trades" / f"{pair}.parquet")
    tob = pl.read_parquet(ds / "tob" / f"{pair}.parquet")
    st = in_period(states(trades, tob), start, end).filter(pl.col("rs_60s").is_not_null())
    if st.height < MIN_TRADES or st["day"].n_unique() < MIN_DAYS:
        return pair, None
    d0 = date.fromisoformat(start)
    day = np.array([(d - d0).days for d in st["day"].to_list()], dtype=np.int64)
    trade_ts = np.sort(trades["ts_us"].to_numpy())
    flags = pause_flags(tob, trade_ts, st["ts_us"].to_numpy(), start, end, latency_us)
    v = st["v"].to_numpy()
    x = st["rs_60s"].to_numpy() - MAKER_FEE
    return pair, evaluate(v, x, day, flags)


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
        "rules": list(RULES),
        "pairs": {p: r for p, r in done if r is not None},
        "too_thin": [p for p, r in done if r is None],
    }


def _med(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def summarize(res: Mapping[str, Any]) -> str:
    pairs = res["pairs"]
    thetas = [e["theta"] for e in pairs.values()]
    bound = [e["theta_without_3_worst_days"] - e["theta"] for e in pairs.values()]
    lines = [
        f"quote pauses, {res['period'][0]} .. {res['period'][1]}, signals known {res['latency_ms']} ms "
        f"before the trade; {len(pairs)} pairs; exploration, no inference",
        f"median theta of all trades {_med(thetas) * 1e4:+.2f} bp; median gain without the 3 worst days "
        f"(ex post) {_med(bound) * 1e4:+.2f} bp",
        f"  {'rule':16} {'delta bp':>9} {'improved':>8} {'kept vol':>8} {'theta kept':>10} "
        f"{'worst day paused':>16}",
    ]
    rows = []
    for rule in res["rules"]:
        es = [e["rules"][rule] for e in pairs.values()]
        deltas = [r["delta"] for r in es if r["delta"] is not None]
        kept = [r["theta_kept"] for r in es if r["theta_kept"] is not None]
        rows.append(
            (
                _med(deltas),
                f"  {rule:16} {_med(deltas) * 1e4:+9.2f} {sum(d > 0 for d in deltas):4d}/{len(deltas):<3d} "
                f"{_med([r['kept_volume'] for r in es]):8.3f} {_med(kept) * 1e4:+10.2f} "
                f"{_med([r['worst_day_paused'] for r in es]):16.3f}",
            )
        )
    lines += [line for _, line in sorted(rows, key=lambda t: -t[0] if t[0] == t[0] else 0.0)]
    if res.get("too_thin"):
        lines.append(f"too thin: {', '.join(res['too_thin'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Explore quote pauses on unusual prices or trading.")
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
