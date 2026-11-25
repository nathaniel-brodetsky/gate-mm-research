"""Exploration: realized spread by the move of a reference market around the trade (stage 3, step 3c).

Is a maker on an illiquid pair picked off when the wider market moves? For every trade of the
exploration period and every research pair, the log change of a reference price (spot BTC_USDT on the
one-second grid of `mmlab.market_data.reference_grid`):

- as of 1 s before the trade, over the last 1, 5, 10, 30 and 60 s (`ref_1s` .. `ref_60s`): tradable;
- ex post, from just before the trade to about 60 s after it (`ref_next_60s_ex_post`): not tradable,
  a ceiling for any signal about the reference.

Changes are in bp and signed so that positive means the reference moved the taker's way. Per bucket,
theta is the volume-weighted RS(60 s) minus the maker fee, as in `state_explore`, for all trades and for
the H-003 subset (quoted half-spread of at least 80 bp). Per feature, the rule "do not quote while the
signed move exceeds 3 bp" gives the kept share of the volume and the change in theta.

The rule (docs/leadlag-plan.md, written before the run): if skipping by the ex-post move gains less
than 3 bp for the median pair on all trades, lead-lag from this reference is closed. Otherwise an as-of
lag whose skip gains at least 3 bp for the median pair while keeping at least half of the volume, on
all trades or on the H-003 subset, is a candidate for the registered batch H-004 (D-002, December).

Usage:
    python -m mmlab.microstructure.leadlag_explore data/processed/U001 \\
        --reference data/processed/reference/BTC_USDT.parquet --out results/h004_explore/leadlag_btc.json
"""

from __future__ import annotations

import argparse
import functools
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
from mmlab.feature_engineering.asof_activity import grid_index
from mmlab.feature_engineering.trade_state import taker_sides
from mmlab.microstructure.markouts import in_period
from mmlab.microstructure.quote_pause import EXPLORATION_END
from mmlab.microstructure.state_explore import bucket_table, labels

F = NDArray[np.float64]

LAGS_S = (1, 5, 10, 30, 60)
AHEAD_S = 60
EX_POST = "ref_next_60s_ex_post"
FEATURES = (*(f"ref_{lag}s" for lag in LAGS_S), EX_POST)
SIGNED_BP = (-math.inf, -10.0, -3.0, -1.0, 1.0, 3.0, 10.0, math.inf)
EDGES: dict[str, tuple[float, ...]] = {f: SIGNED_BP for f in FEATURES}
LATENCY_MS = 1000
SKIP_BP = 3.0
MIN_GAIN_BP = 3.0
MIN_KEPT = 0.5


@functools.lru_cache(maxsize=2)
def load_reference(path: str) -> tuple[int, F]:
    """(first second, log of price_before) of a reference grid; the seconds must be contiguous."""
    df = pl.read_parquet(path, columns=["second", "price_before"])
    sec = df["second"].to_numpy()
    if len(sec) == 0 or int(sec[-1] - sec[0]) + 1 != len(sec):
        raise ValueError(f"{path}: the reference grid must hold contiguous seconds")
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(df["price_before"].to_numpy())
    return int(sec[0]), np.asarray(logp, dtype=np.float64)


def reference_features(ts_us: NDArray[np.int64], d: F, g0: int, logp: F, latency_us: int) -> dict[str, F]:
    """Signed reference moves in bp around trades at ts_us with taker directions d (+1 buy, -1 sell)."""
    n = len(logp)

    def at(i: NDArray[np.int64]) -> F:
        ok = (i >= 0) & (i < n)
        out = np.full(i.shape, np.nan)
        out[ok] = logp[i[ok]]
        return out

    gi = grid_index(ts_us, g0, latency_us)
    now = at(gi)
    feats = {f"ref_{lag}s": d * (now - at(gi - lag)) * 1e4 for lag in LAGS_S}
    gn = grid_index(ts_us, g0, 0)
    feats[EX_POST] = d * (at(gn + AHEAD_S + 1) - at(gn)) * 1e4
    return feats


def skip_effect(v: F, x: F, signed_bp: F, skip_bp: float = SKIP_BP) -> dict[str, float | None]:
    """Kept volume share and theta change when trades with a signed move above skip_bp are not quoted."""
    vol = float(v.sum())
    with np.errstate(invalid="ignore"):
        kept = ~(signed_bp > skip_bp)
    kv = float(v[kept].sum())
    if vol <= 0:
        return {"kept_share": None, "theta_kept": None, "gain": None}
    theta = float((v * x).sum() / vol)
    theta_kept = float((v[kept] * x[kept]).sum() / kv) if kv > 0 else None
    return {
        "kept_share": kv / vol,
        "theta_kept": theta_kept,
        "gain": None if theta_kept is None else theta_kept - theta,
    }


def explore_pair(
    ds: Path, pair: str, start: str, end: str, latency_us: int, reference: str
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
    g0, logp = load_reference(reference)
    feats = reference_features(st["ts_us"].to_numpy(), st["d"].to_numpy(), g0, logp, latency_us)
    wide = np.zeros(st.height, dtype=bool)
    wide[eligible(st, THRESHOLD_BP)["_r"].to_numpy()] = True
    out: dict[str, Any] = {}
    for subset, m in (("all", np.ones(st.height, dtype=bool)), ("q80", wide)):
        vol = float(v[m].sum())
        out[subset] = {
            "trades": int(m.sum()),
            "volume": vol,
            "theta": float((v[m] * x[m]).sum() / vol) if vol > 0 else None,
            "features": bucket_table(v[m], x[m], day[m], {k: f[m] for k, f in feats.items()}, EDGES),
            "skip": {k: skip_effect(v[m], x[m], f[m]) for k, f in feats.items()},
        }
    return pair, out


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def screen(pairs: Mapping[str, Any]) -> dict[str, Any]:
    """Medians over pairs of theta and of the skip effects; the decision of the rule."""
    out: dict[str, Any] = {}
    for subset in ("all", "q80"):
        rows = [p[subset] for p in pairs.values() if p[subset]["trades"] >= MIN_TRADES]
        th = _median([r["theta"] for r in rows if r["theta"] is not None])
        skip = {}
        for f in FEATURES:
            effects = [r["skip"][f] for r in rows if r["skip"][f]["gain"] is not None]
            gains = [float(e["gain"]) for e in effects]
            g = _median(gains)
            skip[f] = {
                "pairs": len(effects),
                "kept_share": _median([float(e["kept_share"]) for e in effects]),
                "gain_bp": None if g is None else round(g * 1e4, 6),  # rounded: 3 bp is 3 bp
                "improved": sum(x > 0 for x in gains),
            }
        theta_bp = None if th is None else round(th * 1e4, 6)
        out[subset] = {"pairs": len(rows), "theta_bp": theta_bp, "skip": skip}
    ceiling = out["all"]["skip"][EX_POST]["gain_bp"]
    candidates = [
        f"{subset} {f}"
        for subset in ("all", "q80")
        for f in FEATURES[:-1]
        if (s := out[subset]["skip"][f])["gain_bp"] is not None
        and s["gain_bp"] >= MIN_GAIN_BP
        and (s["kept_share"] or 0.0) >= MIN_KEPT
    ]
    out["decision"] = {
        "ceiling_gain_bp": ceiling,
        "closed": ceiling is None or ceiling < MIN_GAIN_BP,
        "candidates": candidates,
        "rule": f"closed if the ex-post skip gains < {MIN_GAIN_BP:g} bp (median pair, all trades); "
        f"candidate: an as-of lag gaining >= {MIN_GAIN_BP:g} bp, keeping >= {MIN_KEPT:.0%} of the volume",
    }
    return out


def explore(
    ds: Path,
    pairs: Sequence[str],
    start: str,
    end: str,
    reference: str,
    latency_ms: int = LATENCY_MS,
    workers: int = 1,
) -> dict[str, Any]:
    args = [(ds, p, start, end, latency_ms * 1000, reference) for p in pairs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(explore_pair, *zip(*args, strict=True)))
    else:
        done = [explore_pair(*a) for a in args]
    got = {p: r for p, r in done if r is not None}
    return {
        "period": [start, end],
        "latency_ms": latency_ms,
        "reference": Path(reference).name,
        "skip_bp": SKIP_BP,
        "edges_bp": [str(e) for e in SIGNED_BP],
        "pairs": got,
        "too_thin": [p for p, r in done if r is None],
        "screen": screen(got),
    }


def _bp(x: float | None) -> str:
    return "     n/a" if x is None else f"{x:+8.2f}"


def summarize(res: Mapping[str, Any]) -> str:
    pairs = res["pairs"]
    sc = res["screen"]
    lines = [
        f"realized spread by the move of {res['reference']} around the trade, {res['period'][0]} .. "
        f"{res['period'][1]}; signed bp, + = the taker's way; as of {res['latency_ms']} ms before, except "
        f"{EX_POST}; {len(pairs)} pairs; exploration, no inference"
    ]
    for subset in ("all", "q80"):
        lines.append(f"{subset}: theta of all its trades, median pair {_bp(sc[subset]['theta_bp'])} bp")
        for name in FEATURES:
            lines.append(f"  {name}")
            for lab in labels(SIGNED_BP):
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
                med = _bp(statistics.median(th) * 1e4 if th else None)
                share = f"{statistics.median(shares):.3f}" if shares else "n/a"
                lines.append(f"    {lab:12} pairs {len(th):2d}  volume share {share:>5}  theta bp {med}")
            lines.append(f"    skip > {res['skip_bp']:g} bp: {_skip(sc[subset]['skip'][name])}")
    lines.append(_decision(sc["decision"]))
    if res.get("too_thin"):
        lines.append(f"too thin: {', '.join(res['too_thin'])}")
    return "\n".join(lines)


def _skip(s: Mapping[str, Any]) -> str:
    kept = "  n/a" if s["kept_share"] is None else f"{s['kept_share']:.3f}"
    return f"kept {kept}  gain bp {_bp(s['gain_bp'])}  improved {s['improved']}/{s['pairs']}"


def _decision(d: Mapping[str, Any]) -> str:
    return (
        f"decision: ceiling (ex post) {_bp(d['ceiling_gain_bp'])} bp -> "
        f"{'closed' if d['closed'] else 'open'}; candidates: {', '.join(d['candidates']) or 'none'}"
    )


def brief(res: Mapping[str, Any]) -> str:
    """The screen only: per subset and feature, the effect of not quoting; the decision."""
    sc = res["screen"]
    lines = [
        f"lead-lag with {res['reference']}: no quote while the signed move exceeds {res['skip_bp']:g} bp; "
        f"medians over pairs; exploration, no inference"
    ]
    for subset in ("all", "q80"):
        lines.append(f"{subset}: {sc[subset]['pairs']} pairs, theta {_bp(sc[subset]['theta_bp'])} bp")
        lines += [f"  {f:21} {_skip(sc[subset]['skip'][f])}" for f in FEATURES]
    lines.append(_decision(sc["decision"]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Realized spread by the move of a reference market.")
    p.add_argument("dataset")
    p.add_argument("--reference", required=True, help="parquet from mmlab.market_data.reference_grid")
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
    res = explore(Path(a.dataset), pairs, a.start, a.end, a.reference, a.latency_ms, a.workers)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(summarize(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
