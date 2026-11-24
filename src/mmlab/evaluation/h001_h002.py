"""Tests of H-001 (realized spread above the maker fee) and H-002 (share of impact after 1 s).

Both hypotheses were registered on 2026-11-23 (hypotheses/H-001-*.md, H-002-*.md); this module
implements them as written. Periods: development 2026-08-01 .. 2026-09-30, validation 2026-10-01 ..
2026-10-31. The decisions are taken on those periods only. Once November is opened (stage 2 plan,
step 10) the same statistics are computed as estimates, which do not revise the decisions:
  november  research pairs, 2026-11-01 .. 2026-11-30, after H-003 has been tested (--h003)
  holdout   U-001 holdout pairs, 2026-08-01 .. 2026-11-30, after H-003 is decided (--h003)

Usage:
  python -m mmlab.evaluation.h001_h002 data/processed/U001 --period development \\
      --universe configs/universe_U001.json --out data/results/h001_h002_development.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices
from mmlab.evaluation.h003 import coverage_problems
from mmlab.evaluation.multiple_testing import benjamini_hochberg
from mmlab.hypothesis_registry.holdout import h003_tested, holdout_unlocked
from mmlab.microstructure.markouts import HORIZONS_S, _hkey, daily_sums, in_period, markouts

PERIODS = {"development": ("2026-08-01", "2026-10-01"), "validation": ("2026-10-01", "2026-11-01")}
ESTIMATES = {"november": ("2026-11-01", "2026-12-01"), "holdout": ("2026-08-01", "2026-12-01")}
MAKER_FEE = 0.0010  # global spot VIP 0, from 2026-04-09
RESAMPLES = 10_000
MEAN_BLOCK_DAYS = 3.0
SEED_H001 = 20261123
SEED_H002 = 20261124
BH_Q = 0.10
MIN_TRADES = 30
MIN_DAYS = 5


@dataclass
class PairResult:
    pair: str
    trades: int
    days: int
    es: float | None = None
    rs: dict[str, float | None] = field(default_factory=dict)
    impact: dict[str, float | None] = field(default_factory=dict)
    h001_theta: float | None = None  # RS(60 s) - maker fee
    h001_ci: tuple[float, float] | None = None
    h001_p: float = 1.0
    h001_theta_micro: float | None = None
    h002_s: float | None = None  # impact(1 s) / impact(300 s), trades valid at both horizons
    h002_ci: tuple[float, float] | None = None
    note: str = ""


def _vw(df: pl.DataFrame, col: str) -> float | None:
    d = df.filter(pl.col(col).is_not_null())
    den = float(d["v"].sum()) if d.height else 0.0
    return float((d["v"] * d[col]).sum()) / den if den else None


def evaluate_pair(pair: str, tob: pl.DataFrame, trades: pl.DataFrame, start: str, end: str) -> PairResult:
    mk = in_period(markouts(trades, tob), start, end)
    res = PairResult(pair, mk.height, mk["day"].n_unique() if mk.height else 0)
    if mk.height == 0:
        res.note = "no usable trades"
        return res
    res.es = _vw(mk, "es")
    for h in HORIZONS_S:
        res.rs[_hkey(h)] = _vw(mk, f"rs_{_hkey(h)}")
        res.impact[_hkey(h)] = _vw(mk, f"pi_{_hkey(h)}")

    # H-001
    num, den = daily_sums(mk, "rs_60s")
    if len(num) >= MIN_DAYS and mk["rs_60s"].drop_nulls().len() >= MIN_TRADES:
        n_arr, d_arr = np.asarray(num), np.asarray(den)
        res.h001_theta = float(n_arr.sum() / d_arr.sum()) - MAKER_FEE
        boot = ratio_resamples(
            n_arr, d_arr, stationary_indices(len(num), RESAMPLES, MEAN_BLOCK_DAYS, SEED_H001)
        )
        boot = boot[~np.isnan(boot)] - MAKER_FEE
        res.h001_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        res.h001_p = float((1 + np.sum(boot <= 0)) / (boot.size + 1))
    else:
        res.note = "too few trades or days for H-001"
    micro = in_period(markouts(trades, tob, horizons_s=(60,), price="micro"), start, end)
    t_micro = _vw(micro, "rs_60s") if micro.height else None
    res.h001_theta_micro = None if t_micro is None else t_micro - MAKER_FEE

    # H-002: common subset valid at 1 s and 300 s, S = sum(v PI_1) / sum(v PI_300)
    both = mk.filter(pl.col("pi_1s").is_not_null() & pl.col("pi_300s").is_not_null())
    n1, _ = daily_sums(both, "pi_1s")
    n300, _ = daily_sums(both, "pi_300s")
    if len(n1) >= MIN_DAYS and both.height >= MIN_TRADES and sum(n300) > 0:
        a1, a300 = np.asarray(n1), np.asarray(n300)
        res.h002_s = float(a1.sum() / a300.sum())
        idx = stationary_indices(len(n1), RESAMPLES, MEAN_BLOCK_DAYS, SEED_H002)
        s_boot = ratio_resamples(a1, a300, idx)
        s_boot = s_boot[np.isfinite(s_boot)]
        res.h002_ci = (float(np.percentile(s_boot, 2.5)), float(np.percentile(s_boot, 97.5)))
    elif both.height:
        res.note = (res.note + "; " if res.note else "") + "H-002 undefined (impact at 300 s <= 0 or too few)"
    return res


def decide(results: Sequence[PairResult], registered_pairs: int) -> dict[str, Any]:
    thetas = [r.h001_theta for r in results if r.h001_theta is not None]
    rejects = benjamini_hochberg([r.h001_p for r in results], BH_Q)
    need = registered_pairs // 2
    h001 = {
        "rejections_after_bh": sum(rejects),
        "required": need,
        "median_theta": statistics.median(thetas) if thetas else None,
        "pairs_rejected": [r.pair for r, x in zip(results, rejects, strict=True) if x],
    }
    h001["criterion_met"] = bool(sum(rejects) >= need and thetas and statistics.median(thetas) > 0)

    s_vals = np.asarray([r.h002_s for r in results if r.h002_s is not None])
    h002: dict[str, Any] = {"pairs_defined": int(s_vals.size)}
    if s_vals.size:
        rng = np.random.default_rng(SEED_H002)
        meds = np.median(s_vals[rng.integers(0, s_vals.size, (RESAMPLES, s_vals.size))], axis=1)
        lo, hi = float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
        h002.update(median_s=float(np.median(s_vals)), ci=(lo, hi))
        h002["outcome"] = "accept" if hi < 0.5 else "reject" if lo > 0.5 else "undecided"
    else:
        h002["outcome"] = "undecided"
    return {"H-001": h001, "H-002": h002}


def _bp(v: float | None) -> str:
    return "     n/a" if v is None else f"{v * 1e4:8.1f}"


def _line(r: PairResult) -> str:
    s = "n/a" if r.h002_s is None else f"{r.h002_s:.2f}"
    return (
        f"{r.pair:16} trades {r.trades:6} days {r.days:3} | ES {_bp(r.es)} RS1 {_bp(r.rs.get('1s'))} "
        f"RS60 {_bp(r.rs.get('60s'))} RS300 {_bp(r.rs.get('300s'))} | "
        f"theta {_bp(r.h001_theta)} p {r.h001_p:.4f} | S {s} {r.note}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate H-001 and H-002 on a built dataset.")
    p.add_argument("dataset")
    p.add_argument("--period", required=True, choices=[*sorted(PERIODS), *ESTIMATES])
    p.add_argument("--h003", help="H-003 test result; required for the november and holdout estimates")
    p.add_argument("--universe", default="configs/universe_U001.json")
    p.add_argument("--exclude", default="", help="comma-separated pairs left out (reported variant)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    uni = json.loads(Path(a.universe).read_text())
    part = "holdout" if a.period == "holdout" else "research"
    registered = [x for s in sorted(uni[part], key=int) for x in uni[part][s]]
    excluded = {x.strip() for x in a.exclude.split(",") if x.strip()}
    pairs = [x for x in registered if x not in excluded]
    ds = Path(a.dataset)
    if a.period in ESTIMATES:
        err = h003_tested(a.h003) if a.period == "november" else holdout_unlocked(a.h003)
        if err:
            print(f"refused: {err}")
            return 2
        start, end = ESTIMATES[a.period]
        problems = coverage_problems(ds, pairs, start, end)
        if problems:
            print("dataset does not cover the period:\n  " + "\n  ".join(problems))
            return 2
    else:
        missing = [x for x in pairs if not (ds / "tob" / f"{x}.parquet").exists()]
        if missing:
            print(f"dataset incomplete, missing: {missing}")
            return 2
        start, end = PERIODS[a.period]
    results = []
    for x in pairs:
        r = evaluate_pair(
            x,
            pl.read_parquet(ds / "tob" / f"{x}.parquet"),
            pl.read_parquet(ds / "trades" / f"{x}.parquet"),
            start,
            end,
        )
        results.append(r)
        print(_line(r))
    decision = decide(results, registered_pairs=len(registered))
    doc = {
        "period": a.period,
        "range": [start, end],
        "pairs": pairs,
        "excluded": sorted(excluded),
        "params": {
            "maker_fee": MAKER_FEE,
            "resamples": RESAMPLES,
            "mean_block_days": MEAN_BLOCK_DAYS,
            "seeds": [SEED_H001, SEED_H002],
            "bh_q": BH_Q,
            "min_trades": MIN_TRADES,
            "min_days": MIN_DAYS,
        },
        "results": [asdict(r) for r in results],
        "decision": decision,
        "role": "estimate: the registered decisions stand" if a.period in ESTIMATES else "decision period",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print("bp = basis points (1e-4); theta = RS60 - maker fee")
    print(json.dumps(decision, indent=2, default=str))
    if a.period in ESTIMATES:
        print("estimate only: H-001 and H-002 were decided on the development and validation periods")
    return 0


if __name__ == "__main__":
    sys.exit(main())
