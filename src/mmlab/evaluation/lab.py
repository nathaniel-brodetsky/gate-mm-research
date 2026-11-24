"""Synthetic lab: do our statistical methods behave as claimed on data like ours?

The generator (`mmlab.stochastic_models.markout_panel`) is calibrated per pair on the exploration period
(2026-08-01 .. 2026-10-31, research pairs), then the methods are run many times on synthetic panels whose
truth is known:

  1. day bootstrap (H-001, H-003): coverage of the 95% interval and the rejection rate of the one-sided
     test at 5% when the true theta is 0, for 30 and 61 days;
  2. Benjamini-Hochberg across 24 dependent pairs: false discovery rate and power;
  3. bootstrap over pairs of the median (H-002, H-003): coverage, and how often the lower end is above 0
     when the true median is 0;
  4. the registered H-003 rule on a synthetic November: how often it accepts, rejects or extends when the
     August-October effect persists, halves or is absent.
Monte Carlo errors are reported with every rate. The lab decides nothing; it says how far the methods can
be trusted and how much a test can see.

Usage:
  python -m mmlab.evaluation.lab calibrate data/processed/U001 --out results/lab/calibration.json
  python -m mmlab.evaluation.lab run results/lab/calibration.json --out results/lab/checks.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import statistics
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices
from mmlab.evaluation.h003 import THRESHOLD_BP, eligible, states
from mmlab.evaluation.multiple_testing import benjamini_hochberg
from mmlab.microstructure.markouts import in_period
from mmlab.stochastic_models.markout_panel import PairProfile, ar1, simulate_days

MAKER_FEE = 0.0010
MEAN_BLOCK_DAYS = 3.0
MIN_TRADES = 30
MIN_DAYS = 5
EXPLORATION_END = "2026-11-01"
BH_ALTERNATIVE = 0.0010  # theta of the non-null pairs in the BH check: +10 bp after the fee


# ---------------------------------------------------------------- calibration


def calibrate(
    day: NDArray[np.int64], v: NDArray[np.float64], x: NDArray[np.float64], n_days: int
) -> tuple[PairProfile, float, NDArray[np.float64]]:
    """Profile, volume-weighted theta and the daily theta series (nan on days without trades)."""
    counts = np.bincount(day, minlength=n_days).astype(float)
    m, s2 = counts.mean(), counts.var(ddof=1)
    k = m * m / (s2 - m) if s2 > m > 0 else None
    logv = np.log(v)
    theta = float((v * x).sum() / v.sum())
    num = np.bincount(day, weights=v * x, minlength=n_days)
    den = np.bincount(day, weights=v, minlength=n_days)
    v2 = np.bincount(day, weights=v * v, minlength=n_days)
    active = den > 0
    theta_d = np.full(n_days, np.nan)
    theta_d[active] = num[active] / den[active]
    resid = x - theta_d[day]
    dof = max(x.size - int(active.sum()), 1)
    trade_sd = float(np.sqrt((resid**2).sum() / dof))
    m2 = float((resid**2).mean())
    excess = float((resid**4).mean() / (m2 * m2) - 3) if m2 > 0 else 0.0
    df = float(np.clip(4 + 6 / excess, 2.5, 1000.0)) if excess > 0 else 1000.0
    noise = trade_sd**2 * v2[active] / den[active] ** 2
    obs = theta_d[active]
    day_var = float(obs.var(ddof=1)) if obs.size > 1 else 0.0
    day_sd = math.sqrt(max(0.0, day_var - float(noise.mean())))
    if day_sd <= 1e-9 * (abs(theta) + trade_sd + 1e-12):  # below float precision: no day effect
        day_sd = 0.0
    both = active[1:] & active[:-1]
    ar = 0.0
    if day_sd > 0 and both.sum() >= 3:
        a, b = theta_d[1:][both] - obs.mean(), theta_d[:-1][both] - obs.mean()
        scale = math.sqrt((a * a).sum() * (b * b).sum())
        if scale > 0:
            # trade noise is independent across days and dilutes the lag-1 correlation of daily theta
            ar = float(np.clip((a * b).sum() / scale * day_var / day_sd**2, -0.9, 0.9))
    profile = PairProfile(
        trades_per_day=float(m),
        count_k=None if k is None else float(k),
        logv_mean=float(logv.mean()),
        logv_sd=float(logv.std(ddof=1)) if logv.size > 1 else 0.0,
        trade_sd=trade_sd,
        trade_df=df,
        day_sd=day_sd,
        day_ar=ar,
    )
    return profile, theta, theta_d


def mean_pairwise_corr(series: Sequence[NDArray[np.float64]], min_overlap: int = 10) -> float:
    """Average correlation of daily theta across pairs, over days where both pairs traded."""
    cs = []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            ok = ~np.isnan(series[i]) & ~np.isnan(series[j])
            if ok.sum() >= min_overlap:
                a, b = series[i][ok], series[j][ok]
                if a.std() > 0 and b.std() > 0:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cs)) if cs else 0.0


def calibrate_dataset(ds: Path, pairs: Sequence[str], start: str, end: str) -> dict[str, Any]:
    n_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    d0 = date.fromisoformat(start)
    out: dict[str, Any] = {"period": [start, end], "n_days": n_days, "subsets": {}}
    per: dict[str, dict[str, Any]] = {"all": {}, "q80": {}}
    daily: dict[str, list[NDArray[np.float64]]] = {"all": [], "q80": []}
    for x in pairs:
        st = in_period(
            states(
                pl.read_parquet(ds / "trades" / f"{x}.parquet"), pl.read_parquet(ds / "tob" / f"{x}.parquet")
            ),
            start,
            end,
        )
        for name, sub in (("all", st), ("q80", eligible(st, THRESHOLD_BP))):
            rs = sub.filter(pl.col("rs_60s").is_not_null())
            day = np.array([(d - d0).days for d in rs["day"].to_list()], dtype=np.int64)
            n_active = len(set(day.tolist()))
            entry: dict[str, Any] = {"trades": rs.height, "active_days": n_active}
            if rs.height >= MIN_TRADES and n_active >= MIN_DAYS:
                prof, theta, th_d = calibrate(
                    day, rs["v"].to_numpy(), rs["rs_60s"].to_numpy() - MAKER_FEE, n_days
                )
                entry.update(profile=asdict(prof), theta=theta)
                daily[name].append(th_d)
            else:
                v = rs["v"].to_numpy()
                entry.update(
                    profile=None,
                    theta=float((v * (rs["rs_60s"].to_numpy() - MAKER_FEE)).sum() / v.sum())
                    if v.size
                    else None,
                )
            per[name][x] = entry
    for name in per:
        ars = [e["profile"]["day_ar"] for e in per[name].values() if e["profile"]]
        out["subsets"][name] = {
            "pairs": per[name],
            "rho": mean_pairwise_corr(daily[name]),
            "common_ar": float(statistics.median(ars)) if ars else 0.0,
        }
    return out


def profiles_for(subset: dict[str, Any], n_days: int) -> dict[str, tuple[PairProfile, float]]:
    """Profile and theta per pair; pairs too thin to calibrate borrow the median profile at their own rate."""
    known = {p: e for p, e in subset["pairs"].items() if e["profile"]}
    med = {
        f.name: statistics.median(
            e["profile"][f.name] for e in known.values() if e["profile"][f.name] is not None
        )
        for f in fields(PairProfile)
        if f.name != "count_k"
    }
    out = {}
    for pair, e in subset["pairs"].items():
        if e["profile"]:
            out[pair] = (PairProfile(**e["profile"]), float(e["theta"]))
        else:
            rate = e["trades"] / n_days
            out[pair] = (
                PairProfile(**{**med, "trades_per_day": rate, "count_k": None}),
                float(e["theta"] or 0.0),
            )
    return out


# ---------------------------------------------------------------- checks


def _day_test(
    num: NDArray[np.float64], den: NDArray[np.float64], resamples: int, seed: int
) -> tuple[float, float, float] | None:
    """(lower, upper, one-sided p) of the day bootstrap as in H-001/H-003; None below MIN_DAYS days."""
    keep = den > 0
    if keep.sum() < MIN_DAYS:
        return None
    n, d = num[keep], den[keep]
    boot = ratio_resamples(n, d, stationary_indices(len(n), resamples, MEAN_BLOCK_DAYS, seed))
    boot = boot[~np.isnan(boot)]
    p = float((1 + np.sum(boot <= 0)) / (boot.size + 1))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), p


def _rate(k: int, n: int) -> dict[str, float]:
    r = k / n if n else float("nan")
    return {"rate": r, "mc_se": math.sqrt(r * (1 - r) / n) if n else float("nan")}


def day_bootstrap_check(
    p: PairProfile, theta: float, n_days: int, reps: int, resamples: int, seed: int, alpha: float = 0.05
) -> dict[str, Any]:
    cover = reject = used = 0
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        num, den, _ = simulate_days(p, theta, n_days, rng)
        res = _day_test(num, den, resamples, seed * 100_003 + r)
        if res is None:
            continue
        lo, hi, pv = res
        used += 1
        cover += lo <= theta <= hi
        reject += pv <= alpha
    return {
        "reps": used,
        "coverage": _rate(cover, used),
        "reject_at_alpha": _rate(reject, used),
        "alpha": alpha,
    }


def bh_check(
    profiles: Sequence[PairProfile],
    thetas: Sequence[float],
    n_days: int,
    rho: float,
    common_ar: float,
    reps: int,
    resamples: int,
    seed: int,
    q: float = 0.10,
) -> dict[str, Any]:
    null = np.array([t <= 0 for t in thetas])
    fdp, any_false, power, rejections = [], 0, [], []
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        common = ar1(n_days, common_ar, rng)
        ps = []
        for i, (p, t) in enumerate(zip(profiles, thetas, strict=True)):
            num, den, _ = simulate_days(p, t, n_days, rng, common, rho)
            res = _day_test(num, den, resamples, seed * 100_003 + r * 101 + i)
            ps.append(1.0 if res is None else res[2])
        rej = np.array(benjamini_hochberg(ps, q))
        v, total = int((rej & null).sum()), int(rej.sum())
        fdp.append(v / max(total, 1))
        any_false += v > 0
        rejections.append(total)
        if (~null).any():
            power.append(float(rej[~null].mean()))
    return {
        "reps": reps,
        "q": q,
        "fdr": {
            "rate": float(np.mean(fdp)),
            "mc_se": float(np.std(fdp, ddof=1) / math.sqrt(reps)) if reps > 1 else float("nan"),
        },
        "any_false_rejection": _rate(any_false, reps),
        "power": float(np.mean(power)) if power else None,
        "mean_rejections": float(np.mean(rejections)),
    }


def _median_ci(values: NDArray[np.float64], resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    meds = np.median(values[rng.integers(0, values.size, (resamples, values.size))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def median_ci_check(
    n_pairs: int, mu: float, tau: float, est_sd: float, reps: int, resamples: int, seed: int
) -> dict[str, Any]:
    cover = above = 0
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        vals = mu + tau * rng.standard_normal(n_pairs) + est_sd * rng.standard_normal(n_pairs)
        lo, hi = _median_ci(vals, resamples, rng)
        cover += lo <= mu <= hi
        above += lo > 0
    return {"reps": reps, "coverage": _rate(cover, reps), "lower_above_0": _rate(above, reps)}


def h003_check(
    profiles: dict[str, tuple[PairProfile, float]],
    scale: float,
    n_days: int,
    rho: float,
    common_ar: float,
    reps: int,
    resamples: int,
    seed: int,
    min_pairs: int = 8,
) -> dict[str, Any]:
    """The registered H-003 rule on synthetic months where every pair's theta is `scale` times its own."""
    outcomes = {"accept": 0, "reject": 0, "extend": 0}
    counted = []
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        common = ar1(n_days, common_ar, rng)
        est = []
        for p, theta in profiles.values():
            num, den, cnt = simulate_days(p, theta * scale, n_days, rng, common, rho)
            if cnt.sum() >= MIN_TRADES and (cnt > 0).sum() >= MIN_DAYS:
                est.append(num.sum() / den.sum())
        counted.append(len(est))
        if len(est) < min_pairs:
            outcomes["extend"] += 1
            continue
        lo, _ = _median_ci(np.asarray(est), resamples, rng)
        outcomes["accept" if lo > 0 else "reject"] += 1
    return {
        "reps": reps,
        "scale": scale,
        **{k: _rate(v, reps) for k, v in outcomes.items()},
        "mean_counted_pairs": float(np.mean(counted)),
    }


# ---------------------------------------------------------------- runner


def plan(
    cal: dict[str, Any], reps: int, resamples: int, seed: int
) -> list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]]:
    """All checks as (name, function, keyword arguments); each runs independently."""
    n_cal = int(cal["n_days"])
    allp = profiles_for(cal["subsets"]["all"], n_cal)
    q80 = profiles_for(cal["subsets"]["q80"], n_cal)
    tasks: list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]] = []
    for subset, profs, horizons in (("all", allp, (30, 61)), ("q80", q80, (30,))):
        calibrated = [k for k, e in cal["subsets"][subset]["pairs"].items() if e["profile"]]
        for n_days in horizons:
            for i, pair in enumerate(calibrated):
                tasks.append(
                    (
                        f"day_bootstrap/{subset}/{n_days}d/{pair}",
                        day_bootstrap_check,
                        {
                            "p": profs[pair][0],
                            "theta": 0.0,
                            "n_days": n_days,
                            "reps": reps,
                            "resamples": resamples,
                            "seed": seed + 1000 * n_days + i,
                        },
                    )
                )
    pairs = list(allp)
    sub = cal["subsets"]["all"]
    for n_days in (30, 61):
        for label, thetas in (
            ("all_null", [0.0] * len(pairs)),
            ("half_at_plus10bp", [BH_ALTERNATIVE if i % 2 else 0.0 for i in range(len(pairs))]),
        ):
            tasks.append(
                (
                    f"bh/all/{n_days}d/{label}",
                    bh_check,
                    {
                        "profiles": [allp[p][0] for p in pairs],
                        "thetas": thetas,
                        "n_days": n_days,
                        "rho": sub["rho"],
                        "common_ar": sub["common_ar"],
                        "reps": reps,
                        "resamples": resamples,
                        "seed": seed + 7 * n_days + len(label),
                    },
                )
            )
    # spread of pair thetas as observed in the exploration period (pairs that could be calibrated)
    q80_thetas = np.array(
        [float(e["theta"]) for e in cal["subsets"]["q80"]["pairs"].values() if e["profile"]]
    )
    tau = float(np.std(q80_thetas, ddof=1)) if q80_thetas.size > 1 else 0.0
    for n_pairs in (8, 12, 17, 24):
        tasks.append(
            (
                f"median_ci/{n_pairs}_pairs/mu0",
                median_ci_check,
                {
                    "n_pairs": n_pairs,
                    "mu": 0.0,
                    "tau": tau,
                    "est_sd": 0.0,
                    "reps": 4 * reps,
                    "resamples": resamples,
                    "seed": seed + n_pairs,
                },
            )
        )
    s80 = cal["subsets"]["q80"]
    for scale in (1.0, 0.5, 0.0):
        tasks.append(
            (
                f"h003/november/scale_{scale:g}",
                h003_check,
                {
                    "profiles": q80,
                    "scale": scale,
                    "n_days": 30,
                    "rho": s80["rho"],
                    "common_ar": s80["common_ar"],
                    "reps": 2 * reps,
                    "resamples": resamples,
                    "seed": seed + int(100 * scale),
                },
            )
        )
    # one distinct, reproducible seed per check
    return [(name, fn, {**kw, "seed": seed * 10_000 + i}) for i, (name, fn, kw) in enumerate(tasks)]


def _run(task: tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    name, fn, kw = task
    return name, fn(**kw)


def summarize(results: dict[str, dict[str, Any]]) -> str:
    lines = []
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, res in results.items():
        if name.startswith("day_bootstrap/"):
            groups.setdefault(name.rsplit("/", 1)[0], []).append((name, res))
    for g, items in sorted(groups.items()):
        cov = [r["coverage"]["rate"] for _, r in items if r["reps"]]
        rej = [r["reject_at_alpha"]["rate"] for _, r in items if r["reps"]]
        worst = max(items, key=lambda it: it[1]["reject_at_alpha"]["rate"] if it[1]["reps"] else -1)
        lines.append(
            f"{g:28} pairs {len(items):2}  coverage min/med/max "
            f"{min(cov):.3f}/{statistics.median(cov):.3f}/{max(cov):.3f}"
            f"  reject@5% min/med/max {min(rej):.3f}/{statistics.median(rej):.3f}/{max(rej):.3f}"
            f"  (worst {worst[0].rsplit('/', 1)[1]})"
        )
    for name, r in sorted(results.items()):
        if name.startswith("bh/"):
            power = "n/a" if r["power"] is None else f"{r['power']:.3f}"
            lines.append(
                f"{name:40} FDR {r['fdr']['rate']:.3f} ± {r['fdr']['mc_se']:.3f}"
                f"  any false {r['any_false_rejection']['rate']:.3f}"
                f"  power {power}  rejections {r['mean_rejections']:.2f}"
            )
        elif name.startswith("median_ci/"):
            lines.append(
                f"{name:40} coverage {r['coverage']['rate']:.3f} ± {r['coverage']['mc_se']:.3f}"
                f"  lower>0 at mu=0 {r['lower_above_0']['rate']:.3f}"
            )
        elif name.startswith("h003/"):
            lines.append(
                f"{name:40} accept {r['accept']['rate']:.3f}  reject {r['reject']['rate']:.3f}"
                f"  extend {r['extend']['rate']:.3f}  counted pairs {r['mean_counted_pairs']:.1f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Synthetic lab for the statistical methods.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("dataset")
    c.add_argument("--universe", default="configs/universe_U001.json")
    c.add_argument("--start", default="2026-08-01")
    c.add_argument("--end", default=EXPLORATION_END)
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("calibration")
    r.add_argument("--reps", type=int, default=400)
    r.add_argument("--resamples", type=int, default=10_000)
    r.add_argument("--seed", type=int, default=20261124)
    r.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "calibrate":
        if a.end > EXPLORATION_END:
            print(f"refused: calibration stays on exploration data (end <= {EXPLORATION_END})")
            return 2
        uni = json.loads(Path(a.universe).read_text())
        pairs = [x for s in sorted(uni["research"], key=int) for x in uni["research"][s]]
        cal = calibrate_dataset(Path(a.dataset), pairs, a.start, a.end)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(cal, indent=2) + "\n")
        for name, s in cal["subsets"].items():
            n_cal = sum(1 for e in s["pairs"].values() if e["profile"])
            print(
                f"{name}: {n_cal} pairs calibrated, cross-pair correlation {s['rho']:.3f}, "
                f"median day AR {s['common_ar']:.3f}"
            )
        return 0
    cal = json.loads(Path(a.calibration).read_text())
    tasks = plan(cal, a.reps, a.resamples, a.seed)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=ctx) as ex:
        results = dict(ex.map(_run, tasks))
    doc = {
        "calibration": a.calibration,
        "reps": a.reps,
        "resamples": a.resamples,
        "seed": a.seed,
        "results": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
