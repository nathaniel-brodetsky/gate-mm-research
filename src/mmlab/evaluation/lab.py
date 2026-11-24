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
import functools
import json
import math
import multiprocessing
import os
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices
from mmlab.evaluation.day_tests import METHODS, politis_white_block
from mmlab.evaluation.h003 import THRESHOLD_BP, eligible, states
from mmlab.evaluation.multiple_testing import benjamini_hochberg, benjamini_yekutieli
from mmlab.microstructure.markouts import in_period
from mmlab.stochastic_models.markout_panel import (
    DayPool,
    EmpiricalPool,
    PairProfile,
    ar1,
    day_indices,
    simulate_days,
    simulate_days_calendar,
    simulate_days_empirical,
)

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
                tiny = 1e-12 * (abs(a.mean()) + abs(b.mean()) + 1e-12)  # float noise is not variation
                if a.std() > tiny and b.std() > tiny:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cs)) if cs else 0.0


def pair_trades(
    ds: Path, pair: str, start: str, end: str
) -> dict[str, tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]]:
    """Per subset ("all", "q80"): day index from start, quote value, markout net of the fee.

    Only trades whose RS(60 s) is known are kept.
    """
    d0 = date.fromisoformat(start)
    st = in_period(
        states(
            pl.read_parquet(ds / "trades" / f"{pair}.parquet"),
            pl.read_parquet(ds / "tob" / f"{pair}.parquet"),
        ),
        start,
        end,
    )
    out = {}
    for name, sub in (("all", st), ("q80", eligible(st, THRESHOLD_BP))):
        rs = sub.filter(pl.col("rs_60s").is_not_null())
        day = np.array([(d - d0).days for d in rs["day"].to_list()], dtype=np.int64)
        out[name] = (day, rs["v"].to_numpy(), rs["rs_60s"].to_numpy() - MAKER_FEE)
    return out


def calibrate_dataset(ds: Path, pairs: Sequence[str], start: str, end: str) -> dict[str, Any]:
    n_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    out: dict[str, Any] = {"period": [start, end], "n_days": n_days, "subsets": {}}
    per: dict[str, dict[str, Any]] = {"all": {}, "q80": {}}
    daily: dict[str, list[NDArray[np.float64]]] = {"all": [], "q80": []}
    for x in pairs:
        for name, (day, v, xs) in pair_trades(ds, x, start, end).items():
            n_active = len(set(day.tolist()))
            entry: dict[str, Any] = {"trades": int(day.size), "active_days": n_active}
            if day.size >= MIN_TRADES and n_active >= MIN_DAYS:
                prof, theta, th_d = calibrate(day, v, xs, n_days)
                entry.update(profile=asdict(prof), theta=theta)
                daily[name].append(th_d)
            else:
                entry.update(profile=None, theta=float((v * xs).sum() / v.sum()) if v.size else None)
            per[name][x] = entry
    for name in per:
        ars = [e["profile"]["day_ar"] for e in per[name].values() if e["profile"]]
        out["subsets"][name] = {
            "pairs": per[name],
            "rho": mean_pairwise_corr(daily[name]),
            "common_ar": float(statistics.median(ars)) if ars else 0.0,
        }
    return out


def profiles_for(subset: dict[str, Any], n_days: int) -> dict[str, tuple[Gen, float]]:
    """Generator and theta per pair. Parametric: pairs too thin to calibrate borrow the median profile at
    their own rate. Empirical: every pair has its pool."""
    if any("pool" in e for e in subset["pairs"].values()):
        return {pair: (str(e["pool"]), float(e["theta"])) for pair, e in subset["pairs"].items()}
    known = {p: e for p, e in subset["pairs"].items() if e["profile"]}
    med = {
        f.name: statistics.median(
            e["profile"][f.name] for e in known.values() if e["profile"][f.name] is not None
        )
        for f in fields(PairProfile)
        if f.name != "count_k"
    }
    out: dict[str, tuple[Gen, float]] = {}
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


# ---------------------------------------------------------------- empirical (resampling) generator

Gen = PairProfile | str  # a parametric profile, or the path of a saved EmpiricalPool or DayPool


def build_pool(
    day: NDArray[np.int64], v: NDArray[np.float64], x: NDArray[np.float64], n_days: int
) -> tuple[EmpiricalPool, float]:
    """Pool and volume-weighted theta of one pair; the daily effect is estimated net of trade noise."""
    counts = np.bincount(day, minlength=n_days)
    theta = float((v * x).sum() / v.sum()) if v.size else 0.0
    num = np.bincount(day, weights=v * x, minlength=n_days)
    den = np.bincount(day, weights=v, minlength=n_days)
    th = np.full(n_days, np.nan)
    multi = counts >= 2
    th[den > 0] = num[den > 0] / den[den > 0]
    keep = multi[day]
    n_d = counts[day[keep]].astype(float)
    r = (x[keep] - th[day[keep]]) * np.sqrt(n_d / (n_d - 1))
    vk = v[keep]
    noise = np.bincount(day[keep], weights=vk * vk * r * r, minlength=n_days)[multi] / den[multi] ** 2
    obs = th[multi]
    day_var = float(obs.var(ddof=1)) if obs.size > 1 else 0.0
    day_sd = math.sqrt(max(0.0, day_var - float(noise.mean()))) if obs.size > 1 else 0.0
    noise_scale = float(np.sqrt(np.mean(r * r))) if r.size else 0.0
    if day_sd <= 1e-9 * (abs(theta) + noise_scale) + 1e-15:  # below float precision: no day effect
        day_sd = 0.0
    ar = 0.0
    both = multi[1:] & multi[:-1]
    if day_sd > 0 and both.sum() >= 3:
        a, b = th[1:][both] - obs.mean(), th[:-1][both] - obs.mean()
        scale = math.sqrt((a * a).sum() * (b * b).sum())
        if scale > 0:
            ar = float(np.clip((a * b).sum() / scale * day_var / day_sd**2, -0.9, 0.9))
    return EmpiricalPool(vk, r, counts.astype(np.int64), day_sd, ar), theta


def build_daypool(
    day: NDArray[np.int64], v: NDArray[np.float64], x: NDArray[np.float64], n_days: int
) -> tuple[DayPool, float]:
    """Calendar-day pool and volume-weighted theta of one pair (mean block: Politis-White on its days)."""
    theta = float((v * x).sum() / v.sum()) if v.size else 0.0
    den = np.bincount(day, weights=v, minlength=n_days)
    num_c = np.bincount(day, weights=v * (x - theta), minlength=n_days)
    counts = np.bincount(day, minlength=n_days).astype(np.int64)
    mean_den = float(den.mean())
    block = politis_white_block(num_c / mean_den) if mean_den > 0 and n_days > 10 else 1.0
    return DayPool(num_c, den, counts, block), theta


def save_pool(path: Path, pool: EmpiricalPool | DayPool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(pool, DayPool):
        np.savez(path, kind="days", num_c=pool.num_c, den=pool.den, counts=pool.counts, block=pool.block)
    else:
        np.savez(
            path,
            kind="trades",
            v=pool.v,
            r=pool.r,
            day_counts=pool.day_counts,
            day_sd=pool.day_sd,
            day_ar=pool.day_ar,
        )


@functools.lru_cache(maxsize=64)
def load_pool(path: str) -> EmpiricalPool | DayPool:
    with np.load(path) as z:
        if "kind" in z and str(z["kind"]) == "days":
            return DayPool(z["num_c"], z["den"], z["counts"], float(z["block"]))
        return EmpiricalPool(z["v"], z["r"], z["day_counts"], float(z["day_sd"]), float(z["day_ar"]))


def shared_days(gens: Sequence[Gen], n_days: int, rng: np.random.Generator) -> NDArray[np.int64] | None:
    """One draw of calendar days for every pair when the generators are day pools, else None."""
    pools = [p for p in (load_pool(g) for g in gens if isinstance(g, str)) if isinstance(p, DayPool)]
    if not pools:
        return None
    if len({p.den.size for p in pools}) > 1:
        raise ValueError("day pools must cover the same calendar days to share them")
    return day_indices(pools[0].den.size, n_days, float(statistics.median(p.block for p in pools)), rng)


def simulate(
    gen: Gen,
    theta: float,
    n_days: int,
    rng: np.random.Generator,
    common: NDArray[np.float64] | None = None,
    rho: float = 0.0,
    days: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """One synthetic period. Day pools ignore common and rho: pairs share the calendar days instead."""
    if isinstance(gen, str):
        pool = load_pool(gen)
        if isinstance(pool, DayPool):
            return simulate_days_calendar(pool, theta, n_days, rng, days)
        return simulate_days_empirical(pool, theta, n_days, rng, common, rho)
    return simulate_days(gen, theta, n_days, rng, common, rho)


def pools_dataset(
    ds: Path, pairs: Sequence[str], start: str, end: str, out_dir: Path, kind: str = "trades"
) -> dict[str, Any]:
    """Build and save the pool ("trades": EmpiricalPool, "days": DayPool) of every pair and subset;
    returns a calibration-like summary."""
    n_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    out: dict[str, Any] = {
        "period": [start, end],
        "n_days": n_days,
        "generator": "empirical",
        "kind": kind,
        "subsets": {},
    }
    per: dict[str, dict[str, Any]] = {"all": {}, "q80": {}}
    daily: dict[str, list[NDArray[np.float64]]] = {"all": [], "q80": []}
    for x in pairs:
        for name, (day, v, xs) in pair_trades(ds, x, start, end).items():
            path = out_dir / name / f"{x}.npz"
            n_active = len(set(day.tolist()))
            entry: dict[str, Any] = {"trades": int(day.size), "active_days": n_active, "pool": str(path)}
            if kind == "days":
                dpool, theta = build_daypool(day, v, xs, n_days)
                save_pool(path, dpool)
                entry.update(theta=theta, block=dpool.block)
            else:
                pool, theta = build_pool(day, v, xs, n_days)
                save_pool(path, pool)
                entry.update(
                    pool_trades=int(pool.v.size), theta=theta, day_sd=pool.day_sd, day_ar=pool.day_ar
                )
            per[name][x] = entry
            if day.size >= MIN_TRADES and n_active >= MIN_DAYS:
                th = np.full(n_days, np.nan)
                num = np.bincount(day, weights=v * xs, minlength=n_days)
                den = np.bincount(day, weights=v, minlength=n_days)
                th[den > 0] = num[den > 0] / den[den > 0]
                daily[name].append(th)
    for name in per:
        ars = [e.get("day_ar", 0.0) for e in per[name].values() if _usable(e)]
        out["subsets"][name] = {
            "pairs": per[name],
            "rho": mean_pairwise_corr(daily[name]),
            "common_ar": float(statistics.median(ars)) if ars else 0.0,
        }
    return out


def _usable(e: dict[str, Any]) -> bool:
    """A pair the lab tests on its own: calibrated (parametric) or with enough trades (empirical)."""
    if "pool" in e:
        return bool(e["trades"] >= MIN_TRADES and e["active_days"] >= MIN_DAYS)
    return bool(e["profile"])


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
    p: Gen, theta: float, n_days: int, reps: int, resamples: int, seed: int, alpha: float = 0.05
) -> dict[str, Any]:
    cover = reject = used = 0
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        num, den, _ = simulate(p, theta, n_days, rng)
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
    profiles: Sequence[Gen],
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
        days = shared_days(profiles, n_days, rng)
        ps = []
        for i, (p, t) in enumerate(zip(profiles, thetas, strict=True)):
            num, den, _ = simulate(p, t, n_days, rng, common, rho, days)
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
    profiles: Mapping[str, tuple[Gen, float]],
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
    truth = []  # true cross-pair median of the pairs counted in a period
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        common = ar1(n_days, common_ar, rng)
        days = shared_days([g for g, _ in profiles.values()], n_days, rng)
        est, true = [], []
        for p, theta in profiles.values():
            num, den, cnt = simulate(p, theta * scale, n_days, rng, common, rho, days)
            if cnt.sum() >= MIN_TRADES and (cnt > 0).sum() >= MIN_DAYS:
                est.append(num.sum() / den.sum())
                true.append(theta * scale)
        counted.append(len(est))
        if len(est) < min_pairs:
            outcomes["extend"] += 1
            continue
        truth.append(float(np.median(true)))
        lo, _ = _median_ci(np.asarray(est), resamples, rng)
        outcomes["accept" if lo > 0 else "reject"] += 1
    return {
        "reps": reps,
        "scale": scale,
        **{k: _rate(v, reps) for k, v in outcomes.items()},
        "mean_counted_pairs": float(np.mean(counted)),
        "mean_true_median": float(np.mean(truth)) if truth else None,
    }


# ---------------------------------------------------------------- realism of the generator

REALISM_STATS = ("day_sd", "day_ac1", "active_share", "day_tail", "count_cv")


def day_stats(
    num: NDArray[np.float64], den: NDArray[np.float64], counts: NDArray[np.int64]
) -> dict[str, float]:
    """Statistics of a month of daily sums that the generator should reproduce.

    day_sd: standard deviation of daily theta; day_ac1: its lag-1 correlation over consecutive days with
    trades; active_share: share of days with trades; day_tail: (q95 - q05) / (q75 - q25) of daily theta
    (about 2.4 for a normal distribution, larger with heavy tails); count_cv: coefficient of variation of
    daily trade counts. nan where undefined.
    """
    active = den > 0
    th = np.full(den.size, np.nan)
    th[active] = num[active] / den[active]
    obs = th[active]
    out = {
        "day_sd": float(obs.std(ddof=1)) if obs.size > 1 else math.nan,
        "active_share": float(active.mean()),
    }
    both = active[1:] & active[:-1]
    a, b = th[1:][both], th[:-1][both]
    out["day_ac1"] = (
        float(np.corrcoef(a, b)[0, 1]) if both.sum() >= 3 and a.std() > 0 and b.std() > 0 else math.nan
    )
    if obs.size >= 8:
        q05, q25, q75, q95 = np.percentile(obs, [5, 25, 75, 95])
        out["day_tail"] = float((q95 - q05) / (q75 - q25)) if q75 > q25 else math.nan
    else:
        out["day_tail"] = math.nan
    c = counts.astype(float)
    out["count_cv"] = float(c.std() / c.mean()) if c.mean() > 0 else math.nan
    return out


def _mid_rank(sims: Sequence[float], real: float) -> float:
    """Share of synthetic values below the real one, ties counted half (so 1.0 vs all 1.0 gives 0.5)."""
    if not sims or math.isnan(real):
        return math.nan
    a = np.asarray(sims)
    return float((np.sum(a < real) + 0.5 * np.sum(a == real)) / a.size)


def realism_check(
    day: NDArray[np.int64],
    v: NDArray[np.float64],
    x: NDArray[np.float64],
    n_days: int,
    profile: Gen,
    theta: float,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """Rank of each real statistic among `reps` synthetic periods of the same length (0..1; nan if undefined).

    A generator that fits gives ranks spread evenly between 0 and 1 across pairs; ranks piling up near 0 or
    1 show what it gets wrong.
    """
    real = day_stats(
        np.bincount(day, weights=v * x, minlength=n_days),
        np.bincount(day, weights=v, minlength=n_days),
        np.bincount(day, minlength=n_days),
    )
    sims: dict[str, list[float]] = {k: [] for k in REALISM_STATS}
    for r in range(reps):
        st = day_stats(*simulate(profile, theta, n_days, np.random.default_rng([seed, r])))
        for k in REALISM_STATS:
            if not math.isnan(st[k]):
                sims[k].append(st[k])
    ranks = {k: _mid_rank(sims[k], real[k]) for k in REALISM_STATS}
    return {
        "real": real,
        "rank": ranks,
        "sim_median": {k: float(np.median(sims[k])) if sims[k] else math.nan for k in REALISM_STATS},
    }


def _realism_pair(ds: Path, cal: dict[str, Any], pair: str, reps: int, seed: int) -> dict[str, Any]:
    start, end = cal["period"]
    n_days = int(cal["n_days"])
    trades = pair_trades(ds, pair, start, end)
    out = {}
    for name in ("all", "q80"):
        e = cal["subsets"][name]["pairs"][pair]
        if _usable(e):
            gen: Gen = str(e["pool"]) if "pool" in e else PairProfile(**e["profile"])
            day, v, x = trades[name]
            out[f"{name}/{pair}"] = realism_check(day, v, x, n_days, gen, float(e["theta"]), reps, seed)
    return out


def realism_dataset(ds: Path, cal: dict[str, Any], reps: int, seed: int, workers: int = 1) -> dict[str, Any]:
    pairs = list(cal["subsets"]["all"]["pairs"])
    args = [(ds, cal, pair, reps, seed * 1000 + i) for i, pair in enumerate(pairs)]
    out: dict[str, Any] = {}
    if workers <= 1:
        parts = [_realism_pair(*a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            parts = list(ex.map(_realism_pair, *zip(*args, strict=True)))
    for part in parts:
        out.update(part)
    return out


def summarize_realism(res: dict[str, Any]) -> str:
    lines = [
        "rank of the real statistic among synthetic periods: 0.5 = typical; below 0.025 or above 0.975 ="
        " the generator misses it"
    ]
    for name in ("all", "q80"):
        items = {k.split("/", 1)[1]: v for k, v in res.items() if k.startswith(name + "/")}
        if not items:
            continue
        lines.append(f"{name}: {len(items)} pairs")
        for st in REALISM_STATS:
            ranks = [v["rank"][st] for v in items.values() if not math.isnan(v["rank"][st])]
            low = [p for p, v in items.items() if v["rank"][st] < 0.025]
            high = [p for p, v in items.items() if v["rank"][st] > 0.975]
            med = statistics.median(ranks) if ranks else math.nan
            lines.append(
                f"  {st:13} median rank {med:.2f}  below 2.5%: {len(low):2}  above 97.5%: {len(high):2}"
                + (f"  (low: {', '.join(low[:4])})" if low else "")
                + (f"  (high: {', '.join(high[:4])})" if high else "")
            )
    return "\n".join(lines)


# ---------------------------------------------------------------- method comparison (stage 3, step 2)


def comparison_check(
    p: Gen, theta: float, n_days: int, reps: int, resamples: int, seed: int, alpha: float = 0.05
) -> dict[str, Any]:
    """Every method of day_tests on the same synthetic months: coverage and rejection rate at alpha."""
    cover = dict.fromkeys(METHODS, 0)
    reject = dict.fromkeys(METHODS, 0)
    used = 0
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        num, den, _ = simulate(p, theta, n_days, rng)
        keep = den > 0
        if keep.sum() < MIN_DAYS:
            continue
        used += 1
        for name, test in METHODS.items():
            res = test(num[keep], den[keep], resamples, seed * 100_003 + r)
            cover[name] += res.lower <= theta <= res.upper
            reject[name] += res.p <= alpha
    return {
        "reps": used,
        "theta": theta,
        "coverage": {m: _rate(cover[m], used) for m in METHODS},
        "reject_at_alpha": {m: _rate(reject[m], used) for m in METHODS},
    }


def bh_compare_check(
    profiles: Sequence[Gen],
    thetas: Sequence[float],
    n_days: int,
    rho: float,
    common_ar: float,
    reps: int,
    resamples: int,
    seed: int,
    q: float = 0.10,
) -> dict[str, Any]:
    """Per method and procedure (BH, BY): sums over repetitions, merged by merge_bh()."""
    null = np.array([t <= 0 for t in thetas])
    keys = [f"{m}/{proc}" for m in METHODS for proc in ("bh", "by")]
    out: dict[str, Any] = {
        "reps": 0,
        "fdp": dict.fromkeys(keys, 0.0),
        "any_false": dict.fromkeys(keys, 0),
        "power": dict.fromkeys(keys, 0.0),
    }
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        common = ar1(n_days, common_ar, rng)
        days = shared_days(profiles, n_days, rng)
        ps: dict[str, list[float]] = {m: [] for m in METHODS}
        for i, (p, t) in enumerate(zip(profiles, thetas, strict=True)):
            num, den, _ = simulate(p, t, n_days, rng, common, rho, days)
            keep = den > 0
            for m, test in METHODS.items():
                ps[m].append(
                    test(num[keep], den[keep], resamples, seed * 100_003 + r * 101 + i).p
                    if keep.sum() >= MIN_DAYS
                    else 1.0
                )
        out["reps"] += 1
        for m in METHODS:
            for proc, fn in (("bh", benjamini_hochberg), ("by", benjamini_yekutieli)):
                rej = np.array(fn(ps[m], q))
                v, total = int((rej & null).sum()), int(rej.sum())
                key = f"{m}/{proc}"
                out["fdp"][key] += v / max(total, 1)
                out["any_false"][key] += v > 0
                if (~null).any():
                    out["power"][key] += float(rej[~null].mean())
    out["has_alternatives"] = bool((~null).any())
    return out


def merge_bh(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reps = sum(p["reps"] for p in parts)
    keys = list(parts[0]["fdp"])
    alt = parts[0]["has_alternatives"]
    return {
        "reps": reps,
        **{
            k: {
                "fdr": sum(p["fdp"][k] for p in parts) / reps,
                "any_false": sum(p["any_false"][k] for p in parts) / reps,
                "power": sum(p["power"][k] for p in parts) / reps if alt else None,
            }
            for k in keys
        },
    }


def compare_plan(
    cal: dict[str, Any], reps: int, resamples: int, seed: int, chunks: int = 8
) -> list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]]:
    n_cal = int(cal["n_days"])
    allp = profiles_for(cal["subsets"]["all"], n_cal)
    q80 = profiles_for(cal["subsets"]["q80"], n_cal)
    tasks: list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]] = []
    for subset, profs, horizons in (("all", allp, (30, 61)), ("q80", q80, (30,))):
        calibrated = [k for k, e in cal["subsets"][subset]["pairs"].items() if _usable(e)]
        for n_days in horizons:
            for theta, label in ((0.0, "size"), (BH_ALTERNATIVE, "power")):
                for pair in calibrated:
                    tasks.append(
                        (
                            f"compare/{subset}/{n_days}d/{label}/{pair}",
                            comparison_check,
                            {
                                "p": profs[pair][0],
                                "theta": theta,
                                "n_days": n_days,
                                "reps": reps,
                                "resamples": resamples,
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
            for c in range(chunks):
                tasks.append(
                    (
                        f"bhcmp/all/{n_days}d/{label}#{c}",
                        bh_compare_check,
                        {
                            "profiles": [allp[p][0] for p in pairs],
                            "thetas": thetas,
                            "n_days": n_days,
                            "rho": sub["rho"],
                            "common_ar": sub["common_ar"],
                            "reps": reps // chunks,
                            "resamples": resamples,
                        },
                    )
                )
    return [(name, fn, {**kw, "seed": seed * 10_000 + 5_000 + i}) for i, (name, fn, kw) in enumerate(tasks)]


def summarize_compare(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "method comparison: median over pairs (max in brackets); size = rejections at 5% with theta 0, "
        "power = rejections at 5% with theta +10 bp"
    ]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, res in results.items():
        if name.startswith("compare/") and res["reps"]:
            _, subset, days, label, _ = name.split("/")
            groups.setdefault(f"{subset}/{days}", {}).setdefault(label, []).append(res)
    for g in sorted(groups):
        lines.append(f"{g}:")
        for m in METHODS:
            size = [r["reject_at_alpha"][m]["rate"] for r in groups[g].get("size", [])]
            cov = [r["coverage"][m]["rate"] for r in groups[g].get("size", [])]
            power = [r["reject_at_alpha"][m]["rate"] for r in groups[g].get("power", [])]
            lines.append(
                f"  {m:14} size {statistics.median(size):.3f} ({max(size):.3f})"
                f"  coverage {statistics.median(cov):.3f} ({min(cov):.3f} min)"
                f"  power {statistics.median(power):.3f}"
            )
    for name, r in sorted(results.items()):
        if name.startswith("bhcmp/"):
            lines.append(f"{name} ({r['reps']} months):")
            for k, v in r.items():
                if k == "reps":
                    continue
                pw = "n/a" if v["power"] is None else f"{v['power']:.3f}"
                lines.append(f"  {k:22} FDR {v['fdr']:.3f}  any false {v['any_false']:.3f}  power {pw}")
    return "\n".join(lines)


def show_daypools(cal: dict[str, Any]) -> str:
    lines = []
    for name, sub in cal["subsets"].items():
        lines.append(f"{name}: calendar days of {cal['n_days']}, shared by all pairs")
        lines.append(f"  {'pair':14} {'trades':>8} {'active days':>11} {'mean block':>10} {'theta bp':>8}")
        for pair, e in sub["pairs"].items():
            lines.append(
                f"  {pair:14} {e['trades']:8d} {e['active_days']:11d} {e['block']:10.2f}"
                f" {e['theta'] * 1e4:8.1f}" + ("" if _usable(e) else "  (too thin to test on its own)")
            )
    return "\n".join(lines)


def show_pools(cal: dict[str, Any]) -> str:
    if cal.get("kind") == "days":
        return show_daypools(cal)
    lines = []
    for name, sub in cal["subsets"].items():
        lines.append(f"{name}: cross-pair correlation {sub['rho']:.3f}, median day AR {sub['common_ar']:.3f}")
        lines.append(
            f"  {'pair':14} {'trades':>8} {'in pool':>8} {'active days':>11} {'day sd bp':>9} {'day AR':>6}"
            f" {'theta bp':>8}"
        )
        for pair, e in sub["pairs"].items():
            lines.append(
                f"  {pair:14} {e['trades']:8d} {e['pool_trades']:8d} {e['active_days']:11d}"
                f" {e['day_sd'] * 1e4:9.2f} {e['day_ar']:6.2f} {e['theta'] * 1e4:8.1f}"
                + ("" if _usable(e) else "  (too thin to test on its own)")
            )
    return "\n".join(lines)


def show(cal: dict[str, Any]) -> str:
    if cal.get("generator") == "empirical":
        return show_pools(cal)
    lines = []
    for name, sub in cal["subsets"].items():
        lines.append(f"{name}: cross-pair correlation {sub['rho']:.3f}, median day AR {sub['common_ar']:.3f}")
        lines.append(
            f"  {'pair':14} {'trades':>8} {'per day':>8} {'k':>6} {'log v sd':>8} {'trade sd bp':>11}"
            f" {'df':>6} {'day sd bp':>9} {'day AR':>6} {'theta bp':>8}"
        )
        for pair, e in sub["pairs"].items():
            pr = e["profile"]
            th = "n/a" if e["theta"] is None else f"{e['theta'] * 1e4:8.1f}"
            if pr is None:
                lines.append(f"  {pair:14} {e['trades']:8d} {'(too thin to calibrate)':>51} {th:>8}")
                continue
            k = "Pois" if pr["count_k"] is None else f"{pr['count_k']:6.2f}"
            lines.append(
                f"  {pair:14} {e['trades']:8d} {pr['trades_per_day']:8.1f} {k:>6} {pr['logv_sd']:8.2f}"
                f" {pr['trade_sd'] * 1e4:11.1f} {pr['trade_df']:6.1f} {pr['day_sd'] * 1e4:9.2f}"
                f" {pr['day_ar']:6.2f} {th:>8}"
            )
    return "\n".join(lines)


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
        calibrated = [k for k, e in cal["subsets"][subset]["pairs"].items() if _usable(e)]
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
    q80_thetas = np.array([float(e["theta"]) for e in cal["subsets"]["q80"]["pairs"].values() if _usable(e)])
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


POOLED_TRUTHS = ("zero", "centred", "centred_plus10bp", "explored")


def pooled_profiles(profiles: Mapping[str, tuple[Gen, float]], truth: str) -> dict[str, tuple[Gen, float]]:
    """Pair thetas for a check of the pooled median rule.

    zero: every pair 0. centred: the exploration thetas minus their median, so the true cross-pair median
    is 0 and the pairs differ as they really do. centred_plus10bp: the same plus 10 bp. explored: the
    exploration thetas as they are.
    """
    if truth == "zero":
        return {p: (g, 0.0) for p, (g, _) in profiles.items()}
    if truth == "explored":
        return dict(profiles)
    if truth not in ("centred", "centred_plus10bp"):
        raise ValueError(f"unknown truth {truth!r}")
    med = float(np.median([t for _, t in profiles.values()]))
    add = BH_ALTERNATIVE if truth == "centred_plus10bp" else 0.0
    return {p: (g, t - med + add) for p, (g, t) in profiles.items()}


def pooled_plan(
    cal: dict[str, Any], reps: int, resamples: int, seed: int
) -> list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]]:
    """The pooled median rule (bootstrap over pairs of the median, as H-003) on both subsets, 30 and 61
    days, under each truth of POOLED_TRUTHS. Only pairs the lab tests on their own take part."""
    n_cal = int(cal["n_days"])
    tasks: list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]] = []
    for subset in ("all", "q80"):
        sub = cal["subsets"][subset]
        usable = {k for k, e in sub["pairs"].items() if _usable(e)}
        profs = {k: v for k, v in profiles_for(sub, n_cal).items() if k in usable}
        for n_days in (30, 61):
            for truth in POOLED_TRUTHS:
                tasks.append(
                    (
                        f"pooled/{subset}/{n_days}d/{truth}",
                        h003_check,
                        {
                            "profiles": pooled_profiles(profs, truth),
                            "scale": 1.0,
                            "n_days": n_days,
                            "rho": sub["rho"],
                            "common_ar": sub["common_ar"],
                            "reps": reps,
                            "resamples": resamples,
                        },
                    )
                )
    return [(name, fn, {**kw, "seed": seed * 10_000 + 7_000 + i}) for i, (name, fn, kw) in enumerate(tasks)]


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
        elif name.startswith(("h003/", "pooled/")):
            tm = r.get("mean_true_median")
            lines.append(
                f"{name:40} accept {r['accept']['rate']:.3f}  reject {r['reject']['rate']:.3f}"
                f"  extend {r['extend']['rate']:.3f}  counted pairs {r['mean_counted_pairs']:.1f}"
                + ("" if tm is None else f"  true median {tm * 1e4:+.1f} bp")
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
    rl = sub.add_parser("realism", help="compare real daily statistics with the calibrated generator")
    rl.add_argument("dataset")
    rl.add_argument("calibration")
    rl.add_argument("--reps", type=int, default=1000)
    rl.add_argument("--seed", type=int, default=20261124)
    rl.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    rl.add_argument("--out", required=True)
    po = sub.add_parser("pools", help="build a resampling generator: real trades or real days of every pair")
    po.add_argument("dataset")
    po.add_argument("--universe", default="configs/universe_U001.json")
    po.add_argument("--start", default="2026-08-01")
    po.add_argument("--end", default=EXPLORATION_END)
    po.add_argument("--dir", default="data/interim/lab_pools")
    po.add_argument("--kind", choices=["trades", "days"], default="trades")
    po.add_argument("--out", required=True)
    sh = sub.add_parser("show", help="print a calibration")
    sh.add_argument("calibration")
    cm = sub.add_parser("compare", help="compare the day tests of day_tests.METHODS")
    cm.add_argument("calibration")
    cm.add_argument("--reps", type=int, default=400)
    cm.add_argument("--resamples", type=int, default=2_000)
    cm.add_argument("--seed", type=int, default=20261124)
    cm.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    cm.add_argument("--out", required=True)
    pd = sub.add_parser("pooled", help="check the pooled median rule under several truths")
    pd.add_argument("calibration")
    pd.add_argument("--reps", type=int, default=800)
    pd.add_argument("--resamples", type=int, default=10_000)
    pd.add_argument("--seed", type=int, default=20261124)
    pd.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    pd.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "pools":
        if a.end > EXPLORATION_END:
            print(f"refused: pools stay on exploration data (end <= {EXPLORATION_END})")
            return 2
        uni = json.loads(Path(a.universe).read_text())
        pairs = [x for s in sorted(uni["research"], key=int) for x in uni["research"][s]]
        res = pools_dataset(Path(a.dataset), pairs, a.start, a.end, Path(a.dir), a.kind)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
        print(show_pools(res))
        return 0
    if a.cmd == "realism":
        cal = json.loads(Path(a.calibration).read_text())
        res = realism_dataset(Path(a.dataset), cal, a.reps, a.seed, a.workers)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(
            json.dumps(
                {"calibration": a.calibration, "reps": a.reps, "seed": a.seed, "results": res}, indent=2
            )
            + "\n"
        )
        print(summarize_realism(res))
        return 0
    if a.cmd == "show":
        print(show(json.loads(Path(a.calibration).read_text())))
        return 0
    if a.cmd == "compare":
        cal = json.loads(Path(a.calibration).read_text())
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=ctx) as ex:
            raw = dict(ex.map(_run, compare_plan(cal, a.reps, a.resamples, a.seed)))
        results: dict[str, dict[str, Any]] = {k: v for k, v in raw.items() if not k.startswith("bhcmp/")}
        bh_groups: dict[str, list[dict[str, Any]]] = {}
        for k, v in raw.items():
            if k.startswith("bhcmp/"):
                bh_groups.setdefault(k.split("#")[0], []).append(v)
        results.update({k: merge_bh(v) for k, v in bh_groups.items()})
        doc = {
            "calibration": a.calibration,
            "reps": a.reps,
            "resamples": a.resamples,
            "seed": a.seed,
            "methods": list(METHODS),
            "results": results,
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(summarize_compare(results))
        return 0
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
    tasks = (pooled_plan if a.cmd == "pooled" else plan)(cal, a.reps, a.resamples, a.seed)
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
