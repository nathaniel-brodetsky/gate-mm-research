"""Synthetic lab: do topological features see the regimes that matter to a maker better than simple stats?

Stage 3, step 3b (docs/topology-plan.md). A mid price on a tick grid, one value per second, switches
between two regimes in a Markov chain (exponential durations, mean 900 s). Scenarios:

- volatility: the 1-s changes are 1.5 times as volatile in regime 1 (positive control for the baselines);
- reversion: each 1-s change loads -0.3 (regime 0) or +0.3 (regime 1) on the EMA (half-life 30 s) of
  the past changes, with the same 1-s variance in both: moves revert or continue;
- jumps: in regime 1, 30% of the 1-s variance comes from jumps in 0.5% of the seconds;
- cycles: in regime 1, 10% of the 1-s variance comes from a 40-s cycle (positive control for TDA).

The strengths were set on the baselines alone (`--baselines-only`, seed 999), before any topological
feature was computed, so that the baselines reach an AUC of about 0.6-0.85 and leave room for a gain.

Every 20 s after a 300-s warm-up, features of the trailing 60-s and 300-s windows (61 values; the
300-s window is embedded at 5-s steps) are computed; the label is the regime now. Seven baselines
(`mmlab.feature_engineering.window_stats`) and seven topological features (`mmlab.topology.persistence`):
three of the swing diagram of the path, four of the Rips diagram of the delay embedding (dimension 3)
of the detrended, standardized window. Paths with even numbers fit logistic models, odd ones measure
their AUC: baselines alone, topology alone, both. The bootstrap over test paths gives a 99% interval
for the gain of both over the baselines.

The rule (written before the run): topology goes on to the archive only if, in the reversion or the
jumps scenario, at either window, the gain is at least 0.02 and its 99% interval lies above 0.

Usage:
    python -m mmlab.topology.regime_lab --workers 8 --out results/topology/regime_lab.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from mmlab.feature_engineering import window_stats as ws
from mmlab.topology import persistence as ph

F = NDArray[np.float64]
Ints = NDArray[np.int64]

SCENARIOS = ("volatility", "reversion", "jumps", "cycles")
DECIDING = ("reversion", "jumps")
SIGMA = 0.35  # standard deviation of the latent 1-s change, in ticks
MEAN_DURATION_S = 900.0
LENGTH_S = 14_400
WARMUP_S = 300
EVERY_S = 20
WINDOWS = (60, 300)
EMBED_STEP_S = {60: 1, 300: 5}
VOL_FACTOR = 1.5
PHI = 0.3
EMA_HALF_LIFE_S = 30.0
JUMP_PROB = 0.005
JUMP_SHARE = 0.3
CYCLE_S = 40.0
CYCLE_SHARE = 0.1
SWING_TICKS = 2.0
BASE = (
    "realized_vol",
    "range",
    "variance_ratio",
    "block_autocorr",
    "efficiency",
    "max_share",
    "change_share",
)
TOPO = (
    "swings_2_ticks",
    "swing_entropy",
    "swing_landscape_l1",
    "rips_h0_total",
    "rips_h1_total",
    "rips_h1_longest",
    "rips_h1_landscape_l1",
)
FEATURES = BASE + TOPO
L2 = 1.0
MIN_GAIN = 0.02
LEVEL = 0.99
BOOTS = 1000
BINS_S = (0, 30, 60, 120, 300)
NO_SWITCH = 10**9


def regimes(rng: np.random.Generator, n: int, mean_s: float = MEAN_DURATION_S) -> NDArray[np.int8]:
    """Regime (0 or 1) of every second: alternating spells of exponential length (at least 1 s)."""
    out = np.empty(n, dtype=np.int8)
    t, s = 0, int(rng.integers(2))
    while t < n:
        d = max(1, round(float(rng.exponential(mean_s))))
        out[t : t + d] = s
        t += d
        s = 1 - s
    return out


def ema_innovation_sd(phi: float, half_life_s: float, sigma: float) -> float:
    """Innovation sd for which changes dp_t = e_t + phi * ema_{t-1} have variance sigma^2.

    The EMA of the changes, ema_t = lam * ema_{t-1} + (1 - lam) * dp_t, is then AR(1) with coefficient
    rho = lam + (1 - lam) * phi, so var(dp) = var(e) * (1 + phi^2 (1 - lam)^2 / (1 - rho^2)).
    """
    lam = 0.5 ** (1.0 / half_life_s)
    rho = lam + (1.0 - lam) * phi
    return sigma / math.sqrt(1.0 + phi * phi * (1.0 - lam) ** 2 / (1.0 - rho * rho))


def latent_changes(rng: np.random.Generator, scenario: str, reg: NDArray[np.int8], sigma: float = SIGMA) -> F:
    """1-s changes of the latent price; the same variance in both regimes except in `volatility`."""
    n = len(reg)
    one = reg == 1
    z = rng.standard_normal(n)
    if scenario == "volatility":
        return np.asarray(z * np.where(one, sigma * VOL_FACTOR, sigma), dtype=np.float64)
    if scenario == "reversion":
        lam = 0.5 ** (1.0 / EMA_HALF_LIFE_S)
        phi = np.where(one, PHI, -PHI)
        sd = np.where(
            one,
            ema_innovation_sd(PHI, EMA_HALF_LIFE_S, sigma),
            ema_innovation_sd(-PHI, EMA_HALF_LIFE_S, sigma),
        )
        out = np.empty(n)
        ema = 0.0
        for t in range(n):
            dp = float(sd[t] * z[t] + phi[t] * ema)
            ema = lam * ema + (1.0 - lam) * dp
            out[t] = dp
        return out
    if scenario == "jumps":
        diffusion = z * np.where(one, sigma * math.sqrt(1.0 - JUMP_SHARE), sigma)
        jump = (rng.random(n) < JUMP_PROB) & one
        size = rng.standard_normal(n) * sigma * math.sqrt(JUMP_SHARE / JUMP_PROB)
        return np.asarray(diffusion + jump * size, dtype=np.float64)
    if scenario == "cycles":
        diffusion = z * np.where(one, sigma * math.sqrt(1.0 - CYCLE_SHARE), sigma)
        # a sine with increments of variance 2 amp^2 sin^2(pi / period) = CYCLE_SHARE * sigma^2
        amp = sigma * math.sqrt(CYCLE_SHARE / 2.0) / math.sin(math.pi / CYCLE_S)
        level = cycle_level(rng, one, amp, CYCLE_S)
        return np.asarray(diffusion + np.diff(np.concatenate([[0.0], level])), dtype=np.float64)
    raise ValueError(f"unknown scenario: {scenario!r}")


def cycle_level(rng: np.random.Generator, on: NDArray[np.bool_], amp: float, period_s: float) -> F:
    """A cycle's contribution to the price: a sine during each spell of `on`, from a random phase and
    without a jump at the start; frozen at its last value between spells."""
    n = len(on)
    omega = 2.0 * math.pi / period_s
    level = np.zeros(n)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int64), [0]])))
    last, done = 0.0, 0
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        level[done:a] = last
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        k = np.arange(b - a)
        level[a:b] = last + amp * (np.sin(omega * k + theta) - math.sin(theta))
        last, done = float(level[b - 1]), int(b)
    level[done:] = last
    return level


def mid_path(rng: np.random.Generator, scenario: str, reg: NDArray[np.int8]) -> F:
    """Observed mid in ticks: the latent price, started at a random offset, rounded to the tick."""
    start = float(rng.uniform(-0.5, 0.5))
    return np.asarray(np.round(start + np.cumsum(latent_changes(rng, scenario, reg))), dtype=np.float64)


def since_switch(reg: NDArray[np.int8]) -> Ints:
    """Seconds since the regime last changed (NO_SWITCH before the first change)."""
    idx = np.arange(len(reg))
    change = np.concatenate([[False], reg[1:] != reg[:-1]])
    last = np.maximum.accumulate(np.where(change, idx, -1))
    return np.where(last >= 0, idx - last, NO_SWITCH).astype(np.int64)


def detrend_standardize(y: F) -> F:
    """Residuals from a least-squares line divided by their standard deviation; zeros if they vanish."""
    n = len(y)
    if n < 3:
        return np.zeros(n)
    x = np.arange(n, dtype=np.float64)
    a = np.stack([np.ones(n), x], axis=1)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ coef
    sd = float(np.std(r))
    return np.asarray(r / sd, dtype=np.float64) if sd > 1e-9 else np.zeros(n)


def window_features(m: F, embed_step: int, base_only: bool = False) -> list[float]:
    """The baselines and topological features of one window, in the order of FEATURES."""
    k = max(2, (len(m) - 1) // 12)
    base = [
        ws.realized_vol(m),
        ws.price_range(m),
        ws.variance_ratio(m, k),
        ws.block_autocorr(m, k),
        ws.efficiency_ratio(m),
        ws.max_share(m),
        ws.change_share(m),
    ]
    if base_only:
        return base + [math.nan] * len(TOPO)
    swings = ph.swing_bars(m)
    h0, h1 = ph.rips_bars(ph.delay_embedding(detrend_standardize(m[::embed_step]), 3, 1))
    topo = [
        float(ph.count_at_least(swings, SWING_TICKS)),
        ph.entropy(swings),
        ph.landscape_l1(swings),
        ph.total(h0),
        ph.total(h1),
        ph.longest(h1),
        ph.landscape_l1(h1),
    ]
    return base + topo


def path_task(
    args: tuple[int, int, int, bool, int, int],
) -> dict[str, Any]:
    seed, s_idx, path, base_only, length_s, every_s = args
    rng = np.random.default_rng([seed, s_idx, path])
    reg = regimes(rng, length_s)
    m = mid_path(rng, SCENARIOS[s_idx], reg)
    times = np.arange(WARMUP_S, length_s, every_s)
    feats = {
        w: np.array([window_features(m[t - w : t + 1], EMBED_STEP_S[w], base_only) for t in times])
        for w in WINDOWS
    }
    return {
        "scenario": SCENARIOS[s_idx],
        "path": path,
        "label": reg[times].astype(np.int64),
        "since": since_switch(reg)[times],
        "X": feats,
    }


def fit_logistic(x: F, y: Ints, l2: float = L2, iters: int = 100) -> tuple[F, F, F]:
    """Ridge logistic regression on standardized columns (Newton steps); returns (mean, sd, coefficients)."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    pen = np.full(z.shape[1], l2)
    pen[0] = 0.0
    beta = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -30, 30)))
        grad = z.T @ (p - y) + pen * beta
        hess = z.T @ (z * (p * (1 - p))[:, None]) + np.diag(pen) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return mu, sd, beta


def logistic_score(model: tuple[F, F, F], x: F) -> F:
    mu, sd, beta = model
    return np.asarray(beta[0] + ((x - mu) / sd) @ beta[1:], dtype=np.float64)


def auc(score: F, y: Ints) -> float:
    """Area under the ROC curve (Mann-Whitney, ties count half); NaN without both classes."""
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    r = rankdata(score)
    return float((np.sum(r[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def evaluate(
    paths: Sequence[dict[str, Any]], window: int, rng: np.random.Generator, boots: int, base_only: bool
) -> dict[str, Any]:
    """Fit on even paths, score odd paths; AUCs, the bootstrap of the gain, AUC by time since the switch."""
    train = [p for p in paths if p["path"] % 2 == 0]
    test = [p for p in paths if p["path"] % 2 == 1]
    xtr = np.vstack([p["X"][window] for p in train])
    ytr = np.concatenate([p["label"] for p in train])
    xte = np.vstack([p["X"][window] for p in test])
    yte = np.concatenate([p["label"] for p in test])
    since = np.concatenate([p["since"] for p in test])
    nb = len(BASE)
    cols = {"baselines": list(range(nb))}
    if not base_only:
        cols |= {"topology": list(range(nb, len(FEATURES))), "both": list(range(len(FEATURES)))}
    scores = {k: logistic_score(fit_logistic(xtr[:, c], ytr), xte[:, c]) for k, c in cols.items()}
    out: dict[str, Any] = {
        "train_points": len(ytr),
        "test_points": len(yte),
        "test_share_regime_1": round(float(np.mean(yte)), 4),
        "auc": {k: round(auc(s, yte), 4) for k, s in scores.items()},
        "single_feature_auc": {
            f: round(max(a, 1.0 - a), 4)
            for j, f in enumerate(FEATURES[: len(cols["baselines"]) if base_only else len(FEATURES)])
            for a in [auc(xte[:, j], yte)]
        },
    }
    bins = [*BINS_S, math.inf]
    out["auc_by_seconds_since_switch"] = [
        {"from_s": lo, "points": int(np.sum(mask))}
        | {k: round(auc(s[mask], yte[mask]), 4) for k, s in scores.items()}
        for lo, hi in itertools.pairwise(bins)
        for mask in [(since >= lo) & (since < hi)]
    ]
    if base_only:
        return out
    rows = np.split(np.arange(len(yte)), np.cumsum([len(p["label"]) for p in test])[:-1])
    gains = np.empty(boots)
    for b in range(boots):
        idx = np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        gains[b] = auc(scores["both"][idx], yte[idx]) - auc(scores["baselines"][idx], yte[idx])
    lo, hi = np.quantile(gains, [(1.0 - LEVEL) / 2.0, 1.0 - (1.0 - LEVEL) / 2.0])
    out["gain"] = round(out["auc"]["both"] - out["auc"]["baselines"], 4)
    out["gain_interval"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def decide(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    cells = [
        f"{s} {w}s"
        for s in DECIDING
        for w in WINDOWS
        if (r := results[s][str(w)])["gain"] >= MIN_GAIN and r["gain_interval"][0] > 0
    ]
    return {
        "rule": f"gain >= {MIN_GAIN}, {LEVEL:.0%} interval above 0, in {'/'.join(DECIDING)}, any window",
        "passes": bool(cells),
        "cells": cells,
    }


def lab(
    paths: int,
    seed: int,
    workers: int,
    boots: int = BOOTS,
    base_only: bool = False,
    length_s: int = LENGTH_S,
    every_s: int = EVERY_S,
) -> dict[str, Any]:
    tasks = [(seed, s, p, base_only, length_s, every_s) for s in range(len(SCENARIOS)) for p in range(paths)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(path_task, tasks, chunksize=1))
    else:
        done = [path_task(t) for t in tasks]
    rng = np.random.default_rng([seed, 99])
    results = {
        s: {
            str(w): evaluate([d for d in done if d["scenario"] == s], w, rng, boots, base_only)
            for w in WINDOWS
        }
        for s in SCENARIOS
    }
    res: dict[str, Any] = {
        "config": {
            "paths_per_scenario": paths,
            "seed": seed,
            "length_s": length_s,
            "every_s": every_s,
            "warmup_s": WARMUP_S,
            "windows_s": list(WINDOWS),
            "embed_step_s": {str(k): v for k, v in EMBED_STEP_S.items()},
            "sigma_ticks": SIGMA,
            "mean_duration_s": MEAN_DURATION_S,
            "vol_factor": VOL_FACTOR,
            "phi": PHI,
            "ema_half_life_s": EMA_HALF_LIFE_S,
            "jump_prob": JUMP_PROB,
            "jump_share": JUMP_SHARE,
            "cycle_s": CYCLE_S,
            "cycle_share": CYCLE_SHARE,
            "swing_ticks": SWING_TICKS,
            "l2": L2,
            "boots": boots,
            "baselines_only": base_only,
            "gudhi": version("gudhi"),
        },
        "results": results,
    }
    if not base_only:
        res["decision"] = decide(results)
    return res


def compact(res: dict[str, Any]) -> str:
    base_only = res["config"]["baselines_only"]
    lines = [
        "scenario    window  base   topo   both   gain   99% interval"
        if not base_only
        else "scenario    window  base"
    ]
    for s, by_w in res["results"].items():
        for w, r in by_w.items():
            a = r["auc"]
            if base_only:
                lines.append(f"{s:<11} {w:>5}s  {a['baselines']:.3f}")
            else:
                lo, hi = r["gain_interval"]
                lines.append(
                    f"{s:<11} {w:>5}s  {a['baselines']:.3f}  {a['topology']:.3f}  {a['both']:.3f}  "
                    f"{r['gain']:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
                )
    lines.append("AUC by seconds since the switch (" + ("baselines" if base_only else "baselines/both") + ")")
    ends = [*BINS_S[1:], None]
    width = 5 if base_only else 11
    labels = [f"{lo}-{hi}s" if hi else f"{lo}s+" for lo, hi in zip(BINS_S, ends, strict=True)]
    lines.append("scenario    window  " + "  ".join(f"{x:>{width}}" for x in labels))
    for s, by_w in res["results"].items():
        for w, r in by_w.items():
            cells = [
                f"{b['baselines']:>{width}.3f}" if base_only else f"{b['baselines']:.3f}/{b['both']:.3f}"
                for b in r["auc_by_seconds_since_switch"]
            ]
            lines.append(f"{s:<11} {w:>5}s  " + "  ".join(cells))
    if "decision" in res:
        d = res["decision"]
        lines.append(
            f"decision: {'PASSES' if d['passes'] else 'fails'} ({d['rule']}); cells: {d['cells'] or 'none'}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Synthetic lab: topological features against simple statistics.")
    p.add_argument("--paths", type=int, default=80, help="paths per scenario (even: fit, odd: test)")
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--boots", type=int, default=BOOTS)
    p.add_argument("--length-s", type=int, default=LENGTH_S)
    p.add_argument("--every-s", type=int, default=EVERY_S)
    p.add_argument(
        "--baselines-only", action="store_true", help="skip topology (to set the scenarios' difficulty)"
    )
    p.add_argument("--out")
    a = p.parse_args(argv)
    res = lab(a.paths, a.seed, a.workers, a.boots, a.baselines_only, a.length_s, a.every_s)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1) + "\n")
    print(compact(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
