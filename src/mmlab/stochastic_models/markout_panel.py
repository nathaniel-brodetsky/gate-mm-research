"""Generative model of daily markout panels with known truth (synthetic lab).

One pair over n days. Each day d has a trade count N_d (Poisson with a gamma-distributed intensity, so
counts are over-dispersed like real ones) and a daily effect u_d (stationary AR(1)). Trade i on day d
has quote value v_i (lognormal) and markout

    x_i = theta + u_d + e_i,    e_i ~ Student-t(df), scaled to standard deviation trade_sd.

The volume-weighted mean of x over a long sample is theta, the quantity H-001 and H-003 test (theta is
the markout net of the fee). Daily sums (sum v x, sum v) are what the day bootstrap resamples.

Pairs can share a market-wide daily factor: u_d = day_sd (sqrt(rho) c_d + sqrt(1 - rho) o_d), with c and
o unit-variance AR(1) series.

Days with more than EXACT_MAX trades are summed with a normal approximation (central limit theorem for the
two sums); below that every trade is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

EXACT_MAX = 5_000


@dataclass(frozen=True)
class PairProfile:
    trades_per_day: float  # mean daily trade count
    count_k: float | None  # gamma shape of the daily intensity; None = Poisson counts
    logv_mean: float  # mean of log quote value
    logv_sd: float  # standard deviation of log quote value
    trade_sd: float  # standard deviation of the per-trade markout noise
    trade_df: float  # Student-t degrees of freedom of that noise (> 2); >= 1000 means normal
    day_sd: float  # standard deviation of the daily effect
    day_ar: float  # AR(1) coefficient of the daily effect, in (-1, 1)


def ar1(n: int, ar: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Stationary AR(1) with unit variance."""
    if not -1 < ar < 1:
        raise ValueError("ar must be in (-1, 1)")
    z = rng.standard_normal(n)
    out = np.empty(n)
    if n == 0:
        return out
    out[0] = z[0]
    s = np.sqrt(1 - ar * ar)
    for t in range(1, n):
        out[t] = ar * out[t - 1] + s * z[t]
    return out


def daily_counts(p: PairProfile, n_days: int, rng: np.random.Generator) -> NDArray[np.int64]:
    lam = np.full(n_days, p.trades_per_day)
    if p.count_k is not None:
        lam = lam * rng.gamma(p.count_k, 1.0 / p.count_k, n_days)
    out: NDArray[np.int64] = rng.poisson(lam).astype(np.int64)
    return out


def _noise(n: int, p: PairProfile, rng: np.random.Generator) -> NDArray[np.float64]:
    if p.trade_df >= 1000:
        out: NDArray[np.float64] = p.trade_sd * rng.standard_normal(n)
    else:
        out = p.trade_sd * np.sqrt((p.trade_df - 2) / p.trade_df) * rng.standard_t(p.trade_df, n)
    return out


def simulate_days(
    p: PairProfile,
    theta: float,
    n_days: int,
    rng: np.random.Generator,
    common: NDArray[np.float64] | None = None,
    rho: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Daily (sum v x, sum v, trade count) for one pair; days without trades have zeros."""
    own = ar1(n_days, p.day_ar, rng)
    unit = own if common is None else np.sqrt(rho) * common + np.sqrt(1 - rho) * own
    u = p.day_sd * unit
    counts = daily_counts(p, n_days, rng)
    num, den = np.zeros(n_days), np.zeros(n_days)
    ev = np.exp(p.logv_mean + p.logv_sd**2 / 2)
    ev2 = np.exp(2 * p.logv_mean + 2 * p.logv_sd**2)
    for d in range(n_days):
        n = int(counts[d])
        if n == 0:
            continue
        if n <= EXACT_MAX:
            v = rng.lognormal(p.logv_mean, p.logv_sd, n)
            den[d] = v.sum()
            num[d] = (v * (theta + u[d] + _noise(n, p, rng))).sum()
        else:
            den[d] = max(n * ev + np.sqrt(n * (ev2 - ev * ev)) * rng.standard_normal(), 1e-12)
            num[d] = (theta + u[d]) * den[d] + p.trade_sd * np.sqrt(n * ev2) * rng.standard_normal()
    return num, den, counts


def simulate_trades(
    p: PairProfile, theta: float, n_days: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Trade-level draws (day index, quote value, markout); every trade drawn, for calibration tests."""
    u = p.day_sd * ar1(n_days, p.day_ar, rng)
    counts = daily_counts(p, n_days, rng)
    day = np.repeat(np.arange(n_days), counts)
    v = rng.lognormal(p.logv_mean, p.logv_sd, day.size)
    x = theta + u[day] + _noise(day.size, p, rng)
    return day, v, x


@dataclass(frozen=True)
class EmpiricalPool:
    """Resampling generator: real trades of one pair, with the truth set by construction.

    v, r: quote value and within-day residual (markout minus the day's volume-weighted mean, scaled by
    sqrt(n / (n - 1)) for a day with n trades) of every trade on days with at least two trades. Within a
    day the volume-weighted residuals sum to 0, so over the pool sum(v r) = 0 and the volume-weighted
    mean of theta + u_d + r is theta. day_counts: real trade counts of every calendar day (zeros
    included). day_sd, day_ar: the daily effect, estimated net of the trade noise.
    """

    v: NDArray[np.float64]
    r: NDArray[np.float64]
    day_counts: NDArray[np.int64]
    day_sd: float
    day_ar: float


def simulate_days_empirical(
    pool: EmpiricalPool,
    theta: float,
    n_days: int,
    rng: np.random.Generator,
    common: NDArray[np.float64] | None = None,
    rho: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Daily (sum v x, sum v, trade count): counts drawn from the real days, trades drawn with their size."""
    own = ar1(n_days, pool.day_ar, rng)
    unit = own if common is None else np.sqrt(rho) * common + np.sqrt(1 - rho) * own
    u = pool.day_sd * unit
    counts = rng.choice(pool.day_counts, n_days).astype(np.int64)
    if pool.v.size == 0:
        counts[:] = 0
    day = np.repeat(np.arange(n_days), counts)
    idx = rng.integers(0, max(pool.v.size, 1), day.size)
    v = pool.v[idx] if pool.v.size else np.zeros(0)
    x = theta + u[day] + (pool.r[idx] if pool.r.size else np.zeros(0))
    num = np.bincount(day, weights=v * x, minlength=n_days)
    den = np.bincount(day, weights=v, minlength=n_days)
    return num, den, counts


@dataclass(frozen=True)
class DayPool:
    """Calendar-day generator: the real days of one pair, shifted so that the truth is a chosen theta.

    num_c = sum over the day of v (x - theta_hat), den = sum of v, counts = trades, for every calendar day
    of the calibration period (zeros on days without trades); sum(num_c) = 0. A synthetic period draws
    real days in blocks of consecutive days (mean length `block`) and returns num_c + theta den, so every
    within-day structure (both sides of the book hedging each other, sizes, tails, empty days) is real
    and the volume-weighted mean over the drawn days has expectation theta.
    """

    num_c: NDArray[np.float64]
    den: NDArray[np.float64]
    counts: NDArray[np.int64]
    block: float


def day_indices(n_src: int, n_days: int, block: float, rng: np.random.Generator) -> NDArray[np.int64]:
    """Stationary-bootstrap draw of n_days indices into n_src calendar days (circular, mean block)."""
    out = np.empty(n_days, dtype=np.int64)
    if n_days == 0:
        return out
    out[0] = rng.integers(0, n_src)
    jump = rng.random(n_days) < 1.0 / max(block, 1.0)
    fresh = rng.integers(0, n_src, n_days)
    for t in range(1, n_days):
        out[t] = fresh[t] if jump[t] else (out[t - 1] + 1) % n_src
    return out


def simulate_days_calendar(
    pool: DayPool,
    theta: float,
    n_days: int,
    rng: np.random.Generator,
    days: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Daily (sum v x, sum v, trade count) of real days; pass `days` to share the draw across pairs."""
    idx = day_indices(pool.den.size, n_days, pool.block, rng) if days is None else days
    den = pool.den[idx]
    return pool.num_c[idx] + theta * den, den, pool.counts[idx]
