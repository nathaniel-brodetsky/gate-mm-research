"""One-sided tests of theta = sum(num) / sum(den) > 0 over daily sums, with 95% intervals.

The synthetic lab (2026-11-24, results/lab/checks.json) found the registered day bootstrap
(stationary blocks of mean 3 days, percentile interval) too liberal on data like ours: at a true theta of
0 it rejected at the 5% level in 9.5% (30 days) and 7.9% (61 days) of months for the median pair. The
candidates below are compared on the same synthetic months in stage 3, step 2:

  percentile     stationary bootstrap of the ratio, mean block 3, percentile interval (as registered)
  percentile_pw  the same, mean block chosen from the data (Politis and White)
  student        bootstrap-t: stationary bootstrap (mean block 3) of the ratio studentized with a
                 Newey-West standard error
  student_pw     bootstrap-t with the Politis-White block
  hac            t statistic with a Newey-West standard error and Student t(n - 1) quantiles

The ratio is linearized as z_d = (num_d - theta_hat den_d) / mean(den), so theta_hat - theta ~ mean(z).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices

Array = NDArray[np.float64]
MEAN_BLOCK = 3.0


@dataclass(frozen=True)
class TestResult:
    lower: float
    upper: float
    p: float  # one-sided p-value of H0: theta <= 0


def nw_lag(n: int) -> int:
    """Newey-West bandwidth floor(4 (n / 100)^(2/9)), at least 1."""
    return max(1, int(4 * float(n / 100) ** (2 / 9)))


def nw_variance(z: Array, lag: int) -> Array:
    """Long-run variance with Bartlett weights along the last axis (z has mean ~0 per row)."""
    z = np.atleast_2d(z)
    n = z.shape[-1]
    z = z - z.mean(axis=-1, keepdims=True)
    v = (z * z).sum(axis=-1) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        v = v + 2 * w * (z[..., k:] * z[..., :-k]).sum(axis=-1) / n
    out: Array = np.maximum(v, 0.0)
    return out


def politis_white_block(x: Array) -> float:
    """Mean block length for the stationary bootstrap (Politis and White 2004, corrected by Patton,
    Politis and White 2009; the rule implemented in the arch package), between 1 and min(3 sqrt(n), n/3)."""
    n = x.size
    eps = x - x.mean()
    b_max = math.ceil(min(3 * math.sqrt(n), n / 3))
    kn = max(5, int(math.log10(n)))
    m_max = min(math.ceil(math.sqrt(n)) + kn, n - 2)
    cv = 2 * math.sqrt(math.log10(n) / n)
    acv = np.zeros(m_max + 1)
    acorr = np.zeros(m_max + 1)
    opt_m: int | None = None
    for i in range(m_max + 1):
        cross = float(eps[i:] @ eps[: n - i])
        acv[i] = cross / n
        v1, v2 = float(eps[i + 1 :] @ eps[i + 1 :]), float(eps[: -(i + 1)] @ eps[: -(i + 1)])
        acorr[i] = abs(cross) / math.sqrt(v1 * v2) if v1 > 0 and v2 > 0 else 0.0
        if i >= kn and opt_m is None and np.all(acorr[i - kn : i] < cv):
            opt_m = i - kn
    m = min(2 * max(opt_m, 1) if opt_m is not None else m_max, m_max)
    g, lr = 0.0, float(acv[0])
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2 * (1 - k / m)
        g += 2 * lam * k * float(acv[k])
        lr += 2 * lam * float(acv[k])
    if lr <= 0:
        return 1.0
    b = (2 * g * g / (2 * lr * lr)) ** (1 / 3) * n ** (1 / 3)
    return float(min(max(b, 1.0), b_max))


def _linearized(num: Array, den: Array, theta: float) -> Array:
    out: Array = (num - theta * den) / den.mean()
    return out


def percentile(num: Array, den: Array, resamples: int, seed: int, block: float = MEAN_BLOCK) -> TestResult:
    boot = ratio_resamples(num, den, stationary_indices(num.size, resamples, block, seed))
    boot = boot[~np.isnan(boot)]
    p = float((1 + np.sum(boot <= 0)) / (boot.size + 1))
    return TestResult(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), p)


def percentile_pw(num: Array, den: Array, resamples: int, seed: int) -> TestResult:
    theta = float(num.sum() / den.sum())
    return percentile(num, den, resamples, seed, politis_white_block(_linearized(num, den, theta)))


def student(num: Array, den: Array, resamples: int, seed: int, block: float = MEAN_BLOCK) -> TestResult:
    n = num.size
    theta = float(num.sum() / den.sum())
    lag = nw_lag(n)
    se = math.sqrt(float(nw_variance(_linearized(num, den, theta), lag)[0]) / n)
    idx = stationary_indices(n, resamples, block, seed)
    nb, db = num[idx], den[idx]
    th_b = nb.sum(axis=1) / db.sum(axis=1)
    z_b = (nb - th_b[:, None] * db) / db.mean(axis=1, keepdims=True)
    se_b = np.sqrt(nw_variance(z_b, lag) / n)
    ok = se_b > 0
    t_b = (th_b[ok] - theta) / se_b[ok]
    if se == 0 or t_b.size == 0:
        return TestResult(theta, theta, 1.0 if theta <= 0 else 1 / (resamples + 1))
    t_obs = theta / se
    p = float((1 + np.sum(t_b >= t_obs)) / (t_b.size + 1))
    return TestResult(
        theta - float(np.percentile(t_b, 97.5)) * se, theta - float(np.percentile(t_b, 2.5)) * se, p
    )


def student_pw(num: Array, den: Array, resamples: int, seed: int) -> TestResult:
    theta = float(num.sum() / den.sum())
    return student(num, den, resamples, seed, politis_white_block(_linearized(num, den, theta)))


def hac(num: Array, den: Array, resamples: int = 0, seed: int = 0) -> TestResult:
    """resamples and seed are unused; the signature matches the bootstrap tests."""
    n = num.size
    theta = float(num.sum() / den.sum())
    se = math.sqrt(float(nw_variance(_linearized(num, den, theta), nw_lag(n))[0]) / n)
    if se == 0:
        return TestResult(theta, theta, 1.0 if theta <= 0 else 0.0)
    q = float(stats.t.ppf(0.975, n - 1))
    p = float(stats.t.sf(theta / se, n - 1))
    return TestResult(theta - q * se, theta + q * se, p)


METHODS: dict[str, Callable[[Array, Array, int, int], TestResult]] = {
    "percentile": percentile,
    "percentile_pw": percentile_pw,
    "student": student,
    "student_pw": student_pw,
    "hac": hac,
}
