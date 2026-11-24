"""Day tests: hand-computed pieces, agreement with the registered bootstrap, degenerate inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from mmlab.evaluation import day_tests as dt
from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices
from mmlab.evaluation.multiple_testing import benjamini_hochberg, benjamini_yekutieli


def test_newey_west_lag() -> None:
    assert [dt.nw_lag(n) for n in (5, 30, 61, 100, 1000)] == [2, 3, 3, 4, 6]


def test_newey_west_variance_by_hand() -> None:
    z = np.array([1.0, -1.0, 1.0, -1.0])
    # gamma0 = 1, gamma1 = -3/4, Bartlett weight at lag 1 with bandwidth 1: 1/2
    assert dt.nw_variance(z, 1)[0] == pytest.approx(1 + 2 * 0.5 * (-0.75))
    assert dt.nw_variance(z, 1)[0] == pytest.approx(0.25)
    rows = np.vstack([z, 2 * z])
    assert dt.nw_variance(rows, 1) == pytest.approx([0.25, 1.0])
    assert dt.nw_variance(np.array([1.0, 1.0, 1.0]), 2)[0] == 0.0


def test_politis_white_block() -> None:
    rng = np.random.default_rng(1)
    assert dt.politis_white_block(rng.standard_normal(2000)) < 2.5  # no dependence: short blocks
    x = np.zeros(2000)
    e = rng.standard_normal(2000)
    for t in range(1, 2000):
        x[t] = 0.8 * x[t - 1] + e[t]
    b = dt.politis_white_block(x)
    assert 5 < b <= min(3 * math.sqrt(2000), 2000 / 3)
    assert dt.politis_white_block(np.ones(30)) == 1.0  # no variance
    assert 1.0 <= dt.politis_white_block(rng.standard_normal(30)) <= 10  # capped at n / 3


def test_politis_white_matches_the_arch_package() -> None:
    # reference values from arch 9.0.0, optimal_block_length(x)["stationary"], same series
    rng = np.random.default_rng(1)
    x = np.zeros(500)
    e = rng.standard_normal(500)
    for t in range(1, 500):
        x[t] = 0.6 * x[t - 1] + e[t]
    assert dt.politis_white_block(x) == pytest.approx(10.15189857965761, rel=1e-12)
    assert dt.politis_white_block(np.arange(30.0)) == pytest.approx(6.662832945264733, rel=1e-12)
    assert dt.politis_white_block(np.sin(np.arange(30) / 5.0)) == pytest.approx(6.3004487460705265, rel=1e-12)
    rng41 = np.random.default_rng(41)  # AR(0.3), 61 days: sensitive to the significance threshold of the rule
    y = np.zeros(61)
    e41 = rng41.standard_normal(61)
    for t in range(1, 61):
        y[t] = 0.3 * y[t - 1] + e41[t]
    assert dt.politis_white_block(y) == pytest.approx(2.1451010888855757, rel=1e-12)
    walk = np.cumsum(np.random.default_rng(34).standard_normal(30))
    assert dt.politis_white_block(walk) == 10.0  # arch gives 10.0 too: the cap min(3 sqrt(n), n / 3)


def test_percentile_pw_uses_the_data_driven_block() -> None:
    num, den = _month(7)
    theta = num.sum() / den.sum()
    b = dt.politis_white_block((num - theta * den) / den.mean())
    assert dt.percentile_pw(num, den, 1000, 3) == dt.percentile(num, den, 1000, 3, b)


def test_student_by_hand() -> None:
    num, den = _month(8, theta=0.0005)
    n, theta = 30, num.sum() / den.sum()
    lag = dt.nw_lag(n)

    def se_of(nn: dt.Array, dd: dt.Array) -> float:
        th = nn.sum() / dd.sum()
        z = (nn - th * dd) / dd.mean()
        z = z - z.mean()
        v = (z * z).sum() / n + sum(
            2 * (1 - k / (lag + 1)) * (z[k:] * z[:-k]).sum() / n for k in range(1, lag + 1)
        )
        return math.sqrt(v / n)

    se = se_of(num, den)
    idx = stationary_indices(n, 300, 3.0, 11)
    t = np.array([(num[i].sum() / den[i].sum() - theta) / se_of(num[i], den[i]) for i in idx])
    res = dt.student(num, den, 300, 11)
    assert res.p == pytest.approx((1 + int(np.sum(t >= theta / se))) / 301)
    assert res.lower == pytest.approx(theta - np.percentile(t, 97.5) * se)
    assert res.upper == pytest.approx(theta - np.percentile(t, 2.5) * se)


def _month(seed: int, theta: float = 0.0, n: int = 30) -> tuple[dt.Array, dt.Array]:
    rng = np.random.default_rng(seed)
    den = rng.lognormal(8, 0.5, n)
    num = den * (theta + 0.002 * rng.standard_normal(n))
    return num, den


def test_percentile_is_the_registered_bootstrap() -> None:
    num, den = _month(2)
    res = dt.percentile(num, den, 2000, 99)
    boot = ratio_resamples(num, den, stationary_indices(30, 2000, 3.0, 99))
    assert res.lower == pytest.approx(np.percentile(boot, 2.5))
    assert res.upper == pytest.approx(np.percentile(boot, 97.5))
    assert res.p == pytest.approx((1 + int(np.sum(boot <= 0))) / 2001)


def test_hac_by_hand() -> None:
    num, den = _month(3, theta=0.001)
    theta = num.sum() / den.sum()
    z = (num - theta * den) / den.mean()
    se = math.sqrt(dt.nw_variance(z, 3)[0] / 30)
    res = dt.hac(num, den)
    assert res.p == pytest.approx(stats.t.sf(theta / se, 29))
    assert res.lower == pytest.approx(theta - stats.t.ppf(0.975, 29) * se)
    assert res.upper == pytest.approx(theta + stats.t.ppf(0.975, 29) * se)


@pytest.mark.parametrize("name", list(dt.METHODS))
def test_every_method_detects_a_strong_effect_and_not_a_negative_one(name: str) -> None:
    test = dt.METHODS[name]
    strong = test(*_month(4, theta=0.003), 2000, 5)
    assert strong.p < 0.01
    assert strong.lower > 0
    negative = test(*_month(4, theta=-0.003), 2000, 5)
    assert negative.p > 0.99
    assert negative.upper < 0


@pytest.mark.parametrize("name", ["student", "student_pw", "hac"])
def test_studentized_methods_handle_a_constant_ratio(name: str) -> None:
    den = np.full(10, 100.0)
    res = dt.METHODS[name](0.002 * den, den, 500, 1)
    assert res.lower == pytest.approx(0.002)
    assert res.upper == pytest.approx(0.002)
    assert res.p < 0.01
    res0 = dt.METHODS[name](-0.002 * den, den, 500, 1)
    assert res0.p == 1.0


def test_student_interval_is_wider_than_percentile_on_short_months() -> None:
    widths = {m: [] for m in ("percentile", "student")}  # type: dict[str, list[float]]
    for s in range(20):
        num, den = _month(100 + s, n=15)
        for m in widths:
            r = dt.METHODS[m](num, den, 1000, s)
            widths[m].append(r.upper - r.lower)
    assert np.median(widths["student"]) > np.median(widths["percentile"])


def test_benjamini_yekutieli_is_bh_at_a_smaller_q() -> None:
    p = [0.001, 0.01, 0.02, 0.04, 0.5]
    c = 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5
    assert benjamini_yekutieli(p, 0.10) == benjamini_hochberg(p, 0.10 / c)
    assert benjamini_yekutieli(p, 0.10) == [True, True, True, False, False]  # 0.02 <= 3 * 0.0438 / 5
    assert benjamini_hochberg(p, 0.10) == [True, True, True, True, False]  # BH also takes 0.04 <= 4 * 0.1 / 5
    assert benjamini_yekutieli([], 0.10) == []
