"""The synthetic markout generator produces what its parameters say."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mmlab.stochastic_models import markout_panel as mp

BASE = mp.PairProfile(
    trades_per_day=200.0,
    count_k=None,
    logv_mean=3.0,
    logv_sd=1.0,
    trade_sd=0.01,
    trade_df=1000.0,
    day_sd=0.0,
    day_ar=0.0,
)


def _with(**kw: object) -> mp.PairProfile:
    return dataclasses.replace(BASE, **kw)  # type: ignore[arg-type]


def test_ar1_has_unit_variance_and_the_given_autocorrelation() -> None:
    x = mp.ar1(200_000, 0.6, np.random.default_rng(1))
    assert x.var() == pytest.approx(1.0, abs=0.03)
    assert np.corrcoef(x[1:], x[:-1])[0, 1] == pytest.approx(0.6, abs=0.01)
    assert mp.ar1(0, 0.5, np.random.default_rng(1)).size == 0
    with pytest.raises(ValueError, match="ar must be"):
        mp.ar1(10, 1.0, np.random.default_rng(1))


def test_counts_poisson_and_overdispersed() -> None:
    rng = np.random.default_rng(2)
    c = mp.daily_counts(BASE, 100_000, rng)
    assert (c.mean(), c.var()) == (pytest.approx(200, rel=0.01), pytest.approx(200, rel=0.03))
    g = mp.daily_counts(_with(count_k=4.0), 100_000, rng)
    assert g.mean() == pytest.approx(200, rel=0.01)
    assert g.var() == pytest.approx(200 + 200**2 / 4, rel=0.05)  # Poisson-gamma: m + m^2 / k


def test_volume_weighted_mean_is_theta() -> None:
    num, den, cnt = mp.simulate_days(_with(day_sd=0.002, day_ar=0.5), 0.003, 3000, np.random.default_rng(3))
    assert num.sum() / den.sum() == pytest.approx(0.003, abs=1e-4)
    assert cnt.mean() == pytest.approx(200, rel=0.01)


def test_days_without_trades_are_zero() -> None:
    num, den, cnt = mp.simulate_days(_with(trades_per_day=0.2), 0.0, 500, np.random.default_rng(4))
    empty = cnt == 0
    assert empty.any()
    assert (num[empty] == 0).all()
    assert (den[empty] == 0).all()
    assert (den[~empty] > 0).all()


def test_large_days_use_the_normal_approximation_with_the_same_moments() -> None:
    big = _with(trades_per_day=4 * mp.EXACT_MAX, logv_sd=0.5, trade_sd=0.0)
    num, den, _ = mp.simulate_days(big, 0.002, 200, np.random.default_rng(5))
    ev = np.exp(3.0 + 0.5**2 / 2)
    assert den.mean() == pytest.approx(4 * mp.EXACT_MAX * ev, rel=0.01)
    assert num / den == pytest.approx(np.full(200, 0.002))  # no noise, no day effect
    moving = _with(trades_per_day=4 * mp.EXACT_MAX, trade_sd=0.0, day_sd=0.001)
    num, den, _ = mp.simulate_days(moving, 0.002, 2000, np.random.default_rng(15))
    assert (num / den).std() == pytest.approx(0.001, rel=0.05)  # the day effect survives the approximation


def test_common_factor_correlates_pairs() -> None:
    rng = np.random.default_rng(6)
    p = _with(trade_sd=0.0, day_sd=1.0, day_ar=0.2)
    a, b = [], []
    for _ in range(40):
        common = mp.ar1(200, 0.2, rng)
        na, da, _ = mp.simulate_days(p, 0.0, 200, rng, common, rho=0.5)
        nb, db, _ = mp.simulate_days(p, 0.0, 200, rng, common, rho=0.5)
        a.append(na / da)
        b.append(nb / db)
    assert np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1] == pytest.approx(0.5, abs=0.04)


def test_student_noise_has_the_given_standard_deviation() -> None:
    _, _, x = mp.simulate_trades(_with(trade_df=5.0, trade_sd=0.02), 0.0, 2000, np.random.default_rng(7))
    assert x.std() == pytest.approx(0.02, rel=0.03)
    m2, m4 = (x**2).mean(), (x**4).mean()
    assert m4 / m2**2 - 3 > 2  # heavy tails (Student-t with 5 df: excess kurtosis 6)


def test_simulate_trades_matches_days() -> None:
    day, v, x = mp.simulate_trades(_with(day_sd=0.001), 0.001, 50, np.random.default_rng(8))
    assert day.size == v.size == x.size
    assert (np.diff(day) >= 0).all()
    assert int(np.max(day)) < 50
    assert (v > 0).all()
