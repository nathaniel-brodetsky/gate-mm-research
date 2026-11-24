"""Synthetic lab: calibration recovers known parameters; checks behave as theory says on easy cases."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmlab.evaluation import day_tests, lab
from mmlab.stochastic_models import markout_panel as mp
from test_h003 import _dataset

TRUE = mp.PairProfile(
    trades_per_day=200.0,
    count_k=3.0,
    logv_mean=3.0,
    logv_sd=1.0,
    trade_sd=0.01,
    trade_df=1000.0,
    day_sd=0.002,
    day_ar=0.3,
)
IID = mp.PairProfile(50.0, None, 3.0, 1.0, 0.01, 1000.0, 0.0, 0.0)


def test_calibration_recovers_the_profile() -> None:
    day, v, x = mp.simulate_trades(TRUE, 0.001, 1500, np.random.default_rng(11))
    prof, theta, th_d = lab.calibrate(day, v, x, 1500)
    assert prof.trades_per_day == pytest.approx(200, rel=0.03)
    assert prof.count_k == pytest.approx(3.0, rel=0.15)
    assert prof.logv_mean == pytest.approx(3.0, abs=0.01)
    assert prof.logv_sd == pytest.approx(1.0, abs=0.01)
    assert prof.trade_sd == pytest.approx(0.01, rel=0.01)
    assert prof.trade_df > 100  # normal noise
    assert prof.day_sd == pytest.approx(0.002, rel=0.1)
    assert prof.day_ar == pytest.approx(0.3, abs=0.06)
    assert theta == pytest.approx(0.001, abs=3e-4)
    assert th_d.shape == (1500,)


def test_calibration_separates_poisson_from_gamma_dispersion() -> None:
    p = mp.PairProfile(5.0, 2.0, 3.0, 1.0, 0.01, 1000.0, 0.0, 0.0)  # var = 5 + 25 / 2 = 17.5
    day, v, x = mp.simulate_trades(p, 0.0, 8000, np.random.default_rng(13))
    prof, _, _ = lab.calibrate(day, v, x, 8000)
    assert prof.count_k == pytest.approx(2.0, rel=0.12)


def test_calibration_sees_heavy_tails_and_poisson_counts() -> None:
    p = mp.PairProfile(300.0, None, 2.0, 0.5, 0.01, 6.0, 0.0, 0.0)
    day, v, x = mp.simulate_trades(p, 0.0, 600, np.random.default_rng(12))
    prof, _, _ = lab.calibrate(day, v, x, 600)
    assert prof.count_k is None or prof.count_k > 50  # no over-dispersion
    assert prof.trade_df == pytest.approx(6.0, abs=2.0)
    assert prof.day_sd < 0.0005


def test_calibration_of_a_constant_series() -> None:
    day = np.repeat(np.arange(10), 5)
    prof, theta, _ = lab.calibrate(day, np.full(50, 100.0), np.full(50, 0.002), 10)
    assert (prof.trade_sd, prof.day_sd, prof.day_ar, prof.trade_df) == (0.0, 0.0, 0.0, 1000.0)
    assert theta == pytest.approx(0.002)


def test_mean_pairwise_corr() -> None:
    a = np.arange(20, dtype=float)
    b = a.copy()
    b[:3] = np.nan
    assert lab.mean_pairwise_corr([a, b]) == pytest.approx(1.0)
    assert lab.mean_pairwise_corr([a, -a, a]) == pytest.approx(-1 / 3)
    assert lab.mean_pairwise_corr([a[:5], a[:5]]) == 0.0  # fewer than 10 common days
    flat = np.full(20, 0.019) + np.arange(20) * 1e-19  # constant up to float noise
    assert lab.mean_pairwise_corr([flat, flat]) == 0.0


def test_day_bootstrap_is_calibrated_on_many_iid_days() -> None:
    # with many days the percentile interval is close to nominal; short samples are what the lab measures
    res = lab.day_bootstrap_check(IID, 0.0, 200, reps=300, resamples=2000, seed=1)
    assert res["reps"] == 300
    assert 0.91 <= res["coverage"]["rate"] <= 0.99
    assert res["reject_at_alpha"]["rate"] <= 0.09
    size = lab.day_bootstrap_check(IID, 0.0, 200, reps=600, resamples=2000, seed=21)
    assert 0.035 <= size["reject_at_alpha"]["rate"] <= 0.075  # close to the nominal 5%
    shifted = lab.day_bootstrap_check(IID, 0.002, 200, reps=100, resamples=1000, seed=22)
    assert shifted["coverage"]["rate"] >= 0.9  # covers the true theta, not 0
    strong = lab.day_bootstrap_check(IID, 0.005, 30, reps=50, resamples=1000, seed=2)
    assert strong["reject_at_alpha"]["rate"] == 1.0
    thin = lab.day_bootstrap_check(
        mp.PairProfile(0.05, None, 3.0, 1.0, 0.01, 1000.0, 0.0, 0.0), 0.0, 30, 20, 500, 3
    )
    assert thin["reps"] < 20  # months with fewer than 5 trading days are not tested


def test_bh_controls_false_discoveries_and_finds_strong_effects() -> None:
    profiles = [IID] * 10
    null = lab.bh_check(profiles, [0.0] * 10, 150, 0.3, 0.2, reps=150, resamples=1000, seed=4)
    assert null["fdr"]["rate"] <= 0.17  # q = 0.10 plus Monte Carlo error
    assert null["fdr"]["rate"] == null["any_false_rejection"]["rate"]  # all null: V / R = 1 whenever R > 0
    assert null["power"] is None
    mixed = lab.bh_check(profiles, [0.0] * 5 + [0.005] * 5, 30, 0.3, 0.2, reps=60, resamples=1000, seed=5)
    assert mixed["power"] == pytest.approx(1.0)
    assert mixed["fdr"]["rate"] <= 0.2
    assert mixed["mean_rejections"] >= 5


def test_median_ci() -> None:
    res = lab.median_ci_check(24, 0.0, 0.003, 0.001, reps=400, resamples=2000, seed=6)
    assert 0.88 <= res["coverage"]["rate"] <= 0.99
    assert res["lower_above_0"]["rate"] <= 0.06
    big = lab.median_ci_check(12, 0.01, 0.002, 0.001, reps=50, resamples=1000, seed=7)
    assert big["lower_above_0"]["rate"] == 1.0


def test_h003_rule_on_synthetic_months() -> None:
    profiles = {f"P{i}": (IID, 0.004) for i in range(12)}
    persist = lab.h003_check(profiles, 1.0, 30, 0.2, 0.2, reps=40, resamples=1000, seed=8)
    assert persist["accept"]["rate"] == 1.0
    assert persist["mean_counted_pairs"] == 12
    none = lab.h003_check(profiles, 0.0, 30, 0.2, 0.2, reps=200, resamples=1000, seed=9)
    assert none["accept"]["rate"] <= 0.15
    few = lab.h003_check(dict(list(profiles.items())[:7]), 1.0, 30, 0.2, 0.2, reps=10, resamples=500, seed=10)
    assert few["extend"]["rate"] == 1.0
    eight = lab.h003_check(
        dict(list(profiles.items())[:8]), 1.0, 30, 0.2, 0.2, reps=10, resamples=500, seed=11
    )
    assert eight["accept"]["rate"] == 1.0  # exactly the registered minimum of 8 pairs


def _cal(pairs: dict[str, dict[str, object]]) -> dict[str, object]:
    sub = {"pairs": pairs, "rho": 0.1, "common_ar": 0.2}
    return {"period": ["2026-08-01", "2026-11-01"], "n_days": 92, "subsets": {"all": sub, "q80": sub}}


def test_thin_pairs_borrow_the_median_profile() -> None:
    a = {**asdict(IID), "trades_per_day": 40.0}
    b = {**asdict(IID), "trades_per_day": 60.0, "trade_sd": 0.03}
    cal = _cal(
        {
            "A": {"trades": 3680, "profile": a, "theta": 0.001},
            "B": {"trades": 5520, "profile": b, "theta": 0.002},
            "T": {"trades": 9, "profile": None, "theta": None},
        }
    )
    out = lab.profiles_for(cal["subsets"]["all"], 92)  # type: ignore[index]
    thin, theta = out["T"]
    assert isinstance(thin, mp.PairProfile)
    assert thin.trades_per_day == pytest.approx(9 / 92)
    assert thin.trade_sd == pytest.approx(0.02)  # median of 0.01 and 0.03
    assert thin.count_k is None
    assert theta == 0.0
    assert out["A"] == (mp.PairProfile(**a), 0.001)


def test_plan_covers_every_check() -> None:
    cal = _cal({p: {"trades": 4600, "profile": asdict(IID), "theta": 0.001} for p in ("A", "B", "C")})
    tasks = lab.plan(cal, reps=10, resamples=100, seed=1)
    names = [t[0] for t in tasks]
    assert len(names) == len(set(names))
    assert sum(n.startswith("day_bootstrap/all/30d/") for n in names) == 3
    assert sum(n.startswith("day_bootstrap/all/61d/") for n in names) == 3
    assert sum(n.startswith("day_bootstrap/q80/30d/") for n in names) == 3
    assert {n for n in names if n.startswith("bh/")} == {
        "bh/all/30d/all_null",
        "bh/all/30d/half_at_plus10bp",
        "bh/all/61d/all_null",
        "bh/all/61d/half_at_plus10bp",
    }
    assert [n for n in names if n.startswith("h003/")] == [
        "h003/november/scale_1",
        "h003/november/scale_0.5",
        "h003/november/scale_0",
    ]
    seeds = [t[2]["seed"] for t in tasks]
    assert len(seeds) == len(set(seeds))
    med = dict((n, kw) for n, _, kw in tasks)["median_ci/8_pairs/mu0"]
    assert (med["tau"], med["est_sd"]) == (0.0, 0.0)  # all calibrated thetas equal here


def test_median_check_uses_the_spread_of_calibrated_pairs_only() -> None:
    pairs: dict[str, dict[str, object]] = {
        "A": {"trades": 4600, "profile": asdict(IID), "theta": 0.001},
        "B": {"trades": 4600, "profile": asdict(IID), "theta": 0.003},
        "T": {"trades": 4, "profile": None, "theta": -0.0112},  # thin and noisy: left out of the spread
    }
    tasks = lab.plan(_cal(pairs), reps=10, resamples=100, seed=1)
    med = dict((n, kw) for n, _, kw in tasks)["median_ci/8_pairs/mu0"]
    assert med["tau"] == pytest.approx(np.std([0.001, 0.003], ddof=1))
    assert med["est_sd"] == 0.0


def test_cli_calibrate_and_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "B_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["A_USDT", "B_USDT"], "2026-08-01T00", "2026-10-31T23", days=92)
    cal = tmp_path / "lab" / "cal.json"
    assert (
        lab.main(["calibrate", str(ds), "--universe", str(uni), "--out", str(cal), "--end", "2026-11-02"])
        == 2
    )
    assert lab.main(["calibrate", str(ds), "--universe", str(uni), "--out", str(cal)]) == 0
    doc = json.loads(cal.read_text())
    assert doc["n_days"] == 92
    assert doc["subsets"]["q80"]["pairs"]["A_USDT"]["trades"] == 184
    assert doc["subsets"]["all"]["pairs"]["A_USDT"]["theta"] == pytest.approx(0.019)
    assert "q80: 2 pairs calibrated" in capsys.readouterr().out
    out = tmp_path / "lab" / "checks.json"
    argv = ["run", str(cal), "--reps", "3", "--resamples", "200", "--workers", "1", "--out", str(out)]
    assert lab.main(argv) == 0
    res = json.loads(out.read_text())["results"]
    assert "h003/november/scale_1" in res
    assert res["h003/november/scale_1"]["mean_counted_pairs"] == 2
    assert "day_bootstrap/all/30d" in capsys.readouterr().out


def test_comparison_runs_every_method_on_the_same_months() -> None:
    res = lab.comparison_check(IID, 0.005, 30, reps=20, resamples=500, seed=31)
    assert res["reps"] == 20
    assert set(res["coverage"]) == set(res["reject_at_alpha"]) == set(day_tests.METHODS)
    assert all(v["rate"] == 1.0 for v in res["reject_at_alpha"].values())
    thin = lab.comparison_check(
        mp.PairProfile(0.05, None, 3.0, 1.0, 0.01, 1000.0, 0.0, 0.0), 0.0, 30, 10, 200, 32
    )
    assert thin["reps"] < 10
    sticky = mp.PairProfile(200.0, None, 3.0, 1.0, 0.0001, 1000.0, 0.003, 0.9)  # persistent day effects
    hard = lab.comparison_check(sticky, 0.0, 30, reps=60, resamples=500, seed=36)
    assert hard["coverage"]["percentile"]["rate"] < 0.9  # the short-block bootstrap misses persistence


def test_bh_compare_and_merge() -> None:
    profiles = [IID] * 6
    a = lab.bh_compare_check(profiles, [0.0] * 6, 60, 0.0, 0.0, reps=30, resamples=500, seed=33)
    b = lab.bh_compare_check(profiles, [0.0] * 6, 60, 0.0, 0.0, reps=10, resamples=500, seed=34)
    m = lab.merge_bh([a, b])
    assert m["reps"] == 40
    for key in (f"{x}/{proc}" for x in day_tests.METHODS for proc in ("bh", "by")):
        assert m[key]["fdr"] == pytest.approx((a["fdp"][key] + b["fdp"][key]) / 40)
        assert m[key]["fdr"] == m[key]["any_false"]  # all null
        assert m[key]["power"] is None
    for x in day_tests.METHODS:
        assert m[f"{x}/by"]["any_false"] <= m[f"{x}/bh"]["any_false"]  # BY rejects a subset of BH
    alt = lab.bh_compare_check(
        profiles, [0.0] * 3 + [0.005] * 3, 30, 0.0, 0.0, reps=5, resamples=500, seed=35
    )
    assert lab.merge_bh([alt])["hac/bh"]["power"] == 1.0


def test_compare_plan() -> None:
    cal = _cal({p: {"trades": 4600, "profile": asdict(IID), "theta": 0.001} for p in ("A", "B")})
    tasks = lab.compare_plan(cal, reps=16, resamples=100, seed=1, chunks=4)
    names = [t[0] for t in tasks]
    assert len(names) == len(set(names))
    assert sum(n.startswith("compare/all/30d/size/") for n in names) == 2
    assert sum(n.startswith("compare/q80/30d/power/") for n in names) == 2
    assert sum(n.startswith("bhcmp/all/61d/all_null#") for n in names) == 4
    bh = next(kw for n, _, kw in tasks if n.startswith("bhcmp/"))
    assert bh["reps"] == 4
    power = next(kw for n, _, kw in tasks if n.startswith("compare/all/30d/power/"))
    assert power["theta"] == lab.BH_ALTERNATIVE
    seeds = [kw["seed"] for _, _, kw in tasks]
    assert len(seeds) == len(set(seeds))
    assert not set(seeds) & {kw["seed"] for _, _, kw in lab.plan(cal, 16, 100, 1)}


def test_show_and_compare_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pairs: dict[str, dict[str, object]] = {
        "A_USDT": {"trades": 4600, "profile": asdict(IID), "theta": 0.001},
        "T_USDT": {"trades": 4, "profile": None, "theta": -0.0112},
    }
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(_cal(pairs)))
    assert lab.main(["show", str(cal)]) == 0
    text = capsys.readouterr().out
    assert "too thin to calibrate" in text
    assert "A_USDT" in text
    out = tmp_path / "cmp.json"
    argv = ["compare", str(cal), "--reps", "8", "--resamples", "100", "--workers", "1", "--out", str(out)]
    assert lab.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["methods"] == list(day_tests.METHODS)
    assert doc["results"]["bhcmp/all/30d/all_null"]["reps"] == 8  # 8 chunks of 1 merged
    assert "compare/all/30d/size/A_USDT" in doc["results"]
    assert "all/30d:" in capsys.readouterr().out


def test_day_stats_by_hand() -> None:
    den = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    num = np.array([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    counts = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
    st = lab.day_stats(num, den, counts)
    th = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert st["day_sd"] == pytest.approx(th.std(ddof=1))
    assert st["active_share"] == pytest.approx(0.9)
    assert st["day_ac1"] == pytest.approx(1.0)  # pairs (1,2), (3,4), ..., (8,9): the gap on day 2 is skipped
    q05, q25, q75, q95 = np.percentile(th, [5, 25, 75, 95])
    assert st["day_tail"] == pytest.approx((q95 - q05) / (q75 - q25))
    assert st["count_cv"] == pytest.approx(counts.std() / counts.mean())
    short = lab.day_stats(num[:3], den[:3], counts[:3])
    assert np.isnan(short["day_tail"])
    assert np.isnan(short["day_ac1"])


def test_mid_rank() -> None:
    assert lab._mid_rank([1.0, 1.0, 1.0], 1.0) == 0.5
    assert lab._mid_rank([0.0, 1.0, 2.0, 3.0], 1.5) == 0.5
    assert lab._mid_rank([0.0, 1.0, 2.0, 3.0], 2.0) == pytest.approx(2.5 / 4)
    assert np.isnan(lab._mid_rank([], 1.0))
    assert np.isnan(lab._mid_rank([1.0], float("nan")))


def test_realism_ranks_fit_and_misfit() -> None:
    p = mp.PairProfile(100.0, 1.0, 3.0, 1.0, 0.01, 1000.0, 0.002, 0.3)
    day, v, x = mp.simulate_trades(p, 0.0, 92, np.random.default_rng(41))
    fit = lab.realism_check(day, v, x, 92, p, 0.0, reps=200, seed=42)
    assert all(0.02 < r < 0.98 for r in fit["rank"].values())
    sticky = mp.PairProfile(100.0, 1.0, 3.0, 1.0, 0.0001, 1000.0, 0.003, 0.9)
    day, v, x = mp.simulate_trades(sticky, 0.0, 92, np.random.default_rng(43))
    loose = mp.PairProfile(100.0, 1.0, 3.0, 1.0, 0.0001, 1000.0, 0.003, 0.0)
    miss = lab.realism_check(day, v, x, 92, loose, 0.0, reps=200, seed=44)
    assert miss["rank"]["day_ac1"] > 0.975  # the real days are far more persistent than the profile says
    assert set(miss["sim_median"]) == set(lab.REALISM_STATS)


def test_realism_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["A_USDT"], "2026-08-01T00", "2026-10-31T23", days=92)
    cal = tmp_path / "cal.json"
    assert lab.main(["calibrate", str(ds), "--universe", str(uni), "--out", str(cal)]) == 0
    out = tmp_path / "real.json"
    assert lab.main(["realism", str(ds), str(cal), "--reps", "20", "--workers", "1", "--out", str(out)]) == 0
    res = json.loads(out.read_text())["results"]
    par = tmp_path / "real2.json"
    assert lab.main(["realism", str(ds), str(cal), "--reps", "20", "--workers", "2", "--out", str(par)]) == 0
    assert json.loads(par.read_text())["results"] == res  # same seeds, same answer in parallel
    assert set(res) == {"all/A_USDT", "q80/A_USDT"}
    assert res["all/A_USDT"]["real"]["active_share"] == 1.0
    assert "median rank" in capsys.readouterr().out


def test_build_pool_centres_residuals_and_estimates_the_day_effect() -> None:
    day, v, x = mp.simulate_trades(TRUE, 0.001, 1500, np.random.default_rng(51))
    pool, theta = lab.build_pool(day, v, x, 1500)
    assert theta == pytest.approx(0.001, abs=3e-4)
    assert float((pool.v * pool.r).sum()) == pytest.approx(0.0, abs=1e-9 * float(pool.v.sum()))
    assert (pool.day_counts == np.bincount(day, minlength=1500)).all()
    assert pool.day_sd == pytest.approx(0.002, rel=0.1)
    assert pool.day_ar == pytest.approx(0.3, abs=0.06)
    single = lab.build_pool(np.array([0, 1, 1]), np.array([5.0, 1.0, 3.0]), np.array([0.9, 0.01, -0.01]), 3)[
        0
    ]
    assert single.v.size == 2  # the one-trade day 0 gives no residual
    assert single.r == pytest.approx(np.array([0.01 + 0.005, -0.01 + 0.005]) * np.sqrt(2))


def test_pool_roundtrip_and_dispatch(tmp_path: Path) -> None:
    day, v, x = mp.simulate_trades(IID, 0.0, 60, np.random.default_rng(52))
    pool, _ = lab.build_pool(day, v, x, 60)
    path = tmp_path / "p" / "X.npz"
    lab.save_pool(path, pool)
    back = lab.load_pool(str(path))
    assert isinstance(back, mp.EmpiricalPool)
    assert (back.v == pool.v).all()
    assert (back.r == pool.r).all()
    assert (back.day_sd, back.day_ar) == (pool.day_sd, pool.day_ar)
    assert lab.load_pool(str(path)) is back  # cached per process
    a = lab.simulate(str(path), 0.001, 30, np.random.default_rng(53))
    b = mp.simulate_days_empirical(pool, 0.001, 30, np.random.default_rng(53))
    assert all((x1 == x2).all() for x1, x2 in zip(a, b, strict=True))
    c = lab.simulate(IID, 0.001, 30, np.random.default_rng(53))
    d = mp.simulate_days(IID, 0.001, 30, np.random.default_rng(53))
    assert all((x1 == x2).all() for x1, x2 in zip(c, d, strict=True))


def test_empirical_generator_passes_its_own_realism_check(tmp_path: Path) -> None:
    p = mp.PairProfile(100.0, 1.0, 3.0, 1.5, 0.01, 4.0, 0.001, 0.3)  # heavy tails, uneven days
    day, v, x = mp.simulate_trades(p, 0.0, 92, np.random.default_rng(54))
    pool, theta = lab.build_pool(day, v, x, 92)
    path = tmp_path / "X.npz"
    lab.save_pool(path, pool)
    res = lab.realism_check(day, v, x, 92, str(path), theta, reps=300, seed=55)
    assert all(0.02 < r < 0.98 for r in res["rank"].values())


def test_pools_cli_and_empirical_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "B_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["A_USDT", "B_USDT"], "2026-08-01T00", "2026-10-31T23", days=92)
    pools = tmp_path / "lab" / "pools.json"
    base = ["pools", str(ds), "--universe", str(uni), "--dir", str(tmp_path / "pools"), "--out", str(pools)]
    assert lab.main([*base, "--end", "2026-11-02"]) == 2
    assert lab.main(base) == 0
    doc = json.loads(pools.read_text())
    assert doc["generator"] == "empirical"
    e = doc["subsets"]["q80"]["pairs"]["A_USDT"]
    assert (e["trades"], e["pool_trades"]) == (184, 184)
    assert Path(e["pool"]).exists()
    assert "in pool" in capsys.readouterr().out
    assert lab.main(["show", str(pools)]) == 0
    assert "A_USDT" in capsys.readouterr().out
    out = tmp_path / "lab" / "x.json"
    assert (
        lab.main(
            ["run", str(pools), "--reps", "3", "--resamples", "100", "--workers", "1", "--out", str(out)]
        )
        == 0
    )
    assert json.loads(out.read_text())["results"]["h003/november/scale_1"]["mean_counted_pairs"] == 2
    assert (
        lab.main(
            ["compare", str(pools), "--reps", "8", "--resamples", "100", "--workers", "1", "--out", str(out)]
        )
        == 0
    )
    assert (
        lab.main(["realism", str(ds), str(pools), "--reps", "10", "--workers", "1", "--out", str(out)]) == 0
    )
    assert set(json.loads(out.read_text())["results"]) == {
        "all/A_USDT",
        "all/B_USDT",
        "q80/A_USDT",
        "q80/B_USDT",
    }


def test_thin_pools_are_not_tested_on_their_own() -> None:
    def entry(trades: int, days: int) -> dict[str, object]:
        return {
            "trades": trades,
            "active_days": days,
            "pool": "x.npz",
            "pool_trades": trades,
            "theta": 0.0,
            "day_sd": 0.0,
            "day_ar": 0.0,
        }

    assert lab._usable(entry(30, 5))
    assert not lab._usable(entry(29, 5))
    assert not lab._usable(entry(30, 4))
    sub = {"pairs": {"A": entry(400, 40), "T": entry(9, 2)}, "rho": 0.0, "common_ar": 0.0}
    cal = {
        "period": ["2026-08-01", "2026-11-01"],
        "n_days": 92,
        "generator": "empirical",
        "subsets": {"all": sub, "q80": sub},
    }
    names = [n for n, _, _ in lab.compare_plan(cal, 8, 100, 1)]
    assert any(n.endswith("/A") for n in names)
    assert not any(n.endswith("/T") for n in names)
    assert "(too thin to test on its own)" in lab.show(cal)
    assert lab.profiles_for(sub, 92)["T"] == ("x.npz", 0.0)  # thin pairs still trade in multi-pair checks


def test_daypool_build_save_and_share(tmp_path: Path) -> None:
    day, v, x = mp.simulate_trades(TRUE, 0.001, 92, np.random.default_rng(61))
    pool, theta = lab.build_daypool(day, v, x, 92)
    assert theta == pytest.approx(float((v * x).sum() / v.sum()))
    assert float(pool.num_c.sum()) == pytest.approx(0.0, abs=1e-12 * float(pool.den.sum()))
    assert pool.den == pytest.approx(np.bincount(day, weights=v, minlength=92))
    assert (pool.counts == np.bincount(day, minlength=92)).all()
    assert 1.0 <= pool.block <= 28  # Politis-White cap min(3 sqrt(92), 92 / 3) rounded up
    assert pool.block == day_tests.politis_white_block(pool.num_c / pool.den.mean())
    assert pool.block > 1.0  # the AR(1) day effects of TRUE persist across days
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    lab.save_pool(a, pool)
    lab.save_pool(b, pool)
    back = lab.load_pool(str(a))
    assert isinstance(back, mp.DayPool)
    assert (back.num_c == pool.num_c).all()
    assert back.block == pool.block
    rng = np.random.default_rng(62)
    days = lab.shared_days([str(a), str(b)], 30, rng)
    assert days is not None
    assert days.size == 30
    assert lab.shared_days([IID], 30, rng) is None
    n1, d1, _ = lab.simulate(str(a), 0.0, 30, rng, days=days)
    n2, _, _ = lab.simulate(str(b), 0.0, 30, rng, days=days)
    assert (n1 == n2).all()  # same pool, same calendar days
    assert (d1 == pool.den[days]).all()


def test_daypools_cli_and_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "B_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["A_USDT", "B_USDT"], "2026-08-01T00", "2026-10-31T23", days=92)
    pools = tmp_path / "lab" / "daypools.json"
    argv = [
        "pools",
        str(ds),
        "--universe",
        str(uni),
        "--dir",
        str(tmp_path / "dp"),
        "--kind",
        "days",
        "--out",
        str(pools),
    ]
    assert lab.main(argv) == 0
    doc = json.loads(pools.read_text())
    assert doc["kind"] == "days"
    assert doc["subsets"]["all"]["pairs"]["A_USDT"]["block"] >= 1.0
    assert "mean block" in capsys.readouterr().out
    out = tmp_path / "lab" / "x.json"
    assert (
        lab.main(
            ["run", str(pools), "--reps", "3", "--resamples", "100", "--workers", "1", "--out", str(out)]
        )
        == 0
    )
    res = json.loads(out.read_text())["results"]
    assert res["h003/november/scale_1"]["mean_counted_pairs"] == 2
    assert (
        lab.main(
            ["compare", str(pools), "--reps", "8", "--resamples", "100", "--workers", "1", "--out", str(out)]
        )
        == 0
    )


def test_multi_pair_checks_share_calendar_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []
    for i in range(3):
        day, v, x = mp.simulate_trades(TRUE, 0.0, 92, np.random.default_rng(70 + i))
        pool, _ = lab.build_daypool(day, v, x, 92)
        path = tmp_path / f"P{i}.npz"
        lab.save_pool(path, pool)
        paths.append(str(path))
    seen: list[tuple[str, tuple[int, ...] | None]] = []
    real = lab.simulate

    def spy(
        gen: lab.Gen,
        theta: float,
        n_days: int,
        rng: np.random.Generator,
        common: Any = None,
        rho: float = 0.0,
        days: Any = None,
    ) -> Any:
        seen.append((str(gen), None if days is None else tuple(days.tolist())))
        return real(gen, theta, n_days, rng, common, rho, days)

    monkeypatch.setattr(lab, "simulate", spy)
    runs: list[Callable[[], object]] = [
        lambda: lab.bh_check(paths, [0.0] * 3, 30, 0.0, 0.0, reps=2, resamples=100, seed=1),
        lambda: lab.bh_compare_check(paths, [0.0] * 3, 30, 0.0, 0.0, reps=2, resamples=100, seed=2),
        lambda: lab.h003_check(
            {p: (p, 0.0) for p in paths}, 1.0, 30, 0.0, 0.0, reps=2, resamples=100, seed=3
        ),
    ]
    for run in runs:
        seen.clear()
        run()
        assert len(seen) == 6
        for rep in (seen[:3], seen[3:]):
            assert [g for g, _ in rep] == paths
            assert rep[0][1] is not None
            assert len(rep[0][1]) == 30
            assert len({d for _, d in rep}) == 1  # one calendar draw for all pairs in a period
        assert seen[0][1] != seen[3][1]  # a fresh draw per period


def test_shared_days_use_the_median_block(tmp_path: Path) -> None:
    z = np.zeros(40)
    paths = []
    for i, block in enumerate((1e9, 1e9, 1.0)):  # median: one block per period
        path = tmp_path / f"B{i}.npz"
        lab.save_pool(path, mp.DayPool(z, z + 1.0, np.ones(40, dtype=np.int64), block))
        paths.append(str(path))
    for r in range(20):
        days = lab.shared_days(paths, 25, np.random.default_rng([80, r]))
        assert days is not None
        assert ((np.diff(days) == 1) | (np.diff(days) == -39)).all()


def test_shared_days_need_one_calendar(tmp_path: Path) -> None:
    for n in (40, 41):
        lab.save_pool(
            tmp_path / f"C{n}.npz", mp.DayPool(np.zeros(n), np.ones(n), np.ones(n, dtype=np.int64), 2.0)
        )
    with pytest.raises(ValueError, match="same calendar days"):
        lab.shared_days([str(tmp_path / "C40.npz"), str(tmp_path / "C41.npz")], 10, np.random.default_rng(1))
