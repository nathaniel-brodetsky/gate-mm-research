"""Synthetic lab: calibration recovers known parameters; checks behave as theory says on easy cases."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

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
