from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmlab.topology import persistence as ph
from mmlab.topology import regime_lab as rl


def test_regimes_alternate_and_are_reproducible() -> None:
    r1 = rl.regimes(np.random.default_rng(3), 5000, 100.0)
    r2 = rl.regimes(np.random.default_rng(3), 5000, 100.0)
    assert r1.tolist() == r2.tolist()
    assert set(r1.tolist()) == {0, 1}
    assert 20 < int(np.sum(r1[1:] != r1[:-1])) < 80  # about 5000 / 100 spells
    assert len(set(rl.regimes(np.random.default_rng(3), 500, 1e9).tolist())) == 1


def test_since_switch() -> None:
    reg = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
    assert rl.since_switch(reg).tolist() == [rl.NO_SWITCH, rl.NO_SWITCH, 0, 1, 2, 0]


@pytest.mark.parametrize(
    ("phi", "half_life"), [(0.3, 30.0), (-0.3, 30.0), (0.6, 30.0), (0.6, 1.0), (-0.6, 1.0)]
)
def test_ema_innovation_sd_gives_the_target_variance(phi: float, half_life: float) -> None:
    n, sigma = 400_000, 0.35
    lam = 0.5 ** (1 / half_life)
    e = np.random.default_rng(4).standard_normal(n) * rl.ema_innovation_sd(phi, half_life, sigma)
    dp = np.empty(n)
    ema = 0.0
    for t in range(n):
        dp[t] = e[t] + phi * ema
        ema = lam * ema + (1 - lam) * dp[t]
    assert float(np.var(dp[1000:])) == pytest.approx(sigma**2, rel=0.02)


def _changes(scenario: str, regime: int, n: int = 200_000) -> np.ndarray:
    return rl.latent_changes(np.random.default_rng(5), scenario, np.full(n, regime, dtype=np.int8))


def _vr(d: np.ndarray, k: int) -> float:
    p = np.cumsum(d)
    return float(np.var(p[k:] - p[:-k]) / (k * np.var(d)))


@pytest.mark.parametrize("scenario", rl.SCENARIOS)
def test_scenarios_have_the_designed_1s_variance(scenario: str) -> None:
    v0, v1 = float(np.var(_changes(scenario, 0))), float(np.var(_changes(scenario, 1)))
    assert v0 == pytest.approx(rl.SIGMA**2, rel=0.03)
    factor = rl.VOL_FACTOR**2 if scenario == "volatility" else 1.0
    assert v1 == pytest.approx(rl.SIGMA**2 * factor, rel=0.05)


def test_reversion_reverts_and_momentum_continues() -> None:
    assert _vr(_changes("reversion", 0), 60) < 0.85
    assert _vr(_changes("reversion", 1), 60) > 1.2
    assert _vr(_changes("volatility", 0), 60) == pytest.approx(1.0, abs=0.08)


def test_jumps_and_cycles_have_their_signature() -> None:
    d = _changes("jumps", 1)
    assert float(np.mean(np.abs(d) > 6 * rl.SIGMA)) > 0.0015  # jumps of sd 2.7 ticks in 0.5% of seconds
    assert float(np.mean(np.abs(_changes("jumps", 0)) > 6 * rl.SIGMA)) < 1e-4
    lag = int(rl.CYCLE_S)
    for regime, low, high in [(1, 0.06, 0.14), (0, -0.04, 0.04)]:
        d = _changes("cycles", regime, 20_000)
        assert low < float(np.corrcoef(d[:-lag], d[lag:])[0, 1]) < high  # the cycle carries 10%


def test_cycle_level() -> None:
    on = np.array([False] * 50 + [True] * 100 + [False] * 50 + [True] * 100)
    level = rl.cycle_level(np.random.default_rng(6), on, 2.0, 40.0)
    assert np.all(level[:50] == 0)
    assert level[50] == 0.0  # no jump when a spell starts
    assert np.all(level[150:200] == level[149])  # frozen between spells
    assert level[200] == level[199]
    assert np.allclose(level[60:100], level[100:140])  # period 40 s
    assert float(np.ptp(level[50:150])) == pytest.approx(4.0, abs=0.05)
    assert rl.cycle_level(np.random.default_rng(6), np.zeros(5, dtype=bool), 2.0, 40.0).tolist() == [0.0] * 5


def test_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        rl.latent_changes(np.random.default_rng(0), "nope", np.zeros(3, dtype=np.int8))


def test_mid_path_is_on_the_tick_grid() -> None:
    reg = np.zeros(1000, dtype=np.int8)
    m = rl.mid_path(np.random.default_rng(7), "volatility", reg)
    assert np.all(m == np.round(m))
    assert 0.1 < float(np.mean(np.diff(m) != 0)) < 0.5


def test_detrend_standardize() -> None:
    assert rl.detrend_standardize(np.arange(10.0) * 3 + 1).tolist() == [0.0] * 10
    z = rl.detrend_standardize(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    assert float(np.std(z)) == pytest.approx(1.0)
    assert float(np.mean(z)) == pytest.approx(0.0, abs=1e-12)
    assert rl.detrend_standardize(np.array([1.0, 2.0])).tolist() == [0.0, 0.0]


def test_window_features() -> None:
    m = np.round(np.cumsum(np.random.default_rng(8).standard_normal(61)))
    f = rl.window_features(m, 1)
    assert len(f) == len(rl.FEATURES)
    assert all(math.isfinite(x) for x in f)
    names = dict(zip(rl.FEATURES, f, strict=True))
    assert names["range"] == float(np.max(m) - np.min(m))
    assert names["swings_2_ticks"] >= 0
    flat = rl.window_features(np.zeros(61), 1)
    assert all(math.isfinite(x) for x in flat)
    assert flat[len(rl.BASE) :] == [0.0] * len(rl.TOPO)
    base = rl.window_features(m, 1, base_only=True)
    assert base[: len(rl.BASE)] == f[: len(rl.BASE)]
    assert all(math.isnan(x) for x in base[len(rl.BASE) :])


def test_the_rips_part_uses_the_embedding_step() -> None:
    m = np.round(np.cumsum(np.random.default_rng(12).standard_normal(301)))
    f = dict(zip(rl.FEATURES, rl.window_features(m, 5), strict=True))
    h0, h1 = ph.rips_bars(ph.delay_embedding(rl.detrend_standardize(m[::5]), 3, 1))
    assert f["rips_h0_total"] == pytest.approx(ph.total(h0))
    assert f["rips_h1_total"] == pytest.approx(ph.total(h1))
    swings = ph.swing_bars(m)  # swings use every second
    assert f["swing_landscape_l1"] == pytest.approx(ph.landscape_l1(swings))
    assert f["swings_2_ticks"] == ph.count_at_least(swings, 2.0) < ph.count_at_least(swings, 1.0)


def test_a_cycle_gives_a_long_loop_and_noise_does_not() -> None:
    t = np.arange(61.0)
    cyc = rl.window_features(np.round(3 * np.sin(2 * math.pi * t / 20)), 1)
    noise = rl.window_features(np.round(np.cumsum(np.random.default_rng(9).standard_normal(61))), 1)
    j = rl.FEATURES.index("rips_h1_longest")
    assert cyc[j] > noise[j]


def test_auc() -> None:
    y = np.array([0, 0, 1, 1])
    assert rl.auc(np.array([0.1, 0.4, 0.35, 0.8]), y) == pytest.approx(0.75)
    assert rl.auc(np.array([1.0, 1.0, 1.0, 1.0]), y) == 0.5
    assert rl.auc(np.array([0.0, 0.1, 0.2, 0.3]), y) == 1.0
    assert math.isnan(rl.auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_fit_logistic_recovers_the_direction() -> None:
    rng = np.random.default_rng(10)
    x = rng.standard_normal((4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(1.5 + 2 * x[:, 0] - x[:, 1])))).astype(np.int64)
    model = rl.fit_logistic(x, y, l2=0.0)
    beta = model[2] * np.r_[1.0, 1.0 / model[1]]  # back to raw units
    assert beta[1] == pytest.approx(2.0, rel=0.15)
    assert beta[2] == pytest.approx(-1.0, rel=0.15)
    assert rl.auc(rl.logistic_score(model, x), y) > 0.8
    shrunk = rl.fit_logistic(x, y, l2=1e6)
    assert float(np.max(np.abs(shrunk[2][1:]))) < 0.01
    rate = float(np.mean(y))
    assert shrunk[2][0] == pytest.approx(math.log(rate / (1 - rate)), abs=0.01)  # the intercept is free
    const = rl.fit_logistic(np.ones((10, 1)), np.array([0, 1] * 5))
    assert np.all(np.isfinite(const[2]))


def _paths(signal_col: int, n_paths: int = 9, n: int = 300) -> list[dict[str, Any]]:
    rng = np.random.default_rng(11)
    out = []
    for p in range(n_paths):
        y = rng.integers(2, size=n)
        x = rng.standard_normal((n, len(rl.FEATURES)))
        x[:, 0] += 0.5 * y  # a weak baseline signal
        x[:, signal_col] += 1.5 * y
        out.append({"path": p, "label": y, "since": np.full(n, 500), "X": {60: x, 300: x}})
    return out


def test_evaluate_sees_a_topological_signal() -> None:
    r = rl.evaluate(_paths(len(rl.BASE) + 2), 60, np.random.default_rng(0), 200, False)
    assert r["auc"]["both"] > r["auc"]["baselines"] + 0.1
    assert r["gain"] > 0.1
    assert r["gain_interval"][0] > 0
    assert r["test_points"] == 4 * 300  # odd paths
    assert r["train_points"] == 5 * 300
    assert r["single_feature_auc"]["swing_landscape_l1"] > 0.7
    assert r["auc_by_seconds_since_switch"][-1]["points"] == 1200
    assert r["auc_by_seconds_since_switch"][0]["points"] == 0


def test_evaluate_without_a_topological_signal() -> None:
    r = rl.evaluate(_paths(1), 60, np.random.default_rng(0), 200, False)
    assert abs(r["gain"]) < 0.02
    assert r["gain_interval"][0] < 0 < r["gain_interval"][1]
    b = rl.evaluate(_paths(1), 60, np.random.default_rng(0), 200, True)
    assert set(b["auc"]) == {"baselines"}
    assert "gain" not in b
    assert set(b["single_feature_auc"]) == set(rl.BASE)


def _cell(gain: float, lo: float) -> dict[str, Any]:
    return {"gain": gain, "gain_interval": [lo, lo + 0.05]}


def test_decide() -> None:
    res = {s: {str(w): _cell(0.0, -0.01) for w in rl.WINDOWS} for s in rl.SCENARIOS}
    assert rl.decide(res)["passes"] is False
    res["cycles"]["60"] = _cell(0.2, 0.1)  # a positive control does not count
    assert rl.decide(res)["passes"] is False
    res["jumps"]["300"] = _cell(0.02, 0.001)
    assert rl.decide(res) | {"rule": ""} == {"rule": "", "passes": True, "cells": ["jumps 300s"]}
    res["jumps"]["300"] = _cell(0.019, 0.001)
    assert rl.decide(res)["passes"] is False
    res["reversion"]["60"] = _cell(0.05, 0.0)
    assert rl.decide(res)["passes"] is False


def test_lab_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "lab.json"
    argv = ["--paths", "4", "--workers", "1", "--length-s", "900", "--every-s", "60", "--boots", "20"]
    assert rl.main([*argv, "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert set(res["results"]) == set(rl.SCENARIOS)
    assert set(res["results"]["jumps"]) == {"60", "300"}
    assert res["results"]["jumps"]["60"]["test_points"] == 2 * 10
    assert "decision" in res
    assert res["config"]["gudhi"]
    text = capsys.readouterr().out
    assert "decision:" in text
    assert "reversion" in text
    again = rl.lab(4, 20261125, 1, 20, False, 900, 60)
    assert json.dumps(again["results"], sort_keys=True) == json.dumps(res["results"], sort_keys=True)
    assert rl.main([*argv, "--baselines-only"]) == 0
    assert "decision" not in capsys.readouterr().out
