"""H-003 test: hand-computed markouts, the registered counting and decision rules, and the CLI guards."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from mmlab.evaluation import h003
from mmlab.evaluation.bootstrap import stationary_indices
from mmlab.microstructure import conditional as cd
from mmlab.microstructure.markouts import in_period

S = 1_000_000
T = 1_785_542_400 * S  # 2026-08-01 00:00 UTC
DAY = 86_400 * S


def tob(rows: list[tuple[float, float, float, float, float]], valid: bool = True) -> pl.DataFrame:
    """rows: (label seconds after T, bid, ask, bid_qty, ask_qty); available 100 ms after the label."""
    return pl.DataFrame(
        {
            "label_us": [T + round(r[0] * S) for r in rows],
            "avail_us": [T + round(r[0] * S) + 100_000 for r in rows],
            "bid": [r[1] for r in rows],
            "bid_qty": [r[3] for r in rows],
            "ask": [r[2] for r in rows],
            "ask_qty": [r[4] for r in rows],
            "valid": [valid] * len(rows),
        }
    )


def trades(rows: Sequence[tuple[float, float, int]]) -> pl.DataFrame:
    """rows: (seconds after T, price, taker +1/-1); quantity 1."""
    return pl.DataFrame(
        {
            "ts_us": [T + round(r[0] * S) for r in rows],
            "trade_id": list(range(len(rows))),
            "price": [r[1] for r in rows],
            "qty": [1.0] * len(rows),
            "taker": [r[2] for r in rows],
        },
        schema_overrides={"taker": pl.Int8},
    )


def wide_pair(days: int, per_day: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Constant book 98 / 102 (q = 200 bp); buys at 102 and sells at 98: RS = 200 bp at every horizon."""
    book = tob([(-1, 98.0, 102.0, 1.0, 1.0)])
    rows = []
    for d in range(days):
        for k in range(per_day):
            side = 1 if k % 2 == 0 else -1
            rows.append((d * 86_400 + 3_600 * (k + 1), 102.0 if side > 0 else 98.0, side))
    return book, trades(rows)


def test_states_against_hand_values() -> None:
    book = tob(
        [
            (0, 98.0, 102.0, 3.0, 1.0),  # mid 100, q 200 bp, microprice (98*1 + 102*3) / 4 = 101
            (5, 99.0, 101.0, 1.0, 1.0),  # mid 100
            (49.9, 100.0, 102.0, 1.0, 1.0),  # mid 101 from 50 s
            (200, 97.0, 99.0, 1.0, 1.0),  # mid 98
        ]
    )
    st = h003.states(trades([(1, 102.0, 1), (1, 98.0, -1)]), book)
    buy, sell = st.rows(named=True)
    assert buy["q"] == pytest.approx(0.02)
    assert buy["es"] == pytest.approx(0.02)
    assert (buy["rs_10s"], buy["rs_60s"], buy["rs_300s"]) == (
        pytest.approx(0.02),
        pytest.approx(0.01),
        pytest.approx(0.04),
    )
    assert buy["rs_micro_60s"] == pytest.approx((102 - 101) / 101)
    assert buy["rs_exit_60s"] == pytest.approx(0.0)  # buys back at the ask, 102
    assert sell["rs_60s"] == pytest.approx(0.03)
    assert sell["rs_exit_60s"] == pytest.approx(0.02)  # sells back at the bid, 100
    assert sell["rs_micro_60s"] == pytest.approx((101 - 98) / 101)


def test_invalid_books_drop_the_trade_or_the_horizon() -> None:
    book = pl.concat([tob([(0, 98.0, 102.0, 1.0, 1.0)]), tob([(30, 98.0, 102.0, 1.0, 1.0)], valid=False)])
    st = h003.states(trades([(1, 102.0, 1), (40, 102.0, 1)]), book)
    assert st.height == 1  # the trade at 40 s has an invalid book
    assert (st["rs_10s"][0], st["rs_60s"][0], st["rs_exit_60s"][0]) == (pytest.approx(0.02), None, None)


def test_eligibility_includes_the_threshold_despite_rounding() -> None:
    book = tob([(0, 99.2, 100.8, 1.0, 1.0)])  # q = 80 bp; 100.8 - 99.2 is 1.5999999999999943
    st = h003.states(trades([(1, 100.8, 1)]), book)
    assert st["q"][0] * 1e4 < 80
    assert h003.eligible(st, 80.0).height == 1
    assert h003.eligible(st, 80.000001).height == 0


def test_pair_result_and_counting() -> None:
    book, tr = wide_pair(days=20, per_day=2)
    st = in_period(h003.states(tr, book), "2026-08-01", "2026-11-01")
    r = h003.evaluate_pair("W", st, 80.0, 92)
    assert (r.trades, r.days, r.counted) == (40, 20, True)
    assert r.theta == pytest.approx(0.02 - h003.MAKER_FEE)
    assert r.ci is not None
    assert r.ci[0] == pytest.approx(0.019)
    assert r.ci[1] == pytest.approx(0.019)
    assert r.p == pytest.approx(1 / (h003.RESAMPLES + 1))
    assert r.usdt_per_day == pytest.approx(20 * (102 + 98) / 92)
    assert r.maker_net_per_day == pytest.approx(0.019 * r.usdt_per_day)
    assert r.rs == {"10s": pytest.approx(0.02), "60s": pytest.approx(0.02), "300s": pytest.approx(0.02)}
    assert r.theta_exit == pytest.approx(-h003.MAKER_FEE)  # exit at the same wide quote: nothing left
    assert r.theta_trade_median == pytest.approx(0.019)
    assert r.theta_all_trades == pytest.approx(0.019)
    assert h003.evaluate_pair("W", st, 250.0, 92).trades == 0  # q = 200 bp is below 250


@pytest.mark.parametrize(
    ("days", "per_day", "counted"),
    [(15, 2, True), (5, 6, True), (29, 1, False), (4, 10, False)],  # 30 on 15 days, 30 on 5, 29, 40 on 4
)
def test_minimum_trades_and_days(days: int, per_day: int, counted: bool) -> None:
    book, tr = wide_pair(days, per_day)
    r = h003.evaluate_pair("W", h003.states(tr, book), 80.0, 92)
    assert r.counted is counted


def test_agrees_with_the_exploration_bucket() -> None:
    book = pl.concat(
        [
            tob(
                [(0, 98.0, 102.0, 1.0, 1.0), (3_000, 99.95, 100.05, 1.0, 1.0), (7_000, 97.0, 103.0, 1.0, 1.0)]
            ),
            tob([(9_000, 97.0, 103.0, 1.0, 1.0)], valid=False),
        ]
    )
    # the trade at 8,990 s is eligible but has no RS(60 s): it counts for capacity, not for theta
    tr = trades(
        [
            (10, 102.0, 1),
            (20, 98.0, -1),
            (3_100, 100.05, 1),
            (7_100, 97.0, -1),
            (7_200, 103.0, 1),
            (8_990, 103.0, 1),
        ]
    )
    st = h003.states(tr, book)
    cell = next(c for c in cd.cells(cd.trade_states(tr, book), "X", "p", 92) if c.bucket == "[80, inf) bp")
    r = h003.evaluate_pair("X", st, 80.0, 92)
    assert r.theta == pytest.approx(cell.theta)
    assert r.usdt_per_day == pytest.approx(cell.usdt_per_day)
    assert r.usdt_per_day == pytest.approx((102 + 98 + 97 + 103 + 103) / 92)
    assert r.trades == 4


def test_pair_bootstrap_uses_its_registered_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []
    real = np.random.default_rng

    def spy(seed: int) -> np.random.Generator:
        seeds.append(seed)
        return real(seed)

    monkeypatch.setattr(np.random, "default_rng", spy)
    v = np.array([0.004, -0.001, 0.002, 0.003, 0.0, 0.005, -0.002, 0.001, 0.006])
    mc = h003.median_ci(list(v))
    assert seeds == [20261125]
    assert mc is not None
    meds = np.median(v[real(20261125).integers(0, v.size, (10_000, v.size))], axis=1)
    assert mc == (0.002, (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))))
    assert h003.median_ci([]) is None


def test_day_bootstrap_uses_its_registered_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, float, int]] = []
    real = stationary_indices

    def spy(n: int, resamples: int, mean_block: float, seed: int) -> Any:
        calls.append((n, resamples, mean_block, seed))
        return real(n, resamples, mean_block, seed)

    monkeypatch.setattr(h003, "stationary_indices", spy)
    book, tr = wide_pair(days=6, per_day=6)
    h003.evaluate_pair("W", h003.states(tr, book), 80.0, 92)
    assert calls == [(6, 10_000, 3.0, 20261126)]


def test_days_count_only_trades_with_a_markout() -> None:
    book, tr = wide_pair(days=4, per_day=8)  # 32 trades on 4 days
    x = 10 * 86_400  # a fifth day whose trade has no RS(60 s)
    book = pl.concat(
        [book, tob([(x + 30, 98.0, 102.0, 1.0, 1.0)], valid=False), tob([(x + 120, 98.0, 102.0, 1.0, 1.0)])]
    )
    tr = pl.concat(
        [tr, trades([(x + 1, 102.0, 1)]).with_columns(pl.lit(99, dtype=pl.Int64).alias("trade_id"))]
    )
    r = h003.evaluate_pair("W", h003.states(tr, book), 80.0, 92)
    assert (r.trades, r.days, r.counted) == (32, 4, False)
    assert r.usdt_per_day == pytest.approx((16 * (102 + 98) + 102) / 92)


def _pr(name: str, theta: float, p: float = 0.5, counted: bool = True) -> h003.PairResult:
    return h003.PairResult(name, 80.0, 40, 10, counted, theta=theta, p=p, usdt_per_day=100.0)


def test_decision_rules() -> None:
    accept = h003.summarize([_pr(f"A{i}", 0.002 + i * 1e-4) for i in range(8)], "test")
    assert (accept["outcome"], accept["pairs_counted"]) == ("accept", 8)
    assert accept["ci"][0] > 0
    assert accept["median_theta"] == pytest.approx(0.00235)
    mixed = h003.summarize(
        [_pr(f"M{i}", t) for i, t in enumerate([-3e-3, -2e-3, -1e-3, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3])], "test"
    )
    assert mixed["outcome"] == "reject"
    assert mixed["ci"][0] <= 0
    few = [_pr(f"F{i}", 0.01) for i in range(7)] + [_pr("X", 0.01, counted=False)]
    assert h003.summarize(few, "test")["outcome"] == "extend"
    assert h003.summarize(few, "extended")["outcome"] == "undecided"
    assert h003.summarize(few, "exploration")["outcome"].startswith("none: dry run")
    assert h003.summarize([_pr(f"A{i}", 0.01) for i in range(9)], "exploration")["outcome"].startswith("none")
    assert h003.summarize([_pr(f"A{i}", 0.01) for i in range(9)], "holdout")["outcome"].startswith(
        "none: one-time"
    )
    bh = h003.summarize([_pr("S", 0.01, p=0.0001), *[_pr(f"N{i}", 0.0, p=0.9) for i in range(7)]], "test")
    assert bh["bh_rejections"] == ["S"]
    assert bh["usdt_per_day_all_pairs"] == pytest.approx(800.0)
    assert bh["maker_net_per_day_counted_pairs"] == pytest.approx(0.0)  # maker_net_per_day not set


def test_registered_parameters_are_unchanged() -> None:
    assert (h003.THRESHOLD_BP, h003.SENSITIVITY_BP, h003.MAKER_FEE) == (80.0, (40.0, 120.0), 0.0010)
    assert (h003.MIN_TRADES, h003.MIN_DAYS, h003.MIN_PAIRS) == (30, 5, 8)
    assert (h003.RESAMPLES, h003.MEAN_BLOCK_DAYS, h003.BH_Q) == (10_000, 3.0, 0.10)
    assert (h003.SEED_PAIRS, h003.SEED_DAYS) == (20261125, 20261126)
    assert h003.PERIODS == {
        "exploration": ("2026-08-01", "2026-11-01"),
        "test": ("2026-11-01", "2026-12-01"),
        "extended": ("2026-11-01", "2027-01-01"),
    }
    assert h003.HOLDOUT_START == "2026-08-01"


def _dataset(root: Path, pairs: list[str], first: str, last: str, days: int = 122) -> Path:
    """Trades every day from 2026-08-01 for `days` days (122 = to the end of November)."""
    ds = root / "ds"
    for sub in ("tob", "trades", "hours"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    book, tr = wide_pair(days=days, per_day=2)
    for x in pairs:
        book.write_parquet(ds / "tob" / f"{x}.parquet")
        tr.write_parquet(ds / "trades" / f"{x}.parquet")
        (ds / "hours" / f"{x}.csv").write_text(f"hour,status\n{first},clean\n{last},clean\n")
    return ds


def test_cli_dry_run_and_coverage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT", "B_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["A_USDT", "B_USDT"], "2026-08-01T00", "2026-10-31T23")
    out = tmp_path / "r" / "x.json"
    assert h003.main([str(ds), "--period", "exploration", "--universe", str(uni), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["range"] == ["2026-08-01", "2026-11-01"]
    assert doc["summary"]["80.0"]["outcome"].startswith("none: dry run")
    assert [r["counted"] for r in doc["results"]["80.0"]] == [True, True]
    assert set(doc["results"]) == {"80.0", "40.0", "120.0"}
    assert "decides nothing" in capsys.readouterr().out
    assert h003.main([str(ds), "--period", "test", "--universe", str(uni), "--out", str(out)]) == 2
    short = _dataset(tmp_path / "s", ["A_USDT", "B_USDT"], "2026-08-01T00", "2026-10-31T22")
    assert h003.main([str(short), "--period", "exploration", "--universe", str(uni), "--out", str(out)]) == 2
    late = _dataset(tmp_path / "l", ["A_USDT", "B_USDT"], "2026-08-01T01", "2026-10-31T23")
    assert h003.main([str(late), "--period", "exploration", "--universe", str(uni), "--out", str(out)]) == 2
    assert "do not cover" in capsys.readouterr().out


def test_months_without_trades() -> None:
    _, tr = wide_pair(days=20, per_day=2)  # 2026-08-01 .. 08-20
    assert h003.months_without_trades(tr, "2026-08-01", "2026-11-01") == ["2026-09", "2026-10"]
    assert h003.months_without_trades(tr, "2026-08-01", "2026-09-01") == []
    assert h003.months_without_trades(tr, "2026-11-01", "2027-01-01") == ["2026-11", "2026-12"]


def test_cli_refuses_a_month_without_trades(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["A_USDT"], "2026-11-01T00", "2026-11-30T23", days=92)  # no November trades
    assert (
        h003.main([str(ds), "--period", "test", "--universe", str(uni), "--out", str(tmp_path / "x.json")])
        == 2
    )
    assert "no trades in 2026-11" in capsys.readouterr().out


def test_cli_holdout_needs_a_decided_test(tmp_path: Path) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["A_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    ds = _dataset(tmp_path, ["H_USDT"], "2026-08-01T00", "2026-11-30T23")
    out = tmp_path / "h.json"
    base = [str(ds), "--period", "holdout", "--universe", str(uni), "--out", str(out)]
    assert h003.main(base) == 2
    dec = tmp_path / "d.json"
    for period, outcome, rc in [("test", "extend", 2), ("exploration", "accept", 2), ("test", "reject", 0)]:
        dec.write_text(json.dumps({"period": period, "summary": {"80.0": {"outcome": outcome}}}))
        assert h003.main([*base, "--decision", str(dec)]) == rc
    doc = json.loads(out.read_text())
    assert (doc["pairs"], doc["range"]) == (["H_USDT"], ["2026-08-01", "2026-12-01"])
    dec.write_text(json.dumps({"period": "extended", "summary": {"80.0": {"outcome": "accept"}}}))
    assert h003.main([*base, "--decision", str(dec)]) == 2  # the dataset ends in November, not December
