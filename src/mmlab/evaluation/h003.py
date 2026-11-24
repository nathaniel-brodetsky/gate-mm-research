"""Test of H-003: realized spread of trades at a quoted half-spread of at least 80 bp.

Implements hypotheses/H-003-wide-spread-realized-spread.md as registered on 2026-11-24 (commit 8b2153d).

Periods:
  exploration  2026-08-01 .. 2026-10-31, research pairs: a dry run on data already used to pick the
               threshold; it decides nothing.
  test         2026-11-01 .. 2026-11-30, research pairs: the registered decision.
  extended     2026-11-01 .. 2026-12-31, research pairs: only if the test had fewer than 8 counted pairs.
  holdout      holdout pairs, 2026-08-01 to the end of the deciding period: a one-time estimate, allowed
               only with the decision file of the test (outcome accept or reject).

Usage:
  python -m mmlab.evaluation.h003 data/processed/U001_2026-11 --period test --out results/h003/test.json
  python -m mmlab.evaluation.h003 data/processed/U001_holdout --period holdout \\
      --decision results/h003/test.json --out results/h003/holdout.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices
from mmlab.evaluation.multiple_testing import benjamini_hochberg
from mmlab.microstructure.conditional import book_table
from mmlab.microstructure.markouts import daily_sums, in_period

THRESHOLD_BP = 80.0
SENSITIVITY_BP = (40.0, 120.0)
MAKER_FEE = 0.0010
MIN_TRADES = 30
MIN_DAYS = 5
MIN_PAIRS = 8
RESAMPLES = 10_000
MEAN_BLOCK_DAYS = 3.0
SEED_PAIRS = 20261125
SEED_DAYS = 20261126
BH_Q = 0.10
HORIZONS_S = (10, 60, 300)
PERIODS = {
    "exploration": ("2026-08-01", "2026-11-01"),
    "test": ("2026-11-01", "2026-12-01"),
    "extended": ("2026-11-01", "2027-01-01"),
}
HOLDOUT_START = "2026-08-01"


def states(trades: pl.DataFrame, tob: pl.DataFrame) -> pl.DataFrame:
    """One row per trade with a valid book at t.

    Columns: ts_us, day, v, q, es, rs_10s, rs_60s, rs_300s (mid), rs_micro_60s (microprice at t and
    t + 60 s), rs_exit_60s (against the best ask after 60 s for a taker buy, the best bid for a sell).
    A horizon column is null where the book at t + h is invalid, one-sided or crossed.
    """
    book = book_table(tob)
    df = trades.with_row_index("i").select(
        "i",
        "ts_us",
        pl.col("taker").cast(pl.Float64).alias("d"),
        pl.col("price").alias("p"),
        (pl.col("price") * pl.col("qty")).alias("v"),
    )
    out = (
        df.sort("ts_us", maintain_order=True)
        .join_asof(book.rename({"avail_us": "_a"}), left_on="ts_us", right_on="_a", strategy="backward")
        .filter(pl.col("ok").fill_null(False))
        .with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("m0"),
            (
                (pl.col("bid") * pl.col("ask_qty") + pl.col("ask") * pl.col("bid_qty"))
                / (pl.col("bid_qty") + pl.col("ask_qty"))
            ).alias("micro0"),
        )
    )
    cols = [
        pl.col("ts_us"),
        pl.from_epoch(pl.col("ts_us"), time_unit="us").dt.date().alias("day"),
        pl.col("v"),
        ((pl.col("ask") - pl.col("bid")) / (2 * pl.col("m0"))).alias("q"),
        (pl.col("d") * (pl.col("p") - pl.col("m0")) / pl.col("m0")).alias("es"),
    ]
    for h in HORIZONS_S:
        at_h = (
            df.select("i", "d", (pl.col("ts_us") + h * 1_000_000).alias("_th"))
            .sort("_th", maintain_order=True)
            .join_asof(book, left_on="_th", right_on="avail_us", strategy="backward")
            .select(
                "i",
                pl.when(pl.col("ok")).then((pl.col("bid") + pl.col("ask")) / 2).alias(f"mid_{h}"),
                pl.when(pl.col("ok"))
                .then(
                    (pl.col("bid") * pl.col("ask_qty") + pl.col("ask") * pl.col("bid_qty"))
                    / (pl.col("bid_qty") + pl.col("ask_qty"))
                )
                .alias(f"micro_{h}"),
                pl.when(pl.col("ok"))
                .then(pl.when(pl.col("d") > 0).then(pl.col("ask")).otherwise(pl.col("bid")))
                .alias(f"exit_{h}"),
            )
        )
        out = out.join(at_h, on="i", how="left")
        cols.append((pl.col("d") * (pl.col("p") - pl.col(f"mid_{h}")) / pl.col("m0")).alias(f"rs_{h}s"))
    cols += [
        (pl.col("d") * (pl.col("p") - pl.col("micro_60")) / pl.col("micro0")).alias("rs_micro_60s"),
        (pl.col("d") * (pl.col("p") - pl.col("exit_60")) / pl.col("m0")).alias("rs_exit_60s"),
    ]
    return out.select(cols).sort("ts_us", maintain_order=True)


def eligible(st: pl.DataFrame, threshold_bp: float) -> pl.DataFrame:
    """Trades with q >= threshold, q in bp rounded to 6 decimals first (as in the exploration)."""
    return st.filter((pl.col("q") * 1e4).round(6) >= threshold_bp)


def _vw(df: pl.DataFrame, col: str) -> float | None:
    d = df.filter(pl.col(col).is_not_null())
    den = float(d["v"].sum()) if d.height else 0.0
    return float((d["v"] * d[col]).sum()) / den if den else None


@dataclass
class PairResult:
    pair: str
    threshold_bp: float
    trades: int  # eligible trades with RS(60 s) defined
    days: int
    counted: bool
    theta: float | None = None  # RS(60 s) - maker fee, volume-weighted
    ci: tuple[float, float] | None = None
    p: float = 1.0
    usdt_per_day: float = 0.0  # all eligible trades, per calendar day
    maker_net_per_day: float | None = None  # theta * usdt_per_day
    rs: dict[str, float | None] = field(default_factory=dict)
    theta_micro: float | None = None
    theta_trade_median: float | None = None
    theta_exit: float | None = None  # exit markout - maker fee, before the taker fee
    theta_all_trades: float | None = None  # baseline: every trade, as H-001


def evaluate_pair(pair: str, st: pl.DataFrame, threshold_bp: float, n_days: int) -> PairResult:
    el = eligible(st, threshold_bp)
    rs60 = el.filter(pl.col("rs_60s").is_not_null())
    days = rs60["day"].n_unique() if rs60.height else 0
    res = PairResult(pair, threshold_bp, rs60.height, days, rs60.height >= MIN_TRADES and days >= MIN_DAYS)
    res.usdt_per_day = float(el["v"].sum()) / n_days
    base = _vw(st, "rs_60s")
    res.theta_all_trades = None if base is None else base - MAKER_FEE
    if rs60.height == 0:
        return res
    for h in HORIZONS_S:
        res.rs[f"{h}s"] = _vw(el, f"rs_{h}s")
    res.theta = float(_vw(rs60, "rs_60s") or 0.0) - MAKER_FEE
    res.maker_net_per_day = res.theta * res.usdt_per_day
    micro, exit_ = _vw(el, "rs_micro_60s"), _vw(el, "rs_exit_60s")
    res.theta_micro = None if micro is None else micro - MAKER_FEE
    res.theta_exit = None if exit_ is None else exit_ - MAKER_FEE
    res.theta_trade_median = float(rs60.select(pl.col("rs_60s").median()).item()) - MAKER_FEE
    num, den = daily_sums(rs60, "rs_60s")
    if len(num) >= 2:
        boot = ratio_resamples(
            np.asarray(num),
            np.asarray(den),
            stationary_indices(len(num), RESAMPLES, MEAN_BLOCK_DAYS, SEED_DAYS),
        )
        boot = boot[~np.isnan(boot)] - MAKER_FEE
        res.ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        res.p = float((1 + np.sum(boot <= 0)) / (boot.size + 1))
    return res


def median_ci(values: Sequence[float]) -> tuple[float, tuple[float, float]] | None:
    """Median and 95% interval from a bootstrap over pairs (seed SEED_PAIRS)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(SEED_PAIRS)
    meds = np.median(v[rng.integers(0, v.size, (RESAMPLES, v.size))], axis=1)
    return float(np.median(v)), (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def summarize(results: Sequence[PairResult], period: str) -> dict[str, Any]:
    """Cross-pair summary of one threshold; the outcome is the registered rule (threshold 80 bp only)."""
    counted = [r for r in results if r.counted and r.theta is not None]
    rejects = benjamini_hochberg([r.p for r in counted], BH_Q) if counted else []
    out: dict[str, Any] = {
        "pairs_counted": len(counted),
        "bh_rejections": [r.pair for r, x in zip(counted, rejects, strict=True) if x],
        "usdt_per_day_all_pairs": sum(r.usdt_per_day for r in results),
        "maker_net_per_day_counted_pairs": sum(r.maker_net_per_day or 0.0 for r in counted),
    }
    mc = median_ci([r.theta for r in counted if r.theta is not None])
    if mc is not None:
        out["median_theta"], out["ci"] = mc
    if period == "exploration":
        out["outcome"] = "none: dry run on the data used to pick the threshold"
    elif period == "holdout":
        out["outcome"] = "none: one-time estimate on the holdout pairs"
    elif len(counted) < MIN_PAIRS:
        out["outcome"] = "extend" if period == "test" else "undecided"
    else:
        out["outcome"] = "accept" if out["ci"][0] > 0 else "reject"
    return out


def hour_coverage(path: Path) -> tuple[str, str] | None:
    """First and last hour label of a dataset's hours/{pair}.csv."""
    with path.open() as fh:
        hours = [r["hour"] for r in csv.DictReader(fh)]
    return (hours[0], hours[-1]) if hours else None


def months_without_trades(trades: pl.DataFrame, start: str, end: str) -> list[str]:
    """Calendar months of [start, end) with no trade: usually a monthly deals file not yet published."""
    seen = set(
        in_period(trades, start, end)
        .select(pl.from_epoch(pl.col("ts_us"), time_unit="us").dt.strftime("%Y-%m").unique())
        .to_series()
        .to_list()
    )
    months, t = [], datetime.fromisoformat(start)
    while t < datetime.fromisoformat(end):
        months.append(t.strftime("%Y-%m"))
        t = (t.replace(day=1) + timedelta(days=32)).replace(day=1)
    return [m for m in months if m not in seen]


def _label(day: str, hours_back: int = 0) -> str:
    t = datetime.fromisoformat(day).replace(tzinfo=UTC) - timedelta(hours=hours_back)
    return t.strftime("%Y-%m-%dT%H")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Test of H-003 on a built dataset.")
    p.add_argument("dataset")
    p.add_argument("--period", required=True, choices=[*PERIODS, "holdout"])
    p.add_argument("--universe", default="configs/universe_U001.json")
    p.add_argument("--decision", help="decision JSON of the test; required for --period holdout")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    uni = json.loads(Path(a.universe).read_text())
    if a.period == "holdout":
        if not a.decision:
            print("refused: the holdout pairs need the decision file of the test (--decision)")
            return 2
        dec = json.loads(Path(a.decision).read_text())
        if dec.get("period") not in ("test", "extended") or dec["summary"][str(THRESHOLD_BP)][
            "outcome"
        ] not in (
            "accept",
            "reject",
        ):
            print("refused: the decision file does not hold an accepted or rejected test")
            return 2
        start, end = HOLDOUT_START, PERIODS[dec["period"]][1]
        part = "holdout"
    else:
        start, end = PERIODS[a.period]
        part = "research"
    pairs = [x for s in sorted(uni[part], key=int) for x in uni[part][s]]
    ds = Path(a.dataset)
    problems = []
    for x in pairs:
        cov = hour_coverage(ds / "hours" / f"{x}.csv") if (ds / "hours" / f"{x}.csv").exists() else None
        if not (ds / "tob" / f"{x}.parquet").exists() or cov is None:
            problems.append(f"{x}: missing")
        elif cov[0] > _label(start) or cov[1] < _label(end, 1):
            problems.append(f"{x}: hours {cov[0]} .. {cov[1]} do not cover {start} .. {end}")
        elif empty := months_without_trades(pl.read_parquet(ds / "trades" / f"{x}.parquet"), start, end):
            problems.append(f"{x}: no trades in {', '.join(empty)} (monthly deals file not published yet?)")
    if problems:
        print("dataset does not cover the period:\n  " + "\n  ".join(problems))
        return 2
    n_days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    per_threshold: dict[str, list[PairResult]] = {str(t): [] for t in (THRESHOLD_BP, *SENSITIVITY_BP)}
    for x in pairs:
        st = in_period(
            states(
                pl.read_parquet(ds / "trades" / f"{x}.parquet"), pl.read_parquet(ds / "tob" / f"{x}.parquet")
            ),
            start,
            end,
        )
        for t in (THRESHOLD_BP, *SENSITIVITY_BP):
            per_threshold[str(t)].append(evaluate_pair(x, st, t, n_days))
    summary = {t: summarize(rs, a.period) for t, rs in per_threshold.items()}
    for r in per_threshold[str(THRESHOLD_BP)]:
        print(_line(r))
    doc = {
        "hypothesis": "H-003",
        "period": a.period,
        "range": [start, end],
        "pairs": pairs,
        "params": {
            "threshold_bp": THRESHOLD_BP,
            "sensitivity_bp": list(SENSITIVITY_BP),
            "maker_fee": MAKER_FEE,
            "min_trades": MIN_TRADES,
            "min_days": MIN_DAYS,
            "min_pairs": MIN_PAIRS,
            "resamples": RESAMPLES,
            "mean_block_days": MEAN_BLOCK_DAYS,
            "seeds": [SEED_PAIRS, SEED_DAYS],
            "bh_q": BH_Q,
        },
        "results": {t: [asdict(r) for r in rs] for t, rs in per_threshold.items()},
        "summary": summary,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print("bp = basis points; theta = RS(60 s) - 10 bp over trades at q >= threshold")
    for key, s in summary.items():
        print(f"threshold {key} bp: " + json.dumps(s, default=str))
    if a.period == "exploration":
        print("exploration period: a dry run on data already used to pick the threshold; it decides nothing")
    return 0


def _bp(x: float | None) -> str:
    return "     n/a" if x is None else f"{x * 1e4:8.1f}"


def _line(r: PairResult) -> str:
    return (
        f"{r.pair:16} trades {r.trades:6} days {r.days:3} {'counted' if r.counted else '       '} | "
        f"theta {_bp(r.theta)} p {r.p:.4f} | exit {_bp(r.theta_exit)} | "
        f"all trades {_bp(r.theta_all_trades)} | "
        f"USDT/day {r.usdt_per_day:9.0f}"
    )


if __name__ == "__main__":
    sys.exit(main())
