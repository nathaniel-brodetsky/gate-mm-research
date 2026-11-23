"""Realized spread of trades by the book state just before them (exploration for H-003).

For each trade, from the last valid top of book known at the trade time (availability rule as in
`markouts`):
  q   = (ask - bid) / (2 mid)                quoted half-spread
  rho = maker-side best quantity / (bid quantity + ask quantity)
        (the maker side is the ask for a taker buy and the bid for a taker sell; a small rho means the
        maker sits on the thin side of the book)
and, as in H-001, ES = d (p - m) / m and RS(60 s) = d (p - m(t + 60 s)) / m, with theta = RS - maker fee.

A maker can know q and rho before a trade, so a positive theta in some bucket is a candidate for a
quoting rule. This module only describes; it tests nothing. It refuses the time holdout (from
2026-11-01) and pairs outside the research list of U-001.

Usage:
  python -m mmlab.microstructure.conditional data/processed/U001 --out data/results/h003_explore.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import polars as pl

from mmlab.microstructure.markouts import in_period

HORIZON_US = 60_000_000
MAKER_FEE = 0.0010  # global spot VIP 0, as in H-001
PERIODS = {
    "development": ("2026-08-01", "2026-10-01"),
    "validation": ("2026-10-01", "2026-11-01"),
    "all": ("2026-08-01", "2026-11-01"),
}
TIME_HOLDOUT_START = "2026-11-01"
Q_EDGES_BP: tuple[float, ...] = (0, 5, 10, 20, 40, 80, math.inf)
RHO_EDGES: tuple[float, ...] = (0, 0.2, 0.4, 0.6, 0.8, 1.0)  # rho < 1 while both sides exist
MIN_TRADES = 30
MIN_DAYS = 5


def book_table(tob: pl.DataFrame) -> pl.DataFrame:
    """avail_us, bid, ask, bid_qty, ask_qty, ok; one row per availability time (the last one wins)."""
    ok = (
        pl.col("valid")
        & pl.col("bid").is_not_null()
        & pl.col("ask").is_not_null()
        & (pl.col("bid") < pl.col("ask"))
    )
    return (
        tob.sort("avail_us", maintain_order=True)
        .unique(subset="avail_us", keep="last", maintain_order=True)
        .select("avail_us", "bid", "ask", "bid_qty", "ask_qty", ok.fill_null(False).alias("ok"))
    )


def trade_states(trades: pl.DataFrame, tob: pl.DataFrame) -> pl.DataFrame:
    """One row per trade with a valid book at t: ts_us, day, d, v, q, rho, es, rs_60s.

    rs_60s is null where the book at t + 60 s is invalid, one-sided or crossed.
    """
    book = book_table(tob)
    df = trades.with_row_index("i").select(
        "i",
        "ts_us",
        pl.col("taker").cast(pl.Float64).alias("d"),
        pl.col("price").alias("p"),
        (pl.col("price") * pl.col("qty")).alias("v"),
    )
    at_t = (
        df.sort("ts_us", maintain_order=True)
        .join_asof(book.rename({"avail_us": "_a"}), left_on="ts_us", right_on="_a", strategy="backward")
        .filter(pl.col("ok").fill_null(False))
    )
    mids = book.select(
        "avail_us", ((pl.col("bid") + pl.col("ask")) / 2).alias("m_h"), pl.col("ok").alias("ok_h")
    )
    at_h = (
        df.select("i", (pl.col("ts_us") + HORIZON_US).alias("_th"))
        .sort("_th", maintain_order=True)
        .join_asof(mids, left_on="_th", right_on="avail_us", strategy="backward")
        .select("i", pl.when(pl.col("ok_h")).then(pl.col("m_h")).alias("m_h"))
    )
    m0 = (pl.col("bid") + pl.col("ask")) / 2
    maker_qty = pl.when(pl.col("d") > 0).then(pl.col("ask_qty")).otherwise(pl.col("bid_qty"))
    return (
        at_t.join(at_h, on="i", how="left")
        .with_columns(m0.alias("m0"))
        .select(
            "ts_us",
            pl.from_epoch(pl.col("ts_us"), time_unit="us").dt.date().alias("day"),
            "d",
            "v",
            ((pl.col("ask") - pl.col("bid")) / (2 * pl.col("m0"))).alias("q"),
            (maker_qty / (pl.col("bid_qty") + pl.col("ask_qty"))).alias("rho"),
            (pl.col("d") * (pl.col("p") - pl.col("m0")) / pl.col("m0")).alias("es"),
            (pl.col("d") * (pl.col("p") - pl.col("m_h")) / pl.col("m0")).alias("rs_60s"),
        )
        .sort("ts_us", maintain_order=True)
    )


def bucket_labels(edges: Sequence[float], scale: float, unit: str) -> list[str]:
    def fmt(x: float) -> str:
        return "inf" if math.isinf(x) else f"{x * scale:g}"

    return [f"[{fmt(a)}, {fmt(b)}){unit}" for a, b in itertools.pairwise(edges)]


@dataclass
class Cell:
    period: str
    pair: str
    variable: str  # "q" or "rho"
    bucket: str
    trades: int
    days: int
    usdt_per_day: float  # traded quote value per calendar day of the period
    es: float | None
    rs60: float | None
    theta: float | None


def cells(states: pl.DataFrame, pair: str, period: str, n_days: int) -> list[Cell]:
    out = []
    for var, edges, scale, unit in (("q", Q_EDGES_BP, 1.0, " bp"), ("rho", RHO_EDGES, 1.0, "")):
        # rounded so that a spread of exactly one edge lands in the upper bucket despite float error
        col = (pl.col("q") * 1e4).round(6) if var == "q" else pl.col("rho").round(9)
        for (lo, hi), label in zip(itertools.pairwise(edges), bucket_labels(edges, scale, unit), strict=True):
            sub = states.filter((col >= lo) & (col < hi))
            rs = sub.filter(pl.col("rs_60s").is_not_null())
            v_all, v_rs = float(sub["v"].sum()), float(rs["v"].sum())
            es = float((sub["v"] * sub["es"]).sum()) / v_all if v_all else None
            rs60 = float((rs["v"] * rs["rs_60s"]).sum()) / v_rs if v_rs else None
            out.append(
                Cell(
                    period,
                    pair,
                    var,
                    label,
                    sub.height,
                    sub["day"].n_unique() if sub.height else 0,
                    v_all / n_days,
                    es,
                    rs60,
                    None if rs60 is None else rs60 - MAKER_FEE,
                )
            )
    return out


@dataclass
class Pooled:
    period: str
    variable: str
    bucket: str
    pairs: int  # pairs with enough trades in the bucket
    trades: int  # all pairs
    usdt_per_day: float  # all pairs
    theta_median: float | None  # over the pairs counted in `pairs`
    pairs_theta_pos: int


def _period_rank(period: str) -> int:
    return list(PERIODS).index(period) if period in PERIODS else len(PERIODS)


def _lower(bucket: str) -> float:
    return float(bucket[1:].split(",")[0])


def pooled(rows: Sequence[Cell]) -> list[Pooled]:
    """Per period, variable and bucket: pairs with enough trades, median theta, pairs with theta > 0."""
    out = []
    keys = sorted(
        {(c.period, c.variable, c.bucket) for c in rows},
        key=lambda k: (_period_rank(k[0]), k[1], _lower(k[2])),
    )
    for period, var, bucket in keys:
        cs = [c for c in rows if (c.period, c.variable, c.bucket) == (period, var, bucket)]
        thetas = [
            c.theta for c in cs if c.trades >= MIN_TRADES and c.days >= MIN_DAYS and c.theta is not None
        ]
        out.append(
            Pooled(
                period,
                var,
                bucket,
                len(thetas),
                sum(c.trades for c in cs),
                sum(c.usdt_per_day for c in cs),
                statistics.median(thetas) if thetas else None,
                sum(t > 0 for t in thetas),
            )
        )
    return out


def report(pool: Sequence[Pooled]) -> str:
    lines = [
        f"pairs = pairs with >= {MIN_TRADES} trades on >= {MIN_DAYS} days in the bucket; theta = RS(60 s) - "
        "10 bp, median over those pairs, in bp; trades and USDT/day summed over all pairs",
        f"{'period':12} {'var':4} {'bucket':16} {'pairs':>5} {'trades':>8} {'USDT/day':>11} "
        f"{'theta med':>9} {'theta>0':>7}",
    ]
    for r in pool:
        th = "n/a" if r.theta_median is None else f"{r.theta_median * 1e4:.1f}"
        lines.append(
            f"{r.period:12} {r.variable:4} {r.bucket:16} {r.pairs:5d} {r.trades:8d} {r.usdt_per_day:11.0f} "
            f"{th:>9} {r.pairs_theta_pos:7d}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Realized spread by book state before the trade (exploration).")
    p.add_argument("dataset")
    p.add_argument("--universe", default="configs/universe_U001.json")
    p.add_argument("--periods", default="development,validation,all")
    p.add_argument("--out", required=True, help="per-pair CSV")
    a = p.parse_args(argv)
    periods = [x.strip() for x in a.periods.split(",") if x.strip()]
    bad = [x for x in periods if x not in PERIODS]
    if bad:
        print(f"unknown periods: {bad}; choose from {sorted(PERIODS)}")
        return 2
    uni = json.loads(Path(a.universe).read_text())
    pairs = [x for s in sorted(uni["research"], key=int) for x in uni["research"][s]]
    ds = Path(a.dataset)
    missing = [x for x in pairs if not (ds / "tob" / f"{x}.parquet").exists()]
    if missing:
        print(f"dataset incomplete, missing: {missing}")
        return 2
    rows: list[Cell] = []
    for x in pairs:
        st = trade_states(
            pl.read_parquet(ds / "trades" / f"{x}.parquet"), pl.read_parquet(ds / "tob" / f"{x}.parquet")
        )
        for period in periods:
            start, end = PERIODS[period]
            n_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            rows += cells(in_period(st, start, end), x, period, n_days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        w.writeheader()
        w.writerows(asdict(c) for c in rows)
    print(report(pooled(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
