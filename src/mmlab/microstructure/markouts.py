"""Trade markouts against the reconstructed mid: effective spread, realized spread and price impact.

For a trade at t with price p, taker direction d (+1 buy, -1 sell) and quote value v = p q:
  ES   = d (p - m(t)) / m(t)
  RS_h = d (p - m(t + h)) / m(t)
  PI_h = d (m(t + h) - m(t)) / m(t)          ES = RS_h + PI_h
m(x) is the mid of the latest top-of-book row with availability time <= x (label + 100 ms). A trade is
dropped for a horizon if that row is invalid (anomaly or gap sentinel), one-sided or crossed, at t or
at t + h. Used by H-001 and H-002.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import polars as pl

HORIZONS_S: tuple[float, ...] = (0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 300)


def mid_table(tob: pl.DataFrame) -> pl.DataFrame:
    """avail_us, mid, micro, ok; one row per availability time (the last one wins)."""
    ok = (
        pl.col("valid")
        & pl.col("bid").is_not_null()
        & pl.col("ask").is_not_null()
        & (pl.col("bid") < pl.col("ask"))
    )
    micro = (pl.col("bid") * pl.col("ask_qty") + pl.col("ask") * pl.col("bid_qty")) / (
        pl.col("bid_qty") + pl.col("ask_qty")
    )
    return (
        tob.sort("avail_us", maintain_order=True)
        .unique(subset="avail_us", keep="last", maintain_order=True)
        .select(
            "avail_us",
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
            micro.alias("micro"),
            ok.fill_null(False).alias("ok"),
        )
    )


def _asof(trades: pl.DataFrame, mids: pl.DataFrame, at: str, suffix: str) -> pl.DataFrame:
    right = mids.rename(
        {"avail_us": f"_avail{suffix}", "mid": f"mid{suffix}", "micro": f"micro{suffix}", "ok": f"ok{suffix}"}
    )
    return (
        trades.sort(at)
        .join_asof(right, left_on=at, right_on=f"_avail{suffix}", strategy="backward")
        .drop(f"_avail{suffix}")
    )


def markouts(
    trades: pl.DataFrame, tob: pl.DataFrame, horizons_s: Sequence[float] = HORIZONS_S, price: str = "mid"
) -> pl.DataFrame:
    """One row per trade: ts_us, day, d, v, es, and rs_<h>/pi_<h> (null where the horizon is unusable)."""
    if price not in ("mid", "micro"):
        raise ValueError("price must be 'mid' or 'micro'")
    mids = mid_table(tob)
    df = trades.select(
        "ts_us",
        pl.col("taker").cast(pl.Float64).alias("d"),
        pl.col("price").alias("p"),
        (pl.col("price") * pl.col("qty")).alias("v"),
    )
    df = _asof(df, mids, "ts_us", "_0")
    m0 = pl.when(pl.col("ok_0")).then(pl.col(f"{price}_0"))
    df = df.with_columns(m0.alias("m0")).filter(pl.col("m0").is_not_null())
    out = df.with_columns(
        (pl.col("d") * (pl.col("p") - pl.col("m0")) / pl.col("m0")).alias("es"),
        pl.from_epoch(pl.col("ts_us"), time_unit="us").dt.date().alias("day"),
    )
    for h in horizons_s:
        key = _hkey(h)
        shifted = out.with_columns((pl.col("ts_us") + round(h * 1e6)).alias("_th"))
        joined = _asof(shifted, mids, "_th", "_h").sort("ts_us", maintain_order=True)
        mh = pl.when(pl.col("ok_h")).then(pl.col(f"{price}_h"))
        out = joined.with_columns(
            (pl.col("d") * (pl.col("p") - mh) / pl.col("m0")).alias(f"rs_{key}"),
            (pl.col("d") * (mh - pl.col("m0")) / pl.col("m0")).alias(f"pi_{key}"),
        ).drop("_th", "mid_h", "micro_h", "ok_h")
    return out.select("ts_us", "day", "d", "v", "es", *[c for c in out.columns if c[:3] in ("rs_", "pi_")])


def _hkey(h: float) -> str:
    return f"{h:g}s".replace(".", "p")


def in_period(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """Rows with start <= ts < end (UTC dates, end exclusive)."""
    lo = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1e6)
    hi = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1e6)
    return df.filter((pl.col("ts_us") >= lo) & (pl.col("ts_us") < hi))


def daily_sums(df: pl.DataFrame, column: str) -> tuple[list[float], list[float]]:
    """Per day: sum(v * x) and sum(v) over rows where x is not null (inputs to ratio bootstraps)."""
    d = (
        df.filter(pl.col(column).is_not_null())
        .group_by("day")
        .agg((pl.col("v") * pl.col(column)).sum().alias("num"), pl.col("v").sum().alias("den"))
        .sort("day")
    )
    return d["num"].to_list(), d["den"].to_list()
