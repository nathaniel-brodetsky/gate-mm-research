"""As-of state of the market before each trade (stage 3, step 4).

Every feature of a trade at t uses only information available before t - latency: books with
avail_us < t - latency, trades with ts_us < t - latency. Features signed with the taker direction d
(+1 buy, -1 sell) describe the quote that was hit: positive means the state pointed the taker's way,
against the maker.

- signed_imbalance: d * (bid_qty - ask_qty) / (bid_qty + ask_qty) of the last valid top of book;
- signed_flow(W): d * (taker buy volume - taker sell volume) / volume over [t - latency - W, t - latency);
  NaN without trades in the window;
- since_last_trade: seconds from the last trade before t - latency to t - latency; inf if none.

`taker_sides` recovers d for the rows of `mmlab.evaluation.h003.states` (which does not return it) by
repeating its filter, and refuses if the rows do not line up exactly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.microstructure.conditional import book_table


def taker_sides(trades: pl.DataFrame, tob: pl.DataFrame, st: pl.DataFrame) -> NDArray[np.float64]:
    """Taker direction of every row of states(trades, tob), in its order."""
    book = book_table(tob)
    rows = (
        trades.select(
            "ts_us", pl.col("taker").cast(pl.Float64).alias("d"), (pl.col("price") * pl.col("qty")).alias("v")
        )
        .sort("ts_us", maintain_order=True)
        .join_asof(
            book.select(pl.col("avail_us").alias("_a"), "ok"),
            left_on="ts_us",
            right_on="_a",
            strategy="backward",
        )
        .filter(pl.col("ok").fill_null(False))
    )
    if rows.height != st.height or not (
        np.array_equal(rows["ts_us"].to_numpy(), st["ts_us"].to_numpy())
        and np.array_equal(rows["v"].to_numpy(), st["v"].to_numpy())
    ):
        raise ValueError("trades do not line up with states(); refusing to guess the sides")
    return rows["d"].to_numpy()


def _book_before(tob: pl.DataFrame, t_us: NDArray[np.int64]) -> tuple[NDArray[np.int64], pl.DataFrame]:
    book = book_table(tob).filter(pl.col("ok"))
    idx = np.searchsorted(book["avail_us"].to_numpy(), t_us, side="left") - 1
    return idx, book


def signed_imbalance(
    tob: pl.DataFrame, t_us: NDArray[np.int64], d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """d * top-of-book imbalance of the last valid book with avail_us < t_us; NaN before the first."""
    idx, book = _book_before(tob, t_us)
    bq = book["bid_qty"].to_numpy()
    aq = book["ask_qty"].to_numpy()
    out = np.full(t_us.size, np.nan)
    has = idx >= 0
    i = idx[has]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[has] = d[has] * (bq[i] - aq[i]) / (bq[i] + aq[i])
    return out


def signed_flow(
    all_ts: NDArray[np.int64],
    all_side: NDArray[np.float64],
    all_v: NDArray[np.float64],
    t_us: NDArray[np.int64],
    d: NDArray[np.float64],
    w_s: int,
) -> NDArray[np.float64]:
    """d * (buy - sell) / (buy + sell) taker volume of trades in [t_us - w, t_us); all_ts sorted ascending."""
    cum_v = np.concatenate([[0.0], np.cumsum(all_v)])
    cum_s = np.concatenate([[0.0], np.cumsum(all_v * all_side)])
    hi = np.searchsorted(all_ts, t_us, side="left")
    lo = np.searchsorted(all_ts, t_us - w_s * 1_000_000, side="left")
    vol = cum_v[hi] - cum_v[lo]
    net = cum_s[hi] - cum_s[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(vol > 0, d * net / vol, np.nan), dtype=np.float64)


def since_last_trade(all_ts: NDArray[np.int64], t_us: NDArray[np.int64]) -> NDArray[np.float64]:
    """Seconds from the last trade with ts < t_us to t_us; inf if there is none."""
    k = np.searchsorted(all_ts, t_us, side="left") - 1
    out = np.full(t_us.size, np.inf)
    has = k >= 0
    out[has] = (t_us[has] - all_ts[k[has]]) / 1e6
    return out
