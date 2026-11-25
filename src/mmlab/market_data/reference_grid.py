"""One-second as-of price grid of a reference market (e.g. spot BTC_USDT) from Gate's monthly deals.

Row g (an epoch second) holds `price_before`, the price of the last trade with ts < g * 1e6 (NaN before
the first known trade), and `trades`, the number of trades in [g, g + 1) s. Only the trades files are
read, no order books: the grid is a reference for lead-lag, not a market to quote. A month whose file
is missing is refused rather than bridged with a stale price.

Holdout guard: periods reaching 2026-11-01 or later are refused unless --open-time-holdout is given
(H-001, H-002, D-001).

Usage:
  python -m mmlab.market_data.reference_grid BTC_USDT data/processed/reference \\
      --start 2026-08-01 --end 2026-11-01
Writes {out}/{pair}.parquet with columns second, price_before, trades.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mmlab.data_ingestion.gate_archive import BASE_URL, archive_ref
from mmlab.data_ingestion.gate_download import fetch
from mmlab.data_ingestion.gate_trades_file import Trade, iter_trades_file
from mmlab.feature_engineering.asof_activity import epoch_s

HOLDOUT_START = "2026-11-01"


def months(start: str, end: str) -> list[str]:
    """YYYYMM of every month that [start, end) touches."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    out, y, m = [], a.year, a.month
    while (y, m) < (b.year, b.month) or (b.day > 1 and (y, m) == (b.year, b.month)):
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def grid_from_trades(
    trades: Iterable[Trade], g0: int, n: int
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """(price_before, trades) for the seconds g0 .. g0 + n - 1; trades before g0 seed the first price."""
    last_ts = np.full(n, -1, dtype=np.int64)
    last_px = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int32)
    seed_ts, seed_px = -1, np.nan
    for t in trades:
        s = t.ts_us // 1_000_000 - g0
        if s < 0:
            if t.ts_us >= seed_ts:
                seed_ts, seed_px = t.ts_us, float(t.price)
        elif s < n:
            count[s] += 1
            if t.ts_us >= last_ts[s]:
                last_ts[s] = t.ts_us
                last_px[s] = float(t.price)
    px = np.concatenate([[seed_px], last_px])  # px[g]: last price in second g - 1 (px[0]: the seed)
    idx = np.maximum.accumulate(np.where(np.isfinite(px), np.arange(n + 1), -1))
    filled = np.where(idx >= 0, px[np.maximum(idx, 0)], np.nan)
    return np.asarray(filled[:n], dtype=np.float64), count


def build(
    pair: str, start: str, end: str, raw_root: Path, base: str = BASE_URL, retries: int = 5
) -> pl.DataFrame:
    paths = []
    for ym in months(start, end):
        p = fetch(archive_ref("spot", "deals", pair, ym, base=base), raw_root, retries=retries)
        if p is None:
            raise FileNotFoundError(f"no deals file for {pair} {ym}: refusing to bridge the month")
        paths.append(p)
    g0 = epoch_s(start)
    n = epoch_s(end) - g0
    stream = (t for p in paths for t in iter_trades_file(p, "spot", None, (g0 + n) * 1_000_000))
    price, count = grid_from_trades(stream, g0, n)
    return pl.DataFrame(
        {"second": np.arange(g0, g0 + n, dtype=np.int64), "price_before": price, "trades": count}
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="One-second as-of price grid of a reference market.")
    p.add_argument("pair")
    p.add_argument("out")
    p.add_argument("--start", default="2026-08-01")
    p.add_argument("--end", default="2026-11-01", help="exclusive")
    p.add_argument("--raw-dir", default="data/raw/gate")
    p.add_argument("--base", default=BASE_URL)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--open-time-holdout", action="store_true")
    a = p.parse_args(argv)
    if a.end > HOLDOUT_START and not a.open_time_holdout:
        print(f"refused: the period reaches the time holdout ({HOLDOUT_START}); see --open-time-holdout")
        return 2
    t0 = time.monotonic()
    df = build(a.pair, a.start, a.end, Path(a.raw_dir), a.base, a.retries)
    out = Path(a.out) / f"{a.pair}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(out)
    px = df["price_before"].drop_nans()
    first, last = (float(px[0]), float(px[-1])) if px.len() else (math.nan, math.nan)
    print(
        f"{a.pair} {a.start} .. {a.end}: {df.height} seconds, {int(df['trades'].sum())} trades, "
        f"{int((df['trades'] > 0).sum())} seconds with a trade, {df.height - px.len()} seconds before the "
        f"first price; price {first:g} .. {last:g}; {time.monotonic() - t0:.0f} s -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
