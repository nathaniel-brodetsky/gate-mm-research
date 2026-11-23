"""Parsers for Gate monthly `deals` (spot), `trades` (futures) and `funding_applies` archive files.

Format as verified on archive samples during the stage-1 audit (2026-11-23), not taken from Gate docs:

  spot deals:       ts, trade_id, price, amount, side     taker side: 1 = buy, 2 = sell (see below)
  futures trades:   ts, trade_id, price, size             sign of size = taker side: + buy, - sell
  funding_applies:  ts, rate                              ts in whole seconds

Trade timestamps carry microseconds. Taker sides were verified by matching trades to `take` rows of
the order book within +-1 s (taker buys hit asks, taker sells hit bids): futures GPS_USDT 1052/1054,
spot LUNC_USDT 711/711, RAIN_USDT 912/915 on 2026-10-15. For spot this is the OPPOSITE of the published
"1: sell, 2: buy", which therefore describes the resting (maker) order; read as taker side it matched
only 19% and 9% of trades.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

from mmlab.data_ingestion.gate_orderbook_file import Kind, parse_decimal, parse_ts_us

Taker = Literal["buy", "sell"]
_SPOT_TAKER: Final[dict[str, Taker]] = {"1": "buy", "2": "sell"}


@dataclass(frozen=True, slots=True)
class Trade:
    ts_us: int
    trade_id: int
    price: Decimal
    qty: Decimal  # positive; spot: base amount, futures: contracts
    taker: Taker


@dataclass(frozen=True, slots=True)
class FundingApply:
    ts_us: int
    rate: Decimal


def parse_trade_line(fields: list[str], kind: Kind) -> Trade:
    if kind == "spot":
        if len(fields) != 5:
            raise ValueError(f"spot deal needs 5 fields, got {len(fields)}: {fields}")
        ts, tid, px, amt, side = fields
        try:
            taker = _SPOT_TAKER[side.strip()]
        except KeyError:
            raise ValueError(f"bad spot side: {side!r}") from None
        qty = parse_decimal(amt)
    else:
        if len(fields) != 4:
            raise ValueError(f"futures trade needs 4 fields, got {len(fields)}: {fields}")
        ts, tid, px, sz = fields
        signed = parse_decimal(sz)
        if signed == 0:
            raise ValueError("futures trade size 0 carries no side")
        taker = "buy" if signed > 0 else "sell"
        qty = abs(signed)
    price = parse_decimal(px)
    if price <= 0 or qty <= 0:
        raise ValueError(f"non-positive price or quantity: {fields}")
    trade_id = int(tid)
    if trade_id < 0:
        raise ValueError(f"bad trade id: {tid!r}")
    return Trade(parse_ts_us(ts), trade_id, price, qty, taker)


def _lines(path: Path) -> Iterator[tuple[int, list[str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                yield lineno, line.rstrip("\r\n").split(",")


def iter_trades_file(
    path: str | Path, kind: Kind, start_us: int | None = None, end_us: int | None = None
) -> Iterator[Trade]:
    """Trades with start_us <= ts < end_us (bounds optional)."""
    p = Path(path)
    for lineno, fields in _lines(p):
        try:
            t = parse_trade_line(fields, kind)
        except ValueError as e:
            raise ValueError(f"{p.name}:{lineno}: {e}") from None
        if (start_us is None or t.ts_us >= start_us) and (end_us is None or t.ts_us < end_us):
            yield t


def iter_funding_file(path: str | Path) -> Iterator[FundingApply]:
    p = Path(path)
    for lineno, fields in _lines(p):
        if len(fields) != 2:
            raise ValueError(f"{p.name}:{lineno}: funding row needs 2 fields, got {len(fields)}")
        try:
            yield FundingApply(parse_ts_us(fields[0]), parse_decimal(fields[1]))
        except ValueError as e:
            raise ValueError(f"{p.name}:{lineno}: {e}") from None


@dataclass(frozen=True, slots=True)
class TradeSequenceReport:
    trades: int
    id_gaps: int  # places where the next id is not previous + 1
    missing_ids: int  # ids skipped in total (gaps only, not duplicates)
    non_increasing_ids: int
    non_monotonic_ts: int


def check_sequence(trades: Iterable[Trade]) -> TradeSequenceReport:
    n = gaps = missing = non_inc = non_mono = 0
    prev: Trade | None = None
    for t in trades:
        n += 1
        if prev is not None:
            if t.trade_id <= prev.trade_id:
                non_inc += 1
            elif t.trade_id != prev.trade_id + 1:
                gaps += 1
                missing += t.trade_id - prev.trade_id - 1
            if t.ts_us < prev.ts_us:
                non_mono += 1
        prev = t
    return TradeSequenceReport(n, gaps, missing, non_inc, non_mono)
