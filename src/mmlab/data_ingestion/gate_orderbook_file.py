"""Parser for Gate hourly `orderbooks` archive files.

Format as verified on archive samples during the stage-1 audit (2026-11-23), not taken from Gate docs:

  spot:    ts, side, action, price, amount, begin_id, merged        side: 1 = ask, 2 = bid
  futures: ts, action, price, size, begin_id, merged                sign of size: + bid, - ask

  action:  set  = snapshot row, absolute level quantity (only at file start)
           make = quantity added to the level (delta)
           take = quantity removed from the level (delta; trades AND cancellations)
  ts:      seconds since epoch, UTC; snapshot on the hour, updates in 100 ms steps
  ids:     first update id = snapshot id + 1; next begin_id = begin_id + merged
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, Literal

from mmlab.orderbook.l2 import Side

Kind = Literal["spot", "futures"]
Action = Literal["set", "make", "take"]
_ACTIONS: Final[dict[str, Action]] = {"set": "set", "make": "make", "take": "take"}
_SPOT_SIDE: Final[dict[str, Side]] = {"1": "ask", "2": "bid"}


@dataclass(frozen=True, slots=True)
class BookEvent:
    ts_us: int  # microseconds since epoch, UTC
    side: Side
    action: Action
    price: Decimal
    qty: Decimal  # non-negative; set: level quantity, make/take: delta magnitude
    update_id: int  # begin_id
    merged: int


def parse_ts_us(s: str) -> int:
    whole, _, frac = s.strip().partition(".")
    if not whole.isdigit() or (frac and not frac.isdigit()) or len(frac) > 6:
        raise ValueError(f"bad timestamp: {s!r}")
    return int(whole) * 1_000_000 + int(frac.ljust(6, "0"))


def parse_decimal(s: str) -> Decimal:
    try:
        d = Decimal(s.strip())
    except InvalidOperation:
        raise ValueError(f"bad number: {s!r}") from None
    if not d.is_finite():
        raise ValueError(f"bad number: {s!r}")
    return d


def _action(s: str) -> Action:
    try:
        return _ACTIONS[s.strip()]
    except KeyError:
        raise ValueError(f"bad action: {s!r}") from None


def _ints(begin_id: str, merged: str) -> tuple[int, int]:
    b, m = int(begin_id), int(merged)
    if b < 0 or m < 0:
        raise ValueError(f"bad id/merged: {begin_id!r}/{merged!r}")
    return b, m


def parse_line(fields: list[str], kind: Kind) -> BookEvent:
    if kind == "spot":
        if len(fields) != 7:
            raise ValueError(f"spot row needs 7 fields, got {len(fields)}: {fields}")
        ts, side_s, act, px, amt, bid, mg = fields
        try:
            side = _SPOT_SIDE[side_s.strip()]
        except KeyError:
            raise ValueError(f"bad spot side: {side_s!r}") from None
        qty = parse_decimal(amt)
        if qty < 0:
            raise ValueError(f"negative spot amount: {amt!r}")
    else:
        if len(fields) != 6:
            raise ValueError(f"futures row needs 6 fields, got {len(fields)}: {fields}")
        ts, act, px, sz, bid, mg = fields
        signed = parse_decimal(sz)
        if signed == 0:
            raise ValueError("futures size 0 carries no side")
        side = "bid" if signed > 0 else "ask"
        qty = abs(signed)
    price = parse_decimal(px)
    if price <= 0:
        raise ValueError(f"non-positive price: {px!r}")
    update_id, merged = _ints(bid, mg)
    return BookEvent(parse_ts_us(ts), side, _action(act), price, qty, update_id, merged)


def iter_orderbook_file(path: str | Path, kind: Kind) -> Iterator[BookEvent]:
    """Stream events from a .csv.gz (or plain .csv) hourly orderbooks file."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", newline="") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield parse_line(line.rstrip("\r\n").split(","), kind)
            except ValueError as e:
                raise ValueError(f"{p.name}:{lineno}: {e}") from None
