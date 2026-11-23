"""Price-level (L2) order book with exact Decimal prices and quantities."""

from __future__ import annotations

import bisect
from decimal import Decimal
from typing import Literal

Side = Literal["bid", "ask"]
SIDES: tuple[Side, Side] = ("bid", "ask")
_ZERO = Decimal(0)


class BookError(ValueError):
    """An update would leave the book in an impossible state."""


class L2Book:
    """Aggregated quantity per price level. A level exists only while its quantity is positive."""

    __slots__ = ("_prices", "_qty")

    def __init__(self) -> None:
        self._qty: dict[Side, dict[Decimal, Decimal]] = {"bid": {}, "ask": {}}
        self._prices: dict[Side, list[Decimal]] = {"bid": [], "ask": []}  # ascending

    def copy(self) -> L2Book:
        new = L2Book()
        for s in SIDES:
            new._qty[s] = dict(self._qty[s])
            new._prices[s] = list(self._prices[s])
        return new

    def _put(self, side: Side, price: Decimal, qty: Decimal) -> None:
        levels = self._qty[side]
        if price not in levels:
            bisect.insort(self._prices[side], price)
        levels[price] = qty

    def _remove(self, side: Side, price: Decimal) -> None:
        if self._qty[side].pop(price, None) is not None:
            prices = self._prices[side]
            del prices[bisect.bisect_left(prices, price)]

    def set_level(self, side: Side, price: Decimal, qty: Decimal) -> None:
        """Absolute update: the level becomes `qty`; 0 removes it."""
        if price <= 0 or qty < 0:
            raise BookError(f"bad level {side} {price} {qty}")
        if qty == 0:
            self._remove(side, price)
        else:
            self._put(side, price, qty)

    def apply_delta(self, side: Side, price: Decimal, delta: Decimal) -> None:
        """Incremental update: level += delta. A negative result removes the level and raises BookError."""
        if price <= 0:
            raise BookError(f"bad price {side} {price}")
        new = self._qty[side].get(price, _ZERO) + delta
        if new < 0:
            self._remove(side, price)
            raise BookError(f"negative level {side} {price}: {new}")
        if new == 0:
            self._remove(side, price)
        else:
            self._put(side, price, new)

    def qty(self, side: Side, price: Decimal) -> Decimal:
        return self._qty[side].get(price, _ZERO)

    def depth(self, side: Side) -> int:
        return len(self._prices[side])

    def best_bid(self) -> Decimal | None:
        p = self._prices["bid"]
        return p[-1] if p else None

    def best_ask(self) -> Decimal | None:
        p = self._prices["ask"]
        return p[0] if p else None

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb >= ba

    def levels(self, side: Side, n: int | None = None) -> list[tuple[Decimal, Decimal]]:
        """Levels best-first: bids descending, asks ascending."""
        prices = self._prices[side]
        seq = reversed(prices) if side == "bid" else iter(prices)
        out: list[tuple[Decimal, Decimal]] = []
        for p in seq:
            if n is not None and len(out) >= n:
                break
            out.append((p, self._qty[side][p]))
        return out

    def mismatches(self, other: L2Book) -> int:
        """Number of (side, price) keys whose quantity differs between the two books."""
        n = 0
        for s in SIDES:
            a, b = self._qty[s], other._qty[s]
            n += sum(1 for p in a.keys() | b.keys() if a.get(p) != b.get(p))
        return n
