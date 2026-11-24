"""Queue fills: hand cases for every rule, and the one-pass engine against a per-order reference."""

from __future__ import annotations

import numpy as np
import pytest

from mmlab.execution_simulator import queue as qu

S = 1_000_000
Row = tuple[float, float, float, float, float, bool]


def book(rows: list[Row]) -> qu.Book:
    """rows: (state time in s, bid, ask, bid_qty, ask_qty, ok)."""
    a = np.array([[r[0] * S, *r[1:5]] for r in rows])
    ok = np.array([r[5] for r in rows], dtype=bool)
    return qu.Book(a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4], ok)


def trades(rows: list[tuple[float, float, float, int]]) -> qu.Trades:
    """rows: (time in s, price, qty, taker side)."""
    a = np.array(rows, dtype=float).reshape(-1, 4)
    return qu.Trades((a[:, 0] * S).astype(np.int64), a[:, 1], a[:, 2], a[:, 3].astype(np.int64))


BASE: list[Row] = [(0.0, 99.0, 101.0, 5.0, 3.0, True), (1.4, 99.0, 101.0, 5.0, 4.0, True)]


def sell_at(b: qu.Book, horizon_s: float = 100.0, size: float = 101.0) -> list[qu.Order]:
    """A sell at the best ask decided at 1 s; it arrives at 1.3 s and joins at the end of its batch."""
    return qu.place(b, np.array([1 * S]), 1, size, 300_000, int(horizon_s * S))


def run(b: qu.Book, t: qu.Trades, orders: list[qu.Order]) -> list[tuple[int, float, float]]:
    return [(f.order, f.ts / S, round(f.qty, 12)) for f in qu.simulate(b, t, orders)]


def test_place_joins_the_back_at_the_end_of_the_arrival_batch() -> None:
    o = sell_at(book(BASE))
    assert len(o) == 1
    assert (o[0].price, o[0].size, o[0].ahead, o[0].act_us) == (101.0, 1.0, 4.0, int(1.4 * S))
    assert o[0].expire_us == int(1.4 * S) + 100 * S
    quiet = sell_at(book(BASE[:1]))  # no change since the decision: the same book is in force
    assert (quiet[0].ahead, quiet[0].act_us) == (3.0, int(1.4 * S))


def test_place_skips_invalid_books_and_post_only_rejects() -> None:
    assert sell_at(book([(0.0, 99.0, 101.0, 5, 3, False), (1.4, 99.0, 101.0, 5, 4, True)])) == []
    assert sell_at(book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 99.0, 101.0, 5, 4, False)])) == []
    assert sell_at(book([(1.2, 99.0, 101.0, 5, 3, True)])) == []  # nothing known at the decision
    assert (
        sell_at(book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 101.0, 102.0, 5, 4, True)])) == []
    )  # bid at 101
    lifted = book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 98.0, 99.0, 5, 4, True)])
    assert qu.place(lifted, np.array([S]), -1, 99.0, 300_000, S) == []  # the ask came down to the buy


def test_place_when_the_level_moved() -> None:
    gone = sell_at(book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 99.0, 102.0, 5, 4, True)]))
    assert gone[0].ahead == 0.0  # alone at 101
    better = sell_at(book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 99.0, 100.5, 5, 4, True)]))
    assert better[0].ahead == 3.0  # behind a better price; 101 was last seen with 3


def test_boundaries_of_decision_and_arrival() -> None:
    b = book(
        [
            (0.0, 99.0, 101.0, 5.0, 3.0, True),
            (1.0, 99.0, 102.0, 5.0, 2.0, True),
            (1.4, 99.0, 102.0, 5.0, 7.0, True),
            (1.5, 99.0, 102.0, 5.0, 9.0, True),
        ]
    )
    assert sell_at(b)[0].price == 102.0  # a state available exactly at the decision is known
    before = qu.place(b, np.array([int(1.09 * S)]), 1, 102.0, 300_000, S)[0]
    on_edge = qu.place(b, np.array([int(1.1 * S)]), 1, 102.0, 300_000, S)[0]
    assert (before.act_us, before.ahead) == (int(1.4 * S), 7.0)  # arrives at 1.39 s, batch ends at 1.4 s
    assert (on_edge.act_us, on_edge.ahead) == (int(1.5 * S), 9.0)  # arrives at 1.4 s: the next batch


def test_queue_then_fill_and_activation_timing() -> None:
    b = book(BASE)
    t = trades([(1.35, 101.0, 10.0, 1), (1.4, 101.0, 3.0, 1), (2.0, 99.0, 9.0, -1), (3.0, 101.0, 1.5, 1)])
    # 1.35 s is before the order joins; 1.4 s counts (after the state of 1.4 s); a sell does not touch asks
    assert run(b, t, sell_at(b)) == [(0, 3.0, 0.5)]


def test_display_caps_the_queue_and_partial_fills() -> None:
    b = book([*BASE, (2.5, 99.0, 101.0, 5.0, 1.0, True)])
    t = trades([(2.5, 101.0, 1.25, 1), (4.0, 101.0, 1.0, 1)])
    assert run(b, t, sell_at(b)) == [(0, 2.5, 0.25), (0, 4.0, 0.75)]  # the state at 2.5 s comes first


def test_invalid_states_do_not_cap() -> None:
    b = book([*BASE, (2.5, 99.0, 101.0, 5.0, 1.0, False)])
    assert run(b, trades([(3.0, 101.0, 1.25, 1)]), sell_at(b)) == []


def test_sweep_fills_everything_left() -> None:
    b = book(BASE)
    assert run(b, trades([(2.0, 102.0, 0.01, 1)]), sell_at(b)) == [(0, 2.0, 1.0)]


def test_level_gone_empties_the_queue() -> None:
    b = book([*BASE, (2.0, 99.0, 102.0, 5.0, 4.0, True), (3.0, 99.0, 101.0, 5.0, 2.0, True)])
    assert run(b, trades([(3.5, 101.0, 0.5, 1)]), sell_at(b)) == [(0, 3.5, 0.5)]  # ahead 0, then capped at 0


def test_better_price_in_front_neither_fills_nor_caps() -> None:
    b = book([*BASE, (2.0, 99.0, 100.5, 5.0, 0.1, True)])
    t = trades([(2.5, 100.5, 5.0, 1), (3.0, 101.0, 4.5, 1)])
    assert run(b, t, sell_at(b)) == [(0, 3.0, 0.5)]


def test_expiry() -> None:
    b = book(BASE)
    o = sell_at(b, horizon_s=2.0)
    assert run(b, trades([(3.4, 102.0, 1.0, 1)]), o) == []  # expired at 3.4 s
    assert run(b, trades([(3.39, 102.0, 1.0, 1)]), o) == [(0, 3.39, 1.0)]


def test_orders_join_between_events_and_after_a_state_at_the_same_time() -> None:
    b = book([*BASE, (1.6, 99.0, 101.0, 5.0, 2.0, True)])
    o = qu.place(b, np.array([int(1.25 * S)]), 1, 101.0, 300_000, 100 * S)  # joins at 1.6 s, after that state
    assert (o[0].act_us, o[0].ahead) == (int(1.6 * S), 2.0)
    t = trades([(1.59, 101.0, 5.0, 1), (1.6, 101.0, 2.5, 1)])
    assert run(b, t, o) == [(0, 1.6, 0.5)]


def test_buy_side_and_independent_orders() -> None:
    b = book(BASE)
    buys = qu.place(b, np.array([S, S]), -1, 99.0, 300_000, 100 * S)
    assert [o.ahead for o in buys] == [5.0, 5.0]
    t = trades([(2.0, 99.0, 5.5, -1), (3.0, 98.0, 0.1, -1), (3.5, 99.0, 9.0, 1)])
    assert run(b, t, buys) == [(0, 2.0, 0.5), (1, 2.0, 0.5), (0, 3.0, 0.5), (1, 3.0, 0.5)]


def _reference(b: qu.Book, t: qu.Trades, orders: list[qu.Order]) -> list[tuple[int, float, float]]:
    out = []
    for n, o in enumerate(orders):
        ahead, rem = o.ahead, o.size
        ev = [(int(b.avail[j]), 0, j) for j in range(b.avail.size) if o.act_us < b.avail[j] < o.expire_us]
        ev += [(int(t.ts[k]), 1, k) for k in range(t.ts.size) if o.act_us <= t.ts[k] < o.expire_us]
        for _, kind, i in sorted(ev):
            if rem <= 0:
                break
            if kind == 0:
                if b.ok[i]:
                    best = b.ask[i] if o.side == 1 else b.bid[i]
                    if qu._same(float(best), o.price):
                        ahead = min(ahead, float(b.ask_qty[i] if o.side == 1 else b.bid_qty[i]))
                    elif (best - o.price) * o.side > 0:
                        ahead = 0.0
                continue
            if t.taker[i] != o.side:
                continue
            p, x = float(t.price[i]), float(t.qty[i])
            if qu._same(p, o.price):
                take = min(ahead, x)
                ahead -= take
                got = min(x - take, rem)
            elif (p - o.price) * o.side > 0:
                got = rem
            else:
                continue
            if got > 0:
                out.append((n, int(t.ts[i]) / S, round(got, 12)))
                rem -= got
    return sorted(out, key=lambda r: (r[1], r[0]))


@pytest.mark.parametrize("seed", range(12))
def test_engine_matches_the_per_order_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    times = np.round(np.cumsum(rng.uniform(0.05, 2.0, n)), 1)  # states on 100 ms batch ends
    times = np.unique(times)
    n = times.size
    mid = 100 + np.cumsum(rng.choice([-1, 0, 0, 1], n)) * 0.5
    half = rng.choice([0.5, 1.0, 1.5], n)
    qb, qa, good = rng.integers(1, 6, n), rng.integers(1, 6, n), rng.random(n) > 0.05
    rows: list[Row] = [
        (float(tt), float(m - h), float(m + h), float(x), float(y), bool(g))
        for tt, m, h, x, y, g in zip(times, mid, half, qb, qa, good, strict=True)
    ]
    b = book(rows)
    m = 900
    tt = np.sort(rng.uniform(0, times[-1], m))
    tt[::50] = b.avail[rng.integers(0, n, tt[::50].size)] / S  # some trades exactly at state times
    tt = np.sort(tt)
    side = rng.choice([-1, 1], m)
    idx = np.clip(np.searchsorted(b.avail, tt * S, side="right") - 1, 0, None)
    best = np.where(side == 1, b.ask[idx], b.bid[idx])
    price = best + side * rng.choice([-0.5, 0.0, 0.0, 0.0, 0.5], m)
    qty = rng.uniform(0.1, 4.0, m)
    t = trades(
        [(float(a), float(p), float(q), int(s)) for a, p, q, s in zip(tt, price, qty, side, strict=True)]
    )
    decide = (np.sort(rng.uniform(0, times[-1] - 5, 60)) * S).astype(np.int64)
    decide[::7] = decide[::7] // 100_000 * 100_000 - 300_000  # some orders join exactly at a batch edge
    decide = np.sort(decide)
    orders = sorted(
        qu.place(b, decide, 1, 50.0, 300_000, 20 * S) + qu.place(b, decide, -1, 50.0, 300_000, 20 * S),
        key=lambda o: (o.act_us, o.decided_us, o.side),
    )
    got = sorted(run(b, t, orders), key=lambda r: (r[1], r[0]))
    assert len(orders) > 50
    assert len(got) > 10
    assert got == _reference(b, t, orders)
