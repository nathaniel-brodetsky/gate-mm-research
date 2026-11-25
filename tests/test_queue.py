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


def run(
    b: qu.Book, t: qu.Trades, orders: list[qu.Order], optimistic: bool = False
) -> list[tuple[int, float, float]]:
    return [(f.order, f.ts / S, round(f.qty, 12)) for f in qu.simulate(b, t, orders, optimistic)]


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


def test_sweep_fills_up_to_the_trade_size() -> None:
    b = book(BASE)
    assert run(b, trades([(2.0, 102.0, 0.01, 1)]), sell_at(b)) == [(0, 2.0, 0.01)]
    assert run(b, trades([(2.0, 102.0, 0.01, 1), (2.1, 103.0, 5.0, 1)]), sell_at(b)) == [
        (0, 2.0, 0.01),
        (0, 2.1, 0.99),
    ]


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
    assert run(b, t, buys) == [(0, 2.0, 0.5), (1, 2.0, 0.5), (0, 3.0, 0.1), (1, 3.0, 0.1)]


def _reference(
    b: qu.Book, t: qu.Trades, orders: list[qu.Order], optimistic: bool = False
) -> list[tuple[int, float, float]]:
    out = []
    for n, o in enumerate(orders):
        ahead, rem, last, since = o.ahead, o.size, float("nan"), 0.0
        ev = [(int(b.avail[j]), 0, j) for j in range(b.avail.size) if o.act_us < b.avail[j] < o.expire_us]
        ev += [(int(t.ts[k]), 1, k) for k in range(t.ts.size) if o.act_us <= t.ts[k] < o.expire_us]
        for _, kind, i in sorted(ev):
            if rem <= 0:
                break
            if kind == 0:
                if b.ok[i]:
                    best = b.ask[i] if o.side == 1 else b.bid[i]
                    if qu._same(float(best), o.price):
                        shown = float(b.ask_qty[i] if o.side == 1 else b.bid_qty[i])
                        if optimistic and np.isfinite(last):
                            ahead = max(0.0, ahead - max(0.0, last - since - shown))
                        ahead = min(ahead, shown)
                        last, since = shown, 0.0
                    elif (best - o.price) * o.side > 0:
                        ahead = 0.0
                        last, since = float("nan"), 0.0
                continue
            if t.taker[i] != o.side:
                continue
            p, x = float(t.price[i]), float(t.qty[i])
            if qu._same(p, o.price):
                take = min(ahead, x)
                ahead -= take
                since += x
                got = min(x - take, rem)
            elif (p - o.price) * o.side > 0:
                got = min(x, rem)
            else:
                continue
            if got > 0:
                out.append((n, int(t.ts[i]) / S, round(got, 12)))
                rem -= got
    return sorted(out, key=lambda r: (r[1], r[0]))


@pytest.mark.parametrize("improve", [0, 1])
@pytest.mark.parametrize("optimistic", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_engine_matches_the_per_order_reference(seed: int, optimistic: bool, improve: int) -> None:
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
        qu.place(b, decide, 1, 50.0, 300_000, 20 * S, improve, 0.5)
        + qu.place(b, decide, -1, 50.0, 300_000, 20 * S, improve, 0.5),
        key=lambda o: (o.act_us, o.decided_us, o.side),
    )
    got = sorted(run(b, t, orders, optimistic), key=lambda r: (r[1], r[0]))
    assert len(orders) > 50
    assert len(got) > 10
    assert got == _reference(b, t, orders, optimistic)


def test_optimistic_queue_credits_unexplained_falls_of_the_display() -> None:
    # 4 ahead at 1.4 s; 6 join behind (10 shown); then the display falls to 6 without trades
    b = book([*BASE, (2.0, 99.0, 101.0, 5.0, 10.0, True), (3.0, 99.0, 101.0, 5.0, 6.0, True)])
    t = trades([(3.5, 101.0, 0.5, 1)])
    assert run(b, t, sell_at(b)) == []  # pessimistic: the 4 cancels came from behind
    assert run(b, t, sell_at(b), optimistic=True) == [(0, 3.5, 0.5)]  # optimistic: from ahead


def test_optimistic_queue_does_not_count_trades_as_cancels() -> None:
    b = book([*BASE, (2.0, 99.0, 101.0, 5.0, 10.0, True), (3.0, 99.0, 101.0, 5.0, 7.0, True)])
    t = trades([(2.5, 101.0, 3.0, 1), (3.5, 101.0, 1.5, 1)])  # 10 - 3 traded = 7 shown: no cancels
    assert run(b, t, sell_at(b), optimistic=True) == [(0, 3.5, 0.5)]  # 4 - 3 = 1 ahead, then filled
    hidden = book([*BASE, (2.0, 99.0, 100.5, 5.0, 1.0, True), (3.0, 99.0, 101.0, 5.0, 1.0, True)])
    # 101 was shown with 4, hidden behind 100.5, then shown with 1: 3 cancelled, all from ahead
    assert run(hidden, trades([(3.5, 101.0, 1.5, 1)]), sell_at(hidden), optimistic=True) == [(0, 3.5, 0.5)]


def test_tick_size() -> None:
    b = book(
        [
            (0.0, 0.1234, 0.1237, 1, 1, True),
            (1.0, 0.1233, 0.1236, 1, 1, True),
            (2.0, 0.1233, 0.1240, 1, 1, True),
        ]
    )
    assert qu.tick_size(b) == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="no tick size"):
        qu.tick_size(book([(0.0, np.nan, 1.0, 1, 1, False)]))


def test_improving_orders_stand_inside_the_spread() -> None:
    b = book(BASE)  # 99 / 101, tick 0.5
    sell = qu.place(b, np.array([S]), 1, 100.5, 300_000, 100 * S, improve_ticks=1, tick=0.5)
    buy = qu.place(b, np.array([S]), -1, 99.5, 300_000, 100 * S, improve_ticks=1, tick=0.5)
    assert (sell[0].price, sell[0].ahead, buy[0].price, buy[0].ahead) == (100.5, 0.0, 99.5, 0.0)
    # alone in front: a buy at the old ask fills it up to the trade's size, at its own price
    t = trades([(2.0, 101.0, 0.4, 1), (3.0, 101.0, 2.0, 1)])
    assert run(b, t, sell) == [(0, 2.0, 0.4), (0, 3.0, 0.6)]
    narrow = book([(0.0, 100.0, 100.5, 1, 1, True)])
    assert qu.place(narrow, np.array([S]), 1, 100.0, 300_000, S, improve_ticks=1, tick=0.5) == []  # no room
    two = qu.place(narrow, np.array([S]), 1, 100.0, 300_000, S, improve_ticks=2, tick=0.25)
    assert two == []  # 100.5 - 2 x 0.25 = 100.0 is the bid itself
    joined = book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 99.0, 100.5, 5, 2, True)])
    assert qu.place(joined, np.array([S]), 1, 100.5, 300_000, S, improve_ticks=1, tick=0.5)[0].ahead == 2.0
    beaten = book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 99.0, 100.0, 5, 2, True)])
    assert qu.place(beaten, np.array([S]), 1, 100.5, 300_000, S, improve_ticks=1, tick=0.5)[0].ahead == 0.0
    crossed = book([(0.0, 99.0, 101.0, 5, 3, True), (1.4, 100.5, 102.0, 5, 2, True)])
    assert qu.place(crossed, np.array([S]), 1, 100.5, 300_000, S, improve_ticks=1, tick=0.5) == []
    with pytest.raises(ValueError, match="positive tick"):
        qu.place(b, np.array([S]), 1, 100.5, 300_000, S, improve_ticks=1)


def test_float_noise_in_an_improved_price_still_matches_the_archive() -> None:
    tick = 0.1235 - 0.1234  # 9.999999999998899e-05
    b = book([(0.0, 0.1230, 0.1235, 5, 3, True), (1.4, 0.1230, 0.1234, 5, 2, True)])
    o = qu.place(b, np.array([S]), 1, 0.1234, 300_000, 100 * S, improve_ticks=1, tick=tick)
    assert o[0].ahead == 2.0  # 0.1235 - tick is the archive's 0.1234: joined behind the 2 shown


def test_float_noise_guards() -> None:
    t = 0.6 - 0.5  # a tick measured on another gap: 0.2 - t lands 3e-17 above 0.1
    assert 0.2 - t > 0.1
    # no room at the decision (the improved price is the bid), even if the bid moves away later
    away = book([(0.0, 0.1, 0.2, 1, 1, True), (1.4, 0.05, 0.2, 1, 1, True)])
    assert qu.place(away, np.array([S]), 1, 0.1, 300_000, S, improve_ticks=1, tick=t) == []
    # room at the decision, but at activation the bid has come up to the improved price: post-only
    up = book([(0.0, 0.05, 0.2, 1, 1, True), (1.4, 0.1, 0.2, 1, 1, True)])
    assert qu.place(up, np.array([S]), 1, 0.1, 300_000, S, improve_ticks=1, tick=t) == []
    noisy = book([(0.0, 0.1, 0.2, 1, 1, True), (1.0, 0.2 - t, 0.3, 1, 1, True)])
    assert qu.tick_size(noisy) == pytest.approx(0.1)  # 0.1 and 0.1 + 3e-17 are one price
