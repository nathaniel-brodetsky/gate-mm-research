from __future__ import annotations

from _gate_synth import T0, SynthMarket
from mmlab.data_ingestion.gate_orderbook_file import BookEvent, parse_line
from mmlab.orderbook.gate_replay import ReplayReport, replay
from mmlab.orderbook.l2 import L2Book
from mmlab.orderbook.top_of_book import BATCH_US, top_of_book


def _events(rows: list[str]) -> list[BookEvent]:
    return [parse_line(r.split(","), "spot") for r in rows]


def test_matches_brute_force_and_emits_only_changes() -> None:
    events = _events(SynthMarket("spot", seed=21).hour(T0, steps=800))
    every: list[tuple[int, object]] = []

    def grab(ts: int, b: L2Book, r: ReplayReport) -> None:
        bb, ba = b.best_bid(), b.best_ask()
        state = (bb, b.qty("bid", bb) if bb else None, ba, b.qty("ask", ba) if ba else None, True)
        every.append((ts, state))

    replay(events, on_batch=grab)
    expected: list[tuple[int, object]] = []
    for ts, st in every:
        if not expected or expected[-1][1] != st:
            expected.append((ts, st))
    rows, rep, _ = top_of_book(events)
    assert rep.clean
    assert [(r.label_us, (r.bid, r.bid_qty, r.ask, r.ask_qty, r.valid)) for r in rows] == expected
    assert len(rows) < rep.batches  # deep-book changes do not create rows
    assert all(r.avail_us == r.label_us + BATCH_US for r in rows)
    assert rows[0].label_us > T0 * 1_000_000  # the snapshot itself is not emitted


def test_rows_after_an_anomaly_are_invalid() -> None:
    rows_raw = SynthMarket("spot", seed=22).hour(T0, steps=800)
    first_update = next(i for i, r in enumerate(rows_raw) if ",set," not in r)
    del rows_raw[first_update + 400]  # lost update -> id break
    rows, rep, _ = top_of_book(_events(rows_raw))
    assert rep.id_breaks >= 1
    flags = [r.valid for r in rows]
    assert flags[0] is True
    assert False in flags
    k = flags.index(False)
    assert all(not f for f in flags[k:])  # never valid again within the file
