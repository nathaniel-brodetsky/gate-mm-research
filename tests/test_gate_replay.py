from decimal import Decimal as D
from pathlib import Path

import pytest

from _gate_synth import T0, SynthMarket, write_gz
from mmlab.data_ingestion.gate_orderbook_file import BookEvent, iter_orderbook_file, parse_line
from mmlab.orderbook import gate_replay as gr
from mmlab.orderbook.l2 import L2Book


def _events(rows: list[str], kind: str = "spot") -> list[BookEvent]:
    return [parse_line(r.split(","), kind) for r in rows]  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["spot", "futures"])
def test_clean_two_hour_chain(kind: str, tmp_path: Path) -> None:
    m = SynthMarket(kind)  # type: ignore[arg-type]
    f1 = write_gz(tmp_path / "h12.csv.gz", m.hour(T0))
    truth = (dict(m.bids), dict(m.asks))
    f2 = write_gz(tmp_path / "h13.csv.gz", m.hour(T0 + 3600))
    snaps: list[L2Book] = []
    book1, r1 = gr.replay(iter_orderbook_file(f1, kind))  # type: ignore[arg-type]
    _, r2 = gr.replay(iter_orderbook_file(f2, kind), on_snapshot=lambda b: snaps.append(b.copy()))  # type: ignore[arg-type]
    assert r1.clean, r1
    assert r2.clean, r2
    assert r1.rows_after_hour_end > 0
    assert r1.snapshot_rows == 78
    assert {p: q for p, q in book1.levels("bid")} == truth[0]
    assert {p: q for p, q in book1.levels("ask")} == truth[1]
    ch = gr.check_chain(book1, r1, snaps[0], r2)
    assert ch.ok, ch
    assert gr.main(["--kind", kind, str(f1), str(f2)]) == 0


def test_delta_not_absolute() -> None:
    rows = ["100,2,set,1,5,10,0", "100.1,2,make,1,3,11,1", "100.2,2,take,1,2,12,1"]
    book, rep = gr.replay(_events(rows))
    assert book.qty("bid", D(1)) == D(6)
    assert rep.clean


def test_dropped_row_breaks_id_chain() -> None:
    m = SynthMarket("spot")
    rows = m.hour(T0, steps=200)
    del rows[100]
    _, rep = gr.replay(_events(rows))
    assert rep.id_breaks >= 1
    assert not rep.clean


def test_merged_step_matters() -> None:
    rows = ["100,2,set,1,5,10,0", "100.1,2,make,1,3,11,3", "100.2,2,make,1,3,14,1", "100.3,2,make,1,3,16,1"]
    _, rep = gr.replay(_events(rows))
    assert rep.id_breaks == 1
    assert rep.first_break == (16, 15)
    assert rep.last_covered_id == 16


def test_crossed_negative_nonmonotonic_lateset() -> None:
    rows = [
        "100,2,set,1,5,10,0",
        "100,1,set,2,5,10,0",
        "100.1,2,make,2,1,11,1",  # bid at ask price -> crossed batch
        "100.2,2,take,2,1,12,1",
        "100.3,1,take,2,9,13,1",  # more than available -> negative
        "100.2,1,make,3,1,14,1",  # time goes backwards
        "100.4,1,set,4,1,15,1",  # snapshot row after updates
    ]
    _, rep = gr.replay(_events(rows))
    assert rep.crossed_batches == 1
    assert rep.negative_levels == 1
    assert rep.non_monotonic_ts == 1
    assert rep.late_set_rows == 1
    assert rep.batches == 5


def test_batches_group_rows_with_same_timestamp() -> None:
    rows = ["100,2,set,1,5,10,0", "100.1,2,make,1,1,11,1", "100.1,2,make,1,1,12,1", "100.2,2,take,1,1,13,1"]
    seen: list[int] = []
    _, rep = gr.replay(_events(rows), on_batch=lambda ts, b, r: seen.append(ts))
    assert rep.batches == 2
    assert seen == [100_100_000, 100_200_000]


def test_structure_errors() -> None:
    with pytest.raises(gr.FormatError):
        gr.replay(_events(["100.1,2,make,1,1,11,1"]))
    with pytest.raises(gr.FormatError):
        gr.replay(_events(["100,2,set,1,5,10,0", "100,2,set,2,5,11,0"]))


def test_chain_detects_mismatch(tmp_path: Path) -> None:
    m = SynthMarket("futures")
    f1 = write_gz(tmp_path / "a.csv.gz", m.hour(T0, steps=300))
    rows2 = m.hour(T0 + 3600, steps=300)
    rows2[0] = (
        rows2[0].replace(",set,", ",set,").rsplit(",", 3)[0] + ",123.0," + ",".join(rows2[0].split(",")[-2:])
    )
    f2 = write_gz(tmp_path / "b.csv.gz", rows2)
    assert gr.main(["--kind", "futures", str(f1), str(f2)]) == 1


def test_last_covered_id_uses_merged_and_hour_end_is_inclusive() -> None:
    rows = ["100,2,set,1,5,10,0", "3699.9,2,make,1,1,11,1", "3700.0,2,make,1,1,12,3"]
    _, rep = gr.replay(_events(rows))
    assert rep.last_covered_id == 14
    assert rep.rows_after_hour_end == 1


def test_chain_id_mismatch_with_equal_books() -> None:
    prev_book, prev = gr.replay(_events(["100,2,set,1,5,10,0", "100.1,2,make,1,1,11,1"]))
    snaps: list[L2Book] = []
    _, nxt = gr.replay(_events(["3700,2,set,1,6,12,0"]), on_snapshot=lambda b: snaps.append(b.copy()))
    ch = gr.check_chain(prev_book, prev, snaps[0], nxt)
    assert ch.level_mismatches == 0
    assert not ch.id_match
    assert not ch.ok
