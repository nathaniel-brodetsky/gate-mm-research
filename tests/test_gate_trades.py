from decimal import Decimal as D
from pathlib import Path

import pytest

from _gate_synth import T0, SynthMarket, write_gz
from mmlab.data_ingestion.gate_orderbook_file import BookEvent, iter_orderbook_file
from mmlab.data_ingestion.gate_trades_file import (
    Trade,
    check_sequence,
    iter_funding_file,
    iter_trades_file,
    parse_trade_line,
)
from mmlab.orderbook import gate_reconcile as rc


def row(s: str) -> list[str]:
    return s.split(",")


def test_spot_deal_taker_side() -> None:
    buy = parse_trade_line(row("1790812935.470294,16340555,0.00005277,82518.7,2"), "spot")
    sell = parse_trade_line(row("1793487655.862658,16383437,0.00004287,116938.4,1"), "spot")
    assert buy == Trade(1_790_812_935_470_294, 16340555, D("0.00005277"), D("82518.7"), "buy")
    assert sell.taker == "sell"


def test_futures_trade_sign_is_taker_side() -> None:
    buy = parse_trade_line(row("1790812800.224253,4137792,0.010739,15"), "futures")
    sell = parse_trade_line(row("1793487488.288684,4208056,0.007766,-25"), "futures")
    assert (buy.taker, buy.qty) == ("buy", D(15))
    assert (sell.taker, sell.qty) == ("sell", D(25))


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("1,2,0.1,5,3", "spot"),
        ("1,2,0.1,5", "spot"),
        ("1,2,0.1,0,2", "spot"),
        ("1,2,0,5,2", "spot"),
        ("1,2,0.1,0", "futures"),
        ("1,-2,0.1,5", "futures"),
        ("1,2,0.1,5,1", "futures"),
    ],
)
def test_bad_trade_rows(line: str, kind: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        parse_trade_line(row(line), kind)  # type: ignore[arg-type]


def test_time_filter_and_funding(tmp_path: Path) -> None:
    f = write_gz(tmp_path / "t.csv.gz", ["10.5,1,1,1", "11.0,2,1,-1", "12.0,3,1,2"])
    assert [t.trade_id for t in iter_trades_file(f, "futures", 11_000_000, 12_000_000)] == [2]
    g = write_gz(tmp_path / "f.csv.gz", ["1790812800,0.0001", "1790841600,-0.000009"])
    assert [(x.ts_us, x.rate) for x in iter_funding_file(g)] == [
        (1_790_812_800_000_000, D("0.0001")),
        (1_790_841_600_000_000, D("-0.000009")),
    ]
    bad = write_gz(tmp_path / "b.csv.gz", ["1790812800,0.0001,7"])
    with pytest.raises(ValueError, match=r"b\.csv\.gz:1:"):
        list(iter_funding_file(bad))


def test_sequence_report() -> None:
    def t(ts: int, tid: int) -> Trade:
        return Trade(ts, tid, D(1), D(1), "buy")

    rep = check_sequence([t(1, 10), t(2, 11), t(3, 14), t(2, 15), t(4, 15)])
    assert (rep.trades, rep.id_gaps, rep.missing_ids, rep.non_monotonic_ts, rep.non_increasing_ids) == (
        5,
        1,
        2,
        1,
        1,
    )


@pytest.mark.parametrize(("kind", "convention", "bucket"), [("futures", "end", 0), ("spot", "start", -1)])
def test_reconcile_recovers_batch_convention(kind: str, convention: str, bucket: int, tmp_path: Path) -> None:
    m = SynthMarket(kind, seed=3, batch_ts=convention)  # type: ignore[arg-type]
    book = write_gz(tmp_path / "book.csv.gz", m.hour(T0, steps=1500))
    trades = write_gz(tmp_path / "trades.csv.gz", m.trade_rows())
    rep = rc.reconcile(iter_orderbook_file(book, kind), iter_trades_file(trades, kind))  # type: ignore[arg-type]
    assert rep.trades > 100
    assert rep.matched == rep.trades
    assert set(rep.offsets) == {bucket}
    assert 0 < rep.matched_qty < rep.take_qty  # the rest of take volume is cancellations
    assert rc.main(["--kind", kind, "--trades", str(trades), str(book)]) == 0


def test_reconcile_fails_with_wrong_taker_side(tmp_path: Path) -> None:
    m = SynthMarket("futures", seed=4)
    book = write_gz(tmp_path / "book.csv.gz", m.hour(T0, steps=1500))
    flipped = []
    for r in m.trade_rows():
        f = r.split(",")
        f[3] = f[3][1:] if f[3].startswith("-") else "-" + f[3]
        flipped.append(",".join(f))
    trades = write_gz(tmp_path / "trades.csv.gz", flipped)
    rep = rc.reconcile(iter_orderbook_file(book, "futures"), iter_trades_file(trades, "futures"))
    assert rep.trades > 100
    assert rep.matched < rep.trades * 0.2


def test_reconcile_ignores_trades_outside_book_range(tmp_path: Path) -> None:
    m = SynthMarket("futures", seed=5)
    book = write_gz(tmp_path / "book.csv.gz", m.hour(T0, steps=300))
    rows = [*m.trade_rows(), f"{T0 + 7200}.000001,99999999,0.0000355,1"]
    trades = write_gz(tmp_path / "trades.csv.gz", rows)
    rep = rc.reconcile(iter_orderbook_file(book, "futures"), iter_trades_file(trades, "futures"))
    assert rep.matched == rep.trades


def test_reconcile_consumes_take_quantity() -> None:
    book = [
        BookEvent(100_000_000, "bid", "set", D(1), D(10), 10, 0),
        BookEvent(100_100_000, "bid", "take", D(1), D(5), 11, 1),
        BookEvent(103_000_000, "bid", "make", D(1), D(1), 12, 1),
    ]
    trades = [Trade(101_050_000, 1, D(1), D(3), "sell"), Trade(101_060_000, 2, D(1), D(3), "sell")]
    book[1] = BookEvent(101_100_000, "bid", "take", D(1), D(5), 11, 1)
    rep = rc.reconcile(book, trades)
    assert (rep.trades, rep.matched, rep.matched_qty) == (2, 1, D(3))
