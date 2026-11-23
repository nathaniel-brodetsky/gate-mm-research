from decimal import Decimal as D
from pathlib import Path

import pytest

from mmlab.data_ingestion.gate_orderbook_file import (
    BookEvent,
    iter_orderbook_file,
    parse_decimal,
    parse_line,
    parse_ts_us,
)


def row(s: str) -> list[str]:
    return s.split(",")


def test_timestamps_exact() -> None:
    assert parse_ts_us("1794744000") == 1_794_744_000_000_000
    assert parse_ts_us("1794747599.0") == 1_794_747_599_000_000
    assert parse_ts_us("1794744000.4") == 1_794_744_000_400_000
    assert parse_ts_us("1790812800.224253") == 1_790_812_800_224_253
    for bad in ("", "1.2.3", "-1", "1e9", "1.1234567", "abc"):
        with pytest.raises(ValueError):  # noqa: PT011
            parse_ts_us(bad)


def test_decimal_formats_seen_in_archives() -> None:
    assert parse_decimal("1e-08") == D("0.00000001")
    assert parse_decimal("49643410249.0000000000000000") == D("49643410249")
    for bad in ("x", "nan", "inf", ""):
        with pytest.raises(ValueError):  # noqa: PT011
            parse_decimal(bad)


def test_spot_row_side_mapping() -> None:
    ask = parse_line(row("1794744000.4,1,make,0.00003547,50761421.4,2943585079,1"), "spot")
    bid = parse_line(row("1794744000,2,set,1e-08,49643410249.0000000000000000,2943578795,0"), "spot")
    assert ask == BookEvent(
        1_794_744_000_400_000, "ask", "make", D("0.00003547"), D("50761421.4"), 2943585079, 1
    )
    assert (bid.side, bid.action, bid.price, bid.qty, bid.merged) == (
        "bid",
        "set",
        D("1e-8"),
        D("49643410249"),
        0,
    )


def test_futures_row_sign_is_side() -> None:
    bid = parse_line(row("1794744000,set,1e-06,376800.0,987573008,0"), "futures")
    ask = parse_line(row("1794747599.0,take,0.008115,-2429.0,987582833,1"), "futures")
    assert (bid.side, bid.qty) == ("bid", D("376800"))
    assert (ask.side, ask.qty, ask.action) == ("ask", D("2429"), "take")


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("1794744000,3,make,1,1,1,1", "spot"),
        ("1794744000,1,cancel,1,1,1,1", "spot"),
        ("1794744000,1,make,1,-1,1,1", "spot"),
        ("1794744000,1,make,0,1,1,1", "spot"),
        ("1794744000,1,make,1,1,1", "spot"),
        ("1794744000,make,1,0,1,1", "futures"),
        ("1794744000,make,1,1,1,1,1", "futures"),
        ("1794744000,make,1,1,-5,1", "futures"),
    ],
)
def test_bad_rows_rejected(line: str, kind: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        parse_line(row(line), kind)  # type: ignore[arg-type]


def test_file_errors_carry_line_number(tmp_path: Path) -> None:
    f = tmp_path / "x.csv"
    f.write_text("1794744000,2,set,1,1,5,0\n\n1794744000.1,9,make,1,1,6,1\n")
    with pytest.raises(ValueError, match=r"x\.csv:3:"):
        list(iter_orderbook_file(f, "spot"))
