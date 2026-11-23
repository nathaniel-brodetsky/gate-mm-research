from decimal import Decimal as D

import pytest

from mmlab.orderbook.l2 import BookError, L2Book


def test_best_and_order() -> None:
    b = L2Book()
    for px in ("1.0", "3.0", "2.0"):
        b.set_level("bid", D(px), D(1))
    for px in ("5.0", "4.0", "6.0"):
        b.set_level("ask", D(px), D(2))
    assert (b.best_bid(), b.best_ask()) == (D("3.0"), D("4.0"))
    assert [p for p, _ in b.levels("bid")] == [D("3.0"), D("2.0"), D("1.0")]
    assert [p for p, _ in b.levels("ask", 2)] == [D("4.0"), D("5.0")]
    assert not b.is_crossed()


def test_delta_adds_and_removes() -> None:
    b = L2Book()
    b.set_level("bid", D("1"), D("5"))
    b.apply_delta("bid", D("1"), D("3"))
    assert b.qty("bid", D("1")) == D("8")
    b.apply_delta("bid", D("1"), D("-8"))
    assert b.depth("bid") == 0
    assert b.best_bid() is None


def test_negative_level_raises_and_removes() -> None:
    b = L2Book()
    b.set_level("ask", D("2"), D("1"))
    with pytest.raises(BookError):
        b.apply_delta("ask", D("2"), D("-1.5"))
    assert b.depth("ask") == 0


def test_set_zero_removes_and_bad_inputs() -> None:
    b = L2Book()
    b.set_level("ask", D("2"), D("1"))
    b.set_level("ask", D("2"), D("0"))
    assert b.best_ask() is None
    with pytest.raises(BookError):
        b.set_level("ask", D("0"), D("1"))
    with pytest.raises(BookError):
        b.set_level("ask", D("1"), D("-1"))


def test_crossed_and_mismatches_and_copy() -> None:
    b = L2Book()
    b.set_level("bid", D("2"), D("1"))
    b.set_level("ask", D("2"), D("1"))
    assert b.is_crossed()
    c = b.copy()
    assert b.mismatches(c) == 0
    c.apply_delta("bid", D("2"), D("1"))
    c.set_level("ask", D("3"), D("1"))
    assert b.mismatches(c) == 2
    assert b.qty("bid", D("2")) == D("1")  # copy is independent
