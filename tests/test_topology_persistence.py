from __future__ import annotations

import math

import numpy as np
import pytest

from mmlab.topology import persistence as ph


def bars(b: np.ndarray) -> list[tuple[float, float]]:
    return sorted((float(x), float(y)) for x, y in b)


def test_sublevel_and_swing_bars() -> None:
    y = np.array([0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
    assert bars(ph.sublevel_bars(y)) == [(1.0, 3.0), (2.0, 4.0)]  # minima 1 and 2 merge at 3 and 4
    assert bars(ph.swing_bars(y)) == [(-4.0, -2.0), (-3.0, -1.0), (1.0, 3.0), (2.0, 4.0)]
    assert len(ph.swing_bars(np.arange(6.0))) == 0  # a straight move has no swing


def test_swing_lengths_sum_to_variation_minus_range() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = np.round(np.cumsum(rng.standard_normal(80)) * 2)
        tv = float(np.sum(np.abs(np.diff(y))))
        assert ph.total(ph.swing_bars(y)) == pytest.approx(tv - float(np.max(y) - np.min(y)))


def test_delay_embedding() -> None:
    y = np.arange(5.0)
    assert ph.delay_embedding(y, 3, 1).tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert ph.delay_embedding(y, 3, 2).tolist() == [[0, 2, 4]]
    assert ph.delay_embedding(y, 2, 1).tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert ph.delay_embedding(y, 3, 3).shape == (0, 3)


def test_rips_bars_of_a_square() -> None:
    h0, h1 = ph.rips_bars(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    assert bars(h0) == [(0.0, 1.0)] * 3
    assert h1.shape == (1, 2)
    assert h1[0, 0] == 1.0
    assert h1[0, 1] == pytest.approx(math.sqrt(2))
    e0, e1 = ph.rips_bars(np.array([[0.0, 0.0]]))
    assert len(e0) == len(e1) == 0


def test_a_sampled_circle_has_one_long_loop() -> None:
    t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    _, h1 = ph.rips_bars(np.stack([np.cos(t), np.sin(t)], axis=1))
    assert ph.longest(h1) > 1.0
    assert ph.count_at_least(h1, 0.5) == 1


def test_summaries() -> None:
    b = np.array([[0.0, 2.0], [1.0, 2.0]])
    assert ph.total(b) == 3.0
    assert ph.longest(b) == 2.0
    assert ph.count_at_least(b, 2.0) == 1
    assert ph.count_at_least(b, 1.0) == 2
    assert ph.landscape_l1(b) == pytest.approx((4 + 1) / 4)
    assert ph.entropy(b) == pytest.approx(-(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)))
    empty = np.empty((0, 2))
    assert ph.total(empty) == ph.longest(empty) == ph.landscape_l1(empty) == ph.entropy(empty) == 0.0
    assert ph.entropy(np.array([[0.0, 3.0]])) == 0.0
    assert ph.entropy(np.array([[1.0, 1.0], [0.0, 2.0]])) == 0.0  # zero-length bars do not count
