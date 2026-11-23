from __future__ import annotations

import numpy as np
import pytest

from mmlab.evaluation.bootstrap import ratio_resamples, stationary_indices
from mmlab.evaluation.multiple_testing import benjamini_hochberg


def test_indices_shape_range_and_determinism() -> None:
    a = stationary_indices(20, 500, 3.0, seed=1)
    assert a.shape == (500, 20)
    assert int(np.min(a)) >= 0
    assert int(np.max(a)) < 20
    assert (a == stationary_indices(20, 500, 3.0, seed=1)).all()
    assert not (a == stationary_indices(20, 500, 3.0, seed=2)).all()


def test_mean_block_length() -> None:
    idx = stationary_indices(400, 200, 3.0, seed=7)
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 400).mean()
    assert continues == pytest.approx(1 - 1 / 3.0, abs=0.02)
    iid = stationary_indices(400, 200, 1.0, seed=7)
    assert (iid[:, 1:] == (iid[:, :-1] + 1) % 400).mean() < 0.01


def test_ratio_resamples() -> None:
    num, den = np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 1.0])
    idx = np.array([[0, 2, 2], [1, 1, 1]])
    out = ratio_resamples(num, den, idx)
    assert out[0] == pytest.approx(7.0 / 3.0)  # (1 + 3 + 3) / (1 + 1 + 1)
    assert np.isnan(out[1])
    with pytest.raises(ValueError):  # noqa: PT011
        stationary_indices(0, 10, 3.0, seed=1)


def test_benjamini_hochberg() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.2], 0.10) == [True, True, True, False]
    assert benjamini_hochberg([0.2, 0.3], 0.10) == [False, False]
    assert benjamini_hochberg([0.06, 0.001], 0.10) == [True, True]  # step-up: 0.06 <= 2 * 0.1 / 2
    assert benjamini_hochberg([], 0.10) == []
    assert benjamini_hochberg([0.04, 0.045], 0.05) == [True, True]  # step-up: rank 1 fails, rank 2 passes
