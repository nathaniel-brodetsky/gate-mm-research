"""Benjamini-Hochberg false discovery rate control."""

from __future__ import annotations

from collections.abc import Sequence


def benjamini_hochberg(p_values: Sequence[float], q: float) -> list[bool]:
    """Reject flags in input order: reject the k smallest p-values, k = max{i : p_(i) <= i q / m}."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if p_values[i] <= rank * q / m:
            k = rank
    reject = [False] * m
    for i in order[:k]:
        reject[i] = True
    return reject


def benjamini_yekutieli(p_values: Sequence[float], q: float) -> list[bool]:
    """Benjamini-Yekutieli: BH at q / (1 + 1/2 + ... + 1/m); controls the FDR under any dependence."""
    m = len(p_values)
    return benjamini_hochberg(p_values, q / sum(1 / i for i in range(1, m + 1))) if m else []
