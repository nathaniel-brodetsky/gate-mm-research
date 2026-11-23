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
