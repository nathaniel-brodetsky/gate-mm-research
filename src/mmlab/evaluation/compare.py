"""Compare two result JSON files up to floating-point rounding.

Results are not reproducible bit for bit: polars adds in parallel, so the order of additions, and with it
the last bits of a sum, can change between runs. Measured 2026-11-24 on the H-001/H-002 development
results: two runs differed in 17 floats, by at most 4.1e-14 relative; every integer and string was equal.

Rule: structure, keys, integers, booleans and strings must be equal; two floats are equal if they differ
by at most ABS_TOL absolutely or REL_TOL relatively.

Usage:
  python -m mmlab.evaluation.compare A.json B.json [--ignore role]
Exit status 0 if equal under the rule, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ABS_TOL = 1e-12
REL_TOL = 1e-9
MISSING = "<missing>"


@dataclass(frozen=True)
class Difference:
    path: str
    a: Any
    b: Any

    @property
    def floats(self) -> bool:
        return isinstance(self.a, float) and isinstance(self.b, float)

    @property
    def rel(self) -> float | None:
        if not self.floats:
            return None
        return float(abs(self.a - self.b) / max(abs(self.a), abs(self.b), 1e-300))

    def within_rounding(self, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
        if not self.floats:
            return False
        gap = abs(self.a - self.b)
        return bool(gap <= abs_tol or gap <= rel_tol * max(abs(self.a), abs(self.b)))


def differences(a: Any, b: Any, path: str = "") -> Iterator[Difference]:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                yield Difference(f"{path}/{k}", a.get(k, MISSING), b.get(k, MISSING))
            else:
                yield from differences(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield Difference(f"{path}/len", len(a), len(b))
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            yield from differences(x, y, f"{path}[{i}]")
    elif type(a) is not type(b) or a != b:
        yield Difference(path, a, b)


def compare(
    a: Any, b: Any, ignore: Sequence[str] = (), abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL
) -> tuple[list[Difference], list[Difference]]:
    """(differences beyond rounding, differences within rounding); top-level keys in `ignore` are skipped."""
    if isinstance(a, dict) and isinstance(b, dict):
        a = {k: v for k, v in a.items() if k not in ignore}
        b = {k: v for k, v in b.items() if k not in ignore}
    beyond: list[Difference] = []
    within: list[Difference] = []
    for d in differences(a, b):
        (within if d.within_rounding(abs_tol, rel_tol) else beyond).append(d)
    return beyond, within


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two result JSON files up to floating-point rounding.")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--ignore", action="append", default=[], help="top-level key to skip (repeatable)")
    args = p.parse_args(argv)
    beyond, within = compare(
        json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()), args.ignore
    )
    worst = max((d.rel or 0.0 for d in within), default=0.0)
    print(
        f"{len(within)} floats differ within rounding (max rel {worst:.1e}); {len(beyond)} differences beyond"
    )
    for d in beyond[:20]:
        print(f"  {d.path}: {d.a!r} -> {d.b!r}")
    return 0 if not beyond else 1


if __name__ == "__main__":
    sys.exit(main())
