"""Holdout discipline (D-001, stage 2 plan step 10).

November 2026 and the U-001 holdout pairs are opened once, in this order:
  1. H-003 is tested on the research pairs in November (or November and December);
  2. H-001 and H-002 are estimated on the research pairs in November, after that test has run;
  3. the holdout pairs are built and evaluated, after H-003 is accepted or rejected.
The checks below read the result file of the H-003 test and say whether a step may run.
"""

from __future__ import annotations

import json
from pathlib import Path

H003_KEY = "80.0"  # registered threshold, as written in the result file
H003_TEST_PERIODS = ("test", "extended")


def _h003(path: str | None) -> tuple[dict[str, object] | None, str | None]:
    if not path:
        return None, "the H-003 test result file is required (results/h003/test.json)"
    p = Path(path)
    if not p.is_file():
        return None, f"no H-003 test result file at {path}"
    doc = json.loads(p.read_text())
    if doc.get("hypothesis") != "H-003" or doc.get("period") not in H003_TEST_PERIODS:
        return None, f"{path} is not the result of the H-003 test"
    return doc, None


def h003_tested(path: str | None) -> str | None:
    """None if H-003 has been tested (any outcome); otherwise why not."""
    return _h003(path)[1]


def holdout_unlocked(path: str | None) -> str | None:
    """None if the H-003 test is accepted or rejected, so the holdout pairs may be opened."""
    doc, err = _h003(path)
    if doc is None:
        return err
    summary = doc.get("summary")
    outcome = summary.get(H003_KEY, {}).get("outcome") if isinstance(summary, dict) else None
    if outcome not in ("accept", "reject"):
        return f"H-003 is not decided yet (outcome {outcome!r}); the holdout pairs stay closed"
    return None
