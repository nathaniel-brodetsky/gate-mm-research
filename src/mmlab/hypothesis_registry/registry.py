"""Validation of the hypothesis registry (`hypotheses/H-NNN-*.md`).

Every hypothesis file must follow `hypotheses/TEMPLATE.md`: an `# H-NNN: title` header matching the file
name, a status from the allowed set, a registration date, and all template sections. The registration
commit is the first commit that adds the file (see `git log --diff-filter=A`).

Usage:
  python -m mmlab.hypothesis_registry.registry hypotheses/
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

STATUSES = ("draft", "registered", "running", "rejected", "accepted")
SECTIONS = (
    "Economic rationale",
    "Mathematical formulation",
    "Data",
    "Features",
    "Target variable",
    "Forecast horizon",
    "Baseline",
    "Metric",
    "Success criterion (fixed before the experiment)",
    "Periods",
    "Variants tested (all, including failed ones)",
    "Results",
)
_FILE = re.compile(r"^(H-\d{3})-[a-z0-9-]+\.md$")
_TITLE = re.compile(r"^# (H-\d{3}): (.+)$", re.M)
_STATUS = re.compile(r"^- \*\*Status:\*\* (\S+)\s*$", re.M)
_REGISTERED = re.compile(r"^- \*\*Registered:\*\* (\d{4}-\d{2}-\d{2})\b", re.M)


@dataclass(frozen=True)
class Hypothesis:
    id: str
    title: str
    status: str
    registered: date
    path: Path


def parse(path: Path) -> Hypothesis:
    m = _FILE.match(path.name)
    if not m:
        raise ValueError(f"{path.name}: file name must be H-NNN-short-name.md")
    text = path.read_text()
    errors = []
    t = _TITLE.search(text)
    if not t or t.group(1) != m.group(1):
        errors.append("header '# H-NNN: title' missing or does not match the file name")
    st = _STATUS.search(text)
    if not st or st.group(1) not in STATUSES:
        errors.append(f"status must be one of {STATUSES}")
    rg = _REGISTERED.search(text)
    if not rg:
        errors.append("registration date missing")
    headings = set(re.findall(r"^## (.+?)\s*$", text, re.M))
    errors += [f"section missing: {s}" for s in SECTIONS if s not in headings]
    if errors or t is None or st is None or rg is None:
        raise ValueError(f"{path.name}: " + "; ".join(errors))
    return Hypothesis(m.group(1), t.group(2).strip(), st.group(1), date.fromisoformat(rg.group(1)), path)


def load_all(directory: Path) -> list[Hypothesis]:
    out = [parse(p) for p in sorted(directory.glob("H-*.md"))]
    ids = [h.id for h in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate hypothesis ids: {dupes}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    directory = Path(args[0] if args else "hypotheses")
    try:
        hs = load_all(directory)
    except ValueError as e:
        print(f"INVALID: {e}")
        return 1
    for h in hs:
        print(f"{h.id}  {h.status:10}  {h.registered}  {h.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
