"""Select the pre-registered research universe U-002 of USDT perpetuals (D-003).

Rule (docs/decisions/D-003-perp-universe.md):

1. Candidates of `mmlab.data_ingestion.perp_candidates`: USDT perpetuals listed before 2025-06-01 and
   available on 2025-07-01 according to Tardis.dev's symbol list, so contracts delisted since are
   included. Excluded bases: stablecoins and pegged commodity tokens (D-001's list).
2. Active on at least 28 days of June 2025 (`mmlab.data_ingestion.perp_activity`).
3. Median daily trades in June 2025 in [100, 10,000), four strata equal in log10:
   [100, 316.2), [316.2, 1,000), [1,000, 3,162.3), [3,162.3, 10,000).
4. Within each stratum contracts are ordered by SHA-256(salt + name); the first 6 form the research set,
   the next 3 the holdout set. The salt is fixed here, before the selection is run.

Usage:
  python -m mmlab.market_data.perp_universe results/perp_universe/candidates.csv \\
      results/perp_universe/activity_2025-06.csv configs/universe_U002.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmlab.market_data.universe import STABLE_OR_PEGGED, order_key

SALT = "mmlab-U002-2026-11-26"


@dataclass(frozen=True)
class PerpRule:
    selection_month: str = "2025-06"
    min_active_days: int = 28
    edges: tuple[float, ...] = (100.0, 10**2.5, 1_000.0, 10**3.5, 10_000.0)
    research_per_stratum: int = 6
    holdout_per_stratum: int = 3
    salt: str = SALT


def base_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def exclusion_reason(cand: dict[str, str], act: dict[str, str] | None, rule: PerpRule) -> str | None:
    if base_of(cand["name"]) in STABLE_OR_PEGGED:
        return "stable or pegged"
    if act is None or act.get("status") != "ok":
        return "no trades file"
    if int(act["active_days"]) < rule.min_active_days:
        return f"active on fewer than {rule.min_active_days} days"
    med = float(act["median_daily_trades"])
    if med < rule.edges[0]:
        return "below the lowest stratum"
    if med >= rule.edges[-1]:
        return "above the highest stratum"
    return None


def stratum_of(med: float, rule: PerpRule) -> int:
    for i in range(len(rule.edges) - 1):
        if rule.edges[i] <= med < rule.edges[i + 1]:
            return i
    raise ValueError(f"median {med} outside the strata")


@dataclass(frozen=True)
class PerpSelection:
    eligible: dict[int, list[str]]
    research: dict[int, list[str]]
    holdout: dict[int, list[str]]
    excluded: dict[str, int]


def select(
    candidates: Sequence[dict[str, str]], activity: Sequence[dict[str, str]], rule: PerpRule
) -> PerpSelection:
    act = {a["name"]: a for a in activity}
    eligible: dict[int, list[str]] = {s: [] for s in range(len(rule.edges) - 1)}
    excluded: Counter[str] = Counter()
    for c in candidates:
        why = exclusion_reason(c, act.get(c["name"]), rule)
        if why:
            excluded[why] += 1
            continue
        eligible[stratum_of(float(act[c["name"]]["median_daily_trades"]), rule)].append(c["name"])
    research, holdout = {}, {}
    n_r, n_h = rule.research_per_stratum, rule.holdout_per_stratum
    for s, names in eligible.items():
        ordered = sorted(names, key=lambda n: order_key(n, rule.salt))
        research[s], holdout[s] = ordered[:n_r], ordered[n_r : n_r + n_h]
    return PerpSelection(eligible, research, holdout, dict(excluded))


def _read(path: str) -> tuple[list[dict[str, str]], str]:
    raw = Path(path).read_bytes()
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh)), hashlib.sha256(raw).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Select the pre-registered research universe U-002.")
    p.add_argument("candidates")
    p.add_argument("activity")
    p.add_argument("out")
    a = p.parse_args(argv)
    cands, cand_sha = _read(a.candidates)
    acts, act_sha = _read(a.activity)
    rule = PerpRule()
    sel = select(cands, acts, rule)
    act = {x["name"]: x for x in acts}
    chosen = [n for s in sel.research for n in sel.research[s] + sel.holdout[s]]
    cand = {c["name"]: c for c in cands}
    doc: dict[str, Any] = {
        "rule": asdict(rule),
        "input": {
            "candidates": {"file": Path(a.candidates).name, "sha256": cand_sha},
            "activity": {"file": Path(a.activity).name, "sha256": act_sha},
        },
        "excluded": sel.excluded,
        "eligible_per_stratum": {str(s): len(v) for s, v in sel.eligible.items()},
        "research": {str(s): v for s, v in sel.research.items()},
        "holdout": {str(s): v for s, v in sel.holdout.items()},
        "june_median_daily_trades": {n: float(act[n]["median_daily_trades"]) for n in chosen},
        "delisted_since": sorted(n for n in chosen if cand[n]["listed_on_gate_now"] == "False"),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    for s in sel.eligible:
        lo, hi = rule.edges[s], rule.edges[s + 1]
        gone = sum(cand[n]["listed_on_gate_now"] == "False" for n in sel.eligible[s])
        print(
            f"stratum {s} [{lo:,.1f}, {hi:,.1f}): eligible {len(sel.eligible[s])} ({gone} delisted since), "
            f"research {len(sel.research[s])}, holdout {len(sel.holdout[s])}"
        )
    print(f"excluded: {dict(sorted(sel.excluded.items()))}")
    print(f"selected and delisted since: {len(doc['delisted_since'])} of {len(chosen)}")
    short = [s for s in sel.research if len(sel.research[s]) + len(sel.holdout[s]) < 9]
    if short:
        print(f"WARNING: strata with fewer than 9 eligible contracts: {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
