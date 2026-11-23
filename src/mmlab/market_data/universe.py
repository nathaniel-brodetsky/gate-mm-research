"""Research universe U-001: deterministic, pre-registered selection of illiquid Gate spot pairs.

Rule (docs/decisions/D-001-research-universe.md), applied to daily candles from
`mmlab.data_ingestion.gate_rest_history` (`global_daily_bars.csv`):

1. Quote currency USDT; base is not a stablecoin, a pegged commodity token or a leveraged token.
2. Listed before the selection window: at least one daily candle before 2026-07-01.
3. A candle for every day of the selection window 2026-07-01 .. 2026-07-31, with volume on at least
   28 of the 31 days.
4. Median daily quote volume over the window in [1,000, 100,000) USDT, split into four strata of equal
   width in log10.
5. Within each stratum pairs are ordered by SHA-256(salt + pair); the first 6 form the research set, the
   next 3 the holdout set. The salt is fixed here, before the selection is run.

Usage:
  python -m mmlab.market_data.universe BARS_CSV configs/universe_U001.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

STABLE_OR_PEGGED = frozenset(
    {
        "USDC",
        "USDT",
        "FDUSD",
        "TUSD",
        "DAI",
        "USDE",
        "USD1",
        "PYUSD",
        "EURC",
        "EURT",
        "USDD",
        "USDP",
        "GUSD",
        "RLUSD",
        "BUSD",
        "FRAX",
        "USDS",
        "USDG",
        "USDQ",
        "USDR",
        "LUSD",
        "SUSD",
        "CUSD",
        "PAXG",
        "XAUT",
        "KAU",
        "KAG",
    }
)
LEVERAGED = re.compile(r"\d+[LS]$")


@dataclass(frozen=True)
class UniverseRule:
    id: str = "U-001"
    quote: str = "USDT"
    listed_before: str = "2026-07-01"
    window_start: str = "2026-07-01"
    window_end: str = "2026-07-31"
    min_active_days: int = 28
    strata: tuple[tuple[int, int], ...] = (
        (1_000, 3_162),
        (3_162, 10_000),
        (10_000, 31_623),
        (31_623, 100_000),
    )
    research_per_stratum: int = 6
    holdout_per_stratum: int = 3
    salt: str = "mmlab-U001-2026-11-23"
    excluded_bases: tuple[str, ...] = field(default_factory=lambda: tuple(sorted(STABLE_OR_PEGGED)))


def load_daily(path: str | Path) -> dict[str, dict[str, Decimal]]:
    """pair -> day -> quote volume, from a gate_rest_history daily bars CSV."""
    out: defaultdict[str, dict[str, Decimal]] = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["pair"]][r["day"]] = Decimal(r["quote_volume"])
    return dict(out)


def _days(start: str, end: str) -> list[str]:
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= last:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def exclusion_reason(pair: str, days: dict[str, Decimal], rule: UniverseRule) -> str | None:
    base, _, quote = pair.rpartition("_")
    if quote != rule.quote:
        return "quote"
    if base in rule.excluded_bases:
        return "stable_or_pegged"
    if LEVERAGED.search(base):
        return "leveraged"
    if not any(d < rule.listed_before for d in days):
        return "listed_late"
    window = _days(rule.window_start, rule.window_end)
    if any(d not in days for d in window):
        return "gap_in_window"
    if sum(days[d] > 0 for d in window) < rule.min_active_days:
        return "inactive"
    med = statistics.median(days[d] for d in window)
    lo, hi = rule.strata[0][0], rule.strata[-1][1]
    if not lo <= med < hi:
        return "volume_out_of_range"
    return None


def window_median(days: dict[str, Decimal], rule: UniverseRule) -> Decimal:
    return statistics.median(days[d] for d in _days(rule.window_start, rule.window_end))


def stratum_of(med: Decimal, rule: UniverseRule) -> int:
    for i, (lo, hi) in enumerate(rule.strata):
        if lo <= med < hi:
            return i
    raise ValueError(f"median {med} outside strata")


def order_key(pair: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{pair}".encode()).hexdigest()


@dataclass(frozen=True)
class Selection:
    rule: UniverseRule
    eligible: dict[int, list[str]]
    research: dict[int, list[str]]
    holdout: dict[int, list[str]]
    medians: dict[str, str]
    excluded: dict[str, int]


def select(daily: dict[str, dict[str, Decimal]], rule: UniverseRule) -> Selection:
    excluded: defaultdict[str, int] = defaultdict(int)
    eligible: defaultdict[int, list[str]] = defaultdict(list)
    medians: dict[str, str] = {}
    for pair, days in daily.items():
        why = exclusion_reason(pair, days, rule)
        if why:
            excluded[why] += 1
            continue
        med = window_median(days, rule)
        medians[pair] = str(med)
        eligible[stratum_of(med, rule)].append(pair)
    research, holdout = {}, {}
    for s in range(len(rule.strata)):
        ordered = sorted(eligible.get(s, []), key=lambda p: order_key(p, rule.salt))
        research[s] = ordered[: rule.research_per_stratum]
        holdout[s] = ordered[rule.research_per_stratum : rule.research_per_stratum + rule.holdout_per_stratum]
        eligible[s] = ordered
    return Selection(rule, dict(eligible), research, holdout, medians, dict(excluded))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Select the pre-registered research universe U-001.")
    p.add_argument("bars", help="global_daily_bars.csv from gate_rest_history")
    p.add_argument("out", help="output JSON")
    a = p.parse_args(argv)
    raw = Path(a.bars).read_bytes()
    sel = select(load_daily(a.bars), UniverseRule())
    doc = {
        "rule": asdict(sel.rule),
        "input": {"file": Path(a.bars).name, "sha256": hashlib.sha256(raw).hexdigest()},
        "excluded": sel.excluded,
        "eligible_per_stratum": {str(s): len(v) for s, v in sel.eligible.items()},
        "research": {str(s): v for s, v in sel.research.items()},
        "holdout": {str(s): v for s, v in sel.holdout.items()},
        "window_median": {
            pp: sel.medians[pp] for s in sel.research for pp in sel.research[s] + sel.holdout[s]
        },
    }
    Path(a.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    for s, (lo, hi) in enumerate(sel.rule.strata):
        print(
            f"stratum {s} [{lo:,}, {hi:,}): eligible {len(sel.eligible.get(s, []))}, "
            f"research {len(sel.research[s])}, holdout {len(sel.holdout[s])}"
        )
    print(f"excluded: {dict(sorted(sel.excluded.items()))}")
    short = [s for s in sel.research if len(sel.research[s]) + len(sel.holdout[s]) < 9]
    if short:
        print(f"WARNING: strata with fewer than 9 eligible pairs: {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
