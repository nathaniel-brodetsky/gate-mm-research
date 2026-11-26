"""Exploration: realized spread of USDT perpetuals before and after fees (stage 4, step 3b).

For the research contracts of U-002 over the exploration period (2025-07-01 .. 2025-10-31, D-003),
every trade with a valid book (`mmlab.evaluation.h003.states`): realized spread at 10, 60 and 300 s
against the mid, at 60 s against the microprice and against the far side (exit), volume-weighted per
contract, and the quoted half-spread q. Theta is RS(60 s) minus the maker fee under two scenarios:
VIP 0's 0.02% and the contract field's -0.01% (a rebate); which one an account pays is not established
(stage 4 plan). Per contract also by bucket of q, and the same without each delisted contract's last 14
days before its delisting (D-003, Amendment 1). Medians over contracts with at least 30 trades on 5
days, overall and by stratum. Sizes are in contracts, so volume weights are contracts x price: within a
contract they are proportional to quote volume, and medians over contracts need no multiplier.

Exploration only, no inference: it picks what a registered batch would test on November-December 2025.

Usage:
    python -m mmlab.microstructure.perp_explore data/processed/U002 --out results/perp_explore/markouts.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from mmlab.evaluation.h003 import HORIZONS_S, MIN_DAYS, MIN_TRADES, states
from mmlab.microstructure.markouts import in_period

PERIOD = ("2025-07-01", "2025-11-01")
FEES = {"maker_0.02%": 0.0002, "maker_-0.01%": -0.0001}
Q_EDGES_BP = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, math.inf)
RUNUP_DAYS = 14
COLUMNS = (
    "q_median_bp",
    "rs_10s_bp",
    "rs_60s_bp",
    "rs_300s_bp",
    "rs_exit_60s_bp",
    "theta_maker_0.02%_bp",
    "theta_maker_-0.01%_bp",
)


def _vw(df: pl.DataFrame, col: str) -> float | None:
    d = df.filter(pl.col(col).is_not_null())
    den = float(d["v"].sum()) if d.height else 0.0
    return float((d["v"] * d[col]).sum()) / den if den else None


def q_label(lo: float, hi: float) -> str:
    return f"[{lo:g},{hi:g})"


def block(st: pl.DataFrame) -> dict[str, Any]:
    """Measures of one set of trades (rows of h003.states)."""
    rs60 = st.filter(pl.col("rs_60s").is_not_null())
    days = rs60["day"].n_unique() if rs60.height else 0
    rs = {f"{h}s": _vw(st, f"rs_{h}s") for h in HORIZONS_S}
    vol = float(st["v"].sum()) if st.height else 0.0
    q_bp = st["q"] * 1e4
    buckets = {}
    for lo, hi in itertools.pairwise(Q_EDGES_BP):
        b = st.filter((q_bp >= lo) & (q_bp < hi))
        buckets[q_label(lo, hi)] = {
            "trades": b.height,
            "volume_share": float(b["v"].sum()) / vol if vol else 0.0,
            "rs_60s": _vw(b, "rs_60s"),
        }
    r60 = rs["60s"]
    return {
        "trades": rs60.height,
        "days": days,
        "counted": rs60.height >= MIN_TRADES and days >= MIN_DAYS,
        "q_median_bp": float(st.select((pl.col("q") * 1e4).median()).item()) if st.height else None,
        "rs": rs,
        "rs_micro_60s": _vw(st, "rs_micro_60s"),
        "rs_exit_60s": _vw(st, "rs_exit_60s"),
        "theta": {k: None if r60 is None else r60 - fee for k, fee in FEES.items()},
        "by_q": buckets,
    }


def runup_start(available_to: str) -> int | None:
    """Microseconds since the epoch of the start of the last RUNUP_DAYS before a delisting date."""
    if not available_to:
        return None
    d = date.fromisoformat(available_to) - timedelta(days=RUNUP_DAYS)
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) * 1_000_000


def explore_contract(
    ds: Path, name: str, start: str, end: str, available_to: str
) -> tuple[str, dict[str, Any] | None]:
    tp, bp = ds / "trades" / f"{name}.parquet", ds / "tob" / f"{name}.parquet"
    if not (tp.exists() and bp.exists()):
        return name, None
    st = in_period(states(pl.read_parquet(tp), pl.read_parquet(bp)), start, end)
    out = {"available_to": available_to, "all": block(st)}
    cut = runup_start(available_to)
    out["without_runup"] = out["all"] if cut is None else block(st.filter(pl.col("ts_us") < cut))
    return name, out


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def screen(contracts: Mapping[str, Any], strata: Mapping[str, int]) -> dict[str, Any]:
    """Medians over counted contracts, overall and by stratum, for both variants."""
    out: dict[str, Any] = {}
    for variant in ("all", "without_runup"):
        groups: dict[str, list[dict[str, Any]]] = {"all": []}
        for name, r in contracts.items():
            b = r[variant]
            if b["counted"]:
                groups["all"].append(b)
                groups.setdefault(f"stratum {strata.get(name, '?')}", []).append(b)
        out[variant] = {
            g: {
                "contracts": len(bs),
                "q_median_bp": _median([b["q_median_bp"] for b in bs if b["q_median_bp"] is not None]),
                **{
                    f"rs_{h}_bp": _median([b["rs"][h] * 1e4 for b in bs if b["rs"][h] is not None])
                    for h in ("10s", "60s", "300s")
                },
                "rs_exit_60s_bp": _median(
                    [b["rs_exit_60s"] * 1e4 for b in bs if b["rs_exit_60s"] is not None]
                ),
                **{
                    f"theta_{k}_bp": _median([b["theta"][k] * 1e4 for b in bs if b["theta"][k] is not None])
                    for k in FEES
                },
                "positive_after_0.02%": sum(
                    b["theta"]["maker_0.02%"] is not None and b["theta"]["maker_0.02%"] > 0 for b in bs
                ),
            }
            for g, bs in sorted(groups.items())
        }
    return out


def explore(
    ds: Path, names: Sequence[str], start: str, end: str, available_to: Mapping[str, str], workers: int
) -> dict[str, Any]:
    args = [(ds, n, start, end, available_to.get(n, "")) for n in names]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(explore_contract, *zip(*args, strict=True)))
    else:
        done = [explore_contract(*a) for a in args]
    return {r[0]: r[1] for r in done}


def _bp(x: float | None) -> str:
    return "     n/a" if x is None else f"{x:+8.2f}"


def summarize(res: Mapping[str, Any]) -> str:
    lines = [
        f"USDT perpetuals, realized spread per contract (volume-weighted), {res['period'][0]} .. "
        f"{res['period'][1]}; medians over contracts with >= {MIN_TRADES} trades on >= {MIN_DAYS} days; "
        f"exploration, no inference"
    ]
    for variant, title in (
        ("all", "all trades"),
        ("without_runup", f"without the last {RUNUP_DAYS} days before a delisting"),
    ):
        lines.append(f"{title}:")
        lines.append(
            "  group        n   q med   RS10s   RS60s  RS300s  exit60  th 0.02% th -0.01%  >0 at 0.02%"
        )
        for g, s in res["screen"][variant].items():
            row = "".join(_bp(s[c]) for c in COLUMNS)
            lines.append(
                f"  {g:10} {s['contracts']:3d} {row}  {s['positive_after_0.02%']:3d}/{s['contracts']}"
            )
    lines.append("by quoted half-spread (all trades): median over contracts of RS(60 s) in bp, volume share")
    for lo, hi in itertools.pairwise(Q_EDGES_BP):
        lab = q_label(lo, hi)
        by_q = [r["all"]["by_q"][lab] for r in res["contracts"].values() if r["all"]["counted"]]
        rs = [c["rs_60s"] * 1e4 for c in by_q if c["rs_60s"] is not None and c["trades"] >= MIN_TRADES]
        share = _median([c["volume_share"] for c in by_q])
        lines.append(
            f"  q {lab:10} contracts {len(rs):2d}  RS60 {_bp(_median(rs))}  volume share "
            f"{'n/a' if share is None else f'{share:.3f}'}"
        )
    if res["missing"]:
        lines.append(f"not built: {', '.join(res['missing'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Realized spread of USDT perpetuals before and after fees.")
    p.add_argument("dataset")
    p.add_argument("--universe", default="configs/universe_U002.json")
    p.add_argument("--candidates", default="results/perp_universe/candidates.csv")
    p.add_argument("--start", default=PERIOD[0])
    p.add_argument("--end", default=PERIOD[1])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.end > PERIOD[1]:
        print(f"refused: exploration stays before {PERIOD[1]}, the held-out test period (D-003)")
        return 2
    uni = json.loads(Path(a.universe).read_text())
    strata = {n: int(s) for s, names in uni["research"].items() for n in names}
    names = [n for s in sorted(uni["research"], key=int) for n in uni["research"][s]]
    with open(a.candidates, newline="") as fh:
        available_to = {r["name"]: r["available_to"] for r in csv.DictReader(fh)}
    got = explore(Path(a.dataset), names, a.start, a.end, available_to, a.workers)
    contracts = {n: r for n, r in got.items() if r is not None}
    res = {
        "period": [a.start, a.end],
        "fees": FEES,
        "runup_days": RUNUP_DAYS,
        "contracts": contracts,
        "missing": [n for n, r in got.items() if r is None],
        "screen": screen(contracts, strata),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(summarize(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
