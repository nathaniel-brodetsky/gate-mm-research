"""Exploration: round trips of a new maker on USDT perpetuals (stage 4, step 3c).

The stage 3 round trips (`mmlab.execution_simulator.round_trip`) on the research contracts of U-002
over the exploration period (D-003): every 60 s an order of one contract on each side, one tick
inside the spread, kept 60 s; each filled piece is closed by a passive order one tick inside the other
side, kept 300 s, and what is left is crossed. One contract is the smallest order Gate accepts; its
value in USDT needs the multiplier, which delisted contracts no longer have, so results are per unit
of entry value (bp), not USDT.

Fees under two scenarios: VIP 0 per Gate's announcement (maker 0.02%, taker 0.05%) and the contract
fields seen on 2026-11-26 (maker -0.01%, taker 0.075%); which one an account pays is not established.
Funding is left out: pieces are held for minutes and funding is charged every 8 hours.

Per contract: round trips, passive share, median holding time, gross and net bp per scenario, for all
round trips and without each delisted contract's last 14 days (D-003, Amendment 1). Medians over
contracts with at least 30 round trips on 5 days, overall and by stratum. Exploration only.

Usage:
    python -m mmlab.execution_simulator.perp_round_trip data/processed/U002 \\
        --out results/perp_explore/round_trips.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mmlab.evaluation.h003 import MIN_DAYS, MIN_TRADES
from mmlab.execution_simulator.fill_explore import decision_times
from mmlab.execution_simulator.queue import Book, Trades, place, simulate, tick_size
from mmlab.execution_simulator.round_trip import (
    EVERY_S,
    EXIT_LIFE_S,
    HORIZON_S,
    IMPROVE,
    LATENCY_MS,
    round_trips,
)
from mmlab.microstructure.perp_explore import PERIOD, runup_start

SCENARIOS = {"vip0": (0.0002, 0.0005), "contract_fields": (-0.0001, 0.00075)}  # (maker, taker)
SIZE_CONTRACTS = 1.0


def contract_trips(
    ds: Path,
    name: str,
    start: str,
    end: str,
    every_s: int = EVERY_S,
    horizon_s: int = HORIZON_S,
    exit_life_s: int = EXIT_LIFE_S,
    latency_ms: int = LATENCY_MS,
    improve: int = IMPROVE,
    queue: str = "pessimistic",
    size: float = SIZE_CONTRACTS,
) -> pl.DataFrame | None:
    """Round trips of one contract (rows of round_trip.round_trips); None without data."""
    tp, bp = ds / "trades" / f"{name}.parquet", ds / "tob" / f"{name}.parquet"
    if not (tp.exists() and bp.exists()):
        return None
    book = Book.from_tob(pl.read_parquet(bp))
    tr = Trades.from_frame(pl.read_parquet(tp))
    tick = tick_size(book) if improve else 0.0
    lat, hor = latency_ms * 1000, horizon_s * 1_000_000
    t = decision_times(start, end, every_s, horizon_s)
    sizes = np.full(t.size, size)
    entries = sorted(
        place(book, t, 1, 0.0, lat, hor, improve, tick, sizes)
        + place(book, t, -1, 0.0, lat, hor, improve, tick, sizes),
        key=lambda o: (o.act_us, o.decided_us, o.side),
    )
    optimistic = queue == "optimistic"
    fills = simulate(book, tr, entries, optimistic)
    return round_trips(book, tr, entries, fills, lat, exit_life_s * 1_000_000, improve, tick, optimistic)


def summarize(trips: pl.DataFrame) -> dict[str, Any]:
    """Round trips with a closing price: counts, passive share, holding time, gross and net bp."""
    ok = trips.drop_nulls(["gross"])
    value = float(ok["value"].sum()) if ok.height else 0.0
    days = (
        int(ok.select(pl.from_epoch("ts_us", time_unit="us").dt.date().n_unique()).item()) if ok.height else 0
    )
    gross = float(ok["gross"].sum()) if ok.height else 0.0
    mv = float(ok["maker_value"].sum()) if ok.height else 0.0
    tv = float(ok["taker_value"].sum()) if ok.height else 0.0
    per_bp = 1e4 / value if value > 0 else None
    return {
        "round_trips": ok.height,
        "unpriced": trips.height - ok.height,
        "days": days,
        "counted": ok.height >= MIN_TRADES and days >= MIN_DAYS,
        "passive_share": float((ok["passive_share"] * ok["value"]).sum()) / value if value > 0 else None,
        "median_hold_s": float(np.median(ok["hold_s"].to_numpy())) if ok.height else None,
        "gross_bp": None if per_bp is None else gross * per_bp,
        "net_bp": {
            k: None if per_bp is None else (gross - mk * mv - tk * tv) * per_bp
            for k, (mk, tk) in SCENARIOS.items()
        },
    }


def explore_contract(
    ds: Path, name: str, start: str, end: str, available_to: str, queue: str, improve: int
) -> tuple[str, dict[str, Any] | None]:
    trips = contract_trips(ds, name, start, end, queue=queue, improve=improve)
    if trips is None:
        return name, None
    out = {"available_to": available_to, "all": summarize(trips)}
    cut = runup_start(available_to)
    out["without_runup"] = out["all"] if cut is None else summarize(trips.filter(pl.col("ts_us") < cut))
    return name, out


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def screen(contracts: Mapping[str, Any], strata: Mapping[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in ("all", "without_runup"):
        groups: dict[str, list[dict[str, Any]]] = {"all": []}
        for name, r in contracts.items():
            s = r[variant]
            if s["counted"]:
                groups["all"].append(s)
                groups.setdefault(f"stratum {strata.get(name, '?')}", []).append(s)
        out[variant] = {
            g: {
                "contracts": len(ss),
                "gross_bp": _median([s["gross_bp"] for s in ss if s["gross_bp"] is not None]),
                "passive_share": _median([s["passive_share"] for s in ss if s["passive_share"] is not None]),
                "median_hold_s": _median([s["median_hold_s"] for s in ss if s["median_hold_s"] is not None]),
                **{
                    f"net_{k}_bp": _median([s["net_bp"][k] for s in ss if s["net_bp"][k] is not None])
                    for k in SCENARIOS
                },
                **{f"positive_{k}": sum((s["net_bp"][k] or 0.0) > 0 for s in ss) for k in SCENARIOS},
            }
            for g, ss in sorted(groups.items())
        }
    return out


def _bp(x: float | None) -> str:
    return "     n/a" if x is None else f"{x:+8.2f}"


def report(res: Mapping[str, Any]) -> str:
    lines = [
        f"round trips of a new maker on USDT perpetuals, {res['period'][0]} .. {res['period'][1]}: "
        f"one contract every {EVERY_S} s per side, {res['improve_ticks']} tick inside, {res['queue']} queue; "
        "bp of entry value; fees vip0 maker 0.02%/taker 0.05%, contract fields -0.01%/0.075%; medians over "
        f"contracts with >= {MIN_TRADES} round trips on >= {MIN_DAYS} days; exploration, no inference"
    ]
    for variant in ("all", "without_runup"):
        lines.append(f"{variant}:")
        lines.append("  group        n   gross  net vip0  net fields  passive  hold s  >0 vip0  >0 fields")
        for g, s in res["screen"][variant].items():
            ps = "n/a" if s["passive_share"] is None else f"{s['passive_share']:.2f}"
            hold = "n/a" if s["median_hold_s"] is None else f"{s['median_hold_s']:.0f}"
            n = s["contracts"]
            lines.append(
                f"  {g:10} {n:3d} {_bp(s['gross_bp'])}{_bp(s['net_vip0_bp'])}   "
                f"{_bp(s['net_contract_fields_bp'])}  {ps:>7} {hold:>7}  {s['positive_vip0']:3d}/{n}"
                f"    {s['positive_contract_fields']:3d}/{n}"
            )
    if res["missing"]:
        lines.append(f"not built: {', '.join(res['missing'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Round trips of a new maker on USDT perpetuals.")
    p.add_argument("dataset")
    p.add_argument("--universe", default="configs/universe_U002.json")
    p.add_argument("--candidates", default="results/perp_universe/candidates.csv")
    p.add_argument("--start", default=PERIOD[0])
    p.add_argument("--end", default=PERIOD[1])
    p.add_argument("--queue", choices=["pessimistic", "optimistic"], default="pessimistic")
    p.add_argument("--improve-ticks", type=int, default=IMPROVE)
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
    args = [
        (Path(a.dataset), n, a.start, a.end, available_to.get(n, ""), a.queue, a.improve_ticks) for n in names
    ]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            done = list(ex.map(explore_contract, *zip(*args, strict=True)))
    else:
        done = [explore_contract(*x) for x in args]
    contracts = {n: r for n, r in done if r is not None}
    res = {
        "period": [a.start, a.end],
        "queue": a.queue,
        "improve_ticks": a.improve_ticks,
        "size_contracts": SIZE_CONTRACTS,
        "scenarios": SCENARIOS,
        "contracts": contracts,
        "missing": [n for n, r in done if r is None],
        "screen": screen(contracts, strata),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
