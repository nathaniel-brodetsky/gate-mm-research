"""Which calendar days drive the daily markout of a pair (stage 3, step 2e).

The calendar-day generator rebuilds synthetic periods from the real days of each pair. With it, the day
tests reject a true theta of 0 far more often than with the generators that resample trades. This
diagnostic asks why, pair by pair, from the saved day pools:

- how lopsided the days are: skewness and excess kurtosis of the volume-weighted markout sum of the
  active days, and the share of the three heaviest days in the sum of absolute deviations (against the
  same share for normal days, for scale);
- which days are heaviest: date, how much theta moves if the day is dropped, the day's share of volume
  and trades; with a dataset, also the day's price range and return and its hours with data problems;
- whether the size of the day tests (from `lab compare`) follows the lopsidedness or the block length
  across pairs (Spearman, descriptive only), and which dates are among the worst of several pairs.

Usage:
    python -m mmlab.evaluation.tails results/lab/daypools.json --compare results/lab/compare_days.json \
        --dataset data/processed/U001 --out results/lab/tails_days.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from mmlab.evaluation.lab import _usable, load_pool
from mmlab.stochastic_models.markout_panel import DayPool

TOP = 3
METHOD = "student_pw"  # the best day test of step 2
REGISTERED = "percentile"  # the registered day bootstrap
MIN_SHARED = 3  # a date among the worst days of at least this many pairs is listed


def top_share(x: NDArray[np.float64], k: int = TOP) -> float:
    """Share of the k largest |x| in the sum of |x|."""
    a = np.sort(np.abs(x))[::-1]
    s = float(a.sum())
    return float(a[:k].sum() / s) if s > 0 else 0.0


def normal_top_share(n: int, k: int = TOP, reps: int = 2000, seed: int = 20261124) -> float:
    """Median top-k share of n independent normal days: what a share looks like without heavy days."""
    if n <= k:
        return 1.0
    z = np.abs(np.random.default_rng(seed).standard_normal((reps, n)))
    a = np.sort(z, axis=1)[:, ::-1]
    return float(np.median(a[:, :k].sum(axis=1) / a.sum(axis=1)))


def heavy_days(pool: DayPool, start: str, k: int = TOP) -> dict[str, list[dict[str, Any]]]:
    """The k days with the largest loss and the k with the largest gain against the pair's theta.

    shift_bp: how much theta changes, in bp, if the day is dropped (-num_c / (total volume - day volume)).
    """
    total = float(pool.den.sum())
    d0 = date.fromisoformat(start)
    order = np.argsort(pool.num_c, kind="stable")  # days without trades have num_c = 0 and drop out below

    def entry(i: int) -> dict[str, Any]:
        rest = total - float(pool.den[i])
        return {
            "date": (d0 + timedelta(days=int(i))).isoformat(),
            "shift_bp": -float(pool.num_c[i]) / rest * 1e4 if rest > 0 else None,
            "volume_share": float(pool.den[i]) / total,
            "trades": int(pool.counts[i]),
        }

    worst = [entry(int(i)) for i in order[:k] if pool.num_c[i] < 0]
    best = [entry(int(i)) for i in order[::-1][:k] if pool.num_c[i] > 0]
    return {"worst": worst, "best": best}


def shape(pool: DayPool) -> dict[str, float]:
    """Skewness, excess kurtosis and top-3 share of the active days' centred sums; normal reference."""
    x = pool.num_c[pool.den > 0]
    flat = x.size < 3 or float(np.std(x)) == 0.0
    return {
        "active_days": int(x.size),
        "skew": 0.0 if flat else float(stats.skew(x)),
        "excess_kurtosis": 0.0 if flat else float(stats.kurtosis(x)),
        "top3_share": top_share(x),
        "top3_share_normal": normal_top_share(int(x.size)),
    }


def day_quality(ds: Path, pair: str, days: Sequence[str]) -> dict[str, Any]:
    """Price range and return of each given day (from trades) and its hours with data problems.

    A problem hour is missing, has issues, or does not chain to the previous hour.
    """
    tr = pl.read_parquet(ds / "trades" / f"{pair}.parquet", columns=["ts_us", "price"])
    daily = (
        tr.sort("ts_us", maintain_order=True)
        .with_columns(pl.from_epoch(pl.col("ts_us"), time_unit="us").dt.date().cast(pl.String).alias("day"))
        .group_by("day", maintain_order=True)
        .agg(
            (pl.col("price").max() / pl.col("price").min() - 1).alias("range"),
            (pl.col("price").last() / pl.col("price").first() - 1).alias("ret"),
        )
    )
    by_day = {r["day"]: r for r in daily.iter_rows(named=True)}
    bad: Counter[str] = Counter()
    hours = ds / "hours" / f"{pair}.csv"
    if hours.exists():
        with hours.open() as fh:
            for h in csv.DictReader(fh):
                if h.get("status") != "clean" or h.get("chained") == "false":
                    bad[h["hour"][:10]] += 1
    ranges = daily["range"].to_numpy()
    out: dict[str, Any] = {"median_range": float(np.median(ranges)) if ranges.size else None, "days": {}}
    for d in days:
        r = by_day.get(d)
        out["days"][d] = {
            "range": None if r is None else float(r["range"]),
            "ret": None if r is None else float(r["ret"]),
            "problem_hours": bad[d],
        }
    return out


def _size(cmp: Mapping[str, Any], subset: str, n_days: int, pair: str, method: str) -> float | None:
    r = cmp.get(f"compare/{subset}/{n_days}d/size/{pair}")
    if not r or not r["reps"]:
        return None
    return float(r["reject_at_alpha"][method]["rate"])


def _spearman(a: Sequence[float], b: Sequence[float]) -> dict[str, float] | None:
    if len(a) < 5 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    res = stats.spearmanr(a, b)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "pairs": len(a)}


def tails(cal: Mapping[str, Any], cmp: Mapping[str, Any], ds: Path | None = None) -> dict[str, Any]:
    if cal.get("kind") != "days":
        raise ValueError("tails needs day pools (lab pools --kind days)")
    start = cal["period"][0]
    out: dict[str, Any] = {"period": cal["period"], "method": METHOD, "subsets": {}}
    for subset, sub in cal["subsets"].items():
        pairs: dict[str, Any] = {}
        for pair, e in sub["pairs"].items():
            if not _usable(e):
                continue
            pool = load_pool(e["pool"])
            if not isinstance(pool, DayPool):
                raise ValueError(f"{e['pool']} is not a day pool")
            entry: dict[str, Any] = {"theta": e["theta"], "block": e["block"], **shape(pool)}
            entry.update(heavy_days(pool, start))
            entry["size"] = {
                f"{n}d/{m}": _size(cmp, subset, n, pair, m) for n in (30, 61) for m in (METHOD, REGISTERED)
            }
            if ds is not None:
                q = day_quality(ds, pair, [d["date"] for d in entry["worst"] + entry["best"]])
                entry["median_range"] = q["median_range"]
                for d in entry["worst"] + entry["best"]:
                    d.update(q["days"][d["date"]])
            pairs[pair] = entry
        size = {
            p: e["size"]["30d/" + METHOD] for p, e in pairs.items() if e["size"]["30d/" + METHOD] is not None
        }
        keys = sorted(size)
        corr = {
            what: _spearman([size[p] for p in keys], [float(pairs[p][what]) for p in keys])
            for what in ("top3_share", "skew", "excess_kurtosis", "block")
        }
        shared = Counter(d["date"] for e in pairs.values() for d in e["worst"])
        out["subsets"][subset] = {
            "pairs": pairs,
            "spearman_size30": corr,
            "shared_worst_dates": {d: n for d, n in shared.most_common() if n >= MIN_SHARED},
        }
    return out


def _f(x: float | None, fmt: str, scale: float = 1.0) -> str:
    return "n/a" if x is None else format(x * scale, fmt)


def summarize(res: Mapping[str, Any]) -> str:
    lines = [
        f"heaviest days per pair, {res['period'][0]} .. {res['period'][1]}; size = rejections at 5% with "
        f"theta 0 over 30 days ({res['method']}, calendar-day generator); shift = change of theta in bp "
        "if the worst day is dropped"
    ]
    for subset, sub in res["subsets"].items():
        lines.append(f"{subset}: {len(sub['pairs'])} pairs")
        lines.append(
            f"  {'pair':14} {'theta':>6} {'block':>5} {'skew':>5} {'kurt':>6} {'top3':>5} {'normal':>6} "
            f"{'size30':>6}  {'worst day':10} {'shift':>6} {'vol%':>5} {'range%':>6} {'med%':>5} {'bad h':>5}"
        )
        for pair, e in sorted(
            sub["pairs"].items(), key=lambda kv: -(kv[1]["size"]["30d/" + res["method"]] or 0)
        ):
            w = e["worst"][0] if e["worst"] else {}
            lines.append(
                f"  {pair:14} {e['theta'] * 1e4:6.1f} {e['block']:5.2f} {e['skew']:5.1f} "
                f"{e['excess_kurtosis']:6.1f} {e['top3_share']:5.2f} {e['top3_share_normal']:6.2f} "
                f"{_f(e['size']['30d/' + res['method']], '.3f'):>6}  {w.get('date', '-'):10} "
                f"{_f(w.get('shift_bp'), '+.1f'):>6} {_f(w.get('volume_share'), '.1f', 100):>5} "
                f"{_f(w.get('range'), '.1f', 100):>6} {_f(e.get('median_range'), '.1f', 100):>5} "
                f"{w.get('problem_hours', '-')!s:>5}"
            )
        for what, c in sub["spearman_size30"].items():
            if c is not None:
                lines.append(
                    f"  spearman(size30, {what}) {c['rho']:+.2f} (p {c['p']:.3f}, {c['pairs']} pairs)"
                )
        shared = ", ".join(f"{d} ({n})" for d, n in sub["shared_worst_dates"].items()) or "none"
        lines.append(f"  dates among the {TOP} worst days of >= {MIN_SHARED} pairs: {shared}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heaviest calendar days per pair, from the lab's day pools.")
    p.add_argument("daypools", help="output of: python -m mmlab.evaluation.lab pools --kind days")
    p.add_argument("--compare", required=True, help="output of: python -m mmlab.evaluation.lab compare")
    p.add_argument("--dataset", help="dataset the pools were built from, for prices and data problems")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cal = json.loads(Path(a.daypools).read_text())
    cmp = json.loads(Path(a.compare).read_text())["results"]
    res = tails(cal, cmp, Path(a.dataset) if a.dataset else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(summarize(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
