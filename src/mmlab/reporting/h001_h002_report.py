"""Markdown report of the H-001 and H-002 tests, generated from mmlab.evaluation.h001_h002 JSON output.

The report is never edited by hand: every number comes from the JSON files, whose SHA-256 is printed next
to their paths, so the report can be regenerated and checked.

Usage:
  python -m mmlab.reporting.h001_h002_report results/h001_h002/development.json \\
      results/h001_h002/development_no_spyx.json --out docs/results/h001-h002-development.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from mmlab.microstructure.markouts import HORIZONS_S, _hkey

RS_REPORTED_S = (1.0, 10.0, 60.0, 300.0)  # H-001: 60 s is tested, the others are reported


def _bp(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e4:.1f}"


def _ci_bp(ci: Sequence[float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0] * 1e4:.1f}, {ci[1] * 1e4:.1f}]"


def _ratio(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _ci_ratio(ci: Sequence[float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _median(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return statistics.median(v) if v else None


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _sep(n: int, left: int = 1) -> str:
    """Separator row for n columns: the first `left` aligned left, the rest right."""
    return "|" + "|".join(["---"] * left + ["--:"] * (n - left)) + "|"


def _last_day(end_exclusive: str) -> str:
    return (date.fromisoformat(end_exclusive) - timedelta(days=1)).isoformat()


def _section(path: str, digest: str, doc: dict[str, Any]) -> list[str]:
    results: list[dict[str, Any]] = doc["results"]
    d1 = doc["decision"]["H-001"]
    d2 = doc["decision"]["H-002"]
    prm = doc["params"]
    start, end = doc["range"]
    title = f"## {doc['period'].capitalize()}, {start} to {_last_day(end)}, {len(doc['pairs'])} pairs"
    if doc.get("excluded"):
        title += " (excluded: " + ", ".join(doc["excluded"]) + ")"
    out = [
        title,
        "",
        f"Source: `{path}`, sha256 `{digest}`.",
        f"Parameters: maker fee {prm['maker_fee'] * 1e4:.1f} bp, {prm['resamples']} resamples, mean block "
        f"{prm['mean_block_days']:g} days, seeds {' / '.join(str(s) for s in prm['seeds'])}, "
        f"BH q = {prm['bh_q']:g}, at least {prm['min_trades']} trades and {prm['min_days']} days per pair.",
        "",
        "### H-001: realized spread at 60 s against the maker fee",
        "",
    ]
    rejected = set(d1["pairs_rejected"])
    head = ["Pair", "Trades", "Days", "ES", *[f"RS {h:g} s" for h in RS_REPORTED_S]]
    head += ["θ", "θ 95% CI", "p", "BH", "θ microprice", "Note"]
    out += [_row(head), _sep(len(head))]
    for r in results:
        out.append(
            _row(
                [
                    r["pair"],
                    str(r["trades"]),
                    str(r["days"]),
                    _bp(r["es"]),
                    *[_bp(r["rs"].get(_hkey(h))) for h in RS_REPORTED_S],
                    _bp(r["h001_theta"]),
                    _ci_bp(r["h001_ci"]),
                    f"{r['h001_p']:.4f}",
                    "reject" if r["pair"] in rejected else "",
                    _bp(r["h001_theta_micro"]),
                    r["note"],
                ]
            )
        )
    med = d1["median_theta"]
    out += [
        "",
        f"Criterion inputs: {d1['rejections_after_bh']} of {len(doc['pairs'])} pairs reject H0 after BH "
        f"(required: {d1['required']}); cross-pair median θ = {_bp(med)} bp; "
        f"criterion met in this period: {'yes' if d1['criterion_met'] else 'no'}.",
        "",
        "### H-002: price impact curve Λ(h)",
        "",
        "Λ(h) by pair, bp:",
        "",
    ]
    hs = [_hkey(h) for h in HORIZONS_S]
    head = ["Pair", *[f"{h:g} s" for h in HORIZONS_S], "S", "S 95% CI"]
    out += [_row(head), _sep(len(head))]
    for r in results:
        out.append(
            _row(
                [
                    r["pair"],
                    *[_bp(r["impact"].get(k)) for k in hs],
                    _ratio(r["h002_s"]),
                    _ci_ratio(r["h002_ci"]),
                ]
            )
        )
    meds = [_median([r["impact"].get(k) for r in results]) for k in hs]
    out.append(
        _row(["**Median**", *[_bp(m) for m in meds], _ratio(_median([r["h002_s"] for r in results])), ""])
    )

    shares = [
        (
            r["pair"],
            [r["impact"][k] / r["impact"]["300s"] if r["impact"].get(k) is not None else None for k in hs],
        )
        for r in results
        if (r["impact"].get("300s") or 0) > 0
    ]
    out += ["", "Λ(h) / Λ(300 s), pairs with Λ(300 s) > 0:", ""]
    head = ["Pair", *[f"{h:g} s" for h in HORIZONS_S[:-1]]]
    out += [_row(head), _sep(len(head))]
    for pair, sh in shares:
        out.append(_row([pair, *[_ratio(x) for x in sh[:-1]]]))
    out.append(
        _row(["**Median**", *[_ratio(_median([sh[i] for _, sh in shares])) for i in range(len(hs) - 1)]])
    )
    out += [
        "",
        "Λ(h) uses every trade usable at h; S uses only trades usable at both 1 s and 300 s, so S and the "
        "1 s share above can differ slightly.",
        "",
    ]
    if "median_s" in d2:
        out.append(
            f"Criterion inputs: S defined for {d2['pairs_defined']} of {len(doc['pairs'])} pairs; "
            f"cross-pair median S = "
            f"{d2['median_s']:.3f}, 95% CI {_ci_ratio(d2['ci'])}; outcome in this period: {d2['outcome']}."
        )
    else:
        out.append(
            f"Criterion inputs: S defined for {d2['pairs_defined']} of {len(doc['pairs'])} pairs; "
            f"outcome in this period: {d2['outcome']}."
        )
    return [*out, ""]


def render(inputs: Sequence[tuple[str, str, dict[str, Any]]]) -> str:
    """inputs: (path as shown, sha256 hex digest, parsed JSON) per result file."""
    lines = [
        "# H-001 and H-002 results",
        "",
        "Generated by `python -m mmlab.reporting.h001_h002_report`; do not edit by hand.",
        "Hypotheses: [H-001](../../hypotheses/H-001-maker-realized-spread.md), "
        "[H-002](../../hypotheses/H-002-latency-share-of-impact.md).",
        "",
        "Units: bp = 0.01%. ES = effective half-spread, RS = realized spread, θ = RS(60 s) - maker fee, "
        "Λ(h) = volume-weighted price impact h after the trade, S = Λ(1 s) / Λ(300 s). "
        "Trades = trades with a usable mid at the trade time. "
        "Resolution limit: the book is known in 100 ms batches, so 0.1 s means the next batch.",
        "",
    ]
    for path, digest, doc in inputs:
        lines += _section(path, digest, doc)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render H-001 / H-002 result JSON files as Markdown.")
    p.add_argument("results", nargs="+", help="JSON files written by mmlab.evaluation.h001_h002")
    p.add_argument("--out", help="write the report here instead of stdout")
    a = p.parse_args(argv)
    inputs = []
    for path in a.results:
        raw = Path(path).read_bytes()
        inputs.append((path, hashlib.sha256(raw).hexdigest(), json.loads(raw)))
    text = render(inputs)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
