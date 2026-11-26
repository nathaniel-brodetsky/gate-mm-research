"""Candidate USDT perpetuals for a past window, without survivorship bias (stage 4, step 2b).

Gate's API lists only the contracts trading today, so a universe for 2025 chosen from it would leave
out every contract delisted since. Tardis.dev publishes, free and without a key, the symbols it has
recorded on Gate futures with the dates they were available, delisted ones included
(https://api.tardis.dev/v1/exchanges/gate-io-futures). Both lists are read (public, read-only) and
saved as raw JSON; a contract is a candidate for the window if it was listed at least `--listed-before`
and was still available when the window started. Gate's list adds today's multiplier and launch time.
For every candidate the size of its monthly trades file of the selection month is asked from the
archive (headers only), a first measure of activity before the window.

Usage:
  python -m mmlab.data_ingestion.perp_candidates data/raw/meta results/perp_universe/candidates.csv \\
      --window-start 2025-07-01 --listed-before 2025-06-01 --selection-month 2025-06
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

from mmlab.data_ingestion.gate_archive import BASE_URL, ArchiveRef, ProbeResult, archive_ref, probe

TARDIS_URL: Final = "https://api.tardis.dev/v1/exchanges/gate-io-futures"
GATE_URL: Final = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
USER_AGENT: Final = "mmlab-perp-candidates/0.1"
PAGE: Final = 100  # a page of 1,000 is refused with HTTP 400 (2026-11-26); 100 is accepted
FIELDS: Final = (
    "name",
    "available_since",
    "available_to",
    "listed_on_gate_now",
    "gate_launch_time",
    "gate_quanto_multiplier",
    "gate_in_delisting",
    "trades_status",
    "trades_bytes",
)

Fetch = Callable[[str], Any]


def fetch_json(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gate_contracts(fetch: Fetch, sleep_s: float = 0.2) -> list[dict[str, Any]]:
    """All USDT perpetual contracts Gate lists now, page by page (stops when a page adds nothing new)."""
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        if offset and sleep_s:
            time.sleep(sleep_s)
        page = fetch(f"{GATE_URL}?limit={PAGE}&offset={offset}")
        new = [c for c in page if c["name"] not in out]
        for c in new:
            out[c["name"]] = c
        if len(page) < PAGE or not new:
            return list(out.values())
        offset += PAGE


def _day(iso: str | None) -> date | None:
    return None if not iso else datetime.fromisoformat(iso.replace("Z", "+00:00")).date()


def candidates(
    tardis: dict[str, Any], gate: Sequence[dict[str, Any]], window_start: date, listed_before: date
) -> list[dict[str, Any]]:
    """USDT perpetuals listed before `listed_before` and still available at `window_start`, by name."""
    now = {c["name"]: c for c in gate}
    out = []
    for s in tardis.get("availableSymbols", []):
        name = str(s["id"])
        if s.get("type") != "perpetual" or not name.endswith("_USDT"):
            continue
        since, to = _day(s.get("availableSince")), _day(s.get("availableTo"))
        if since is None or since >= listed_before or (to is not None and to <= window_start):
            continue
        g = now.get(name, {})
        launch = g.get("launch_time")
        out.append(
            {
                "name": name,
                "available_since": since.isoformat(),
                "available_to": "" if to is None else to.isoformat(),
                "listed_on_gate_now": name in now,
                "gate_launch_time": ""
                if not launch
                else datetime.fromtimestamp(int(launch), UTC).date().isoformat(),
                "gate_quanto_multiplier": g.get("quanto_multiplier", ""),
                "gate_in_delisting": g.get("in_delisting", ""),
            }
        )
    return sorted(out, key=lambda r: r["name"])


def gate_only(tardis: dict[str, Any], gate: Sequence[dict[str, Any]], listed_before: date) -> list[str]:
    """Contracts Gate lists now, launched before `listed_before`, that Tardis does not know: a check."""
    known = {str(s["id"]) for s in tardis.get("availableSymbols", [])}
    cut = datetime.combine(listed_before, datetime.min.time(), UTC).timestamp()
    return sorted(
        c["name"]
        for c in gate
        if c["name"] not in known and c.get("launch_time") and int(c["launch_time"]) < cut
    )


def add_trades_sizes(
    rows: list[dict[str, Any]],
    month: str,
    probe_fn: Callable[[ArchiveRef], ProbeResult],
    base: str = BASE_URL,
    sleep_s: float = 0.25,
) -> None:
    for i, r in enumerate(rows):
        if i:
            time.sleep(sleep_s)
        res = probe_fn(archive_ref("futures_usdt", "trades", r["name"], month.replace("-", ""), base=base))
        r["trades_status"] = res.status
        r["trades_bytes"] = res.size_bytes


def main(argv: Sequence[str] | None = None, fetch: Fetch = fetch_json) -> int:
    p = argparse.ArgumentParser(description="Candidate USDT perpetuals for a past window.")
    p.add_argument("raw_dir", help="where the raw JSON answers are saved")
    p.add_argument("out", help="candidates CSV")
    p.add_argument("--window-start", default="2025-07-01")
    p.add_argument("--listed-before", default="2025-06-01")
    p.add_argument("--selection-month", default="2025-06")
    p.add_argument("--base", default=BASE_URL)
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--timeout", type=float, default=20.0)
    a = p.parse_args(argv)
    raw = Path(a.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tardis = fetch(TARDIS_URL)
    gate = gate_contracts(fetch)
    (raw / f"tardis_gate-io-futures_{stamp}.json").write_text(json.dumps(tardis))
    (raw / f"gate_futures_usdt_contracts_{stamp}.json").write_text(json.dumps(gate))
    ws, lb = date.fromisoformat(a.window_start), date.fromisoformat(a.listed_before)
    rows = candidates(tardis, gate, ws, lb)
    add_trades_sizes(rows, a.selection_month, lambda r: probe(r, timeout=a.timeout), a.base, a.sleep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    missing = gate_only(tardis, gate, lb)
    delisted = sum(not r["listed_on_gate_now"] for r in rows)
    with_file = sum(r["trades_status"] in (200, 206) for r in rows)
    print(
        f"raw answers saved {stamp}; Tardis symbols {len(tardis.get('availableSymbols', []))}, Gate USDT "
        f"contracts now {len(gate)}; candidates {len(rows)} (listed before {lb}, available on {ws}), "
        f"{delisted} of them no longer listed; {with_file} with a trades file for {a.selection_month}; "
        f"Gate contracts launched before {lb} unknown to Tardis: {len(missing)}"
        + (f" ({', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''})" if missing else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
