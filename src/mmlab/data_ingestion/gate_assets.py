"""Asset metadata for the research pairs of a universe file, from Gate public REST (no credentials).

Used to identify assets that are not crypto-native (for example tokenized equities with external
trading hours), which D-001 requires to be flagged, not removed. Only research pairs are queried:
holdout pairs stay closed (D-001).

Usage:
  python -m mmlab.data_ingestion.gate_assets configs/universe_U001.json configs/universe_U001_assets.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mmlab.data_ingestion.gate_venues import GLOBAL_URL, Venue

_ERRORS = (RuntimeError, OSError, http.client.HTTPException, ValueError)


def research_pairs(universe: dict[str, Any]) -> list[str]:
    return [p for s in sorted(universe["research"], key=int) for p in universe["research"][s]]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch Gate asset metadata for the research pairs of a universe.")
    p.add_argument("universe")
    p.add_argument("out")
    p.add_argument("--base", default=GLOBAL_URL)
    a = p.parse_args(argv)
    universe = json.loads(Path(a.universe).read_text())
    v = Venue("global", a.base)
    out: dict[str, Any] = {}
    failed = 0
    for pair in research_pairs(universe):
        base = pair.rsplit("_", 1)[0]
        try:
            cur, _ = v.get(f"/spot/currencies/{base}")
            pr, _ = v.get(f"/spot/currency_pairs/{pair}")
            out[pair] = {"currency": cur, "pair": pr}
        except _ERRORS as e:
            out[pair] = {"error": str(e)[:300]}
            failed += 1
        info = out[pair]
        name = info.get("currency", {}).get("name", "?") if "currency" in info else "ERROR"
        print(f"  {pair:16} {name}")
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
