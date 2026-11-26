"""Universe U-002: exclusions, strata edges, ordering by the salted hash, the CLI output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mmlab.market_data import perp_universe as pu
from mmlab.market_data.universe import order_key

RULE = pu.PerpRule()


def _act(name: str, med: float, days: int = 30, status: str = "ok") -> dict[str, str]:
    return {"name": name, "status": status, "active_days": str(days), "median_daily_trades": str(med)}


def _cand(name: str, listed: bool = True) -> dict[str, str]:
    return {"name": name, "listed_on_gate_now": str(listed)}


def test_exclusion_reasons() -> None:
    assert pu.exclusion_reason(_cand("USDC_USDT"), _act("USDC_USDT", 500), RULE) == "stable or pegged"
    assert pu.exclusion_reason(_cand("A_USDT"), None, RULE) == "no trades file"
    assert (
        pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 500, status="missing"), RULE) == "no trades file"
    )
    assert pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 500, days=27), RULE) == (
        "active on fewer than 28 days"
    )
    assert pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 500, days=28), RULE) is None
    assert pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 99.5), RULE) == "below the lowest stratum"
    assert pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 100), RULE) is None
    assert pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 10_000), RULE) == "above the highest stratum"
    assert pu.exclusion_reason(_cand("A_USDT"), _act("A_USDT", 9_999.5), RULE) is None


def test_strata_edges() -> None:
    assert pu.stratum_of(100, RULE) == 0
    assert pu.stratum_of(316.2, RULE) == 0
    assert pu.stratum_of(316.3, RULE) == 1
    assert pu.stratum_of(999.5, RULE) == 1
    assert pu.stratum_of(1_000, RULE) == 2
    assert pu.stratum_of(3_162.2, RULE) == 2
    assert pu.stratum_of(3_162.3, RULE) == 3
    with pytest.raises(ValueError, match="outside"):
        pu.stratum_of(10_000, RULE)
    assert pu.base_of("1000PEPE_USDT") == "1000PEPE"


def test_select_orders_by_the_salted_hash() -> None:
    names = [f"C{i:02d}_USDT" for i in range(12)]
    cands = [_cand(n) for n in [*names, "LOW_USDT", "DAI_USDT"]]
    acts = [_act(n, 2_000) for n in names] + [_act("LOW_USDT", 150), _act("DAI_USDT", 2_000)]
    sel = pu.select(cands, acts, RULE)
    ordered = sorted(names, key=lambda n: order_key(n, pu.SALT))
    assert ordered != sorted(names)  # the hash, not the name, decides
    assert sel.research[2] == ordered[:6]
    assert sel.holdout[2] == ordered[6:9]
    assert sel.research[0] == ["LOW_USDT"]
    assert sel.holdout[0] == []
    assert sel.eligible[1] == sel.eligible[3] == []
    assert sel.excluded == {"stable or pegged": 1}


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    names = [f"C{i:02d}_USDT" for i in range(10)]
    _write(tmp_path / "cand.csv", [_cand(n, listed=i % 2 == 0) for i, n in enumerate(names)])
    _write(tmp_path / "act.csv", [_act(n, 500 + i) for i, n in enumerate(names)])
    out = tmp_path / "cfg" / "u.json"
    assert pu.main([str(tmp_path / "cand.csv"), str(tmp_path / "act.csv"), str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["rule"]["salt"] == pu.SALT
    assert doc["eligible_per_stratum"] == {"0": 0, "1": 10, "2": 0, "3": 0}
    assert len(doc["research"]["1"]) == 6
    assert len(doc["holdout"]["1"]) == 3
    chosen = doc["research"]["1"] + doc["holdout"]["1"]
    assert set(doc["june_median_daily_trades"]) == set(chosen)
    assert doc["delisted_since"] == sorted(n for n in chosen if int(n[1:3]) % 2 == 1)
    assert len(doc["input"]["activity"]["sha256"]) == 64
    text = capsys.readouterr().out
    assert "stratum 1 [316.2, 1,000.0): eligible 10 (5 delisted since), research 6, holdout 3" in text
    assert "WARNING: strata with fewer than 9 eligible contracts: [0, 2, 3]" in text
