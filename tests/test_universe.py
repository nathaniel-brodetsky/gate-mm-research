from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest

from mmlab.market_data import universe as u

RULE = u.UniverseRule()


def _series(start: str, end: str, value: str, zero_days: int = 0, skip: str | None = None) -> dict[str, D]:
    d, last, out = date.fromisoformat(start), date.fromisoformat(end), {}
    while d <= last:
        out[d.isoformat()] = D(value)
        d += timedelta(days=1)
    for k in sorted(out)[-zero_days:] if zero_days else []:
        out[k] = D(0)
    if skip:
        del out[skip]
    return out


def ok(value: str) -> dict[str, D]:
    return _series("2026-06-01", "2026-08-31", value)


def test_exclusions() -> None:
    cases = {
        "ABC_BTC": (ok("5000"), "quote"),
        "USDC_USDT": (ok("5000"), "stable_or_pegged"),
        "PAXG_USDT": (ok("5000"), "stable_or_pegged"),
        "BTC3L_USDT": (ok("5000"), "leveraged"),
        "NEW_USDT": (_series("2026-07-01", "2026-08-31", "5000"), "listed_late"),
        "GAP_USDT": (_series("2026-06-01", "2026-08-31", "5000", skip="2026-07-15"), "gap_in_window"),
        "BIG_USDT": (ok("100000"), "volume_out_of_range"),
        "TINY_USDT": (ok("999.99"), "volume_out_of_range"),
        "FINE_USDT": (ok("1000"), None),
    }
    for pair, (days, why) in cases.items():
        assert u.exclusion_reason(pair, days, RULE) == why, pair


def test_inactive_days_threshold() -> None:
    days = _series("2026-06-01", "2026-07-31", "5000")
    for k in ("2026-07-01", "2026-07-02", "2026-07-03"):
        days[k] = D(0)
    assert u.exclusion_reason("X_USDT", days, RULE) is None  # 28 active days
    days["2026-07-04"] = D(0)
    assert u.exclusion_reason("X_USDT", days, RULE) == "inactive"  # 27


def test_median_uses_window_only() -> None:
    days = ok("2000")
    days.update({k: D("90000") for k in days if not k.startswith("2026-07")})  # outside the window: ignored
    assert u.window_median(days, RULE) == D(2000)
    assert u.stratum_of(D(2000), RULE) == 0
    assert [u.stratum_of(D(x), RULE) for x in ("3161.99", "3162", "31623", "99999")] == [0, 1, 3, 3]


def _universe() -> dict[str, dict[str, D]]:
    daily = {}
    for s, value in enumerate(("2000", "5000", "20000", "50000")):
        for i in range(12):
            daily[f"S{s}P{i:02d}_USDT"] = ok(value)
    daily["USDC_USDT"] = ok("5000")
    return daily


def test_selection_is_deterministic_disjoint_and_salted() -> None:
    sel = u.select(_universe(), RULE)
    assert sel.excluded == {"stable_or_pegged": 1}
    for s in range(4):
        assert len(sel.research[s]) == 6
        assert len(sel.holdout[s]) == 3
        assert not set(sel.research[s]) & set(sel.holdout[s])
        assert all(p.startswith(f"S{s}") for p in sel.research[s] + sel.holdout[s])
        assert sel.research[s] == sorted(sel.research[s], key=lambda p: u.order_key(p, RULE.salt))
    again = u.select(dict(reversed(list(_universe().items()))), RULE)
    assert again.research == sel.research  # input order does not matter
    other = u.select(_universe(), replace(RULE, salt="another salt"))
    assert other.research != sel.research


def test_short_stratum_takes_what_exists() -> None:
    daily = {f"A{i}_USDT": ok("2000") for i in range(7)}
    sel = u.select(daily, RULE)
    assert (len(sel.research[0]), len(sel.holdout[0]), sel.research[1]) == (6, 1, [])


def test_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bars = tmp_path / "global_daily_bars.csv"
    with bars.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "day", "quote_volume", "close"])
        for pair, days in _universe().items():
            w.writerows((pair, d, v, "1") for d, v in days.items())
    out = tmp_path / "u.json"
    assert u.main([str(bars), str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["rule"]["id"] == "U-001"
    assert len(doc["input"]["sha256"]) == 64
    assert doc["eligible_per_stratum"] == {"0": 12, "1": 12, "2": 12, "3": 12}
    assert sum(len(v) for v in doc["research"].values()) == 24
    assert len(doc["window_median"]) == 36
    assert "WARNING" not in capsys.readouterr().out
