"""The H-001 / H-002 report renders exactly what the evaluation JSON contains."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mmlab.evaluation import h001_h002 as hh
from mmlab.reporting import h001_h002_report as rep
from test_h001_h002 import impact_pair, spread_pair


def _row(text: str, section: str, pair: str) -> list[str]:
    part = text.split(section, 1)[1]
    line = next(x for x in part.splitlines() if x.startswith(f"| {pair} |"))
    return [c.strip() for c in line.strip("|").split("|")]


@pytest.fixture
def results(tmp_path: Path) -> tuple[Path, Path]:
    ds = tmp_path / "ds"
    for sub in ("tob", "trades"):
        (ds / sub).mkdir(parents=True)
    pairs = {"GOOD_USDT": spread_pair(0.01), "IMP_USDT": impact_pair(), "SPYX_USDT": spread_pair(0.0)}
    for name, (tob, tr) in pairs.items():
        tob.write_parquet(ds / "tob" / f"{name}.parquet")
        tr.write_parquet(ds / "trades" / f"{name}.parquet")
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": list(pairs)}, "holdout": {"0": ["HOLD_USDT"]}}))
    full, part = tmp_path / "dev.json", tmp_path / "dev_no_spyx.json"
    base = [str(ds), "--period", "development", "--universe", str(uni)]
    assert hh.main([*base, "--out", str(full)]) == 0
    assert hh.main([*base, "--exclude", "SPYX_USDT", "--out", str(part)]) == 0
    return full, part


def test_report_matches_results(results: tuple[Path, Path], tmp_path: Path) -> None:
    full, part = results
    out = tmp_path / "docs" / "report.md"
    assert rep.main([str(full), str(part), "--out", str(out)]) == 0
    text = out.read_text()

    assert "## Development, 2026-08-01 to 2026-09-30, 3 pairs\n" in text
    assert "## Development, 2026-08-01 to 2026-09-30, 2 pairs (excluded: SPYX_USDT)\n" in text
    assert f"sha256 `{hashlib.sha256(full.read_bytes()).hexdigest()}`" in text
    assert f"sha256 `{hashlib.sha256(part.read_bytes()).hexdigest()}`" in text

    head = "| Pair | Trades | Days | ES | RS 1 s | RS 10 s | RS 60 s | RS 300 s | θ | θ 95% CI | p | BH |"
    assert head in text
    good = _row(text, "### H-001", "GOOD_USDT")
    assert good[3:8] == ["100.0", "100.0", "100.0", "100.0", "100.0"]  # ES, RS 1/10/60/300 s
    assert good[8] == "90.0"  # theta = 1% - 0.10% fee
    assert good[11] == "reject"
    flat = _row(text, "### H-001", "SPYX_USDT")
    assert (flat[8], flat[11]) == ("-10.0", "")
    first, second = text.split("## Development")[1:]
    assert "1 of 3 pairs reject H0 after BH (required: 1)" in first
    assert "criterion met in this period: no" in first  # median theta of (90, -30, -10) bp <= 0
    assert "1 of 2 pairs reject H0 after BH (required: 1)" in second
    assert "criterion met in this period: yes" in second  # median of (90, -30) bp > 0

    imp = _row(text, "Λ(h) by pair", "IMP_USDT")
    assert imp[1:4] == ["0.0", "0.0", "20.0"]  # 0.1 s, 0.2 s: not yet moved; 0.5 s: +0.2%
    assert imp[4] == "20.0"
    assert imp[10] == "100.0"
    assert imp[11:13] == ["0.20", "[0.20, 0.20]"]
    med = _row(text, "Λ(h) by pair", "**Median**")
    assert med[1:12] == ["0.0"] * 10 + ["0.20"]  # medians over (GOOD, IMP, SPYX); S only defined for IMP
    share = _row(text, "Λ(h) / Λ(300 s)", "IMP_USDT")
    assert share[1:5] == ["0.00", "0.00", "0.20", "0.20"]
    assert share[-1] == "0.20"  # 60 s
    assert _row(text, "Λ(h) / Λ(300 s)", "**Median**") == ["**Median**", "0.00", "0.00"] + ["0.20"] * 7
    assert "| GOOD_USDT |" not in text.split("Λ(h) / Λ(300 s)", 1)[1].split("##", 1)[0]  # Λ(300 s) = 0
    assert "S defined for 1 of 3 pairs; cross-pair median S = 0.200, 95% CI [0.20, 0.20]" in text
    assert "outcome in this period: accept" in text


def test_tables_are_well_formed(results: tuple[Path, Path]) -> None:
    text = rep.render([("r.json", "0" * 64, json.loads(results[0].read_text()))])
    lines = text.splitlines()
    tables = 0
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            tables += 1
            width = lines[i - 1].count("|")
            assert line.count("|") == width
            body = lines[i + 1 : next(j for j in range(i + 1, len(lines)) if not lines[j].startswith("|"))]
            assert body
            assert all(row.count("|") == width for row in body)
    assert tables == 3


def test_h002_without_defined_pairs() -> None:
    doc = {
        "period": "validation",
        "range": ["2026-10-01", "2026-11-01"],
        "pairs": ["A_USDT"],
        "excluded": [],
        "params": {
            "maker_fee": 0.001,
            "resamples": 10,
            "mean_block_days": 3.0,
            "seeds": [1, 2],
            "bh_q": 0.1,
            "min_trades": 30,
            "min_days": 5,
        },
        "results": [
            {
                "pair": "A_USDT",
                "trades": 0,
                "days": 0,
                "es": None,
                "rs": {},
                "impact": {},
                "h001_theta": None,
                "h001_ci": None,
                "h001_p": 1.0,
                "h001_theta_micro": None,
                "h002_s": None,
                "h002_ci": None,
                "note": "no usable trades",
            }
        ],
        "decision": {
            "H-001": {
                "rejections_after_bh": 0,
                "required": 1,
                "median_theta": None,
                "pairs_rejected": [],
                "criterion_met": False,
            },
            "H-002": {"pairs_defined": 0, "outcome": "undecided"},
        },
    }
    text = rep.render([("v.json", "f" * 64, doc)])
    assert "## Validation, 2026-10-01 to 2026-10-31, 1 pairs\n" in text
    assert (
        "| A_USDT | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.0000 |  | n/a | no usable trades |"
        in text
    )
    assert "0 of 1 pairs reject H0 after BH (required: 1)" in text
    assert "cross-pair median θ = n/a bp" in text
    assert "S defined for 0 of 1 pairs; outcome in this period: undecided." in text


def test_report_to_stdout(
    results: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rep.main([str(results[0])]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("# H-001 and H-002 results\n")
    assert "excluded" not in printed
    assert rep.main([str(results[0]), "--out", str(tmp_path / "r.md")]) == 0
    assert (tmp_path / "r.md").read_text() == printed


def test_helpers() -> None:
    assert rep._median([None, 3.0, 1.0, None, 2.0]) == 2.0
    assert rep._median([None]) is None
    assert rep._last_day("2026-10-01") == "2026-09-30"
    assert rep._last_day("2026-11-01") == "2026-10-31"
    assert (rep._bp(None), rep._ci_bp(None), rep._ratio(None), rep._ci_ratio(None)) == ("n/a",) * 4
    assert rep._bp(-0.00096339) == "-9.6"
    assert rep._ci_bp((-0.0012, 0.0003)) == "[-12.0, 3.0]"
