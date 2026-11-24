"""Opening order of November and the holdout pairs: H-003 test, then estimates, then holdout pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmlab.evaluation import h001_h002 as hh
from mmlab.evaluation import h003
from mmlab.hypothesis_registry import holdout
from test_h003 import _dataset


def _result(path: Path, outcome: str, period: str = "test", hypothesis: str = "H-003") -> str:
    path.write_text(
        json.dumps({"hypothesis": hypothesis, "period": period, "summary": {"80.0": {"outcome": outcome}}})
    )
    return str(path)


def test_checks_read_the_h003_result(tmp_path: Path) -> None:
    f = tmp_path / "t.json"
    assert holdout.h003_tested(None) is not None
    assert "no H-003 test result" in str(holdout.h003_tested(str(tmp_path / "none.json")))
    for outcome, tested, unlocked in [
        ("accept", True, True),
        ("reject", True, True),
        ("extend", True, False),
    ]:
        _result(f, outcome)
        assert (holdout.h003_tested(str(f)) is None, holdout.holdout_unlocked(str(f)) is None) == (
            tested,
            unlocked,
        )
    _result(f, "accept", period="extended")
    assert holdout.holdout_unlocked(str(f)) is None
    _result(f, "undecided", period="extended")
    assert "not decided" in str(holdout.holdout_unlocked(str(f)))
    _result(f, "accept", period="exploration")  # a dry run is not a test
    assert holdout.h003_tested(str(f)) is not None
    _result(f, "accept", hypothesis="H-001")
    assert holdout.holdout_unlocked(str(f)) is not None
    f.write_text(json.dumps({"hypothesis": "H-003", "period": "test"}))
    assert holdout.h003_tested(str(f)) is None
    assert holdout.holdout_unlocked(str(f)) is not None
    assert str(h003.THRESHOLD_BP) == holdout.H003_KEY


@pytest.fixture
def universe(tmp_path: Path) -> Path:
    u = tmp_path / "u.json"
    u.write_text(json.dumps({"research": {"0": ["A_USDT", "B_USDT"]}, "holdout": {"0": ["H_USDT"]}}))
    return u


def test_november_estimate_needs_the_h003_test(
    universe: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _dataset(tmp_path, ["A_USDT", "B_USDT"], "2026-11-01T00", "2026-11-30T23")
    out = tmp_path / "n.json"
    argv = [str(ds), "--period", "november", "--universe", str(universe), "--out", str(out)]
    assert hh.main(argv) == 2
    assert hh.main([*argv, "--h003", _result(tmp_path / "t.json", "extend")]) == 0
    doc = json.loads(out.read_text())
    assert (doc["range"], doc["pairs"]) == (["2026-11-01", "2026-12-01"], ["A_USDT", "B_USDT"])
    assert doc["role"].startswith("estimate")
    assert "estimate only" in capsys.readouterr().out
    short = _dataset(tmp_path / "s", ["A_USDT", "B_USDT"], "2026-11-01T00", "2026-11-29T23")
    assert hh.main([str(short), *argv[1:], "--h003", str(tmp_path / "t.json")]) == 2
    nodeals = _dataset(tmp_path / "d", ["A_USDT", "B_USDT"], "2026-11-01T00", "2026-11-30T23", days=92)
    assert hh.main([str(nodeals), *argv[1:], "--h003", str(tmp_path / "t.json")]) == 2


def test_holdout_estimate_needs_a_decided_h003(universe: Path, tmp_path: Path) -> None:
    ds = _dataset(tmp_path, ["H_USDT"], "2026-08-01T00", "2026-11-30T23")
    out = tmp_path / "h.json"
    argv = [str(ds), "--period", "holdout", "--universe", str(universe), "--out", str(out)]
    assert hh.main([*argv, "--h003", _result(tmp_path / "t.json", "extend")]) == 2
    assert hh.main([*argv, "--h003", _result(tmp_path / "t.json", "reject")]) == 0
    doc = json.loads(out.read_text())
    assert (doc["range"], doc["pairs"]) == (["2026-08-01", "2026-12-01"], ["H_USDT"])
    assert doc["decision"]["H-001"]["required"] == 0  # half of the 1 registered holdout pair, rounded down


def test_decision_periods_are_unchanged(universe: Path, tmp_path: Path) -> None:
    ds = _dataset(tmp_path, ["A_USDT", "B_USDT"], "2026-08-01T00", "2026-10-31T23")
    out = tmp_path / "v.json"
    assert hh.main([str(ds), "--period", "validation", "--universe", str(universe), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["role"] == "decision period"
