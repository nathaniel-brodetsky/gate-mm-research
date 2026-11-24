"""Comparison of result files up to floating-point rounding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmlab.evaluation import compare as cmp


def test_floats_within_and_beyond_rounding() -> None:
    a = {"x": 0.1, "y": [1.0, 2.0], "z": {"w": 1e-15}, "n": 3}
    b = {"x": 0.1 + 1e-17, "y": [1.0, 2.0 + 1e-8], "z": {"w": 2e-15}, "n": 3}
    beyond, within = cmp.compare(a, b)
    assert [d.path for d in beyond] == ["/y[1]"]  # 5e-9 relative
    assert sorted(d.path for d in within) == ["/x", "/z/w"]  # 1.4e-16 relative; 1e-15 absolute


@pytest.mark.parametrize(
    ("a", "b", "ok"),
    [
        (1.0, 1.0 + 0.9e-9, True),  # relative limit 1e-9
        (1.0, 1.0 + 1.1e-9, False),
        (1e-6, 1e-6 + 0.9e-12, True),  # absolute limit 1e-12 (here 9e-7 relative)
        (1e-6, 1e-6 + 1.1e-12, False),
        (0.0, 0.9e-12, True),
        (0.0, 1.1e-12, False),
    ],
)
def test_tolerance_edges(a: float, b: float, ok: bool) -> None:
    beyond, _ = cmp.compare({"v": a}, {"v": b})
    assert (not beyond) is ok


def test_everything_else_must_be_equal() -> None:
    a = {"i": 1, "s": "x", "t": True, "l": [1, 2], "only_a": 1, "f": 1.0, "n": None}
    b = {"i": 2, "s": "y", "t": False, "l": [1, 2, 3], "only_b": 1, "f": 1, "n": 0.0}
    beyond, within = cmp.compare(a, b)
    assert within == []
    got = {d.path: (d.a, d.b) for d in beyond}
    assert got == {
        "/f": (1.0, 1),  # float against int: a type change, not rounding
        "/i": (1, 2),
        "/l/len": (2, 3),
        "/n": (None, 0.0),
        "/only_a": (1, cmp.MISSING),
        "/only_b": (cmp.MISSING, 1),
        "/s": ("x", "y"),
        "/t": (True, False),
    }


def test_ignored_top_level_keys() -> None:
    beyond, _ = cmp.compare({"role": "estimate", "v": 1}, {"v": 1}, ignore=["role"])
    assert beyond == []
    beyond, _ = cmp.compare({"role": "estimate", "v": 1}, {"v": 1})
    assert [d.path for d in beyond] == ["/role"]


def test_rel_is_reported_for_floats_only() -> None:
    assert cmp.Difference("/x", 2.0, 1.0).rel == pytest.approx(0.5)
    assert cmp.Difference("/x", 2, 1).rel is None
    assert cmp.Difference("/x", 0.0, 0.0).rel == 0.0


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"role": "x", "r": [{"t": 1.0, "n": 5}]}))
    b.write_text(json.dumps({"r": [{"t": 1.0 + 4e-14, "n": 5}]}))
    assert cmp.main([str(a), str(b), "--ignore", "role"]) == 0
    assert (
        "1 floats differ within rounding (max rel 4.0e-14); 0 differences beyond" in capsys.readouterr().out
    )
    assert cmp.main([str(a), str(b)]) == 1
    assert "/role: 'x' -> '<missing>'" in capsys.readouterr().out
