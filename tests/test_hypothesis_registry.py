from pathlib import Path

import pytest

from mmlab.hypothesis_registry import registry as r

REPO = Path(__file__).resolve().parents[1]


def test_repository_registry_is_valid() -> None:
    hs = r.load_all(REPO / "hypotheses")
    assert [h.id for h in hs][:2] == ["H-001", "H-002"]
    assert all(h.status in r.STATUSES for h in hs)
    assert r.main([str(REPO / "hypotheses")]) == 0


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def _valid(hid: str = "H-007") -> str:
    body = "\n".join(f"## {s}\n\ntext\n" for s in r.SECTIONS)
    return f"# {hid}: Title\n\n- **Status:** draft\n- **Registered:** 2026-11-23\n\n{body}"


def test_valid_file_parses(tmp_path: Path) -> None:
    h = r.parse(_write(tmp_path, "H-007-x.md", _valid()))
    assert (h.id, h.title, h.status, str(h.registered)) == ("H-007", "Title", "draft", "2026-11-23")


@pytest.mark.parametrize(
    ("name", "text", "msg"),
    [
        ("H-7-x.md", _valid(), "file name"),
        ("H-008-x.md", _valid("H-007"), "does not match"),
        ("H-007-x.md", _valid().replace("draft", "maybe"), "status"),
        ("H-007-x.md", _valid().replace("- **Registered:** 2026-11-23\n", ""), "registration date"),
        ("H-007-x.md", _valid().replace("## Baseline\n", "## Base line\n"), "section missing: Baseline"),
    ],
)
def test_invalid_files(tmp_path: Path, name: str, text: str, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        r.parse(_write(tmp_path, name, text))


def test_duplicate_ids(tmp_path: Path) -> None:
    _write(tmp_path, "H-007-a.md", _valid())
    _write(tmp_path, "H-007-b.md", _valid())
    with pytest.raises(ValueError, match="duplicate"):
        r.load_all(tmp_path)
    assert r.main([str(tmp_path)]) == 1
