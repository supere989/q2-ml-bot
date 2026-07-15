from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import tools.check_python_syntax_floor as syntax_floor


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools/check_python_syntax_floor.py"


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _snapshot(root: Path) -> dict[str, tuple[int, int, int, str]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        metadata = path.stat()
        payload = path.read_bytes()
        result[path.relative_to(root).as_posix()] = (
            metadata.st_mode,
            metadata.st_mtime_ns,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
    return result


def test_git_enumeration_checks_exact_tracked_python_files(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "good.py").write_text("value = 1\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "also_good.py").write_text("def answer():\n    return 42\n")
    (tmp_path / "untracked_bad.py").write_text("broken = [\n")
    _git(tmp_path, "add", "good.py", "nested/also_good.py")

    report = syntax_floor.check_syntax(tmp_path)

    assert report["enumeration"] == "git-tracked"
    assert report["file_count"] == 2
    assert report["failures"] == []
    assert report["passed"] is True


def test_snapshot_fallback_has_fixed_exclusions_and_reports_syntax(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(syntax_floor.shutil, "which", lambda _name: None)
    (tmp_path / "good.py").write_text("value = 1\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "bad.py").write_text("value = (\n")
    for excluded in ("__pycache__", ".venv", "target", "training-data"):
        directory = tmp_path / excluded
        directory.mkdir()
        (directory / "ignored_bad.py").write_text("value = (\n")

    report = syntax_floor.check_syntax(tmp_path)

    assert report["enumeration"] == "recursive-snapshot"
    assert report["file_count"] == 2
    assert report["passed"] is False
    assert [failure["path"] for failure in report["failures"]] == [
        "nested/bad.py"
    ]
    assert report["failures"][0]["kind"] == "syntax"


def test_cli_is_canonical_nonwriting_and_nonzero_on_failure(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    source = tmp_path / "checked.py"
    source.write_text("value = 1\n")
    _git(tmp_path, "add", "checked.py")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    before = _snapshot(tmp_path)
    passed = subprocess.run(
        [sys.executable, "-B", str(CHECKER), "--root", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )
    after = _snapshot(tmp_path)

    assert passed.returncode == 0
    assert passed.stderr == b""
    passed_report = json.loads(passed.stdout)
    assert passed.stdout == syntax_floor.canonical_bytes(passed_report)
    assert passed_report["passed"] is True
    assert before == after
    assert not list(tmp_path.rglob("*.pyc"))

    source.write_text("value = (\n")
    failed = subprocess.run(
        [sys.executable, "-B", str(CHECKER), "--root", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )
    failed_report = json.loads(failed.stdout)
    assert failed.returncode == 1
    assert failed.stderr == b""
    assert failed.stdout == syntax_floor.canonical_bytes(failed_report)
    assert failed_report["passed"] is False
    assert failed_report["failures"][0]["path"] == "checked.py"
