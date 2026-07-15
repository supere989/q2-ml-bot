from __future__ import annotations

import ast
import io
from pathlib import Path
import subprocess
import token
import tokenize

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _tracked_python_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = tuple(
        ROOT / entry.decode("utf-8")
        for entry in result.stdout.split(b"\0")
        if entry
    )
    assert paths, "git reported no tracked Python files"
    return paths


TRACKED_PYTHON_FILES = _tracked_python_files()


def _source(path: Path) -> str:
    with tokenize.open(path) as stream:
        return stream.read()


@pytest.mark.parametrize(
    "path",
    TRACKED_PYTHON_FILES,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_tracked_python_file_uses_python310_grammar(path: Path) -> None:
    ast.parse(
        _source(path),
        filename=str(path.relative_to(ROOT)),
        feature_version=(3, 10),
    )


@pytest.mark.parametrize(
    "path",
    TRACKED_PYTHON_FILES,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_tracked_python_file_has_no_multiline_nontriple_fstring(
    path: Path,
) -> None:
    source = _source(path)
    fstring_start = getattr(token, "FSTRING_START", None)
    fstring_end = getattr(token, "FSTRING_END", None)
    if fstring_start is None or fstring_end is None:
        # Python 3.10 rejects this pre-PEP-701 construct directly. Keep this
        # test useful when the suite itself is run by the deployment runtime.
        compile(source, str(path.relative_to(ROOT)), "exec", dont_inherit=True)
        return

    starts: list[tokenize.TokenInfo] = []
    violations: list[tuple[int, int]] = []
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        if item.type == fstring_start:
            starts.append(item)
        elif item.type == fstring_end:
            assert starts, "tokenizer emitted an unmatched FSTRING_END"
            start = starts.pop()
            triple_quoted = start.string.endswith(("'''", '"""'))
            if not triple_quoted and start.start[0] != item.end[0]:
                violations.append((start.start[0], item.end[0]))

    assert not starts, "tokenizer emitted an unmatched FSTRING_START"
    assert not violations, (
        "Python 3.10 cannot parse non-triple f-strings spanning physical "
        f"lines: {violations}"
    )
