#!/usr/bin/env python3
"""Check every repository Python source under the running interpreter.

The checker deliberately compiles source bytes in memory.  It does not import
repository modules, create bytecode, or write an output file.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "q2-python-syntax-floor-v1"
SNAPSHOT_EXCLUDED_DIRECTORIES = frozenset({
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "env",
    "node_modules",
    "runs",
    "target",
    "training-data",
    "venv",
})


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_sort_key(path: Path) -> bytes:
    return os.fsencode(path.as_posix())


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git_root(root: Path) -> Path | None:
    git = shutil.which("git")
    if git is None:
        return None
    completed = subprocess.run(
        [git, "-C", str(root), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_git_environment(),
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        repository = Path(os.fsdecode(completed.stdout.rstrip(b"\n"))).resolve()
        root.relative_to(repository)
    except (OSError, ValueError):
        return None
    return repository


def _git_python_files(root: Path, repository: Path) -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise OSError("git disappeared during tracked-file enumeration")
    completed = subprocess.run(
        [git, "-C", str(repository), "ls-files", "-z", "--cached", "--", "*.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"git ls-files failed: {message or completed.returncode}")
    paths: list[Path] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        tracked = Path(os.fsdecode(raw_path))
        if tracked.is_absolute() or ".." in tracked.parts:
            raise OSError("git returned a non-relative tracked path")
        try:
            relative = (repository / tracked).relative_to(root)
        except ValueError:
            continue
        paths.append(relative)
    return sorted(paths, key=_path_sort_key)


def _snapshot_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directories, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error,
    ):
        base = Path(directory)
        kept_directories = []
        for name in sorted(directories, key=os.fsencode):
            candidate = base / name
            if name in SNAPSHOT_EXCLUDED_DIRECTORIES or candidate.is_symlink():
                continue
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(filenames, key=os.fsencode):
            if name.endswith(".py"):
                paths.append((base / name).relative_to(root))
    return sorted(paths, key=_path_sort_key)


def enumerate_python_files(root: Path) -> tuple[str, list[Path]]:
    repository = _git_root(root)
    if repository is not None:
        return "git-tracked", _git_python_files(root, repository)
    return "recursive-snapshot", _snapshot_python_files(root)


def _files_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    first = True
    for path in paths:
        if not first:
            digest.update(b"\0")
        digest.update(os.fsencode(path.as_posix()))
        first = False
    return digest.hexdigest()


def _read_failure(path: str, error: OSError) -> dict[str, Any]:
    if isinstance(error, FileNotFoundError):
        message = "file not found"
    elif isinstance(error, PermissionError):
        message = "permission denied"
    elif isinstance(error, IsADirectoryError):
        message = "not a regular file"
    else:
        message = error.strerror or type(error).__name__
    return {"kind": "read", "message": message, "path": path}


def _syntax_failure(path: str, error: BaseException) -> dict[str, Any]:
    if isinstance(error, SyntaxError):
        return {
            "kind": "syntax",
            "line": error.lineno,
            "message": error.msg,
            "offset": error.offset,
            "path": path,
        }
    return {
        "kind": "syntax",
        "line": None,
        "message": str(error),
        "offset": None,
        "path": path,
    }


def check_syntax(root: Path) -> dict[str, Any]:
    root = root.resolve()
    failures: list[dict[str, Any]] = []
    try:
        if not root.is_dir():
            raise OSError("root is not a directory")
        enumeration, files = enumerate_python_files(root)
    except OSError as error:
        enumeration = "failed"
        files = []
        failures.append({
            "kind": "enumeration",
            "message": error.strerror or str(error),
            "path": ".",
        })

    for relative in files:
        display_path = relative.as_posix()
        source_path = root / relative
        try:
            metadata = source_path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise IsADirectoryError(display_path)
            payload = source_path.read_bytes()
        except OSError as error:
            failures.append(_read_failure(display_path, error))
            continue
        try:
            compile(payload, display_path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OverflowError) as error:
            failures.append(_syntax_failure(display_path, error))

    executable = Path(sys.executable).resolve()
    executable_sha256 = None
    try:
        executable_sha256 = _file_sha256(executable)
    except OSError as error:
        failures.append(_read_failure("@interpreter", error))

    failures.sort(key=lambda item: (
        os.fsencode(str(item["path"])),
        str(item["kind"]),
        str(item["message"]),
    ))
    return {
        "enumeration": enumeration,
        "failures": failures,
        "file_count": len(files),
        "files_sha256": _files_sha256(files),
        "interpreter": {
            "executable": str(executable),
            "implementation": sys.implementation.name,
            "sha256": executable_sha256,
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
        },
        "passed": not failures,
        "schema": SCHEMA,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    report = check_syntax(arguments.root)
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
