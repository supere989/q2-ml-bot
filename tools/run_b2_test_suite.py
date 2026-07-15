#!/usr/bin/env python3
"""Run the frozen B2 repository suites and publish implementation-bound logs."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import canonical_bytes, repository_binding  # noqa: E402


REPORT_NAME = "b2-test-report.json"
PYTEST_SUMMARY = re.compile(r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?")
CARGO_SUMMARY = re.compile(
    r"test result: ok\. (?P<passed>\d+) passed; 0 failed; "
    r"(?P<ignored>\d+) ignored"
)
RENAME_NOREPLACE = 1
AT_FDCWD = -100


class B2TestSuiteError(ValueError):
    """Raised when the suite cannot produce honest complete evidence."""


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise B2TestSuiteError("Linux renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise B2TestSuiteError(
            "test evidence destination appeared before publication"
        )
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise B2TestSuiteError(
            "filesystem lacks renameat2(RENAME_NOREPLACE) authority"
        )
    raise OSError(error, os.strerror(error), str(destination))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _parse_counts(name: str, payload: bytes, exit_code: int) -> tuple[int, int, int]:
    text = payload.decode("utf-8", errors="replace")
    if name == "python":
        matches = list(PYTEST_SUMMARY.finditer(text))
        if len(matches) != 1:
            raise B2TestSuiteError("pytest log lacks one unambiguous pass summary")
        return (
            int(matches[0].group("passed")),
            int(matches[0].group("skipped") or 0),
            0,
        )
    if name in {"rust-tests", "dyn-tests"}:
        matches = list(CARGO_SUMMARY.finditer(text))
        if not matches:
            raise B2TestSuiteError(f"{name} log lacks Cargo test summaries")
        return (
            sum(int(match.group("passed")) for match in matches),
            0,
            sum(int(match.group("ignored")) for match in matches),
        )
    if exit_code != 0:
        return 0, 0, 0
    return 1, 0, 0


def _commands(python: str) -> tuple[tuple[str, list[str]], ...]:
    dyn_manifest = "tools/q2-dyn-evidence/Cargo.toml"
    return (
        ("python", [python, "-m", "pytest", "-q"]),
        ("rust-fmt", ["cargo", "fmt", "--all", "--", "--check"]),
        ("rust-clippy", ["cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings"]),
        ("rust-tests", ["cargo", "test", "--locked", "--all-targets"]),
        ("dyn-fmt", ["cargo", "fmt", "--manifest-path", dyn_manifest, "--", "--check"]),
        ("dyn-clippy", ["cargo", "clippy", "--locked", "--manifest-path", dyn_manifest, "--all-targets", "--", "-D", "warnings"]),
        ("dyn-tests", ["cargo", "test", "--locked", "--manifest-path", dyn_manifest]),
    )


def run_suite(output: Path, *, python: str = sys.executable) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise B2TestSuiteError("test evidence output already exists")
    if not output.parent.is_dir():
        raise B2TestSuiteError("test evidence parent is missing")
    if _path_is_within(output, ROOT):
        raise B2TestSuiteError(
            "test evidence output must be outside the implementation repository"
        )
    binding = repository_binding(ROOT)
    if binding.get("git_clean") is not True:
        raise B2TestSuiteError("implementation repository is not clean")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    published = False
    try:
        runs = []
        failures = []
        for name, command in _commands(python):
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            payload = completed.stdout
            log_path = stage / f"{name}.log"
            _write_new(log_path, payload)
            passed, skipped, ignored = _parse_counts(
                name, payload, completed.returncode
            )
            if completed.returncode != 0:
                failures.append(f"{name}: exit {completed.returncode}")
            runs.append({
                "name": name,
                "command": command,
                "exit_code": completed.returncode,
                "passed_count": passed,
                "skipped_count": skipped,
                "ignored_count": ignored,
                "log": {
                    "path": str((output / log_path.name).resolve()),
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                },
            })
        if repository_binding(ROOT) != binding:
            raise B2TestSuiteError(
                "implementation repository changed while tests were running"
            )
        report = {
            "schema": "q2-b2-test-report-v1",
            "implementation": binding,
            "runs": runs,
            "failures": failures,
            "passed": not failures,
        }
        _write_new(stage / REPORT_NAME, canonical_bytes(report))
        _fsync_directory(stage)
        _rename_noreplace(stage, output)
        _fsync_directory(output.parent)
        published = True
        return report
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        report = run_suite(args.output, python=args.python)
    except (B2TestSuiteError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"B2 test suite failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
