#!/usr/bin/env python3
"""Attest the exact no-write argv for the sole final-cohort Dyn producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import canonical_bytes, repository_binding  # noqa: E402


SCHEMA = "q2-b2-dyn-argv-preflight-v1"
EXPECTED_STDOUT = canonical_bytes({"passed": True, "schema": SCHEMA})


class DynInvocationPreflightError(ValueError):
    """Raised before publication when the planned Dyn invocation is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _origin(value: str) -> str:
    fields = value.split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("origin must be X,Y,Z")
    try:
        parsed = [int(field) for field in fields]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("origin fields must be integers") from exc
    return ",".join(str(field) for field in parsed)


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DynInvocationPreflightError(f"{label} must be a regular file: {path}")


def _abs(path: Path) -> str:
    if not path.is_absolute():
        raise DynInvocationPreflightError(f"planned Dyn path must be absolute: {path}")
    return str(path)


def _producer_argv(args: argparse.Namespace) -> list[str]:
    return [
        _abs(args.executable),
        "--repo-root",
        _abs(args.repo_root),
        "--atlas",
        _abs(args.atlas),
        "--manifest",
        _abs(args.manifest),
        "--bsp",
        _abs(args.bsp),
        "--expected-map-id",
        args.expected_map_id,
        "--expected-origin",
        args.expected_origin,
        "--expected-analyzer-authority",
        args.expected_analyzer_authority,
        "--expected-crate-commit",
        args.expected_crate_commit,
        "--map-epoch",
        str(args.map_epoch),
        "--environment-steps",
        str(args.environment_steps),
        "--samples",
        str(args.samples),
        "--output",
        _abs(args.output),
    ]


def preflight(args: argparse.Namespace) -> dict[str, object]:
    _regular_file(args.executable, "Dyn evidence executable")
    if args.output.exists() or args.output.is_symlink():
        raise DynInvocationPreflightError("planned Dyn output already exists")
    if args.report.exists() or args.report.is_symlink():
        raise DynInvocationPreflightError("Dyn argv preflight report already exists")
    if not args.report.parent.is_dir():
        raise DynInvocationPreflightError("Dyn argv preflight report parent is absent")
    binding_before = repository_binding(args.repo_root)
    if not binding_before.get("git_clean"):
        raise DynInvocationPreflightError("Dyn argv preflight repository is not clean")
    if binding_before.get("repository_commit") != args.expected_crate_commit:
        raise DynInvocationPreflightError("planned Dyn commit differs from repository")

    producer_argv = _producer_argv(args)
    preflight_argv = [*producer_argv, "--preflight-only", "true"]
    completed = subprocess.run(
        preflight_argv,
        cwd=args.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise DynInvocationPreflightError(
            f"Dyn argv parser refused the planned invocation with exit {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    if completed.stdout != EXPECTED_STDOUT or completed.stderr != b"":
        raise DynInvocationPreflightError("Dyn argv preflight output differs")
    if args.output.exists() or args.output.is_symlink():
        raise DynInvocationPreflightError("Dyn argv preflight touched the producer output")
    binding_after = repository_binding(args.repo_root)
    if binding_after != binding_before:
        raise DynInvocationPreflightError("repository changed during Dyn argv preflight")

    return {
        "schema": SCHEMA,
        "passed": True,
        "repository": binding_before,
        "executable": {
            "path": str(args.executable),
            "sha256": _sha256(args.executable),
            "size_bytes": args.executable.stat().st_size,
        },
        "producer_argv": producer_argv,
        "preflight_argv": preflight_argv,
        "producer_output_absent_before": True,
        "producer_output_absent_after": True,
        "preflight_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "preflight_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bsp", type=Path, required=True)
    parser.add_argument("--expected-map-id", required=True)
    parser.add_argument("--expected-origin", type=_origin, required=True)
    parser.add_argument("--expected-analyzer-authority", required=True)
    parser.add_argument("--expected-crate-commit", required=True)
    parser.add_argument("--map-epoch", type=int, required=True)
    parser.add_argument("--environment-steps", type=int, required=True)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.map_epoch <= 0 or args.environment_steps < 0 or args.samples < 2000:
            raise DynInvocationPreflightError("planned Dyn numeric domain differs")
        report = preflight(args)
        payload = canonical_bytes(report)
        descriptor = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        sys.stdout.buffer.write(payload)
        return 0
    except (DynInvocationPreflightError, OSError, subprocess.SubprocessError) as exc:
        print(f"Dyn argv preflight refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
