#!/usr/bin/env python3
"""Attest pre-source Dyn argv shape while deferring artifact-derived origin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    canonical_bytes,
    repository_binding,
)


SCHEMA = "q2-b2-dyn-argv-shape-preflight-v2"
EXPECTED_STDOUT = canonical_bytes({"passed": True, "schema": SCHEMA})


class DynInvocationPreflightError(ValueError):
    """Raised before publication when the planned Dyn invocation is unsafe."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DynInvocationPreflightError(
                f"duplicate JSON key in Dyn shape preflight: {key}"
            )
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DynInvocationPreflightError(f"{label} must be a regular file: {path}")


def _abs(path: Path) -> str:
    if not path.is_absolute():
        raise DynInvocationPreflightError(f"planned Dyn path must be absolute: {path}")
    return str(path)


def argv_flag(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise DynInvocationPreflightError(
            f"Dyn argv must contain exactly one {name}"
        )
    ordinal = argv.index(name)
    if ordinal + 1 >= len(argv):
        raise DynInvocationPreflightError(f"Dyn argv lacks the {name} value")
    return argv[ordinal + 1]


def load_shape_preflight(path: Path) -> dict[str, Any]:
    """Load the canonical non-executable Phase-A shape authority."""

    _regular_file(path, "Dyn argv shape preflight")
    raw = path.read_bytes()
    try:
        report = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DynInvocationPreflightError(
            f"invalid Dyn argv shape preflight JSON: {exc}"
        ) from exc
    if not isinstance(report, dict) or raw != canonical_bytes(report):
        raise DynInvocationPreflightError(
            "Dyn argv shape preflight is not canonical"
        )
    expected_keys = {
        "schema",
        "passed",
        "repository",
        "executable",
        "origin_binding_status",
        "producer_argv_without_origin",
        "preflight_argv",
        "producer_output_absent_before",
        "producer_output_absent_after",
        "preflight_stdout_sha256",
        "preflight_stderr_sha256",
    }
    if set(report) != expected_keys or report["schema"] != SCHEMA:
        raise DynInvocationPreflightError(
            "Dyn argv shape preflight schema or keys differ"
        )
    if report["passed"] is not True:
        raise DynInvocationPreflightError("Dyn argv shape preflight is not green")
    if report["origin_binding_status"] != "deferred-until-promoted-artifact":
        raise DynInvocationPreflightError("Dyn origin was not explicitly deferred")
    if (
        report["producer_output_absent_before"] is not True
        or report["producer_output_absent_after"] is not True
    ):
        raise DynInvocationPreflightError(
            "Dyn argv shape preflight did not preserve output absence"
        )
    producer_argv = report["producer_argv_without_origin"]
    preflight_argv = report["preflight_argv"]
    if (
        not isinstance(producer_argv, list)
        or not producer_argv
        or any(not isinstance(value, str) for value in producer_argv)
        or preflight_argv
        != [*producer_argv, "--preflight-only", "true"]
    ):
        raise DynInvocationPreflightError(
            "Dyn shape-preflight and origin-free argv differ"
        )
    if any(value.startswith("--expected-origin") for value in producer_argv):
        raise DynInvocationPreflightError(
            "pre-source Dyn argv must not contain a concrete origin"
        )
    flags = (
        "--repo-root",
        "--atlas",
        "--manifest",
        "--bsp",
        "--expected-map-id",
        "--expected-analyzer-authority",
        "--expected-crate-commit",
        "--map-epoch",
        "--environment-steps",
        "--samples",
        "--output",
    )
    if len(producer_argv) != 1 + 2 * len(flags):
        raise DynInvocationPreflightError("Dyn origin-free argv shape differs")
    for flag in flags:
        argv_flag(producer_argv, flag)
    path_values = [producer_argv[0]] + [
        argv_flag(producer_argv, name)
        for name in ("--repo-root", "--atlas", "--manifest", "--bsp", "--output")
    ]
    if any(not Path(value).is_absolute() for value in path_values):
        raise DynInvocationPreflightError(
            "shape-preflighted Dyn paths must all be absolute"
        )
    if report["preflight_stdout_sha256"] != hashlib.sha256(EXPECTED_STDOUT).hexdigest():
        raise DynInvocationPreflightError(
            "Dyn argv shape preflight stdout digest differs"
        )
    if report["preflight_stderr_sha256"] != hashlib.sha256(b"").hexdigest():
        raise DynInvocationPreflightError(
            "Dyn argv shape preflight stderr digest differs"
        )
    return report


def _producer_argv_without_origin(args: argparse.Namespace) -> list[str]:
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
    if args.map_epoch <= 0 or args.environment_steps < 0 or args.samples < 2000:
        raise DynInvocationPreflightError("planned Dyn numeric domain differs")
    if not args.report.is_absolute():
        raise DynInvocationPreflightError("Dyn argv preflight report must be absolute")
    try:
        args.report.resolve().relative_to(args.repo_root.resolve())
    except ValueError:
        pass
    else:
        raise DynInvocationPreflightError(
            "Dyn argv preflight report must be outside the repository"
        )
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

    producer_argv_without_origin = _producer_argv_without_origin(args)
    if any(
        value.startswith("--expected-origin")
        for value in producer_argv_without_origin
    ):
        raise DynInvocationPreflightError(
            "pre-source Dyn origin must remain deferred"
        )
    preflight_argv = [
        *producer_argv_without_origin,
        "--preflight-only",
        "true",
    ]
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
        "origin_binding_status": "deferred-until-promoted-artifact",
        "producer_argv_without_origin": producer_argv_without_origin,
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
        report = preflight(args)
        payload = canonical_bytes(report)
        descriptor = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        sys.stdout.buffer.write(payload)
        return 0
    except (
        DynInvocationPreflightError,
        GeneratorCohortError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Dyn argv preflight refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
