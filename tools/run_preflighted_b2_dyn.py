#!/usr/bin/env python3
"""Execute the sole B2 Dyn producer from a retained exact-argv preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.preflight_b2_dyn_invocation import EXPECTED_STDOUT, SCHEMA  # noqa: E402
from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    canonical_bytes,
    repository_binding,
)


class PreflightedDynError(ValueError):
    """Raised before the sole producer starts when its preflight has drifted."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightedDynError(f"duplicate JSON key in Dyn preflight: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flag(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise PreflightedDynError(f"preflighted producer argv must contain one {name}")
    ordinal = argv.index(name)
    if ordinal + 1 >= len(argv):
        raise PreflightedDynError(f"preflighted producer argv lacks the {name} value")
    return argv[ordinal + 1]


def load_preflight(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightedDynError("Dyn argv preflight must be a regular file")
    raw = path.read_bytes()
    try:
        report = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PreflightedDynError(f"invalid Dyn argv preflight JSON: {exc}") from exc
    if not isinstance(report, dict) or raw != canonical_bytes(report):
        raise PreflightedDynError("Dyn argv preflight is not canonical")
    expected_keys = {
        "schema",
        "passed",
        "repository",
        "executable",
        "producer_argv",
        "preflight_argv",
        "producer_output_absent_before",
        "producer_output_absent_after",
        "preflight_stdout_sha256",
        "preflight_stderr_sha256",
    }
    if set(report) != expected_keys or report["schema"] != SCHEMA:
        raise PreflightedDynError("Dyn argv preflight schema or keys differ")
    if report["passed"] is not True:
        raise PreflightedDynError("Dyn argv preflight is not green")
    if (
        report["producer_output_absent_before"] is not True
        or report["producer_output_absent_after"] is not True
    ):
        raise PreflightedDynError("Dyn argv preflight did not preserve output absence")
    producer_argv = report["producer_argv"]
    preflight_argv = report["preflight_argv"]
    if (
        not isinstance(producer_argv, list)
        or not producer_argv
        or any(not isinstance(value, str) for value in producer_argv)
        or preflight_argv != [*producer_argv, "--preflight-only", "true"]
    ):
        raise PreflightedDynError("Dyn preflight and producer argv differ")
    if any(value.startswith("--expected-origin=") for value in producer_argv):
        raise PreflightedDynError("equals-glued Dyn origin is forbidden")
    _flag(producer_argv, "--expected-origin")
    path_values = [producer_argv[0]] + [
        _flag(producer_argv, name)
        for name in ("--repo-root", "--atlas", "--manifest", "--bsp", "--output")
    ]
    if any(not Path(value).is_absolute() for value in path_values):
        raise PreflightedDynError("preflighted Dyn paths must all be absolute")
    if report["preflight_stdout_sha256"] != hashlib.sha256(EXPECTED_STDOUT).hexdigest():
        raise PreflightedDynError("Dyn argv preflight stdout digest differs")
    if report["preflight_stderr_sha256"] != hashlib.sha256(b"").hexdigest():
        raise PreflightedDynError("Dyn argv preflight stderr digest differs")
    return report


def execute(report: dict[str, Any]) -> Path:
    producer_argv = list(report["producer_argv"])
    executable = Path(producer_argv[0])
    executable_record = report["executable"]
    if (
        not isinstance(executable_record, dict)
        or executable_record.get("path") != str(executable)
        or executable.is_symlink()
        or not executable.is_file()
        or executable_record.get("sha256") != _sha256(executable)
        or executable_record.get("size_bytes") != executable.stat().st_size
    ):
        raise PreflightedDynError("preflighted Dyn executable bytes differ")
    repo_root = Path(_flag(producer_argv, "--repo-root"))
    if repository_binding(repo_root) != report["repository"]:
        raise PreflightedDynError("preflighted Dyn repository binding differs")
    output = Path(_flag(producer_argv, "--output"))
    if output.exists() or output.is_symlink():
        raise PreflightedDynError("preflighted Dyn output already exists")

    completed = subprocess.run(
        producer_argv,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr)
        raise PreflightedDynError(
            f"sole preflighted Dyn producer failed with exit {completed.returncode}"
        )
    expected_report = output / "b2-dyn-evidence.json"
    if completed.stdout != (str(expected_report) + "\n").encode() or not expected_report.is_file():
        raise PreflightedDynError("preflighted Dyn producer publication differs")
    sys.stderr.buffer.write(completed.stderr)
    return expected_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report_path = execute(load_preflight(args.preflight_report))
        print(report_path)
        return 0
    except (
        PreflightedDynError,
        GeneratorCohortError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"preflighted B2 Dyn refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
