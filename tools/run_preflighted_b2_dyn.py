#!/usr/bin/env python3
"""Execute sole B2 Dyn from pre-source shape and artifact-origin authorities."""

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

from tools.bind_b2_dyn_origin import (  # noqa: E402
    ARTIFACT_PREFLIGHT_STDOUT,
    SCHEMA as ORIGIN_BINDING_SCHEMA,
    DynOriginBindingError,
    _load_canonical,
    require_versioned_declaration_path,
)
from tools.preflight_b2_dyn_invocation import (  # noqa: E402
    EXPECTED_STDOUT,
    DynInvocationPreflightError,
    argv_flag,
    load_shape_preflight,
)
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


def load_origin_binding(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightedDynError("Dyn origin binding must be a regular file")
    raw = path.read_bytes()
    try:
        report = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PreflightedDynError(f"invalid Dyn origin-binding JSON: {exc}") from exc
    if not isinstance(report, dict) or raw != canonical_bytes(report):
        raise PreflightedDynError("Dyn origin binding is not canonical")
    expected_keys = {
        "schema",
        "passed",
        "shape_preflight",
        "declaration",
        "promotion",
        "repository",
        "executable",
        "artifacts",
        "identity",
        "producer_argv",
        "parser_preflight_argv",
        "artifact_preflight_argv",
        "producer_output_absent_before",
        "producer_output_absent_after",
        "parser_preflight_stdout_sha256",
        "parser_preflight_stderr_sha256",
        "artifact_preflight_stdout_sha256",
        "artifact_preflight_stderr_sha256",
    }
    if set(report) != expected_keys or report["schema"] != ORIGIN_BINDING_SCHEMA:
        raise PreflightedDynError("Dyn origin-binding schema or keys differ")
    if report["passed"] is not True:
        raise PreflightedDynError("Dyn origin binding is not green")
    if (
        report["producer_output_absent_before"] is not True
        or report["producer_output_absent_after"] is not True
    ):
        raise PreflightedDynError("Dyn origin binding did not preserve output absence")
    producer_argv = report["producer_argv"]
    parser_preflight_argv = report["parser_preflight_argv"]
    artifact_preflight_argv = report["artifact_preflight_argv"]
    if (
        not isinstance(producer_argv, list)
        or not producer_argv
        or any(not isinstance(value, str) for value in producer_argv)
        or parser_preflight_argv
        != [*producer_argv, "--preflight-only", "true"]
        or artifact_preflight_argv
        != [*producer_argv, "--verify-artifacts-only", "true"]
    ):
        raise PreflightedDynError("Dyn origin-bound parser and producer argv differ")
    if any(value.startswith("--expected-origin=") for value in producer_argv):
        raise PreflightedDynError("equals-glued Dyn origin is forbidden")
    argv_flag(producer_argv, "--expected-origin")
    path_values = [producer_argv[0]] + [
        argv_flag(producer_argv, name)
        for name in ("--repo-root", "--atlas", "--manifest", "--bsp", "--output")
    ]
    if any(not Path(value).is_absolute() for value in path_values):
        raise PreflightedDynError("origin-bound Dyn paths must all be absolute")
    if report["parser_preflight_stdout_sha256"] != hashlib.sha256(EXPECTED_STDOUT).hexdigest():
        raise PreflightedDynError("Dyn origin-binding stdout digest differs")
    if report["parser_preflight_stderr_sha256"] != hashlib.sha256(b"").hexdigest():
        raise PreflightedDynError("Dyn origin-binding stderr digest differs")
    if report["artifact_preflight_stdout_sha256"] != hashlib.sha256(
        ARTIFACT_PREFLIGHT_STDOUT
    ).hexdigest():
        raise PreflightedDynError("Dyn artifact-preflight stdout digest differs")
    if report["artifact_preflight_stderr_sha256"] != hashlib.sha256(b"").hexdigest():
        raise PreflightedDynError("Dyn artifact-preflight stderr digest differs")
    return report


# Compatibility name for callers that explicitly load the executable Phase-B
# authority. Phase A is intentionally available only as load_shape_preflight.
load_preflight = load_origin_binding


def _record_matches(record: object, path: Path) -> bool:
    return (
        isinstance(record, dict)
        and record.get("path") == str(path)
        and not path.is_symlink()
        and path.is_file()
        and record.get("sha256") == _sha256(path)
        and record.get("size_bytes") == path.stat().st_size
    )


def _bound_argv(shape_argv: list[str], origin_token: str) -> list[str]:
    if any(value.startswith("--expected-origin") for value in shape_argv):
        raise PreflightedDynError("pre-source Dyn argv contains an origin")
    ordinal = shape_argv.index("--expected-analyzer-authority")
    return [
        *shape_argv[:ordinal],
        "--expected-origin",
        origin_token,
        *shape_argv[ordinal:],
    ]


def execute(
    shape: dict[str, Any],
    binding: dict[str, Any],
    declaration_path: Path,
) -> Path:
    shape_argv = list(shape["producer_argv_without_origin"])
    producer_argv = list(binding["producer_argv"])
    identity = binding.get("identity")
    if not isinstance(identity, dict):
        raise PreflightedDynError("Dyn origin-binding identity differs")
    origin = identity.get("origin")
    origin_token = identity.get("origin_token")
    if (
        not isinstance(origin, list)
        or len(origin) != 3
        or any(not isinstance(axis, int) or isinstance(axis, bool) for axis in origin)
        or not isinstance(origin_token, str)
        or origin_token != ",".join(str(axis) for axis in origin)
        or producer_argv != _bound_argv(shape_argv, origin_token)
    ):
        raise PreflightedDynError("Dyn complete argv was not derived from both authorities")

    shape_record = binding.get("shape_preflight")
    shape_path = Path(shape_record.get("path", "")) if isinstance(shape_record, dict) else Path()
    if not _record_matches(shape_record, shape_path):
        raise PreflightedDynError("Dyn Phase-A report bytes differ")
    declaration_record = binding.get("declaration")
    bound_declaration_path = (
        Path(declaration_record.get("path", ""))
        if isinstance(declaration_record, dict)
        else Path()
    )
    if bound_declaration_path != declaration_path:
        raise PreflightedDynError("Dyn active-declaration path differs")
    if not _record_matches(declaration_record, declaration_path):
        raise PreflightedDynError("Dyn active-declaration bytes differ")
    promotion_record = binding.get("promotion")
    promotion_path = (
        Path(promotion_record.get("path", ""))
        if isinstance(promotion_record, dict)
        else Path()
    )
    if not _record_matches(promotion_record, promotion_path):
        raise PreflightedDynError("Dyn generated-promotion bytes differ")
    executable = Path(producer_argv[0])
    if (
        not _record_matches(binding.get("executable"), executable)
        or binding.get("executable") != shape.get("executable")
    ):
        raise PreflightedDynError("origin-bound Dyn executable bytes differ")
    repo_root = Path(argv_flag(producer_argv, "--repo-root"))
    if (
        binding.get("repository") != shape.get("repository")
        or repository_binding(repo_root) != binding.get("repository")
    ):
        raise PreflightedDynError("origin-bound Dyn repository binding differs")
    try:
        declaration = _load_canonical(
            declaration_path, "active cohort declaration"
        )
        require_versioned_declaration_path(
            repo_root, declaration_path, declaration
        )
    except DynOriginBindingError as exc:
        raise PreflightedDynError(str(exc)) from exc
    output = Path(argv_flag(producer_argv, "--output"))
    if promotion_path != output.parent / "reports/generated-promotion.json":
        raise PreflightedDynError("Dyn generated-promotion path differs")

    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "atlas", "atlas_manifest", "analysis_manifest", "bsp"
    }:
        raise PreflightedDynError("Dyn origin-binding artifact records differ")
    map_id = argv_flag(producer_argv, "--expected-map-id")
    expected_paths = {
        "atlas": Path(argv_flag(producer_argv, "--atlas")),
        "atlas_manifest": Path(argv_flag(producer_argv, "--manifest")),
        "analysis_manifest": Path(argv_flag(producer_argv, "--manifest")).parent
        / f"{map_id}.analysis.manifest.json",
        "bsp": Path(argv_flag(producer_argv, "--bsp")),
    }
    if any(
        not _record_matches(artifacts[name], path)
        for name, path in expected_paths.items()
    ):
        raise PreflightedDynError("origin-bound Dyn artifact bytes differ")
    if (
        identity.get("canonical_map_id") != map_id
        or identity.get("analyzer_authority_sha256")
        != argv_flag(producer_argv, "--expected-analyzer-authority")
        or identity.get("atlas_sha256") != artifacts["atlas"].get("sha256")
        or identity.get("atlas_manifest_sha256")
        != artifacts["atlas_manifest"].get("sha256")
        or identity.get("analysis_manifest_sha256")
        != artifacts["analysis_manifest"].get("sha256")
        or identity.get("bsp_sha256") != artifacts["bsp"].get("sha256")
    ):
        raise PreflightedDynError("Dyn origin-binding identity does not match artifacts")

    if output.exists() or output.is_symlink():
        raise PreflightedDynError("origin-bound Dyn output already exists")

    artifact_preflight = subprocess.run(
        binding["artifact_preflight_argv"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        artifact_preflight.returncode != 0
        or artifact_preflight.stdout != ARTIFACT_PREFLIGHT_STDOUT
        or artifact_preflight.stderr != b""
    ):
        raise PreflightedDynError(
            "final no-write Dyn artifact verification differs"
        )
    if output.exists() or output.is_symlink():
        raise PreflightedDynError(
            "final no-write Dyn artifact verification touched output"
        )
    if repository_binding(repo_root) != binding.get("repository"):
        raise PreflightedDynError(
            "repository changed during final no-write Dyn verification"
        )

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
    parser.add_argument("--shape-preflight-report", type=Path, required=True)
    parser.add_argument("--origin-binding-report", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        shape = load_shape_preflight(args.shape_preflight_report)
        binding = load_origin_binding(args.origin_binding_report)
        if binding.get("shape_preflight", {}).get("path") != str(
            args.shape_preflight_report
        ):
            raise PreflightedDynError(
                "Dyn origin binding names a different Phase-A report"
            )
        report_path = execute(shape, binding, args.declaration)
        print(report_path)
        return 0
    except (
        PreflightedDynError,
        DynInvocationPreflightError,
        GeneratorCohortError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"preflighted B2 Dyn refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
