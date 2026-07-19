#!/usr/bin/env python3
"""Bind admitted promoted Atlas origin into the sole executable Dyn argv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.preflight_b2_dyn_invocation import (  # noqa: E402
    EXPECTED_STDOUT,
    DynInvocationPreflightError,
    argv_flag,
    load_shape_preflight,
)
from harness.atlas_analyzer import _snapped_origin  # noqa: E402
from tools.retired_cohort_registry import (  # noqa: E402
    RetiredCohortRegistryError,
    require_unretired_identity,
)
from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    canonical_bytes,
    repository_binding,
)


SCHEMA = "q2-b2-dyn-origin-binding-v1"
ARTIFACT_PREFLIGHT_SCHEMA = "q2-b2-dyn-artifact-preflight-v1"
ARTIFACT_PREFLIGHT_STDOUT = canonical_bytes(
    {"passed": True, "schema": ARTIFACT_PREFLIGHT_SCHEMA}
)
VERSIONED_DECLARATION_NAME = re.compile(
    r"^B2-GENERATED-COHORT-([0-9]+)-DECLARATION\.json$"
)


class DynOriginBindingError(ValueError):
    """Raised before Dyn execution when promoted artifacts do not bind exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DynOriginBindingError(f"Dyn artifact must be a regular file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DynOriginBindingError(f"duplicate JSON key in Dyn artifact: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise DynOriginBindingError(
        f"non-finite JSON token in Dyn artifact: {token}"
    )


def _load_json_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DynOriginBindingError(f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except DynOriginBindingError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DynOriginBindingError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DynOriginBindingError(f"{label} must be a JSON object")
    return raw, value


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw, value = _load_json_object(path, label)
    if raw != canonical_bytes(value):
        raise DynOriginBindingError(f"{label} is not canonical JSON")
    return value


def _load_atlas_compact(path: Path) -> dict[str, Any]:
    """Load exactly the compact JSON-plus-LF form admitted by promotion."""

    raw, value = _load_json_object(path, "Atlas manifest")
    compact = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n"
    ).encode("ascii")
    if raw != compact:
        raise DynOriginBindingError(
            "Atlas manifest is not canonical compact JSON"
        )
    return value


def require_versioned_declaration_path(
    repo_root: Path,
    declaration_path: Path,
    declaration: dict[str, Any],
) -> None:
    """Require the exact immutable declaration leaf, never the current alias."""

    if not declaration_path.is_absolute():
        raise DynOriginBindingError("Dyn declaration path must be absolute")
    if declaration_path.is_symlink() or not declaration_path.is_file():
        raise DynOriginBindingError("Dyn declaration must be a regular file")
    try:
        resolved = declaration_path.resolve(strict=True)
        expected_parent = (repo_root / "docs/multires").resolve(strict=True)
    except OSError as exc:
        raise DynOriginBindingError(
            f"Dyn declaration path cannot be resolved: {exc}"
        ) from exc
    if resolved != declaration_path or resolved.parent != expected_parent:
        raise DynOriginBindingError(
            "Dyn declaration is not the direct immutable repository path"
        )
    matched = VERSIONED_DECLARATION_NAME.fullmatch(declaration_path.name)
    if matched is None:
        raise DynOriginBindingError(
            "Dyn declaration path is not a versioned immutable declaration"
        )
    cohort_id = declaration.get("cohort_id")
    if (
        not isinstance(cohort_id, str)
        or not cohort_id.endswith(f"_{matched.group(1)}")
    ):
        raise DynOriginBindingError(
            "Dyn declaration path number differs from cohort identity"
        )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DynOriginBindingError(f"{label} must be an object")
    return value


def _origin(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(axis, int) or isinstance(axis, bool) for axis in value)
    ):
        raise DynOriginBindingError(f"{label} must be three integer axes")
    if any(axis % 256 != 0 for axis in value):
        raise DynOriginBindingError(f"{label} is not snapped to the 256-unit grid")
    return list(value)


def _insert_origin(argv_without_origin: list[str], token: str) -> list[str]:
    if any(value.startswith("--expected-origin") for value in argv_without_origin):
        raise DynOriginBindingError("pre-source argv already contains an origin")
    ordinal = argv_without_origin.index("--expected-analyzer-authority")
    return [
        *argv_without_origin[:ordinal],
        "--expected-origin",
        token,
        *argv_without_origin[ordinal:],
    ]


def bind_origin(
    shape_path: Path,
    promotion_path: Path,
    declaration_path: Path,
    report_path: Path,
) -> dict[str, object]:
    if not shape_path.is_absolute():
        raise DynOriginBindingError("Dyn shape-preflight path must be absolute")
    if not promotion_path.is_absolute():
        raise DynOriginBindingError("generated-promotion report path must be absolute")
    shape = load_shape_preflight(shape_path)
    argv_without_origin = list(shape["producer_argv_without_origin"])
    repo_root = Path(argv_flag(argv_without_origin, "--repo-root"))
    try:
        shape_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise DynOriginBindingError("Dyn shape preflight is inside the repository")
    if not report_path.is_absolute():
        raise DynOriginBindingError("Dyn origin-binding report must be absolute")
    try:
        report_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise DynOriginBindingError("Dyn origin-binding report is inside the repository")
    if report_path.exists() or report_path.is_symlink():
        raise DynOriginBindingError("Dyn origin-binding report already exists")
    if not report_path.parent.is_dir():
        raise DynOriginBindingError("Dyn origin-binding report parent is absent")

    output = Path(argv_flag(argv_without_origin, "--output"))
    expected_promotion_path = output.parent / "reports/generated-promotion.json"
    if promotion_path != expected_promotion_path:
        raise DynOriginBindingError(
            "generated-promotion report path differs from the final workspace"
        )
    if output.exists() or output.is_symlink():
        raise DynOriginBindingError("planned Dyn output already exists")
    binding_before = repository_binding(repo_root)
    if binding_before != shape["repository"] or not binding_before.get("git_clean"):
        raise DynOriginBindingError("Dyn repository binding differs from Phase A")

    executable = Path(argv_without_origin[0])
    executable_record = _file_record(executable)
    if executable_record != shape["executable"]:
        raise DynOriginBindingError("Dyn executable bytes differ from Phase A")

    declaration = _load_canonical(declaration_path, "active cohort declaration")
    require_versioned_declaration_path(repo_root, declaration_path, declaration)
    declaration_maps = declaration.get("maps")
    if (
        not isinstance(declaration.get("cohort_id"), str)
        or not isinstance(declaration_maps, list)
        or len(declaration_maps) != 28
        or any(not isinstance(row, dict) for row in declaration_maps)
    ):
        raise DynOriginBindingError("active cohort declaration shape differs")
    try:
        require_unretired_identity(
            declaration["cohort_id"], _sha256(declaration_path)
        )
    except RetiredCohortRegistryError as exc:
        raise DynOriginBindingError(
            f"active cohort declaration is permanently retired: {exc}"
        ) from exc

    atlas_path = Path(argv_flag(argv_without_origin, "--atlas"))
    manifest_path = Path(argv_flag(argv_without_origin, "--manifest"))
    bsp_path = Path(argv_flag(argv_without_origin, "--bsp"))
    map_id = argv_flag(argv_without_origin, "--expected-map-id")
    expected_analysis_name = f"{map_id}.analysis.manifest.json"
    analysis_path = manifest_path.parent / expected_analysis_name

    manifest = _load_atlas_compact(manifest_path)
    analysis = _load_canonical(analysis_path, "analysis manifest")
    promotion = _load_canonical(promotion_path, "generated promotion report")
    manifest_grid = _mapping(manifest.get("grid"), "Atlas manifest grid")
    origin = _origin(manifest_grid.get("origin"), "Atlas manifest origin")
    model0_mins = manifest_grid.get("model0_mins")
    if (
        not isinstance(model0_mins, list)
        or len(model0_mins) != 3
        or any(
            not isinstance(axis, int) or isinstance(axis, bool)
            for axis in model0_mins
        )
    ):
        raise DynOriginBindingError("Atlas manifest model0_mins differ")
    snapped = list(_snapped_origin(model0_mins))
    if origin != snapped:
        raise DynOriginBindingError(
            "Atlas manifest origin does not equal snapped model0_mins"
        )

    analysis_grid = _mapping(analysis.get("grid"), "analysis manifest grid")
    if _origin(analysis_grid.get("origin"), "analysis manifest origin") != origin:
        raise DynOriginBindingError("analysis and Atlas manifest origins differ")
    manifest_bsp = _mapping(manifest.get("bsp"), "Atlas manifest BSP")
    if (
        manifest_bsp.get("canonical_map_id") != map_id
        or analysis.get("canonical_map_id") != map_id
    ):
        raise DynOriginBindingError("promoted artifact map identity differs from Phase A")

    artifacts = {
        "atlas": _file_record(atlas_path),
        "atlas_manifest": _file_record(manifest_path),
        "analysis_manifest": _file_record(analysis_path),
        "bsp": _file_record(bsp_path),
    }
    with atlas_path.open("rb") as stream:
        atlas_header = stream.read(40)
    if len(atlas_header) != 40 or atlas_header[:8] != b"Q2ATL001":
        raise DynOriginBindingError("raw Atlas header differs")
    schema, byte_order, header_bytes, *atlas_origin = struct.unpack(
        "<HHIqqq", atlas_header[8:]
    )
    if (schema, byte_order, header_bytes) != (1, 0x454C, 136):
        raise DynOriginBindingError("raw Atlas schema or byte order differs")
    if atlas_origin != origin:
        raise DynOriginBindingError("raw Atlas and canonical manifest origins differ")
    atlas_manifest_artifacts = _mapping(
        manifest.get("artifacts"), "Atlas manifest artifacts"
    )
    atlas_entry = _mapping(
        atlas_manifest_artifacts.get(atlas_path.name), "Atlas artifact entry"
    )
    analysis_identity = _mapping(analysis.get("identity"), "analysis identity")
    analysis_atlas_manifest = _mapping(
        _mapping(analysis.get("artifacts"), "analysis artifacts").get(
            "atlas_manifest"
        ),
        "analysis Atlas manifest",
    )
    manifest_analyzer = _mapping(manifest.get("analyzer"), "Atlas analyzer")
    expected_analyzer = argv_flag(
        argv_without_origin, "--expected-analyzer-authority"
    )
    if (
        atlas_entry.get("sha256_uncompressed") != artifacts["atlas"]["sha256"]
        or atlas_entry.get("uncompressed_size") != artifacts["atlas"]["size_bytes"]
        or manifest_bsp.get("sha256") != artifacts["bsp"]["sha256"]
        or manifest_bsp.get("size_bytes") != artifacts["bsp"]["size_bytes"]
        or analysis_identity.get("atlas_sha256") != artifacts["atlas"]["sha256"]
        or analysis_identity.get("bsp_sha256") != artifacts["bsp"]["sha256"]
        or analysis_identity.get("atlas_manifest_sha256")
        != artifacts["atlas_manifest"]["sha256"]
        or analysis_atlas_manifest.get("sha256")
        != artifacts["atlas_manifest"]["sha256"]
        or manifest_analyzer.get("sha256") != expected_analyzer
        or analysis_identity.get("analyzer_sha256") != expected_analyzer
    ):
        raise DynOriginBindingError("promoted artifact digest authority differs")

    promotion_maps = promotion.get("maps")
    if (
        promotion.get("schema") != "q2-generator-claim-campaign-v2"
        or promotion.get("phase") != "compiled_validation"
        or promotion.get("passed") is not True
        or promotion.get("cohort_id") != declaration["cohort_id"]
        or promotion.get("declaration_sha256") != _sha256(declaration_path)
        or promotion.get("expected_count") != 28
        or promotion.get("map_count") != 28
        or promotion.get("pass_count") != 28
        or promotion.get("failures") != []
        or not isinstance(promotion_maps, list)
        or len(promotion_maps) != 28
    ):
        raise DynOriginBindingError("generated promotion report is not green")
    declared_rows = [
        row for row in declaration_maps if row.get("map") == map_id
    ]
    if len(declared_rows) != 1:
        raise DynOriginBindingError(
            "representative Dyn map is not unique in the active declaration"
        )
    promoted_rows = [
        row
        for row in promotion_maps
        if isinstance(row, dict) and row.get("map") == map_id
    ]
    if (
        len(promoted_rows) != 1
        or promoted_rows[0].get("passed") is not True
        or promoted_rows[0].get("failures") != []
        or promoted_rows[0].get("atlas_sha256")
        != artifacts["atlas"]["sha256"]
        or promoted_rows[0].get("bsp_sha256") != artifacts["bsp"]["sha256"]
        or promoted_rows[0].get("atlas_manifest_sha256")
        != artifacts["atlas_manifest"]["sha256"]
        or promoted_rows[0].get("analysis_manifest_sha256")
        != artifacts["analysis_manifest"]["sha256"]
    ):
        raise DynOriginBindingError(
            "representative Dyn artifacts are not admitted by promotion"
        )

    origin_token = ",".join(str(axis) for axis in origin)
    producer_argv = _insert_origin(argv_without_origin, origin_token)
    parser_preflight_argv = [*producer_argv, "--preflight-only", "true"]
    completed = subprocess.run(
        parser_preflight_argv,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise DynOriginBindingError(
            "Dyn parser refused artifact-bound argv with exit "
            f"{completed.returncode}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    if completed.stdout != EXPECTED_STDOUT or completed.stderr != b"":
        raise DynOriginBindingError("Dyn artifact-bound parser preflight output differs")
    artifact_preflight_argv = [
        *producer_argv,
        "--verify-artifacts-only",
        "true",
    ]
    artifact_completed = subprocess.run(
        artifact_preflight_argv,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if artifact_completed.returncode != 0:
        raise DynOriginBindingError(
            "Dyn artifact verifier refused the admitted artifacts with exit "
            f"{artifact_completed.returncode}: "
            f"{artifact_completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    if (
        artifact_completed.stdout != ARTIFACT_PREFLIGHT_STDOUT
        or artifact_completed.stderr != b""
    ):
        raise DynOriginBindingError("Dyn artifact verifier output differs")
    if output.exists() or output.is_symlink():
        raise DynOriginBindingError("Dyn artifact binding touched producer output")
    if repository_binding(repo_root) != binding_before:
        raise DynOriginBindingError("repository changed during Dyn origin binding")

    return {
        "schema": SCHEMA,
        "passed": True,
        "shape_preflight": _file_record(shape_path),
        "declaration": _file_record(declaration_path),
        "promotion": _file_record(promotion_path),
        "repository": binding_before,
        "executable": executable_record,
        "artifacts": artifacts,
        "identity": {
            "canonical_map_id": map_id,
            "origin": origin,
            "origin_token": origin_token,
            "analyzer_authority_sha256": expected_analyzer,
            "atlas_sha256": artifacts["atlas"]["sha256"],
            "atlas_manifest_sha256": artifacts["atlas_manifest"]["sha256"],
            "analysis_manifest_sha256": artifacts["analysis_manifest"]["sha256"],
            "bsp_sha256": artifacts["bsp"]["sha256"],
        },
        "producer_argv": producer_argv,
        "parser_preflight_argv": parser_preflight_argv,
        "artifact_preflight_argv": artifact_preflight_argv,
        "producer_output_absent_before": True,
        "producer_output_absent_after": True,
        "parser_preflight_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "parser_preflight_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "artifact_preflight_stdout_sha256": hashlib.sha256(
            artifact_completed.stdout
        ).hexdigest(),
        "artifact_preflight_stderr_sha256": hashlib.sha256(
            artifact_completed.stderr
        ).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-preflight-report", type=Path, required=True)
    parser.add_argument("--generated-promotion-report", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = bind_origin(
            args.shape_preflight_report,
            args.generated_promotion_report,
            args.declaration,
            args.report,
        )
        payload = canonical_bytes(report)
        descriptor = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        sys.stdout.buffer.write(payload)
        return 0
    except (
        DynOriginBindingError,
        DynInvocationPreflightError,
        GeneratorCohortError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Dyn origin binding refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
