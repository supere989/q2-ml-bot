#!/usr/bin/env python3
"""Fail-closed admission for permanently retired generated cohorts.

Failure authorities are historical records, but their identities are active
admission constraints.  A new producer may not reuse a retired cohort ID,
declaration digest, map ID, or seed.  Read-only declaration and stage
verification intentionally live outside this module and remain available for
forensics.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools.run_generator_cohort import GeneratorCohortError


ROOT = Path(__file__).resolve().parents[1]
FAILURE_REGISTRY_DIR = ROOT / "docs/multires"
FAILURE_GLOB = "B2-GENERATED-COHORT-*-FAILURE.json"
FAILURE_FILE_RE = re.compile(
    r"^B2-GENERATED-COHORT-(?P<number>[0-9]+)-FAILURE\.json$"
)
FAILURE_SCHEMA = "q2-b2-generated-cohort-failure-v1"
FAILURE_STATUS_RE = re.compile(r"^permanently-failed-[a-z0-9-]+$")
TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "cohort_id",
    "declaration",
    "admission",
    "evidence",
    "failure",
    "operator_preflight_attempts",
    "operator_transcript",
}
REQUIRED_TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "cohort_id",
    "declaration",
    "admission",
    "evidence",
    "failure",
}
REQUIRED_FALSE_ADMISSION_FIELDS = {
    "older_population_reuse_allowed",
    "passing_subset_allowed",
    "regeneration_under_same_declaration_allowed",
    "replacement_member_allowed",
    "retry_under_same_declaration_allowed",
    "salvage_allowed",
}
OPTIONAL_FALSE_ADMISSION_FIELDS = {
    "dyn_execution_allowed",
    "release_artifact_copy_allowed",
    "release_artifact_execution_allowed",
    "release_artifact_reuse_allowed",
    "retired_artifact_reuse_allowed",
    "substitution_allowed",
}
PUBLICATION_ADMISSION_FIELDS = {
    "analysis_stage_published",
    "claims_stage_published",
    "compiled_stage_published",
    "materialized_stage_published",
    "source_stage_published",
}
ALLOWED_ADMISSION_FIELDS = (
    {
        "permanently_non_admissible",
        "replacement_declaration_status",
    }
    | REQUIRED_FALSE_ADMISSION_FIELDS
    | OPTIONAL_FALSE_ADMISSION_FIELDS
    | PUBLICATION_ADMISSION_FIELDS
)


class RetiredCohortRegistryError(GeneratorCohortError):
    """Raised when retirement authority is invalid or denies admission."""


def _no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RetiredCohortRegistryError(
                f"retired cohort registry contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetiredCohortRegistryError(f"{label} must be an object")
    return value


def _load_canonical_json(path: Path) -> Mapping[str, Any]:
    from tools.run_generator_cohort import canonical_bytes

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RetiredCohortRegistryError(
                    f"{path.name} contains non-finite JSON token {token}"
                )
            ),
        )
    except RetiredCohortRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetiredCohortRegistryError(
            f"cannot read retired cohort authority {path.name}: {exc}"
        ) from exc
    authority = _mapping(value, f"retired cohort authority {path.name}")
    try:
        canonical = canonical_bytes(authority)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RetiredCohortRegistryError(
            f"retired cohort authority {path.name} is not canonical JSON: {exc}"
        ) from exc
    if raw != canonical:
        raise RetiredCohortRegistryError(
            f"retired cohort authority {path.name} is not canonical JSON"
        )
    return authority


def _require_no_symlink_components(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise RetiredCohortRegistryError(
            f"{label} is outside the canonical repository"
        ) from exc
    current = root.absolute()
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RetiredCohortRegistryError(
                f"{label} contains symlink component {current}"
            )


def _validate_admission(value: object, filename: str) -> None:
    admission = _mapping(value, f"{filename} admission")
    extra = set(admission) - ALLOWED_ADMISSION_FIELDS
    if extra:
        raise RetiredCohortRegistryError(
            f"{filename} admission contains unknown fields: {sorted(extra)}"
        )
    if admission.get("permanently_non_admissible") is not True:
        raise RetiredCohortRegistryError(
            f"{filename} does not assert permanent non-admission"
        )
    missing = REQUIRED_FALSE_ADMISSION_FIELDS - set(admission)
    if missing:
        raise RetiredCohortRegistryError(
            f"{filename} admission omits retirement controls: {sorted(missing)}"
        )
    for field in sorted(
        REQUIRED_FALSE_ADMISSION_FIELDS | OPTIONAL_FALSE_ADMISSION_FIELDS
    ):
        if field in admission and admission[field] is not False:
            raise RetiredCohortRegistryError(
                f"{filename} admission field {field} must be false"
            )
    for field in sorted(PUBLICATION_ADMISSION_FIELDS & set(admission)):
        if not isinstance(admission[field], bool):
            raise RetiredCohortRegistryError(
                f"{filename} admission field {field} must be boolean"
            )
    replacement = admission.get("replacement_declaration_status")
    if not isinstance(replacement, str) or TOKEN_RE.fullmatch(replacement) is None:
        raise RetiredCohortRegistryError(
            f"{filename} replacement declaration status is malformed"
        )


def _load_registry() -> tuple[
    dict[str, str], dict[str, str], dict[str, str], dict[int, str]
]:
    """Load the complete canonical registry before making an admission choice."""

    from tools.run_generator_cohort import GeneratorCohortError, load_declaration

    registry_dir = FAILURE_REGISTRY_DIR
    _require_no_symlink_components(
        registry_dir, ROOT, "retired cohort registry directory"
    )
    if (
        not registry_dir.exists()
        or not registry_dir.is_dir()
        or registry_dir.is_symlink()
    ):
        raise RetiredCohortRegistryError(
            "retired cohort registry directory is missing, invalid, or a symlink"
        )
    paths = sorted(registry_dir.glob(FAILURE_GLOB), key=lambda item: item.name)
    if not paths:
        raise RetiredCohortRegistryError("retired cohort registry is empty")

    cohorts: dict[str, str] = {}
    declarations: dict[str, str] = {}
    maps: dict[str, str] = {}
    seeds: dict[int, str] = {}
    for path in paths:
        match = FAILURE_FILE_RE.fullmatch(path.name)
        if match is None:
            raise RetiredCohortRegistryError(
                f"retired cohort authority filename is malformed: {path.name}"
            )
        if not path.is_file() or path.is_symlink():
            raise RetiredCohortRegistryError(
                f"retired cohort authority is not a regular file: {path.name}"
            )
        authority = _load_canonical_json(path)
        keys = set(authority)
        missing = REQUIRED_TOP_LEVEL_KEYS - keys
        extra = keys - ALLOWED_TOP_LEVEL_KEYS
        if missing or extra:
            raise RetiredCohortRegistryError(
                f"{path.name} authority keys differ; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if authority["schema"] != FAILURE_SCHEMA:
            raise RetiredCohortRegistryError(f"{path.name} schema differs")
        status = authority["status"]
        if not isinstance(status, str) or FAILURE_STATUS_RE.fullmatch(status) is None:
            raise RetiredCohortRegistryError(
                f"{path.name} does not carry a permanent failure status"
            )

        number = match.group("number")
        if str(int(number)) != number:
            raise RetiredCohortRegistryError(
                f"{path.name} cohort number is not canonical"
            )
        cohort_id = authority["cohort_id"]
        expected_cohort_id = f"b2g26_final_{number}"
        if cohort_id != expected_cohort_id:
            raise RetiredCohortRegistryError(
                f"{path.name} cohort_id does not match its filename"
            )
        if cohort_id in cohorts:
            raise RetiredCohortRegistryError(
                f"contradictory retired cohort_id {cohort_id} in "
                f"{cohorts[cohort_id]} and {path.name}"
            )

        declaration = _mapping(authority["declaration"], f"{path.name} declaration")
        if set(declaration) != {"path", "sha256"}:
            raise RetiredCohortRegistryError(
                f"{path.name} declaration binding keys differ"
            )
        expected_relative = (
            f"docs/multires/B2-GENERATED-COHORT-{number}-DECLARATION.json"
        )
        if declaration["path"] != expected_relative:
            raise RetiredCohortRegistryError(
                f"{path.name} declaration path is not canonical"
            )
        declaration_sha256 = declaration["sha256"]
        if (
            not isinstance(declaration_sha256, str)
            or HEX_RE.fullmatch(declaration_sha256) is None
        ):
            raise RetiredCohortRegistryError(
                f"{path.name} declaration SHA-256 is malformed"
            )
        if declaration_sha256 in declarations:
            raise RetiredCohortRegistryError(
                f"contradictory retired declaration SHA-256 {declaration_sha256} in "
                f"{declarations[declaration_sha256]} and {path.name}"
            )
        _validate_admission(authority["admission"], path.name)
        _mapping(authority["evidence"], f"{path.name} evidence")
        _mapping(authority["failure"], f"{path.name} failure")
        if "operator_transcript" in authority:
            _mapping(
                authority["operator_transcript"],
                f"{path.name} operator transcript",
            )
        if "operator_preflight_attempts" in authority:
            attempts = authority["operator_preflight_attempts"]
            if not isinstance(attempts, list) or any(
                not isinstance(attempt, Mapping) for attempt in attempts
            ):
                raise RetiredCohortRegistryError(
                    f"{path.name} operator preflight attempts must be an object array"
                )

        declaration_path = ROOT / expected_relative
        _require_no_symlink_components(
            declaration_path, ROOT, f"{path.name} bound declaration"
        )
        if not declaration_path.is_file() or declaration_path.is_symlink():
            raise RetiredCohortRegistryError(
                f"{path.name} bound declaration is missing, invalid, or a symlink"
            )
        try:
            bound, actual_sha256 = load_declaration(declaration_path)
        except (GeneratorCohortError, OSError) as exc:
            raise RetiredCohortRegistryError(
                f"{path.name} bound declaration is invalid: {exc}"
            ) from exc
        if actual_sha256 != declaration_sha256:
            raise RetiredCohortRegistryError(
                f"{path.name} declaration SHA-256 does not match bound bytes"
            )
        if bound["cohort_id"] != cohort_id:
            raise RetiredCohortRegistryError(
                f"{path.name} declaration cohort_id does not match authority"
            )

        for row in bound["maps"]:
            map_id = row["map"]
            seed = row["seed"]
            if map_id in maps:
                raise RetiredCohortRegistryError(
                    f"contradictory retired map ID {map_id} belongs to "
                    f"{maps[map_id]} and {cohort_id}"
                )
            if seed in seeds:
                raise RetiredCohortRegistryError(
                    f"contradictory retired seed {seed} belongs to "
                    f"{seeds[seed]} and {cohort_id}"
                )
            maps[map_id] = cohort_id
            seeds[seed] = cohort_id
        cohorts[cohort_id] = path.name
        declarations[declaration_sha256] = path.name

    return cohorts, declarations, maps, seeds


def require_unretired_declaration(
    declaration_path: Path,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
) -> None:
    """Reject retired identity reuse before a producer creates any artifact."""

    from tools.run_generator_cohort import (
        GeneratorCohortError,
        load_declaration,
        validate_declaration,
    )

    try:
        normalized = validate_declaration(declaration)
        loaded, actual_sha256 = load_declaration(Path(declaration_path))
    except (GeneratorCohortError, OSError) as exc:
        raise RetiredCohortRegistryError(
            f"candidate declaration is invalid: {exc}"
        ) from exc
    if normalized != loaded:
        raise RetiredCohortRegistryError(
            "candidate declaration mapping does not match declaration bytes"
        )
    if (
        not isinstance(declaration_sha256, str)
        or HEX_RE.fullmatch(declaration_sha256) is None
        or declaration_sha256 != actual_sha256
    ):
        raise RetiredCohortRegistryError(
            "candidate declaration SHA-256 does not match declaration bytes"
        )

    cohorts, declarations, maps, seeds = _load_registry()
    cohort_id = normalized["cohort_id"]
    if cohort_id in cohorts:
        raise RetiredCohortRegistryError(
            f"cohort_id {cohort_id} is permanently retired by {cohorts[cohort_id]}"
        )
    if declaration_sha256 in declarations:
        raise RetiredCohortRegistryError(
            "declaration SHA-256 is permanently retired by "
            f"{declarations[declaration_sha256]}"
        )
    for row in normalized["maps"]:
        map_id = row["map"]
        if map_id in maps:
            raise RetiredCohortRegistryError(
                f"map ID {map_id} is permanently retired with {maps[map_id]}"
            )
        seed = row["seed"]
        if seed in seeds:
            raise RetiredCohortRegistryError(
                f"seed {seed} is permanently retired with {seeds[seed]}"
            )


def require_unretired_identity(cohort_id: str, declaration_sha256: str) -> None:
    """Reject an authority identity without trusting a gate's declaration body."""

    if not isinstance(cohort_id, str) or not cohort_id:
        raise RetiredCohortRegistryError("candidate cohort identity is malformed")
    if (
        not isinstance(declaration_sha256, str)
        or HEX_RE.fullmatch(declaration_sha256) is None
    ):
        raise RetiredCohortRegistryError("candidate declaration identity is malformed")
    cohorts, declarations, _maps, _seeds = _load_registry()
    if cohort_id in cohorts:
        raise RetiredCohortRegistryError(
            f"cohort_id {cohort_id} is permanently retired by {cohorts[cohort_id]}"
        )
    if declaration_sha256 in declarations:
        raise RetiredCohortRegistryError(
            "declaration SHA-256 is permanently retired by "
            f"{declarations[declaration_sha256]}"
        )
