"""Fail-closed admission for the exact Batch B1 physics authorities.

The repository's B1 gate is the trust root.  A caller supplies executable and
attestation files; recorded filesystem paths are deliberately not trusted.
Every supplied file is admitted by bytes, and every oracle must reproduce the
identity preimage sealed by the gate and hook-parity attestation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


GATE_SCHEMA = "q2-multires-batch-gate-v1"
SEAL_SCHEMA = "q2-b1-runtime-authority-seal-v1"
REQUALIFICATION_SCHEMA = "q2-b1-authority-requalification-v1"
DESIGN_RELATIVE_PATH = Path(
    "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"
)
PLAN_RELATIVE_PATH = Path(
    "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md"
)
GATE_RELATIVE_PATH = Path("docs/multires/B1-GATE.json")

# These are the accepted, externally reviewed and methodology-amended
# documents.  Checking only a mutable gate against mutable files would let a
# replacement gate bless a replacement specification.
ACCEPTED_DESIGN_SHA256 = (
    "c55fc7ffc32bd0e88410b8493b46c179f3333f3806632ff8e6530f1c717508e6"
)
ACCEPTED_PLAN_SHA256 = (
    "371577feb8c40f542c90eec4b4aa91ef84c4a8e2019bf1614e59c46aedfec410"
)

# The only historical gate that may seed the 2026-07-16 requalification.  Its
# B1 test and parity evidence remain historical evidence; it is not itself a
# current authority.  Requalification must prove these exact bytes, then
# re-admit the byte-identical oracle binaries and their live identities.
HISTORICAL_GATE_SHA256 = (
    "909b1e46b4c3dca8adb6ab9017cd8716daa8c6cdd3eb106ae11aa09bee0572f8"
)
HISTORICAL_DESIGN_SHA256 = (
    "eab02d2269f250a26f45bb5d3b1f66ffab2c34ba3ee958d2f8b5bd2a14fef8b5"
)
HISTORICAL_PLAN_SHA256 = (
    "970e97b9478b27ad1f1cd35d29a74b2ed2cd51ed1ae8b4af82605615d5b5ba6b"
)
HISTORICAL_HOOK_EXECUTABLE_SHA256 = (
    "cd8bc4107ae2e9f4ac006fbe469b360832db80b96a5597c2e5dfe12c32dc9284"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{7,40}\Z")
_HOOK_CASE_IDS = (
    "world-pull",
    "client-pull",
    "zero-pull",
    "launch",
    "backoff",
    "sky",
    "attach",
    "hook-landing",
)
_HOOK_CHECKS = {
    "bsp_identity",
    "collision_parity",
    "hook_identity_consistency",
    "hook_vector_parity",
    "pmove_landing_parity",
    "q2ded_parameter_block",
    "strict_hook_responses",
}
_CLEAN_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


class B1AuthorityError(ValueError):
    """A purported B1 authority is absent, malformed, stale, or mismatched."""


@dataclass(frozen=True)
class B1AuthorityGate:
    """The immutable authority fields extracted from the repository B1 gate."""

    repo_root: Path
    design_sha256: str
    plan_sha256: str
    hook_attestation_sha256: str
    hook_schema: str
    hook_case_count: int
    hook_pullspeed: int
    hook_physics_identity: str
    hook_tool_identity: str
    collision_fixture_identity: str
    pmove_fixture_identity: str
    fixture_bsp_sha256: str
    vector_results_sha256: str
    oracle_tool_identity: str
    oracle_source_closure_sha256: str
    cm_executable_sha256: str
    pmove_executable_sha256: str
    fall_schema: str
    fall_executable_sha256: str
    fall_tool_identity: str
    fall_default_physics_identity: str
    collision_source_sha256: str
    pmove_source_sha256: str
    shared_source_sha256: str
    shared_header_sha256: str
    hook_shared_source_sha256: str
    hook_shared_header_sha256: str
    fall_shared_source_sha256: str
    fall_shared_header_sha256: str
    fall_integration_sha256: str
    fall_constants_sha256: str


@dataclass(frozen=True)
class HookParityAuthority:
    """Fields admitted from the exact parity-attestation bytes."""

    attestation_sha256: str
    fixture_bsp_sha256: str
    fixture_bsp_bytes: int
    hook_executable_sha256: str
    hook_physics_identity: str
    hook_tool_identity: str
    collision_physics_identity: str
    pmove_physics_identity: str
    hook_speed: float
    hook_pullspeed: float
    hook_sky: bool
    hook_maxtime: float


@dataclass(frozen=True)
class B1RuntimeAuthoritySeal:
    """A successful all-or-nothing admission result."""

    schema: str
    design_sha256: str
    plan_sha256: str
    hook_parity_attestation_sha256: str
    fixture_bsp_sha256: str
    analysis_bsp_sha256: str
    cm_executable_sha256: str
    pmove_executable_sha256: str
    hook_executable_sha256: str
    fall_executable_sha256: str
    collision_tool_identity: str
    collision_physics_identity: str
    pmove_tool_identity: str
    pmove_physics_identity: str
    hook_tool_identity: str
    hook_physics_identity: str
    fall_tool_identity: str
    fall_physics_identity: str

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic manifest-ready representation."""

        return {
            "schema": self.schema,
            "normative_documents": {
                "design_sha256": self.design_sha256,
                "plan_sha256": self.plan_sha256,
            },
            "hook_parity_attestation_sha256": (
                self.hook_parity_attestation_sha256
            ),
            "fixture_bsp_sha256": self.fixture_bsp_sha256,
            "analysis_bsp_sha256": self.analysis_bsp_sha256,
            "executables": {
                "cm_sha256": self.cm_executable_sha256,
                "pmove_sha256": self.pmove_executable_sha256,
                "hook_sha256": self.hook_executable_sha256,
                "fall_sha256": self.fall_executable_sha256,
            },
            "identities": {
                "collision": {
                    "tool_identity": self.collision_tool_identity,
                    "physics_identity": self.collision_physics_identity,
                },
                "pmove": {
                    "tool_identity": self.pmove_tool_identity,
                    "physics_identity": self.pmove_physics_identity,
                },
                "hook": {
                    "tool_identity": self.hook_tool_identity,
                    "physics_identity": self.hook_physics_identity,
                },
                "fall": {
                    "tool_identity": self.fall_tool_identity,
                    "physics_identity": self.fall_physics_identity,
                },
            },
        }


def _reject(condition: bool, message: str) -> None:
    if condition:
        raise B1AuthorityError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _reject(not isinstance(value, Mapping), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    _reject(
        actual != keys,
        f"{label} fields differ; missing={sorted(keys - actual)}, "
        f"extra={sorted(actual - keys)}",
    )


def _sha256(value: object, label: str) -> str:
    _reject(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None,
        f"{label} must be a lowercase SHA-256",
    )
    return str(value)


def canonical_cm_physics_identity(
    tool_identity: object, map_sha256: object,
) -> str:
    """Return the sole admitted CM physics identity for one BSP."""

    tool = _sha256(tool_identity, "CM tool identity")
    map_digest = _sha256(map_sha256, "CM map digest")
    return hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=cm;tool_identity="
            f"{tool};map={map_digest}"
        ).encode("ascii")
    ).hexdigest()


def _positive_int(value: object, label: str) -> int:
    _reject(type(value) is not int or value <= 0, f"{label} must be positive")
    return int(value)


def _finite_number(value: object, label: str) -> float:
    _reject(
        isinstance(value, bool) or not isinstance(value, (int, float)),
        f"{label} must be a finite number",
    )
    result = float(value)
    _reject(not math.isfinite(result), f"{label} must be a finite number")
    return result


def _regular_file_bytes(path: Path, label: str, *, limit: int) -> bytes:
    try:
        with path.open("rb") as handle:
            file_stat = os.fstat(handle.fileno())
            _reject(not stat.S_ISREG(file_stat.st_mode), f"{label} is not regular")
            _reject(file_stat.st_size > limit, f"{label} exceeds {limit} bytes")
            data = handle.read(limit + 1)
    except OSError as error:
        raise B1AuthorityError(f"cannot read {label}: {error}") from error
    _reject(len(data) > limit, f"{label} exceeds {limit} bytes")
    return data


def _file_sha256(path: Path, label: str) -> str:
    try:
        with path.open("rb") as handle:
            file_stat = os.fstat(handle.fileno())
            _reject(not stat.S_ISREG(file_stat.st_mode), f"{label} is not regular")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise B1AuthorityError(f"cannot hash {label}: {error}") from error
    return digest.hexdigest()


def _decode_json_object(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise B1AuthorityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise B1AuthorityError(f"{label} contains non-finite number {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except B1AuthorityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise B1AuthorityError(f"{label} is not strict UTF-8 JSON: {error}") from error
    _reject(not isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise B1AuthorityError(f"cannot canonicalize JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _require_true_fields(value: Mapping[str, Any], label: str) -> None:
    for name, item in value.items():
        _reject(item is not True, f"{label}.{name} must be true")


def _validate_gate_document(
    document: Mapping[str, Any],
    repo_root: Path,
    *,
    expected_design_sha256: str = ACCEPTED_DESIGN_SHA256,
    expected_plan_sha256: str = ACCEPTED_PLAN_SHA256,
    verify_normative_files: bool = True,
    require_requalification: bool = True,
) -> B1AuthorityGate:
    top_level_keys = {
        "schema", "batch", "status", "recorded_at", "amended_at",
        "owner_directive", "normative_documents", "integration_heads",
        "integrated_work", "artifacts", "physics_source_sha256", "tests",
        "admission_invariants", "limitations_carried_to_B2", "deployment",
        "gate",
    }
    if require_requalification:
        top_level_keys.add("authority_requalification")
    _exact_keys(
        document,
        top_level_keys,
        "B1 gate",
    )
    _reject(document["schema"] != GATE_SCHEMA, "B1 gate schema mismatch")
    _reject(document["batch"] != "B1", "gate is not Batch B1")
    _reject(document["status"] != "green", "Batch B1 status is not green")
    for timestamp in ("recorded_at", "amended_at"):
        _reject(
            not isinstance(document[timestamp], str) or not document[timestamp],
            f"B1 gate {timestamp} is invalid",
        )

    directive = _mapping(document["owner_directive"], "owner directive")
    _exact_keys(
        directive,
        {
            "replacement", "legacy_model_lineages", "operational_fallback",
            "legacy_runtime_or_model_used",
        },
        "owner directive",
    )
    _reject(
        directive != {
            "replacement": "one-way",
            "legacy_model_lineages": "retired",
            "operational_fallback": "forbidden",
            "legacy_runtime_or_model_used": False,
        },
        "B1 gate does not preserve the one-way retirement directive",
    )

    normative = _mapping(document["normative_documents"], "normative documents")
    _exact_keys(normative, {"design_sha256", "plan_sha256"}, "normative documents")
    design_sha = _sha256(normative["design_sha256"], "design digest")
    plan_sha = _sha256(normative["plan_sha256"], "plan digest")
    _reject(design_sha != expected_design_sha256, "unaccepted design digest in B1 gate")
    _reject(plan_sha != expected_plan_sha256, "unaccepted plan digest in B1 gate")
    if verify_normative_files:
        actual_design = _file_sha256(
            repo_root / DESIGN_RELATIVE_PATH, "normative design"
        )
        actual_plan = _file_sha256(repo_root / PLAN_RELATIVE_PATH, "normative plan")
        _reject(actual_design != design_sha, "normative design bytes changed")
        _reject(actual_plan != plan_sha, "normative plan bytes changed")

    heads = _mapping(document["integration_heads"], "integration heads")
    _exact_keys(heads, {"q2-ml-bot", "q2-ml-client", "q2-lithium-3zb2"}, "integration heads")
    for name, commit in heads.items():
        _reject(
            not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None,
            f"integration head {name} is invalid",
        )

    integrated = _mapping(document["integrated_work"], "integrated work")
    _exact_keys(
        integrated,
        {
            "atlas_core", "corpus_quarantine", "atlas_oracle_admission",
            "collision_pmove_oracles", "client_oracle_provenance",
            "transformed_inline_collision", "hook_oracle",
            "hook_parity_attestation", "fall_damage_oracle",
        },
        "integrated work",
    )
    for name, raw in integrated.items():
        item = _mapping(raw, f"integrated work {name}")
        expected = {"agent_commit", "integration_commit"}
        if name in {"transformed_inline_collision", "fall_damage_oracle"}:
            expected.add("reason")
        _exact_keys(item, expected, f"integrated work {name}")
        for field in ("agent_commit", "integration_commit"):
            commit = item[field]
            _reject(
                not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None,
                f"integrated work {name}.{field} is invalid",
            )
        if "reason" in item:
            _reject(not isinstance(item["reason"], str) or not item["reason"], f"{name} reason is invalid")

    artifacts = _mapping(document["artifacts"], "B1 artifacts")
    _exact_keys(
        artifacts,
        {
            "atlas_fixture_uncompressed_sha256", "atlas_hardened_manifest_sha256",
            "stock_corpus_report_sha256", "stock_pak_sha256",
            "hook_parity_attestation", "transformed_inline_collision",
            "fall_damage_oracle",
        },
        "B1 artifacts",
    )
    for name in (
        "atlas_fixture_uncompressed_sha256", "atlas_hardened_manifest_sha256",
        "stock_corpus_report_sha256", "stock_pak_sha256",
    ):
        _sha256(artifacts[name], name)

    hook = _mapping(artifacts["hook_parity_attestation"], "hook parity gate")
    _exact_keys(
        hook,
        {
            "path", "sha256", "schema", "passed", "case_count",
            "hook_pullspeed", "hook_physics_identity", "hook_tool_identity",
            "collision_fixture_identity", "pmove_fixture_identity",
            "fixture_bsp_sha256", "vector_results_sha256",
        },
        "hook parity gate",
    )
    _reject(not isinstance(hook["path"], str) or not hook["path"], "recorded hook path is invalid")
    _reject(hook["schema"] != "q2-hook-parity-attestation-v1", "hook attestation schema mismatch")
    _reject(hook["passed"] is not True, "hook parity is not green")
    hook_cases = _positive_int(hook["case_count"], "hook case count")
    _reject(hook_cases != len(_HOOK_CASE_IDS), "unexpected B1 hook case count")
    _reject(type(hook["hook_pullspeed"]) is not int or hook["hook_pullspeed"] != 1700, "unexpected B1 hook pullspeed")

    transformed = _mapping(artifacts["transformed_inline_collision"], "transformed collision gate")
    _exact_keys(
        transformed,
        {
            "oracle_tool_identity", "oracle_source_closure_sha256",
            "cm_oracle_sha256", "pmove_oracle_sha256", "operations",
        },
        "transformed collision gate",
    )
    _reject(
        transformed["operations"] != [
            "transformed_point_contents", "transformed_box_trace"
        ],
        "required transformed collision operations are not sealed",
    )

    fall = _mapping(artifacts["fall_damage_oracle"], "fall gate")
    _exact_keys(
        fall,
        {
            "schema", "executable_sha256", "tool_identity",
            "default_physics_identity", "fall_damagemod_1_5_physics_identity",
            "dmflags_no_falling_physics_identity",
        },
        "fall gate",
    )
    _reject(fall["schema"] != "q2-fall-oracle-v1", "fall gate schema mismatch")

    sources = _mapping(document["physics_source_sha256"], "physics sources")
    _exact_keys(
        sources,
        {
            "collision_c", "pmove_c", "yamagi_shared_c", "yamagi_shared_h",
            "hook_shared_c", "hook_shared_h", "fall_shared_c", "fall_shared_h",
            "fall_runtime_adapter", "fall_constants",
        },
        "physics sources",
    )
    for name, digest in sources.items():
        _sha256(digest, f"physics source {name}")

    invariants = _mapping(document["admission_invariants"], "admission invariants")
    _exact_keys(
        invariants,
        {
            "collision_failure_rejects_build",
            "pmove_absence_forbids_jump_and_controlled_drop",
            "hook_or_parity_absence_forbids_hook_edges",
            "all_edges_require_nonzero_evidence_and_validation_version",
            "bsp_tool_schema_parameter_source_and_parity_mismatches_reject",
            "canonical_reencode_blocks_same_count_artifact_substitution",
        },
        "admission invariants",
    )
    _require_true_fields(invariants, "admission invariant")

    limitations = document["limitations_carried_to_B2"]
    _reject(
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item for item in limitations),
        "B1 limitations are malformed",
    )
    deployment = _mapping(document["deployment"], "B1 deployment")
    _exact_keys(
        deployment,
        {
            "public_or_teacher_service_changed", "cross_host_runtime_copy_performed",
            "external_or_community_download_performed", "trainer_or_tensorboard_started",
        },
        "B1 deployment",
    )
    for name, value in deployment.items():
        _reject(value is not False, f"B1 deployment.{name} must be false")

    gate = _mapping(document["gate"], "B1 gate predicate")
    _exact_keys(
        gate,
        {
            "physics_oracles_build_and_match_engine",
            "atlas_serialization_deterministic",
            "all_stock_maps_structurally_admitted", "oracle_admission_fail_closed",
            "worktrees_clean", "failures", "green",
        },
        "B1 gate predicate",
    )
    _require_true_fields(
        {name: gate[name] for name in gate if name != "failures"},
        "B1 gate predicate",
    )
    _reject(gate["failures"] != [], "B1 gate records failures")
    _reject(not isinstance(document["tests"], Mapping), "B1 tests must be an object")

    result = B1AuthorityGate(
        repo_root=repo_root,
        design_sha256=design_sha,
        plan_sha256=plan_sha,
        hook_attestation_sha256=_sha256(hook["sha256"], "hook attestation digest"),
        hook_schema=str(hook["schema"]),
        hook_case_count=hook_cases,
        hook_pullspeed=int(hook["hook_pullspeed"]),
        hook_physics_identity=_sha256(hook["hook_physics_identity"], "hook physics identity"),
        hook_tool_identity=_sha256(hook["hook_tool_identity"], "hook tool identity"),
        collision_fixture_identity=_sha256(hook["collision_fixture_identity"], "collision fixture identity"),
        pmove_fixture_identity=_sha256(hook["pmove_fixture_identity"], "Pmove fixture identity"),
        fixture_bsp_sha256=_sha256(hook["fixture_bsp_sha256"], "fixture BSP digest"),
        vector_results_sha256=_sha256(hook["vector_results_sha256"], "hook vector digest"),
        oracle_tool_identity=_sha256(transformed["oracle_tool_identity"], "oracle tool identity"),
        oracle_source_closure_sha256=_sha256(transformed["oracle_source_closure_sha256"], "oracle source closure"),
        cm_executable_sha256=_sha256(transformed["cm_oracle_sha256"], "CM executable digest"),
        pmove_executable_sha256=_sha256(transformed["pmove_oracle_sha256"], "Pmove executable digest"),
        fall_schema=str(fall["schema"]),
        fall_executable_sha256=_sha256(fall["executable_sha256"], "fall executable digest"),
        fall_tool_identity=_sha256(fall["tool_identity"], "fall tool identity"),
        fall_default_physics_identity=_sha256(fall["default_physics_identity"], "fall default physics identity"),
        collision_source_sha256=str(sources["collision_c"]),
        pmove_source_sha256=str(sources["pmove_c"]),
        shared_source_sha256=str(sources["yamagi_shared_c"]),
        shared_header_sha256=str(sources["yamagi_shared_h"]),
        hook_shared_source_sha256=str(sources["hook_shared_c"]),
        hook_shared_header_sha256=str(sources["hook_shared_h"]),
        fall_shared_source_sha256=str(sources["fall_shared_c"]),
        fall_shared_header_sha256=str(sources["fall_shared_h"]),
        fall_integration_sha256=str(sources["fall_runtime_adapter"]),
        fall_constants_sha256=str(sources["fall_constants"]),
    )
    if require_requalification:
        _validate_authority_requalification(
            document["authority_requalification"], result
        )
    return result


def _validate_authority_requalification(
    value: object, gate: B1AuthorityGate
) -> None:
    record = _mapping(value, "B1 authority requalification")
    _exact_keys(
        record,
        {
            "schema", "status", "recorded_at", "historical_gate_sha256",
            "historical_normative_documents", "current_normative_documents",
            "probe_bsp_sha256", "repository", "inputs", "live_identities",
            "probe_runtime_authority_seal", "checks", "failures",
        },
        "B1 authority requalification",
    )
    _reject(
        record["schema"] != REQUALIFICATION_SCHEMA,
        "B1 authority requalification schema mismatch",
    )
    _reject(
        record["status"] != "green",
        "B1 authority requalification is not green",
    )
    _reject(
        not isinstance(record["recorded_at"], str) or not record["recorded_at"],
        "B1 authority requalification timestamp is invalid",
    )
    _reject(
        record["historical_gate_sha256"] != HISTORICAL_GATE_SHA256,
        "B1 authority requalification does not bind the historical gate",
    )

    historical = _mapping(
        record["historical_normative_documents"],
        "historical B1 normative documents",
    )
    current = _mapping(
        record["current_normative_documents"],
        "current B1 normative documents",
    )
    for documents, label in (
        (historical, "historical B1 normative documents"),
        (current, "current B1 normative documents"),
    ):
        _exact_keys(documents, {"design_sha256", "plan_sha256"}, label)
        for name, digest in documents.items():
            _sha256(digest, f"{label} {name}")
    _reject(
        historical
        != {
            "design_sha256": HISTORICAL_DESIGN_SHA256,
            "plan_sha256": HISTORICAL_PLAN_SHA256,
        },
        "historical B1 normative binding differs",
    )
    _reject(
        current
        != {
            "design_sha256": gate.design_sha256,
            "plan_sha256": gate.plan_sha256,
        },
        "current B1 normative binding differs from the gate",
    )
    probe_bsp_sha256 = _sha256(
        record["probe_bsp_sha256"], "B1 requalification probe BSP"
    )

    repository = _mapping(record["repository"], "B1 requalification repository")
    _exact_keys(repository, {"commit", "tree", "clean"}, "B1 requalification repository")
    for name in ("commit", "tree"):
        _reject(
            not isinstance(repository[name], str)
            or re.fullmatch(r"[0-9a-f]{40}", repository[name]) is None,
            f"B1 requalification repository {name} is invalid",
        )
    _reject(repository["clean"] is not True, "B1 requalification repository was dirty")

    inputs = _mapping(record["inputs"], "B1 requalification inputs")
    _exact_keys(
        inputs,
        {"hook_parity_attestation_sha256", "executables"},
        "B1 requalification inputs",
    )
    _reject(
        inputs["hook_parity_attestation_sha256"]
        != gate.hook_attestation_sha256,
        "requalified hook attestation differs from B1",
    )
    executables = _mapping(
        inputs["executables"], "B1 requalification executables"
    )
    _exact_keys(
        executables,
        {"cm_sha256", "pmove_sha256", "hook_sha256", "fall_sha256"},
        "B1 requalification executables",
    )
    for name, digest in executables.items():
        _sha256(digest, f"B1 requalification executable {name}")
    _reject(
        executables
        != {
            "cm_sha256": gate.cm_executable_sha256,
            "pmove_sha256": gate.pmove_executable_sha256,
            "hook_sha256": HISTORICAL_HOOK_EXECUTABLE_SHA256,
            "fall_sha256": gate.fall_executable_sha256,
        },
        "requalified executable bytes differ from historical B1",
    )

    identities = _mapping(
        record["live_identities"], "B1 requalification live identities"
    )
    _exact_keys(
        identities,
        {"collision", "pmove", "hook", "fall"},
        "B1 requalification live identities",
    )
    admitted: dict[str, Mapping[str, Any]] = {}
    for name, raw in identities.items():
        identity = _mapping(raw, f"B1 requalification {name} identity")
        expected_keys = {"tool_identity", "physics_identity"}
        if name == "pmove":
            expected_keys.add("parameters")
        _exact_keys(
            identity,
            expected_keys,
            f"B1 requalification {name} identity",
        )
        _sha256(identity["tool_identity"], f"B1 requalification {name} tool")
        _sha256(
            identity["physics_identity"],
            f"B1 requalification {name} physics",
        )
        admitted[name] = identity

    collision_expected = canonical_cm_physics_identity(
        gate.oracle_tool_identity, probe_bsp_sha256
    )
    _reject(
        admitted["collision"]
        != {
            "tool_identity": gate.oracle_tool_identity,
            "physics_identity": collision_expected,
        },
        "requalified collision identity is not canonical",
    )
    # Pmove's constants are a sealed part of its live identity response.  The
    # producer validates the complete response and records the resulting
    # physics digest; the gate independently fixes its tool identity here.
    _reject(
        admitted["pmove"]["tool_identity"] != gate.oracle_tool_identity,
        "requalified Pmove tool identity differs from B1",
    )
    pmove_parameters = _mapping(
        admitted["pmove"]["parameters"],
        "B1 requalification Pmove parameters",
    )
    _exact_keys(
        pmove_parameters,
        {"gravity", "airaccelerate", "constants"},
        "B1 requalification Pmove parameters",
    )
    _reject(
        pmove_parameters["gravity"] != 800
        or pmove_parameters["airaccelerate"] != 0
        or not isinstance(pmove_parameters["constants"], str)
        or not pmove_parameters["constants"],
        "requalified Pmove parameters differ from B1",
    )
    pmove_expected = hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=pmove;tool_identity="
            f"{gate.oracle_tool_identity};map={probe_bsp_sha256};gravity=800;"
            f"airaccelerate=0;constants={pmove_parameters['constants']}"
        ).encode()
    ).hexdigest()
    _reject(
        admitted["pmove"]["physics_identity"] != pmove_expected,
        "requalified Pmove identity is not canonical",
    )
    _reject(
        admitted["hook"]
        != {
            "tool_identity": gate.hook_tool_identity,
            "physics_identity": gate.hook_physics_identity,
        },
        "requalified hook identity differs from B1",
    )
    _reject(
        admitted["fall"]
        != {
            "tool_identity": gate.fall_tool_identity,
            "physics_identity": gate.fall_default_physics_identity,
        },
        "requalified fall identity differs from B1",
    )

    probe_seal = _mapping(
        record["probe_runtime_authority_seal"],
        "B1 requalification probe runtime authority seal",
    )
    _exact_keys(
        probe_seal,
        {
            "schema", "normative_documents", "hook_parity_attestation_sha256",
            "fixture_bsp_sha256", "analysis_bsp_sha256", "executables",
            "identities",
        },
        "B1 requalification probe runtime authority seal",
    )
    _reject(
        probe_seal["schema"] != SEAL_SCHEMA,
        "B1 requalification probe seal schema mismatch",
    )
    _reject(
        probe_seal["normative_documents"] != current,
        "B1 requalification probe seal does not bind current documents",
    )
    _reject(
        probe_seal["hook_parity_attestation_sha256"]
        != gate.hook_attestation_sha256
        or probe_seal["fixture_bsp_sha256"] != gate.fixture_bsp_sha256
        or probe_seal["analysis_bsp_sha256"] != probe_bsp_sha256,
        "B1 requalification probe seal artifact binding differs",
    )
    _reject(
        probe_seal["executables"] != executables,
        "B1 requalification probe seal executable binding differs",
    )
    projected_identities = {
        name: {
            "tool_identity": identity["tool_identity"],
            "physics_identity": identity["physics_identity"],
        }
        for name, identity in admitted.items()
    }
    _reject(
        probe_seal["identities"] != projected_identities,
        "B1 requalification probe seal identity binding differs",
    )

    checks = _mapping(record["checks"], "B1 requalification checks")
    _exact_keys(
        checks,
        {
            "historical_gate_exact_bytes", "normative_documents_rehashed",
            "repository_clean", "executable_bytes_match_historical_gate",
            "hook_attestation_revalidated", "live_identities_recomputed",
            "live_identity_preimages_validated",
        },
        "B1 requalification checks",
    )
    _require_true_fields(checks, "B1 requalification check")
    _reject(record["failures"] != [], "B1 authority requalification records failures")


def load_b1_authority_gate(repo_root: Path | str | None = None) -> B1AuthorityGate:
    """Load only ``docs/multires/B1-GATE.json`` from a repository root.

    The gate cannot redirect its normative inputs or parity artifact.  Its
    recorded artifact path is informational and is never dereferenced.
    """

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    data = _regular_file_bytes(root / GATE_RELATIVE_PATH, "repository B1 gate", limit=1024 * 1024)
    return _validate_gate_document(_decode_json_object(data, "repository B1 gate"), root)


def load_historical_b1_authority_gate(
    path: Path | str, *, repo_root: Path | str | None = None
) -> B1AuthorityGate:
    """Load only the exact historical gate as a requalification input.

    This does not make the historical gate current.  It exists solely so the
    requalification producer can verify the old evidence root byte-for-byte
    before challenging the sealed binaries again under amended documents.
    """

    gate_path = Path(path)
    data = _regular_file_bytes(
        gate_path, "historical B1 gate", limit=1024 * 1024
    )
    _reject(
        hashlib.sha256(data).hexdigest() != HISTORICAL_GATE_SHA256,
        "historical B1 gate bytes differ from the accepted evidence root",
    )
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return _validate_gate_document(
        _decode_json_object(data, "historical B1 gate"),
        root,
        expected_design_sha256=HISTORICAL_DESIGN_SHA256,
        expected_plan_sha256=HISTORICAL_PLAN_SHA256,
        verify_normative_files=False,
        require_requalification=False,
    )


def _hook_physics_identity(parameters: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    number = lambda value: format(float(value), ".9g")
    preimage = (
        f"schema=q2-hook-oracle-v1;shared_c={source['shared_c_sha256']};"
        f"shared_h={source['shared_h_sha256']};"
        f"integration={source['integration_sha256']};"
        f"math={source['math_sha256']};build={source['build_contract']};"
        f"tool_closure={source['tool_closure_sha256']};"
        f"hook_speed={number(parameters['hook_speed'])};"
        f"hook_pullspeed={number(parameters['hook_pullspeed'])};"
        f"hook_sky={1 if parameters['hook_sky'] else 0};"
        f"hook_maxtime={number(parameters['hook_maxtime'])};"
        "full_velocity_overwrite=1"
    )
    return hashlib.sha256(preimage.encode()).hexdigest()


def _validate_hook_parameters(value: object, gate: B1AuthorityGate) -> dict[str, Any]:
    parameters = _mapping(value, "hook parameters")
    _exact_keys(
        parameters,
        {"hook_speed", "hook_pullspeed", "hook_sky", "hook_maxtime", "full_velocity_overwrite"},
        "hook parameters",
    )
    speed = _finite_number(parameters["hook_speed"], "hook speed")
    pullspeed = _finite_number(parameters["hook_pullspeed"], "hook pullspeed")
    maxtime = _finite_number(parameters["hook_maxtime"], "hook maxtime")
    _reject(min(speed, pullspeed, maxtime) < 0, "hook parameters must be nonnegative")
    _reject(parameters["hook_sky"] is not False, "B1 hook_sky must be false")
    _reject(parameters["full_velocity_overwrite"] is not True, "B1 hook must overwrite full velocity")
    _reject(speed != 900.0 or pullspeed != float(gate.hook_pullspeed) or maxtime != 5.0, "hook parameter block differs from B1")
    return dict(parameters)


def _validate_hook_source(value: object, gate: B1AuthorityGate) -> dict[str, Any]:
    source = _mapping(value, "hook source")
    _exact_keys(
        source,
        {
            "shared_c_sha256", "shared_h_sha256", "integration_sha256",
            "math_sha256", "build_contract", "tool_closure_sha256",
        },
        "hook source",
    )
    for name in set(source) - {"build_contract"}:
        _sha256(source[name], f"hook source {name}")
    _reject(source["shared_c_sha256"] != gate.hook_shared_source_sha256, "hook shared C source is stale")
    _reject(source["shared_h_sha256"] != gate.hook_shared_header_sha256, "hook shared header is stale")
    _reject(source["build_contract"] != "lithium-linux-c99-o1-f32-shared-hook-v2", "hook build contract mismatch")
    _reject(source["tool_closure_sha256"] != gate.hook_tool_identity, "hook source closure differs from B1")
    return dict(source)


def _validate_hook_attestation_record(
    document: Mapping[str, Any], gate: B1AuthorityGate, digest: str
) -> tuple[HookParityAuthority, dict[str, Any]]:
    _reject(digest != gate.hook_attestation_sha256, "hook parity bytes are not the B1 artifact")
    _exact_keys(
        document,
        {
            "schema", "passed", "parameters", "fixture", "identities",
            "binaries", "attestor_closure_sha256", "evidence", "checks",
        },
        "hook parity attestation",
    )
    _reject(document["schema"] != gate.hook_schema, "hook parity schema differs from B1")
    _reject(document["passed"] is not True, "hook parity attestation did not pass")
    parameters = _validate_hook_parameters(document["parameters"], gate)

    fixture = _mapping(document["fixture"], "hook fixture")
    _exact_keys(fixture, {"name", "bsp_sha256", "bsp_bytes", "ibsp_version"}, "hook fixture")
    _reject(fixture["name"] != "hookprobe-v1", "hook fixture name mismatch")
    _reject(fixture["ibsp_version"] != 38, "hook fixture is not IBSP-38")
    fixture_bytes = _positive_int(fixture["bsp_bytes"], "hook fixture bytes")
    fixture_sha = _sha256(fixture["bsp_sha256"], "hook fixture digest")
    _reject(fixture_sha != gate.fixture_bsp_sha256, "hook fixture digest differs from B1")

    identities = _mapping(document["identities"], "hook identities")
    _exact_keys(identities, {"collision", "pmove", "hook"}, "hook identities")
    hook = _mapping(identities["hook"], "hook identity")
    _exact_keys(hook, {"schema", "physics_identity", "tool_identity", "source"}, "hook identity")
    _reject(hook["schema"] != "q2-hook-oracle-v1", "hook oracle schema mismatch")
    hook_physics = _sha256(hook["physics_identity"], "hook physics identity")
    hook_tool = _sha256(hook["tool_identity"], "hook tool identity")
    _reject(hook_physics != gate.hook_physics_identity, "hook physics identity differs from B1")
    _reject(hook_tool != gate.hook_tool_identity, "hook tool identity differs from B1")
    source = _validate_hook_source(hook["source"], gate)
    _reject(hook_tool != source["tool_closure_sha256"], "hook tool/source identity mismatch")
    _reject(hook_physics != _hook_physics_identity(parameters, source), "hook physics preimage mismatch")

    fixture_identities: dict[str, str] = {}
    for name, schema, expected in (
        ("collision", "q2-cm-oracle-v1", gate.collision_fixture_identity),
        ("pmove", "q2-pmove-oracle-v1", gate.pmove_fixture_identity),
    ):
        identity = _mapping(identities[name], f"{name} fixture identity")
        _exact_keys(identity, {"schema", "physics_identity", "map_sha256"}, f"{name} fixture identity")
        _reject(identity["schema"] != schema, f"{name} fixture schema mismatch")
        physics = _sha256(identity["physics_identity"], f"{name} fixture physics identity")
        _reject(physics != expected, f"{name} fixture physics identity differs from B1")
        _reject(identity["map_sha256"] != fixture_sha, f"{name} fixture does not bind the BSP")
        fixture_identities[name] = physics

    binaries = _mapping(document["binaries"], "hook parity binaries")
    _exact_keys(
        binaries,
        {
            "q2ded_sha256", "probe_game_module_sha256", "hook_oracle_sha256",
            "cm_oracle_sha256", "pmove_oracle_sha256",
        },
        "hook parity binaries",
    )
    for name, value in binaries.items():
        _sha256(value, f"hook parity binary {name}")
    _reject(binaries["cm_oracle_sha256"] != gate.cm_executable_sha256, "attested CM binary differs from B1")
    _reject(binaries["pmove_oracle_sha256"] != gate.pmove_executable_sha256, "attested Pmove binary differs from B1")
    _sha256(document["attestor_closure_sha256"], "hook attestor closure")

    evidence = _mapping(document["evidence"], "hook parity evidence")
    _exact_keys(evidence, {"case_count", "case_ids", "vector_results_sha256"}, "hook parity evidence")
    _reject(evidence["case_count"] != gate.hook_case_count, "hook parity case count differs from B1")
    _reject(evidence["case_ids"] != list(_HOOK_CASE_IDS), "hook parity case IDs differ from B1")
    _reject(evidence["vector_results_sha256"] != gate.vector_results_sha256, "hook vector evidence differs from B1")
    checks = _mapping(document["checks"], "hook parity checks")
    _exact_keys(checks, _HOOK_CHECKS, "hook parity checks")
    _require_true_fields(checks, "hook parity check")

    authority = HookParityAuthority(
        attestation_sha256=digest,
        fixture_bsp_sha256=fixture_sha,
        fixture_bsp_bytes=fixture_bytes,
        hook_executable_sha256=str(binaries["hook_oracle_sha256"]),
        hook_physics_identity=hook_physics,
        hook_tool_identity=hook_tool,
        collision_physics_identity=fixture_identities["collision"],
        pmove_physics_identity=fixture_identities["pmove"],
        hook_speed=float(parameters["hook_speed"]),
        hook_pullspeed=float(parameters["hook_pullspeed"]),
        hook_sky=bool(parameters["hook_sky"]),
        hook_maxtime=float(parameters["hook_maxtime"]),
    )
    return authority, dict(document)


def _load_hook_parity_attestation(
    path: Path, gate: B1AuthorityGate
) -> tuple[HookParityAuthority, dict[str, Any]]:
    data = _regular_file_bytes(path, "supplied hook parity attestation", limit=1024 * 1024)
    digest = hashlib.sha256(data).hexdigest()
    document = _decode_json_object(data, "supplied hook parity attestation")
    _reject(data != _canonical_json_bytes(document), "hook parity attestation is not canonical JSON")
    return _validate_hook_attestation_record(document, gate, digest)


def admit_hook_parity_attestation(
    path: Path | str, *, repo_root: Path | str | None = None
) -> HookParityAuthority:
    """Admit a relocated attestation by bytes; never dereference its gate path."""

    gate = load_b1_authority_gate(repo_root)
    authority, _ = _load_hook_parity_attestation(Path(path), gate)
    return authority


def _validate_provenance(value: object, gate: B1AuthorityGate) -> dict[str, Any]:
    provenance = _mapping(value, "oracle provenance")
    _exact_keys(
        provenance,
        {
            "schema", "tool_identity", "source_closure_sha256",
            "source_closure_count", "build_identity_sha256", "compiler",
            "archiver", "build",
        },
        "oracle provenance",
    )
    _reject(provenance["schema"] != "q2-oracle-tool-identity-v1", "oracle provenance schema mismatch")
    _reject(provenance["tool_identity"] != gate.oracle_tool_identity, "oracle provenance tool identity mismatch")
    _reject(provenance["source_closure_sha256"] != gate.oracle_source_closure_sha256, "oracle source closure differs from B1")
    _positive_int(provenance["source_closure_count"], "oracle source closure count")
    _sha256(provenance["build_identity_sha256"], "oracle build identity")
    compiler = _mapping(provenance["compiler"], "oracle compiler")
    _exact_keys(compiler, {"command", "version", "target", "executable_sha256"}, "oracle compiler")
    archiver = _mapping(provenance["archiver"], "oracle archiver")
    _exact_keys(archiver, {"command", "version", "executable_sha256"}, "oracle archiver")
    build = _mapping(provenance["build"], "oracle build")
    _exact_keys(build, {"cflags", "ldflags"}, "oracle build")
    for tool, label in ((compiler, "compiler"), (archiver, "archiver")):
        for field in set(tool) - {"executable_sha256"}:
            _reject(not isinstance(tool[field], str), f"oracle {label}.{field} must be text")
        _sha256(tool["executable_sha256"], f"oracle {label} executable")
    for name, value in build.items():
        _reject(not isinstance(value, str), f"oracle build.{name} must be text")
    return dict(provenance)


def _validate_cm_identity(
    record: Mapping[str, Any], gate: B1AuthorityGate, analysis_bsp_sha256: str
) -> None:
    _exact_keys(
        record,
        {
            "ok", "id", "op", "schema", "tool_identity", "physics_identity",
            "map_sha256", "map_checksum", "provenance", "source", "map",
            "model0", "clusters", "inline_models",
        },
        "CM identity response",
    )
    _reject(record["ok"] is not True or record["id"] != "b1-authority" or record["op"] != "identity", "CM identity response is not successful/causal")
    _reject(record["schema"] != "q2-cm-oracle-v1", "CM schema mismatch")
    _reject(record["tool_identity"] != gate.oracle_tool_identity, "CM tool identity differs from B1")
    expected_physics = canonical_cm_physics_identity(
        gate.oracle_tool_identity, analysis_bsp_sha256
    )
    _reject(record["physics_identity"] != expected_physics, "CM analysis-map physics identity is not canonical")
    _reject(record["map_sha256"] != analysis_bsp_sha256, "CM identity does not bind the analysis BSP")
    _reject(type(record["map_checksum"]) is not int, "CM map checksum is invalid")
    _validate_provenance(record["provenance"], gate)
    source = _mapping(record["source"], "CM source")
    _exact_keys(source, {"collision_sha256", "shared_header_sha256", "shared_source_sha256"}, "CM source")
    _reject(source["collision_sha256"] != gate.collision_source_sha256, "CM collision source is stale")
    _reject(source["shared_header_sha256"] != gate.shared_header_sha256, "CM shared header is stale")
    _reject(source["shared_source_sha256"] != gate.shared_source_sha256, "CM shared source is stale")
    _reject(not isinstance(record["map"], str) or not record["map"], "CM map label is invalid")
    model = _mapping(record["model0"], "CM model0")
    _exact_keys(model, {"mins", "maxs", "headnode"}, "CM model0")
    for name in ("mins", "maxs"):
        _reject(
            not isinstance(model[name], list)
            or len(model[name]) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in model[name]
            ),
            f"CM model0.{name} is not finite engine geometry",
        )
    for name in ("headnode",):
        _reject(type(model[name]) is not int, f"CM model0.{name} is invalid")
    for name in ("clusters", "inline_models"):
        _reject(type(record[name]) is not int or record[name] < 0, f"CM {name} is invalid")


def _validate_pmove_identity(
    record: Mapping[str, Any], gate: B1AuthorityGate, analysis_bsp_sha256: str
) -> None:
    _exact_keys(
        record,
        {
            "ok", "id", "op", "schema", "tool_identity", "physics_identity",
            "map_sha256", "map_checksum", "parameters", "provenance", "source",
        },
        "Pmove identity response",
    )
    _reject(record["ok"] is not True or record["id"] != "b1-authority" or record["op"] != "identity", "Pmove identity response is not successful/causal")
    _reject(record["schema"] != "q2-pmove-oracle-v1", "Pmove schema mismatch")
    _reject(record["tool_identity"] != gate.oracle_tool_identity, "Pmove tool identity differs from B1")
    _reject(record["map_sha256"] != analysis_bsp_sha256, "Pmove identity does not bind the analysis BSP")
    _reject(type(record["map_checksum"]) is not int, "Pmove map checksum is invalid")
    parameters = _mapping(record["parameters"], "Pmove parameters")
    _exact_keys(parameters, {"gravity", "airaccelerate", "constants"}, "Pmove parameters")
    _reject(parameters["gravity"] != 800 or parameters["airaccelerate"] != 0, "Pmove fixture parameters differ from B1")
    _reject(not isinstance(parameters["constants"], str) or not parameters["constants"], "Pmove constants are empty")
    expected_physics = hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=pmove;tool_identity="
            f"{gate.oracle_tool_identity};map={analysis_bsp_sha256};gravity=800;"
            f"airaccelerate=0;constants={parameters['constants']}"
        ).encode()
    ).hexdigest()
    _reject(record["physics_identity"] != expected_physics, "Pmove analysis-map physics identity is not canonical")
    _validate_provenance(record["provenance"], gate)
    source = _mapping(record["source"], "Pmove source")
    _exact_keys(source, {"collision_sha256", "pmove_sha256", "shared_header_sha256", "shared_source_sha256"}, "Pmove source")
    _reject(source["collision_sha256"] != gate.collision_source_sha256, "Pmove collision source is stale")
    _reject(source["pmove_sha256"] != gate.pmove_source_sha256, "Pmove source is stale")
    _reject(source["shared_header_sha256"] != gate.shared_header_sha256, "Pmove shared header is stale")
    _reject(source["shared_source_sha256"] != gate.shared_source_sha256, "Pmove shared source is stale")


def _validate_hook_identity(
    record: Mapping[str, Any], gate: B1AuthorityGate,
    parity: HookParityAuthority, attestation: Mapping[str, Any]
) -> None:
    _exact_keys(
        record,
        {"ok", "id", "op", "schema", "physics_identity", "tool_identity", "parameters", "source"},
        "hook identity response",
    )
    _reject(record["ok"] is not True or record["id"] != "b1-authority" or record["op"] != "identity", "hook identity response is not successful/causal")
    _reject(record["schema"] != "q2-hook-oracle-v1", "hook schema mismatch")
    _reject(record["tool_identity"] != parity.hook_tool_identity, "hook tool identity differs from B1")
    _reject(record["physics_identity"] != parity.hook_physics_identity, "hook physics identity differs from B1")
    parameters = _validate_hook_parameters(record["parameters"], gate)
    source = _validate_hook_source(record["source"], gate)
    expected_hook = _mapping(_mapping(attestation["identities"], "attested identities")["hook"], "attested hook identity")
    _reject(parameters != attestation["parameters"], "hook runtime parameters differ from attestation")
    _reject(source != expected_hook["source"], "hook runtime source differs from attestation")
    _reject(record["physics_identity"] != _hook_physics_identity(parameters, source), "hook runtime physics preimage mismatch")


def _fall_physics_identity(parameters: Mapping[str, Any], source: Mapping[str, Any], constants: str) -> str:
    number = lambda value: format(float(value), ".9g")
    preimage = (
        f"schema=q2-fall-oracle-v1;tool={source['tool_closure_sha256']};"
        f"shared_c={source['shared_c_sha256']};shared_h={source['shared_h_sha256']};"
        f"integration={source['integration_sha256']};"
        f"game_header={source['game_header_sha256']};"
        f"constants_sha256={source['constants_sha256']};constants={constants};"
        f"build={source['build_contract']};"
        f"fall_damagemod={number(parameters['fall_damagemod'])};"
        f"deathmatch={1 if parameters['deathmatch'] else 0};dmflags={parameters['dmflags']}"
    )
    return hashlib.sha256(preimage.encode()).hexdigest()


def _validate_fall_identity(record: Mapping[str, Any], gate: B1AuthorityGate) -> None:
    _exact_keys(
        record,
        {"ok", "id", "op", "schema", "physics_identity", "tool_identity", "parameters", "constants", "source"},
        "fall identity response",
    )
    _reject(record["ok"] is not True or record["id"] != "b1-authority" or record["op"] != "identity", "fall identity response is not successful/causal")
    _reject(record["schema"] != gate.fall_schema, "fall schema mismatch")
    _reject(record["tool_identity"] != gate.fall_tool_identity, "fall tool identity differs from B1")
    _reject(record["physics_identity"] != gate.fall_default_physics_identity, "fall default physics identity differs from B1")
    parameters = _mapping(record["parameters"], "fall parameters")
    _exact_keys(parameters, {"fall_damagemod", "deathmatch", "dmflags"}, "fall parameters")
    _reject(_finite_number(parameters["fall_damagemod"], "fall damage modifier") != 1.0, "fall damage modifier is not the B1 default")
    _reject(parameters["deathmatch"] is not True or parameters["dmflags"] != 0, "fall runtime parameters are not the B1 defaults")
    constants = record["constants"]
    _reject(not isinstance(constants, str) or not constants, "fall constants are empty")
    source = _mapping(record["source"], "fall source")
    _exact_keys(
        source,
        {
            "shared_c_sha256", "shared_h_sha256", "integration_sha256",
            "game_header_sha256", "constants_sha256", "build_contract",
            "tool_closure_sha256",
        },
        "fall source",
    )
    for name in set(source) - {"build_contract"}:
        _sha256(source[name], f"fall source {name}")
    _reject(source["shared_c_sha256"] != gate.fall_shared_source_sha256, "fall shared C source is stale")
    _reject(source["shared_h_sha256"] != gate.fall_shared_header_sha256, "fall shared header is stale")
    _reject(source["integration_sha256"] != gate.fall_integration_sha256, "fall runtime adapter is stale")
    _reject(source["constants_sha256"] != gate.fall_constants_sha256, "fall constants source is stale")
    _reject(source["tool_closure_sha256"] != gate.fall_tool_identity, "fall source closure differs from B1")
    _reject(source["build_contract"] != "lithium-linux-c99-o1-f32-shared-fall-v1", "fall build contract mismatch")
    _reject(hashlib.sha256(constants.encode()).hexdigest() != gate.fall_constants_sha256, "fall constants bytes differ from B1")
    _reject(record["physics_identity"] != _fall_physics_identity(parameters, source, constants), "fall physics preimage mismatch")


def _run_identity(
    executable: Path,
    request: Mapping[str, Any],
    *,
    arguments: Sequence[str] = (),
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = _canonical_json_bytes(request)
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            env=_CLEAN_ENV,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise B1AuthorityError(f"identity probe failed for {executable.name}: {error}") from error
    _reject(completed.returncode != 0, f"identity probe {executable.name} exited {completed.returncode}")
    _reject(bool(completed.stderr), f"identity probe {executable.name} wrote stderr")
    _reject(len(completed.stdout) > 1024 * 1024, f"identity probe {executable.name} output is oversized")
    lines = completed.stdout.splitlines()
    _reject(len(lines) != 1, f"identity probe {executable.name} did not emit exactly one record")
    return _decode_json_object(lines[0], f"{executable.name} identity response")


def _admit_executable(path: Path, expected_sha256: str, label: str) -> str:
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise B1AuthorityError(f"cannot stat {label}: {error}") from error
    _reject(not stat.S_ISREG(mode), f"{label} is not a regular file")
    _reject(mode & 0o111 == 0, f"{label} is not executable")
    digest = _file_sha256(path, label)
    _reject(digest != expected_sha256, f"{label} bytes differ from B1")
    return digest


def _resolve_supplied_path(value: Path | str, label: str) -> Path:
    try:
        return Path(value).resolve(strict=True)
    except OSError as error:
        raise B1AuthorityError(f"cannot resolve {label}: {error}") from error


def admit_b1_runtime_authorities(
    *,
    cm_oracle: Path | str,
    pmove_oracle: Path | str,
    hook_oracle: Path | str,
    fall_oracle: Path | str,
    hook_parity_attestation: Path | str,
    analysis_bsp: Path | str,
    repo_root: Path | str | None = None,
    timeout_seconds: float = 5.0,
) -> B1RuntimeAuthoritySeal:
    """Admit the complete B1 authority set or raise without a partial result.

    ``analysis_bsp`` is the map the caller will analyze.  CM and Pmove must
    bind their canonical physics identities to those exact bytes.  The
    isolated hook fixture need not remain installed: its digest and physics
    identities are sealed by the exact parity-attestation bytes and B1 gate.
    """

    timeout = _finite_number(timeout_seconds, "identity timeout")
    _reject(timeout <= 0 or timeout > 30, "identity timeout must be in (0, 30]")
    gate = load_b1_authority_gate(repo_root)
    parity, attestation = _load_hook_parity_attestation(
        _resolve_supplied_path(
            hook_parity_attestation, "supplied hook parity attestation"
        ),
        gate,
    )

    map_path = _resolve_supplied_path(analysis_bsp, "supplied analysis BSP")
    analysis_digest = _file_sha256(map_path, "supplied analysis BSP")

    cm = _resolve_supplied_path(cm_oracle, "supplied CM oracle")
    pmove = _resolve_supplied_path(pmove_oracle, "supplied Pmove oracle")
    hook = _resolve_supplied_path(hook_oracle, "supplied hook oracle")
    fall = _resolve_supplied_path(fall_oracle, "supplied fall oracle")
    cm_digest = _admit_executable(cm, gate.cm_executable_sha256, "supplied CM oracle")
    pmove_digest = _admit_executable(pmove, gate.pmove_executable_sha256, "supplied Pmove oracle")
    hook_digest = _admit_executable(hook, parity.hook_executable_sha256, "supplied hook oracle")
    fall_digest = _admit_executable(fall, gate.fall_executable_sha256, "supplied fall oracle")

    cm_identity = _run_identity(
        cm,
        {"id": "b1-authority", "op": "identity"},
        arguments=("--map", str(map_path)),
        timeout_seconds=timeout,
    )
    pmove_identity = _run_identity(
        pmove,
        {
            "id": "b1-authority", "op": "identity", "gravity": 800,
            "airaccelerate": 0,
        },
        arguments=("--map", str(map_path)),
        timeout_seconds=timeout,
    )
    hook_identity = _run_identity(
        hook,
        {
            "id": "b1-authority", "op": "identity",
            "hook_speed": parity.hook_speed,
            "hook_pullspeed": parity.hook_pullspeed,
            "hook_sky": parity.hook_sky,
            "hook_maxtime": parity.hook_maxtime,
        },
        timeout_seconds=timeout,
    )
    fall_identity = _run_identity(
        fall,
        {
            "id": "b1-authority", "op": "identity",
            "fall_damagemod": 1, "deathmatch": True, "dmflags": 0,
        },
        timeout_seconds=timeout,
    )

    _validate_cm_identity(cm_identity, gate, analysis_digest)
    _validate_pmove_identity(pmove_identity, gate, analysis_digest)
    _validate_hook_identity(hook_identity, gate, parity, attestation)
    _validate_fall_identity(fall_identity, gate)

    # Re-hash after execution so a pathname replacement cannot become the
    # admitted file between the first byte check and the identity probe.
    for path, expected, label in (
        (cm, cm_digest, "supplied CM oracle"),
        (pmove, pmove_digest, "supplied Pmove oracle"),
        (hook, hook_digest, "supplied hook oracle"),
        (fall, fall_digest, "supplied fall oracle"),
        (map_path, analysis_digest, "supplied analysis BSP"),
    ):
        _reject(_file_sha256(path, label) != expected, f"{label} changed during admission")

    return B1RuntimeAuthoritySeal(
        schema=SEAL_SCHEMA,
        design_sha256=gate.design_sha256,
        plan_sha256=gate.plan_sha256,
        hook_parity_attestation_sha256=parity.attestation_sha256,
        fixture_bsp_sha256=parity.fixture_bsp_sha256,
        analysis_bsp_sha256=analysis_digest,
        cm_executable_sha256=cm_digest,
        pmove_executable_sha256=pmove_digest,
        hook_executable_sha256=hook_digest,
        fall_executable_sha256=fall_digest,
        collision_tool_identity=str(cm_identity["tool_identity"]),
        collision_physics_identity=str(cm_identity["physics_identity"]),
        pmove_tool_identity=str(pmove_identity["tool_identity"]),
        pmove_physics_identity=str(pmove_identity["physics_identity"]),
        hook_tool_identity=str(hook_identity["tool_identity"]),
        hook_physics_identity=str(hook_identity["physics_identity"]),
        fall_tool_identity=str(fall_identity["tool_identity"]),
        fall_physics_identity=str(fall_identity["physics_identity"]),
    )


__all__ = [
    "B1AuthorityError",
    "B1AuthorityGate",
    "B1RuntimeAuthoritySeal",
    "HookParityAuthority",
    "admit_b1_runtime_authorities",
    "admit_hook_parity_attestation",
    "load_b1_authority_gate",
    "load_historical_b1_authority_gate",
]
