#!/usr/bin/env python3
"""Fail-closed executable integration gate for the multires B3-B6 milestones.

This tool consumes an explicit evidence envelope rather than inferring
readiness from component tests.  Every gate is evaluated in one run so a
partially built milestone still reports its complete failure surface.  The
report is canonical JSON with a deterministic SHA-256 over content only
(evidence bytes, never filesystem paths), so the same evidence produces the
same digest on any host.

Missing, legacy, or placeholder evidence fails the owning gate; no metric is
ever defaulted to zero.

Usage:
    python3 tools/verify_multires_integration.py --evidence envelope.json \
        [--out report.json]

The envelope maps each gate to one evidence JSON path:

    {
      "schema": "multires-integration-evidence-v1",
      "evidence": {
        "feature_action_contract": "...",
        "bundle_v3_atlas": "...",
        "runtime_epoch_fencing": "...",
        "b4_wire_generation": "...",
        "causal_reward_admission": "...",
        "lineage_attestation": "...",
        "legacy_selector_deactivation": "...",
        "deterministic_transitions": "...",
        "wsl_b6_campaign": "..."
      }
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.multires_contract import (
    ACTION_DIM,
    DYN_DIM,
    FACTUAL_DIM,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    GUIDE_CANDIDATE_DIM,
    GUIDE_CANDIDATES,
    GUIDE_CLASS_NAMES,
    GUIDE_DIM,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
    RECOVERY_DIM,
)
from harness.multires_lineage import (
    ALLOWED_INITIALIZATIONS,
    CHECKPOINT_FORMAT,
    LineageError,
)
from harness.multires_reward import CausalRewardFrame, RewardAdmissionError
from harness.multires_runtime import (
    B4_ACTION_MAGIC,
    B4_CLIENT_WIRE_VERSION,
    B4_OBSERVATION_MAGIC,
    B4_PROTOCOL_GENERATION,
    B4_ROLLOUT_SCHEMA,
    B4_TEACHER_VERSION,
    LEGACY_ACTION_MAGIC,
    LEGACY_CLIENT_WIRE_VERSION,
    LEGACY_OBSERVATION_MAGIC,
    LEGACY_ROLLOUT_SCHEMA,
    LEGACY_TEACHER_VERSION,
    adapt_b4_observation_descriptor,
    validate_runtime_evidence,
)

REPORT_SCHEMA = "multires-integration-report-v1"
ENVELOPE_SCHEMA = "multires-integration-evidence-v1"
TOOL_NAME = "verify_multires_integration"

ACTION_CARDINALITIES = {
    "vertical_intent": POSTURE_CLASSES,
    "fire": 2,
    "hook": 4,
    "weapon": 10,
}

# Private causal facts the B4 wire must carry (debug/teacher channel) so the
# causal reward reducer and vertical-echo admission can operate.  These are
# never policy inputs; their absence makes reward attribution unprovable.
REQUIRED_PRIVATE_CAUSAL_FACTS = (
    "tick",
    "client_life_epoch",
    "target_id",
    "target_epoch",
    "environmental_source_id",
    "environmental_source_epoch",
    "environmental_mod",
    "environmental_damage",
    "crouch_edge_id",
    "crouch_edge_epoch",
    "echo_tick",
    "action_generation",
    "hook_zone_id",
    "hook_attempt_tick",
    "hook_action_generation",
    "causal_flags",
)

GATE_ORDER = (
    "feature_action_contract",
    "bundle_v3_atlas",
    "runtime_epoch_fencing",
    "b4_wire_generation",
    "causal_reward_admission",
    "lineage_attestation",
    "legacy_selector_deactivation",
    "deterministic_transitions",
    "wsl_b6_campaign",
)

DETERMINISTIC_TRANSITION_COUNT = 500
MIN_ACCEPTED_TRANSITIONS_B6 = 16384
FEATURE_ASSEMBLY_P99_BUDGET_MS = 0.5
RESIDENT_ATLAS_BUDGET_BYTES = 32 * 1024 * 1024
DYN_PAYLOAD_HARD_LIMIT_BYTES = 8 * 1024 * 1024
DETERMINISTIC_STACK = "multires-stack"
DETERMINISTIC_COLLECTOR = "MultiresSynchronousCollector"
DETERMINISTIC_PROVIDER = "RustAtlasSpatialProvider"
DETERMINISTIC_LATTICE_NAMES = frozenset(
    {"q2_lattice", "q2_lattice_rs", "q2-lattice"}
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_PLACEHOLDER_TOKENS = ("placeholder", "tbd", "todo", "fixme", "changeme")


class EvidenceError(ValueError):
    """Raised when a gate's evidence is missing, malformed, or placeholder."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
        and value != "0" * 64
    )


def _reject_placeholders(value: Any, where: str) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in _PLACEHOLDER_TOKENS):
            raise EvidenceError(f"{where}: placeholder marker in {value!r}")
        if value == "0" * 64:
            raise EvidenceError(f"{where}: all-zero digest is placeholder evidence")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(key, where)
            _reject_placeholders(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{where}[{index}]")


def _require(evidence: Mapping[str, Any], key: str) -> Any:
    if key not in evidence:
        raise EvidenceError(f"required evidence field {key!r} is missing")
    return evidence[key]


def _require_number(evidence: Mapping[str, Any], key: str) -> float:
    value = _require(evidence, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"evidence field {key!r} must be numeric")
    return float(value)


def _require_int(evidence: Mapping[str, Any], key: str) -> int:
    value = _require(evidence, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"evidence field {key!r} must be an integer")
    return value


def _require_digest(evidence: Mapping[str, Any], key: str) -> str:
    value = _require(evidence, key)
    if not _valid_sha256(value):
        raise EvidenceError(f"evidence field {key!r} is not a lowercase SHA-256")
    return value


def _cross_check(
    context: Mapping[str, Any], key: str, found: Any, where: str
) -> None:
    recorded = context.get(key)
    if recorded is not None and found != recorded:
        raise EvidenceError(
            f"{where}: {key} {found!r} disagrees with previously admitted"
            f" {recorded!r}"
        )


# --- gate checks ------------------------------------------------------------


def _check_feature_action_contract(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    expected = {
        "policy_generation": POLICY_GENERATION,
        "feature_schema": FEATURE_SCHEMA,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_dim": OBS_DIM,
        "factual_dim": FACTUAL_DIM,
        "dyn_dim": DYN_DIM,
        "recovery_dim": RECOVERY_DIM,
        "guide_dim": GUIDE_DIM,
        "logical_action_dim": ACTION_DIM,
        "posture_classes": POSTURE_CLASSES,
    }
    for name, wanted in sorted(expected.items()):
        found = _require(evidence, name)
        if found != wanted:
            raise EvidenceError(
                f"contract field {name}={found!r} does not match frozen {wanted!r}"
            )
    cardinalities = _require(evidence, "action_cardinalities")
    if cardinalities != ACTION_CARDINALITIES:
        raise EvidenceError(
            f"action_cardinalities {cardinalities!r} do not match frozen"
            f" {ACTION_CARDINALITIES!r}"
        )


def _check_bundle_v3_atlas(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    bundle_version = _require_int(evidence, "bundle_version")
    if bundle_version != 3:
        raise EvidenceError(
            f"bundle_version={bundle_version} is not the mandatory-Atlas bundle v3"
        )
    artifact_state = _require(evidence, "artifact_state")
    if artifact_state != "admitted":
        raise EvidenceError(
            f"artifact_state={artifact_state!r}: only 'admitted' authorizes the"
            " runtime (built/published/validated are non-admissible)"
        )
    atlas_sha256 = _require_digest(evidence, "atlas_sha256")
    _require_digest(evidence, "objective_identity_sha256")
    context["atlas_sha256"] = atlas_sha256

    fixture = _require(evidence, "objective_fixture")
    if not isinstance(fixture, Mapping):
        raise EvidenceError("objective_fixture must be an object")
    map_name = _require(fixture, "map_name")
    if not isinstance(map_name, str) or not map_name:
        raise EvidenceError("objective_fixture.map_name must be a nonempty string")
    classes = _require(fixture, "objective_classes_present")
    if (
        not isinstance(classes, list)
        or not classes
        or not set(classes) <= set(GUIDE_CLASS_NAMES)
    ):
        raise EvidenceError(
            "objective_classes_present must be a nonempty subset of the eight"
            f" frozen classes {GUIDE_CLASS_NAMES!r}"
        )
    guide_vector = _require(fixture, "guide_vector")
    if not isinstance(guide_vector, list) or len(guide_vector) != GUIDE_DIM:
        raise EvidenceError(
            f"guide_vector must contain exactly {GUIDE_DIM} floats"
        )
    values = []
    for index, raw in enumerate(guide_vector):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EvidenceError(f"guide_vector[{index}] is not numeric")
        values.append(float(raw))
    if all(value == 0.0 for value in values):
        raise EvidenceError(
            "guide proof is universal-zero on an objective-bearing fixture;"
            " a placeholder guide field cannot admit bundle v3"
        )
    hot_classes = set()
    for candidate in range(GUIDE_CANDIDATES):
        row = values[
            candidate * GUIDE_CANDIDATE_DIM: (candidate + 1) * GUIDE_CANDIDATE_DIM
        ]
        onehot = row[7:15]
        hot = [index for index, bit in enumerate(onehot) if bit != 0.0]
        if len(hot) > 1 or any(bit not in (0.0, 1.0) for bit in onehot):
            raise EvidenceError(
                f"guide candidate {candidate} class encoding is not a valid one-hot"
            )
        if hot:
            hot_classes.add(GUIDE_CLASS_NAMES[hot[0]])
    if not hot_classes & set(classes):
        raise EvidenceError(
            "no guide candidate carries a declared objective class; the"
            " objective-bearing fixture proof is vacuous"
        )


def _check_runtime_epoch_fencing(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    clients = _require(evidence, "clients")
    if not isinstance(clients, list) or not clients:
        raise EvidenceError("clients must be a nonempty list of epoch records")
    epochs, maps, digests = set(), set(), set()
    for index, client in enumerate(clients):
        if not isinstance(client, Mapping):
            raise EvidenceError(f"clients[{index}] must be an object")
        client_id = _require(client, "client_id")
        if not isinstance(client_id, str) or not client_id:
            raise EvidenceError(f"clients[{index}].client_id must be nonempty")
        epochs.add(_require_int(client, "map_epoch"))
        map_name = _require(client, "map_name")
        if not isinstance(map_name, str) or not map_name:
            raise EvidenceError(f"clients[{index}].map_name must be nonempty")
        maps.add(map_name)
        digests.add(_require_digest(client, "atlas_sha256"))
    if len(epochs) != 1:
        raise EvidenceError(f"mixed client map epochs {sorted(epochs)} are fenced out")
    if len(maps) != 1:
        raise EvidenceError(f"mixed client maps {sorted(maps)} are fenced out")
    if len(digests) != 1:
        raise EvidenceError(
            f"mixed client Atlas digests {sorted(digests)} are fenced out"
        )
    _cross_check(context, "atlas_sha256", next(iter(digests)), "epoch fencing")
    if _require_int(evidence, "actions_dispatched_during_epoch_barrier") != 0:
        raise EvidenceError(
            "actions were dispatched through the map-epoch barrier"
        )
    if _require_int(evidence, "stale_epoch_transitions_admitted") != 0:
        raise EvidenceError("stale-epoch transitions were admitted as trainable")


def _check_b4_wire_generation(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    descriptor = _require(evidence, "descriptor")
    if not isinstance(descriptor, Mapping):
        raise EvidenceError("descriptor must be an object")
    atlas_sha256 = _require_digest(evidence, "atlas_sha256")
    _cross_check(context, "atlas_sha256", atlas_sha256, "B4 wire generation")
    violations = _require_int(evidence, "public_teacher_field_violations")
    try:
        runtime_evidence = adapt_b4_observation_descriptor(
            descriptor,
            atlas_sha256=atlas_sha256,
            teacher_field_violations=violations,
        )
        runtime = validate_runtime_evidence(
            runtime_evidence, expected_atlas_sha256=atlas_sha256
        )
    except LineageError as error:
        raise EvidenceError(str(error)) from error
    context["runtime_manifest_sha256"] = runtime.runtime_manifest_sha256
    facts = _require(evidence, "private_causal_facts")
    if not isinstance(facts, list):
        raise EvidenceError("private_causal_facts must be a list of field names")
    missing = sorted(set(REQUIRED_PRIVATE_CAUSAL_FACTS) - set(facts))
    if missing:
        raise EvidenceError(
            f"B4 private/debug channel is missing required causal facts: {missing}"
        )


def _check_causal_reward_admission(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    frames = _require(evidence, "frames")
    if not isinstance(frames, list) or not frames:
        raise EvidenceError("frames must be a nonempty list of B4 fact records")
    for index, raw in enumerate(frames):
        if not isinstance(raw, Mapping):
            raise EvidenceError(f"frames[{index}] must be an object")
        try:
            frame = CausalRewardFrame(**raw)
            frame.validate()
        except (RewardAdmissionError, TypeError, ValueError) as error:
            raise EvidenceError(
                f"frames[{index}] is not admissible as a CausalRewardFrame:"
                f" {error}"
            ) from error


def _check_lineage_attestation(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    if _require(evidence, "checkpoint_format") != CHECKPOINT_FORMAT:
        raise EvidenceError(
            f"checkpoint_format must be {CHECKPOINT_FORMAT!r}; raw or legacy"
            " checkpoints are not attestable"
        )
    if _require(evidence, "policy_generation") != POLICY_GENERATION:
        raise EvidenceError("checkpoint policy_generation is not the multires v1")
    if _require(evidence, "feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise EvidenceError("checkpoint feature schema digest mismatch")
    atlas_sha256 = _require_digest(evidence, "atlas_sha256")
    _cross_check(context, "atlas_sha256", atlas_sha256, "lineage attestation")
    runtime_digest = _require_digest(evidence, "runtime_manifest_sha256")
    _cross_check(
        context, "runtime_manifest_sha256", runtime_digest, "lineage attestation"
    )
    initialization = _require(evidence, "initialization")
    if initialization not in ALLOWED_INITIALIZATIONS:
        raise EvidenceError(
            f"initialization {initialization!r} is not an allowed fresh-lineage"
            f" origin {sorted(ALLOWED_INITIALIZATIONS)}"
        )
    legacy_paths = _require(evidence, "legacy_resume_paths")
    if not isinstance(legacy_paths, list) or legacy_paths:
        raise EvidenceError(
            f"legacy resume paths remain reachable: {legacy_paths!r}"
        )


def _check_legacy_selector_deactivation(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    _require_digest(evidence, "retirement_manifest_sha256")
    selectors = _require(evidence, "selectors")
    if not isinstance(selectors, list) or not selectors:
        raise EvidenceError(
            "selectors must be a nonempty inventory of operational selectors"
        )
    for index, selector in enumerate(selectors):
        if not isinstance(selector, Mapping):
            raise EvidenceError(f"selectors[{index}] must be an object")
        name = _require(selector, "name")
        legacy = _require(selector, "legacy")
        active = _require(selector, "active")
        if not isinstance(legacy, bool) or not isinstance(active, bool):
            raise EvidenceError(f"selectors[{index}] legacy/active must be booleans")
        if legacy and active:
            raise EvidenceError(
                f"legacy selector {name!r} is still active; the retirement"
                " manifest is not satisfied"
            )
    wire_versions = _require(evidence, "active_wire_versions")
    if not isinstance(wire_versions, list) or not wire_versions:
        raise EvidenceError("active_wire_versions must be a nonempty list")
    if any(version != B4_CLIENT_WIRE_VERSION for version in wire_versions):
        raise EvidenceError(
            f"active wire versions {wire_versions!r} include a non-B4 generation"
            f" (legacy is {LEGACY_CLIENT_WIRE_VERSION})"
        )
    magics = _require(evidence, "active_observation_magics")
    if not isinstance(magics, list) or not magics:
        raise EvidenceError("active_observation_magics must be a nonempty list")
    if any(magic != B4_OBSERVATION_MAGIC for magic in magics):
        raise EvidenceError(
            f"active observation magics {magics!r} include a non-B4 magic"
            f" (legacy is {LEGACY_OBSERVATION_MAGIC:#x})"
        )
    schemas = _require(evidence, "active_rollout_schemas")
    if not isinstance(schemas, list) or not schemas:
        raise EvidenceError("active_rollout_schemas must be a nonempty list")
    if any(schema != B4_ROLLOUT_SCHEMA for schema in schemas):
        raise EvidenceError(
            f"active rollout schemas {schemas!r} include a non-B4 schema"
            f" (legacy is {LEGACY_ROLLOUT_SCHEMA!r})"
        )


def _check_deterministic_transitions(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    count = _require_int(evidence, "transition_count")
    if count != DETERMINISTIC_TRANSITION_COUNT:
        raise EvidenceError(
            f"transition_count={count} is not the required"
            f" {DETERMINISTIC_TRANSITION_COUNT}-transition proof"
        )
    runs = _require(evidence, "runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise EvidenceError("at least two independent runs are required")
    launch_ids, stacks, seeds, game_seeds, digests = set(), set(), set(), set(), set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise EvidenceError(f"runs[{index}] must be an object")
        launch_id = _require(run, "launch_id")
        if not isinstance(launch_id, str) or not launch_id:
            raise EvidenceError(f"runs[{index}].launch_id must be nonempty")
        launch_ids.add(launch_id)
        stack = _require(run, "stack")
        if stack != DETERMINISTIC_STACK:
            raise EvidenceError(
                f"runs[{index}].stack={stack!r} is not {DETERMINISTIC_STACK!r}"
            )
        stacks.add(stack)
        if _require(run, "fresh_subprocess") is not True:
            raise EvidenceError(f"runs[{index}] is not a fresh subprocess launch")
        if _require(run, "collector") != DETERMINISTIC_COLLECTOR:
            raise EvidenceError(f"runs[{index}] did not use the multires collector")
        if _require(run, "spatial_provider") != DETERMINISTIC_PROVIDER:
            raise EvidenceError(f"runs[{index}] did not use the Rust Atlas provider")
        lattice = _require(run, "lattice_crate")
        if lattice not in DETERMINISTIC_LATTICE_NAMES:
            raise EvidenceError(
                f"runs[{index}] did not attest the Rust lattice boundary"
            )
        for counter in (
            "partial_admissions", "stale_admissions", "resync_admissions"
        ):
            if _require_int(run, counter) != 0:
                raise EvidenceError(f"runs[{index}] recorded nonzero {counter}")
        seeds.add(_require_int(run, "seed"))
        game_seeds.add(_require_int(run, "game_seed"))
        if _require_int(run, "transition_count") != DETERMINISTIC_TRANSITION_COUNT:
            raise EvidenceError(
                f"runs[{index}] did not complete the full"
                f" {DETERMINISTIC_TRANSITION_COUNT}-transition trajectory"
            )
        digests.add(_require_digest(run, "trajectory_sha256"))
    if len(launch_ids) != len(runs):
        raise EvidenceError(
            "deterministic proof reused a launch identity instead of two fresh runs"
        )
    if len(stacks) != 1 or len(seeds) != 1 or len(game_seeds) != 1:
        raise EvidenceError(
            "same-seed runs do not describe the same complete stack and seeds"
        )
    if len(digests) != 1:
        raise EvidenceError(
            f"trajectory digests diverge across runs: {sorted(digests)}"
        )
    divergence = _require(evidence, "divergence_run")
    if not isinstance(divergence, Mapping):
        raise EvidenceError("divergence_run must be an object")
    divergence_launch = _require(divergence, "launch_id")
    if not isinstance(divergence_launch, str) or not divergence_launch:
        raise EvidenceError("divergence_run.launch_id must be nonempty")
    if divergence_launch in launch_ids:
        raise EvidenceError("divergence run reused a same-seed launch identity")
    if _require(divergence, "fresh_subprocess") is not True:
        raise EvidenceError("divergence run is not a fresh subprocess launch")
    if _require(divergence, "stack") != DETERMINISTIC_STACK:
        raise EvidenceError("divergence run did not use the same complete stack")
    if _require(divergence, "collector") != DETERMINISTIC_COLLECTOR:
        raise EvidenceError("divergence run did not use the multires collector")
    if _require(divergence, "spatial_provider") != DETERMINISTIC_PROVIDER:
        raise EvidenceError("divergence run did not use the Rust Atlas provider")
    if _require(divergence, "lattice_crate") not in DETERMINISTIC_LATTICE_NAMES:
        raise EvidenceError("divergence run did not attest the Rust lattice boundary")
    for counter in (
        "partial_admissions", "stale_admissions", "resync_admissions"
    ):
        if _require_int(divergence, counter) != 0:
            raise EvidenceError(f"divergence run recorded nonzero {counter}")
    if _require_int(divergence, "transition_count") != DETERMINISTIC_TRANSITION_COUNT:
        raise EvidenceError("divergence run did not complete exactly 500 transitions")
    if _require_int(divergence, "seed") != next(iter(seeds)):
        raise EvidenceError("divergence run changed the policy seed")
    if _require_int(divergence, "game_seed") in game_seeds:
        raise EvidenceError("divergence run did not change the game seed")
    divergence_digest = _require_digest(divergence, "trajectory_sha256")
    if divergence_digest in digests:
        raise EvidenceError("different game seed did not change the trajectory digest")


def _check_wsl_b6_campaign(
    evidence: Mapping[str, Any], context: dict[str, Any]
) -> None:
    host_platform = _require(evidence, "host_platform")
    if not isinstance(host_platform, str) or "wsl" not in host_platform.lower():
        raise EvidenceError(
            f"host_platform={host_platform!r}: B6 evidence must come from the"
            " isolated WSL staging host"
        )
    for gate in ("g0_identity_pass", "g1_transport_pass"):
        if _require(evidence, gate) is not True:
            raise EvidenceError(f"B6 precondition {gate} did not pass")
    atlas_sha256 = _require_digest(evidence, "atlas_sha256")
    _cross_check(context, "atlas_sha256", atlas_sha256, "B6 campaign")
    accepted = _require_int(evidence, "accepted_transitions")
    if accepted < MIN_ACCEPTED_TRANSITIONS_B6:
        raise EvidenceError(
            f"accepted_transitions={accepted} is below the"
            f" {MIN_ACCEPTED_TRANSITIONS_B6} G1 minimum"
        )
    if _require_int(evidence, "failed_rounds") != 0:
        raise EvidenceError("B6 campaign recorded failed rounds")
    if _require_int(evidence, "echo_timeouts") != 0:
        raise EvidenceError("B6 campaign recorded echo timeouts")
    p99 = _require_number(evidence, "feature_assembly_p99_ms")
    if p99 <= 0.0:
        raise EvidenceError(
            "feature_assembly_p99_ms is not a positive measurement; a zero"
            " latency is placeholder evidence"
        )
    if p99 >= FEATURE_ASSEMBLY_P99_BUDGET_MS:
        raise EvidenceError(
            f"feature_assembly_p99_ms={p99} breaches the"
            f" {FEATURE_ASSEMBLY_P99_BUDGET_MS} ms budget"
        )
    resident = _require_int(evidence, "resident_atlas_bytes")
    if not 0 < resident <= RESIDENT_ATLAS_BUDGET_BYTES:
        raise EvidenceError(
            f"resident_atlas_bytes={resident} is outside (0,"
            f" {RESIDENT_ATLAS_BUDGET_BYTES}]"
        )
    dyn_bytes = _require_int(evidence, "dyn_payload_bytes_combined")
    if not 0 < dyn_bytes <= DYN_PAYLOAD_HARD_LIMIT_BYTES:
        raise EvidenceError(
            f"dyn_payload_bytes_combined={dyn_bytes} is outside (0,"
            f" {DYN_PAYLOAD_HARD_LIMIT_BYTES}]"
        )
    if _require(evidence, "public_services_modified") is not False:
        raise EvidenceError("a public/teacher service was modified during B6")
    if _require_int(evidence, "orphan_processes_after_teardown") != 0:
        raise EvidenceError("staging teardown left orphan client/server processes")


_GATE_CHECKS: dict[str, Callable[[Mapping[str, Any], dict[str, Any]], None]] = {
    "feature_action_contract": _check_feature_action_contract,
    "bundle_v3_atlas": _check_bundle_v3_atlas,
    "runtime_epoch_fencing": _check_runtime_epoch_fencing,
    "b4_wire_generation": _check_b4_wire_generation,
    "causal_reward_admission": _check_causal_reward_admission,
    "lineage_attestation": _check_lineage_attestation,
    "legacy_selector_deactivation": _check_legacy_selector_deactivation,
    "deterministic_transitions": _check_deterministic_transitions,
    "wsl_b6_campaign": _check_wsl_b6_campaign,
}

assert tuple(_GATE_CHECKS) == GATE_ORDER


# --- envelope evaluation ----------------------------------------------------


def _load_evidence(base: Path, raw_path: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError("evidence path must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise EvidenceError(f"evidence file {path.name!r} does not exist")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"evidence {path.name!r} is not valid JSON: {error}")
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence {path.name!r} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def run_gates(envelope_path: Path) -> dict[str, Any]:
    """Evaluate every gate and return the canonical report structure."""
    try:
        envelope_bytes = envelope_path.read_bytes()
        envelope = json.loads(envelope_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        envelope = None
        envelope_error = f"evidence envelope is unreadable: {error}"
        envelope_bytes = b""
    else:
        envelope_error = None
        if not isinstance(envelope, dict) or envelope.get("schema") != ENVELOPE_SCHEMA:
            envelope_error = (
                f"evidence envelope must declare schema={ENVELOPE_SCHEMA!r}"
            )
    evidence_map = (
        envelope.get("evidence") if isinstance(envelope, dict) else None
    )
    if envelope_error is None and not isinstance(evidence_map, dict):
        envelope_error = "evidence envelope has no 'evidence' object"

    context: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    base = envelope_path.resolve().parent
    for rank, gate in enumerate(GATE_ORDER, start=1):
        reasons: list[str] = []
        evidence_sha256: Optional[str] = None
        if envelope_error is not None:
            reasons.append(envelope_error)
        else:
            try:
                evidence, evidence_sha256 = _load_evidence(
                    base, evidence_map.get(gate)
                )
                _reject_placeholders(evidence, gate)
                _GATE_CHECKS[gate](evidence, context)
            except EvidenceError as error:
                reasons.append(str(error))
            except Exception as error:  # fail closed on anything unexpected
                reasons.append(f"unexpected {type(error).__name__}: {error}")
        results.append({
            "rank": rank,
            "gate": gate,
            "status": "pass" if not reasons else "fail",
            "evidence_sha256": evidence_sha256,
            "reasons": reasons,
        })

    failed = [entry["gate"] for entry in results if entry["status"] == "fail"]
    body = {
        "schema": REPORT_SCHEMA,
        "tool": TOOL_NAME,
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "envelope_sha256": hashlib.sha256(envelope_bytes).hexdigest(),
        "gates": results,
        "failed_gates": failed,
        "overall": "pass" if not failed else "fail",
    }
    body["report_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(report)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evidence", required=True, type=Path,
        help="path to the multires integration evidence envelope JSON",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="optional path to write the canonical report JSON",
    )
    args = parser.parse_args(argv)
    report = run_gates(args.evidence)
    payload = canonical_report_bytes(report)
    if args.out is not None:
        args.out.write_bytes(payload + b"\n")
    sys.stdout.write(payload.decode("utf-8") + "\n")
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
