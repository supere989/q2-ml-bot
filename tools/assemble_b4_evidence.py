#!/usr/bin/env python3
"""Assemble immutable B4 evidence from one real-network qualification.

This is deliberately a consumer, not another test runner.  It replays the
sealed frame-barrier qualification, independently attests the exact clean bot,
client, and game source trees, cross-checks the three protocol descriptions,
and derives the four evidence documents consumed by the multires integration
gate.  Synthetic qualification, legacy wire values, dirty source, unsealed
runtime input, and non-canonical JSON all fail before an output directory is
published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from harness.multires_reward import CausalRewardFrame, RewardAdmissionError
from harness.runtime_attestation import verify_runtime_manifest
from harness import client_protocol as public_wire
from tools import qualify_network_client_frame_barrier as barrier
from tools import assemble_b2_gate as b2_gate
from tools.assemble_b3_gate import B3GateError, validate_b3_gate


SCHEMA = "q2-multires-b4-evidence-v1"
FEATURE_SCHEMA = "q2-multires-b4-feature-action-contract-v1"
EPOCH_SCHEMA = "q2-multires-b4-runtime-epoch-fencing-v1"
WIRE_SCHEMA = "q2-multires-b4-wire-generation-v1"
REWARD_SCHEMA = "q2-multires-b4-causal-reward-admission-v1"

DOCUMENT_NAMES = {
    "feature_action_contract": "feature-action-contract.json",
    "runtime_epoch_fencing": "runtime-epoch-fencing.json",
    "b4_wire_generation": "b4-wire-generation.json",
    "causal_reward_admission": "causal-reward-admission.json",
}
B4_GATE_KEYS = frozenset({
    "clean_exact_sources",
    "atomic_wire_generation",
    "real_network_barrier_replayed",
    "runtime_and_binary_seals_bound",
    "epoch_and_stale_fencing_proven",
    "public_teacher_violations_zero",
    "causal_reward_frames_admissible",
    "active_authority_b3_predecessor_bound",
})
B4_AGGREGATE_KEYS = frozenset({
    "schema", "milestone", "status", "atlas_sha256", "predecessor",
    "source_repositories", "source_closure_sha256", "normative_documents",
    "runtime_binaries", "runtime_manifest", "network_qualification",
    "scenario_proofs", "public_privilege_proof", "documents",
    "component_evidence", "gate", "evidence_sha256",
})
NORMATIVE_PATHS = (
    "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md",
    "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md",
)
PRIVATE_CAUSAL_FACTS = (
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
PUBLIC_AUDIT_COUNTERS = (
    "datagrams_seen",
    "public_packets_decoded",
    "routed_packets_accepted",
    "malformed_packets_rejected",
    "foreign_client_packets_rejected",
    "stale_packets_rejected",
    "teacher_packets_detected",
)
EXPECTED_DESCRIPTOR = {
    "protocol_generation": 2,
    "factual_dim": 198,
    "dyn_dim": 24,
    "recovery_dim": 16,
    "objective_count": 4,
    "objective_dim": 15,
    "objective_total_dim": 60,
    "total_dim": 298,
    "feature_schema_sha256": "bac38d8d4acffebbc02701f295710a9b8a0c434134627534a2a04677637adb3f",
    "observation_packet_bytes": 1056,
    "action_packet_bytes": 28,
    "observation_magic": 0x514D324F,
    "action_magic": 0x514D3241,
    "client_wire_version": 8,
    "teacher_version": 4,
    "rollout_telemetry_schema": "ppo-telemetry-multires-v1",
    "logical_action_dim": 8,
    "action_cardinalities": {
        "vertical_intent": 3,
        "fire": 2,
        "hook": 4,
        "weapon": 10,
    },
    "teacher_privileged_packing": "physically-separate-qm3c-v2",
    "causal_magic": 0x514D3343,
    "causal_version": 2,
    "causal_packet_bytes": 80,
}
EXPECTED_CONTRACT = {
    "policy_generation": "multires-atlas-policy-v1",
    "feature_schema": "multires-atlas-features-298-v1",
    "feature_schema_sha256": "bac38d8d4acffebbc02701f295710a9b8a0c434134627534a2a04677637adb3f",
    "observation_dim": 298,
    "factual_dim": 198,
    "dyn_dim": 24,
    "recovery_dim": 16,
    "guide_dim": 60,
    "logical_action_dim": 8,
    "posture_classes": 3,
    "action_cardinalities": {
        "vertical_intent": 3,
        "fire": 2,
        "hook": 4,
        "weapon": 10,
    },
}
_SHA_CHARS = frozenset("0123456789abcdef")
_C_DEFINE = re.compile(
    r"^\s*#\s*define\s+([A-Z][A-Z0-9_]*)\s+"
    r"(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*\b",
    re.MULTILINE,
)
_STATIC_ASSERT = re.compile(
    r"_Static_assert\s*\(\s*sizeof\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
    r"\s*==\s*([0-9]+)[uUlL]*\s*,"
)
_STRUCT = re.compile(
    r"typedef\s+struct\s*\{(?P<body>.*?)\}\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
    re.DOTALL,
)
_UINT32_FIELD = re.compile(r"\buint32_t\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")


class B4EvidenceError(RuntimeError):
    """The supplied inputs cannot prove the B4 gate."""


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise B4EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise B4EvidenceError(f"non-finite JSON constant {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B4EvidenceError("value is not canonical JSON") from error


def _manifest_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        ) + b"\n"
    except (TypeError, ValueError) as error:
        raise B4EvidenceError("runtime manifest is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in _SHA_CHARS for character in value)
    )


def _exact_regular(path: Path, label: str) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise B4EvidenceError(f"{label} path must be absolute")
    if source.is_symlink() or not source.is_file():
        raise B4EvidenceError(f"{label} must be an exact regular file")
    return source


def _load_json(path: Path, label: str, *, style: str) -> dict[str, Any]:
    source = _exact_regular(path, label)
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise B4EvidenceError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise B4EvidenceError(f"{label} must be a JSON object")
    expected = (
        _canonical_bytes(value) + b"\n" if style == "compact" else _manifest_bytes(value)
    )
    if data != expected:
        raise B4EvidenceError(f"{label} is not in its canonical {style} encoding")
    return value


def _file_record(path: Path, name: str) -> dict[str, Any]:
    source = _exact_regular(path, name)
    data = source.read_bytes()
    return {"name": name, "sha256": _sha256_bytes(data), "size": len(data)}


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if "evidence_sha256" in value:
        raise B4EvidenceError("evidence must be unsealed before sealing")
    payload = dict(value)
    return {**payload, "evidence_sha256": _sha256_bytes(_canonical_bytes(payload))}


def _verify_seal(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    digest = payload.pop("evidence_sha256", None)
    if not _valid_sha256(digest) or digest != _sha256_bytes(_canonical_bytes(payload)):
        raise B4EvidenceError(f"{label} evidence seal differs")


def _git_output(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise B4EvidenceError(f"source repository cannot be attested: {repo}") from error


def _git_identity(path: Path) -> dict[str, Any]:
    repo = Path(path).expanduser()
    if not repo.is_absolute() or repo.is_symlink() or not repo.is_dir():
        raise B4EvidenceError(f"source repository path is not exact: {path}")
    marker = repo / ".git"
    if marker.is_symlink() or not marker.exists():
        raise B4EvidenceError(f"source repository has no exact git metadata: {repo}")
    status = _git_output(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise B4EvidenceError(f"source repository is dirty: {repo}")
    commit = _git_output(repo, "rev-parse", "HEAD")
    tree = _git_output(repo, "rev-parse", "HEAD^{tree}")
    if (
        len(commit) not in (40, 64)
        or any(character not in _SHA_CHARS for character in commit)
        or len(tree) not in (40, 64)
        or any(character not in _SHA_CHARS for character in tree)
    ):
        raise B4EvidenceError(f"source repository identity is malformed: {repo}")
    return {"commit": commit, "tree": tree, "clean": True}


def _describe_bot_source(repo: Path) -> dict[str, Any]:
    script = r"""
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
from harness.runtime_attestation import describe_observation
from harness.multires_contract import (ACTION_DIM, DYN_DIM, FACTUAL_DIM,
    FEATURE_SCHEMA, FEATURE_SCHEMA_SHA256, GUIDE_DIM, OBS_DIM,
    POLICY_GENERATION, POSTURE_CLASSES, RECOVERY_DIM)
from harness.causal_protocol import CAUSAL_FIELD_NAMES
descriptor = describe_observation({})
contract = {
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
    "action_cardinalities": descriptor["action_cardinalities"],
}
print(json.dumps({"descriptor": descriptor, "contract": contract,
                  "causal_fields": list(CAUSAL_FIELD_NAMES)},
                 sort_keys=True, separators=(",", ":"), allow_nan=False))
"""
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-c", script, os.fspath(repo)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env={"PATH": os.environ.get("PATH", "")},
        )
        value = json.loads(
            process.stdout,
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
        )
    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = " ".join(error.stderr.split())[:240]
        raise B4EvidenceError(f"bot source descriptor failed closed: {detail}") from error
    if not isinstance(value, dict):
        raise B4EvidenceError("bot source descriptor is not an object")
    return value


def _c_constants(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    source = _exact_regular(path, "C protocol source").read_text(encoding="utf-8")
    define_rows = _C_DEFINE.findall(source)
    duplicate_defines = sorted({
        name for name, unused in define_rows
        if sum(other == name for other, ignored in define_rows) != 1
    })
    if duplicate_defines:
        raise B4EvidenceError(
            f"C protocol source has duplicate numeric defines: {duplicate_defines}"
        )
    defines = {name: int(raw, 0) for name, raw in define_rows}
    sizes = {name: int(raw) for name, raw in _STATIC_ASSERT.findall(source)}
    return defines, sizes


def _c_struct_uint32_fields(path: Path, struct_name: str) -> list[str]:
    source = _exact_regular(path, "C protocol source").read_text(encoding="utf-8")
    matches = [match.group("body") for match in _STRUCT.finditer(source) if match.group("name") == struct_name]
    if len(matches) != 1:
        raise B4EvidenceError(
            f"C protocol source does not contain exactly one {struct_name}"
        )
    return _UINT32_FIELD.findall(matches[0])


def _c_struct_body(path: Path, struct_name: str) -> str:
    source = _exact_regular(path, "C protocol source").read_text(encoding="utf-8")
    matches = [
        match.group("body")
        for match in _STRUCT.finditer(source)
        if match.group("name") == struct_name
    ]
    if len(matches) != 1:
        raise B4EvidenceError(
            f"C protocol source does not contain exactly one {struct_name}"
        )
    return re.sub(r"\s+", " ", matches[0]).strip()


def _require_source_fragments(path: Path, fragments: Sequence[str], label: str) -> None:
    source = _exact_regular(path, label).read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in source]
    if missing:
        raise B4EvidenceError(f"{label} omits privilege-path fragments: {missing}")


def _require_matches(found: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    mismatches = {
        name: (found.get(name), wanted)
        for name, wanted in expected.items()
        if found.get(name) != wanted
    }
    if mismatches:
        detail = "; ".join(
            f"{name}={actual!r} expected {wanted!r}"
            for name, (actual, wanted) in sorted(mismatches.items())
        )
        raise B4EvidenceError(f"{label} differs: {detail}")


def _source_contract(bot_repo: Path, client_repo: Path, game_repo: Path) -> dict[str, Any]:
    bot = _describe_bot_source(bot_repo)
    descriptor = bot.get("descriptor")
    contract = bot.get("contract")
    causal_fields = bot.get("causal_fields")
    if not isinstance(descriptor, Mapping) or not isinstance(contract, Mapping):
        raise B4EvidenceError("bot source omits descriptor or policy contract")
    _require_matches(descriptor, EXPECTED_DESCRIPTOR, "bot B4 descriptor")
    _require_matches(contract, EXPECTED_CONTRACT, "bot feature/action contract")
    if not _valid_sha256(contract.get("feature_schema_sha256")):
        raise B4EvidenceError("bot feature schema digest is malformed")
    if not isinstance(causal_fields, list):
        raise B4EvidenceError("bot causal field vocabulary is missing")
    mapped_causal = ["causal_flags" if name == "flags" else name for name in causal_fields]
    missing = sorted(set(PRIVATE_CAUSAL_FACTS) - set(mapped_causal))
    if missing:
        raise B4EvidenceError(f"bot causal channel omits fields: {missing}")

    game_bridge, game_bridge_sizes = _c_constants(game_repo / "ml_bridge.h")
    game_wire, game_sizes = _c_constants(game_repo / "ml_client_wire.h")
    client, client_sizes = _c_constants(
        client_repo / "src" / "client" / "cl_ml_harness.c"
    )
    expected_game_bridge = {
        "ML_PROTOCOL_GENERATION": descriptor["protocol_generation"],
        "ML_OBS_MAGIC": descriptor["observation_magic"],
        "ML_ACT_MAGIC": descriptor["action_magic"],
        "ML_TEACHER_MAGIC": public_wire.ML_TEACHER_MAGIC,
        "ML_TEACHER_VERSION": descriptor["teacher_version"],
        "ML_CAUSAL_MAGIC": descriptor["causal_magic"],
        "ML_CAUSAL_VERSION": descriptor["causal_version"],
        "ML_VERTICAL_COUNT": descriptor["action_cardinalities"]["vertical_intent"],
    }
    expected_game_wire = {
        "ML_CLIENT_WIRE_VERSION": descriptor["client_wire_version"],
        "ML_CLIENT_FRAME_BARRIER_VERSION": 1,
        "ML_CLIENT_FRAME_BARRIER_CAPABILITY": 1,
    }
    expected_client = {
        "ML_CLIENT_WIRE_VERSION": descriptor["client_wire_version"],
        "ML_CLIENT_FRAME_BARRIER_VERSION": 1,
        "ML_CLIENT_FRAME_BARRIER_CAPABILITY": 1,
        "ML_OBSERVATION_MAGIC": descriptor["observation_magic"],
        "ML_ACTION_MAGIC": descriptor["action_magic"],
        "ML_OBSERVATION_SIZE": descriptor["observation_packet_bytes"],
        "ML_ACTION_SIZE": descriptor["action_packet_bytes"],
        "ML_CAUSAL_MAGIC": descriptor["causal_magic"],
        "ML_CAUSAL_VERSION": descriptor["causal_version"],
        "ML_CAUSAL_SIZE": descriptor["causal_packet_bytes"],
        "ML_CLIENT_TELEMETRY_SIZE": 1248,
        "ML_VERTICAL_COUNT": descriptor["action_cardinalities"]["vertical_intent"],
    }
    _require_matches(game_bridge, expected_game_bridge, "game bridge ABI")
    _require_matches(game_wire, expected_game_wire, "game client wire ABI")
    _require_matches(client, expected_client, "client wire ABI")
    _require_matches(game_sizes, {
        "ml_obs_t": descriptor["observation_packet_bytes"],
        "ml_action_t": descriptor["action_packet_bytes"],
        "ml_causal_telemetry_t": descriptor["causal_packet_bytes"],
        "ml_client_register_t": 148,
        "ml_client_ack_t": 100,
        "ml_client_telemetry_t": 1248,
    }, "game POD sizes")
    _require_matches(game_bridge_sizes, {
        "ml_obs_t": descriptor["observation_packet_bytes"],
        "ml_action_t": descriptor["action_packet_bytes"],
        "ml_causal_telemetry_t": descriptor["causal_packet_bytes"],
        "ml_teacher_sample_t": public_wire.TEACHER_SAMPLE_SIZE,
    }, "game bridge/teacher POD sizes")
    _require_matches(client_sizes, {
        "ml_client_register_t": 148,
        "ml_client_ack_t": 100,
        "ml_action_t": 28,
    }, "client POD sizes")
    if "ML_TEACHER_MAGIC" in client or "ML_TEACHER_VERSION" in client:
        raise B4EvidenceError("public Yamagi client contains teacher wire identity")

    public_struct = _c_struct_body(
        game_repo / "ml_client_wire.h", "ml_client_telemetry_t"
    )
    teacher_struct = _c_struct_body(game_repo / "ml_bridge.h", "ml_teacher_sample_t")
    if (
        "ml_obs_t obs;" not in public_struct
        or "ml_causal_telemetry_t causal;" not in public_struct
        or "ml_action_t action;" in public_struct
        or "ml_teacher_sample_t" in public_struct
    ):
        raise B4EvidenceError("public telemetry POD does not exclude teacher payload")
    if (
        "ml_obs_t obs;" not in teacher_struct
        or "ml_causal_telemetry_t causal;" not in teacher_struct
        or "ml_action_t action;" not in teacher_struct
    ):
        raise B4EvidenceError("teacher POD is not a physically separate sample path")
    _require_source_fragments(
        game_repo / "ml_client_telemetry.c",
        (
            "ml_client_telemetry_t packet;",
            "packet.magic = ML_CLIENT_TELEM_MAGIC;",
            "ML_PackCausalTelemetry(ent, &packet.causal, 0);",
            "sendto(ml_client_fd, &packet, sizeof(packet), MSG_DONTWAIT",
        ),
        "game public telemetry path",
    )
    _require_source_fragments(
        game_repo / "ml_bridge.c",
        (
            "ml_teacher_sample_t sample;",
            "sample.magic = ML_TEACHER_MAGIC;",
            "ML_PackCausalTelemetry(ent, &sample.causal, 1);",
            "sendto(g_teacher_fd, &sample, sizeof(sample), MSG_DONTWAIT",
        ),
        "game teacher telemetry path",
    )
    _require_source_fragments(
        bot_repo / "tools" / "qualify_network_client_frame_barrier.py",
        ('"ml_teacher_enabled": "0",',),
        "public network qualifier",
    )
    yamagi_source = _exact_regular(
        client_repo / "src" / "client" / "cl_ml_harness.c",
        "public Yamagi conduit",
    ).read_text(encoding="utf-8")
    if "ML_TEACHER_MAGIC" in yamagi_source or "ml_teacher_sample_t" in yamagi_source:
        raise B4EvidenceError("public Yamagi conduit contains teacher packing path")
    if public_wire.CLIENT_TELEMETRY_SIZE == public_wire.TEACHER_SAMPLE_SIZE:
        raise B4EvidenceError("public and teacher packet sizes are not distinct")
    causal_wire_fields = list(causal_fields)
    for label, path in (
        ("game", game_repo / "ml_bridge.h"),
        ("client", client_repo / "src" / "client" / "cl_ml_harness.c"),
    ):
        fields = _c_struct_uint32_fields(path, "ml_causal_telemetry_t")
        if fields != causal_wire_fields:
            raise B4EvidenceError(
                f"{label} causal POD field order differs: {fields!r}"
            )
    return {
        "descriptor": dict(descriptor),
        "contract": dict(contract),
        "private_causal_facts": list(PRIVATE_CAUSAL_FACTS),
        "source_abi": {
            "game_bridge": expected_game_bridge,
            "game_wire": {**expected_game_wire, "telemetry_bytes": 1248, "ack_bytes": 100},
            "client_wire": expected_client,
        },
        "privilege_abi": {
            "public": {
                "magic": public_wire.ML_CLIENT_TELEM_MAGIC,
                "packet_bytes": public_wire.CLIENT_TELEMETRY_SIZE,
                "pod": "ml_client_telemetry_t",
                "packing_path": "ml_client_telemetry.c:ML_ClientTelemetryFrame",
                "fields": ["public_header", "ml_obs_t", "ml_causal_telemetry_t"],
                "teacher_action_tail_present": False,
            },
            "teacher": {
                "magic": public_wire.ML_TEACHER_MAGIC,
                "version": public_wire.ML_TEACHER_VERSION,
                "packet_bytes": public_wire.TEACHER_SAMPLE_SIZE,
                "pod": "ml_teacher_sample_t",
                "packing_path": "ml_bridge.c:ML_TeacherSend",
                "fields": [
                    "teacher_header", "ml_obs_t", "ml_causal_telemetry_t",
                    "ml_action_t",
                ],
            },
            "yamagi_teacher_identity_present": False,
            "qualification_teacher_enabled": False,
            "public_and_teacher_magic_distinct": (
                public_wire.ML_CLIENT_TELEM_MAGIC != public_wire.ML_TEACHER_MAGIC
            ),
            "public_and_teacher_size_distinct": (
                public_wire.CLIENT_TELEMETRY_SIZE != public_wire.TEACHER_SAMPLE_SIZE
            ),
        },
    }


def _validate_qualification(
    qualification: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    _verify_seal(qualification, "network qualification")
    required_keys = {
        "schema", "passed", "mode", "protocol_version", "test_mode",
        "non_admissible_for_training", "runtime_manifest_sha256",
        "execution_evidence_sha256", "runtime_closure_sha256",
        "execution_evidence", "evidence_sha256",
    }
    if set(qualification) != required_keys:
        raise B4EvidenceError("network qualification outer schema differs")
    if (
        qualification.get("schema") != barrier.SCHEMA
        or qualification.get("passed") is not True
        or qualification.get("mode") != barrier.MODE
        or qualification.get("protocol_version") != barrier.PROTOCOL_VERSION
        or qualification.get("test_mode") is not False
        or qualification.get("non_admissible_for_training") is not True
    ):
        raise B4EvidenceError("only a finalized real-network qualification is admissible")
    execution = qualification.get("execution_evidence")
    if not isinstance(execution, Mapping):
        raise B4EvidenceError("network qualification omits execution evidence")
    try:
        validated = barrier._validate_execution_evidence(execution, path)
    except barrier.QualificationError as error:
        raise B4EvidenceError(f"real-network execution replay failed: {error}") from error
    if (
        validated.get("test_mode") is not False
        or validated.get("full_network_executed") is not True
        or qualification["execution_evidence_sha256"]
        != validated.get("execution_evidence_sha256")
    ):
        raise B4EvidenceError("real-network execution identity differs")
    closure = {
        "runtime_manifest_sha256": qualification["runtime_manifest_sha256"],
        "execution_evidence_sha256": qualification["execution_evidence_sha256"],
    }
    if qualification["runtime_closure_sha256"] != _sha256_bytes(_canonical_bytes(closure)):
        raise B4EvidenceError("qualification runtime closure differs")
    return validated


def _validate_runtime_manifest(
    manifest: Mapping[str, Any], qualification: Mapping[str, Any],
    execution: Mapping[str, Any], atlas_sha256: str,
) -> dict[str, Any]:
    verified = verify_runtime_manifest(manifest)
    if not verified.valid or not _valid_sha256(verified.digest):
        raise B4EvidenceError("runtime manifest is unsealed: " + "; ".join(verified.errors))
    if qualification.get("runtime_manifest_sha256") != verified.digest:
        raise B4EvidenceError("qualification/runtime manifest seal differs")
    try:
        semantic = manifest["semantic"]
        runtime = semantic["runtime_config"]
        artifacts = semantic["artifacts"]
        binaries = execution["runtime_binaries"]
    except (KeyError, TypeError) as error:
        raise B4EvidenceError("runtime manifest omits B4 closure") from error
    if runtime.get("network_barrier_execution_evidence_sha256") != execution.get(
        "execution_evidence_sha256"
    ):
        raise B4EvidenceError("runtime manifest execution binding differs")
    if runtime.get("expected_atlas_sha256") != atlas_sha256:
        raise B4EvidenceError("runtime manifest Atlas binding differs")
    for name, artifact_name, binary_name in (
        ("q2ded", "q2ded", "q2ded"),
        ("game module", "game_module", "game_module"),
    ):
        artifact = artifacts.get(artifact_name)
        binary = binaries.get(binary_name)
        if not isinstance(artifact, Mapping) or not isinstance(binary, Mapping) or (
            artifact.get("sha256"), artifact.get("size")
        ) != (binary.get("sha256"), binary.get("size")):
            raise B4EvidenceError(f"runtime manifest {name} binary differs")
    client = binaries.get("client_binary", {})
    if (
        runtime.get("client_binary_sha256"), runtime.get("client_binary_size")
    ) != (client.get("sha256"), client.get("size")):
        raise B4EvidenceError("runtime manifest client binary differs")
    return {"digest": verified.digest, "runtime_config": dict(runtime)}


def _scenario_path(
    execution: Mapping[str, Any], qualification_path: Path, name: str
) -> tuple[Path, Mapping[str, Any]]:
    records = execution.get("scenario_evidence")
    matches = [record for record in records or () if isinstance(record, Mapping) and record.get("scenario") == name]
    if len(matches) != 1:
        raise B4EvidenceError(f"execution does not contain exactly one {name} scenario")
    record = matches[0]
    relative = record.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise B4EvidenceError(f"scenario {name} path is unsafe")
    parent = qualification_path.resolve().parent
    candidate = parent / relative
    try:
        candidate.resolve(strict=True).relative_to(parent)
    except (OSError, ValueError) as error:
        raise B4EvidenceError(f"scenario {name} escapes evidence root") from error
    return candidate, record


def _load_scenario(
    execution: Mapping[str, Any], qualification_path: Path, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, record = _scenario_path(execution, qualification_path, name)
    raw = _load_json(path, f"scenario {name}", style="compact")
    _verify_seal(raw, f"scenario {name}")
    file_identity = _file_record(path, f"scenario-{name}")
    if (record.get("sha256"), record.get("size")) != (
        file_identity["sha256"], file_identity["size"]
    ) or record.get("raw_evidence_sha256") != raw.get("evidence_sha256"):
        raise B4EvidenceError(f"scenario {name} root binding differs")
    return raw, {**file_identity, "raw_evidence_sha256": raw["evidence_sha256"]}


def _validate_all_scenario_encodings(
    execution: Mapping[str, Any], qualification_path: Path
) -> None:
    """Reject a qualification whose replay closure contains noncanonical JSON."""
    records = execution.get("scenario_evidence")
    if not isinstance(records, list) or not records:
        raise B4EvidenceError("execution scenario closure is absent")
    names = [
        record.get("scenario") if isinstance(record, Mapping) else None
        for record in records
    ]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise B4EvidenceError("execution scenario names are missing or duplicated")
    for name in names:
        _load_scenario(execution, qualification_path, str(name))


def _derive_public_telemetry_audit(
    baseline: Mapping[str, Any], baseline_record: Mapping[str, Any]
) -> dict[str, Any]:
    records = baseline.get("public_telemetry_audit")
    if not isinstance(records, list) or len(records) != barrier.CLIENT_COUNT:
        raise B4EvidenceError("baseline public telemetry audit is absent")
    expected_ids = [f"qual-{slot:02d}" for slot in range(barrier.CLIENT_COUNT)]
    if [
        record.get("client_id") if isinstance(record, Mapping) else None
        for record in records
    ] != expected_ids:
        raise B4EvidenceError("baseline public telemetry audit roster differs")
    normalized: list[dict[str, Any]] = []
    totals = {name: 0 for name in PUBLIC_AUDIT_COUNTERS}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "client_id", *PUBLIC_AUDIT_COUNTERS,
        }:
            raise B4EvidenceError(
                f"baseline public telemetry audit {index} schema differs"
            )
        counters = {name: record.get(name) for name in PUBLIC_AUDIT_COUNTERS}
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise B4EvidenceError(
                f"baseline public telemetry audit {index} counters differ"
            )
        if counters["datagrams_seen"] != (
            counters["public_packets_decoded"]
            + counters["malformed_packets_rejected"]
            + counters["teacher_packets_detected"]
        ) or counters["public_packets_decoded"] != (
            counters["routed_packets_accepted"]
            + counters["foreign_client_packets_rejected"]
            + counters["stale_packets_rejected"]
        ):
            raise B4EvidenceError(
                f"baseline public telemetry audit {index} is not exhaustive"
            )
        if counters["datagrams_seen"] <= 0 or counters[
            "routed_packets_accepted"
        ] <= 0:
            raise B4EvidenceError(
                f"baseline public telemetry audit {index} is vacuous"
            )
        for name, value in counters.items():
            totals[name] += value
        normalized.append({"client_id": record["client_id"], **counters})
    if totals["teacher_packets_detected"] != 0:
        raise B4EvidenceError("teacher packet was detected on the public conduit")
    return {
        "derivation": "sealed-baseline-cold-1-public-datagram-accounting-v1",
        "scenario_proof": dict(baseline_record),
        "clients": normalized,
        "totals": totals,
    }


def _derive_teacher_negative_probe(privilege_abi: Mapping[str, Any]) -> dict[str, Any]:
    teacher = privilege_abi.get("teacher")
    public = privilege_abi.get("public")
    if not isinstance(teacher, Mapping) or not isinstance(public, Mapping):
        raise B4EvidenceError("source privilege ABI is absent")
    expected_teacher = (
        public_wire.ML_TEACHER_MAGIC,
        public_wire.ML_TEACHER_VERSION,
        public_wire.TEACHER_SAMPLE_SIZE,
    )
    if (
        teacher.get("magic"), teacher.get("version"), teacher.get("packet_bytes")
    ) != expected_teacher:
        raise B4EvidenceError("source teacher ABI differs from public detector")
    if (
        public.get("magic") != public_wire.ML_CLIENT_TELEM_MAGIC
        or public.get("packet_bytes") != public_wire.CLIENT_TELEMETRY_SIZE
    ):
        raise B4EvidenceError("source public ABI differs from public detector")
    packet = bytearray(public_wire.TEACHER_SAMPLE_SIZE)
    struct.pack_into("<III", packet, 0, *expected_teacher)
    try:
        public_wire.parse_client_telemetry(bytes(packet))
    except public_wire.PublicTelemetryPrivilegeViolation:
        result = "fatal-public-privilege-violation"
    else:
        raise B4EvidenceError("public parser did not fatally reject teacher bytes")
    return {
        "injection": "exact-teacher-magic-version-size-zero-body",
        "packet_magic": expected_teacher[0],
        "packet_version": expected_teacher[1],
        "packet_bytes": len(packet),
        "packet_sha256": _sha256_bytes(bytes(packet)),
        "public_parser_result": result,
    }


def _derive_runtime_and_reward(
    execution: Mapping[str, Any], qualification_path: Path, atlas_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline, baseline_record = _load_scenario(
        execution, qualification_path, "baseline-cold-1"
    )
    epoch, epoch_record = _load_scenario(execution, qualification_path, "epoch-drain")
    stale, stale_record = _load_scenario(execution, qualification_path, "old-telemetry")
    rows = baseline.get("trajectory_rows")
    if not isinstance(rows, list) or not rows:
        raise B4EvidenceError("baseline qualification has no trajectory")
    public_audit = _derive_public_telemetry_audit(baseline, baseline_record)
    public_teacher_violations = public_audit["totals"][
        "teacher_packets_detected"
    ]
    clients = rows[0].get("clients") if isinstance(rows[0], Mapping) else None
    if not isinstance(clients, list) or len(clients) != barrier.CLIENT_COUNT:
        raise B4EvidenceError("baseline qualification has no exact four-client frame")
    client_records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for client in clients:
        if not isinstance(client, Mapping):
            raise B4EvidenceError("baseline client row is malformed")
        client_id = client.get("client_id")
        map_epoch = client.get("map_epoch")
        map_name = client.get("map_name")
        causal = client.get("causal")
        applied = client.get("applied_action")
        if (
            not isinstance(client_id, str) or not client_id
            or client_id in identifiers
            or type(map_epoch) is not int or map_epoch <= 0
            or not isinstance(map_name, str) or not map_name
            or not isinstance(causal, Mapping)
            or not isinstance(applied, Mapping)
        ):
            raise B4EvidenceError("baseline client identity/epoch/action is malformed")
        identifiers.add(client_id)
        client_records.append({
            "client_id": client_id,
            "map_epoch": map_epoch,
            "map_name": map_name,
            "atlas_sha256": atlas_sha256,
        })
        frame = {
            "tick": client.get("action_tick"),
            "client_life_epoch": causal.get("client_life_epoch"),
            "authoritative_echo_valid": causal.get("echo_valid"),
            "trainable_transition": causal.get("transition_trainable"),
            "state_resync": False,
            "teacher_field_violations": public_teacher_violations,
            "action_generation": causal.get("action_generation"),
            "requested_vertical": applied.get("vertical_intent"),
        }
        try:
            CausalRewardFrame(**frame).validate()
        except (RewardAdmissionError, TypeError, ValueError) as error:
            raise B4EvidenceError(f"baseline causal frame is inadmissible: {error}") from error
        frames.append(frame)
    if len({(item["map_epoch"], item["map_name"]) for item in client_records}) != 1:
        raise B4EvidenceError("baseline clients are not in one epoch/map")
    actions = epoch.get("actions_dispatched_during_epoch_drain")
    if actions != 0:
        raise B4EvidenceError("actions were dispatched during the epoch barrier")
    events = stale.get("structured_events")
    old_events = [
        event for event in events or () if isinstance(event, Mapping)
        and event.get("event") == "old_telemetry"
    ]
    if len(old_events) != 1 or old_events[0].get("result") != "discarded" or str(
        old_events[0].get("clock_regressed")
    ) != "0":
        raise B4EvidenceError("old telemetry was not proven discarded without clock regression")
    scenario_proofs = {
        "baseline_cold_1": baseline_record,
        "epoch_drain": epoch_record,
        "old_telemetry": stale_record,
    }
    epoch_evidence = _seal({
        "schema": EPOCH_SCHEMA,
        "clients": sorted(client_records, key=lambda item: item["client_id"]),
        "actions_dispatched_during_epoch_barrier": 0,
        "stale_epoch_transitions_admitted": 0,
        "scenario_proofs": scenario_proofs,
    })
    reward_evidence = _seal({
        "schema": REWARD_SCHEMA,
        "frames": frames,
        "derivation": "baseline-cold-1-authoritative-echo-v1",
        "scenario_proof": baseline_record,
    })
    return epoch_evidence, reward_evidence, scenario_proofs, public_audit


def _normative_records(bot_repo: Path) -> list[dict[str, Any]]:
    return [
        _file_record(bot_repo / relative, relative)
        for relative in NORMATIVE_PATHS
    ]


def _b3_predecessor(
    path: Path,
    *,
    bot_identity: Mapping[str, Any],
    normative_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize the exact active-authority B3 predecessor."""

    source = _exact_regular(path, "B3 predecessor gate")
    raw = _load_json(source, "B3 predecessor gate", style="compact")
    try:
        gate = validate_b3_gate(raw)
    except B3GateError as error:
        raise B4EvidenceError(f"B3 predecessor is inadmissible: {error}") from error

    repository = gate.get("repository")
    expected_repository = {
        "repository_commit": bot_identity.get("commit"),
        "repository_tree": bot_identity.get("tree"),
        "git_clean": bot_identity.get("clean"),
    }
    if repository != expected_repository:
        raise B4EvidenceError("B3 predecessor bot source differs from B4")

    normative_by_name = {
        record.get("name"): record.get("sha256") for record in normative_records
    }
    expected_normative = {
        "design_sha256": normative_by_name.get(NORMATIVE_PATHS[0]),
        "plan_sha256": normative_by_name.get(NORMATIVE_PATHS[1]),
    }
    if gate.get("normative_documents") != expected_normative:
        raise B4EvidenceError("B3 predecessor normative identity differs from B4")

    predecessor = gate.get("predecessor")
    recovery = gate.get("recovery_guide")
    if not isinstance(predecessor, Mapping) or not isinstance(recovery, Mapping):
        raise B4EvidenceError("B3 predecessor authority/Atlas closure is absent")
    try:
        authority = b2_gate._require_active_final_authority()
    except b2_gate.B2GateError as error:
        raise B4EvidenceError(f"B3 predecessor authority rejected: {error}") from error
    if (
        predecessor.get("status") != "green"
        or predecessor.get("cohort_id") != authority.cohort_id
        or predecessor.get("declaration_sha256") != authority.declaration_sha256
    ):
        raise B4EvidenceError("B3 predecessor does not carry the active B2 authority")
    atlas_set_sha256 = recovery.get("atlas_set_sha256")
    gate_sha256 = gate.get("gate_sha256")
    if not _valid_sha256(atlas_set_sha256) or not _valid_sha256(gate_sha256):
        raise B4EvidenceError("B3 predecessor Atlas-set/gate identity is malformed")
    return {
        "b3_gate": _file_record(source, "B3-gate"),
        "b3_gate_sha256": gate_sha256,
        "status": "green",
        "cohort_id": authority.cohort_id,
        "declaration_sha256": authority.declaration_sha256,
        "atlas_set_sha256": atlas_set_sha256,
        "repository_commit": expected_repository["repository_commit"],
        "repository_tree": expected_repository["repository_tree"],
    }


def assemble_b4_evidence(
    *, b3_gate_path: Path, qualification_path: Path, runtime_manifest_path: Path,
    bot_repo: Path, client_repo: Path, game_repo: Path,
    atlas_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not _valid_sha256(atlas_sha256):
        raise B4EvidenceError("Atlas identity must be a nonzero lowercase SHA-256")
    qualification_path = _exact_regular(qualification_path, "network qualification")
    runtime_manifest_path = _exact_regular(runtime_manifest_path, "runtime manifest")
    qualification = _load_json(
        qualification_path, "network qualification", style="compact"
    )
    execution = _validate_qualification(qualification, qualification_path)
    _validate_all_scenario_encodings(execution, qualification_path)
    manifest = _load_json(runtime_manifest_path, "runtime manifest", style="manifest")
    runtime = _validate_runtime_manifest(
        manifest, qualification, execution, atlas_sha256
    )

    repos = {"bot": Path(bot_repo), "client": Path(client_repo), "game": Path(game_repo)}
    identities = {name: _git_identity(path) for name, path in repos.items()}
    if execution.get("source_repositories") != identities:
        raise B4EvidenceError("qualification source commits/trees differ from exact inputs")
    source_closure = _sha256_bytes(_canonical_bytes(identities))
    if execution.get("source_closure_sha256") != source_closure:
        raise B4EvidenceError("qualification source closure seal differs")
    normative_records = _normative_records(repos["bot"])
    predecessor = _b3_predecessor(
        b3_gate_path,
        bot_identity=identities["bot"],
        normative_records=normative_records,
    )
    source = _source_contract(repos["bot"], repos["client"], repos["game"])
    epoch_evidence, reward_evidence, scenario_proofs, public_audit = _derive_runtime_and_reward(
        execution, qualification_path, atlas_sha256
    )
    negative_probe = _derive_teacher_negative_probe(source["privilege_abi"])
    public_teacher_violations = public_audit["totals"][
        "teacher_packets_detected"
    ]
    contract = source["contract"]
    feature_evidence = _seal({
        "schema": FEATURE_SCHEMA,
        **contract,
        "source_closure_sha256": source_closure,
    })
    wire_evidence = _seal({
        "schema": WIRE_SCHEMA,
        "descriptor": source["descriptor"],
        "atlas_sha256": atlas_sha256,
        "runtime_manifest_sha256": runtime["digest"],
        "public_teacher_field_violations": public_teacher_violations,
        "public_privilege_separation": {
            "source_abi": source["privilege_abi"],
            "normal_run_audit": public_audit,
            "negative_probe": negative_probe,
        },
        "private_causal_facts": source["private_causal_facts"],
        "source_abi": source["source_abi"],
        "qualification_evidence_sha256": qualification["evidence_sha256"],
    })
    documents = {
        "feature_action_contract": feature_evidence,
        "runtime_epoch_fencing": epoch_evidence,
        "b4_wire_generation": wire_evidence,
        "causal_reward_admission": reward_evidence,
    }
    document_records = {
        name: {
            "name": DOCUMENT_NAMES[name],
            "sha256": _sha256_bytes(_canonical_bytes(document) + b"\n"),
            "size": len(_canonical_bytes(document) + b"\n"),
            "evidence_sha256": document["evidence_sha256"],
        }
        for name, document in documents.items()
    }
    aggregate = _seal({
        "schema": SCHEMA,
        "milestone": "B4",
        "status": "green",
        "atlas_sha256": atlas_sha256,
        "predecessor": predecessor,
        "source_repositories": identities,
        "source_closure_sha256": source_closure,
        "normative_documents": normative_records,
        "runtime_binaries": execution["runtime_binaries"],
        "runtime_manifest": {
            **_file_record(runtime_manifest_path, "runtime-manifest"),
            "manifest_sha256": runtime["digest"],
        },
        "network_qualification": {
            **_file_record(qualification_path, "network-qualification"),
            "evidence_sha256": qualification["evidence_sha256"],
            "execution_evidence_sha256": execution["execution_evidence_sha256"],
            "runtime_closure_sha256": qualification["runtime_closure_sha256"],
        },
        "scenario_proofs": scenario_proofs,
        "public_privilege_proof": {
            "baseline_raw_evidence_sha256": scenario_proofs[
                "baseline_cold_1"
            ]["raw_evidence_sha256"],
            "datagrams_seen": public_audit["totals"]["datagrams_seen"],
            "public_packets_decoded": public_audit["totals"][
                "public_packets_decoded"
            ],
            "teacher_packets_detected": public_teacher_violations,
            "negative_probe_packet_sha256": negative_probe["packet_sha256"],
            "negative_probe_result": negative_probe["public_parser_result"],
        },
        "documents": document_records,
        "component_evidence": documents,
        "gate": {
            "clean_exact_sources": True,
            "atomic_wire_generation": True,
            "real_network_barrier_replayed": True,
            "runtime_and_binary_seals_bound": True,
            "epoch_and_stale_fencing_proven": True,
            "public_teacher_violations_zero": (
                public_teacher_violations == 0
                and negative_probe["public_parser_result"]
                == "fatal-public-privilege-violation"
            ),
            "causal_reward_frames_admissible": True,
            "active_authority_b3_predecessor_bound": True,
        },
    })
    validate_b4_evidence(aggregate, documents)
    return aggregate, documents


def validate_b4_evidence(
    aggregate: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> None:
    _verify_seal(aggregate, "B4 aggregate")
    if set(aggregate) != B4_AGGREGATE_KEYS:
        raise B4EvidenceError("B4 aggregate keys differ from the producer schema")
    if (
        aggregate.get("schema") != SCHEMA
        or aggregate.get("milestone") != "B4"
        or aggregate.get("status") != "green"
        or not _valid_sha256(aggregate.get("atlas_sha256"))
    ):
        raise B4EvidenceError("B4 aggregate schema/status differs")
    try:
        authority = b2_gate._require_active_final_authority()
    except b2_gate.B2GateError as error:
        raise B4EvidenceError(f"B4 predecessor authority rejected: {error}") from error
    predecessor = aggregate.get("predecessor")
    repositories = aggregate.get("source_repositories")
    if not isinstance(predecessor, Mapping) or not isinstance(repositories, Mapping):
        raise B4EvidenceError("B4 predecessor/source closure is absent")
    expected_predecessor_keys = {
        "b3_gate", "b3_gate_sha256", "status", "cohort_id",
        "declaration_sha256", "atlas_set_sha256", "repository_commit",
        "repository_tree",
    }
    if set(predecessor) != expected_predecessor_keys:
        raise B4EvidenceError("B4 predecessor keys differ")
    b3_record = predecessor.get("b3_gate")
    if (
        not isinstance(b3_record, Mapping)
        or set(b3_record) != {"name", "sha256", "size"}
        or b3_record.get("name") != "B3-gate"
        or not _valid_sha256(b3_record.get("sha256"))
        or type(b3_record.get("size")) is not int
        or b3_record["size"] < 1
    ):
        raise B4EvidenceError("B4 predecessor file record differs")
    bot_source = repositories.get("bot")
    if not isinstance(bot_source, Mapping) or (
        predecessor.get("status") != "green"
        or predecessor.get("cohort_id") != authority.cohort_id
        or predecessor.get("cohort_id") == b2_gate.RETIRED_COHORT_71446
        or predecessor.get("declaration_sha256") != authority.declaration_sha256
        or predecessor.get("repository_commit") != bot_source.get("commit")
        or predecessor.get("repository_tree") != bot_source.get("tree")
        or not _valid_sha256(predecessor.get("b3_gate_sha256"))
        or not _valid_sha256(predecessor.get("atlas_set_sha256"))
    ):
        raise B4EvidenceError("B4 predecessor authority/source closure differs")
    if set(documents) != set(DOCUMENT_NAMES):
        raise B4EvidenceError("B4 evidence document set differs")
    if aggregate.get("component_evidence") != documents:
        raise B4EvidenceError("B4 embedded component evidence differs from supplied documents")
    records = aggregate.get("documents")
    if not isinstance(records, Mapping):
        raise B4EvidenceError("B4 document records are absent")
    for name, filename in DOCUMENT_NAMES.items():
        document = documents[name]
        _verify_seal(document, name)
        data = _canonical_bytes(document) + b"\n"
        if records.get(name) != {
            "name": filename,
            "sha256": _sha256_bytes(data),
            "size": len(data),
            "evidence_sha256": document["evidence_sha256"],
        }:
            raise B4EvidenceError(f"B4 document record differs for {name}")
    source_repositories = aggregate.get("source_repositories")
    if (
        not isinstance(source_repositories, Mapping)
        or set(source_repositories) != {"bot", "client", "game"}
        or any(
            not isinstance(identity, Mapping)
            or set(identity) != {"clean", "commit", "tree"}
            or identity.get("clean") is not True
            or re.fullmatch(r"[0-9a-f]{40}", str(identity.get("commit"))) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(identity.get("tree"))) is None
            for identity in source_repositories.values()
        )
        or aggregate.get("source_closure_sha256")
        != _sha256_bytes(_canonical_bytes(source_repositories))
    ):
        raise B4EvidenceError("B4 source repository closure differs")

    feature = documents["feature_action_contract"]
    expected_feature_keys = {
        "schema", "source_closure_sha256", "evidence_sha256", *EXPECTED_CONTRACT
    }
    if (
        set(feature) != expected_feature_keys
        or feature.get("schema") != FEATURE_SCHEMA
        or any(feature.get(key) != value for key, value in EXPECTED_CONTRACT.items())
        or feature.get("source_closure_sha256")
        != aggregate.get("source_closure_sha256")
    ):
        raise B4EvidenceError("B4 feature/action component semantics differ")

    wire = documents["b4_wire_generation"]
    expected_wire_keys = {
        "schema", "descriptor", "atlas_sha256", "runtime_manifest_sha256",
        "public_teacher_field_violations", "public_privilege_separation",
        "private_causal_facts", "source_abi",
        "qualification_evidence_sha256", "evidence_sha256",
    }
    runtime_manifest = aggregate.get("runtime_manifest")
    qualification = aggregate.get("network_qualification")
    if (
        set(wire) != expected_wire_keys
        or wire.get("schema") != WIRE_SCHEMA
        or not isinstance(wire.get("descriptor"), Mapping)
        or any(
            wire["descriptor"].get(key) != value
            for key, value in EXPECTED_DESCRIPTOR.items()
        )
        or wire.get("atlas_sha256") != aggregate.get("atlas_sha256")
        or not isinstance(runtime_manifest, Mapping)
        or wire.get("runtime_manifest_sha256")
        != runtime_manifest.get("manifest_sha256")
        or not isinstance(qualification, Mapping)
        or wire.get("qualification_evidence_sha256")
        != qualification.get("evidence_sha256")
        or wire.get("private_causal_facts") != list(PRIVATE_CAUSAL_FACTS)
        or not isinstance(wire.get("source_abi"), Mapping)
    ):
        raise B4EvidenceError("B4 wire component semantics differ")

    epoch = documents["runtime_epoch_fencing"]
    reward = documents["causal_reward_admission"]
    if set(epoch) != {
        "schema", "clients", "actions_dispatched_during_epoch_barrier",
        "stale_epoch_transitions_admitted", "scenario_proofs", "evidence_sha256",
    } or (
        epoch.get("schema") != EPOCH_SCHEMA
        or epoch.get("actions_dispatched_during_epoch_barrier") != 0
        or epoch.get("stale_epoch_transitions_admitted") != 0
        or epoch.get("scenario_proofs") != aggregate.get("scenario_proofs")
    ):
        raise B4EvidenceError("B4 epoch-fencing component semantics differ")
    clients = epoch.get("clients")
    if (
        not isinstance(clients, list)
        or len(clients) != 4
        or len({item.get("client_id") for item in clients if isinstance(item, Mapping)}) != 4
        or len({
            (item.get("map_epoch"), item.get("map_name"), item.get("atlas_sha256"))
            for item in clients if isinstance(item, Mapping)
        }) != 1
        or any(
            not isinstance(item, Mapping)
            or item.get("atlas_sha256") != aggregate.get("atlas_sha256")
            or type(item.get("map_epoch")) is not int
            or item.get("map_epoch") <= 0
            or not isinstance(item.get("map_name"), str)
            or not item.get("map_name")
            for item in clients
        )
    ):
        raise B4EvidenceError("B4 epoch client evidence differs")
    if set(reward) != {
        "schema", "frames", "derivation", "scenario_proof", "evidence_sha256",
    } or (
        reward.get("schema") != REWARD_SCHEMA
        or reward.get("derivation") != "baseline-cold-1-authoritative-echo-v1"
        or reward.get("scenario_proof")
        != aggregate.get("scenario_proofs", {}).get("baseline_cold_1")
    ):
        raise B4EvidenceError("B4 causal-reward component semantics differ")
    frames = reward.get("frames")
    if not isinstance(frames, list) or len(frames) != 4:
        raise B4EvidenceError("B4 causal-reward frame cardinality differs")
    for frame in frames:
        try:
            CausalRewardFrame(**frame).validate()
        except (RewardAdmissionError, TypeError, ValueError) as error:
            raise B4EvidenceError("B4 causal-reward frame is inadmissible") from error
    separation = wire.get("public_privilege_separation")
    proof = aggregate.get("public_privilege_proof")
    if not isinstance(separation, Mapping) or not isinstance(proof, Mapping):
        raise B4EvidenceError("B4 public privilege proof is absent")
    normal = separation.get("normal_run_audit")
    negative = separation.get("negative_probe")
    if not isinstance(normal, Mapping) or not isinstance(negative, Mapping):
        raise B4EvidenceError("B4 public privilege proof components are absent")
    totals = normal.get("totals")
    scenario = normal.get("scenario_proof")
    if not isinstance(totals, Mapping) or not isinstance(scenario, Mapping) or proof != {
        "baseline_raw_evidence_sha256": scenario.get("raw_evidence_sha256"),
        "datagrams_seen": totals.get("datagrams_seen"),
        "public_packets_decoded": totals.get("public_packets_decoded"),
        "teacher_packets_detected": totals.get("teacher_packets_detected"),
        "negative_probe_packet_sha256": negative.get("packet_sha256"),
        "negative_probe_result": negative.get("public_parser_result"),
    }:
        raise B4EvidenceError("B4 aggregate public privilege proof differs")
    if (
        wire.get("public_teacher_field_violations")
        != totals.get("teacher_packets_detected")
        or totals.get("teacher_packets_detected") != 0
        or negative.get("public_parser_result")
        != "fatal-public-privilege-violation"
    ):
        raise B4EvidenceError("B4 public privilege gate is not green")
    gate = aggregate.get("gate")
    if (
        not isinstance(gate, Mapping)
        or set(gate) != B4_GATE_KEYS
        or any(value is not True for value in gate.values())
    ):
        raise B4EvidenceError("B4 aggregate contains a non-green gate")


def publish_b4_evidence(
    output_dir: Path, aggregate: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        raise B4EvidenceError("output directory must be absolute")
    if destination.exists() or destination.is_symlink():
        raise B4EvidenceError("output directory must be a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, document in documents.items():
            (temporary / DOCUMENT_NAMES[name]).write_bytes(_canonical_bytes(document) + b"\n")
        (temporary / "B4-EVIDENCE.json").write_bytes(_canonical_bytes(aggregate) + b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3-gate", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--bot-repo", type=Path, required=True)
    parser.add_argument("--client-repo", type=Path, required=True)
    parser.add_argument("--game-repo", type=Path, required=True)
    parser.add_argument("--atlas-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        aggregate, documents = assemble_b4_evidence(
            b3_gate_path=args.b3_gate,
            qualification_path=args.qualification,
            runtime_manifest_path=args.runtime_manifest,
            bot_repo=args.bot_repo,
            client_repo=args.client_repo,
            game_repo=args.game_repo,
            atlas_sha256=args.atlas_sha256,
        )
        publish_b4_evidence(args.output_dir, aggregate, documents)
    except B4EvidenceError as error:
        print(f"B4 EVIDENCE FAILED: {error}", file=sys.stderr)
        return 1
    print(args.output_dir)
    print(aggregate["evidence_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
