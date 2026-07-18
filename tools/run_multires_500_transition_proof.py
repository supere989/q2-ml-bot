#!/usr/bin/env python3
"""Fail-closed 500-transition determinism proof for the isolated multires trainer.

Milestone M6d.  Production mode is self-contained: it launches three fresh
subprocesses of an absolute operational one-run trainer command (no shell, no
Python callback injection).  Two identical same-seed full-stack launches must
match byte-for-byte; a launch that differs only in game seed must diverge.

The production one-run contract never admits a digest-only assertion.  Each
launch must emit exactly 500 fully parsed canonical transition records in
time-major / client-minor order for four clients (125 transitions each).  The
trajectory digest is recomputed from those records; identity fields, dual
boundary digests, and freshness facts must match the admitted launch.
Each Rust-feature digest is recomputed from the admitted observation's frozen
``DYN.slice``; an optional separate ``rust_features`` payload is only an exact
equality cross-check and can never substitute for observation-bound evidence.

Client identities are not hardcoded: the four unique ordered client IDs are
derived from the first time step (records 0..3) and must hold that same order
on every subsequent round.  ``batch_round_id`` is exactly ``0..124``.
``server_frame`` values must be synchronized across clients and advance by
exactly +1 per accepted round from a positive base frame observed at the first
time step — the parser does not assume a magic starting frame.

Teardown admission does not trust ``orphan_processes_after_teardown`` alone:
every PID the trainer reported in ``process_ids`` / launched PIDs must be
proven dead after exit.  Timeout cleanup discovers Linux descendants via
``/proc`` PPID walks (scoped to the owned trainer tree only; no pattern kills)
before SIGTERM/SIGKILL, then re-verifies death.

Usage (production):

    python3 tools/run_multires_500_transition_proof.py \\
        --mode production \\
        --seed 7142026 --game_seed 4242 --divergence_game_seed 4243 \\
        --trainer_executable /abs/path/to/python3 \\
        --trainer_arg -m --trainer_arg train.multires_one_run \\
        --q2ded /abs/q2ded --client_binary /abs/q2 \\
        --runtime_root /abs/runtime \\
        --bundle_manifest /abs/map.bundle.manifest.json \\
        --objectives /abs/map.objectives.json \\
        --atlas_bin /abs/map.atlas.bin \\
        --checkpoint /abs/attested_fresh.pt \\
        --training_manifest /abs/training_manifest.json \\
        --runtime_evidence /abs/runtime_evidence.json \\
        --out /abs/deterministic_transitions.json

Injected transports remain available for non-admissible synthetic unit tests
only.  Production rejects callbacks, injected transports, legacy selectors,
synthetic trainer output, digest-only trajectories, missing/extra/truncated/
duplicate/reordered/noncanonical/unbound records, cross-client frame skew,
non-unit frame jumps, partial/stale/resync admissions, wrong provider/
collector/generation/digests, and leftover owned/descendant PIDs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_catalog import (  # noqa: E402
    ATLAS_CATALOG_DOMAIN,
    ATLAS_CATALOG_SCHEMA,
    AtlasCatalogError,
    canonical_bytes as atlas_catalog_canonical_bytes,
    load_atlas_catalog,
)
from harness.multires_admission import SPATIAL_PROVIDER_SCHEMA  # noqa: E402
from harness.multires_contract import (  # noqa: E402
    ACTION_DIM,
    DYN,
    FEATURE_SCHEMA_SHA256,
    OBS_DIM,
    POLICY_GENERATION,
)
from harness.multires_lineage import (  # noqa: E402
    CHECKPOINT_FORMAT,
    load_attested_checkpoint,
)
from harness.multires_runtime import (  # noqa: E402
    B4_ACTION_MAGIC,
    B4_CAUSAL_MAGIC,
    B4_CAUSAL_PACKET_BYTES,
    B4_CAUSAL_VERSION,
    B4_CLIENT_WIRE_VERSION,
    B4_OBSERVATION_MAGIC,
    B4_PROTOCOL_GENERATION,
    B4_ROLLOUT_SCHEMA,
    B4_TEACHER_VERSION,
    LEGACY_CLIENT_WIRE_VERSION,
    LEGACY_OBSERVATION_MAGIC,
    LEGACY_ROLLOUT_SCHEMA,
    validate_runtime_evidence,
)
from harness.multires_training_config import (  # noqa: E402
    MultiresTrainingConfiguration,
    TRAINING_CONFIG_SCHEMA,
)


TOOL_NAME = "run_multires_500_transition_proof"
PROOF_SCHEMA = "q2-multires-500-transition-proof-v1"
VERIFIER_GATE = "deterministic_transitions"
REQUIRED_TRANSITION_COUNT = 500
REQUIRED_CLIENT_COUNT = 4
TRANSITIONS_PER_CLIENT = REQUIRED_TRANSITION_COUNT // REQUIRED_CLIENT_COUNT
REQUIRED_PROCESS_ROLES = (
    "q2ded",
    *(f"network-client-{index:02d}" for index in range(REQUIRED_CLIENT_COUNT)),
)
assert (
    REQUIRED_CLIENT_COUNT * TRANSITIONS_PER_CLIENT == REQUIRED_TRANSITION_COUNT
), "500-transition proof layout must be exactly four clients x 125 transitions"
# Synthetic fixtures only — production derives ordered IDs from the first round.
DEFAULT_SYNTHETIC_CLIENT_IDS: tuple[str, ...] = tuple(
    f"mrproof-{index:02d}" for index in range(REQUIRED_CLIENT_COUNT)
)
DEFAULT_SYNTHETIC_BASE_SERVER_FRAME = 4827
RECORD_ORDERING = "time-major/client-minor"
MODE_PRODUCTION = "production"
MODE_SYNTHETIC = "synthetic_injected"
TRAJECTORY_DOMAIN = b"q2-multires-500-transition-trajectory-v1\0"
STACK_LANGUAGE = "multires-stack"
PYTHON_COLLECTOR_SCHEMA = "q2-multires-collected-rollout-v1"
RUST_PROVIDER_SCHEMA = SPATIAL_PROVIDER_SCHEMA
COLLECTOR_CLASS_NAME = "MultiresSynchronousCollector"
SPATIAL_PROVIDER_CLASS_NAME = "RustAtlasSpatialProvider"
LATTICE_CRATE_NAME = "q2_lattice"
ONE_RUN_SCHEMA = "q2-multires-one-run-proof-v1"
ONE_RUN_PROTOCOL_VERSION = 1
# Documented M4 one-run entry expected once operational CLI is stable.
DOCUMENTED_M4_ONE_RUN_MODULE = "train.multires_one_run"
FORBIDDEN_LEGACY_SELECTORS = (
    "train.ppo",
    "models.policy",
    "public_network_thermal_bc_live_v2",
    "public_network_engagement_anchor_v3",
    "public_network_thermal_fresh_v1",
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_PLACEHOLDER_TOKENS = ("placeholder", "tbd", "todo", "fixme", "changeme")
_SHELL_METACHARACTERS = frozenset("|&;<>()$`\\\"'\n\r")
_RUST_FEATURE_WIDTH = 24


class Multires500ProofError(RuntimeError):
    """Raised when the 500-transition proof cannot admit a production result."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_json(value: object) -> str:
    return _canonical_bytes(value).decode("utf-8")


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
            raise Multires500ProofError(f"{where}: placeholder marker in {value!r}")
        if value == "0" * 64:
            raise Multires500ProofError(f"{where}: all-zero digest is placeholder evidence")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(key, where)
            _reject_placeholders(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{where}[{index}]")


def _require_path(path: Optional[Path], label: str, *, executable: bool = False) -> Path:
    if path is None:
        raise Multires500ProofError(f"production mode requires explicit {label}")
    resolved = path.expanduser()
    if not resolved.is_absolute():
        raise Multires500ProofError(f"{label} must be an absolute path: {path}")
    if not resolved.exists():
        raise Multires500ProofError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise Multires500ProofError(f"{label} is not a file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise Multires500ProofError(f"{label} is not executable: {resolved}")
    return resolved.resolve()


def _require_dir(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise Multires500ProofError(f"production mode requires explicit {label}")
    resolved = path.expanduser()
    if not resolved.is_absolute():
        raise Multires500ProofError(f"{label} must be an absolute path: {path}")
    if not resolved.is_dir():
        raise Multires500ProofError(f"{label} is not a directory: {resolved}")
    return resolved.resolve()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Multires500ProofError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise Multires500ProofError(f"{label} must be a JSON object")
    _reject_placeholders(payload, label)
    return payload


def _load_rust_extension(path: Path) -> Any:
    """Load only the exact Rust authority sealed into an admitted catalog."""
    sys.modules.pop("q2_lattice_rs", None)
    spec = importlib.util.spec_from_file_location("q2_lattice_rs", path)
    if spec is None or spec.loader is None:
        raise Multires500ProofError(
            "cannot load the Atlas catalog Rust Dyn authority"
        )
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError) as error:
        raise Multires500ProofError(
            f"cannot load the Atlas catalog Rust Dyn authority: {error}"
        ) from error
    return module


def _admit_atlas_catalog(path: Path, expected_sha256: str):
    """Seal the catalog before loading its exact native-code authority."""
    document = _load_json_object(path, "atlas_catalog")
    exact_fields = {
        "schema", "domain", "map_count", "rust_extension", "dyn_creation",
        "maps", "atlas_catalog_sha256",
    }
    if (
        set(document) != exact_fields
        or document.get("schema") != ATLAS_CATALOG_SCHEMA
        or document.get("domain") != ATLAS_CATALOG_DOMAIN
    ):
        raise Multires500ProofError("Atlas catalog fields/schema/domain differ")
    content = {
        "domain": ATLAS_CATALOG_DOMAIN,
        "rust_extension": document["rust_extension"],
        "dyn_creation": document["dyn_creation"],
        "maps": document["maps"],
    }
    semantic_sha256 = hashlib.sha256(
        atlas_catalog_canonical_bytes(content)
    ).hexdigest()
    if (
        document.get("atlas_catalog_sha256") != semantic_sha256
        or expected_sha256 != semantic_sha256
    ):
        raise Multires500ProofError("Atlas catalog content seal differs")

    rust_record = document.get("rust_extension")
    if not isinstance(rust_record, Mapping) or set(rust_record) != {"path", "sha256"}:
        raise Multires500ProofError("Atlas catalog Rust extension record is malformed")
    relative = Path(str(rust_record["path"]))
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise Multires500ProofError(
            "Atlas catalog Rust extension path is not portable catalog-relative"
        )
    rust_extension = _require_path(
        path.parent.resolve() / relative, "Atlas catalog Rust extension"
    )
    try:
        rust_extension.relative_to(path.parent.resolve())
    except ValueError as error:
        raise Multires500ProofError(
            "Atlas catalog Rust extension path escapes the catalog root"
        ) from error
    if not _valid_sha256(rust_record["sha256"]) or _file_sha256(
        rust_extension
    ) != rust_record["sha256"]:
        raise Multires500ProofError("Atlas catalog Rust extension digest differs")

    extension_module = _load_rust_extension(rust_extension)
    try:
        return load_atlas_catalog(
            path,
            expected_sha256=expected_sha256,
            extension_module=extension_module,
        )
    except (AtlasCatalogError, OSError, TypeError, ValueError) as error:
        raise Multires500ProofError(f"Atlas catalog admission failed: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(value: Any, width: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise Multires500ProofError(f"{label} must be a length-{width} sequence")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Multires500ProofError(f"{label}[{index}] is not numeric")
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise Multires500ProofError(f"{label}[{index}] is non-finite")
        result.append(number)
    return result


def _argv_token_safe(token: str, label: str) -> str:
    if not isinstance(token, str) or not token:
        raise Multires500ProofError(f"{label} must be a nonempty string argv token")
    if any(character in _SHELL_METACHARACTERS for character in token):
        raise Multires500ProofError(
            f"{label} rejects shell metacharacters (no shell execution): {token!r}"
        )
    if "\0" in token:
        raise Multires500ProofError(f"{label} rejects NUL bytes")
    return token


@dataclass(frozen=True)
class TransitionRecord:
    """One admitted transition with dual-boundary provenance."""

    index: int
    observation: tuple[float, ...]
    action: tuple[float, ...]
    reward: float
    client_id: str
    server_frame: int
    batch_round_id: int
    policy_version: int
    map_name: str
    map_epoch: int
    atlas_sha256: str
    runtime_manifest_sha256: str
    rust_provider_schema: str
    rust_feature_digest: str
    python_collector_schema: str
    identity_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "observation": list(self.observation),
            "action": list(self.action),
            "reward": self.reward,
            "client_id": self.client_id,
            "server_frame": self.server_frame,
            "batch_round_id": self.batch_round_id,
            "policy_version": self.policy_version,
            "map_name": self.map_name,
            "map_epoch": self.map_epoch,
            "atlas_sha256": self.atlas_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "rust_provider_schema": self.rust_provider_schema,
            "rust_feature_digest": self.rust_feature_digest,
            "python_collector_schema": self.python_collector_schema,
            "identity_digest": self.identity_digest,
        }


def transition_identity_digest(
    *,
    client_id: str,
    server_frame: int,
    batch_round_id: int,
    policy_version: int,
    map_name: str,
    map_epoch: int,
    atlas_sha256: str,
    runtime_manifest_sha256: str,
) -> str:
    payload = {
        "atlas_sha256": atlas_sha256,
        "batch_round_id": int(batch_round_id),
        "client_id": str(client_id),
        "map_epoch": int(map_epoch),
        "map_name": str(map_name),
        "policy_version": int(policy_version),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "server_frame": int(server_frame),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def rust_feature_digest(features: Sequence[float]) -> str:
    values = _float_list(list(features), _RUST_FEATURE_WIDTH, "rust_features")
    return hashlib.sha256(_canonical_bytes({"features": values})).hexdigest()


def make_transition_record(
    *,
    index: int,
    observation: Sequence[float],
    action: Sequence[float],
    reward: float,
    client_id: str,
    server_frame: int,
    batch_round_id: int,
    policy_version: int,
    map_name: str,
    map_epoch: int,
    atlas_sha256: str,
    runtime_manifest_sha256: str,
    rust_features: Sequence[float],
) -> TransitionRecord:
    obs = tuple(_float_list(list(observation), OBS_DIM, "observation"))
    act = tuple(_float_list(list(action), ACTION_DIM, "action"))
    emitted_rust_features = tuple(
        _float_list(list(rust_features), _RUST_FEATURE_WIDTH, "rust_features")
    )
    observation_dyn = obs[DYN.slice]
    if emitted_rust_features != observation_dyn:
        raise Multires500ProofError(
            "rust_features must exactly equal observation[DYN.slice]"
        )
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise Multires500ProofError("reward must be numeric")
    reward_value = float(reward)
    if reward_value != reward_value or reward_value in (float("inf"), float("-inf")):
        raise Multires500ProofError("reward must be finite")
    if not _valid_sha256(atlas_sha256):
        raise Multires500ProofError("atlas_sha256 is not a lowercase SHA-256")
    if not _valid_sha256(runtime_manifest_sha256):
        raise Multires500ProofError("runtime_manifest_sha256 is not a lowercase SHA-256")
    identity = transition_identity_digest(
        client_id=client_id,
        server_frame=server_frame,
        batch_round_id=batch_round_id,
        policy_version=policy_version,
        map_name=map_name,
        map_epoch=map_epoch,
        atlas_sha256=atlas_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
    )
    return TransitionRecord(
        index=int(index),
        observation=obs,
        action=act,
        reward=reward_value,
        client_id=str(client_id),
        server_frame=int(server_frame),
        batch_round_id=int(batch_round_id),
        policy_version=int(policy_version),
        map_name=str(map_name),
        map_epoch=int(map_epoch),
        atlas_sha256=atlas_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        rust_provider_schema=RUST_PROVIDER_SCHEMA,
        rust_feature_digest=rust_feature_digest(observation_dyn),
        python_collector_schema=PYTHON_COLLECTOR_SCHEMA,
        identity_digest=identity,
    )


def trajectory_sha256(records: Sequence[TransitionRecord]) -> str:
    if not records:
        raise Multires500ProofError("trajectory is empty")
    digest = hashlib.sha256()
    digest.update(TRAJECTORY_DOMAIN)
    digest.update(int(len(records)).to_bytes(8, "little", signed=False))
    for record in records:
        digest.update(_canonical_bytes(record.to_mapping()))
        digest.update(b"\0")
    return digest.hexdigest()


def expected_client_id(
    client_index: int, client_ids: Sequence[str]
) -> str:
    if not 0 <= int(client_index) < REQUIRED_CLIENT_COUNT:
        raise Multires500ProofError(
            f"client_index={client_index} outside 0..{REQUIRED_CLIENT_COUNT - 1}"
        )
    if len(client_ids) != REQUIRED_CLIENT_COUNT:
        raise Multires500ProofError(
            f"client_ids must contain exactly {REQUIRED_CLIENT_COUNT} ordered IDs"
        )
    return str(client_ids[int(client_index)])


def layout_for_flat_index(
    flat_index: int,
    client_ids: Sequence[str] = DEFAULT_SYNTHETIC_CLIENT_IDS,
) -> tuple[int, int, str]:
    """Return (time_step, client_index, client_id) for time-major/client-minor."""
    if not 0 <= int(flat_index) < REQUIRED_TRANSITION_COUNT:
        raise Multires500ProofError(
            f"flat index {flat_index} is outside the exact-500 layout"
        )
    time_step = int(flat_index) // REQUIRED_CLIENT_COUNT
    client_index = int(flat_index) % REQUIRED_CLIENT_COUNT
    return time_step, client_index, expected_client_id(client_index, client_ids)


def client_ids_from_first_round(
    records: Sequence[TransitionRecord],
) -> tuple[str, ...]:
    """Derive the exact four unique ordered client IDs from time step 0."""
    if len(records) < REQUIRED_CLIENT_COUNT:
        raise Multires500ProofError(
            "cannot derive client IDs: first time step is incomplete"
        )
    first_round = tuple(records[index] for index in range(REQUIRED_CLIENT_COUNT))
    client_ids = tuple(record.client_id for record in first_round)
    if any(not client_id for client_id in client_ids):
        raise Multires500ProofError(
            "first time step contains an empty client_id"
        )
    if len(set(client_ids)) != REQUIRED_CLIENT_COUNT:
        raise Multires500ProofError(
            f"first time step must contain {REQUIRED_CLIENT_COUNT} unique "
            f"client_ids; got {client_ids!r}"
        )
    return client_ids


def base_server_frame_from_first_round(
    records: Sequence[TransitionRecord],
) -> int:
    """Derive the positive synchronized base server_frame from time step 0."""
    if len(records) < REQUIRED_CLIENT_COUNT:
        raise Multires500ProofError(
            "cannot derive base server_frame: first time step is incomplete"
        )
    frames = {
        int(records[index].server_frame) for index in range(REQUIRED_CLIENT_COUNT)
    }
    if len(frames) != 1:
        raise Multires500ProofError(
            f"first time step has mixed server_frame values {sorted(frames)}; "
            "cross-client frame skew is not admissible"
        )
    base_frame = int(records[0].server_frame)
    if base_frame <= 0:
        raise Multires500ProofError(
            f"base server_frame must be a positive engine frame; got {base_frame}"
        )
    return base_frame


def enforce_exact_500_layout(
    records: Sequence[TransitionRecord],
) -> tuple[tuple[str, ...], int]:
    """Bind 4x125 time-major/client-minor order and unit server_frame advance.

    Returns ``(ordered_client_ids, base_server_frame)``.
    """
    if len(records) != REQUIRED_TRANSITION_COUNT:
        raise Multires500ProofError(
            f"exact-500 layout requires {REQUIRED_TRANSITION_COUNT} records; "
            f"got {len(records)}"
        )
    for flat_index, record in enumerate(records):
        if int(record.index) != flat_index:
            raise Multires500ProofError(
                f"records[{flat_index}] reordered/noncanonical index="
                f"{record.index} (expected {flat_index})"
            )

    client_ids = client_ids_from_first_round(records)
    base_frame = base_server_frame_from_first_round(records)
    client_counts = {client_id: 0 for client_id in client_ids}

    for time_step in range(TRANSITIONS_PER_CLIENT):
        expected_frame = base_frame + time_step
        frames_this_round: set[int] = set()
        for client_index in range(REQUIRED_CLIENT_COUNT):
            flat_index = time_step * REQUIRED_CLIENT_COUNT + client_index
            record = records[flat_index]
            expected_client = client_ids[client_index]
            if record.client_id != expected_client:
                raise Multires500ProofError(
                    f"records[{flat_index}] reordered client_id="
                    f"{record.client_id!r} (expected {expected_client!r} for "
                    "time-major/client-minor; order is bound from the first "
                    "time step and must not change)"
                )
            if int(record.batch_round_id) != time_step:
                raise Multires500ProofError(
                    f"records[{flat_index}] batch_round_id="
                    f"{record.batch_round_id} != time_step={time_step} "
                    f"(batch_round_id must be exactly 0..{TRANSITIONS_PER_CLIENT - 1})"
                )
            if int(record.server_frame) != expected_frame:
                if time_step == 0:
                    raise Multires500ProofError(
                        f"records[{flat_index}] server_frame="
                        f"{record.server_frame} != first-round base {base_frame}"
                    )
                previous = base_frame + time_step - 1
                delta = int(record.server_frame) - previous
                if delta != 1:
                    raise Multires500ProofError(
                        f"records[{flat_index}] non-unit server_frame jump: "
                        f"got {record.server_frame} after previous round frame "
                        f"{previous} (delta={delta}; required exactly +1 per "
                        "accepted round)"
                    )
                raise Multires500ProofError(
                    f"records[{flat_index}] server_frame={record.server_frame} "
                    f"!= expected synchronized frame {expected_frame} "
                    f"(base {base_frame} + time_step {time_step})"
                )
            frames_this_round.add(int(record.server_frame))
            client_counts[record.client_id] += 1
        if len(frames_this_round) != 1:
            raise Multires500ProofError(
                f"time_step={time_step} has mixed server_frame values "
                f"{sorted(frames_this_round)}; cross-client frame skew is not "
                "admissible"
            )

    for client_id, count in client_counts.items():
        if count != TRANSITIONS_PER_CLIENT:
            raise Multires500ProofError(
                f"client {client_id!r} has {count} transitions; exact-500 "
                f"layout requires {TRANSITIONS_PER_CLIENT} each across "
                f"{REQUIRED_CLIENT_COUNT} clients"
            )
    return client_ids, base_frame


def parse_canonical_transition_record(
    item: Mapping[str, Any],
    *,
    flat_index: int,
    expected_atlas_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_map_name: str,
    expected_map_epoch: int,
    expected_policy_version: int,
    require_exact_layout: bool,
) -> TransitionRecord:
    """Fully parse one canonical transition; reject truncated/noncanonical/unbound."""
    if not isinstance(item, Mapping):
        raise Multires500ProofError(
            f"records[{flat_index}] must be an object (truncated/noncanonical)"
        )
    required_keys = (
        "index",
        "observation",
        "action",
        "reward",
        "client_id",
        "server_frame",
        "batch_round_id",
        "policy_version",
        "map_name",
        "map_epoch",
        "atlas_sha256",
        "runtime_manifest_sha256",
        "rust_provider_schema",
        "rust_feature_digest",
        "python_collector_schema",
        "identity_digest",
    )
    missing = [key for key in required_keys if key not in item]
    if missing:
        raise Multires500ProofError(
            f"records[{flat_index}] truncated; missing fields: {missing}"
        )

    try:
        index = int(item["index"])
        server_frame = int(item["server_frame"])
        batch_round_id = int(item["batch_round_id"])
        policy_version = int(item["policy_version"])
        map_epoch = int(item["map_epoch"])
    except (TypeError, ValueError) as error:
        raise Multires500ProofError(
            f"records[{flat_index}] has non-integer identity fields: {error}"
        ) from error

    client_id = item["client_id"]
    map_name = item["map_name"]
    if not isinstance(client_id, str) or not client_id:
        raise Multires500ProofError(
            f"records[{flat_index}] client_id must be a nonempty string"
        )
    if not isinstance(map_name, str) or not map_name:
        raise Multires500ProofError(
            f"records[{flat_index}] map_name must be a nonempty string"
        )
    _reject_placeholders(client_id, f"records[{flat_index}].client_id")
    _reject_placeholders(map_name, f"records[{flat_index}].map_name")

    atlas_sha256 = item["atlas_sha256"]
    runtime_manifest_sha256 = item["runtime_manifest_sha256"]
    if not _valid_sha256(atlas_sha256):
        raise Multires500ProofError(
            f"records[{flat_index}] atlas_sha256 is not a lowercase SHA-256"
        )
    if not _valid_sha256(runtime_manifest_sha256):
        raise Multires500ProofError(
            f"records[{flat_index}] runtime_manifest_sha256 is not a lowercase SHA-256"
        )
    if atlas_sha256 != expected_atlas_sha256:
        raise Multires500ProofError(
            f"records[{flat_index}] unbound atlas_sha256 (does not match launch)"
        )
    if runtime_manifest_sha256 != expected_runtime_manifest_sha256:
        raise Multires500ProofError(
            f"records[{flat_index}] unbound runtime_manifest_sha256 "
            "(does not match launch)"
        )
    if map_name != expected_map_name:
        raise Multires500ProofError(
            f"records[{flat_index}] unbound map_name={map_name!r}"
        )
    if map_epoch != int(expected_map_epoch):
        raise Multires500ProofError(
            f"records[{flat_index}] unbound map_epoch={map_epoch}"
        )
    if policy_version != int(expected_policy_version):
        raise Multires500ProofError(
            f"records[{flat_index}] unbound policy_version={policy_version}"
        )

    if item["rust_provider_schema"] != RUST_PROVIDER_SCHEMA:
        raise Multires500ProofError(
            f"records[{flat_index}] noncanonical rust_provider_schema"
        )
    if item["python_collector_schema"] != PYTHON_COLLECTOR_SCHEMA:
        raise Multires500ProofError(
            f"records[{flat_index}] noncanonical python_collector_schema"
        )

    obs = tuple(_float_list(
        list(item["observation"]), OBS_DIM,
        f"records[{flat_index}].observation",
    ))
    observation_dyn = obs[DYN.slice]
    rust_feature_digest_value = item["rust_feature_digest"]
    if not _valid_sha256(rust_feature_digest_value):
        raise Multires500ProofError(
            f"records[{flat_index}] rust_feature_digest is not a lowercase SHA-256"
        )
    if "rust_features" in item:
        emitted_rust_features = tuple(_float_list(
            item["rust_features"],
            _RUST_FEATURE_WIDTH,
            f"records[{flat_index}].rust_features",
        ))
        if emitted_rust_features != observation_dyn:
            raise Multires500ProofError(
                f"records[{flat_index}] rust_features do not exactly equal "
                "observation[DYN.slice]"
            )
    recomputed_feature = rust_feature_digest(observation_dyn)
    if recomputed_feature != rust_feature_digest_value:
        raise Multires500ProofError(
            f"records[{flat_index}] rust_feature_digest does not match "
            "observation[DYN.slice]"
        )

    emitted_identity = item["identity_digest"]
    if not _valid_sha256(emitted_identity):
        raise Multires500ProofError(
            f"records[{flat_index}] identity_digest is not a lowercase SHA-256"
        )
    recomputed_identity = transition_identity_digest(
        client_id=client_id,
        server_frame=server_frame,
        batch_round_id=batch_round_id,
        policy_version=policy_version,
        map_name=map_name,
        map_epoch=map_epoch,
        atlas_sha256=atlas_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
    )
    if emitted_identity != recomputed_identity:
        raise Multires500ProofError(
            f"records[{flat_index}] identity_digest does not match identity fields"
        )

    # Dense contiguous indices always; exact 4x125 client/frame layout is
    # enforced in admit_canonical_records after the first round is known.
    if index != flat_index:
        raise Multires500ProofError(
            f"records[{flat_index}] reordered/noncanonical index={index} "
            f"(expected {flat_index})"
        )
    if server_frame <= 0:
        raise Multires500ProofError(
            f"records[{flat_index}] server_frame must be a positive engine frame"
        )
    if require_exact_layout:
        time_step = int(flat_index) // REQUIRED_CLIENT_COUNT
        if not 0 <= batch_round_id < TRANSITIONS_PER_CLIENT:
            raise Multires500ProofError(
                f"records[{flat_index}] batch_round_id={batch_round_id} outside "
                f"0..{TRANSITIONS_PER_CLIENT - 1}"
            )
        if batch_round_id != time_step:
            raise Multires500ProofError(
                f"records[{flat_index}] batch_round_id={batch_round_id} "
                f"!= time_step={time_step} (time-major/client-minor)"
            )

    act = tuple(_float_list(list(item["action"]), ACTION_DIM, f"records[{flat_index}].action"))
    reward_raw = item["reward"]
    if isinstance(reward_raw, bool) or not isinstance(reward_raw, (int, float)):
        raise Multires500ProofError(f"records[{flat_index}] reward must be numeric")
    reward_value = float(reward_raw)
    if reward_value != reward_value or reward_value in (float("inf"), float("-inf")):
        raise Multires500ProofError(f"records[{flat_index}] reward must be finite")

    return TransitionRecord(
        index=index,
        observation=obs,
        action=act,
        reward=reward_value,
        client_id=str(client_id),
        server_frame=server_frame,
        batch_round_id=batch_round_id,
        policy_version=policy_version,
        map_name=str(map_name),
        map_epoch=map_epoch,
        atlas_sha256=str(atlas_sha256),
        runtime_manifest_sha256=str(runtime_manifest_sha256),
        rust_provider_schema=RUST_PROVIDER_SCHEMA,
        rust_feature_digest=recomputed_feature,
        python_collector_schema=PYTHON_COLLECTOR_SCHEMA,
        identity_digest=recomputed_identity,
    )


def admit_canonical_records(
    raw_records: Any,
    *,
    expected_count: int,
    expected_atlas_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_map_name: str,
    expected_map_epoch: int,
    expected_policy_version: int,
    expected_trajectory_sha256: Optional[str] = None,
) -> tuple[TransitionRecord, ...]:
    """Admit a full trajectory of canonical records; never digest-only."""
    if raw_records is None:
        raise Multires500ProofError(
            "canonical transition records are missing; digest-only assertion "
            "is not admissible"
        )
    if not isinstance(raw_records, list):
        raise Multires500ProofError(
            "records must be a list of canonical transition objects"
        )
    if len(raw_records) == 0:
        raise Multires500ProofError(
            "records list is empty; digest-only assertion is not admissible"
        )
    if len(raw_records) != int(expected_count):
        raise Multires500ProofError(
            f"records length {len(raw_records)} != expected {expected_count} "
            f"(missing/extra/truncated trajectory)"
        )
    require_exact_layout = int(expected_count) == REQUIRED_TRANSITION_COUNT

    built: list[TransitionRecord] = []
    seen_identity: set[str] = set()
    seen_keys: set[tuple[str, int, int]] = set()

    for flat_index, item in enumerate(raw_records):
        record = parse_canonical_transition_record(
            item,
            flat_index=flat_index,
            expected_atlas_sha256=expected_atlas_sha256,
            expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
            expected_map_name=expected_map_name,
            expected_map_epoch=expected_map_epoch,
            expected_policy_version=expected_policy_version,
            require_exact_layout=require_exact_layout,
        )
        if record.identity_digest in seen_identity:
            raise Multires500ProofError(
                f"records[{flat_index}] duplicates identity_digest"
            )
        key = (record.client_id, record.server_frame, record.batch_round_id)
        if key in seen_keys:
            raise Multires500ProofError(
                f"records[{flat_index}] duplicates client/frame/round identity"
            )
        seen_identity.add(record.identity_digest)
        seen_keys.add(key)
        built.append(record)

    if require_exact_layout:
        enforce_exact_500_layout(built)

    records = tuple(built)
    recomputed = trajectory_sha256(records)
    if expected_trajectory_sha256 is not None:
        if not _valid_sha256(expected_trajectory_sha256):
            raise Multires500ProofError("trajectory_sha256 is invalid")
        if recomputed != expected_trajectory_sha256:
            raise Multires500ProofError(
                "trajectory_sha256 does not match recomputation from canonical records"
            )
    return records


def _pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process table entry exists but is not owned by us.
        return True
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class ProcessIdentity:
    role: str
    pid: int
    start_ticks: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pid": self.pid,
            "start_ticks": self.start_ticks,
        }


def _proc_state_start_ticks(pid: int) -> Optional[tuple[str, int]]:
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return str(tail[0]), int(tail[19])
    except (OSError, IndexError, ValueError):
        return None


def process_identity_alive(identity: ProcessIdentity) -> bool:
    """True only for the same Linux process instance, never PID reuse."""
    current = _proc_state_start_ticks(identity.pid)
    return (
        current is not None
        and current[0] != "Z"
        and current[1] == identity.start_ticks
    )


def prove_process_records_dead(
    records: Sequence[ProcessIdentity], *, where: str
) -> tuple[ProcessIdentity, ...]:
    if not records:
        raise Multires500ProofError(
            f"{where}: process_records are empty; PID integers alone are insufficient"
        )
    alive = [record.to_mapping() for record in records if process_identity_alive(record)]
    if alive:
        raise Multires500ProofError(
            f"{where}: same-identity processes survived teardown: {alive}"
        )
    return tuple(records)


def _ppid_of(pid: int) -> Optional[int]:
    """Read parent PID from Linux /proc; None when the entry is gone/unreadable."""
    status_path = Path(f"/proc/{int(pid)}/status")
    try:
        text = status_path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    for line in text.splitlines():
        if line.startswith("PPid:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def discover_descendant_pids(root_pid: int) -> tuple[int, ...]:
    """Scoped Linux /proc PPID walk under ``root_pid`` only (no pattern matching)."""
    root = int(root_pid)
    if root <= 0:
        return ()
    children_by_ppid: dict[int, list[int]] = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid <= 0 or pid == root:
                continue
            ppid = _ppid_of(pid)
            if ppid is None:
                continue
            children_by_ppid.setdefault(int(ppid), []).append(pid)
    except OSError:
        return ()

    found: list[int] = []
    seen: set[int] = set()
    stack = list(children_by_ppid.get(root, ()))
    while stack:
        pid = stack.pop()
        if pid in seen or pid == root:
            continue
        seen.add(pid)
        found.append(pid)
        stack.extend(children_by_ppid.get(pid, ()))
    return tuple(sorted(found))


def terminate_owned_pid_tree(
    root_pid: int,
    *,
    extra_pids: Sequence[int] = (),
    grace_seconds: float = 2.0,
) -> CleanupReport:
    """Terminate a root PID plus scoped /proc descendants and optional extras."""
    root = int(root_pid)
    notes: list[str] = []
    killed: list[int] = []
    scoped: list[int] = []
    targets: list[int] = []
    seen: set[int] = set()

    def _add(pid: int) -> None:
        value = int(pid)
        if value <= 0 or value in seen:
            return
        seen.add(value)
        targets.append(value)
        scoped.append(value)

    if root > 0:
        _add(root)
        for child in discover_descendant_pids(root):
            _add(child)
    for pid in extra_pids:
        _add(int(pid))

    # Children first, then root, so session leaders do not reparent mid-walk.
    kill_order = list(reversed(targets))
    for pid in kill_order:
        if not _pid_is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as error:
            notes.append(f"SIGTERM pid={pid}: {error}")

    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        if not any(_pid_is_alive(pid) for pid in targets):
            break
        time.sleep(0.05)

    for pid in kill_order:
        if not _pid_is_alive(pid):
            if pid in targets:
                killed.append(pid)
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError) as error:
            notes.append(f"SIGKILL pid={pid}: {error}")
        try:
            # Reap if this PID is a direct child of the current process.
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass

    time.sleep(0.05)
    orphans = sum(1 for pid in targets if _pid_is_alive(pid))
    for pid in targets:
        if not _pid_is_alive(pid) and pid not in killed:
            killed.append(pid)
    return CleanupReport(
        cleaned=orphans == 0,
        orphan_processes=orphans,
        killed_process_ids=tuple(dict.fromkeys(killed)),
        notes=tuple(notes),
        scoped_child_pids=tuple(scoped),
        verified_dead_process_ids=tuple(
            pid for pid in targets if not _pid_is_alive(pid)
        ),
        still_alive_process_ids=tuple(
            pid for pid in targets if _pid_is_alive(pid)
        ),
    )


def prove_process_ids_dead(
    process_ids: Sequence[int],
    *,
    where: str,
) -> tuple[int, ...]:
    """Fail closed unless every reported/launched PID is proven not alive."""
    if not process_ids:
        raise Multires500ProofError(
            f"{where}: process_ids/launched PIDs are empty; cannot prove teardown"
        )
    still_alive = [int(pid) for pid in process_ids if _pid_is_alive(int(pid))]
    if still_alive:
        raise Multires500ProofError(
            f"{where}: owned process_ids still alive after teardown: {still_alive}"
        )
    return tuple(int(pid) for pid in process_ids)


@dataclass(frozen=True)
class CleanupReport:
    cleaned: bool
    orphan_processes: int
    killed_process_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()
    scoped_child_pids: tuple[int, ...] = ()
    verified_dead_process_ids: tuple[int, ...] = ()
    still_alive_process_ids: tuple[int, ...] = ()

    @property
    def orphan_processes_after_teardown(self) -> int:
        return int(self.orphan_processes)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cleaned": self.cleaned,
            "orphan_processes": self.orphan_processes,
            "orphan_processes_after_teardown": self.orphan_processes_after_teardown,
            "killed_process_ids": list(self.killed_process_ids),
            "scoped_child_pids": list(self.scoped_child_pids),
            "verified_dead_process_ids": list(self.verified_dead_process_ids),
            "still_alive_process_ids": list(self.still_alive_process_ids),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LaunchResult:
    launch_id: str
    seed: int
    game_seed: int
    language: str
    transition_count: int
    trajectory_sha256: str
    records: tuple[TransitionRecord, ...]
    atlas_sha256: str
    map_epoch: int
    policy_version: int
    partial_admissions: int = 0
    stale_admissions: int = 0
    resync_admissions: int = 0
    process_ids: tuple[int, ...] = ()
    process_records: tuple[ProcessIdentity, ...] = ()
    collector: str = COLLECTOR_CLASS_NAME
    spatial_provider: str = SPATIAL_PROVIDER_CLASS_NAME
    lattice_crate: str = LATTICE_CRATE_NAME

    def to_mapping(self) -> dict[str, Any]:
        return {
            "launch_id": self.launch_id,
            "seed": self.seed,
            "game_seed": self.game_seed,
            "language": self.language,
            "transition_count": self.transition_count,
            "trajectory_sha256": self.trajectory_sha256,
            "atlas_sha256": self.atlas_sha256,
            "map_epoch": self.map_epoch,
            "policy_version": self.policy_version,
            "partial_admissions": self.partial_admissions,
            "stale_admissions": self.stale_admissions,
            "resync_admissions": self.resync_admissions,
            "process_ids": list(self.process_ids),
            "process_records": [
                record.to_mapping() for record in self.process_records
            ],
            "collector": self.collector,
            "spatial_provider": self.spatial_provider,
            "lattice_crate": self.lattice_crate,
        }


@dataclass(frozen=True)
class LaunchRequest:
    launch_id: str
    seed: int
    game_seed: int
    language: str
    transition_count: int
    policy_version: int
    map_name: str
    map_epoch: int
    atlas_sha256: str
    runtime_manifest_sha256: str
    timeout_seconds: float


class LaunchTransport(Protocol):
    def launch(self, request: LaunchRequest) -> LaunchResult:
        ...

    def cleanup(self) -> CleanupReport:
        ...


@dataclass
class ProofConfig:
    mode: str
    seed: int
    game_seed: int
    divergence_game_seed: int
    transition_count: int = REQUIRED_TRANSITION_COUNT
    policy_version: int = 1
    map_name: str = "mlstage_0001"
    map_epoch: int = 1
    timeout_seconds: float = 600.0
    out_path: Optional[Path] = None
    q2ded: Optional[Path] = None
    client_binary: Optional[Path] = None
    runtime_root: Optional[Path] = None
    bundle_manifest: Optional[Path] = None
    objectives: Optional[Path] = None
    atlas_bin: Optional[Path] = None
    atlas_catalog: Optional[Path] = None
    expected_atlas_catalog_sha256: Optional[str] = None
    checkpoint: Optional[Path] = None
    training_manifest: Optional[Path] = None
    runtime_evidence: Optional[Path] = None
    evidence_dir: Optional[Path] = None
    legacy_selector: Optional[str] = None
    # Absolute operational one-run trainer argv prefix (no shell).
    trainer_executable: Optional[Path] = None
    trainer_args: tuple[str, ...] = ()
    # Test-only injection hooks — rejected in production mode.
    production_process_factory: Optional[Callable[..., Any]] = None
    production_collect_fn: Optional[Callable[..., Any]] = None


@dataclass(frozen=True)
class ProductionAdmission:
    atlas_sha256: str
    atlas_catalog_sha256: str
    runtime_manifest_sha256: str
    objective_identity_sha256: str
    training_manifest_sha256: str
    checkpoint_sha256: str
    bundle_version: int
    artifact_state: str
    policy_generation: str
    checkpoint_format: str
    checkpoint_initialization: str
    checkpoint_training_step: int
    checkpoint_lineage_root_sha256: str
    b4_protocol_generation: int
    qm3c_causal_magic: int
    client_wire_version: int
    observation_magic: int
    action_magic: int
    teacher_version: int
    rollout_schema: str
    map_name: str
    q2ded: Path
    client_binary: Path
    runtime_root: Path
    bundle_manifest: Path
    objectives: Path
    atlas_bin: Path
    atlas_catalog: Path
    checkpoint: Path
    training_manifest: Path
    runtime_evidence: Path
    trainer_argv_prefix: tuple[str, ...]
    evidence_dir: Path


def _canonical_training_configuration(
    path: Path,
) -> tuple[MultiresTrainingConfiguration, str]:
    document = _load_json_object(path, "training_manifest")
    allowed = {"schema", "reward", "guide_dropout", "ppo"}
    if set(document) not in (allowed, allowed | {"sha256"}):
        raise Multires500ProofError(
            "training_manifest must contain only the canonical training fields"
        )
    if document.get("schema") != TRAINING_CONFIG_SCHEMA:
        raise Multires500ProofError(
            f"training_manifest schema must be {TRAINING_CONFIG_SCHEMA!r}"
        )
    if not all(
        isinstance(document.get(name), Mapping)
        for name in ("reward", "guide_dropout", "ppo")
    ):
        raise Multires500ProofError(
            "training_manifest reward/guide_dropout/ppo sections must be objects"
        )
    try:
        configuration = MultiresTrainingConfiguration.create(
            reward=document["reward"],
            guide_dropout=document["guide_dropout"],
            ppo=document["ppo"],
        )
    except ValueError as error:
        raise Multires500ProofError(
            f"training_manifest is not canonical training data: {error}"
        ) from error
    declared = document.get("sha256")
    if declared is not None and declared != configuration.sha256:
        raise Multires500ProofError(
            "training_manifest declared SHA-256 differs from its canonical body"
        )
    return configuration, configuration.sha256


def _configured_optimizer(runtime_root: Path) -> Mapping[str, Any]:
    runtime_config = _load_json_object(
        _require_path(
            runtime_root / "multires-one-run.json", "multires one-run runtime config"
        ),
        "multires one-run runtime config",
    )
    optimizer = runtime_config.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != {"class", "learning_rate", "kwargs"}
        or optimizer.get("class") != "torch.optim.Adam"
        or isinstance(optimizer.get("learning_rate"), bool)
        or not isinstance(optimizer.get("learning_rate"), (int, float))
        or not math.isfinite(float(optimizer["learning_rate"]))
        or float(optimizer["learning_rate"]) <= 0.0
        or not isinstance(optimizer.get("kwargs"), Mapping)
        or "lr" in optimizer["kwargs"]
    ):
        raise Multires500ProofError(
            "multires one-run runtime optimizer must be exact configured Adam"
        )
    return optimizer


def _validate_production_checkpoint(
    checkpoint: Path,
    *,
    atlas_catalog_sha256: str,
    runtime_manifest_sha256: str,
    training_configuration: MultiresTrainingConfiguration,
    optimizer_configuration: Mapping[str, Any],
) -> tuple[str, int, str]:
    """Load the real safe envelope; byte markers and sidecars have no authority."""
    try:
        import torch
        from models.multires_policy import MultiresQ2BotPolicy

        policy = MultiresQ2BotPolicy().to(torch.device("cpu"))
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(optimizer_configuration["learning_rate"]),
            **dict(optimizer_configuration["kwargs"]),
        )
        manifest = load_attested_checkpoint(
            checkpoint,
            policy,
            expected_atlas_catalog_sha256=atlas_catalog_sha256,
            expected_runtime_manifest_sha256=runtime_manifest_sha256,
            expected_training_config=training_configuration,
            optimizer=optimizer,
            map_location="cpu",
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise Multires500ProofError(
            f"checkpoint is not a real attested weights-only step-zero envelope: {error}"
        ) from error
    if (
        manifest.initialization != "random"
        or manifest.training_step != 0
        or optimizer.state_dict().get("state") != {}
        or not _valid_sha256(manifest.lineage_root_sha256)
    ):
        raise Multires500ProofError(
            "production proof requires random step-zero checkpoint and fresh optimizer"
        )
    return (
        manifest.initialization,
        int(manifest.training_step),
        manifest.lineage_root_sha256,
    )


def _reject_legacy_selector(value: Optional[str]) -> None:
    if value is None:
        return
    lowered = value.strip().lower()
    for forbidden in FORBIDDEN_LEGACY_SELECTORS:
        if forbidden.lower() in lowered:
            raise Multires500ProofError(
                f"legacy selector rejected: {value!r} matches {forbidden!r}"
            )
    if "train.ppo" in lowered or "models.policy" in lowered:
        raise Multires500ProofError(f"legacy trainer path rejected: {value!r}")


def resolve_trainer_argv(
    *,
    trainer_executable: Optional[Path],
    trainer_args: Sequence[str] = (),
    trainer_command: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Resolve an absolute trainer argv prefix with no shell interpolation."""
    if trainer_command is not None:
        if trainer_executable is not None or trainer_args:
            raise Multires500ProofError(
                "use either --trainer_command or --trainer_executable/--trainer_arg, not both"
            )
        tokens = [str(item) for item in trainer_command]
        if not tokens:
            raise Multires500ProofError("trainer_command must include an absolute executable")
        executable = Path(tokens[0]).expanduser()
        if not executable.is_absolute():
            raise Multires500ProofError(
                f"trainer_command executable must be absolute: {tokens[0]!r}"
            )
        resolved_exec = _require_path(executable, "trainer_command[0]", executable=True)
        rest = [
            _argv_token_safe(token, f"trainer_command[{index + 1}]")
            for index, token in enumerate(tokens[1:])
        ]
        for token in rest:
            _reject_legacy_selector(token)
        return (str(resolved_exec), *rest)

    if trainer_executable is None:
        raise Multires500ProofError(
            "production mode requires absolute --trainer_executable or --trainer_command"
        )
    resolved_exec = _require_path(
        trainer_executable, "trainer_executable", executable=True
    )
    rest = [
        _argv_token_safe(str(token), f"trainer_arg[{index}]")
        for index, token in enumerate(trainer_args)
    ]
    for token in rest:
        _reject_legacy_selector(token)
    _reject_legacy_selector(str(resolved_exec))
    return (str(resolved_exec), *rest)


def admit_production_config(config: ProofConfig) -> ProductionAdmission:
    if config.mode != MODE_PRODUCTION:
        raise Multires500ProofError("production admission requires mode=production")
    _reject_legacy_selector(config.legacy_selector)

    if config.production_collect_fn is not None:
        raise Multires500ProofError(
            "production mode rejects production_collect_fn callback injection; "
            "drive the operational one-run trainer via --trainer_executable/--trainer_command"
        )
    if config.production_process_factory is not None:
        raise Multires500ProofError(
            "production mode rejects production_process_factory injection; "
            "the CLI launches the operational trainer subprocess itself"
        )

    q2ded = _require_path(config.q2ded, "q2ded", executable=True)
    client_binary = _require_path(config.client_binary, "client_binary", executable=True)
    runtime_root = _require_dir(config.runtime_root, "runtime_root")
    bundle_manifest = _require_path(config.bundle_manifest, "bundle_manifest")
    objectives = _require_path(config.objectives, "objectives")
    atlas_bin = _require_path(config.atlas_bin, "atlas_bin")
    atlas_catalog = _require_path(config.atlas_catalog, "atlas_catalog")
    if not _valid_sha256(config.expected_atlas_catalog_sha256):
        raise Multires500ProofError("Atlas catalog content digest is invalid")
    checkpoint = _require_path(config.checkpoint, "checkpoint")
    training_manifest = _require_path(config.training_manifest, "training_manifest")
    runtime_evidence_path = _require_path(config.runtime_evidence, "runtime_evidence")
    trainer_argv_prefix = resolve_trainer_argv(
        trainer_executable=config.trainer_executable,
        trainer_args=config.trainer_args,
    )

    if config.evidence_dir is not None:
        evidence_dir = _require_dir(config.evidence_dir, "evidence_dir")
    elif config.out_path is not None:
        out_parent = config.out_path.expanduser()
        if not out_parent.is_absolute():
            raise Multires500ProofError(f"--out must be an absolute path: {config.out_path}")
        evidence_dir = out_parent.parent.resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise Multires500ProofError(
            "production mode requires --evidence_dir or absolute --out for per-launch artifacts"
        )

    bundle = _load_json_object(bundle_manifest, "bundle_manifest")
    if int(bundle.get("bundle_version", -1)) != 3:
        raise Multires500ProofError("bundle_manifest must declare bundle_version=3")
    artifact_state = bundle.get("artifact_state", bundle.get("state"))
    if artifact_state != "admitted":
        raise Multires500ProofError(
            f"bundle artifact_state={artifact_state!r}: only 'admitted' is production-admissible"
        )
    map_name = bundle.get("name") or bundle.get("map_name") or config.map_name
    if not isinstance(map_name, str) or not map_name:
        raise Multires500ProofError("bundle_manifest is missing map name")

    _load_json_object(objectives, "objectives")
    objective_identity = _file_sha256(objectives)
    if not _valid_sha256(objective_identity):
        raise Multires500ProofError("objective identity digest is invalid")

    atlas_sha256 = bundle.get("atlas_sha256")
    if not _valid_sha256(atlas_sha256):
        atlas_sha256 = _file_sha256(atlas_bin)
    if not _valid_sha256(atlas_sha256):
        raise Multires500ProofError("atlas_sha256 is invalid")
    if _file_sha256(atlas_bin) != atlas_sha256:
        raise Multires500ProofError("atlas_bin digest differs from admitted atlas_sha256")

    atlas_catalog_admission = _admit_atlas_catalog(
        atlas_catalog, str(config.expected_atlas_catalog_sha256)
    )
    selected_catalog_map = atlas_catalog_admission.by_name().get(str(map_name))
    if selected_catalog_map is None:
        raise Multires500ProofError(
            f"active map {map_name!r} is not present in the admitted Atlas catalog"
        )
    if (
        selected_catalog_map.atlas != atlas_bin
        or selected_catalog_map.atlas_sha256 != atlas_sha256
        or selected_catalog_map.bundle_manifest != bundle_manifest
        or selected_catalog_map.objectives != objectives
    ):
        raise Multires500ProofError(
            "active map Atlas/bundle/objective tuple differs from the admitted catalog"
        )

    runtime_evidence = _load_json_object(runtime_evidence_path, "runtime_evidence")
    if int(runtime_evidence.get("client_wire_version", -1)) == LEGACY_CLIENT_WIRE_VERSION:
        raise Multires500ProofError("legacy client wire version is not admissible")
    if int(runtime_evidence.get("observation_magic", -1)) == LEGACY_OBSERVATION_MAGIC:
        raise Multires500ProofError("legacy observation magic is not admissible")
    if runtime_evidence.get("rollout_schema") == LEGACY_ROLLOUT_SCHEMA:
        raise Multires500ProofError("legacy rollout schema is not admissible")
    if int(runtime_evidence.get("causal_magic", -1)) != B4_CAUSAL_MAGIC:
        raise Multires500ProofError(
            f"QM3C causal magic mismatch: expected {B4_CAUSAL_MAGIC:#x}"
        )
    if int(runtime_evidence.get("protocol_generation", -1)) != B4_PROTOCOL_GENERATION:
        raise Multires500ProofError(
            f"B4 protocol generation mismatch: expected {B4_PROTOCOL_GENERATION}"
        )
    validated = validate_runtime_evidence(
        runtime_evidence, expected_atlas_sha256=str(atlas_sha256)
    )

    training_configuration, training_manifest_sha256 = (
        _canonical_training_configuration(training_manifest)
    )
    optimizer_configuration = _configured_optimizer(runtime_root)

    try:
        checkpoint_payload = checkpoint.read_bytes()
    except OSError as error:
        raise Multires500ProofError(f"cannot read checkpoint: {error}") from error
    if not checkpoint_payload:
        raise Multires500ProofError("checkpoint is empty")
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    (
        checkpoint_initialization,
        checkpoint_training_step,
        checkpoint_lineage_root_sha256,
    ) = _validate_production_checkpoint(
        checkpoint,
        atlas_catalog_sha256=atlas_catalog_admission.atlas_catalog_sha256,
        runtime_manifest_sha256=validated.runtime_manifest_sha256,
        training_configuration=training_configuration,
        optimizer_configuration=optimizer_configuration,
    )

    return ProductionAdmission(
        atlas_sha256=str(atlas_sha256),
        atlas_catalog_sha256=atlas_catalog_admission.atlas_catalog_sha256,
        runtime_manifest_sha256=validated.runtime_manifest_sha256,
        objective_identity_sha256=str(objective_identity),
        training_manifest_sha256=str(training_manifest_sha256),
        checkpoint_sha256=checkpoint_sha256,
        bundle_version=3,
        artifact_state="admitted",
        policy_generation=POLICY_GENERATION,
        checkpoint_format=CHECKPOINT_FORMAT,
        checkpoint_initialization=checkpoint_initialization,
        checkpoint_training_step=checkpoint_training_step,
        checkpoint_lineage_root_sha256=checkpoint_lineage_root_sha256,
        b4_protocol_generation=B4_PROTOCOL_GENERATION,
        qm3c_causal_magic=B4_CAUSAL_MAGIC,
        client_wire_version=validated.client_wire_version,
        observation_magic=validated.observation_magic,
        action_magic=validated.action_magic,
        teacher_version=validated.teacher_version,
        rollout_schema=validated.rollout_schema,
        map_name=str(map_name),
        q2ded=q2ded,
        client_binary=client_binary,
        runtime_root=runtime_root,
        bundle_manifest=bundle_manifest,
        objectives=objectives,
        atlas_bin=atlas_bin,
        atlas_catalog=atlas_catalog,
        checkpoint=checkpoint,
        training_manifest=training_manifest,
        runtime_evidence=runtime_evidence_path,
        trainer_argv_prefix=trainer_argv_prefix,
        evidence_dir=evidence_dir,
    )


@dataclass
class ManagedProcessGroup:
    processes: list[subprocess.Popen] = field(default_factory=list)
    extra_owned_pids: list[int] = field(default_factory=list)

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(process.pid for process in self.processes if process.pid)

    def terminate_all(self, *, grace_seconds: float = 2.0) -> CleanupReport:
        """Kill owned trainer processes and their /proc-discovered descendants."""
        killed: list[int] = []
        notes: list[str] = []
        scoped: list[int] = []
        verified_dead: list[int] = []
        still_alive: list[int] = []
        orphan_total = 0
        cleaned = True

        roots = [int(process.pid) for process in self.processes if process.pid]
        # Also terminate any previously recorded extras (e.g. trainer-reported PIDs).
        extras = [int(pid) for pid in self.extra_owned_pids if int(pid) > 0]

        if not roots and extras:
            for pid in extras:
                report = terminate_owned_pid_tree(
                    pid, grace_seconds=grace_seconds
                )
                killed.extend(report.killed_process_ids)
                notes.extend(report.notes)
                scoped.extend(report.scoped_child_pids)
                verified_dead.extend(report.verified_dead_process_ids)
                still_alive.extend(report.still_alive_process_ids)
                orphan_total += report.orphan_processes
                cleaned = cleaned and report.cleaned
        else:
            for root in roots:
                report = terminate_owned_pid_tree(
                    root,
                    extra_pids=extras,
                    grace_seconds=grace_seconds,
                )
                killed.extend(report.killed_process_ids)
                notes.extend(report.notes)
                scoped.extend(report.scoped_child_pids)
                verified_dead.extend(report.verified_dead_process_ids)
                still_alive.extend(report.still_alive_process_ids)
                orphan_total += report.orphan_processes
                cleaned = cleaned and report.cleaned
                # Extras are only attached once (to the first root).
                extras = []

        for process in list(self.processes):
            try:
                if process.poll() is None:
                    process.wait(timeout=0.2)
            except Exception:
                pass

        self.processes.clear()
        self.extra_owned_pids.clear()
        # De-duplicate while preserving order.
        def _unique(values: Sequence[int]) -> tuple[int, ...]:
            return tuple(dict.fromkeys(int(value) for value in values))

        still_alive_unique = _unique(
            pid for pid in still_alive if _pid_is_alive(pid)
        )
        orphan_total = max(orphan_total, len(still_alive_unique))
        cleaned = cleaned and orphan_total == 0
        return CleanupReport(
            cleaned=cleaned,
            orphan_processes=orphan_total,
            killed_process_ids=_unique(killed),
            notes=tuple(notes),
            scoped_child_pids=_unique(scoped),
            verified_dead_process_ids=_unique(verified_dead),
            still_alive_process_ids=still_alive_unique,
        )


def build_one_run_argv(
    *,
    trainer_argv_prefix: Sequence[str],
    request: LaunchRequest,
    admission: ProductionAdmission,
    out_path: Path,
) -> list[str]:
    """Construct the fail-closed operational one-run argv (list form, no shell).

    Documented M4 contract (not yet implemented as a stable CLI module):
    expose ``python3 -m train.multires_one_run`` (or an absolute executable)
    that accepts every flag below, launches MultiresSynchronousCollector over
    RustAtlasSpatialProvider/q2_lattice with B4/QM3C wire evidence, collects
    exactly ``--transition_count`` transitions, and writes
    ``ONE_RUN_SCHEMA`` JSON to ``--out`` with dual-boundary attestation and
    trajectory digest.  Required methods to wire on the M4 side:

      MultiresTrainerRuntime.fresh|resume(...)
      MultiresLiveTrainer(..., deterministic_collection=True)
      MultiresLiveTrainer.collector.collect(policy_version=...)
        or MultiresLiveTrainer.train_update / a dedicated collect_only path
      emit ONE_RUN_SCHEMA with received_inputs echo + trajectory_sha256
    """
    if not out_path.is_absolute():
        raise Multires500ProofError(f"one-run --out must be absolute: {out_path}")
    if int(request.transition_count) != REQUIRED_TRANSITION_COUNT:
        # Still pass the requested count so misconfigured trainers surface
        # evidence; the proof rejects non-500 results later.
        pass
    argv = [str(token) for token in trainer_argv_prefix]
    for token in argv:
        _argv_token_safe(token, "trainer argv")
    required = [
        ("--seed", str(int(request.seed))),
        ("--game_seed", str(int(request.game_seed))),
        ("--q2ded", str(admission.q2ded)),
        ("--client_binary", str(admission.client_binary)),
        ("--runtime_root", str(admission.runtime_root)),
        ("--bundle_manifest", str(admission.bundle_manifest)),
        ("--objectives", str(admission.objectives)),
        ("--atlas_bin", str(admission.atlas_bin)),
        ("--atlas_catalog", str(admission.atlas_catalog)),
        ("--checkpoint", str(admission.checkpoint)),
        ("--training_manifest", str(admission.training_manifest)),
        ("--runtime_evidence", str(admission.runtime_evidence)),
        ("--transition_count", str(int(request.transition_count))),
        ("--policy_version", str(int(request.policy_version))),
        ("--map_epoch", str(int(request.map_epoch))),
        ("--map_name", str(request.map_name)),
        ("--out", str(out_path)),
        ("--launch_id", str(request.launch_id)),
        ("--expected_atlas_sha256", str(admission.atlas_sha256)),
        (
            "--expected_atlas_catalog_sha256",
            str(admission.atlas_catalog_sha256),
        ),
        (
            "--expected_runtime_manifest_sha256",
            str(admission.runtime_manifest_sha256),
        ),
    ]
    for flag, value in required:
        _argv_token_safe(flag, "one-run flag")
        _argv_token_safe(value, f"one-run {flag}")
        argv.extend([flag, value])
    return argv


def expected_received_inputs(
    request: LaunchRequest,
    admission: ProductionAdmission,
    out_path: Path,
) -> dict[str, Any]:
    return {
        "seed": int(request.seed),
        "game_seed": int(request.game_seed),
        "q2ded": str(admission.q2ded),
        "client_binary": str(admission.client_binary),
        "runtime_root": str(admission.runtime_root),
        "bundle_manifest": str(admission.bundle_manifest),
        "objectives": str(admission.objectives),
        "atlas_bin": str(admission.atlas_bin),
        "atlas_catalog": str(admission.atlas_catalog),
        "checkpoint": str(admission.checkpoint),
        "training_manifest": str(admission.training_manifest),
        "runtime_evidence": str(admission.runtime_evidence),
        "transition_count": int(request.transition_count),
        "policy_version": int(request.policy_version),
        "map_epoch": int(request.map_epoch),
        "map_name": str(request.map_name),
        "out": str(out_path),
        "launch_id": str(request.launch_id),
        "expected_atlas_sha256": str(admission.atlas_sha256),
        "expected_atlas_catalog_sha256": str(admission.atlas_catalog_sha256),
        "expected_runtime_manifest_sha256": str(admission.runtime_manifest_sha256),
    }


def parse_one_run_output(
    payload: Mapping[str, Any],
    *,
    request: LaunchRequest,
    admission: ProductionAdmission,
    out_path: Path,
) -> LaunchResult:
    """Admit one operational one-run trainer artifact."""
    _reject_placeholders(payload, "one-run output")
    if payload.get("schema") != ONE_RUN_SCHEMA:
        raise Multires500ProofError(
            f"one-run schema must be {ONE_RUN_SCHEMA!r}; got {payload.get('schema')!r}"
        )
    if int(payload.get("protocol_version", -1)) != ONE_RUN_PROTOCOL_VERSION:
        raise Multires500ProofError(
            f"one-run protocol_version must be {ONE_RUN_PROTOCOL_VERSION}"
        )
    if payload.get("synthetic") is True:
        raise Multires500ProofError(
            "one-run output reports synthetic=true; production rejects synthetic data"
        )
    if payload.get("legacy") is True:
        raise Multires500ProofError("one-run output reports legacy=true")

    collector = payload.get("collector")
    if collector != COLLECTOR_CLASS_NAME:
        raise Multires500ProofError(
            f"one-run collector must be {COLLECTOR_CLASS_NAME!r}; got {collector!r}"
        )
    if payload.get("python_collector_schema") != PYTHON_COLLECTOR_SCHEMA:
        raise Multires500ProofError(
            "one-run python_collector_schema must be "
            f"{PYTHON_COLLECTOR_SCHEMA!r}"
        )
    provider = payload.get("spatial_provider")
    if provider != SPATIAL_PROVIDER_CLASS_NAME:
        raise Multires500ProofError(
            f"one-run spatial_provider must be {SPATIAL_PROVIDER_CLASS_NAME!r}; "
            f"got {provider!r}"
        )
    if payload.get("rust_provider_schema") != RUST_PROVIDER_SCHEMA:
        raise Multires500ProofError(
            f"one-run rust_provider_schema must be {RUST_PROVIDER_SCHEMA!r}"
        )
    lattice = payload.get("lattice_crate")
    if lattice not in (LATTICE_CRATE_NAME, "q2_lattice_rs", "q2-lattice"):
        raise Multires500ProofError(
            f"one-run lattice_crate must attest {LATTICE_CRATE_NAME!r}; got {lattice!r}"
        )

    if int(payload.get("b4_protocol_generation", -1)) != B4_PROTOCOL_GENERATION:
        raise Multires500ProofError("one-run B4 protocol generation mismatch")
    if int(payload.get("qm3c_causal_magic", -1)) != B4_CAUSAL_MAGIC:
        raise Multires500ProofError("one-run QM3C causal magic mismatch")
    if int(payload.get("client_wire_version", -1)) != admission.client_wire_version:
        raise Multires500ProofError("one-run client_wire_version mismatch")
    if payload.get("policy_generation") != POLICY_GENERATION:
        raise Multires500ProofError("one-run policy_generation mismatch")
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise Multires500ProofError("one-run checkpoint_format mismatch")
    if payload.get("checkpoint_training_step") != admission.checkpoint_training_step:
        raise Multires500ProofError("one-run checkpoint_training_step mismatch")

    for key, expected in (
        ("atlas_sha256", admission.atlas_sha256),
        ("atlas_catalog_sha256", admission.atlas_catalog_sha256),
        ("runtime_manifest_sha256", admission.runtime_manifest_sha256),
        ("objective_identity_sha256", admission.objective_identity_sha256),
        ("training_manifest_sha256", admission.training_manifest_sha256),
        ("checkpoint_sha256", admission.checkpoint_sha256),
    ):
        actual = payload.get(key)
        if not _valid_sha256(actual):
            raise Multires500ProofError(f"one-run {key} is not a valid SHA-256")
        if actual != expected:
            raise Multires500ProofError(
                f"one-run {key} digests differ from admitted artifacts"
            )

    received = payload.get("received_inputs")
    if not isinstance(received, Mapping):
        raise Multires500ProofError(
            "one-run output must echo received_inputs to prove argument handshake"
        )
    expected = expected_received_inputs(request, admission, out_path)
    for key, wanted in expected.items():
        actual = received.get(key)
        if actual is None:
            raise Multires500ProofError(
                f"one-run ignored required input {key!r} (missing from received_inputs)"
            )
        if str(actual) != str(wanted):
            raise Multires500ProofError(
                f"one-run ignored or rewrote required input {key!r}: "
                f"expected {wanted!r}, got {actual!r}"
            )

    transition_count = int(payload.get("transition_count", -1))
    if transition_count != int(request.transition_count):
        raise Multires500ProofError(
            f"one-run transition_count={transition_count} != "
            f"request {request.transition_count}"
        )
    if transition_count != REQUIRED_TRANSITION_COUNT:
        raise Multires500ProofError(
            f"one-run transition_count={transition_count} is not the required "
            f"{REQUIRED_TRANSITION_COUNT}"
        )
    trajectory = payload.get("trajectory_sha256")
    if not _valid_sha256(trajectory):
        raise Multires500ProofError("one-run trajectory_sha256 is invalid")

    partial = int(payload.get("partial_admissions", 0))
    stale = int(payload.get("stale_admissions", 0))
    resync = int(payload.get("resync_admissions", 0))
    if partial or stale or resync:
        raise Multires500ProofError(
            f"one-run reported partial={partial} stale={stale} resync={resync}; "
            "proof requires zero"
        )

    if int(payload.get("seed", -1)) != int(request.seed):
        raise Multires500ProofError("one-run seed does not match launch request")
    if int(payload.get("game_seed", -1)) != int(request.game_seed):
        raise Multires500ProofError("one-run game_seed does not match launch request")
    if int(payload.get("policy_version", -1)) != int(request.policy_version):
        raise Multires500ProofError("one-run policy_version does not match launch request")
    if int(payload.get("map_epoch", -1)) != int(request.map_epoch):
        raise Multires500ProofError("one-run map_epoch does not match launch request")
    if str(payload.get("map_name", "")) != str(request.map_name):
        raise Multires500ProofError("one-run map_name does not match launch request")

    raw_process_records = payload.get("process_records")
    if not isinstance(raw_process_records, list) or len(raw_process_records) != len(
        REQUIRED_PROCESS_ROLES
    ):
        raise Multires500ProofError(
            "one-run process_records must contain q2ded plus exactly four clients"
        )
    process_records: list[ProcessIdentity] = []
    seen_pids: set[int] = set()
    for index, (raw, required_role) in enumerate(
        zip(raw_process_records, REQUIRED_PROCESS_ROLES)
    ):
        if not isinstance(raw, Mapping) or set(raw) != {
            "role", "pid", "start_ticks"
        }:
            raise Multires500ProofError(
                f"one-run process_records[{index}] is not canonical"
            )
        pid = raw["pid"]
        start_ticks = raw["start_ticks"]
        if (
            raw["role"] != required_role
            or type(pid) is not int
            or pid <= 1
            or pid in seen_pids
            or type(start_ticks) is not int
            or start_ticks <= 0
        ):
            raise Multires500ProofError(
                f"one-run process_records[{index}] role/PID/start_ticks is invalid"
            )
        seen_pids.add(pid)
        process_records.append(
            ProcessIdentity(
                role=required_role, pid=int(pid), start_ticks=int(start_ticks)
            )
        )
    process_ids = [record.pid for record in process_records]
    for field in (
        "process_ids", "launched_process_ids", "terminated_process_ids"
    ):
        if payload.get(field) != process_ids:
            raise Multires500ProofError(
                f"one-run {field} does not reconcile with process_records"
            )
    terminated_records = payload.get("terminated_process_records")
    if terminated_records != [record.to_mapping() for record in process_records]:
        raise Multires500ProofError(
            "one-run terminated_process_records do not reconcile with process_records"
        )

    # M6c: never admit production evidence from a digest-only assertion.
    if "records" not in payload:
        raise Multires500ProofError(
            "one-run records are missing; digest-only assertion is not admissible"
        )
    records = admit_canonical_records(
        payload.get("records"),
        expected_count=REQUIRED_TRANSITION_COUNT,
        expected_atlas_sha256=admission.atlas_sha256,
        expected_runtime_manifest_sha256=admission.runtime_manifest_sha256,
        expected_map_name=str(request.map_name),
        expected_map_epoch=int(request.map_epoch),
        expected_policy_version=int(request.policy_version),
        expected_trajectory_sha256=str(trajectory),
    )

    return finalize_launch_result(
        request=request,
        records=records,
        process_ids=tuple(process_ids),
        process_records=tuple(process_records),
        partial_admissions=0,
        stale_admissions=0,
        resync_admissions=0,
        collector=COLLECTOR_CLASS_NAME,
        spatial_provider=SPATIAL_PROVIDER_CLASS_NAME,
        lattice_crate=LATTICE_CRATE_NAME,
    )


class ProductionTransport:
    """Production launcher: three fresh subprocess one-run trainer executions."""

    def __init__(self, admission: ProductionAdmission):
        self.admission = admission
        self._groups: list[ManagedProcessGroup] = []
        self._last_cleanup = CleanupReport(cleaned=True, orphan_processes=0)
        self._reported_process_ids: list[int] = []
        self._reported_process_records: list[ProcessIdentity] = []
        self.launches: list[str] = []
        self.last_argv: list[str] = []

    def launch(self, request: LaunchRequest) -> LaunchResult:
        self.launches.append(request.launch_id)
        out_path = (
            self.admission.evidence_dir / f"one_run_{request.launch_id}.json"
        ).resolve()
        if out_path.exists():
            out_path.unlink()
        argv = build_one_run_argv(
            trainer_argv_prefix=self.admission.trainer_argv_prefix,
            request=request,
            admission=self.admission,
            out_path=out_path,
        )
        self.last_argv = list(argv)
        group = ManagedProcessGroup()
        self._groups.append(group)
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            group.processes.append(process)
            try:
                stdout, stderr = process.communicate(timeout=request.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                # Discover detached/descendant sessions under the owned trainer
                # via /proc before killing; do not use pattern-based kills.
                descendants = discover_descendant_pids(int(process.pid))
                group.extra_owned_pids.extend(descendants)
                timeout_report = group.terminate_all()
                still_alive = [
                    pid
                    for pid in (int(process.pid), *descendants)
                    if _pid_is_alive(pid)
                ]
                if still_alive:
                    raise Multires500ProofError(
                        f"launch {request.launch_id!r} timed out after "
                        f"{request.timeout_seconds}s and left owned PIDs alive "
                        f"after scoped teardown: {still_alive} "
                        f"(notes={list(timeout_report.notes)})"
                    ) from error
                raise Multires500ProofError(
                    f"launch {request.launch_id!r} timed out after "
                    f"{request.timeout_seconds}s "
                    f"(terminated scoped pids={list(timeout_report.scoped_child_pids)})"
                ) from error
            returncode = process.returncode
            if returncode != 0:
                detail = (stderr or b"").decode("utf-8", errors="replace")[-2000:]
                raise Multires500ProofError(
                    f"launch {request.launch_id!r} trainer exit={returncode}: {detail}"
                )
            if not out_path.is_file():
                # Allow stdout JSON if --out was not materialized but process succeeded.
                if stdout:
                    try:
                        payload = json.loads(stdout.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise Multires500ProofError(
                            f"launch {request.launch_id!r} missing --out artifact "
                            f"and stdout is not JSON: {error}"
                        ) from error
                    out_path.write_bytes(_canonical_bytes(payload))
                else:
                    raise Multires500ProofError(
                        f"launch {request.launch_id!r} did not write one-run output "
                        f"at {out_path}"
                    )
            payload = _load_json_object(out_path, f"one-run[{request.launch_id}]")
            result = parse_one_run_output(
                payload,
                request=request,
                admission=self.admission,
                out_path=out_path,
            )
            # Trainer child should have exited; re-check no orphans from this group.
            if process.poll() is None:
                raise Multires500ProofError(
                    f"launch {request.launch_id!r} left trainer process running"
                )
            # PID integers alone cannot distinguish teardown from PID reuse.
            # Any same-identity survivor is added to the owned cleanup set
            # before the proof is rejected.
            group.extra_owned_pids.extend(
                record.pid
                for record in result.process_records
                if process_identity_alive(record)
            )
            prove_process_records_dead(
                result.process_records,
                where=f"launch {request.launch_id!r} post-exit process_records",
            )
            if _pid_is_alive(int(process.pid)):
                raise Multires500ProofError(
                    f"launch {request.launch_id!r} trainer pid {process.pid} "
                    "still alive after exit"
                )
            self._reported_process_ids.extend(int(pid) for pid in result.process_ids)
            self._reported_process_records.extend(result.process_records)
            return result
        except Exception:
            self.cleanup()
            raise

    def cleanup(self) -> CleanupReport:
        orphan_total = 0
        killed: list[int] = []
        notes: list[str] = []
        scoped: list[int] = []
        verified_dead: list[int] = []
        still_alive: list[int] = []
        cleaned = True
        for group in self._groups:
            # Only the same attested process instance may enter the kill set;
            # never target a reused PID with different start_ticks.
            for record in self._reported_process_records:
                if (
                    process_identity_alive(record)
                    and record.pid not in group.extra_owned_pids
                ):
                    group.extra_owned_pids.append(record.pid)
            report = group.terminate_all()
            orphan_total += report.orphan_processes
            killed.extend(report.killed_process_ids)
            notes.extend(report.notes)
            scoped.extend(report.scoped_child_pids)
            verified_dead.extend(report.verified_dead_process_ids)
            still_alive.extend(report.still_alive_process_ids)
            cleaned = cleaned and report.cleaned
        self._groups.clear()

        # Final admission: every exact process identity reported across launches
        # is dead. A live reused PID with different start_ticks is not an orphan.
        reported = tuple(dict.fromkeys(int(pid) for pid in self._reported_process_ids))
        reported_identity_alive = [
            record
            for record in self._reported_process_records
            if process_identity_alive(record)
        ]
        if reported_identity_alive:
            cleaned = False
            orphan_total = max(orphan_total, len(reported_identity_alive))
            still_alive.extend(record.pid for record in reported_identity_alive)
            notes.append(
                "reported same-identity process_records still alive after teardown: "
                f"{[record.to_mapping() for record in reported_identity_alive]}"
            )
        else:
            verified_dead.extend(reported)

        def _unique(values: Sequence[int]) -> tuple[int, ...]:
            return tuple(dict.fromkeys(int(value) for value in values))

        self._last_cleanup = CleanupReport(
            cleaned=cleaned and orphan_total == 0 and not reported_identity_alive,
            orphan_processes=orphan_total,
            killed_process_ids=_unique(killed),
            notes=tuple(notes),
            scoped_child_pids=_unique(scoped),
            verified_dead_process_ids=_unique(verified_dead),
            still_alive_process_ids=_unique(
                pid for pid in still_alive if _pid_is_alive(pid)
            ),
        )
        return self._last_cleanup


@dataclass
class InjectedScenario:
    """Unit-test control surface for the injected transport (synthetic only)."""

    transition_count: int = REQUIRED_TRANSITION_COUNT
    force_partial_admissions: int = 0
    force_stale_admissions: int = 0
    force_resync_admissions: int = 0
    mix_atlas_digest: bool = False
    mix_map_epoch: bool = False
    mix_policy_version: bool = False
    fail_launch_id: Optional[str] = None
    fail_message: str = "injected launch failure"
    timeout_launch_id: Optional[str] = None
    leave_orphan_on_failure: bool = False
    custom_records_fn: Optional[
        Callable[[LaunchRequest], Sequence[TransitionRecord]]
    ] = None


class InjectedTransport:
    """Deterministic in-process transport for unit tests only (non-admissible)."""

    def __init__(
        self,
        *,
        atlas_sha256: str,
        runtime_manifest_sha256: str,
        scenario: Optional[InjectedScenario] = None,
    ):
        if not _valid_sha256(atlas_sha256):
            raise Multires500ProofError("injected atlas_sha256 is invalid")
        if not _valid_sha256(runtime_manifest_sha256):
            raise Multires500ProofError("injected runtime_manifest_sha256 is invalid")
        self.atlas_sha256 = atlas_sha256
        self.runtime_manifest_sha256 = runtime_manifest_sha256
        self.scenario = scenario or InjectedScenario()
        self._live_pids: list[int] = []
        self._next_fake_pid = 10_000
        self.cleanup_calls = 0
        self.launches: list[str] = []

    def _alloc_pid(self) -> int:
        pid = self._next_fake_pid
        self._next_fake_pid += 1
        self._live_pids.append(pid)
        return pid

    def _synthetic_records(self, request: LaunchRequest) -> list[TransitionRecord]:
        scenario = self.scenario
        count = int(scenario.transition_count)
        if scenario.custom_records_fn is not None:
            return list(scenario.custom_records_fn(request))
        records: list[TransitionRecord] = []
        # Language is not a launch variant: same seed/game_seed match byte-for-byte.
        material = hashlib.sha256(
            f"seed={request.seed}:game={request.game_seed}:map={request.map_name}".encode()
        ).digest()
        use_exact_layout = count == REQUIRED_TRANSITION_COUNT
        for index in range(count):
            step_material = hashlib.sha256(
                material + index.to_bytes(4, "little")
            ).digest()
            obs = [
                ((step_material[i % len(step_material)] / 255.0) * 2.0 - 1.0)
                for i in range(OBS_DIM)
            ]
            obs[0] = float(request.game_seed % 997) / 997.0
            obs[1] = float(request.seed % 991) / 991.0
            obs[2] = float(index) / max(count, 1)
            act = [0.0] * ACTION_DIM
            act[0] = float(step_material[0] % 11) / 10.0 - 0.5
            act[4] = float(step_material[1] % 3)
            act[5] = float(step_material[2] % 2)
            reward = float(step_material[3]) / 255.0
            atlas = request.atlas_sha256
            map_epoch = request.map_epoch
            policy_version = request.policy_version
            if scenario.mix_atlas_digest and index == max(count // 2, 0):
                atlas = hashlib.sha256(atlas.encode() + b"-mixed").hexdigest()
            if scenario.mix_map_epoch and index == max(count // 3, 0):
                map_epoch = request.map_epoch + 7
            if scenario.mix_policy_version and index == max(count // 4, 0):
                policy_version = request.policy_version + 3
            rust_features = [
                float(step_material[4 + (i % 12)]) / 255.0
                for i in range(_RUST_FEATURE_WIDTH)
            ]
            obs[DYN.slice] = rust_features
            if use_exact_layout:
                time_step, _client_index, client_id = layout_for_flat_index(
                    index, DEFAULT_SYNTHETIC_CLIENT_IDS
                )
                server_frame = DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + time_step
                batch_round_id = time_step
            else:
                # Non-500 negative fixtures keep dense single-stream indices.
                client_id = DEFAULT_SYNTHETIC_CLIENT_IDS[0]
                server_frame = DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + index
                batch_round_id = index
            records.append(
                make_transition_record(
                    index=index,
                    observation=obs,
                    action=act,
                    reward=reward,
                    client_id=client_id,
                    server_frame=server_frame,
                    batch_round_id=batch_round_id,
                    policy_version=policy_version,
                    map_name=request.map_name,
                    map_epoch=map_epoch,
                    atlas_sha256=atlas,
                    runtime_manifest_sha256=request.runtime_manifest_sha256,
                    rust_features=rust_features,
                )
            )
        return records

    def launch(self, request: LaunchRequest) -> LaunchResult:
        self.launches.append(request.launch_id)
        pid = self._alloc_pid()
        scenario = self.scenario
        if scenario.timeout_launch_id == request.launch_id:
            if not scenario.leave_orphan_on_failure:
                self._live_pids = [item for item in self._live_pids if item != pid]
            raise Multires500ProofError(
                f"launch {request.launch_id!r} timed out after {request.timeout_seconds}s"
            )
        if scenario.fail_launch_id == request.launch_id:
            if not scenario.leave_orphan_on_failure:
                self._live_pids = [item for item in self._live_pids if item != pid]
            raise Multires500ProofError(scenario.fail_message)
        records = self._synthetic_records(request)
        return finalize_launch_result(
            request=request,
            records=records,
            process_ids=(pid,),
            partial_admissions=scenario.force_partial_admissions,
            stale_admissions=scenario.force_stale_admissions,
            resync_admissions=scenario.force_resync_admissions,
        )

    def cleanup(self) -> CleanupReport:
        self.cleanup_calls += 1
        killed = tuple(self._live_pids)
        scoped = tuple(self._live_pids)
        self._live_pids.clear()
        return CleanupReport(
            cleaned=True,
            orphan_processes=0,
            killed_process_ids=killed,
            notes=("injected-transport-cleanup",),
            scoped_child_pids=scoped,
        )


def finalize_launch_result(
    *,
    request: LaunchRequest,
    records: Sequence[TransitionRecord],
    process_ids: Sequence[int] = (),
    process_records: Sequence[ProcessIdentity] = (),
    partial_admissions: int = 0,
    stale_admissions: int = 0,
    resync_admissions: int = 0,
    collector: str = COLLECTOR_CLASS_NAME,
    spatial_provider: str = SPATIAL_PROVIDER_CLASS_NAME,
    lattice_crate: str = LATTICE_CRATE_NAME,
) -> LaunchResult:
    records_tuple = tuple(records)
    if not records_tuple:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} produced zero admitted transitions"
        )
    atlas_values = {record.atlas_sha256 for record in records_tuple}
    epoch_values = {record.map_epoch for record in records_tuple}
    policy_values = {record.policy_version for record in records_tuple}
    if len(atlas_values) != 1:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} mixed atlas digests: {sorted(atlas_values)}"
        )
    if len(epoch_values) != 1:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} mixed map epochs: {sorted(epoch_values)}"
        )
    if len(policy_values) != 1:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} mixed policy versions: {sorted(policy_values)}"
        )
    atlas_sha256 = next(iter(atlas_values))
    map_epoch = next(iter(epoch_values))
    policy_version = next(iter(policy_values))
    if atlas_sha256 != request.atlas_sha256:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} atlas digest diverges from request"
        )
    if map_epoch != request.map_epoch:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} map epoch diverges from request"
        )
    if policy_version != request.policy_version:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} policy version diverges from request"
        )
    if partial_admissions:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} recorded partial batch admissions="
            f"{partial_admissions}; proof requires zero"
        )
    if stale_admissions:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} recorded stale admissions="
            f"{stale_admissions}; proof requires zero"
        )
    if resync_admissions:
        raise Multires500ProofError(
            f"launch {request.launch_id!r} recorded resync admissions="
            f"{resync_admissions}; proof requires zero"
        )
    # Re-admit through the canonical parser so synthetic and production share
    # the same fail-closed layout/identity/binding checks (never digest-only).
    records_tuple = admit_canonical_records(
        [record.to_mapping() for record in records_tuple],
        expected_count=len(records_tuple),
        expected_atlas_sha256=request.atlas_sha256,
        expected_runtime_manifest_sha256=request.runtime_manifest_sha256,
        expected_map_name=str(request.map_name),
        expected_map_epoch=int(request.map_epoch),
        expected_policy_version=int(request.policy_version),
        expected_trajectory_sha256=None,
    )
    digest = trajectory_sha256(records_tuple)
    return LaunchResult(
        launch_id=request.launch_id,
        seed=request.seed,
        game_seed=request.game_seed,
        language=STACK_LANGUAGE,
        transition_count=len(records_tuple),
        trajectory_sha256=digest,
        records=records_tuple,
        atlas_sha256=atlas_sha256,
        map_epoch=map_epoch,
        policy_version=policy_version,
        partial_admissions=0,
        stale_admissions=0,
        resync_admissions=0,
        process_ids=tuple(int(pid) for pid in process_ids),
        process_records=tuple(process_records),
        collector=collector,
        spatial_provider=spatial_provider,
        lattice_crate=lattice_crate,
    )


def build_verifier_evidence(
    *,
    mode: str,
    admissible: bool,
    production_pass: bool,
    transition_count: int,
    run_a: LaunchResult,
    run_b: LaunchResult,
    divergence_run: LaunchResult,
) -> dict[str, Any]:
    """Subset consumed by tools/verify_multires_integration.py.

    Both entries are independent fresh launches of the same complete stack.
    Python-collector and Rust-provider provenance is asserted on every run;
    it is never represented by relabeling one launch as a different language.
    """
    def record(run: LaunchResult) -> dict[str, Any]:
        return {
            "launch_id": run.launch_id,
            "stack": STACK_LANGUAGE,
            "fresh_subprocess": True,
            "transition_count": int(run.transition_count),
            "trajectory_sha256": run.trajectory_sha256,
            "game_seed": run.game_seed,
            "seed": run.seed,
            "collector": COLLECTOR_CLASS_NAME,
            "spatial_provider": SPATIAL_PROVIDER_CLASS_NAME,
            "lattice_crate": LATTICE_CRATE_NAME,
            "partial_admissions": int(run.partial_admissions),
            "stale_admissions": int(run.stale_admissions),
            "resync_admissions": int(run.resync_admissions),
        }

    return {
        "mode": mode,
        "admissible": admissible,
        "production_pass": production_pass,
        "transition_count": int(transition_count),
        "runs": [record(run_a), record(run_b)],
        "divergence_run": record(divergence_run),
    }


def _request_for(
    config: ProofConfig,
    *,
    launch_id: str,
    game_seed: int,
    atlas_sha256: str,
    runtime_manifest_sha256: str,
    map_name: str,
) -> LaunchRequest:
    return LaunchRequest(
        launch_id=launch_id,
        seed=int(config.seed),
        game_seed=int(game_seed),
        language=STACK_LANGUAGE,
        transition_count=int(config.transition_count),
        policy_version=int(config.policy_version),
        map_name=map_name,
        map_epoch=int(config.map_epoch),
        atlas_sha256=atlas_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        timeout_seconds=float(config.timeout_seconds),
    )


def run_proof(
    config: ProofConfig,
    *,
    transport: Optional[LaunchTransport] = None,
) -> dict[str, Any]:
    """Execute the fail-closed 500-transition proof and return canonical evidence."""
    if config.mode not in (MODE_PRODUCTION, MODE_SYNTHETIC):
        raise Multires500ProofError(
            f"mode must be {MODE_PRODUCTION!r} or {MODE_SYNTHETIC!r}"
        )
    if config.transition_count != REQUIRED_TRANSITION_COUNT:
        raise Multires500ProofError(
            f"transition_count={config.transition_count} is not the required "
            f"{REQUIRED_TRANSITION_COUNT}-transition proof"
        )
    if config.game_seed == config.divergence_game_seed:
        raise Multires500ProofError(
            "divergence_game_seed must differ from game_seed"
        )
    _reject_legacy_selector(config.legacy_selector)

    admission: Optional[ProductionAdmission] = None
    active_transport: LaunchTransport
    mode = config.mode

    if mode == MODE_PRODUCTION:
        if transport is not None:
            raise Multires500ProofError(
                "production mode rejects injected transports; the CLI drives "
                "the operational one-run trainer subprocess directly"
            )
        if config.production_collect_fn is not None or config.production_process_factory is not None:
            raise Multires500ProofError(
                "production mode rejects callback/process factory injection"
            )
        admission = admit_production_config(config)
        active_transport = ProductionTransport(admission)
        atlas_sha256 = admission.atlas_sha256
        runtime_manifest_sha256 = admission.runtime_manifest_sha256
        map_name = admission.map_name
    else:
        if transport is None:
            raise Multires500ProofError(
                "synthetic_injected mode requires an injected transport"
            )
        active_transport = transport
        atlas_sha256 = getattr(transport, "atlas_sha256", None)
        runtime_manifest_sha256 = getattr(transport, "runtime_manifest_sha256", None)
        if not _valid_sha256(atlas_sha256) or not _valid_sha256(runtime_manifest_sha256):
            raise Multires500ProofError(
                "injected transport must expose valid atlas and runtime digests"
            )
        map_name = config.map_name

    failures: list[str] = []
    same_seed_a: Optional[LaunchResult] = None
    same_seed_b: Optional[LaunchResult] = None
    divergence: Optional[LaunchResult] = None
    cleanup_report = CleanupReport(cleaned=False, orphan_processes=-1)

    try:
        # Both same-seed launches are identical complete stack invocations.
        # Language labels are not used as launch variants.
        same_seed_a = active_transport.launch(
            _request_for(
                config,
                launch_id="same_seed_run_a",
                game_seed=config.game_seed,
                atlas_sha256=atlas_sha256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                map_name=map_name,
            )
        )
        same_seed_b = active_transport.launch(
            _request_for(
                config,
                launch_id="same_seed_run_b",
                game_seed=config.game_seed,
                atlas_sha256=atlas_sha256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                map_name=map_name,
            )
        )
        divergence = active_transport.launch(
            _request_for(
                config,
                launch_id="divergence_game_seed",
                game_seed=config.divergence_game_seed,
                atlas_sha256=atlas_sha256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                map_name=map_name,
            )
        )
    except Exception as error:
        failures.append(str(error))
        raise Multires500ProofError(str(error)) from error
    finally:
        cleanup_report = active_transport.cleanup()

    assert same_seed_a is not None and same_seed_b is not None and divergence is not None

    for label, launch in (
        ("same_seed_run_a", same_seed_a),
        ("same_seed_run_b", same_seed_b),
        ("divergence_game_seed", divergence),
    ):
        if launch.transition_count != REQUIRED_TRANSITION_COUNT:
            failures.append(
                f"{label} transition_count={launch.transition_count} "
                f"!= {REQUIRED_TRANSITION_COUNT}"
            )
        if launch.partial_admissions or launch.stale_admissions or launch.resync_admissions:
            failures.append(
                f"{label} non-zero admission defects: "
                f"partial={launch.partial_admissions} "
                f"stale={launch.stale_admissions} "
                f"resync={launch.resync_admissions}"
            )
        if launch.language != STACK_LANGUAGE:
            failures.append(
                f"{label} language={launch.language!r} is not the full stack label "
                f"{STACK_LANGUAGE!r}"
            )

    same_seed_match = (
        same_seed_a.trajectory_sha256 == same_seed_b.trajectory_sha256
        and same_seed_a.transition_count == REQUIRED_TRANSITION_COUNT
        and same_seed_b.transition_count == REQUIRED_TRANSITION_COUNT
    )
    if not same_seed_match:
        failures.append(
            "same-seed trajectories diverge or are incomplete: "
            f"{same_seed_a.trajectory_sha256} vs {same_seed_b.trajectory_sha256}"
        )

    different_seed_diverges = (
        divergence.trajectory_sha256 != same_seed_a.trajectory_sha256
        and divergence.transition_count == REQUIRED_TRANSITION_COUNT
    )
    if not different_seed_diverges:
        failures.append(
            "different game seed did not diverge from the same-seed trajectory"
        )

    orphan_after = int(cleanup_report.orphan_processes_after_teardown)
    if orphan_after != 0 or not cleanup_report.cleaned:
        failures.append(
            f"teardown left orphan_processes_after_teardown={orphan_after}"
        )
    if cleanup_report.still_alive_process_ids:
        failures.append(
            "teardown still_alive_process_ids="
            f"{list(cleanup_report.still_alive_process_ids)}"
        )

    # M6f: after production launches, every exact reported process identity
    # must be dead. PID-only liveness would misclassify reuse.
    if mode == MODE_PRODUCTION:
        for label, launch in (
            ("same_seed_run_a", same_seed_a),
            ("same_seed_run_b", same_seed_b),
            ("divergence_game_seed", divergence),
        ):
            if not launch.process_records:
                failures.append(
                    f"{label} process_records empty; PID integers are insufficient"
                )
                continue
            alive = [
                record.to_mapping()
                for record in launch.process_records
                if process_identity_alive(record)
            ]
            if alive:
                failures.append(
                    f"{label} same-identity processes still alive after teardown: {alive}"
                )

    # Production requires every launch to carry fully admitted records (not digests).
    derived_client_ids: tuple[str, ...] = ()
    derived_base_server_frame: Optional[int] = None
    for label, launch in (
        ("same_seed_run_a", same_seed_a),
        ("same_seed_run_b", same_seed_b),
        ("divergence_game_seed", divergence),
    ):
        if len(launch.records) != REQUIRED_TRANSITION_COUNT:
            failures.append(
                f"{label} records length {len(launch.records)} != "
                f"{REQUIRED_TRANSITION_COUNT} (digest-only is not admissible)"
            )
        elif launch.trajectory_sha256 != trajectory_sha256(launch.records):
            failures.append(
                f"{label} trajectory_sha256 does not match recomputed records"
            )
        else:
            try:
                client_ids, base_frame = enforce_exact_500_layout(launch.records)
            except Multires500ProofError as error:
                failures.append(f"{label} layout admission failed: {error}")
            else:
                if not derived_client_ids:
                    derived_client_ids = client_ids
                    derived_base_server_frame = base_frame
                elif client_ids != derived_client_ids:
                    failures.append(
                        f"{label} client_id order {client_ids!r} differs from "
                        f"first launch {derived_client_ids!r}"
                    )
                elif (
                    derived_base_server_frame is not None
                    and base_frame != derived_base_server_frame
                ):
                    failures.append(
                        f"{label} base_server_frame {base_frame} differs from "
                        f"first launch {derived_base_server_frame}"
                    )

    non_admissible_reason = ""
    if mode == MODE_SYNTHETIC:
        admissible = False
        production_pass = False
        non_admissible_reason = (
            "synthetic_injected mode is labeled non-admissible and cannot "
            "produce a production pass"
        )
    elif not failures:
        admissible = True
        production_pass = True
    else:
        admissible = False
        production_pass = False

    verifier_evidence = build_verifier_evidence(
        mode=mode,
        admissible=admissible,
        production_pass=production_pass,
        transition_count=REQUIRED_TRANSITION_COUNT,
        run_a=same_seed_a,
        run_b=same_seed_b,
        divergence_run=divergence,
    )

    report: dict[str, Any] = {
        "schema": PROOF_SCHEMA,
        "tool": TOOL_NAME,
        "mode": mode,
        "admissible": admissible,
        "production_pass": production_pass,
        "non_admissible_reason": non_admissible_reason,
        "transition_count": REQUIRED_TRANSITION_COUNT,
        "required_client_count": REQUIRED_CLIENT_COUNT,
        "transitions_per_client": TRANSITIONS_PER_CLIENT,
        "record_ordering": RECORD_ORDERING,
        "proof_client_ids": list(derived_client_ids),
        "base_server_frame": derived_base_server_frame,
        "records_required": True,
        "digest_only_admissible": False,
        "seed": int(config.seed),
        "game_seed": int(config.game_seed),
        "divergence_game_seed": int(config.divergence_game_seed),
        "same_seed_match": same_seed_match,
        "different_seed_diverges": different_seed_diverges,
        "partial_admissions": 0,
        "stale_admissions": 0,
        "resync_admissions": 0,
        "orphan_processes_after_teardown": orphan_after,
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "b4_protocol_generation": B4_PROTOCOL_GENERATION,
        "qm3c_causal_magic": B4_CAUSAL_MAGIC,
        "qm3c_causal_magic_hex": f"{B4_CAUSAL_MAGIC:#x}",
        "b4_client_wire_version": B4_CLIENT_WIRE_VERSION,
        "b4_observation_magic": B4_OBSERVATION_MAGIC,
        "b4_action_magic": B4_ACTION_MAGIC,
        "b4_teacher_version": B4_TEACHER_VERSION,
        "b4_rollout_schema": B4_ROLLOUT_SCHEMA,
        "b4_causal_version": B4_CAUSAL_VERSION,
        "b4_causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "collector": COLLECTOR_CLASS_NAME,
        "spatial_provider": SPATIAL_PROVIDER_CLASS_NAME,
        "lattice_crate": LATTICE_CRATE_NAME,
        "python_collector_schema": PYTHON_COLLECTOR_SCHEMA,
        "rust_provider_schema": RUST_PROVIDER_SCHEMA,
        "one_run_schema": ONE_RUN_SCHEMA,
        "one_run_protocol_version": ONE_RUN_PROTOCOL_VERSION,
        "documented_m4_one_run_module": DOCUMENTED_M4_ONE_RUN_MODULE,
        "atlas_sha256": atlas_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "map_name": map_name,
        "map_epoch": int(config.map_epoch),
        "policy_version": int(config.policy_version),
        "runs": [
            same_seed_a.to_mapping(),
            same_seed_b.to_mapping(),
        ],
        "divergence_run": divergence.to_mapping(),
        "teardown": cleanup_report.to_mapping(),
        "verifier_gate": VERIFIER_GATE,
        "verifier_evidence": verifier_evidence,
        "failures": failures,
        "legacy_modules_forbidden": list(FORBIDDEN_LEGACY_SELECTORS),
    }
    if admission is not None:
        report["atlas_catalog_sha256"] = admission.atlas_catalog_sha256
        report["production_admission"] = {
            "bundle_version": admission.bundle_version,
            "artifact_state": admission.artifact_state,
            "objective_identity_sha256": admission.objective_identity_sha256,
            "training_manifest_sha256": admission.training_manifest_sha256,
            "checkpoint_sha256": admission.checkpoint_sha256,
            "checkpoint_initialization": admission.checkpoint_initialization,
            "checkpoint_training_step": admission.checkpoint_training_step,
            "checkpoint_lineage_root_sha256": (
                admission.checkpoint_lineage_root_sha256
            ),
            "q2ded": str(admission.q2ded),
            "client_binary": str(admission.client_binary),
            "runtime_root": str(admission.runtime_root),
            "atlas_catalog": str(admission.atlas_catalog),
            "trainer_argv_prefix": list(admission.trainer_argv_prefix),
            "evidence_dir": str(admission.evidence_dir),
        }

    if mode == MODE_SYNTHETIC:
        report["admissible"] = False
        report["production_pass"] = False
        report["non_admissible_reason"] = non_admissible_reason or (
            "synthetic_injected mode is labeled non-admissible and cannot "
            "produce a production pass"
        )

    if mode == MODE_PRODUCTION and failures:
        raise Multires500ProofError(
            "production 500-transition proof failed: " + "; ".join(failures)
        )
    if mode == MODE_SYNTHETIC and failures:
        report["deterministic_ok"] = False
    else:
        report["deterministic_ok"] = (
            same_seed_match
            and different_seed_diverges
            and not failures
        )
    if config.out_path is not None:
        _write_evidence(config.out_path, report)
    return report


def _write_evidence(path: Path, report: Mapping[str, Any]) -> None:
    target = path.expanduser()
    if not target.is_absolute():
        raise Multires500ProofError(f"--out must be an absolute path: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    if target.name == f"{VERIFIER_GATE}.json" or target.name == "deterministic_transitions.json":
        if (
            report.get("mode") != MODE_PRODUCTION
            or report.get("admissible") is not True
            or report.get("production_pass") is not True
        ):
            raise Multires500ProofError(
                "deterministic verifier evidence requires a passing production proof"
            )
        body = _canonical_bytes(report["verifier_evidence"])
    else:
        body = _canonical_bytes(payload)
        if (
            report.get("mode") == MODE_PRODUCTION
            and report.get("admissible") is True
            and report.get("production_pass") is True
        ):
            sibling = target.with_name(f"{VERIFIER_GATE}.json")
            sibling.write_bytes(_canonical_bytes(report["verifier_evidence"]))
    target.write_bytes(body)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed multires 500-transition same-seed determinism proof "
            "(M6d). Production mode launches three fresh operational one-run "
            "trainer subprocesses with absolute argv (no shell, no callbacks) "
            "and requires exactly 500 canonical records (never digest-only) "
            "with derived client IDs and unit server_frame advance."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_PRODUCTION, MODE_SYNTHETIC),
        required=True,
        help="production or synthetic_injected (tests only)",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--game_seed", type=int, required=True)
    parser.add_argument("--divergence_game_seed", type=int, required=True)
    parser.add_argument(
        "--transition_count",
        type=int,
        default=REQUIRED_TRANSITION_COUNT,
        help=f"must be {REQUIRED_TRANSITION_COUNT}",
    )
    parser.add_argument("--policy_version", type=int, default=1)
    parser.add_argument("--map_name", default="mlstage_0001")
    parser.add_argument("--map_epoch", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--q2ded", type=Path, default=None)
    parser.add_argument("--client_binary", type=Path, default=None)
    parser.add_argument("--runtime_root", type=Path, default=None)
    parser.add_argument("--bundle_manifest", type=Path, default=None)
    parser.add_argument("--objectives", type=Path, default=None)
    parser.add_argument("--atlas_bin", type=Path, default=None)
    parser.add_argument("--atlas_catalog", type=Path, default=None)
    parser.add_argument("--expected_atlas_catalog_sha256", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training_manifest", type=Path, default=None)
    parser.add_argument("--runtime_evidence", type=Path, default=None)
    parser.add_argument("--evidence_dir", type=Path, default=None)
    parser.add_argument(
        "--trainer_executable",
        type=Path,
        default=None,
        help="absolute path to the operational one-run trainer executable",
    )
    parser.add_argument(
        "--trainer_arg",
        action="append",
        default=[],
        help="additional argv token after --trainer_executable (repeatable, no shell)",
    )
    parser.add_argument(
        "--trainer_command",
        nargs="+",
        default=None,
        help=(
            "absolute executable plus optional argv tokens as a single command "
            "list (mutually exclusive with --trainer_executable/--trainer_arg)"
        ),
    )
    parser.add_argument(
        "--legacy_selector",
        default=None,
        help="optional selector string; legacy names are rejected",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ProofConfig:
    trainer_executable = args.trainer_executable
    trainer_args: tuple[str, ...] = tuple(args.trainer_arg or ())
    if args.trainer_command is not None:
        resolved = resolve_trainer_argv(trainer_command=args.trainer_command)
        trainer_executable = Path(resolved[0])
        trainer_args = tuple(resolved[1:])
    return ProofConfig(
        mode=str(args.mode),
        seed=int(args.seed),
        game_seed=int(args.game_seed),
        divergence_game_seed=int(args.divergence_game_seed),
        transition_count=int(args.transition_count),
        policy_version=int(args.policy_version),
        map_name=str(args.map_name),
        map_epoch=int(args.map_epoch),
        timeout_seconds=float(args.timeout_seconds),
        out_path=args.out,
        q2ded=args.q2ded,
        client_binary=args.client_binary,
        runtime_root=args.runtime_root,
        bundle_manifest=args.bundle_manifest,
        objectives=args.objectives,
        atlas_bin=args.atlas_bin,
        atlas_catalog=args.atlas_catalog,
        expected_atlas_catalog_sha256=args.expected_atlas_catalog_sha256,
        checkpoint=args.checkpoint,
        training_manifest=args.training_manifest,
        runtime_evidence=args.runtime_evidence,
        evidence_dir=args.evidence_dir,
        legacy_selector=args.legacy_selector,
        trainer_executable=trainer_executable,
        trainer_args=trainer_args,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    if config.mode == MODE_SYNTHETIC:
        print(
            "synthetic_injected mode is non-admissible and cannot produce a "
            "production pass; provide an injected transport via the Python API",
            file=sys.stderr,
        )
        return 2
    try:
        report = run_proof(config)
    except Multires500ProofError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(_canonical_json({
        "schema": report["schema"],
        "mode": report["mode"],
        "admissible": report["admissible"],
        "production_pass": report["production_pass"],
        "same_seed_match": report["same_seed_match"],
        "different_seed_diverges": report["different_seed_diverges"],
        "trajectory_sha256": report["runs"][0]["trajectory_sha256"],
        "divergence_trajectory_sha256": report["divergence_run"]["trajectory_sha256"],
        "collector": report["collector"],
        "spatial_provider": report["spatial_provider"],
        "out": str(config.out_path) if config.out_path else None,
        "documented_m4_one_run_module": report["documented_m4_one_run_module"],
    }))
    return 0 if report["production_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
