"""One exact, self-contained multires network rollout qualification run.

This is the operational side of ``tools.run_multires_500_transition_proof``.
It has no synthetic transport and no legacy trainer/model selector.  A launch
is admitted only after the sealed semantic runtime manifest is reconciled
against the actual q2ded, game, client, Rust extension, map, Atlas, training,
optimizer, and retirement artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.multires_contract import DYN, POLICY_GENERATION  # noqa: E402
from harness.multires_lineage import (  # noqa: E402
    CHECKPOINT_FORMAT,
    LineageError,
    load_attested_checkpoint,
)
from harness.multires_runtime import (  # noqa: E402
    B4_CAUSAL_MAGIC,
    B4_PROTOCOL_GENERATION,
    validate_runtime_evidence,
)
from harness.multires_training_config import (  # noqa: E402
    MultiresTrainingConfiguration,
    TRAINING_CONFIG_SCHEMA,
    canonical_json as canonical_training_json,
)
from harness.runtime_attestation import (  # noqa: E402
    load_runtime_manifest,
    verify_runtime_manifest,
)
from tools.run_multires_500_transition_proof import (  # noqa: E402
    COLLECTOR_CLASS_NAME,
    LATTICE_CRATE_NAME,
    ONE_RUN_PROTOCOL_VERSION,
    ONE_RUN_SCHEMA,
    PYTHON_COLLECTOR_SCHEMA,
    RUST_PROVIDER_SCHEMA,
    SPATIAL_PROVIDER_CLASS_NAME,
    make_transition_record,
    trajectory_sha256,
)
from tools.qualify_network_client_frame_barrier import (  # noqa: E402
    EXECUTION_SCHEMA as NETWORK_BARRIER_EXECUTION_SCHEMA,
    QualificationError as NetworkBarrierQualificationError,
    _validate_execution_evidence,
)


RUNTIME_CONFIG_SCHEMA = "q2-multires-one-run-runtime-v1"
RETIREMENT_SCHEMA = "q2-multires-runtime-retirement-v1"
RUNTIME_CONFIG_NAME = "multires-one-run.json"
RUNTIME_MANIFEST_NAME = "runtime-manifest.json"
NETWORK_BARRIER_SCHEMA = "q2-network-client-frame-barrier-qualification-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,63}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{32,63}\Z")


class OneRunError(RuntimeError):
    """Raised before proof publication when operational admission fails."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value)) and value != "0" * 64


def _file(path: Path, label: str, *, executable: bool = False) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise OneRunError(f"{label} must be an absolute, regular, non-symlink file")
    resolved = source.resolve()
    if executable and not os.access(resolved, os.X_OK):
        raise OneRunError(f"{label} is not executable")
    return resolved


def _directory(path: Path, label: str) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise OneRunError(f"{label} must be an absolute, non-symlink directory")
    return source.resolve()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OneRunError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise OneRunError(f"{label} must be a JSON object")
    return value


def _record_matches(path: Path, record: Mapping[str, Any], label: str) -> None:
    if (
        record.get("sha256") != _sha256(path)
        or record.get("size") != path.stat().st_size
    ):
        raise OneRunError(f"sealed runtime {label} differs from actual bytes")


def _git_identity(path: Path, label: str) -> dict[str, Any]:
    repo = Path(path).resolve()
    if not repo.is_dir() or Path(path).is_symlink():
        raise OneRunError(f"{label} source repository is invalid")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo,
            text=True,
            stderr=subprocess.STDOUT,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise OneRunError(f"{label} source repository cannot be attested") from error
    if status:
        raise OneRunError(f"{label} source repository is dirty")
    return {"commit": commit, "tree": tree, "clean": True}


def _current_source_repositories() -> dict[str, dict[str, Any]]:
    return {
        "bot": _git_identity(ROOT, "bot"),
        "client": _git_identity(ROOT.parent / "q2-ml-client", "client"),
        "game": _git_identity(ROOT.parent / "q2-lithium-3zb2", "game"),
    }


def _validate_network_barrier(
    *,
    barrier_path: Path,
    barrier: Mapping[str, Any],
    runtime_manifest_sha256: str,
    runtime_semantics: Mapping[str, Any],
    q2ded: Path,
    game_module: Path,
    client_binary: Path,
) -> str:
    """Recompute the full execution/raw/runtime/source qualification closure."""
    envelope = dict(barrier)
    outer_digest = envelope.pop("evidence_sha256", None)
    if (
        envelope.get("schema") != NETWORK_BARRIER_SCHEMA
        or envelope.get("passed") is not True
        or envelope.get("test_mode") is not False
        or envelope.get("mode") != "client-telemetry-frame-ack-v1"
        or envelope.get("protocol_version") != 1
        or envelope.get("non_admissible_for_training") is not True
        or not _valid_sha256(outer_digest)
        or outer_digest
        != hashlib.sha256(_canonical_bytes(envelope)).hexdigest()
    ):
        raise OneRunError("network frame-barrier qualification envelope differs")
    execution_value = envelope.get("execution_evidence")
    if not isinstance(execution_value, Mapping):
        raise OneRunError("network frame-barrier execution evidence is absent")
    try:
        execution = _validate_execution_evidence(execution_value, barrier_path)
    except (
        NetworkBarrierQualificationError, KeyError, TypeError, ValueError,
        IndexError, OSError,
    ) as error:
        raise OneRunError(
            f"network frame-barrier execution evidence differs: {error}"
        ) from error
    execution_digest = execution["execution_evidence_sha256"]
    if (
        execution.get("schema") != NETWORK_BARRIER_EXECUTION_SCHEMA
        or execution.get("test_mode") is not False
        or execution.get("full_network_executed") is not True
        or envelope.get("execution_evidence_sha256") != execution_digest
        or envelope.get("runtime_manifest_sha256") != runtime_manifest_sha256
    ):
        raise OneRunError("network frame-barrier execution/runtime binding differs")
    closure = hashlib.sha256(_canonical_bytes({
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "execution_evidence_sha256": execution_digest,
    })).hexdigest()
    if envelope.get("runtime_closure_sha256") != closure:
        raise OneRunError("network frame-barrier runtime closure differs")
    if runtime_semantics.get(
        "network_barrier_execution_evidence_sha256"
    ) != execution_digest:
        raise OneRunError("sealed runtime frame-barrier execution binding differs")

    binaries = execution["runtime_binaries"]
    _record_matches(q2ded, binaries["q2ded"], "qualified q2ded")
    _record_matches(game_module, binaries["game_module"], "qualified game module")
    _record_matches(
        client_binary, binaries["client_binary"], "qualified client binary"
    )
    design = _file(
        ROOT / "docs" / "NETWORK-CLIENT-FRAME-BARRIER.md",
        "network frame-barrier design",
    )
    if execution.get("design_sha256") != _sha256(design):
        raise OneRunError("network frame-barrier design identity differs")
    current_sources = _current_source_repositories()
    recorded_sources = execution.get("source_repositories")
    if current_sources != recorded_sources:
        raise OneRunError("network frame-barrier source commit/tree closure differs")
    source_digest = hashlib.sha256(_canonical_bytes({
        name: current_sources[name] for name in sorted(current_sources)
    })).hexdigest()
    if execution.get("source_closure_sha256") != source_digest:
        raise OneRunError("network frame-barrier source closure digest differs")
    for key in (
        "fault_injection_passed",
        "ack_timeout_rejection_passed",
        "unsupported_mode_rejected",
        "deterministic_client_id_slot_admission",
        "all_clients_registered_before_bootstrap",
        "fresh_usercmd_per_client_frame",
        "modulo_generation_enforced",
        "reliable_hook_weapon_deferred_ordered",
    ):
        if execution.get(key) is not True:
            raise OneRunError(f"network frame-barrier conclusion {key} differs")
    if (
        execution.get("action_free_bootstrap_frames") != 1
        or execution.get("usercmd_application_order") != "client-id-then-slot"
        or execution.get("automatic_promotion") is not False
    ):
        raise OneRunError("network frame-barrier application conclusions differ")
    return execution_digest


def _training_configuration(
    path: Path,
) -> tuple[MultiresTrainingConfiguration, str, dict[str, Any]]:
    document = _json(path, "training_manifest")
    body = {key: document[key] for key in ("schema", "reward", "guide_dropout", "ppo") if key in document}
    if set(body) != {"schema", "reward", "guide_dropout", "ppo"}:
        raise OneRunError("training_manifest envelope differs")
    if body["schema"] != TRAINING_CONFIG_SCHEMA:
        raise OneRunError("training_manifest schema differs")
    configuration = MultiresTrainingConfiguration.create(
        reward=body["reward"],
        guide_dropout=body["guide_dropout"],
        ppo=body["ppo"],
    )
    canonical_digest = configuration.sha256
    declared = document.get("sha256", canonical_digest)
    if declared != canonical_digest:
        raise OneRunError("training_manifest declared digest is not its canonical body")
    if set(document) not in (
        {"schema", "reward", "guide_dropout", "ppo"},
        {"schema", "reward", "guide_dropout", "ppo", "sha256"},
    ):
        raise OneRunError("training_manifest has unrecognized fields")
    return configuration, canonical_digest, body


def _validate_step_zero_checkpoint(
    checkpoint: Path,
    *,
    atlas_sha256: str,
    runtime_manifest_sha256: str,
    training_configuration: MultiresTrainingConfiguration,
    optimizer_configuration: Mapping[str, Any],
) -> None:
    """Weights-only load the exact fresh checkpoint before proof admission."""
    try:
        import torch
        from models.multires_policy import MultiresQ2BotPolicy
    except ImportError as error:
        raise OneRunError(
            "PyTorch and the multires policy are required for checkpoint admission"
        ) from error
    policy = MultiresQ2BotPolicy().to(torch.device("cpu"))
    try:
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(optimizer_configuration["learning_rate"]),
            **dict(optimizer_configuration["kwargs"]),
        )
        manifest = load_attested_checkpoint(
            checkpoint,
            policy,
            expected_atlas_sha256=atlas_sha256,
            expected_runtime_manifest_sha256=runtime_manifest_sha256,
            expected_training_config=training_configuration,
            optimizer=optimizer,
            map_location="cpu",
        )
    except (KeyError, TypeError, ValueError, RuntimeError, LineageError) as error:
        raise OneRunError(
            f"step-zero checkpoint admission failed: {error}"
        ) from error
    if (
        manifest.initialization != "random"
        or manifest.training_step != 0
        or not _valid_sha256(manifest.lineage_root_sha256)
        or optimizer.state_dict().get("state") != {}
    ):
        raise OneRunError(
            "qualification requires a real random step-zero checkpoint with "
            "fresh-empty optimizer state"
        )


@dataclass(frozen=True)
class OneRunAdmission:
    args: argparse.Namespace
    runtime_root: Path
    runtime_config: Mapping[str, Any]
    runtime_manifest_sha256: str
    runtime_evidence: Mapping[str, Any]
    training_configuration: MultiresTrainingConfiguration
    training_manifest_sha256: str
    objective_identity_sha256: str
    checkpoint_sha256: str
    q2_root: Path
    q2ded: Path
    client_binary: Path
    bundle_manifest: Path
    objectives: Path
    atlas_bin: Path
    dyn_snapshots: tuple[Path, ...]
    rust_extension: Path
    retirement_manifest_sha256: str
    network_barrier_execution_evidence_sha256: str


def preflight(args: argparse.Namespace) -> OneRunAdmission:
    for name in ("launch_id", "map_name"):
        if not _SAFE_NAME.fullmatch(str(getattr(args, name))):
            raise OneRunError(f"{name} contains unsafe characters")
    if args.seed < 0 or args.game_seed < 0 or args.policy_version < 0 or args.map_epoch < 0:
        raise OneRunError("seed, game seed, policy version, and map epoch must be nonnegative")
    if args.transition_count < 1:
        raise OneRunError("transition_count must be positive")
    if not _valid_sha256(args.expected_atlas_sha256) or not _valid_sha256(
        args.expected_runtime_manifest_sha256
    ):
        raise OneRunError("expected Atlas/runtime digests must be non-placeholder SHA-256")

    runtime_root = _directory(args.runtime_root, "runtime_root")
    runtime_config_path = _file(
        runtime_root / RUNTIME_CONFIG_NAME, "runtime configuration"
    )
    config = _json(runtime_config_path, "runtime configuration")
    required_config = {
        "schema", "q2_root", "rust_extension", "dyn_snapshots",
        "retirement_manifest", "network_barrier_qualification", "client_count",
        "server_host", "server_port",
        "telemetry_host", "telemetry_port", "telemetry_token_env",
        "harness_host", "harness_port_base", "qport_base", "client_data_root",
        "client_id_prefix", "name_prefix", "game", "client_timeout",
        "round_timeout", "max_rejected_echoes", "movement_tolerance",
        "look_tolerance", "maximum_boundary_rounds", "device",
        "optimizer", "debug_clients", "server_warmup_seconds",
    }
    if set(config) != required_config or config.get("schema") != RUNTIME_CONFIG_SCHEMA:
        raise OneRunError("runtime configuration fields/schema differ")
    client_count = config["client_count"]
    if type(client_count) is not int or not 1 <= client_count <= 64:
        raise OneRunError("runtime client_count is invalid")
    if args.transition_count % client_count:
        raise OneRunError(
            "transition_count must divide exactly across all configured clients"
        )
    for field in ("server_host", "telemetry_host", "harness_host"):
        if config[field] != "127.0.0.1":
            raise OneRunError(f"runtime {field} must be exact IPv4 loopback")
    ports: dict[str, int] = {}
    for field in ("server_port", "telemetry_port", "harness_port_base", "qport_base"):
        value = config[field]
        if type(value) is not int or not 1 <= value <= 65535:
            raise OneRunError(f"runtime {field} is invalid")
        ports[field] = value
    harness_ports = set(
        range(ports["harness_port_base"], ports["harness_port_base"] + client_count)
    )
    qports = set(range(ports["qport_base"], ports["qport_base"] + client_count))
    if (
        max(harness_ports) > 65535
        or max(qports) > 65535
        or ports["server_port"] == ports["telemetry_port"]
        or ports["server_port"] in harness_ports | qports
        or ports["telemetry_port"] in harness_ports | qports
        or harness_ports & qports
    ):
        raise OneRunError("runtime server/telemetry/client port ranges overlap")
    for field, maximum in (
        ("client_id_prefix", 60), ("name_prefix", 12), ("game", 63)
    ):
        value = config[field]
        if not isinstance(value, str) or len(value) > maximum or not _SAFE_NAME.fullmatch(value):
            raise OneRunError(f"runtime {field} contains unsafe characters")
    for field in (
        "client_timeout", "round_timeout", "movement_tolerance",
        "look_tolerance", "server_warmup_seconds",
    ):
        value = config[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise OneRunError(f"runtime {field} must be finite numeric")
    if (
        float(config["client_timeout"]) <= 0.0
        or float(config["round_timeout"]) <= 0.0
        or float(config["client_timeout"]) < float(config["round_timeout"])
        or float(config["movement_tolerance"]) < 0.0
        or float(config["look_tolerance"]) < 0.0
        or not 0.0 <= float(config["server_warmup_seconds"]) <= 30.0
    ):
        raise OneRunError("runtime timeout/tolerance/warmup values are invalid")
    if (
        type(config["max_rejected_echoes"]) is not int
        or config["max_rejected_echoes"] < 0
        or config["maximum_boundary_rounds"] != 0
        or type(config["debug_clients"]) is not bool
    ):
        raise OneRunError("runtime echo/boundary/debug settings are invalid")
    device_name = config["device"]
    if not isinstance(device_name, str) or not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", device_name):
        raise OneRunError("runtime device is invalid")
    if device_name.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8", ":16:8",
    }:
        raise OneRunError(
            "CUDA qualification requires deterministic CUBLAS_WORKSPACE_CONFIG"
        )

    q2_root = _directory(Path(config["q2_root"]), "q2_root")
    q2ded = _file(args.q2ded, "q2ded", executable=True)
    expected_q2ded = _file(q2_root / "q2ded", "q2_root/q2ded", executable=True)
    if q2ded != expected_q2ded:
        raise OneRunError("q2ded argument differs from runtime q2_root")
    client_binary = _file(args.client_binary, "client_binary", executable=True)
    rust_extension = _file(Path(config["rust_extension"]), "Rust extension")
    client_data_root = _directory(
        Path(config["client_data_root"]), "client_data_root"
    )
    dyn_values = config["dyn_snapshots"]
    if not isinstance(dyn_values, list) or len(dyn_values) != client_count:
        raise OneRunError("runtime Dyn snapshot cardinality differs from clients")
    dyn_snapshots = tuple(
        _file(Path(value), f"Dyn snapshot[{index}]")
        for index, value in enumerate(dyn_values)
    )

    bundle_manifest = _file(args.bundle_manifest, "bundle_manifest")
    objectives = _file(args.objectives, "objectives")
    atlas_bin = _file(args.atlas_bin, "atlas_bin")
    checkpoint = _file(args.checkpoint, "checkpoint")
    training_manifest = _file(args.training_manifest, "training_manifest")
    runtime_evidence_path = _file(args.runtime_evidence, "runtime_evidence")
    out = Path(args.out)
    if not out.is_absolute() or out.is_symlink():
        raise OneRunError("out must be an absolute non-symlink path")
    _directory(out.parent, "out parent")

    bundle = _json(bundle_manifest, "bundle_manifest")
    if (
        bundle.get("bundle_version") != 3
        or bundle.get("name") != args.map_name
        or bundle.get("artifact_state") not in (None, "admitted")
    ):
        raise OneRunError("only the requested exact bundle-v3 map is allowed")
    directory = bundle_manifest.parent
    expected_objectives = directory / f"{args.map_name}.objectives.json"
    if objectives != expected_objectives.resolve():
        raise OneRunError("objectives argument is not the selected bundle member")
    if bundle.get("analysis_files", {}).get(objectives.name) != _sha256(objectives):
        raise OneRunError("objectives bytes differ from admitted bundle")
    atlas_manifest_path = _file(
        directory / f"{args.map_name}.atlas.manifest.json", "Atlas manifest"
    )
    atlas_manifest = _json(atlas_manifest_path, "Atlas manifest")
    try:
        atlas_record = atlas_manifest["artifacts"][f"{args.map_name}.atlas.bin"]
    except (KeyError, TypeError) as error:
        raise OneRunError("Atlas manifest lacks raw Atlas identity") from error
    if (
        _sha256(atlas_bin) != args.expected_atlas_sha256
        or atlas_record.get("sha256_uncompressed") != args.expected_atlas_sha256
        or atlas_record.get("uncompressed_size") != atlas_bin.stat().st_size
    ):
        raise OneRunError("raw Atlas differs from the admitted Atlas manifest")

    objectives_document = _json(objectives, "objectives")
    objective_identity = objectives_document.get(
        "objective_identity_sha256", _sha256(objectives)
    )
    if not _valid_sha256(objective_identity):
        raise OneRunError("objective identity is invalid")
    training_configuration, training_sha256, _training_body = (
        _training_configuration(training_manifest)
    )

    retirement_path = _file(
        Path(config["retirement_manifest"]), "runtime retirement manifest"
    )
    retirement = _json(retirement_path, "runtime retirement manifest")
    retirement_sha256 = _sha256(retirement_path)
    if (
        retirement.get("schema") != RETIREMENT_SCHEMA
        or retirement.get("status") != "legacy-runtime-retired"
        or retirement.get("fallback_allowed") is not False
    ):
        raise OneRunError("runtime retirement manifest does not disable fallback")

    barrier_path = _file(
        Path(config["network_barrier_qualification"]),
        "network client frame-barrier qualification",
    )
    barrier = _json(barrier_path, "network client frame-barrier qualification")

    full_manifest_path = _file(
        runtime_root / RUNTIME_MANIFEST_NAME, "sealed runtime manifest"
    )
    full_manifest = load_runtime_manifest(full_manifest_path)
    verified = verify_runtime_manifest(full_manifest)
    if not verified.valid:
        raise OneRunError(
            "sealed runtime manifest is invalid: " + "; ".join(verified.errors)
        )
    if verified.digest != args.expected_runtime_manifest_sha256:
        raise OneRunError("sealed runtime manifest digest differs from CLI admission")
    runtime_evidence = _json(runtime_evidence_path, "runtime_evidence")
    validated_runtime = validate_runtime_evidence(
        runtime_evidence, expected_atlas_sha256=args.expected_atlas_sha256
    )
    if validated_runtime.runtime_manifest_sha256 != verified.digest:
        raise OneRunError("compact runtime evidence is not bound to full manifest")

    semantic = full_manifest.get("semantic")
    try:
        artifacts = semantic["artifacts"]
        _record_matches(q2ded, artifacts["q2ded"], "q2ded")
        game_module = _file(q2_root / config["game"] / "game.so", "game module")
        _record_matches(game_module, artifacts["game_module"], "game module")
        rust_record = artifacts["rust_lattice"]
        if rust_record.get("enabled") is not True:
            raise OneRunError("sealed runtime does not enable the Rust lattice")
        _record_matches(rust_extension, rust_record, "Rust extension")
        runtime_semantics = semantic["runtime_config"]
    except (KeyError, TypeError) as error:
        raise OneRunError("sealed runtime artifact envelope is incomplete") from error
    barrier_execution_sha256 = _validate_network_barrier(
        barrier_path=barrier_path,
        barrier=barrier,
        runtime_manifest_sha256=verified.digest,
        runtime_semantics=runtime_semantics,
        q2ded=q2ded,
        game_module=game_module,
        client_binary=client_binary,
    )
    expected_semantics = {
        "proof_module": "train.multires_one_run",
        "trainer_module": "train.multires_primary",
        "legacy_fallback_enabled": False,
        "deterministic_collection": True,
        "client_count": client_count,
        "use_startobserver": 0,
        "use_startchasecam": 0,
        "training_config_sha256": training_sha256,
        "retirement_manifest_sha256": retirement_sha256,
        "network_barrier_execution_evidence_sha256": barrier_execution_sha256,
        "optimizer": config["optimizer"],
        "client_binary_sha256": _sha256(client_binary),
        "client_binary_size": client_binary.stat().st_size,
        "dyn_snapshots": [
            {
                "client_id": index,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for index, path in enumerate(dyn_snapshots)
        ],
    }
    if not isinstance(runtime_semantics, Mapping) or any(
        runtime_semantics.get(key) != value
        for key, value in expected_semantics.items()
    ):
        raise OneRunError("sealed runtime semantic trainer/artifact bindings differ")

    bundle_bsp = _file(directory / f"{args.map_name}.bsp", "bundle BSP")
    bundle_bsp_sha256 = _sha256(bundle_bsp)
    map_records = semantic.get("maps", [])
    selected = [item for item in map_records if item.get("name") == args.map_name]
    if len(selected) != 1:
        raise OneRunError("sealed runtime does not select exactly one requested map")
    bsp_records = [
        item for item in selected[0].get("files", [])
        if str(item.get("name", "")).endswith(f"/{args.map_name}.bsp")
    ]
    if len(bsp_records) != 1 or bsp_records[0].get("sha256") != bundle_bsp_sha256:
        raise OneRunError("sealed runtime map BSP differs from admitted bundle")
    installed_bsp = q2_root / str(bsp_records[0]["name"])
    _record_matches(_file(installed_bsp, "installed BSP"), bsp_records[0], "map BSP")

    optimizer = config["optimizer"]
    if (
        not isinstance(optimizer, Mapping)
        or optimizer.get("class") != "torch.optim.Adam"
        or not isinstance(optimizer.get("learning_rate"), (int, float))
        or isinstance(optimizer.get("learning_rate"), bool)
        or not math.isfinite(float(optimizer["learning_rate"]))
        or float(optimizer["learning_rate"]) <= 0.0
        or not isinstance(optimizer.get("kwargs"), Mapping)
        or "lr" in optimizer["kwargs"]
    ):
        raise OneRunError("runtime optimizer configuration is not admitted Adam")
    if config["telemetry_token_env"] != "Q2_ML_CLIENT_TELEMETRY_TOKEN":
        raise OneRunError("telemetry token must use the frozen environment name")
    token = os.environ.get(config["telemetry_token_env"], "")
    if not _TOKEN.fullmatch(token):
        raise OneRunError("telemetry token environment is missing or malformed")
    if checkpoint.stat().st_size <= 0:
        raise OneRunError("checkpoint is empty")
    _validate_step_zero_checkpoint(
        checkpoint,
        atlas_sha256=args.expected_atlas_sha256,
        runtime_manifest_sha256=verified.digest,
        training_configuration=training_configuration,
        optimizer_configuration=optimizer,
    )

    return OneRunAdmission(
        args=args,
        runtime_root=runtime_root,
        runtime_config=config,
        runtime_manifest_sha256=verified.digest,
        runtime_evidence=runtime_evidence,
        training_configuration=training_configuration,
        training_manifest_sha256=training_sha256,
        objective_identity_sha256=str(objective_identity),
        checkpoint_sha256=_sha256(checkpoint),
        q2_root=q2_root,
        q2ded=q2ded,
        client_binary=client_binary,
        bundle_manifest=bundle_manifest,
        objectives=objectives,
        atlas_bin=atlas_bin,
        dyn_snapshots=dyn_snapshots,
        rust_extension=rust_extension,
        retirement_manifest_sha256=retirement_sha256,
        network_barrier_execution_evidence_sha256=barrier_execution_sha256,
    )


def _load_extension(path: Path) -> ModuleType:
    sys.modules.pop("q2_lattice_rs", None)
    spec = importlib.util.spec_from_file_location("q2_lattice_rs", path)
    if spec is None or spec.loader is None:
        raise OneRunError("cannot load the admitted q2_lattice_rs extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_server_config(admission: OneRunAdmission) -> Path:
    args = admission.args
    config = admission.runtime_config
    token = os.environ[str(config["telemetry_token_env"])]
    game_directory = admission.q2_root / str(config["game"])
    game_directory.mkdir(parents=True, exist_ok=True)
    destination = game_directory / f"multires_one_run_{args.launch_id}.cfg"
    if destination.is_symlink():
        raise OneRunError("server launch config symlinks are rejected")
    lines = [
        "set dedicated 1",
        "set deathmatch 1",
        "set cheats 1",
        "set timedemo 0",
        "set timescale 1",
        "set sv_fps 10",
        f"set maxclients {int(config['client_count'])}",
        "set autospawn 0",
        'set botlist ""',
        "set allow_client_bot_controls 0",
        # Global ML mode supplies deterministic gameplay seeding and hook-zone
        # sidecars only; slot 99 plus empty botlist prevents the retired
        # in-process bot runtime from owning any real client.
        "set ml_enabled 1",
        "set ml_async 0",
        "set sv_ml_frame_barrier 1",
        f"set sv_ml_frame_barrier_clients {int(config['client_count'])}",
        "set sv_ml_frame_barrier_timeout_ms 5000",
        # Qualification controls are startup-sealed by the first map.  A
        # training launch must therefore prove a fresh, fault-free process.
        "set sv_ml_frame_barrier_test_mode 0",
        'set sv_ml_frame_barrier_test_fault ""',
        "set sv_ml_frame_barrier_test_tick 0",
        "set ml_bot_slot 99",
        "set ml_teacher_enabled 0",
        f"set ml_game_seed {int(args.game_seed)}",
        "set ml_client_telemetry 1",
        f"set ml_client_telemetry_port {int(config['telemetry_port'])}",
        f'set ml_client_telemetry_token "{token}"',
        "set use_mapqueue 0",
        'set mapqueue ""',
        # A barrier cohort must enter as ordinary players.  Never inherit
        # Lithium's observer-first defaults from a host installation.
        "set use_startobserver 0",
        "set use_startchasecam 0",
        f"map {args.map_name}",
        "",
    ]
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _stop_server(process: subprocess.Popen, *, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)


def _process_record(role: str, pid: int) -> dict[str, Any]:
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        state = str(tail[0])
        start_ticks = int(tail[19])
    except (OSError, IndexError, ValueError) as error:
        raise OneRunError(f"cannot attest launched process {pid}") from error
    if state == "Z" or int(pid) <= 1 or start_ticks <= 0:
        raise OneRunError(f"launched process {pid} is not live/attestable")
    return {"role": role, "pid": int(pid), "start_ticks": start_ticks}


def _process_record_alive(record: Mapping[str, Any]) -> bool:
    try:
        current = _process_record(str(record["role"]), int(record["pid"]))
    except OneRunError:
        return False
    return current["start_ticks"] == int(record["start_ticks"])


def _capture_client_process_records(
    batch: Any, process_records: list[dict[str, Any]]
) -> None:
    known = {int(record["pid"]) for record in process_records}
    for index, env in enumerate(batch.envs):
        process = getattr(env, "_process", None)
        if process is None or int(process.pid) in known:
            continue
        record = _process_record(f"network-client-{index:02d}", int(process.pid))
        process_records.append(record)
        known.add(int(process.pid))


def _records(rollout: Any, admission: OneRunAdmission) -> tuple[list[dict[str, Any]], str]:
    args = admission.args
    clients, time_steps = rollout.observations.shape[:2]
    if clients * time_steps != args.transition_count:
        raise OneRunError("collector rollout cardinality differs from requested total")
    built = []
    emitted = []
    index = 0
    for time_index in range(time_steps):
        infos = rollout.infos[time_index]
        if len(infos) != clients:
            raise OneRunError("collector info cardinality differs")
        for client_index in range(clients):
            info = infos[client_index]
            attestation = info.get("_multires_spatial_attestation")
            if not isinstance(attestation, Mapping):
                raise OneRunError("transition lacks spatial attestation")
            if (
                info.get("map") != args.map_name
                or attestation.get("map_epoch") != args.map_epoch
                or attestation.get("atlas_sha256") != args.expected_atlas_sha256
                or attestation.get("runtime_manifest_sha256")
                != admission.runtime_manifest_sha256
            ):
                raise OneRunError("transition map/Atlas/runtime identity differs")
            observation = rollout.observations[client_index, time_index]
            rust_features = observation[DYN.slice]
            record = make_transition_record(
                index=index,
                observation=[float(value) for value in observation],
                action=[
                    float(value)
                    for value in rollout.actions[client_index, time_index]
                ],
                reward=float(rollout.rewards[client_index, time_index]),
                client_id=str(info["client_id"]),
                server_frame=int(info["server_frame"]),
                batch_round_id=int(info["batch_round_id"]),
                policy_version=int(info["policy_version"]),
                map_name=str(info["map"]),
                map_epoch=int(attestation["map_epoch"]),
                atlas_sha256=str(attestation["atlas_sha256"]),
                runtime_manifest_sha256=str(
                    attestation["runtime_manifest_sha256"]
                ),
                rust_features=[float(value) for value in rust_features],
            )
            item = record.to_mapping()
            item["rust_features"] = [float(value) for value in rust_features]
            built.append(record)
            emitted.append(item)
            index += 1
    return emitted, trajectory_sha256(tuple(built))


def execute(admission: OneRunAdmission) -> dict[str, Any]:
    # Heavy imports occur only after the pure artifact preflight succeeded.
    import numpy as np
    import torch

    from harness.client_batch import build_network_client_batch
    from harness.multires_collector import CollectorConfig
    from harness.multires_reward import CausalRewardConfig
    from harness.multires_contract import GuideDropoutConfig
    from harness.rust_multires_provider import (
        RustAtlasProviderFactory,
        RustMapArtifacts,
    )
    from train.multires_live import MultiresLiveTrainer
    from train.multires_ppo import MultiresPPOConfig
    from train.multires_runtime import MultiresTrainerRuntime

    args = admission.args
    config = admission.runtime_config
    client_count = int(config["client_count"])
    per_client = args.transition_count // client_count
    device = torch.device(str(config["device"]))
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    training = admission.training_configuration
    reward_config = CausalRewardConfig(**dict(training.reward))
    guide_dropout = GuideDropoutConfig(**dict(training.guide_dropout))
    ppo_config = MultiresPPOConfig(**dict(training.ppo))
    optimizer_config = config["optimizer"]

    def optimizer_factory(parameters):
        return torch.optim.Adam(
            parameters,
            lr=float(optimizer_config["learning_rate"]),
            **dict(optimizer_config["kwargs"]),
        )

    runtime, optimizer, checkpoint_manifest = MultiresTrainerRuntime.resume(
        Path(args.checkpoint),
        admission.runtime_evidence,
        expected_atlas_sha256=args.expected_atlas_sha256,
        device=device,
        optimizer_factory=optimizer_factory,
        reward_config=reward_config,
        guide_dropout=guide_dropout,
        ppo_config=ppo_config,
    )
    if runtime.runtime.runtime_manifest_sha256 != admission.runtime_manifest_sha256:
        raise OneRunError("trainer runtime differs from sealed runtime manifest")
    if (
        checkpoint_manifest.initialization != "random"
        or checkpoint_manifest.training_step != 0
        or not _valid_sha256(checkpoint_manifest.lineage_root_sha256)
    ):
        raise OneRunError(
            "one-run qualification requires an attested fresh random step-zero lineage"
        )

    extension = _load_extension(admission.rust_extension)
    artifacts = RustMapArtifacts(
        bundle_manifest_path=admission.bundle_manifest,
        uncompressed_atlas_path=admission.atlas_bin,
        dyn_snapshot_path=admission.dyn_snapshots[0],
        expected_atlas_sha256=args.expected_atlas_sha256,
    )
    factories = [
        RustAtlasProviderFactory(
            extension_module=extension,
            artifacts_by_map={
                args.map_name: RustMapArtifacts(
                    bundle_manifest_path=artifacts.bundle_manifest_path,
                    uncompressed_atlas_path=artifacts.uncompressed_atlas_path,
                    dyn_snapshot_path=admission.dyn_snapshots[index],
                    expected_atlas_sha256=artifacts.expected_atlas_sha256,
                )
            },
            runtime_manifest_sha256=admission.runtime_manifest_sha256,
            bound_client_id=f"{config['client_id_prefix']}-{index:02d}",
            rust_client_id=index,
            client_count=client_count,
        )
        for index in range(client_count)
    ]

    launch_cfg = _write_server_config(admission)
    server_command = [
        str(admission.q2ded),
        "+set", "game", str(config["game"]),
        "+set", "port", str(int(config["server_port"])),
        "+exec", launch_cfg.name,
    ]
    server = subprocess.Popen(
        server_command,
        cwd=admission.q2_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    process_records: list[dict[str, Any]] = []
    batch = None
    records: list[dict[str, Any]] | None = None
    trajectory = ""
    metrics = None
    cleanup_error: Exception | None = None
    try:
        process_records.append(_process_record("q2ded", int(server.pid)))
        warmup = float(config["server_warmup_seconds"])
        if not math.isfinite(warmup) or not 0.0 <= warmup <= 30.0:
            raise OneRunError("server_warmup_seconds must be within [0, 30]")
        if warmup:
            time.sleep(warmup)
        if server.poll() is not None:
            raise OneRunError("q2ded exited before clients connected")
        token = os.environ[str(config["telemetry_token_env"])]
        batch = build_network_client_batch(
            n_clients=client_count,
            server=f"{config['server_host']}:{int(config['server_port'])}",
            telemetry_server=(
                f"{config['telemetry_host']}:{int(config['telemetry_port'])}"
            ),
            telemetry_token=token,
            client_binary=str(admission.client_binary),
            client_root=str(admission.q2_root),
            client_data_root=str(config["client_data_root"]),
            harness_host=str(config["harness_host"]),
            harness_port_base=int(config["harness_port_base"]),
            qport_base=int(config["qport_base"]),
            client_id_prefix=str(config["client_id_prefix"]),
            name_prefix=str(config["name_prefix"]),
            game=str(config["game"]),
            client_timeout=float(config["client_timeout"]),
            round_timeout=float(config["round_timeout"]),
            max_rejected_echoes=int(config["max_rejected_echoes"]),
            movement_tolerance=float(config["movement_tolerance"]),
            look_tolerance=float(config["look_tolerance"]),
            spatial_seed=args.seed,
            debug=bool(config["debug_clients"]),
            deterministic_frame_barrier=True,
            multires_spatial_provider_factories=factories,
            expected_runtime_manifest_sha256=admission.runtime_manifest_sha256,
        )
        trainer = MultiresLiveTrainer(
            runtime,
            batch,
            optimizer,
            device=device,
            collector_config=CollectorConfig(
                transitions_per_client=per_client,
                maximum_boundary_rounds=int(config["maximum_boundary_rounds"]),
            ),
            deterministic_collection=True,
        )
        initial_observations, initial_infos = batch.reset()
        _capture_client_process_records(batch, process_records)
        trainer.collector.prime(initial_observations, initial_infos)
        rollout = trainer.collector.collect(policy_version=args.policy_version)
        if rollout.boundary_rounds != 0:
            raise OneRunError("proof run encountered a resync/boundary admission")
        metrics = batch.metrics
        for field in (
            "stale_policy_rounds_rejected", "stale_echoes_rejected",
            "mismatched_echoes_rejected", "map_epoch_resyncs",
            "telemetry_gap_resyncs", "realtime_catchup_resyncs",
            "action_state_resyncs",
        ):
            if int(getattr(metrics, field)) != 0:
                raise OneRunError(f"proof run transport metric {field} is nonzero")
        records, trajectory = _records(rollout, admission)
    finally:
        if batch is not None:
            try:
                _capture_client_process_records(batch, process_records)
            except Exception as error:
                cleanup_error = error
        try:
            if batch is not None:
                batch.close()
        except Exception as error:  # preserve cleanup evidence over convenience
            cleanup_error = error
        try:
            _stop_server(server)
        except Exception as error:
            cleanup_error = cleanup_error or error
        launch_cfg.unlink(missing_ok=True)

    orphans = [
        int(record["pid"])
        for record in process_records
        if _process_record_alive(record)
    ]
    if cleanup_error is not None or orphans:
        raise OneRunError(
            f"scoped teardown failed cleanup={cleanup_error!r} orphans={orphans!r}"
        )
    if records is None or len(records) != args.transition_count:
        raise OneRunError("run ended without the exact requested records")
    if len(process_records) != 1 + int(config["client_count"]):
        raise OneRunError("run lacks exact q2ded/client PID start-tick evidence")
    launched_pids = [int(record["pid"]) for record in process_records]

    received = {
        "seed": args.seed,
        "game_seed": args.game_seed,
        "q2ded": str(admission.q2ded),
        "client_binary": str(admission.client_binary),
        "runtime_root": str(admission.runtime_root),
        "bundle_manifest": str(admission.bundle_manifest),
        "objectives": str(admission.objectives),
        "atlas_bin": str(admission.atlas_bin),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "training_manifest": str(Path(args.training_manifest).resolve()),
        "runtime_evidence": str(Path(args.runtime_evidence).resolve()),
        "transition_count": args.transition_count,
        "policy_version": args.policy_version,
        "map_epoch": args.map_epoch,
        "map_name": args.map_name,
        "out": str(Path(args.out).resolve()),
        "launch_id": args.launch_id,
        "expected_atlas_sha256": args.expected_atlas_sha256,
        "expected_runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
    }
    return {
        "schema": ONE_RUN_SCHEMA,
        "protocol_version": ONE_RUN_PROTOCOL_VERSION,
        "synthetic": False,
        "legacy": False,
        "collector": COLLECTOR_CLASS_NAME,
        "python_collector_schema": PYTHON_COLLECTOR_SCHEMA,
        "spatial_provider": SPATIAL_PROVIDER_CLASS_NAME,
        "rust_provider_schema": RUST_PROVIDER_SCHEMA,
        "lattice_crate": LATTICE_CRATE_NAME,
        "b4_protocol_generation": B4_PROTOCOL_GENERATION,
        "qm3c_causal_magic": B4_CAUSAL_MAGIC,
        "client_wire_version": runtime.runtime.client_wire_version,
        "policy_generation": POLICY_GENERATION,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "atlas_sha256": args.expected_atlas_sha256,
        "runtime_manifest_sha256": admission.runtime_manifest_sha256,
        "objective_identity_sha256": admission.objective_identity_sha256,
        "training_manifest_sha256": admission.training_manifest_sha256,
        "checkpoint_sha256": admission.checkpoint_sha256,
        "received_inputs": received,
        "transition_count": args.transition_count,
        "trajectory_sha256": trajectory,
        "partial_admissions": 0,
        "stale_admissions": 0,
        "resync_admissions": 0,
        "seed": args.seed,
        "game_seed": args.game_seed,
        "policy_version": args.policy_version,
        "map_epoch": args.map_epoch,
        "map_name": args.map_name,
        "process_ids": launched_pids,
        "launched_process_ids": launched_pids,
        "terminated_process_ids": launched_pids,
        "process_records": process_records,
        "terminated_process_records": process_records,
        "orphan_processes_after_teardown": 0,
        "checkpoint_training_step": checkpoint_manifest.training_step,
        "records": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--game_seed", type=int, required=True)
    parser.add_argument("--q2ded", type=Path, required=True)
    parser.add_argument("--client_binary", type=Path, required=True)
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument("--bundle_manifest", type=Path, required=True)
    parser.add_argument("--objectives", type=Path, required=True)
    parser.add_argument("--atlas_bin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training_manifest", type=Path, required=True)
    parser.add_argument("--runtime_evidence", type=Path, required=True)
    parser.add_argument("--transition_count", type=int, required=True)
    parser.add_argument("--policy_version", type=int, required=True)
    parser.add_argument("--map_epoch", type=int, required=True)
    parser.add_argument("--map_name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch_id", required=True)
    parser.add_argument("--expected_atlas_sha256", required=True)
    parser.add_argument("--expected_runtime_manifest_sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        admission = preflight(args)
        payload = execute(admission)
        destination = Path(args.out)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_bytes(_canonical_bytes(payload))
        os.replace(temporary, destination)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except Exception as error:
        print(f"multires one-run failed: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
