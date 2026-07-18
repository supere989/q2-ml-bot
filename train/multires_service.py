"""PID-scoped supervisor for proof and the continuous multires trainer.

``prove`` is always the finite 500-transition, zero-update qualification.
``start`` supervises only ``train.multires_primary`` and therefore cannot be
confused with proof execution or any retired trainer generation.  There is no
crash-unrecoverable foreground training selector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.multires_one_run import (  # noqa: E402
    OneRunAdmission,
    OneRunError,
    RUNTIME_CONFIG_NAME,
    preflight as one_run_preflight,
)
from train.multires_primary import (  # noqa: E402
    PRIMARY_CHILD_INVENTORY_NAME,
    PRIMARY_CHILD_INVENTORY_SCHEMA,
    PRIMARY_ATTEMPT_NAME,
    PRIMARY_ATTEMPT_FIELDS,
    PRIMARY_ATTEMPT_SCHEMA,
    PRIMARY_SELECTOR_TOKEN_ENV,
    PRIMARY_TERMINAL_NAME,
    PRIMARY_TERMINAL_FIELDS,
    PRIMARY_TERMINAL_SCHEMA,
    PRIMARY_TRAINER_MODULE,
    PROOF_MODULE,
    PrimaryTrainingAdmission,
    PrimaryTrainingError,
    admit_primary_training,
)
from tools.verify_multires_integration import (  # noqa: E402
    ENVELOPE_SCHEMA as INTEGRATION_ENVELOPE_SCHEMA,
    GATE_ORDER as INTEGRATION_GATE_ORDER,
    REPORT_SCHEMA as INTEGRATION_REPORT_SCHEMA,
    canonical_report_bytes as canonical_integration_report_bytes,
    run_gates as run_integration_gates,
)


SERVICE_CONFIG_SCHEMA = "q2-multires-service-v2"
SERVICE_PREFLIGHT_SCHEMA = "q2-multires-service-preflight-v2"
SERVICE_STATE_SCHEMA = "q2-multires-service-state-v2"
SERVICE_LEASE_SCHEMA = "q2-multires-service-lease-v1"
RETIREMENT_VALIDATION_SCHEMA = "q2-multires-retirement-validation-v1"
SERVICE_MODE = "multires-primary-trainer"
PROOF_ROLE = "qualification-proof"
PRIMARY_ROLE = "primary-trainer"
SERVICE_CONFIG_NAME = "multires-service.json"
SERVICE_STATE_NAME = "multires-service-state.json"
SERVICE_LEASE_NAME = "multires-service.lock"
PRIMARY_ADMISSION_TIMEOUT_SECONDS = 30.0
_SHA256_CHARS = frozenset("0123456789abcdef")


class MultiresServiceError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
        and value != "0" * 64
    )


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiresServiceError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise MultiresServiceError(f"{label} must be a JSON object")
    return value


def _absolute_file(value: object, label: str, *, executable: bool = False) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MultiresServiceError(f"{label} must be an absolute non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise MultiresServiceError(f"{label} is not executable")
    return path.resolve()


def _absolute_directory(value: object, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise MultiresServiceError(
            f"{label} must be an existing absolute non-symlink directory"
        )
    return path.resolve()


@dataclass(frozen=True)
class ServiceAdmission:
    runtime_root: Path
    config: Mapping[str, Any]
    one_run: OneRunAdmission
    proof_command: tuple[str, ...]
    evidence_dir: Path
    proof_out: Path
    log_path: Path
    primary: PrimaryTrainingAdmission
    trainer_command: tuple[str, ...]
    current_run_root: Path
    tensorboard_root: Path
    current_season_report: Path
    child_inventory: Path
    terminal_evidence: Path
    launch_config: Path
    shutdown_grace_seconds: float
    tensorboard_command: tuple[str, ...] | None
    retirement_validation_command: tuple[str, ...]
    retirement_validation: Mapping[str, Any]
    integration_envelope: Path
    integration_report: Path
    integration_report_sha256: str
    integration_verification: Mapping[str, Any]


def _artifact_record(
    value: object, label: str,
) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise MultiresServiceError(f"{label} record is malformed")
    path = _absolute_file(value["path"], label)
    digest = value["sha256"]
    if not _valid_sha256(digest) or digest != _sha256(path):
        raise MultiresServiceError(f"{label} byte digest differs")
    return path, {"path": str(path), "sha256": str(digest)}


def _integration_evidence_paths(envelope_path: Path) -> dict[str, Path]:
    envelope = _json(envelope_path, "integration evidence envelope")
    if set(envelope) != {"schema", "evidence"} or envelope.get(
        "schema"
    ) != INTEGRATION_ENVELOPE_SCHEMA:
        raise MultiresServiceError(
            "integration evidence envelope fields/schema differ"
        )
    evidence = envelope.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        INTEGRATION_GATE_ORDER
    ):
        raise MultiresServiceError("integration evidence inventory differs")
    paths: dict[str, Path] = {}
    for gate in INTEGRATION_GATE_ORDER:
        raw = evidence[gate]
        if not isinstance(raw, str) or not raw:
            raise MultiresServiceError(
                f"integration evidence path for {gate} is invalid"
            )
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = envelope_path.parent / candidate
        paths[gate] = _absolute_file(candidate, f"integration {gate} evidence")
    return paths


def _require_b6_file_binding(
    bindings: Mapping[str, Any], name: str, expected: Path,
) -> None:
    record = bindings.get(name)
    if (
        not isinstance(record, Mapping)
        or set(record) != {"bytes", "sha256"}
        or record.get("bytes") != expected.stat().st_size
        or record.get("sha256") != _sha256(expected)
    ):
        raise MultiresServiceError(
            f"integration B6 {name} binding differs from service input"
        )


def _current_bot_source_identity() -> dict[str, Any]:
    top = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    try:
        top_level = Path(top.stdout.strip()).resolve(strict=True)
    except (OSError, RuntimeError):
        top_level = None
    if top.returncode != 0 or top_level != ROOT.resolve():
        raise MultiresServiceError(
            "primary trainer must run from the exact bot Git worktree root"
        )
    values: list[str] = []
    for revision in ("HEAD^{commit}", "HEAD^{tree}"):
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", revision],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0 or not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value
        ):
            raise MultiresServiceError("cannot attest current bot Git source")
        values.append(value)
    status = subprocess.run(
        [
            "git", "-C", str(ROOT), "status", "--porcelain=v1",
            "--untracked-files=normal",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if status.returncode != 0 or status.stdout:
        raise MultiresServiceError("current bot Git source is dirty or unreadable")
    return {"commit": values[0], "tree": values[1], "clean": True}


def _verify_integration_admission(
    *, runtime_root: Path, config: Mapping[str, Any],
    cold_inputs: Mapping[str, Any], retirement_manifest: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Re-evaluate the final B2-B6 envelope and bind it to this service."""
    admission = config.get("integration_admission")
    if not isinstance(admission, Mapping) or set(admission) != {
        "envelope", "report", "bot_source"
    }:
        raise MultiresServiceError("integration admission records are malformed")
    bot_source = admission["bot_source"]
    if (
        not isinstance(bot_source, Mapping)
        or set(bot_source) != {"commit", "tree"}
        or any(
            not isinstance(bot_source.get(name), str)
            or re.fullmatch(
                r"(?:[0-9a-f]{40}|[0-9a-f]{64})", bot_source[name]
            ) is None
            for name in ("commit", "tree")
        )
    ):
        raise MultiresServiceError("integration bot source record is malformed")
    if cold_inputs.get("integration_bot_source") != dict(bot_source):
        raise MultiresServiceError("cold-start integration bot source differs")
    envelope_path, envelope_record = _artifact_record(
        admission["envelope"], "integration evidence envelope"
    )
    report_path, report_record = _artifact_record(
        admission["report"], "precomputed integration report"
    )
    if cold_inputs.get("integration_envelope") != envelope_record:
        raise MultiresServiceError(
            "cold-start integration envelope binding differs"
        )
    if cold_inputs.get("integration_report") != report_record:
        raise MultiresServiceError(
            "cold-start integration report binding differs"
        )

    evidence_paths = _integration_evidence_paths(envelope_path)
    rerun = run_integration_gates(envelope_path)
    rerun_payload = canonical_integration_report_bytes(rerun) + b"\n"
    try:
        precomputed_payload = report_path.read_bytes()
    except OSError as error:
        raise MultiresServiceError(
            "precomputed integration report is unreadable"
        ) from error
    if precomputed_payload != rerun_payload:
        raise MultiresServiceError(
            "precomputed integration report bytes differ from fresh verification"
        )
    report = _json(report_path, "precomputed integration report")
    unsigned = dict(report)
    report_seal = unsigned.pop("report_sha256", None)
    if (
        report != rerun
        or report.get("schema") != INTEGRATION_REPORT_SCHEMA
        or report.get("overall") != "pass"
        or report.get("failed_gates") != []
        or not _valid_sha256(report_seal)
        or report_seal != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or report.get("envelope_sha256") != envelope_record["sha256"]
        or report_record["sha256"] != hashlib.sha256(
            precomputed_payload
        ).hexdigest()
    ):
        raise MultiresServiceError(
            "integration report status/hash/seal differs"
        )
    gates = report.get("gates")
    if (
        not isinstance(gates, list)
        or [entry.get("gate") for entry in gates if isinstance(entry, Mapping)]
        != list(INTEGRATION_GATE_ORDER)
        or any(
            not isinstance(entry, Mapping)
            or entry.get("status") != "pass"
            or entry.get("reasons") != []
            or entry.get("evidence_sha256")
            != _sha256(evidence_paths[str(entry.get("gate"))])
            for entry in gates
        )
    ):
        raise MultiresServiceError("integration report gate inventory differs")

    proof = config["proof"]
    atlas_sha256 = proof["expected_atlas_sha256"]
    runtime_sha256 = config["runtime_manifest_sha256"]
    retirement_sha256 = config["retirement_manifest_sha256"]
    bundle_gate = _json(
        evidence_paths["bundle_v3_atlas"], "bundle-v3 integration evidence"
    )
    lineage_gate = _json(
        evidence_paths["lineage_attestation"], "lineage integration evidence"
    )
    retirement_gate = _json(
        evidence_paths["legacy_selector_deactivation"],
        "retirement integration evidence",
    )
    b6 = _json(evidence_paths["wsl_b6_campaign"], "B6 integration evidence")
    bindings = b6.get("bindings")
    b6_sources = b6.get("source_repositories")
    expected_bot_source = {**dict(bot_source), "clean": True}
    if (
        bundle_gate.get("atlas_sha256") != atlas_sha256
        or lineage_gate.get("atlas_sha256") != atlas_sha256
        or lineage_gate.get("runtime_manifest_sha256") != runtime_sha256
        or retirement_gate.get("retirement_manifest_sha256")
        != retirement_sha256
        or not isinstance(bindings, Mapping)
        or bindings.get("atlas_sha256") != atlas_sha256
        or bindings.get("runtime_manifest_identity_sha256") != runtime_sha256
        or not isinstance(bindings.get("retirement_manifest"), Mapping)
        or bindings["retirement_manifest"].get("sha256") != retirement_sha256
        or not isinstance(b6_sources, Mapping)
        or b6_sources.get("bot") != expected_bot_source
        or _current_bot_source_identity() != expected_bot_source
    ):
        raise MultiresServiceError(
            "integration runtime/Atlas/retirement/source identities differ from service"
        )

    exact_files = {
        "runtime_evidence": Path(str(proof["runtime_evidence"])).resolve(),
        "runtime_manifest": (runtime_root / "runtime-manifest.json").resolve(),
        "checkpoint": Path(str(proof["checkpoint"])).resolve(),
        "training_manifest": Path(str(proof["training_manifest"])).resolve(),
        "objectives": Path(str(proof["objectives"])).resolve(),
        "bundle_manifest": Path(str(proof["bundle_manifest"])).resolve(),
        "atlas": Path(str(proof["atlas_bin"])).resolve(),
    }
    for name, path in exact_files.items():
        _require_b6_file_binding(bindings, name, _absolute_file(path, name))
    _require_b6_file_binding(bindings, "retirement_manifest", retirement_manifest)
    return envelope_path, report_path, report


def _one_run_namespace(
    runtime_root: Path, proof: Mapping[str, Any], evidence_dir: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        seed=int(proof["seed"]),
        game_seed=int(proof["game_seed"]),
        q2ded=Path(proof["q2ded"]),
        client_binary=Path(proof["client_binary"]),
        runtime_root=runtime_root,
        bundle_manifest=Path(proof["bundle_manifest"]),
        objectives=Path(proof["objectives"]),
        atlas_bin=Path(proof["atlas_bin"]),
        checkpoint=Path(proof["checkpoint"]),
        training_manifest=Path(proof["training_manifest"]),
        runtime_evidence=Path(proof["runtime_evidence"]),
        atlas_catalog=Path(proof["atlas_catalog"]),
        transition_count=int(proof["transition_count"]),
        policy_version=int(proof["policy_version"]),
        map_epoch=int(proof["map_epoch"]),
        map_name=str(proof["map_name"]),
        out=(evidence_dir / "service-preflight-one-run.json").resolve(),
        launch_id="service-preflight",
        expected_atlas_sha256=str(proof["expected_atlas_sha256"]),
        expected_atlas_catalog_sha256=str(
            proof["expected_atlas_catalog_sha256"]
        ),
        expected_runtime_manifest_sha256=str(
            proof["expected_runtime_manifest_sha256"]
        ),
    )


def service_preflight(runtime_root: Path) -> ServiceAdmission:
    runtime_root = _absolute_directory(runtime_root, "runtime_root")
    config_path = _absolute_file(
        runtime_root / SERVICE_CONFIG_NAME, "service configuration"
    )
    config = _json(config_path, "service configuration")
    if set(config) != {
        "schema", "retirement_manifest_sha256", "runtime_manifest_sha256",
        "retirement_manifest", "retirement_cold_start", "operational_roots",
        "service_selectors", "modules", "proof", "training_runtime",
        "integration_admission", "evidence_dir", "log_path", "tensorboard",
    } or config.get("schema") != SERVICE_CONFIG_SCHEMA:
        raise MultiresServiceError("service configuration fields/schema differ")
    if config["modules"] != {
        "proof_module": PROOF_MODULE,
        "trainer_module": PRIMARY_TRAINER_MODULE,
    }:
        raise MultiresServiceError("service proof/trainer module selectors differ")
    proof = config["proof"]
    required_proof = {
        "seed", "game_seed", "divergence_game_seed", "transition_count",
        "policy_version", "map_name", "map_epoch", "timeout_seconds", "q2ded",
        "client_binary", "bundle_manifest", "objectives", "atlas_bin",
        "atlas_catalog",
        "checkpoint", "training_manifest", "runtime_evidence",
        "expected_atlas_sha256", "expected_atlas_catalog_sha256",
        "expected_runtime_manifest_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != required_proof:
        raise MultiresServiceError("service proof fields differ")
    for field in (
        "seed", "game_seed", "divergence_game_seed", "transition_count",
        "policy_version", "map_epoch",
    ):
        if type(proof[field]) is not int or int(proof[field]) < 0:
            raise MultiresServiceError(f"service proof {field} type/value is invalid")
    if (
        isinstance(proof["timeout_seconds"], bool)
        or not isinstance(proof["timeout_seconds"], (int, float))
        or not math.isfinite(float(proof["timeout_seconds"]))
        or not float(proof["timeout_seconds"]) > 0.0
    ):
        raise MultiresServiceError("service proof timeout type/value is invalid")
    if not isinstance(proof["map_name"], str) or not proof["map_name"]:
        raise MultiresServiceError("service proof map_name is invalid")
    if int(proof["transition_count"]) != 500:
        raise MultiresServiceError("service qualification requires exactly 500 transitions")
    if proof["expected_runtime_manifest_sha256"] != config["runtime_manifest_sha256"]:
        raise MultiresServiceError("service/full runtime manifest digests differ")
    if not _valid_sha256(proof["expected_atlas_catalog_sha256"]):
        raise MultiresServiceError("service Atlas catalog digest is invalid")
    evidence_dir = _absolute_directory(config["evidence_dir"], "evidence_dir")
    log_path = Path(str(config["log_path"]))
    if not log_path.is_absolute() or log_path.is_symlink():
        raise MultiresServiceError("log_path must be absolute and non-symlink")
    _absolute_directory(log_path.parent, "log_path parent")
    if log_path.resolve() != runtime_root / "multires-service.log":
        raise MultiresServiceError(
            "service log_path must be the wrapper's exact runtime-root log"
        )

    # The retirement validator is deliberately a separate, read-only process.
    # It is the first semantic gate in service preflight and therefore runs
    # before one-run preflight can create a client-data/output directory, and
    # before any service state or child process is created.
    retirement_manifest = _absolute_file(
        config["retirement_manifest"], "retirement manifest"
    )
    retirement_cold_start = _absolute_file(
        config["retirement_cold_start"], "retirement cold-start declaration"
    )
    cold_document = _json(
        retirement_cold_start, "retirement cold-start declaration"
    )
    cold_inputs = cold_document.get("inputs")
    if not isinstance(cold_inputs, Mapping):
        raise MultiresServiceError("cold-start input bindings are missing")
    integration_envelope, integration_report, integration_verification = (
        _verify_integration_admission(
            runtime_root=runtime_root,
            config=config,
            cold_inputs=cold_inputs,
            retirement_manifest=retirement_manifest,
        )
    )
    root_values = config["operational_roots"]
    if not isinstance(root_values, list) or not root_values:
        raise MultiresServiceError("operational_roots must be a nonempty list")
    operational_roots = tuple(
        _absolute_directory(value, f"operational_roots[{index}]")
        for index, value in enumerate(root_values)
    )
    if len(operational_roots) != len(set(operational_roots)):
        raise MultiresServiceError("operational_roots repeat")
    selector_values = config["service_selectors"]
    if not isinstance(selector_values, list) or not selector_values:
        raise MultiresServiceError("service_selectors must be a nonempty list")
    service_selectors = tuple(
        _absolute_file(value, f"service_selectors[{index}]")
        for index, value in enumerate(selector_values)
    )
    if len(service_selectors) != len(set(service_selectors)):
        raise MultiresServiceError("service_selectors repeat")

    python = _absolute_file(
        Path(sys.executable).resolve(), "Python executable", executable=True
    )
    retirement_tool = _absolute_file(
        ROOT / "tools" / "validate_multires_retirement.py",
        "runtime retirement validator",
    )
    retirement_command_values = [
        str(python), str(retirement_tool),
        "--manifest", str(retirement_manifest),
        "--expected-manifest-sha256", str(config["retirement_manifest_sha256"]),
        "--cold-start", str(retirement_cold_start),
    ]
    for root in operational_roots:
        retirement_command_values.extend(("--operational-root", str(root)))
    for selector in service_selectors:
        retirement_command_values.extend(("--service-selector", str(selector)))
    retirement_validation_command = tuple(retirement_command_values)
    completed = subprocess.run(
        retirement_validation_command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise MultiresServiceError(
            f"cold-start retirement validation failed: {detail}"
        )
    try:
        retirement_validation = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise MultiresServiceError(
            "cold-start retirement validator did not emit JSON"
        ) from error
    if (
        not isinstance(retirement_validation, dict)
        or retirement_validation.get("schema") != RETIREMENT_VALIDATION_SCHEMA
        or retirement_validation.get("status") != "pass"
        or retirement_validation.get("read_only") is not True
        or retirement_validation.get("manifest_sha256")
        != config["retirement_manifest_sha256"]
    ):
        raise MultiresServiceError("cold-start retirement validation report differs")
    if cold_document.get("runtime_manifest_sha256") != config["runtime_manifest_sha256"]:
        raise MultiresServiceError("cold-start/service runtime digests differ")
    if cold_document.get("atlas_catalog_sha256") != proof[
        "expected_atlas_catalog_sha256"
    ]:
        raise MultiresServiceError("cold-start/service Atlas catalog digests differ")
    one_run_config_path = _absolute_file(
        runtime_root / RUNTIME_CONFIG_NAME, "one-run runtime configuration"
    )
    one_run_config = _json(one_run_config_path, "one-run runtime configuration")
    if cold_document.get("optimizer") != one_run_config.get("optimizer"):
        raise MultiresServiceError("cold-start/one-run optimizer configurations differ")
    def require_cold_input(name: str, expected: object) -> None:
        record = cold_inputs.get(name)
        expected_path = Path(str(expected)).resolve()
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or Path(str(record["path"])).resolve() != expected_path
            or record["sha256"] != _sha256(expected_path)
        ):
            raise MultiresServiceError(
                f"cold-start {name} differs from the service proof input"
            )

    require_cold_input("checkpoint", proof["checkpoint"])
    require_cold_input("training_manifest", proof["training_manifest"])
    require_cold_input("runtime_evidence", proof["runtime_evidence"])
    require_cold_input("bundle_manifest", proof["bundle_manifest"])
    require_cold_input("atlas_catalog", proof["atlas_catalog"])
    training_runtime_record = config["training_runtime"]
    if (
        not isinstance(training_runtime_record, Mapping)
        or set(training_runtime_record) != {"path", "sha256"}
    ):
        raise MultiresServiceError("service training_runtime record is malformed")
    training_runtime_path = _absolute_file(
        training_runtime_record["path"], "primary training runtime configuration"
    )
    if (
        training_runtime_record["sha256"] != _sha256(training_runtime_path)
        or cold_inputs.get("training_runtime") != dict(training_runtime_record)
    ):
        raise MultiresServiceError(
            "service/cold-start primary training runtime binding differs"
        )
    cold_dyn = cold_inputs.get("dyn_snapshots")
    configured_dyn = one_run_config.get("dyn_snapshots")
    if (
        not isinstance(cold_dyn, list)
        or not isinstance(configured_dyn, list)
        or len(cold_dyn) != len(configured_dyn)
    ):
        raise MultiresServiceError("cold-start Dyn bindings differ from one-run")
    for index, expected in enumerate(configured_dyn):
        record = cold_dyn[index]
        expected_path = Path(str(expected)).resolve()
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or Path(str(record["path"])).resolve() != expected_path
            or record["sha256"] != _sha256(expected_path)
        ):
            raise MultiresServiceError(
                f"cold-start Dyn snapshot[{index}] differs from one-run"
            )

    one_run = one_run_preflight(
        _one_run_namespace(runtime_root, proof, evidence_dir)
    )
    if one_run.runtime_manifest_sha256 != config["runtime_manifest_sha256"]:
        raise MultiresServiceError("one-run runtime digest differs from service")
    if one_run.retirement_manifest_sha256 != config["retirement_manifest_sha256"]:
        raise MultiresServiceError("service retirement digest differs from one-run")
    if Path(one_run.runtime_config["retirement_manifest"]).resolve() != retirement_manifest:
        raise MultiresServiceError("service/one-run retirement manifest paths differ")

    primary = admit_primary_training(
        SimpleNamespace(one_run=one_run),
        config_path=training_runtime_path,
        config_sha256=str(training_runtime_record["sha256"]),
        cold_document=cold_document,
    )
    selected_checkpoint = cold_inputs.get("trainer_checkpoint")
    expected_checkpoint = {
        "path": str(primary.checkpoint), "sha256": _sha256(primary.checkpoint)
    }
    if selected_checkpoint != expected_checkpoint:
        raise MultiresServiceError(
            "cold-start trainer checkpoint differs from primary selection"
        )
    selected_season = cold_inputs.get("trainer_current_season")
    expected_season = None
    if primary.current_season_report is not None:
        expected_season = {
            "path": str(primary.current_season_report),
            "sha256": _sha256(primary.current_season_report),
        }
    if selected_season != expected_season:
        raise MultiresServiceError(
            "cold-start trainer season differs from primary selection"
        )

    proof_out = (evidence_dir / "deterministic_transitions.json").resolve()
    proof_tool = _absolute_file(
        ROOT / "tools" / "run_multires_500_transition_proof.py",
        "500-transition proof tool",
    )
    command = (
        str(python), str(proof_tool),
        "--mode", "production",
        "--seed", str(int(proof["seed"])),
        "--game_seed", str(int(proof["game_seed"])),
        "--divergence_game_seed", str(int(proof["divergence_game_seed"])),
        "--transition_count", "500",
        "--policy_version", str(int(proof["policy_version"])),
        "--map_name", str(proof["map_name"]),
        "--map_epoch", str(int(proof["map_epoch"])),
        "--timeout_seconds", str(float(proof["timeout_seconds"])),
        "--q2ded", str(one_run.q2ded),
        "--client_binary", str(one_run.client_binary),
        "--runtime_root", str(runtime_root),
        "--bundle_manifest", str(one_run.bundle_manifest),
        "--objectives", str(one_run.objectives),
        "--atlas_bin", str(one_run.atlas_bin),
        "--atlas_catalog", str(one_run.atlas_catalog.path),
        "--expected_atlas_catalog_sha256",
        str(one_run.atlas_catalog.atlas_catalog_sha256),
        "--checkpoint", str(Path(proof["checkpoint"]).resolve()),
        "--training_manifest", str(Path(proof["training_manifest"]).resolve()),
        "--runtime_evidence", str(Path(proof["runtime_evidence"]).resolve()),
        "--evidence_dir", str(evidence_dir),
        "--trainer_executable", str(python),
        "--trainer_arg=-m",
        "--trainer_arg=train.multires_one_run",
        "--out", str(proof_out),
    )

    tensorboard = config["tensorboard"]
    if not isinstance(tensorboard, Mapping) or set(tensorboard) != {
        "enabled", "executable", "port", "bind_all"
    }:
        raise MultiresServiceError("tensorboard service fields differ")
    if (
        type(tensorboard["enabled"]) is not bool
        or type(tensorboard["bind_all"]) is not bool
        or type(tensorboard["port"]) is not int
    ):
        raise MultiresServiceError("tensorboard service field types differ")
    tensorboard_command = None
    if tensorboard["enabled"] is True:
        executable = _absolute_file(
            tensorboard["executable"], "TensorBoard executable", executable=True
        )
        port = tensorboard["port"]
        if not 1 <= port <= 65535:
            raise MultiresServiceError("TensorBoard port is invalid")
        values = [
            str(executable), "--logdir", str(primary.tensorboard_root),
            "--port", str(port)
        ]
        if tensorboard["bind_all"] is True:
            values.append("--bind_all")
        tensorboard_command = tuple(values)
    elif tensorboard["enabled"] is not False:
        raise MultiresServiceError("tensorboard.enabled must be boolean")
    trainer_command = (
        str(python), "-u", "-m", PRIMARY_TRAINER_MODULE,
        "--runtime_root", str(runtime_root),
    )
    launch_config = (
        one_run.q2_root / str(one_run.runtime_config["game"])
        / "multires_one_run_primary-trainer.cfg"
    ).resolve()
    return ServiceAdmission(
        runtime_root=runtime_root,
        config=config,
        one_run=one_run,
        proof_command=command,
        evidence_dir=evidence_dir,
        proof_out=proof_out,
        log_path=log_path,
        primary=primary,
        trainer_command=trainer_command,
        current_run_root=primary.current_run_root,
        tensorboard_root=primary.tensorboard_root,
        current_season_report=(primary.season_report_root / "current.json"),
        child_inventory=(runtime_root / PRIMARY_CHILD_INVENTORY_NAME),
        terminal_evidence=(runtime_root / PRIMARY_TERMINAL_NAME),
        launch_config=launch_config,
        shutdown_grace_seconds=float(
            primary.config["execution"]["shutdown_grace_seconds"]
        ),
        tensorboard_command=tensorboard_command,
        retirement_validation_command=retirement_validation_command,
        retirement_validation=retirement_validation,
        integration_envelope=integration_envelope,
        integration_report=integration_report,
        integration_report_sha256=str(
            integration_verification["report_sha256"]
        ),
        integration_verification=integration_verification,
    )


def _state_path(runtime_root: Path) -> Path:
    return runtime_root / SERVICE_STATE_NAME


def _lease_path(runtime_root: Path) -> Path:
    return runtime_root / SERVICE_LEASE_NAME


def _proc_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return int(tail[19])
    except (OSError, IndexError, ValueError) as error:
        raise MultiresServiceError(f"cannot attest process {pid}") from error


def _proc_identity(pid: int) -> tuple[int, int, int, int]:
    """Return (ppid, process-group, session, start-ticks) from Linux /proc."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return int(tail[1]), int(tail[2]), int(tail[3]), int(tail[19])
    except (OSError, IndexError, ValueError) as error:
        raise MultiresServiceError(f"cannot attest process {pid}") from error


def _proc_state_and_start_ticks(pid: int) -> tuple[str, int]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return str(tail[0]), int(tail[19])
    except (OSError, IndexError, ValueError) as error:
        raise MultiresServiceError(f"cannot attest process {pid}") from error


def _process_record(role: str, process: subprocess.Popen) -> dict[str, Any]:
    _ppid, process_group, session, start_ticks = _proc_identity(process.pid)
    return {
        "role": role,
        "pid": int(process.pid),
        "start_ticks": start_ticks,
        "process_group": process_group,
        "session": session,
    }


def _record_alive(record: Mapping[str, Any]) -> bool:
    try:
        state, start_ticks = _proc_state_and_start_ticks(int(record["pid"]))
        return state != "Z" and start_ticks == int(record["start_ticks"])
    except MultiresServiceError:
        return False


def _current_process_record(role: str) -> dict[str, Any]:
    _ppid, pgrp, session, ticks = _proc_identity(os.getpid())
    return {
        "role": role, "pid": os.getpid(), "start_ticks": ticks,
        "process_group": pgrp, "session": session,
    }


def _write_private_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    if path.is_symlink():
        raise MultiresServiceError(f"private state symlink rejected: {path}")
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical(payload))
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _lease_payload(
    mode: str, owner: Mapping[str, Any], selector_token_sha256: str
) -> dict[str, Any]:
    if not _valid_sha256(selector_token_sha256):
        raise MultiresServiceError("service selector token digest is invalid")
    return {
        "schema": SERVICE_LEASE_SCHEMA, "mode": mode, "owner": dict(owner),
        "selector_token_sha256": selector_token_sha256,
    }


def _acquire_lease(runtime_root: Path, mode: str) -> str:
    if mode not in {"prove", "start"}:
        raise MultiresServiceError("runtime lease selector is not operational")
    path = _lease_path(runtime_root)
    owner = _current_process_record(mode)
    selector_token = secrets.token_hex(32)
    selector_token_sha256 = hashlib.sha256(
        selector_token.encode("ascii")
    ).hexdigest()
    try:
        _write_private_json(
            path, _lease_payload(mode, owner, selector_token_sha256),
            exclusive=True,
        )
    except FileExistsError as error:
        try:
            existing = _json(path, "service runtime lease")
            active = isinstance(existing.get("owner"), Mapping) and _record_alive(
                existing["owner"]
            )
        except Exception:
            active = False
        qualifier = "active" if active else "stale"
        raise MultiresServiceError(
            f"{qualifier} exclusive runtime lease exists; run stop before another selector"
        ) from error
    return selector_token


def _transfer_lease(
    runtime_root: Path, mode: str, owner: Mapping[str, Any], selector_token: str
) -> None:
    path = _lease_path(runtime_root)
    if not path.is_file() or path.is_symlink():
        raise MultiresServiceError("exclusive runtime lease disappeared")
    token_sha256 = hashlib.sha256(selector_token.encode("ascii")).hexdigest()
    existing = _json(path, "service runtime lease")
    if existing.get("selector_token_sha256") != token_sha256:
        raise MultiresServiceError("service selector token changed before transfer")
    _write_private_json(
        path, _lease_payload(mode, owner, token_sha256), exclusive=False
    )


def _release_lease(runtime_root: Path) -> None:
    path = _lease_path(runtime_root)
    if path.is_symlink():
        raise MultiresServiceError("service runtime lease symlink rejected")
    path.unlink(missing_ok=True)


def _validated_lease(runtime_root: Path) -> dict[str, Any] | None:
    path = _lease_path(runtime_root)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise MultiresServiceError("service runtime lease is not a regular file")
    lease = _json(path, "service runtime lease")
    if (
        set(lease) != {"schema", "mode", "owner", "selector_token_sha256"}
        or lease.get("schema") != SERVICE_LEASE_SCHEMA
        or lease.get("mode") not in {"prove", "start"}
        or not _valid_sha256(lease.get("selector_token_sha256"))
    ):
        raise MultiresServiceError("service runtime lease fields/schema differ")
    owner = lease.get("owner")
    if not isinstance(owner, Mapping) or set(owner) != {
        "role", "pid", "start_ticks", "process_group", "session"
    }:
        raise MultiresServiceError("service runtime lease owner is malformed")
    for field in ("pid", "start_ticks", "process_group", "session"):
        if type(owner[field]) is not int or owner[field] <= 1:
            raise MultiresServiceError("service runtime lease owner is invalid")
    return lease


def _lease_matches(runtime_root: Path, primary_record: Mapping[str, Any]) -> bool:
    path = _lease_path(runtime_root)
    if path.is_symlink() or not path.is_file():
        return False
    try:
        lease = _json(path, "service runtime lease")
    except MultiresServiceError:
        return False
    owner = lease.get("owner")
    return bool(
        set(lease) == {"schema", "mode", "owner", "selector_token_sha256"}
        and lease.get("schema") == SERVICE_LEASE_SCHEMA
        and lease.get("mode") == "start"
        and _valid_sha256(lease.get("selector_token_sha256"))
        and isinstance(owner, Mapping)
        and all(
            owner.get(field) == primary_record.get(field)
            for field in ("pid", "start_ticks", "process_group", "session")
        )
    )


def _reconcile_zero_update_attempt(runtime_root: Path) -> dict[str, Any]:
    """Remove only dead-attempt artifacts that cannot contain sealed progress."""
    attempt_path = runtime_root / PRIMARY_ATTEMPT_NAME
    if not attempt_path.exists():
        return {"status": "none", "removed": []}
    if attempt_path.is_symlink() or not attempt_path.is_file():
        raise MultiresServiceError("primary attempt record is not a regular file")
    if _state_path(runtime_root).exists():
        raise MultiresServiceError(
            "primary attempt has service state; stop must run before reconciliation"
        )
    attempt = _json(attempt_path, "primary attempt record")
    unsigned = dict(attempt)
    declared = unsigned.pop("evidence_sha256", None)
    if (
        set(attempt) != PRIMARY_ATTEMPT_FIELDS
        or attempt.get("schema") != PRIMARY_ATTEMPT_SCHEMA
        or not _valid_sha256(declared)
        or declared != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or not _valid_sha256(attempt.get("runtime_manifest_sha256"))
        or not _valid_sha256(attempt.get("atlas_catalog_sha256"))
        or not _valid_sha256(attempt.get("training_runtime_sha256"))
        or not _valid_sha256(attempt.get("selected_checkpoint_sha256"))
        or not _valid_sha256(attempt.get("selector_token_sha256"))
    ):
        raise MultiresServiceError("primary attempt evidence seal/fields differ")
    owner = attempt.get("owner")
    if not isinstance(owner, Mapping) or set(owner) != {
        "role", "pid", "start_ticks", "process_group", "session"
    } or owner.get("role") != PRIMARY_TRAINER_MODULE:
        raise MultiresServiceError("primary attempt owner is malformed")
    if _record_alive(owner):
        raise MultiresServiceError("primary attempt owner is still live")

    service_path = _absolute_file(
        runtime_root / SERVICE_CONFIG_NAME, "service configuration"
    )
    service_config = _json(service_path, "service configuration")
    training_record = service_config.get("training_runtime")
    cold_start_value = service_config.get("retirement_cold_start")
    if (
        service_config.get("schema") != SERVICE_CONFIG_SCHEMA
        or service_config.get("runtime_manifest_sha256")
        != attempt["runtime_manifest_sha256"]
        or not isinstance(training_record, Mapping)
        or set(training_record) != {"path", "sha256"}
        or training_record.get("sha256") != attempt["training_runtime_sha256"]
        or not isinstance(cold_start_value, str)
    ):
        raise MultiresServiceError("primary attempt/service runtime binding differs")
    training_path = _absolute_file(
        training_record["path"], "primary training runtime configuration"
    )
    if _sha256(training_path) != attempt["training_runtime_sha256"]:
        raise MultiresServiceError("primary attempt training runtime bytes differ")
    training_config = _json(
        training_path, "primary training runtime configuration"
    )
    checkpoint_selector = training_config.get("checkpoint")
    if training_config.get("atlas_catalog_sha256") != attempt[
        "atlas_catalog_sha256"
    ]:
        raise MultiresServiceError(
            "primary attempt Atlas catalog differs from training runtime"
        )
    if (
        training_config.get("schema") != "q2-multires-primary-training-runtime-v1"
        or training_config.get("runtime_manifest_sha256")
        != attempt["runtime_manifest_sha256"]
        or not isinstance(checkpoint_selector, Mapping)
        or checkpoint_selector.get("mode") != attempt["checkpoint_mode"]
        or Path(str(checkpoint_selector.get("path"))).resolve()
        != Path(str(attempt["selected_checkpoint"])).resolve()
        or checkpoint_selector.get("sha256")
        != attempt["selected_checkpoint_sha256"]
    ):
        raise MultiresServiceError("primary attempt/checkpoint selector differs")
    cold_start_path = _absolute_file(
        cold_start_value, "retirement cold-start declaration"
    )
    cold_start = _json(cold_start_path, "retirement cold-start declaration")
    cold_lineage = cold_start.get("lineage")
    if cold_start.get("atlas_catalog_sha256") != attempt[
        "atlas_catalog_sha256"
    ]:
        raise MultiresServiceError(
            "primary attempt Atlas catalog differs from cold start"
        )
    if (
        cold_start.get("runtime_manifest_sha256")
        != attempt["runtime_manifest_sha256"]
        or not isinstance(cold_lineage, Mapping)
    ):
        raise MultiresServiceError("primary attempt/cold-start lineage differs")

    current = Path(str(attempt["current_run_root"]))
    checkpoint_root = Path(str(attempt["checkpoint_root"]))
    tensorboard_root = Path(str(attempt["tensorboard_root"]))
    rollout_root = Path(str(attempt["rollout_root"]))
    update_root = Path(str(attempt["update_root"]))
    season = Path(str(attempt["current_season_report"]))
    selected = Path(str(attempt["selected_checkpoint"]))
    if (
        not current.is_absolute() or current.is_symlink() or not current.is_dir()
        or checkpoint_root != current / "checkpoints"
        or tensorboard_root != current / "tensorboard"
        or rollout_root != current / "evidence/rollouts"
        or update_root != current / "evidence/updates"
        or season != current / "season/current.json"
        or selected.parent != checkpoint_root
        or selected.is_symlink() or not selected.is_file()
        or _sha256(selected) != attempt["selected_checkpoint_sha256"]
        or Path(str(cold_lineage.get("current_run_root"))).resolve()
        != current.resolve()
        or Path(str(cold_lineage.get("checkpoint_root"))).resolve()
        != checkpoint_root.resolve()
        or Path(str(cold_lineage.get("tensorboard_root"))).resolve()
        != tensorboard_root.resolve()
        or Path(str(cold_lineage.get("rollout_root"))).resolve()
        != rollout_root.resolve()
        or Path(str(cold_lineage.get("update_root"))).resolve()
        != update_root.resolve()
        or Path(str(cold_lineage.get("season_report_root"))).resolve()
        != season.parent.resolve()
    ):
        raise MultiresServiceError("primary attempt run-local roots differ")
    for root, label in (
        (checkpoint_root, "checkpoint"), (tensorboard_root, "TensorBoard"),
        (rollout_root, "rollout"), (update_root, "update"),
    ):
        if root.is_symlink() or not root.is_dir():
            raise MultiresServiceError(f"primary attempt {label} root differs")

    if season.exists():
        if not _current_season_health(
            season,
            runtime_manifest_sha256=str(attempt["runtime_manifest_sha256"]),
            atlas_catalog_sha256=str(attempt["atlas_catalog_sha256"]),
        ):
            raise MultiresServiceError(
                "primary attempt has invalid/nonterminal season evidence"
            )
        attempt_path.unlink()
        return {"status": "sealed-progress-preserved", "removed": [str(attempt_path)]}
    if attempt.get("checkpoint_mode") != "fresh-step-zero":
        raise MultiresServiceError(
            "automatic zero-update reconciliation is fresh-attempt only"
        )
    if any(rollout_root.iterdir()) or any(update_root.iterdir()):
        raise MultiresServiceError(
            "zero-update attempt contains immutable rollout/update evidence"
        )

    owner_pid = int(owner["pid"])
    pending_pattern = re.compile(
        rf"\.{{1,2}}checkpoint-[0-9]{{12}}-{owner_pid}\.pending"
        rf"(?:\.[A-Za-z0-9_-]+\.tmp)?\Z"
    )
    pending: list[Path] = []
    for child in checkpoint_root.iterdir():
        if child == selected:
            continue
        if child.is_symlink() or not child.is_file() or not pending_pattern.fullmatch(
            child.name
        ):
            raise MultiresServiceError(
                "zero-update checkpoint root contains non-attempt content"
            )
        pending.append(child)

    suffix = attempt.get("tensorboard_filename_suffix")
    expected_suffix = f".attempt-{attempt['selector_token_sha256'][:16]}"
    if suffix != expected_suffix:
        raise MultiresServiceError("primary attempt TensorBoard suffix differs")
    tensorboard_files: list[Path] = []
    for child in tensorboard_root.iterdir():
        if (
            child.is_symlink() or not child.is_file()
            or not child.name.startswith("events.out.tfevents.")
            or f".{owner_pid}." not in child.name
            or not child.name.endswith(expected_suffix)
        ):
            raise MultiresServiceError(
                "zero-update TensorBoard root contains non-attempt content"
            )
        tensorboard_files.append(child)

    removed: list[str] = []
    for child in (*pending, *tensorboard_files):
        child.unlink()
        removed.append(str(child))
    attempt_path.unlink()
    removed.append(str(attempt_path))
    return {"status": "zero-update-reconciled", "removed": removed}


def _child_inventory(
    path: Path, *, runtime_manifest_sha256: str,
    primary_record: Mapping[str, Any], require_complete: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise MultiresServiceError("primary child inventory is not a regular file")
    inventory = _json(path, "primary child inventory")
    if set(inventory) != {
        "schema", "runtime_manifest_sha256", "owner", "launch_config",
        "expected_client_count", "complete", "processes",
    } or inventory.get("schema") != PRIMARY_CHILD_INVENTORY_SCHEMA:
        raise MultiresServiceError("primary child inventory fields/schema differ")
    if inventory.get("runtime_manifest_sha256") != runtime_manifest_sha256:
        raise MultiresServiceError("primary child inventory runtime differs")
    owner = inventory.get("owner")
    identity_fields = ("pid", "start_ticks", "process_group", "session")
    if not isinstance(owner, Mapping) or any(
        owner.get(field) != primary_record.get(field) for field in identity_fields
    ):
        raise MultiresServiceError("primary child inventory owner differs")
    count = inventory.get("expected_client_count")
    complete = inventory.get("complete")
    records = inventory.get("processes")
    if type(count) is not int or count != 4 or type(complete) is not bool:
        raise MultiresServiceError("primary child inventory cardinality differs")
    if not isinstance(records, list) or not records:
        raise MultiresServiceError("primary child inventory processes are missing")
    expected_roles = {"q2ded"} | {f"network-client-{index:02d}" for index in range(count)}
    roles: set[str] = set()
    identities: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "role", "pid", "start_ticks", "process_group", "session"
        }:
            raise MultiresServiceError(f"primary child process[{index}] is malformed")
        role = record["role"]
        if role not in expected_roles or role in roles:
            raise MultiresServiceError("primary child inventory roles differ")
        roles.add(role)
        for field in ("pid", "start_ticks", "process_group", "session"):
            if type(record[field]) is not int or record[field] <= 1:
                raise MultiresServiceError("primary child process identity is invalid")
        identity = (int(record["pid"]), int(record["start_ticks"]))
        if identity in identities:
            raise MultiresServiceError("primary child process identities repeat")
        identities.add(identity)
    if complete and roles != expected_roles:
        raise MultiresServiceError("complete primary child inventory is not q2ded+4")
    if require_complete and (complete is not True or roles != expected_roles):
        return None
    if require_complete and any(not _record_alive(record) for record in records):
        return None
    return inventory


def _owned_process_tree(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Snapshot the exact /proc descendant tree under one attested service PID."""
    root_pid = int(root["pid"])
    identities: dict[int, tuple[int, int, int, int]] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as error:
        raise MultiresServiceError("cannot enumerate owned service descendants") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            identities[pid] = _proc_identity(pid)
        except MultiresServiceError:
            continue
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgrp, _session, _ticks) in identities.items():
        children.setdefault(ppid, []).append(pid)
    declared_group = int(root["process_group"])
    declared_session = int(root["session"])
    selected = {
        pid for pid, (_ppid, pgrp, session, _ticks) in identities.items()
        if session == declared_session and pgrp == declared_group
    }
    pending = [root_pid]
    traversed = {root_pid}
    while pending:
        parent = pending.pop()
        for pid in children.get(parent, []):
            if pid in traversed:
                continue
            traversed.add(pid)
            selected.add(pid)
            pending.append(pid)
    records: list[dict[str, Any]] = []
    # start_new_session=True makes the supervisor record both leader and
    # durable ownership namespace.  Session members remain discoverable even
    # if that leader crashes before stop/status runs.
    for pid in selected:
        _ppid, pgrp, session, ticks = identities[pid]
        records.append({
            "role": dict(root).get("role") if pid == root_pid else "owned-descendant",
            "pid": pid, "start_ticks": ticks,
            "process_group": pgrp, "session": session,
        })
    return records


def _signal_owned_tree(records: Sequence[Mapping[str, Any]], signum: int) -> None:
    """Signal only attested PIDs/groups whose leader is in the owned tree."""
    owned_pids = {int(record["pid"]) for record in records if _record_alive(record)}
    group_leaders = {
        int(record.get("process_group", -1)) for record in records
        if int(record.get("process_group", -1)) > 1
        and any(
            int(member["pid"]) in owned_pids
            and int(member.get("process_group", -1))
            == int(record.get("process_group", -1))
            for member in records
        )
    }
    for pgrp in sorted(group_leaders, reverse=True):
        try:
            os.killpg(pgrp, signum)
        except ProcessLookupError:
            pass
    grouped = {
        int(record["pid"])
        for record in records
        if int(record.get("process_group", -1)) in group_leaders
    }
    for pid in sorted(owned_pids - grouped, reverse=True):
        record = next(item for item in records if int(item["pid"]) == pid)
        if not _record_alive(record):
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _terminate_owned_tree(
    root: Mapping[str, Any], *, timeout: float,
    additional_records: Sequence[Mapping[str, Any]] = (),
) -> list[int]:
    root_pid = int(root["pid"])
    if int(root.get("process_group", -1)) != root_pid or int(
        root.get("session", -1)
    ) != root_pid:
        raise MultiresServiceError(
            f"owned service root {root_pid} is not its declared process/session leader"
        )
    if _record_alive(root):
        _ppid, pgrp, session, ticks = _proc_identity(root_pid)
        if (
            pgrp != root_pid or session != root_pid
            or ticks != int(root["start_ticks"])
        ):
            raise MultiresServiceError(
                f"owned service root {root_pid} live identity differs"
            )
    records_by_identity = {
        (int(item["pid"]), int(item["start_ticks"])): dict(item)
        for item in (*_owned_process_tree(root), *additional_records)
        if _record_alive(item)
    }
    records = list(records_by_identity.values())
    if not records:
        return []
    pids = [int(record["pid"]) for record in records]
    cooperative_primary = root.get("role") == PRIMARY_ROLE and _record_alive(root)
    if cooperative_primary:
        try:
            os.kill(root_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        _signal_owned_tree(records, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    group_term_sent = not cooperative_primary
    while time.monotonic() < deadline and any(_record_alive(item) for item in records):
        if cooperative_primary and not _record_alive(root) and not group_term_sent:
            _signal_owned_tree(records, signal.SIGTERM)
            group_term_sent = True
        time.sleep(0.05)
    survivors = [item for item in records if _record_alive(item)]
    if survivors:
        _signal_owned_tree(survivors, signal.SIGKILL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(
            _record_alive(item) for item in survivors
        ):
            time.sleep(0.05)
    survivors = [int(item["pid"]) for item in records if _record_alive(item)]
    if survivors:
        raise MultiresServiceError(
            f"owned service process tree survived teardown: {survivors}"
        )
    return pids


def _write_state(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    runtime_manifest_sha256: str,
    atlas_catalog_sha256: str,
    current_run_root: Path,
    tensorboard_root: Path,
    current_season_report: Path,
    child_inventory: Path,
    terminal_evidence: Path,
    launch_config: Path,
    shutdown_grace_seconds: float,
) -> None:
    payload = {
        "schema": SERVICE_STATE_SCHEMA,
        "service_role": PRIMARY_ROLE,
        "training_updates_enabled": True,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "atlas_catalog_sha256": atlas_catalog_sha256,
        "current_run_root": str(current_run_root),
        "tensorboard_root": str(tensorboard_root),
        "current_season_report": str(current_season_report),
        "child_inventory": str(child_inventory),
        "terminal_evidence": str(terminal_evidence),
        "launch_config": str(launch_config),
        "shutdown_grace_seconds": shutdown_grace_seconds,
        "processes": list(records),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_state(path: Path) -> dict[str, Any]:
    state_path = _absolute_file(path, "service state")
    state = _json(state_path, "service state")
    if set(state) != {
        "schema", "service_role", "training_updates_enabled",
        "runtime_manifest_sha256", "atlas_catalog_sha256", "current_run_root",
        "tensorboard_root",
        "current_season_report", "child_inventory", "terminal_evidence",
        "launch_config", "shutdown_grace_seconds", "processes",
    } or state.get("schema") != SERVICE_STATE_SCHEMA:
        raise MultiresServiceError("service state fields/schema differ")
    if (
        state["service_role"] != PRIMARY_ROLE
        or state["training_updates_enabled"] is not True
        or not _valid_sha256(state["runtime_manifest_sha256"])
        or not _valid_sha256(state["atlas_catalog_sha256"])
    ):
        raise MultiresServiceError("service state primary-trainer identity differs")
    current = Path(str(state["current_run_root"]))
    tensorboard = Path(str(state["tensorboard_root"]))
    season = Path(str(state["current_season_report"]))
    child_inventory = Path(str(state["child_inventory"]))
    terminal_evidence = Path(str(state["terminal_evidence"]))
    launch_config = Path(str(state["launch_config"]))
    if (
        not current.is_absolute()
        or tensorboard != current / "tensorboard"
        or season != current / "season" / "current.json"
        or not child_inventory.is_absolute()
        or child_inventory.parent != state_path.parent
        or child_inventory.name != PRIMARY_CHILD_INVENTORY_NAME
        or not terminal_evidence.is_absolute()
        or terminal_evidence.parent != state_path.parent
        or terminal_evidence.name != PRIMARY_TERMINAL_NAME
        or not launch_config.is_absolute()
        or launch_config.name != "multires_one_run_primary-trainer.cfg"
    ):
        raise MultiresServiceError("service state run-local roots differ")
    grace = state["shutdown_grace_seconds"]
    if (
        isinstance(grace, bool) or not isinstance(grace, (int, float))
        or not math.isfinite(float(grace)) or not 30.0 <= float(grace) <= 3600.0
    ):
        raise MultiresServiceError("service state shutdown grace is invalid")
    records = state["processes"]
    allowed_roles = {PRIMARY_ROLE, "tensorboard-current-run"}
    if not isinstance(records, list) or not records:
        raise MultiresServiceError("service state process inventory is invalid")
    roles: set[str] = set()
    pids: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "role", "pid", "start_ticks", "process_group", "session"
        }:
            raise MultiresServiceError(f"service state process[{index}] is malformed")
        role = record["role"]
        pid = record["pid"]
        ticks = record["start_ticks"]
        pgrp = record["process_group"]
        session = record["session"]
        if (
            role not in allowed_roles
            or role in roles
            or type(pid) is not int
            or pid <= 1
            or pid in pids
            or type(ticks) is not int
            or ticks <= 0
            or type(pgrp) is not int or pgrp != pid
            or type(session) is not int or session != pid
        ):
            raise MultiresServiceError(f"service state process[{index}] is invalid")
        roles.add(role)
        pids.add(pid)
    if PRIMARY_ROLE not in roles:
        raise MultiresServiceError("service state lacks the primary trainer")
    return state


def _revalidate_service_integration(admission: ServiceAdmission) -> None:
    """Close the preflight-to-launch mutation window before process creation."""
    current_config = _json(
        admission.runtime_root / SERVICE_CONFIG_NAME, "service configuration"
    )
    if current_config != admission.config:
        raise MultiresServiceError(
            "service configuration changed after integration preflight"
        )
    cold_start = _json(
        _absolute_file(
            current_config["retirement_cold_start"],
            "retirement cold-start declaration",
        ),
        "retirement cold-start declaration",
    )
    cold_inputs = cold_start.get("inputs")
    if not isinstance(cold_inputs, Mapping):
        raise MultiresServiceError("cold-start input bindings are missing")
    envelope, report, verification = _verify_integration_admission(
        runtime_root=admission.runtime_root,
        config=current_config,
        cold_inputs=cold_inputs,
        retirement_manifest=_absolute_file(
            current_config["retirement_manifest"], "retirement manifest"
        ),
    )
    if (
        envelope != admission.integration_envelope
        or report != admission.integration_report
        or verification.get("report_sha256")
        != admission.integration_report_sha256
    ):
        raise MultiresServiceError(
            "integration admission changed after service preflight"
        )


def _wait_for_primary_admission(
    admission: ServiceAdmission,
    trainer: subprocess.Popen,
    trainer_record: Mapping[str, Any],
    selector_token: str,
    *,
    timeout_seconds: float = PRIMARY_ADMISSION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Require the child to publish its sealed post-preflight attempt first."""
    attempt_path = admission.runtime_root / PRIMARY_ATTEMPT_NAME
    deadline = time.monotonic() + timeout_seconds
    token_sha256 = hashlib.sha256(selector_token.encode("ascii")).hexdigest()
    while True:
        returncode = trainer.poll()
        if attempt_path.exists() or attempt_path.is_symlink():
            if attempt_path.is_symlink() or not attempt_path.is_file():
                raise MultiresServiceError(
                    "primary admission attempt is not a regular file"
                )
            attempt = _json(attempt_path, "primary admission attempt")
            unsigned = dict(attempt)
            seal = unsigned.pop("evidence_sha256", None)
            owner = attempt.get("owner")
            owner_matches = bool(
                isinstance(owner, Mapping)
                and set(owner) == {
                    "role", "pid", "start_ticks", "process_group", "session"
                }
                and owner.get("role") == PRIMARY_TRAINER_MODULE
                and all(
                    owner.get(field) == trainer_record.get(field)
                    for field in (
                        "pid", "start_ticks", "process_group", "session"
                    )
                )
            )
            if (
                set(attempt) != PRIMARY_ATTEMPT_FIELDS
                or attempt.get("schema") != PRIMARY_ATTEMPT_SCHEMA
                or not _valid_sha256(seal)
                or seal != hashlib.sha256(_canonical(unsigned)).hexdigest()
                or attempt.get("runtime_manifest_sha256")
                != admission.one_run.runtime_manifest_sha256
                or attempt.get("atlas_catalog_sha256")
                != admission.one_run.atlas_catalog.atlas_catalog_sha256
                or attempt.get("training_runtime_sha256")
                != admission.primary.config_sha256
                or attempt.get("checkpoint_mode")
                != admission.primary.checkpoint_mode
                or Path(str(attempt.get("selected_checkpoint"))).resolve()
                != admission.primary.checkpoint
                or attempt.get("selected_checkpoint_sha256")
                != _sha256(admission.primary.checkpoint)
                or Path(str(attempt.get("current_run_root"))).resolve()
                != admission.current_run_root
                or Path(str(attempt.get("checkpoint_root"))).resolve()
                != admission.primary.checkpoint_root
                or Path(str(attempt.get("tensorboard_root"))).resolve()
                != admission.tensorboard_root
                or Path(str(attempt.get("rollout_root"))).resolve()
                != admission.primary.rollout_root
                or Path(str(attempt.get("update_root"))).resolve()
                != admission.primary.update_root
                or Path(str(attempt.get("current_season_report"))).resolve()
                != admission.current_season_report
                or attempt.get("selector_token_sha256") != token_sha256
                or attempt.get("tensorboard_filename_suffix")
                != f".attempt-{token_sha256[:16]}"
                or not owner_matches
            ):
                raise MultiresServiceError(
                    "primary child admission attempt fields/bindings differ"
                )
            if returncode is not None or not _record_alive(trainer_record):
                raise MultiresServiceError(
                    "primary trainer exited during child admission"
                )
            return attempt
        if returncode is not None:
            raise MultiresServiceError(
                f"primary trainer exited before child admission (status {returncode})"
            )
        if time.monotonic() >= deadline:
            raise MultiresServiceError(
                "primary trainer did not publish child admission before timeout"
            )
        time.sleep(0.01)


def start_service(
    admission: ServiceAdmission, *, lease_acquired: bool = False,
    selector_token: str | None = None,
) -> dict[str, Any]:
    # This is deliberately before lease acquisition, state reconciliation,
    # log creation, and Popen.  CLI preflight performs the same verification
    # before its selector lease, and this second pass closes the launch window.
    _revalidate_service_integration(admission)
    if not lease_acquired:
        selector_token = _acquire_lease(admission.runtime_root, "start")
    if not isinstance(selector_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", selector_token
    ):
        if lease_acquired:
            _release_lease(admission.runtime_root)
        raise MultiresServiceError("start selector token is missing or malformed")
    state_path = _state_path(admission.runtime_root)
    if state_path.is_symlink():
        _release_lease(admission.runtime_root)
        raise MultiresServiceError("service state symlinks are rejected")
    if state_path.exists():
        _release_lease(admission.runtime_root)
        raise MultiresServiceError(
            "service state already exists; stop must reconcile it before start"
        )
    for stale, label in (
        (admission.terminal_evidence, "terminal evidence"),
        (admission.child_inventory, "child inventory"),
    ):
        if stale.is_symlink():
            _release_lease(admission.runtime_root)
            raise MultiresServiceError(f"stale {label} symlinks are rejected")
        try:
            stale.unlink(missing_ok=True)
        except Exception:
            _release_lease(admission.runtime_root)
            raise
    try:
        log = admission.log_path.open("ab", buffering=0)
    except Exception:
        _release_lease(admission.runtime_root)
        raise
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        trainer_environment = dict(os.environ)
        trainer_environment[PRIMARY_SELECTOR_TOKEN_ENV] = selector_token
        trainer = subprocess.Popen(
            admission.trainer_command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=trainer_environment,
        )
        processes.append((PRIMARY_ROLE, trainer))
        trainer_record = _process_record(PRIMARY_ROLE, trainer)
        _transfer_lease(
            admission.runtime_root, "start", trainer_record, selector_token
        )
        primary_attempt = _wait_for_primary_admission(
            admission, trainer, trainer_record, selector_token
        )
        # The child has independently completed service_preflight and sealed
        # its exact selection. Recheck the final envelope/source immediately
        # before allowing the observability process to exist.
        _revalidate_service_integration(admission)
        if trainer.poll() is not None or not _record_alive(trainer_record):
            raise MultiresServiceError(
                "primary trainer exited before TensorBoard admission"
            )
        if admission.tensorboard_command is not None:
            tensorboard = subprocess.Popen(
                admission.tensorboard_command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(("tensorboard-current-run", tensorboard))
        records = [trainer_record] + [
            _process_record(role, process) for role, process in processes[1:]
        ]
        _write_state(
            state_path,
            records,
            runtime_manifest_sha256=admission.one_run.runtime_manifest_sha256,
            atlas_catalog_sha256=(
                admission.one_run.atlas_catalog.atlas_catalog_sha256
            ),
            current_run_root=admission.current_run_root,
            tensorboard_root=admission.tensorboard_root,
            current_season_report=admission.current_season_report,
            child_inventory=admission.child_inventory,
            terminal_evidence=admission.terminal_evidence,
            launch_config=admission.launch_config,
            shutdown_grace_seconds=admission.shutdown_grace_seconds,
        )
        return {
            "started": True,
            "service_mode": SERVICE_MODE,
            "service_role": PRIMARY_ROLE,
            "training_updates_enabled": True,
            "health": "starting",
            "healthy": False,
            "current_season_evidence": False,
            "primary_admission_evidence_sha256": primary_attempt[
                "evidence_sha256"
            ],
            "processes": records,
        }
    except Exception:
        for role, process in processes:
            if process.poll() is None:
                _terminate_owned_tree(
                    _process_record(role, process),
                    timeout=admission.shutdown_grace_seconds,
                )
        _release_lease(admission.runtime_root)
        raise
    finally:
        log.close()


def stop_service(runtime_root: Path) -> dict[str, Any]:
    runtime_root = _absolute_directory(runtime_root, "runtime_root")
    state_path = _state_path(runtime_root)
    if state_path.is_symlink():
        raise MultiresServiceError("service state symlinks are rejected")
    if not state_path.exists():
        lease = _validated_lease(runtime_root)
        if lease is not None and _record_alive(lease["owner"]):
            raise MultiresServiceError(
                f"refusing to unlink live {lease['mode']} selector lease"
            )
        # Explicit stop may reconcile only a dead selector lease.
        _release_lease(runtime_root)
        return {
            "stopped": True, "service_mode": SERVICE_MODE,
            "service_role": PRIMARY_ROLE, "training_updates_enabled": True,
            "terminated": [], "already_stopped": True,
        }
    state = _validated_state(state_path)
    primary_record = next(
        record for record in state["processes"] if record["role"] == PRIMARY_ROLE
    )
    lease = _validated_lease(runtime_root)
    if (
        lease is not None and _record_alive(lease["owner"])
        and not (
            lease["mode"] == "start"
            and all(
                lease["owner"].get(field) == primary_record.get(field)
                for field in ("pid", "start_ticks", "process_group", "session")
            )
        )
    ):
        raise MultiresServiceError("live selector lease does not own service state")
    errors: list[str] = []
    try:
        inventory = _child_inventory(
            Path(str(state["child_inventory"])),
            runtime_manifest_sha256=str(state["runtime_manifest_sha256"]),
            primary_record=primary_record,
            require_complete=False,
        )
    except Exception as error:
        errors.append(f"child_inventory={error}")
        inventory = None
    additional = [] if inventory is None else list(inventory["processes"])
    terminated = []
    for record in state.get("processes", []):
        try:
            terminated.extend(_terminate_owned_tree(
                record,
                timeout=float(state["shutdown_grace_seconds"]),
                additional_records=(additional if record["role"] == PRIMARY_ROLE else ()),
            ))
        except Exception as error:
            errors.append(f"terminate_{record['role']}={error}")
    launch_config = Path(str(state["launch_config"]))
    try:
        if launch_config.is_symlink():
            raise MultiresServiceError("primary launch config symlink rejected")
        launch_config.unlink(missing_ok=True)
    except Exception as error:
        errors.append(f"launch_config={error}")
    try:
        child_inventory = Path(str(state["child_inventory"]))
        if child_inventory.is_symlink():
            raise MultiresServiceError("primary child inventory symlink rejected")
        child_inventory.unlink(missing_ok=True)
    except Exception as error:
        errors.append(f"child_inventory_cleanup={error}")
    if errors:
        raise MultiresServiceError(
            "service teardown incomplete: " + "; ".join(errors)
        )
    state_path.unlink(missing_ok=True)
    _release_lease(runtime_root)
    return {
        "stopped": True, "service_mode": SERVICE_MODE,
        "service_role": PRIMARY_ROLE, "training_updates_enabled": True,
        "terminated": terminated, "already_stopped": False,
    }


def service_status(runtime_root: Path) -> dict[str, Any]:
    runtime_root = _absolute_directory(runtime_root, "runtime_root")
    path = _state_path(runtime_root)
    if path.is_symlink():
        raise MultiresServiceError("service state symlinks are rejected")
    if not path.exists():
        return {
            "running": False, "service_mode": SERVICE_MODE,
            "service_role": PRIMARY_ROLE, "training_updates_enabled": True,
            "health": "stopped", "healthy": False,
            "current_season_evidence": False, "processes": [],
        }
    state = _validated_state(path)
    processes = [
        {**record, "alive": _record_alive(record)}
        for record in state.get("processes", [])
    ]
    trainer_alive = any(
        item["role"] == PRIMARY_ROLE and item["alive"] for item in processes
    )
    primary_record = next(
        record for record in state["processes"] if record["role"] == PRIMARY_ROLE
    )
    lease_held = _lease_matches(runtime_root, primary_record)
    try:
        inventory = _child_inventory(
            Path(str(state["child_inventory"])),
            runtime_manifest_sha256=str(state["runtime_manifest_sha256"]),
            primary_record=primary_record,
            require_complete=True,
        )
    except MultiresServiceError:
        inventory = None
    inventory_complete = inventory is not None
    season_evidence = _current_season_health(
        Path(str(state["current_season_report"])),
        runtime_manifest_sha256=str(state["runtime_manifest_sha256"]),
        atlas_catalog_sha256=str(state["atlas_catalog_sha256"]),
    )
    terminal_evidence = _terminal_health(
        Path(str(state["terminal_evidence"])), state=state
    )
    health = (
        "training-active" if (
            trainer_alive and season_evidence and inventory_complete and lease_held
        )
        else "starting" if trainer_alive and lease_held
        else "lease-lost" if trainer_alive
        else "completed" if season_evidence and terminal_evidence
        else "crashed"
    )
    return {
        "running": trainer_alive,
        "service_mode": SERVICE_MODE,
        "service_role": PRIMARY_ROLE,
        "training_updates_enabled": True,
        "health": health,
        "healthy": health == "training-active",
        "current_season_evidence": season_evidence,
        "child_inventory_complete": inventory_complete,
        "terminal_completion_evidence": terminal_evidence,
        "exclusive_runtime_lease": lease_held,
        "processes": processes,
    }


def _current_season_health(
    path: Path, *, runtime_manifest_sha256: str, atlas_catalog_sha256: str
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        report = _json(path, "current season report")
    except MultiresServiceError:
        return False
    counters = report.get("counters")
    declared_evidence = report.get("evidence_sha256")
    unsigned = dict(report)
    unsigned.pop("evidence_sha256", None)
    sealed = hashlib.sha256(_canonical(unsigned)).hexdigest()
    run_root = path.parent.parent
    try:
        checkpoint_candidate = run_root / str(report["last_checkpoint"])
        if checkpoint_candidate.is_symlink():
            return False
        checkpoint = checkpoint_candidate.resolve()
    except (KeyError, TypeError, OSError):
        return False
    checkpoint_manifest = report.get("checkpoint_manifest")
    return bool(
        report.get("schema") == "q2-multires-current-season-v1"
        and report.get("health") == "training-active"
        and report.get("runtime_manifest_sha256") == runtime_manifest_sha256
        and report.get("atlas_catalog_sha256") == atlas_catalog_sha256
        and declared_evidence == sealed
        and isinstance(counters, Mapping)
        and type(counters.get("policy_updates")) is int
        and counters["policy_updates"] >= 1
        and type(counters.get("optimizer_steps")) is int
        and counters["optimizer_steps"] >= counters["policy_updates"]
        and counters.get("next_policy_version") == counters["policy_updates"]
        and type(counters.get("accepted_transitions")) is int
        and counters["accepted_transitions"] >= 1
        and isinstance(checkpoint_manifest, Mapping)
        and checkpoint_manifest.get("training_step")
        == counters["accepted_transitions"]
        and checkpoint_manifest.get("runtime_manifest_sha256")
        == runtime_manifest_sha256
        and checkpoint_manifest.get("atlas_catalog_sha256")
        == atlas_catalog_sha256
        and checkpoint.is_file()
        and checkpoint.parent == run_root / "checkpoints"
        and report.get("checkpoint_sha256") == _sha256(checkpoint)
    )


def _terminal_health(path: Path, *, state: Mapping[str, Any]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        terminal = _json(path, "primary terminal evidence")
        if set(terminal) != PRIMARY_TERMINAL_FIELDS:
            return False
        unsigned = dict(terminal)
        declared = unsigned.pop("evidence_sha256")
        if declared != hashlib.sha256(_canonical(unsigned)).hexdigest():
            return False
        season = Path(str(terminal["current_season_report"])).resolve()
        checkpoint = Path(str(terminal["checkpoint"])).resolve()
        current_root = Path(str(state["current_run_root"]))
        if (
            terminal["schema"] != PRIMARY_TERMINAL_SCHEMA
            or terminal["status"] != "completed"
            or terminal["runtime_manifest_sha256"]
            != state["runtime_manifest_sha256"]
            or terminal["atlas_catalog_sha256"]
            != state["atlas_catalog_sha256"]
            or season != Path(str(state["current_season_report"])).resolve()
            or season.is_symlink() or not season.is_file()
            or terminal["current_season_sha256"] != _sha256(season)
            or checkpoint.parent != current_root / "checkpoints"
            or checkpoint.is_symlink() or not checkpoint.is_file()
            or terminal["checkpoint_sha256"] != _sha256(checkpoint)
            or type(terminal["accepted_transitions"]) is not int
            or terminal["accepted_transitions"] < 1
            or type(terminal["policy_updates"]) is not int
            or terminal["policy_updates"] < 1
            or type(terminal["optimizer_steps"]) is not int
            or terminal["optimizer_steps"] < terminal["policy_updates"]
            or terminal["next_policy_version"] != terminal["policy_updates"]
        ):
            return False
        report = _json(season, "terminal current-season report")
        counters = report.get("counters")
        return bool(
            isinstance(counters, Mapping)
            and counters.get("accepted_transitions")
            == terminal["accepted_transitions"]
            and counters.get("policy_updates") == terminal["policy_updates"]
            and counters.get("optimizer_steps") == terminal["optimizer_steps"]
            and counters.get("next_policy_version")
            == terminal["next_policy_version"]
            and report.get("lineage_root_sha256")
            == terminal["lineage_root_sha256"]
            and report.get("atlas_catalog_sha256")
            == terminal["atlas_catalog_sha256"]
            and isinstance(report.get("checkpoint_manifest"), Mapping)
            and report["checkpoint_manifest"].get("atlas_catalog_sha256")
            == terminal["atlas_catalog_sha256"]
            and (current_root / str(report.get("last_checkpoint", ""))).resolve()
            == checkpoint
        )
    except (MultiresServiceError, OSError, TypeError, ValueError):
        return False


def run_proof(
    admission: ServiceAdmission, *, lease_acquired: bool = False,
    selector_token: str | None = None,
) -> int:
    if not lease_acquired:
        selector_token = _acquire_lease(admission.runtime_root, "prove")
    if not isinstance(selector_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", selector_token
    ):
        _release_lease(admission.runtime_root)
        raise MultiresServiceError("proof selector token is missing or malformed")
    process: subprocess.Popen | None = None
    record: dict[str, Any] | None = None
    try:
        process = subprocess.Popen(
            admission.proof_command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        record = _process_record(PROOF_ROLE, process)
        _transfer_lease(
            admission.runtime_root, "prove", record, selector_token
        )
        return int(process.wait())
    finally:
        if process is not None and process.poll() is None and record is not None:
            _terminate_owned_tree(
                record,
                timeout=min(
                    3600.0,
                    float(admission.config["proof"]["timeout_seconds"]) + 30.0,
                ),
            )
        _release_lease(admission.runtime_root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument(
        "command", choices=("preflight", "prove", "start", "stop", "status")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selector_lease = False
    selector_token: str | None = None
    reconciliation: dict[str, Any] | None = None
    try:
        if args.command == "stop":
            result = stop_service(args.runtime_root)
        elif args.command == "status":
            result = service_status(args.runtime_root)
        else:
            # The complete integration envelope is re-evaluated before a
            # selector lease or zero-update reconciliation mutates runtime
            # state. start_service performs one more pass immediately before
            # child creation.
            admission = service_preflight(args.runtime_root)
            if args.command in ("prove", "start"):
                runtime_root = _absolute_directory(
                    args.runtime_root, "runtime_root"
                )
                selector_token = _acquire_lease(runtime_root, args.command)
                selector_lease = True
                if args.command == "start":
                    reconciliation = _reconcile_zero_update_attempt(runtime_root)
            if args.command == "preflight":
                result = {
                    "schema": SERVICE_PREFLIGHT_SCHEMA,
                    "passed": True,
                    "service_mode": SERVICE_MODE,
                    "proof": {
                        "service_role": PROOF_ROLE,
                        "training_updates_enabled": False,
                        "transition_count": 500,
                        "command": list(admission.proof_command),
                    },
                    "trainer": {
                        "service_role": PRIMARY_ROLE,
                        "training_updates_enabled": True,
                        "module": PRIMARY_TRAINER_MODULE,
                        "command": list(admission.trainer_command),
                        "current_run_root": str(admission.current_run_root),
                        "tensorboard_root": str(admission.tensorboard_root),
                        "current_season_report": str(
                            admission.current_season_report
                        ),
                    },
                    "runtime_manifest_sha256": (
                        admission.one_run.runtime_manifest_sha256
                    ),
                    "retirement_manifest_sha256": (
                        admission.one_run.retirement_manifest_sha256
                    ),
                    "retirement_validation": dict(
                        admission.retirement_validation
                    ),
                    "retirement_validation_command": list(
                        admission.retirement_validation_command
                    ),
                    "integration_admission": {
                        "envelope": str(admission.integration_envelope),
                        "report": str(admission.integration_report),
                        "report_file_sha256": admission.config[
                            "integration_admission"
                        ]["report"]["sha256"],
                        "report_sha256": (
                            admission.integration_report_sha256
                        ),
                        "bot_source": dict(admission.config[
                            "integration_admission"
                        ]["bot_source"]),
                        "verification": dict(
                            admission.integration_verification
                        ),
                    },
                }
            elif args.command == "prove":
                selector_lease = False
                return run_proof(
                    admission, lease_acquired=True,
                    selector_token=selector_token,
                )
            else:
                selector_lease = False
                result = start_service(
                    admission, lease_acquired=True,
                    selector_token=selector_token,
                )
                result["reconciliation"] = reconciliation
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (
        MultiresServiceError, PrimaryTrainingError, OneRunError,
        ValueError, OSError,
    ) as error:
        print(f"multires service failed: {error}", file=sys.stderr)
        return 2
    finally:
        if selector_lease:
            try:
                _release_lease(
                    _absolute_directory(args.runtime_root, "runtime_root")
                )
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
