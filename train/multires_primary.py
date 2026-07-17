"""Operational composition root for the continuous multires primary trainer.

This module owns no alternative collector or policy path.  It composes the
sealed network-client runtime admitted by ``multires_service`` with the frozen
B7 continuous core, one exact checkpoint, and one run-local SummaryWriter.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import hashlib
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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRIMARY_TRAINING_CONFIG_SCHEMA = "q2-multires-primary-training-runtime-v1"
PRIMARY_TRAINER_MODULE = "train.multires_primary"
PROOF_MODULE = "train.multires_one_run"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,63}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{32,63}\Z")
PRIMARY_CHILD_INVENTORY_SCHEMA = "q2-multires-primary-child-inventory-v1"
PRIMARY_TERMINAL_SCHEMA = "q2-multires-primary-terminal-v1"
PRIMARY_ATTEMPT_SCHEMA = "q2-multires-primary-attempt-v1"
PRIMARY_CHILD_INVENTORY_NAME = "multires-primary-children.json"
PRIMARY_TERMINAL_NAME = "multires-primary-terminal.json"
PRIMARY_ATTEMPT_NAME = "multires-primary-attempt.json"
PRIMARY_SELECTOR_TOKEN_ENV = "Q2_MULTIRES_PRIMARY_SELECTOR_TOKEN"


class PrimaryTrainingError(RuntimeError):
    """Raised before or during the one admitted continuous trainer."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value)) and value != "0" * 64


def _file(value: object, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PrimaryTrainingError(
            f"{label} must be an absolute regular non-symlink file"
        )
    return path.resolve()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrimaryTrainingError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PrimaryTrainingError(f"{label} must be a JSON object")
    return value


def _artifact_record(value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PrimaryTrainingError(f"{label} artifact record is malformed")
    path = _file(value["path"], label)
    digest = value["sha256"]
    if not _valid_sha256(digest) or _sha256(path) != digest:
        raise PrimaryTrainingError(f"{label} artifact digest differs")
    return path, str(digest)


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise PrimaryTrainingError(f"refusing to replace evidence: {path}")
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _proc_record(role: str, pid: int) -> dict[str, Any]:
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        state = str(tail[0])
        pgrp, session, start_ticks = int(tail[2]), int(tail[3]), int(tail[19])
    except (OSError, IndexError, ValueError) as error:
        raise PrimaryTrainingError(f"cannot attest primary process {pid}") from error
    if state == "Z" or int(pid) <= 1 or min(pgrp, session, start_ticks) <= 0:
        raise PrimaryTrainingError(f"primary process {pid} is not live/attestable")
    return {
        "role": role, "pid": int(pid), "start_ticks": start_ticks,
        "process_group": pgrp, "session": session,
    }


def _verify_primary_selector_lease(runtime_root: Path) -> tuple[dict[str, Any], str]:
    token = os.environ.get(PRIMARY_SELECTOR_TOKEN_ENV, "")
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise PrimaryTrainingError("primary selector token is missing or malformed")
    token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
    current = _proc_record(PRIMARY_TRAINER_MODULE, os.getpid())
    identity_fields = ("pid", "start_ticks", "process_group", "session")
    lease_path = runtime_root / "multires-service.lock"
    deadline = time.monotonic() + 5.0
    while True:
        if lease_path.is_symlink() or not lease_path.is_file():
            raise PrimaryTrainingError("primary has no owned service runtime lease")
        lease = _json(lease_path, "primary service runtime lease")
        owner = lease.get("owner")
        if (
            set(lease) != {"schema", "mode", "owner", "selector_token_sha256"}
            or lease.get("schema") != "q2-multires-service-lease-v1"
            or lease.get("mode") != "start"
            or lease.get("selector_token_sha256") != token_sha256
            or not isinstance(owner, Mapping)
        ):
            raise PrimaryTrainingError("primary selector lease ownership differs")
        if all(owner.get(field) == current[field] for field in identity_fields):
            break
        if time.monotonic() >= deadline:
            raise PrimaryTrainingError("primary selector lease transfer timed out")
        time.sleep(0.01)
    # Prevent the private supervisor capability from reaching q2ded/clients.
    os.environ.pop(PRIMARY_SELECTOR_TOKEN_ENV, None)
    return current, token_sha256


def _write_attempt_evidence(
    path: Path,
    admission: "PrimaryTrainingAdmission",
    *,
    owner: Mapping[str, Any],
    selector_token_sha256: str,
) -> dict[str, Any]:
    suffix = f".attempt-{selector_token_sha256[:16]}"
    payload = {
        "schema": PRIMARY_ATTEMPT_SCHEMA,
        "runtime_manifest_sha256": admission.service.one_run.runtime_manifest_sha256,
        "training_runtime_sha256": admission.config_sha256,
        "checkpoint_mode": admission.checkpoint_mode,
        "selected_checkpoint": str(admission.checkpoint),
        "selected_checkpoint_sha256": _sha256(admission.checkpoint),
        "current_run_root": str(admission.current_run_root),
        "checkpoint_root": str(admission.checkpoint_root),
        "tensorboard_root": str(admission.tensorboard_root),
        "rollout_root": str(admission.rollout_root),
        "update_root": str(admission.update_root),
        "current_season_report": str(admission.season_report_root / "current.json"),
        "owner": dict(owner),
        "selector_token_sha256": selector_token_sha256,
        "tensorboard_filename_suffix": suffix,
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _atomic_json(path, payload, replace=False)
    return payload


def _write_child_inventory(
    path: Path,
    *,
    runtime_manifest_sha256: str,
    launch_config: Path,
    processes: list[dict[str, Any]],
    expected_client_count: int,
    complete: bool,
) -> None:
    expected_roles = {"q2ded"} | {
        f"network-client-{index:02d}" for index in range(expected_client_count)
    }
    roles = [record.get("role") for record in processes]
    if len(roles) != len(set(roles)) or not set(roles) <= expected_roles:
        raise PrimaryTrainingError("primary child inventory roles differ")
    if complete and set(roles) != expected_roles:
        raise PrimaryTrainingError("primary lacks exact q2ded/client process evidence")
    owner = _proc_record(PRIMARY_TRAINER_MODULE, os.getpid())
    payload = {
        "schema": PRIMARY_CHILD_INVENTORY_SCHEMA,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "owner": owner,
        "launch_config": str(launch_config.resolve()),
        "expected_client_count": expected_client_count,
        "complete": complete,
        "processes": processes,
    }
    _atomic_json(path, payload, replace=True)


def _write_terminal_evidence(
    path: Path, admission: "PrimaryTrainingAdmission", summary: Any
) -> dict[str, Any]:
    season_path = Path(summary.current_season_report).resolve()
    season = _json(season_path, "terminal current-season report")
    checkpoint = (
        admission.current_run_root / str(season.get("last_checkpoint", ""))
    ).resolve()
    if (
        checkpoint.parent != admission.checkpoint_root
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or season.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise PrimaryTrainingError("terminal checkpoint/current-season identity differs")
    payload = {
        "schema": PRIMARY_TERMINAL_SCHEMA,
        "status": "completed",
        "runtime_manifest_sha256": admission.service.one_run.runtime_manifest_sha256,
        "lineage_root_sha256": summary.lineage_root_sha256,
        "accepted_transitions": summary.accepted_transitions,
        "policy_updates": summary.policy_updates,
        "optimizer_steps": summary.optimizer_steps,
        "next_policy_version": summary.next_policy_version,
        "current_season_report": str(season_path),
        "current_season_sha256": _sha256(season_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _atomic_json(path, payload, replace=True)
    return payload


@dataclass(frozen=True)
class PrimaryTrainingAdmission:
    service: Any
    config_path: Path
    config_sha256: str
    config: Mapping[str, Any]
    checkpoint: Path
    checkpoint_mode: str
    expected_lineage_root_sha256: str
    current_season_report: Path | None
    stage: Any
    predecessor_gate: Mapping[str, Any] | None
    collector_config: Any
    current_run_root: Path
    checkpoint_root: Path
    tensorboard_root: Path
    rollout_root: Path
    update_root: Path
    season_report_root: Path


@dataclass
class PrimaryOwnedResources:
    """Idempotent, scoped close order for trainer-owned runtime resources."""

    server: Any
    launch_config: Path
    stop_server: Any
    batch: Any = None
    writer: Any = None
    child_inventory: Path | None = None
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors: list[str] = []
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception as error:
                errors.append(f"writer_flush={error}")
            try:
                self.writer.close()
            except Exception as error:
                errors.append(f"writer_close={error}")
        if self.batch is not None:
            try:
                self.batch.close()
            except Exception as error:
                errors.append(f"batch={error}")
        try:
            self.stop_server(self.server)
        except Exception as error:
            errors.append(f"server={error}")
        try:
            self.launch_config.unlink(missing_ok=True)
        except Exception as error:
            errors.append(f"launch_config={error}")
        if not errors and self.child_inventory is not None:
            try:
                self.child_inventory.unlink(missing_ok=True)
            except Exception as error:
                errors.append(f"child_inventory={error}")
        if errors:
            raise PrimaryTrainingError(
                "primary scoped resource close failed: " + "; ".join(errors)
            )


def run_admitted_core(core: Any, execution: Mapping[str, Any], should_stop) -> Any:
    """One service seam for finite and continuous B7 core execution."""
    summary = core.run(
        maximum_updates=execution["maximum_updates"],
        should_stop=should_stop,
    )
    if (
        type(getattr(summary, "policy_updates", None)) is not int
        or summary.policy_updates < 1
        or type(getattr(summary, "optimizer_steps", None)) is not int
        or summary.optimizer_steps < summary.policy_updates
        or type(getattr(summary, "accepted_transitions", None)) is not int
        or summary.accepted_transitions < 1
    ):
        raise PrimaryTrainingError(
            "primary core returned without a real optimizer update"
        )
    return summary


def admit_primary_training(
    service: Any,
    *,
    config_path: Path,
    config_sha256: str,
    cold_document: Mapping[str, Any],
) -> PrimaryTrainingAdmission:
    """Validate the training-only runtime without reinterpreting proof knobs."""
    from harness.multires_collector import CollectorConfig
    from train.multires_train import CurriculumStage, progress_from_season_report

    path = _file(config_path, "primary training runtime configuration")
    if not _valid_sha256(config_sha256) or _sha256(path) != config_sha256:
        raise PrimaryTrainingError("primary training runtime configuration digest differs")
    config = _json(path, "primary training runtime configuration")
    required = {
        "schema", "runtime_manifest_sha256",
        "network_barrier_execution_evidence_sha256", "seed", "game_seed",
        "map_name", "map_epoch", "collector", "optimizer", "curriculum",
        "execution", "checkpoint",
    }
    if set(config) != required or config.get("schema") != PRIMARY_TRAINING_CONFIG_SCHEMA:
        raise PrimaryTrainingError("primary training runtime fields/schema differ")
    one_run = service.one_run
    if config["runtime_manifest_sha256"] != one_run.runtime_manifest_sha256:
        raise PrimaryTrainingError("primary/service runtime manifest digests differ")
    if (
        not _valid_sha256(config["network_barrier_execution_evidence_sha256"])
        or config["network_barrier_execution_evidence_sha256"]
        != one_run.network_barrier_execution_evidence_sha256
    ):
        raise PrimaryTrainingError(
            "primary frame-barrier execution evidence digest differs"
        )
    for field in ("seed", "game_seed", "map_epoch"):
        if type(config[field]) is not int or int(config[field]) < 0:
            raise PrimaryTrainingError(f"primary {field} is invalid")
    if (
        not isinstance(config["map_name"], str)
        or not _SAFE_NAME.fullmatch(config["map_name"])
        or config["map_name"] != one_run.args.map_name
    ):
        raise PrimaryTrainingError("primary map name differs from the admitted bundle")

    collector = config["collector"]
    if not isinstance(collector, Mapping) or set(collector) != {
        "transitions_per_client", "gamma", "gae_lambda",
        "maximum_boundary_rounds",
    }:
        raise PrimaryTrainingError("primary collector configuration fields differ")
    if (
        type(collector["transitions_per_client"]) is not int
        or collector["transitions_per_client"] < 1
        or type(collector["maximum_boundary_rounds"]) is not int
        or collector["maximum_boundary_rounds"] < 0
        or any(
            isinstance(collector[name], bool)
            or not isinstance(collector[name], (int, float))
            or not math.isfinite(float(collector[name]))
            for name in ("gamma", "gae_lambda")
        )
    ):
        raise PrimaryTrainingError("primary collector value types are invalid")
    try:
        collector_config = CollectorConfig(
            transitions_per_client=collector["transitions_per_client"],
            gamma=collector["gamma"],
            gae_lambda=collector["gae_lambda"],
            maximum_boundary_rounds=collector["maximum_boundary_rounds"],
        )
        collector_config.validate()
    except (TypeError, ValueError) as error:
        raise PrimaryTrainingError(f"primary collector configuration is invalid: {error}") from error
    if config["optimizer"] != one_run.runtime_config.get("optimizer"):
        raise PrimaryTrainingError("primary/one-run optimizer configurations differ")
    if config["optimizer"] != cold_document.get("optimizer"):
        raise PrimaryTrainingError("primary/cold-start optimizer configurations differ")
    optimizer = config["optimizer"]
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != {"class", "learning_rate", "kwargs"}
        or optimizer["class"] != "torch.optim.Adam"
        or isinstance(optimizer["learning_rate"], bool)
        or not isinstance(optimizer["learning_rate"], (int, float))
        or not math.isfinite(float(optimizer["learning_rate"]))
        or float(optimizer["learning_rate"]) <= 0.0
        or not isinstance(optimizer["kwargs"], Mapping)
        or "lr" in optimizer["kwargs"]
    ):
        raise PrimaryTrainingError("primary optimizer configuration is invalid")

    curriculum = config["curriculum"]
    if not isinstance(curriculum, Mapping) or set(curriculum) != {
        "stage_id", "configuration", "configuration_sha256",
        "predecessor_stage_sha256", "predecessor_gate",
    }:
        raise PrimaryTrainingError("primary curriculum fields differ")
    if (
        type(curriculum["stage_id"]) is not int
        or not isinstance(curriculum["configuration"], Mapping)
        or not curriculum["configuration"]
    ):
        raise PrimaryTrainingError("primary curriculum value types are invalid")
    try:
        stage = CurriculumStage.create(
            curriculum["stage_id"],
            curriculum["configuration"],
            predecessor_stage_sha256=curriculum["predecessor_stage_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise PrimaryTrainingError(f"primary curriculum stage is invalid: {error}") from error
    if stage.configuration_sha256 != curriculum["configuration_sha256"]:
        raise PrimaryTrainingError("primary curriculum configuration digest differs")
    gate_record = curriculum["predecessor_gate"]
    predecessor_gate = None
    if gate_record is not None:
        gate_path, _gate_sha = _artifact_record(gate_record, "predecessor gate")
        predecessor_gate = _json(gate_path, "predecessor gate")
    if (stage.stage_id == 1) != (predecessor_gate is None):
        raise PrimaryTrainingError("primary predecessor gate/stage relationship differs")

    execution = config["execution"]
    if not isinstance(execution, Mapping) or set(execution) != {
        "maximum_updates", "continuous", "device", "deterministic_collection",
        "deterministic_algorithms", "cuda_cublas_workspace_config",
        "shutdown_grace_seconds",
    }:
        raise PrimaryTrainingError("primary execution fields differ")
    continuous = execution["continuous"]
    maximum_updates = execution["maximum_updates"]
    if type(continuous) is not bool or (
        continuous and maximum_updates is not None
    ) or (
        not continuous and (type(maximum_updates) is not int or maximum_updates < 1)
    ):
        raise PrimaryTrainingError("primary continuous/maximum update contract differs")
    device = execution["device"]
    if not isinstance(device, str) or not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", device):
        raise PrimaryTrainingError("primary execution device is invalid")
    if execution["deterministic_collection"] is not True or execution[
        "deterministic_algorithms"
    ] is not True:
        raise PrimaryTrainingError("primary deterministic execution must be enabled")
    workspace = execution["cuda_cublas_workspace_config"]
    if device.startswith("cuda"):
        if workspace not in (":4096:8", ":16:8"):
            raise PrimaryTrainingError("primary CUDA workspace determinism is invalid")
    elif workspace is not None:
        raise PrimaryTrainingError("CPU primary runtime rejects CUDA workspace settings")
    shutdown_grace = execution["shutdown_grace_seconds"]
    minimum_shutdown_grace = (
        float(one_run.runtime_config["client_timeout"])
        + (
            int(collector["transitions_per_client"])
            + int(collector["maximum_boundary_rounds"])
        ) * float(one_run.runtime_config["round_timeout"])
        + 30.0
    )
    if (
        isinstance(shutdown_grace, bool)
        or not isinstance(shutdown_grace, (int, float))
        or not math.isfinite(float(shutdown_grace))
        or float(shutdown_grace) < minimum_shutdown_grace
        or float(shutdown_grace) > 3600.0
    ):
        raise PrimaryTrainingError(
            "primary shutdown grace cannot cover one full admitted update"
        )

    checkpoint = config["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "mode", "path", "sha256", "lineage_root_sha256",
        "current_season_report",
    }:
        raise PrimaryTrainingError("primary checkpoint selector fields differ")
    checkpoint_path, _checkpoint_sha = _artifact_record(
        {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
        "primary checkpoint",
    )
    mode = checkpoint["mode"]
    lineage = checkpoint["lineage_root_sha256"]
    if mode not in ("fresh-step-zero", "same-lineage-resume") or not _valid_sha256(lineage):
        raise PrimaryTrainingError("primary checkpoint mode/lineage is invalid")
    season_path = None
    season_record = checkpoint["current_season_report"]
    if mode == "fresh-step-zero":
        if season_record is not None or checkpoint_path != Path(one_run.args.checkpoint).resolve():
            raise PrimaryTrainingError("fresh primary checkpoint differs from proof step zero")
    else:
        season_path, _season_sha = _artifact_record(
            season_record, "current season resume report"
        )

    lineage_document = cold_document.get("lineage")
    if not isinstance(lineage_document, Mapping):
        raise PrimaryTrainingError("cold-start lineage roots are missing")
    root_names = (
        "current_run_root", "checkpoint_root", "tensorboard_root",
        "rollout_root", "update_root", "season_report_root",
    )
    roots = {name: Path(str(lineage_document[name])).resolve() for name in root_names}
    current = roots["current_run_root"]
    exact = {
        "checkpoint_root": current / "checkpoints",
        "tensorboard_root": current / "tensorboard",
        "rollout_root": current / "evidence" / "rollouts",
        "update_root": current / "evidence" / "updates",
        "season_report_root": current / "season",
    }
    for name, expected in exact.items():
        if roots[name] != expected.resolve() or not roots[name].is_dir() or roots[name].is_symlink():
            raise PrimaryTrainingError(f"cold-start {name} differs from the training core layout")
    if checkpoint_path.parent != roots["checkpoint_root"]:
        raise PrimaryTrainingError("primary checkpoint is outside the declared checkpoint root")
    expected_season = roots["season_report_root"] / "current.json"
    if season_path is not None and season_path != expected_season:
        raise PrimaryTrainingError("resume report differs from declared season root")

    def regular_children(root: Path) -> tuple[Path, ...]:
        children = tuple(sorted(root.iterdir()))
        if any(child.is_symlink() or not child.is_file() for child in children):
            raise PrimaryTrainingError(
                f"primary run root contains non-regular content: {root}"
            )
        return children

    if mode == "fresh-step-zero":
        checkpoint_children = regular_children(roots["checkpoint_root"])
        if checkpoint_children != (checkpoint_path,):
            raise PrimaryTrainingError(
                "fresh checkpoint root must contain only the selected step-zero file"
            )
        for name in (
            "tensorboard_root", "rollout_root", "update_root", "season_report_root"
        ):
            if regular_children(roots[name]):
                raise PrimaryTrainingError(
                    f"fresh {name} must be empty before primary start"
                )
    else:
        report = _json(season_path, "current season resume report")
        try:
            reported_checkpoint = (
                current / str(report["last_checkpoint"])
            ).resolve()
        except (KeyError, TypeError) as error:
            raise PrimaryTrainingError(
                "resume report lacks its exact checkpoint selection"
            ) from error
        if (
            reported_checkpoint != checkpoint_path
            or report.get("checkpoint_sha256") != _sha256(checkpoint_path)
            or report.get("lineage_root_sha256") != lineage
        ):
            raise PrimaryTrainingError(
                "resume report/checkpoint/lineage selection differs"
            )
        try:
            progress_from_season_report(
                report,
                stage,
                runtime_manifest_sha256=one_run.runtime_manifest_sha256,
                atlas_sha256=one_run.args.expected_atlas_sha256,
                lineage_root_sha256=str(lineage),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise PrimaryTrainingError(
                f"resume current-season evidence is invalid: {error}"
            ) from error
        for name in (
            "checkpoint_root", "tensorboard_root", "rollout_root",
            "update_root", "season_report_root",
        ):
            regular_children(roots[name])

    return PrimaryTrainingAdmission(
        service=service,
        config_path=path,
        config_sha256=str(config_sha256),
        config=config,
        checkpoint=checkpoint_path,
        checkpoint_mode=str(mode),
        expected_lineage_root_sha256=str(lineage),
        current_season_report=season_path,
        stage=stage,
        predecessor_gate=predecessor_gate,
        collector_config=collector_config,
        current_run_root=current,
        checkpoint_root=roots["checkpoint_root"],
        tensorboard_root=roots["tensorboard_root"],
        rollout_root=roots["rollout_root"],
        update_root=roots["update_root"],
        season_report_root=roots["season_report_root"],
    )


def execute_primary(admission: PrimaryTrainingAdmission) -> dict[str, Any]:
    """Run finite or continuous training and close every owned resource."""
    execution = admission.config["execution"]
    one_run = admission.service.one_run
    token_name = one_run.runtime_config.get("telemetry_token_env")
    if token_name != "Q2_ML_CLIENT_TELEMETRY_TOKEN":
        raise PrimaryTrainingError("primary telemetry token selector differs")
    token = os.environ.get(token_name, "")
    if not _TOKEN.fullmatch(token):
        raise PrimaryTrainingError(
            "primary telemetry token environment is missing or malformed"
        )
    runtime_root = Path(one_run.runtime_root).resolve()
    lease_owner, selector_token_sha256 = _verify_primary_selector_lease(
        runtime_root
    )
    child_inventory = runtime_root / PRIMARY_CHILD_INVENTORY_NAME
    terminal_path = runtime_root / PRIMARY_TERMINAL_NAME
    attempt_path = runtime_root / PRIMARY_ATTEMPT_NAME
    if attempt_path.is_symlink() or attempt_path.exists():
        raise PrimaryTrainingError(
            "unreconciled primary attempt blocks direct execution"
        )
    for stale, label in (
        (child_inventory, "primary child inventory"),
        (terminal_path, "primary terminal evidence"),
    ):
        if stale.is_symlink():
            raise PrimaryTrainingError(f"{label} symlinks are rejected")
        stale.unlink(missing_ok=True)
    attempt = _write_attempt_evidence(
        attempt_path, admission, owner=lease_owner,
        selector_token_sha256=selector_token_sha256,
    )
    workspace = execution["cuda_cublas_workspace_config"]
    if workspace is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(workspace)

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from harness.client_batch import build_network_client_batch
    from harness.multires_contract import GuideDropoutConfig
    from harness.multires_metrics import MultiresSeasonMetrics
    from harness.multires_reward import CausalRewardConfig
    from harness.rust_multires_provider import RustAtlasProviderFactory, RustMapArtifacts
    from train.multires_live import MultiresLiveTrainer
    from train.multires_one_run import (
        _load_extension,
        _stop_server,
        _write_server_config,
    )
    from train.multires_ppo import MultiresPPOConfig
    from train.multires_runtime import MultiresTrainerRuntime
    from train.multires_train import (
        MultiresContinuousTrainingCore,
        SanitizedCausalMetricsAccumulator,
        TrainingProgress,
        progress_from_season_report,
    )

    config = one_run.runtime_config
    seed = int(admission.config["seed"])
    device = torch.device(str(execution["device"]))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    training = one_run.training_configuration
    reward_config = CausalRewardConfig(**dict(training.reward))
    guide_dropout = GuideDropoutConfig(**dict(training.guide_dropout))
    ppo_config = MultiresPPOConfig(**dict(training.ppo))
    optimizer_config = admission.config["optimizer"]

    def optimizer_factory(parameters):
        return torch.optim.Adam(
            parameters,
            lr=float(optimizer_config["learning_rate"]),
            **dict(optimizer_config["kwargs"]),
        )

    runtime, optimizer, manifest = MultiresTrainerRuntime.resume(
        admission.checkpoint,
        one_run.runtime_evidence,
        expected_atlas_sha256=one_run.args.expected_atlas_sha256,
        device=device,
        optimizer_factory=optimizer_factory,
        reward_config=reward_config,
        guide_dropout=guide_dropout,
        ppo_config=ppo_config,
        expected_lineage_root_sha256=admission.expected_lineage_root_sha256,
    )
    if admission.checkpoint_mode == "fresh-step-zero":
        if (
            manifest.initialization != "random"
            or manifest.training_step != 0
            or optimizer.state_dict().get("state") != {}
        ):
            raise PrimaryTrainingError("fresh primary checkpoint is not exact random step zero")
        progress = TrainingProgress(lineage_root_sha256=manifest.lineage_root_sha256)
    else:
        if manifest.training_step < 1 or admission.current_season_report is None:
            raise PrimaryTrainingError("resume checkpoint has no positive training progress")
        report = _json(admission.current_season_report, "current season resume report")
        progress = progress_from_season_report(
            report,
            admission.stage,
            runtime_manifest_sha256=runtime.runtime.runtime_manifest_sha256,
            atlas_sha256=runtime.runtime.atlas_sha256,
            lineage_root_sha256=manifest.lineage_root_sha256,
        )
        if progress.accepted_transitions != manifest.training_step:
            raise PrimaryTrainingError("resume checkpoint/report training steps differ")

    extension = _load_extension(one_run.rust_extension)
    count = int(config["client_count"])
    if count != 4:
        raise PrimaryTrainingError("primary trainer requires exact four-client cohort")
    factories = [
        RustAtlasProviderFactory(
            extension_module=extension,
            artifacts_by_map={
                admission.config["map_name"]: RustMapArtifacts(
                    bundle_manifest_path=one_run.bundle_manifest,
                    uncompressed_atlas_path=one_run.atlas_bin,
                    dyn_snapshot_path=one_run.dyn_snapshots[index],
                    expected_atlas_sha256=one_run.args.expected_atlas_sha256,
                )
            },
            runtime_manifest_sha256=one_run.runtime_manifest_sha256,
            bound_client_id=f"{config['client_id_prefix']}-{index:02d}",
            rust_client_id=index,
            client_count=count,
        )
        for index in range(count)
    ]

    trainer_args = argparse.Namespace(**vars(one_run.args))
    trainer_args.seed = seed
    trainer_args.game_seed = int(admission.config["game_seed"])
    trainer_args.map_epoch = int(admission.config["map_epoch"])
    trainer_args.map_name = str(admission.config["map_name"])
    trainer_args.launch_id = "primary-trainer"
    trainer_one_run = dataclasses.replace(one_run, args=trainer_args)
    launch_cfg = _write_server_config(trainer_one_run)
    server = subprocess.Popen(
        [
            str(one_run.q2ded), "+set", "game", str(config["game"]),
            "+set", "port", str(int(config["server_port"])),
            "+exec", launch_cfg.name,
        ],
        cwd=one_run.q2_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    resources = PrimaryOwnedResources(
        server=server,
        launch_config=launch_cfg,
        stop_server=_stop_server,
        child_inventory=child_inventory,
    )
    child_records = [_proc_record("q2ded", int(server.pid))]
    _write_child_inventory(
        child_inventory,
        runtime_manifest_sha256=one_run.runtime_manifest_sha256,
        launch_config=launch_cfg,
        processes=child_records,
        expected_client_count=count,
        complete=False,
    )
    stop_requested = False
    prior_handlers: dict[int, Any] = {}

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        prior_handlers[signum] = signal.signal(signum, request_stop)
    try:
        warmup = float(config["server_warmup_seconds"])
        if warmup:
            time.sleep(warmup)
        if server.poll() is not None:
            raise PrimaryTrainingError("q2ded exited before primary clients connected")
        batch = build_network_client_batch(
            n_clients=count,
            server=f"{config['server_host']}:{int(config['server_port'])}",
            telemetry_server=f"{config['telemetry_host']}:{int(config['telemetry_port'])}",
            telemetry_token=token,
            client_binary=str(one_run.client_binary),
            client_root=str(one_run.q2_root),
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
            spatial_seed=seed,
            debug=bool(config["debug_clients"]),
            deterministic_frame_barrier=True,
            multires_spatial_provider_factories=factories,
            expected_runtime_manifest_sha256=one_run.runtime_manifest_sha256,
        )
        resources.batch = batch
        metrics = SanitizedCausalMetricsAccumulator(MultiresSeasonMetrics(
            season_id=f"stage-{admission.stage.stage_id:02d}",
            atlas_sha256=runtime.runtime.atlas_sha256,
            policy_start_version=progress.next_policy_version,
        ))
        trainer = MultiresLiveTrainer(
            runtime,
            batch,
            optimizer,
            device=device,
            collector_config=admission.collector_config,
            deterministic_collection=True,
            transition_observer=metrics,
        )
        initial_observations, initial_infos = batch.reset()
        for index, environment in enumerate(batch.envs):
            process = getattr(environment, "_process", None)
            if process is None:
                raise PrimaryTrainingError(
                    f"network client {index} has no launched process identity"
                )
            child_records.append(_proc_record(
                f"network-client-{index:02d}", int(process.pid)
            ))
        _write_child_inventory(
            child_inventory,
            runtime_manifest_sha256=one_run.runtime_manifest_sha256,
            launch_config=launch_cfg,
            processes=child_records,
            expected_client_count=count,
            complete=True,
        )
        trainer.collector.prime(initial_observations, initial_infos)
        writer = SummaryWriter(
            log_dir=str(admission.tensorboard_root),
            filename_suffix=str(attempt["tensorboard_filename_suffix"]),
        )
        resources.writer = writer
        if Path(writer.log_dir).resolve() != admission.tensorboard_root:
            raise PrimaryTrainingError("SummaryWriter escaped the declared tensorboard root")
        core = MultiresContinuousTrainingCore(
            trainer,
            output_root=admission.current_run_root,
            stage=admission.stage,
            writer=writer,
            causal_metrics=metrics,
            predecessor_gate=admission.predecessor_gate,
            progress=progress,
        )
        summary = run_admitted_core(
            core,
            execution,
            lambda: stop_requested,
        )
        terminal = _write_terminal_evidence(terminal_path, admission, summary)
        attempt_path.unlink()
        return {
            "schema": "q2-multires-primary-training-result-v1",
            "service_role": "primary-trainer",
            "training_updates_enabled": True,
            "accepted_transitions": summary.accepted_transitions,
            "policy_updates": summary.policy_updates,
            "optimizer_steps": summary.optimizer_steps,
            "next_policy_version": summary.next_policy_version,
            "lineage_root_sha256": summary.lineage_root_sha256,
            "current_season_report": str(summary.current_season_report),
            "tensorboard_root": str(admission.tensorboard_root),
            "terminal_evidence": str(terminal_path),
            "terminal_evidence_sha256": terminal["evidence_sha256"],
        }
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
        resources.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from train.multires_service import service_preflight

        service = service_preflight(args.runtime_root)
        result = execute_primary(service.primary)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as error:
        print(f"multires primary trainer failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
