#!/usr/bin/env python3
"""Derive B6 G1 recovery evidence from a real full-network qualification run.

This assembler does not accept scenario pass booleans.  It revalidates the
full-network execution and all of its hashed raw scenario files, then extracts
the map-epoch drain, whole-cohort telemetry gap, and one-client fatal timeout
observations required by G1.  It never launches a trainer or updates a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.multires_training_config import MultiresTrainingConfiguration
from harness.runtime_attestation import load_runtime_manifest, verify_runtime_manifest
from tools.assemble_b3_gate import validate_b3_gate
from tools.qualify_network_client_frame_barrier import (
    _load_json_file,
    _validate_execution_evidence,
)

SCHEMA = "q2-multires-b6-g1-fault-probe-v1"
TOOL = "assemble_b6_g1_fault_probe"
_SHA = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")


class FaultProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FaultProbeError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path, label: str) -> Path:
    source = Path(os.path.abspath(path.expanduser()))
    _require(source.is_file() and not source.is_symlink(), f"{label} is not a regular file")
    return source


def _record(path: Path) -> dict[str, Any]:
    source = _file(path, str(path))
    return {"bytes": source.stat().st_size, "sha256": _sha256(source)}


def _git_identity(path: Path, label: str) -> dict[str, Any]:
    repo = Path(os.path.abspath(path.expanduser()))
    _require(repo.is_dir() and not repo.is_symlink(), f"{label} repository is invalid")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo, text=True, stderr=subprocess.STDOUT,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise FaultProbeError(f"{label} Git identity is unavailable") from error
    _require(not status, f"{label} repository is dirty")
    return {"commit": commit, "tree": tree, "clean": True}


def _training_identity(path: Path) -> str:
    value = _load_json_file(path, "training manifest")
    required = {"schema", "reward", "guide_dropout", "ppo"}
    body = {key: value[key] for key in required if key in value}
    _require(set(body) == required, "training manifest body differs")
    configuration = MultiresTrainingConfiguration.create(
        reward=body["reward"], guide_dropout=body["guide_dropout"], ppo=body["ppo"]
    )
    _require(value.get("sha256", configuration.sha256) == configuration.sha256,
             "training manifest identity differs")
    return configuration.sha256


def _raw_scenarios(execution: Mapping[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    root = source.parent.resolve()
    for record in execution["scenario_evidence"]:
        raw_path = source.parent / str(record["path"])
        try:
            resolved = raw_path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise FaultProbeError("raw scenario escapes the execution root") from error
        _require(not raw_path.is_symlink() and resolved.is_file(),
                 "raw scenario is not an exact regular file")
        try:
            data = resolved.read_bytes()
            raw = json.loads(data)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FaultProbeError("raw scenario is not readable JSON") from error
        _require(isinstance(raw, Mapping), "raw scenario is not a JSON object")
        _require(
            type(record.get("size")) is int
            and record["size"] == len(data)
            and record.get("sha256") == hashlib.sha256(data).hexdigest(),
            "raw scenario file identity changed after execution validation",
        )
        _require(raw.get("evidence_sha256") == record.get("raw_evidence_sha256"),
                 "raw scenario/root digest differs")
        result[str(record["scenario"])] = {
            "raw": raw,
            "record": {
                "bytes": int(record["size"]), "sha256": str(record["sha256"]),
                "evidence_sha256": str(record["raw_evidence_sha256"]),
            },
        }
    return result


def assemble(
    *, execution_path: Path, runtime_manifest_path: Path,
    training_manifest_path: Path, objectives_path: Path, checkpoint_path: Path,
    bundle_manifest_path: Path, atlas_path: Path, atlas_manifest_path: Path,
    b3_gate_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    execution_source = _file(execution_path, "full-network execution")
    execution = _validate_execution_evidence(
        _load_json_file(execution_source, "full-network execution"), execution_source
    )
    _require(execution.get("test_mode") is False
             and execution.get("full_network_executed") is True,
             "fault source is not a real full-network execution")
    execution_host = execution.get("execution_host")
    _require(isinstance(execution_host, Mapping)
             and execution_host.get("hostname") == "DESKTOP-RTX2080"
             and "microsoft-standard-WSL2" in str(
                 execution_host.get("kernel_release")
             )
             and execution_host.get("architecture") == "x86_64",
             "full-network execution was not captured on DESKTOP-RTX2080 WSL2")
    manifest = load_runtime_manifest(_file(runtime_manifest_path, "runtime manifest"))
    verified = verify_runtime_manifest(manifest)
    _require(verified.valid and _SHA.fullmatch(verified.digest) is not None,
             "runtime manifest is not sealed")
    runtime_config = manifest["semantic"]["runtime_config"]
    _require(runtime_config.get("network_barrier_execution_evidence_sha256")
             == execution["execution_evidence_sha256"],
             "runtime does not bind the exact full-network execution")

    roots = {
        "bot": Path(repo_root),
        "client": Path(repo_root).resolve().parent / "q2-ml-client",
        "game": Path(repo_root).resolve().parent / "q2-lithium-3zb2",
    }
    sources = {name: _git_identity(path, name) for name, path in roots.items()}
    _require(sources == execution.get("source_repositories"),
             "current source closure differs from the full-network run")

    training_identity = _training_identity(_file(training_manifest_path, "training manifest"))
    _require(runtime_config.get("training_config_sha256") == training_identity,
             "runtime/training semantic identity differs")
    _load_json_file(_file(objectives_path, "objectives"), "objectives")
    objective_identity = _sha256(objectives_path)
    _require(isinstance(objective_identity, str) and _SHA.fullmatch(objective_identity),
             "objective identity is invalid")
    bundle = _load_json_file(_file(bundle_manifest_path, "bundle manifest"), "bundle")
    _require(bundle.get("analysis_files", {}).get(Path(objectives_path).name)
             == _sha256(Path(objectives_path)), "bundle/objectives binding differs")
    atlas_manifest = _load_json_file(
        _file(atlas_manifest_path, "Atlas manifest"), "Atlas manifest"
    )
    _require(atlas_manifest.get("recovery_physics") == {
        "hook_walk_budget_ticks": 15, "game_tick_hz": 10,
        "walk_speed_q8_per_second": 76800,
    }, "Atlas hook-necessity budget/cadence identity differs")
    atlas_record = atlas_manifest.get("artifacts", {}).get(Path(atlas_path).name, {})
    _require(atlas_record.get("sha256_uncompressed") == _sha256(Path(atlas_path))
             and atlas_record.get("uncompressed_size") == Path(atlas_path).stat().st_size,
             "Atlas manifest/raw Atlas binding differs")
    b3_gate = validate_b3_gate(
        _load_json_file(_file(b3_gate_path, "B3 gate"), "B3 gate")
    )
    _require(b3_gate.get("schema") == "q2-multires-b3-gate-v1"
             and b3_gate.get("batch") == "B3"
             and b3_gate.get("status") == "green"
             and b3_gate.get("recovery_guide", {}).get("hook_walk_budget_ticks") == 15
             and b3_gate.get("recovery_guide", {}).get("game_tick_hz") == 10
             and b3_gate.get("recovery_guide", {}).get(
                 "walk_speed_q8_per_second"
             ) == 76800,
             "B3 hook-necessity gate identity differs")
    rust_extension = b3_gate.get("recovery_guide", {}).get("rust_extension")
    runtime_rust = manifest.get("semantic", {}).get("artifacts", {}).get(
        "rust_lattice", {}
    )
    _require(isinstance(rust_extension, Mapping)
             and runtime_rust.get("enabled") is True
             and runtime_rust.get("sha256") == rust_extension.get("sha256")
             and runtime_rust.get("size") == rust_extension.get("bytes"),
             "B3/runtime Rust extension bytes differ")

    raw = _raw_scenarios(execution, execution_source)
    _require({
        "epoch-drain", "whole-batch-telemetry-gap-recovery",
        "partial-client-telemetry-timeout",
    } <= set(raw),
             "full-network execution lacks required G1 scenarios")
    epoch = raw["epoch-drain"]["raw"]
    gap = raw["whole-batch-telemetry-gap-recovery"]["raw"]
    partial = raw["partial-client-telemetry-timeout"]["raw"]
    epoch_rows = epoch.get("trajectory_rows")
    gap_rows = gap.get("trajectory_rows")
    _require(isinstance(epoch_rows, list) and epoch_rows
             and isinstance(gap_rows, list) and gap_rows,
             "recovery scenarios lack raw trajectories")
    recovered_epoch_clients = epoch_rows[-1].get("clients", [])
    recovered_gap_clients = gap_rows[-1].get("clients", [])
    gap_frames = []
    for row in gap_rows:
        clients = row.get("clients", []) if isinstance(row, Mapping) else []
        frames = {
            client.get("server_frame") for client in clients
            if isinstance(client, Mapping)
        }
        _require(len(clients) == 4 and len(frames) == 1,
                 "whole-gap trajectory row is not synchronized")
        gap_frames.append(next(iter(frames)))
    gap_discontinuities = [
        {
            "prior_frame": prior,
            "current_frame": current,
            "delta": current - prior,
            "trajectory_ordinal": index + 1,
        }
        for index, (prior, current) in enumerate(
            zip(gap_frames, gap_frames[1:]), start=1
        )
        if current != prior + 1
    ]
    gap_events = gap.get("transport_fault_events")
    gap_metrics = gap.get("transport_metrics")
    gap_clients = sorted(
        event.get("client_id") for event in gap_events or ()
        if isinstance(event, Mapping)
        and event.get("event") == "udp_telemetry_held_and_released"
    )
    gap_event_sha256 = hashlib.sha256(_canonical(gap_events)).hexdigest()
    held_gap_frames = {
        event.get("server_frame") for event in gap_events or ()
        if isinstance(event, Mapping)
        and event.get("event") == "udp_telemetry_held_and_released"
    }
    _require(gap_discontinuities and len(gap_discontinuities) == 1
             and gap_discontinuities[0]["delta"] == 2
             and gap_clients == [f"qual-{slot:02d}" for slot in range(4)]
             and len(gap_events or ()) == 4
             and held_gap_frames == {gap_discontinuities[0]["current_frame"] - 1}
             and isinstance(gap_metrics, Mapping)
             and gap_metrics.get("telemetry_gap_resyncs") == 1
             and gap_metrics.get("failed_rounds") == 0
             and gap_metrics.get("echo_timeouts") == 0,
             "whole-batch telemetry-gap recovery raw proof differs")
    partial_events = partial.get("transport_fault_events")
    partial_metrics = partial.get("transport_metrics")
    partial_clients = [
        event.get("client_id") for event in partial_events or ()
        if isinstance(event, Mapping)
        and event.get("event") == "udp_telemetry_held_and_released"
    ]
    partial_event_sha256 = hashlib.sha256(_canonical(partial_events)).hexdigest()
    _require(partial_clients == ["qual-00"]
             and len(partial_events or ()) == 1
             and isinstance(partial_metrics, Mapping)
             and partial_metrics.get("telemetry_gap_resyncs") == 0
             and partial_metrics.get("failed_rounds") == 1
             and partial_metrics.get("echo_timeouts") == 1
             and partial.get("observed_outcome") == "fatal"
             and partial.get("exception") == "AuthoritativeEchoError",
             "partial-client live telemetry timeout raw proof differs")

    file_bindings = {
        "runtime_manifest": _record(runtime_manifest_path),
        "training_manifest": _record(training_manifest_path),
        "objectives": _record(objectives_path),
        "checkpoint": _record(checkpoint_path),
        "bundle_manifest": _record(bundle_manifest_path),
        "atlas": _record(atlas_path),
        "atlas_manifest": _record(atlas_manifest_path),
        "b3_gate": _record(b3_gate_path),
        "b3_gate_sha256": b3_gate["gate_sha256"],
    }
    bindings = {
        **file_bindings,
        "runtime_manifest_identity_sha256": verified.digest,
        "training_config_sha256": training_identity,
        "objective_identity_sha256": objective_identity,
        "network_barrier_execution_evidence_sha256": execution[
            "execution_evidence_sha256"
        ],
        "hook_necessity_runtime": {
            "hook_walk_budget_ticks": 15, "game_tick_hz": 10,
            "walk_speed_q8_per_second": 76800,
        },
        "rust_extension": dict(rust_extension),
    }
    scenarios = [
        {
            "name": "map-epoch-recovery",
            "source_scenario": "epoch-drain",
            "source": raw["epoch-drain"]["record"],
            "observed_outcome": epoch.get("observed_outcome"),
            "epoch_drains": epoch.get("epoch_drains"),
            "new_epoch_bootstrap_frames": epoch.get("new_epoch_bootstrap_frames"),
            "actions_dispatched_during_epoch_drain": epoch.get(
                "actions_dispatched_during_epoch_drain"
            ),
            "boundary_rounds": epoch.get("boundary_rounds"),
            "accepted_after_recovery": len(recovered_epoch_clients),
            "recovered_map_epoch": (
                recovered_epoch_clients[0].get("map_epoch")
                if recovered_epoch_clients else None
            ),
        },
        {
            "name": "whole-batch-telemetry-gap-recovery",
            "source_scenario": "whole-batch-telemetry-gap-recovery",
            "source": raw["whole-batch-telemetry-gap-recovery"]["record"],
            "observed_outcome": gap.get("observed_outcome"),
            "boundary_rounds": gap.get("boundary_rounds"),
            "accepted_synchronized_frames": gap.get("accepted_synchronized_frames"),
            "fault_event_count": gap.get("fault_event_count"),
            "accepted_after_recovery": len(recovered_gap_clients),
            "frame_discontinuities": gap_discontinuities,
            "timeout_client_ids": gap_clients,
            "relay_event_count": len(gap_events),
            "relay_events_sha256": gap_event_sha256,
            "transport_metrics": dict(gap_metrics),
        },
        {
            "name": "partial-client-timeout-fatal",
            "source_scenario": "partial-client-telemetry-timeout",
            "source": raw["partial-client-telemetry-timeout"]["record"],
            "observed_outcome": partial.get("observed_outcome"),
            "fault_event_count": partial.get("fault_event_count"),
            "exception": partial.get("exception"),
            "accepted_synchronized_frames": partial.get("accepted_synchronized_frames"),
            "timeout_client_ids": partial_clients,
            "relay_event_count": len(partial_events),
            "relay_events_sha256": partial_event_sha256,
            "transport_metrics": dict(partial_metrics),
        },
    ]
    result = {
        "schema": SCHEMA,
        "tool": TOOL,
        "synthetic": False,
        "host": dict(execution_host),
        "source_repositories": sources,
        "bindings": bindings,
        "full_network_execution": {
            **_record(execution_source),
            "execution_evidence_sha256": execution["execution_evidence_sha256"],
        },
        "scenarios": scenarios,
    }
    result["evidence_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(path.expanduser()))
    _require(not destination.exists() and not destination.is_symlink(),
             "fault-probe output already exists")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "execution", "runtime_manifest", "training_manifest", "objectives",
        "checkpoint", "bundle_manifest", "atlas", "atlas_manifest", "b3_gate",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        value = assemble(
            execution_path=args.execution, runtime_manifest_path=args.runtime_manifest,
            training_manifest_path=args.training_manifest, objectives_path=args.objectives,
            checkpoint_path=args.checkpoint,
            bundle_manifest_path=args.bundle_manifest, atlas_path=args.atlas,
            atlas_manifest_path=args.atlas_manifest, b3_gate_path=args.b3_gate,
            repo_root=args.repo_root,
        )
        _publish(args.output, value)
    except Exception as error:
        print(f"B6 fault probe refused: {error}", file=sys.stderr)
        return 1
    print(_canonical(value).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
