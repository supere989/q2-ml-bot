"""Unit tests for the fail-closed multires 500-transition proof harness (M6d)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
import time

import pytest

from harness.multires_contract import (
    ACTION_DIM,
    DYN,
    FEATURE_SCHEMA_SHA256,
    OBS_DIM,
    POLICY_GENERATION,
)
from harness.multires_lineage import CHECKPOINT_FORMAT, save_attested_checkpoint
from harness.multires_runtime import (
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
    adapt_b4_observation_descriptor,
    validate_runtime_evidence,
)
from harness.multires_training_config import (
    TRAINING_CONFIG_SCHEMA,
    MultiresTrainingConfiguration,
)
import tools.run_multires_500_transition_proof as proof_module
from tools.run_multires_500_transition_proof import (
    COLLECTOR_CLASS_NAME,
    DEFAULT_SYNTHETIC_BASE_SERVER_FRAME,
    DEFAULT_SYNTHETIC_CLIENT_IDS,
    FORBIDDEN_LEGACY_SELECTORS,
    LATTICE_CRATE_NAME,
    MODE_PRODUCTION,
    MODE_SYNTHETIC,
    ONE_RUN_PROTOCOL_VERSION,
    ONE_RUN_SCHEMA,
    PYTHON_COLLECTOR_SCHEMA,
    RECORD_ORDERING,
    REQUIRED_CLIENT_COUNT,
    REQUIRED_TRANSITION_COUNT,
    RUST_PROVIDER_SCHEMA,
    SPATIAL_PROVIDER_CLASS_NAME,
    STACK_LANGUAGE,
    TRANSITIONS_PER_CLIENT,
    InjectedScenario,
    InjectedTransport,
    Multires500ProofError,
    ProcessIdentity,
    ProductionTransport,
    ProofConfig,
    admit_canonical_records,
    admit_production_config,
    build_one_run_argv,
    build_verifier_evidence,
    client_ids_from_first_round,
    discover_descendant_pids,
    layout_for_flat_index,
    make_transition_record,
    process_identity_alive,
    prove_process_ids_dead,
    prove_process_records_dead,
    rust_feature_digest,
    run_proof,
    trajectory_sha256,
)
from tools.verify_multires_integration import (
    DETERMINISTIC_TRANSITION_COUNT,
    _check_deterministic_transitions,
)


ATLAS = hashlib.sha256(b"multires-500-test-atlas").hexdigest()
RUNTIME = hashlib.sha256(b"multires-500-test-runtime").hexdigest()
OBJECTIVE = hashlib.sha256(b"multires-500-test-objectives").hexdigest()

try:
    import torch
    from models.multires_policy import MultiresQ2BotPolicy
except ImportError:
    torch = None
    MultiresQ2BotPolicy = None


@pytest.fixture(autouse=True)
def _checkpoint_loader_without_local_torch(monkeypatch):
    """Keep non-checkpoint outer-gate tests runnable in minimal CI images."""
    if torch is None:
        monkeypatch.setattr(
            proof_module,
            "_validate_production_checkpoint",
            lambda *args, **kwargs: ("random", 0, "a" * 64),
        )


def _observation_with_dyn(
    first: float = 0.0,
    rust_features: list[float] | None = None,
) -> list[float]:
    features = list(rust_features if rust_features is not None else [0.1] * 24)
    observation = [0.0] * OBS_DIM
    observation[0] = float(first)
    observation[DYN.slice] = features
    return observation


def _transport(scenario: InjectedScenario | None = None) -> InjectedTransport:
    return InjectedTransport(
        atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        scenario=scenario,
    )


def _synthetic_config(**overrides) -> ProofConfig:
    base = dict(
        mode=MODE_SYNTHETIC,
        seed=7142026,
        game_seed=4242,
        divergence_game_seed=4243,
        transition_count=REQUIRED_TRANSITION_COUNT,
        policy_version=1,
        map_name="mlstage_0001",
        map_epoch=3,
        timeout_seconds=30.0,
    )
    base.update(overrides)
    return ProofConfig(**base)


def test_same_seed_determinism_and_different_seed_divergence():
    transport = _transport()
    report = run_proof(_synthetic_config(), transport=transport)

    assert report["schema"] == "q2-multires-500-transition-proof-v1"
    assert report["mode"] == MODE_SYNTHETIC
    assert report["same_seed_match"] is True
    assert report["different_seed_diverges"] is True
    assert report["deterministic_ok"] is True
    assert report["runs"][0]["trajectory_sha256"] == report["runs"][1]["trajectory_sha256"]
    assert (
        report["runs"][0]["trajectory_sha256"]
        != report["divergence_run"]["trajectory_sha256"]
    )
    assert report["runs"][0]["transition_count"] == REQUIRED_TRANSITION_COUNT
    assert report["runs"][1]["transition_count"] == REQUIRED_TRANSITION_COUNT
    # Same-seed launches are identical complete stacks, not language variants.
    assert report["runs"][0]["language"] == STACK_LANGUAGE
    assert report["runs"][1]["language"] == STACK_LANGUAGE
    assert report["runs"][0]["launch_id"] == "same_seed_run_a"
    assert report["runs"][1]["launch_id"] == "same_seed_run_b"
    assert transport.cleanup_calls >= 1


def test_synthetic_mode_is_non_admissible_and_cannot_production_pass():
    report = run_proof(_synthetic_config(), transport=_transport())

    assert report["admissible"] is False
    assert report["production_pass"] is False
    assert "non-admissible" in report["non_admissible_reason"]
    assert report["deterministic_ok"] is True
    assert report["same_seed_match"] is True


def test_production_rejects_injected_transport(tmp_path):
    config = _materialize_production_tree(tmp_path / "inj")
    with pytest.raises(Multires500ProofError, match="rejects injected transports"):
        run_proof(config, transport=_transport())


def test_production_rejects_callback_injection(tmp_path):
    config = _materialize_production_tree(tmp_path / "cb")
    config.production_collect_fn = lambda *args, **kwargs: []
    with pytest.raises(Multires500ProofError, match="callback"):
        admit_production_config(config)
    config2 = _materialize_production_tree(tmp_path / "cb2")
    config2.production_process_factory = lambda *args, **kwargs: None
    with pytest.raises(Multires500ProofError, match="injection"):
        admit_production_config(config2)


def test_verifier_evidence_is_consumable_by_verify_multires_integration():
    report = run_proof(_synthetic_config(), transport=_transport())
    evidence = report["verifier_evidence"]
    assert evidence["transition_count"] == DETERMINISTIC_TRANSITION_COUNT
    assert evidence["runs"][0]["stack"] == STACK_LANGUAGE
    assert evidence["runs"][1]["stack"] == STACK_LANGUAGE
    assert evidence["runs"][0]["fresh_subprocess"] is True
    assert evidence["runs"][1]["fresh_subprocess"] is True
    assert evidence["runs"][0]["collector"] == COLLECTOR_CLASS_NAME
    assert evidence["runs"][1]["spatial_provider"] == SPATIAL_PROVIDER_CLASS_NAME
    _check_deterministic_transitions(evidence, context={})

    rebuilt = build_verifier_evidence(
        transition_count=REQUIRED_TRANSITION_COUNT,
        run_a=type("R", (), {
            "transition_count": report["runs"][0]["transition_count"],
            "trajectory_sha256": report["runs"][0]["trajectory_sha256"],
            "launch_id": report["runs"][0]["launch_id"],
            "game_seed": report["runs"][0]["game_seed"],
            "seed": report["runs"][0]["seed"],
            "partial_admissions": 0,
            "stale_admissions": 0,
            "resync_admissions": 0,
        })(),
        run_b=type("R", (), {
            "transition_count": report["runs"][1]["transition_count"],
            "trajectory_sha256": report["runs"][1]["trajectory_sha256"],
            "launch_id": report["runs"][1]["launch_id"],
            "game_seed": report["runs"][1]["game_seed"],
            "seed": report["runs"][1]["seed"],
            "partial_admissions": 0,
            "stale_admissions": 0,
            "resync_admissions": 0,
        })(),
        divergence_run=type("R", (), {
            "transition_count": report["divergence_run"]["transition_count"],
            "trajectory_sha256": report["divergence_run"]["trajectory_sha256"],
            "launch_id": report["divergence_run"]["launch_id"],
            "game_seed": report["divergence_run"]["game_seed"],
            "seed": report["divergence_run"]["seed"],
            "partial_admissions": 0,
            "stale_admissions": 0,
            "resync_admissions": 0,
        })(),
    )
    _check_deterministic_transitions(rebuilt, context={})


def test_499_transition_count_rejected_on_config():
    with pytest.raises(Multires500ProofError, match="not the required 500"):
        run_proof(
            _synthetic_config(transition_count=499),
            transport=_transport(),
        )


def test_499_launch_records_reject_deterministic_ok():
    transport = _transport(InjectedScenario(transition_count=499))
    report = run_proof(_synthetic_config(), transport=transport)
    assert report["deterministic_ok"] is False
    assert report["production_pass"] is False
    assert report["admissible"] is False
    assert any("transition_count=499" in item for item in report["failures"])


def test_mixed_atlas_digest_rejection():
    transport = _transport(InjectedScenario(mix_atlas_digest=True))
    with pytest.raises(Multires500ProofError, match="mixed atlas digests"):
        run_proof(_synthetic_config(), transport=transport)
    assert transport.cleanup_calls >= 1


def test_mixed_map_epoch_rejection():
    transport = _transport(InjectedScenario(mix_map_epoch=True))
    with pytest.raises(Multires500ProofError, match="mixed map epochs"):
        run_proof(_synthetic_config(), transport=transport)
    assert transport.cleanup_calls >= 1


def test_mixed_policy_version_rejection():
    transport = _transport(InjectedScenario(mix_policy_version=True))
    with pytest.raises(Multires500ProofError, match="mixed policy versions"):
        run_proof(_synthetic_config(), transport=transport)
    assert transport.cleanup_calls >= 1


def test_legacy_selector_rejection():
    for selector in (
        "train.ppo",
        "models.policy",
        "public_network_thermal_bc_live_v2",
        "resume:public_network_engagement_anchor_v3",
    ):
        with pytest.raises(Multires500ProofError, match="legacy"):
            run_proof(
                _synthetic_config(legacy_selector=selector),
                transport=_transport(),
            )


def test_partial_batch_admission_rejection():
    transport = _transport(InjectedScenario(force_partial_admissions=1))
    with pytest.raises(Multires500ProofError, match="partial batch"):
        run_proof(_synthetic_config(), transport=transport)
    assert transport.cleanup_calls >= 1


def test_stale_and_resync_admission_rejection():
    with pytest.raises(Multires500ProofError, match="stale admissions"):
        run_proof(
            _synthetic_config(),
            transport=_transport(InjectedScenario(force_stale_admissions=2)),
        )
    with pytest.raises(Multires500ProofError, match="resync admissions"):
        run_proof(
            _synthetic_config(),
            transport=_transport(InjectedScenario(force_resync_admissions=1)),
        )


def test_process_failure_triggers_cleanup():
    transport = _transport(
        InjectedScenario(
            fail_launch_id="same_seed_run_b",
            fail_message="injected process crash",
        )
    )
    with pytest.raises(Multires500ProofError, match="injected process crash"):
        run_proof(_synthetic_config(), transport=transport)
    assert transport.cleanup_calls >= 1
    assert transport._live_pids == []


def test_timeout_triggers_cleanup():
    transport = _transport(
        InjectedScenario(timeout_launch_id="same_seed_run_a")
    )
    with pytest.raises(Multires500ProofError, match="timed out"):
        run_proof(_synthetic_config(), transport=transport)
    assert transport.cleanup_calls >= 1
    assert transport._live_pids == []


def test_tool_module_does_not_import_legacy_trainers():
    source = Path(__file__).resolve().parents[1].joinpath(
        "tools/run_multires_500_transition_proof.py"
    ).read_text(encoding="utf-8")
    assert "train.ppo" in source  # forbidden list mentions it
    assert "import train.ppo" not in source
    assert "from train.ppo" not in source
    assert "import models.policy" not in source
    assert "from models.policy" not in source
    assert "from train import ppo" not in source
    assert "from models import policy" not in source
    isolated = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import tools.run_multires_500_transition_proof; "
                "assert 'train.ppo' not in sys.modules; "
                "assert 'models.policy' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert isolated.returncode == 0, isolated.stdout
    for name in FORBIDDEN_LEGACY_SELECTORS:
        assert f"import {name}" not in source
        assert f"from {name}" not in source


def test_trajectory_hash_is_byte_stable():
    records = [
        make_transition_record(
            index=index,
            observation=_observation_with_dyn(float(index)),
            action=[0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            reward=0.25,
            client_id="c0",
            server_frame=10 + index,
            batch_round_id=index,
            policy_version=1,
            map_name="mlstage_0001",
            map_epoch=1,
            atlas_sha256=ATLAS,
            runtime_manifest_sha256=RUNTIME,
            rust_features=[0.1] * 24,
        )
        for index in range(3)
    ]
    first = trajectory_sha256(records)
    second = trajectory_sha256(records)
    assert first == second
    assert len(first) == 64
    mutated = list(records)
    mutated[1] = make_transition_record(
        index=1,
        observation=_observation_with_dyn(9.0),
        action=[0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        reward=0.25,
        client_id="c0",
        server_frame=11,
        batch_round_id=1,
        policy_version=1,
        map_name="mlstage_0001",
        map_epoch=1,
        atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        rust_features=[0.1] * 24,
    )
    assert trajectory_sha256(mutated) != first


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _b4_runtime_evidence(atlas_sha256: str) -> dict:
    descriptor = {
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "factual_dim": 198,
        "dyn_dim": 24,
        "recovery_dim": 16,
        "objective_count": 4,
        "objective_dim": 15,
        "total_dim": OBS_DIM,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_magic": B4_OBSERVATION_MAGIC,
        "action_magic": B4_ACTION_MAGIC,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": B4_TEACHER_VERSION,
        "rollout_telemetry_schema": B4_ROLLOUT_SCHEMA,
        "logical_action_dim": ACTION_DIM,
        "action_cardinalities": {
            "vertical_intent": 3,
            "fire": 2,
            "hook": 4,
            "weapon": 10,
        },
        "teacher_privileged_packing": "physically-separate-qm3c-v2",
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
    }
    evidence = adapt_b4_observation_descriptor(
        descriptor, atlas_sha256=atlas_sha256, teacher_field_violations=0
    )
    sealed = validate_runtime_evidence(
        evidence, expected_atlas_sha256=atlas_sha256
    )
    evidence["runtime_manifest_sha256"] = sealed.runtime_manifest_sha256
    return evidence


FAKE_TRAINER_SCRIPT = textwrap.dedent(
    r'''
    #!/usr/bin/env python3
    """Test-only operational one-run trainer fixture for M6d subprocess proof."""
    from __future__ import annotations
    import argparse
    import hashlib
    import json
    import os
    import subprocess as sp
    import sys
    import time
    from pathlib import Path

    ONE_RUN_SCHEMA = "q2-multires-one-run-proof-v1"
    ONE_RUN_PROTOCOL_VERSION = 1
    TRAJECTORY_DOMAIN = b"q2-multires-500-transition-trajectory-v1\0"
    OBS_DIM = 298
    ACTION_DIM = 8
    REQUIRED_CLIENT_COUNT = 4
    TRANSITIONS_PER_CLIENT = 125
    RUST_PROVIDER_SCHEMA = "q2-multires-spatial-provider-v1"
    PYTHON_COLLECTOR_SCHEMA = "q2-multires-collected-rollout-v1"
    CLIENT_IDS = tuple(f"mrproof-{index:02d}" for index in range(REQUIRED_CLIENT_COUNT))
    PROCESS_ROLES = ("q2ded", *(f"network-client-{index:02d}" for index in range(4)))
    BASE_SERVER_FRAME = 4827

    def canonical_bytes(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def parse_args(argv):
        p = argparse.ArgumentParser()
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--game_seed", type=int, required=True)
        p.add_argument("--q2ded", required=True)
        p.add_argument("--client_binary", required=True)
        p.add_argument("--runtime_root", required=True)
        p.add_argument("--bundle_manifest", required=True)
        p.add_argument("--objectives", required=True)
        p.add_argument("--atlas_bin", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--training_manifest", required=True)
        p.add_argument("--runtime_evidence", required=True)
        p.add_argument("--transition_count", type=int, required=True)
        p.add_argument("--policy_version", type=int, required=True)
        p.add_argument("--map_epoch", type=int, required=True)
        p.add_argument("--map_name", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--launch_id", required=True)
        p.add_argument("--expected_atlas_sha256", required=True)
        p.add_argument("--expected_runtime_manifest_sha256", required=True)
        return p.parse_args(argv)

    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def identity_digest(**fields):
        payload = {
            "atlas_sha256": fields["atlas_sha256"],
            "batch_round_id": int(fields["batch_round_id"]),
            "client_id": str(fields["client_id"]),
            "map_epoch": int(fields["map_epoch"]),
            "map_name": str(fields["map_name"]),
            "policy_version": int(fields["policy_version"]),
            "runtime_manifest_sha256": fields["runtime_manifest_sha256"],
            "server_frame": int(fields["server_frame"]),
        }
        return hashlib.sha256(canonical_bytes(payload)).hexdigest()

    def feature_digest(features):
        return hashlib.sha256(canonical_bytes({"features": list(features)})).hexdigest()

    def process_start_ticks(pid):
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        return int(raw[raw.rfind(")") + 2:].split()[19])

    def build_records(args, atlas_sha, runtime_sha, count, mode):
        material = hashlib.sha256(
            f"seed={args.seed}:game={args.game_seed}:map={args.map_name}".encode()
        ).digest()
        records = []
        for index in range(count):
            step_material = hashlib.sha256(material + index.to_bytes(4, "little")).digest()
            obs = [
                ((step_material[i % len(step_material)] / 255.0) * 2.0 - 1.0)
                for i in range(OBS_DIM)
            ]
            obs[0] = float(args.game_seed % 997) / 997.0
            obs[1] = float(args.seed % 991) / 991.0
            obs[2] = float(index) / max(count, 1)
            act = [0.0] * ACTION_DIM
            act[0] = float(step_material[0] % 11) / 10.0 - 0.5
            act[4] = float(step_material[1] % 3)
            act[5] = float(step_material[2] % 2)
            reward = float(step_material[3]) / 255.0
            rust_features = [float(step_material[4 + (i % 12)]) / 255.0 for i in range(24)]
            obs[198:222] = rust_features
            time_step = index // REQUIRED_CLIENT_COUNT
            client_index = index % REQUIRED_CLIENT_COUNT
            client_id = CLIENT_IDS[client_index]
            server_frame = BASE_SERVER_FRAME + time_step
            batch_round_id = time_step
            if mode == "reorder" and index in (4, 5):
                # First round binds order; later round swaps two minor-axis slots
                # without creating a duplicate client/frame/round identity.
                client_id = CLIENT_IDS[1] if index == 4 else CLIENT_IDS[0]
            if mode == "duplicate_client_ids" and index == 1:
                # First round has only three unique IDs.
                client_id = CLIENT_IDS[0]
            if mode == "frame_skew" and index == 5:
                # Same time step, one client off by +1 frame.
                server_frame = BASE_SERVER_FRAME + time_step + 1
            if mode == "frame_jump" and time_step >= 3:
                # Non-unit advance after round 2 (skip a frame forever after).
                server_frame = BASE_SERVER_FRAME + time_step + 1
            if mode == "bad_identity" and index == 0:
                # Corrupt later after building.
                pass
            if mode == "unbound_record" and index == 0:
                record_atlas = hashlib.sha256(atlas_sha.encode() + b"-unbound").hexdigest()
            else:
                record_atlas = atlas_sha
            if mode == "duplicate" and index == 1:
                # Force duplicate of record 0 identity fields after layout break.
                time_step = 0
                client_id = CLIENT_IDS[0]
                server_frame = BASE_SERVER_FRAME
                batch_round_id = 0
            rec = {
                "index": index if mode != "reindex" else (index + 1 if index == 0 else index),
                "observation": obs,
                "action": act,
                "reward": reward,
                "client_id": client_id,
                "server_frame": server_frame,
                "batch_round_id": batch_round_id,
                "policy_version": int(args.policy_version),
                "map_name": str(args.map_name),
                "map_epoch": int(args.map_epoch),
                "atlas_sha256": record_atlas,
                "runtime_manifest_sha256": runtime_sha,
                "rust_provider_schema": RUST_PROVIDER_SCHEMA,
                "rust_features": rust_features,
                "rust_feature_digest": feature_digest(rust_features),
                "python_collector_schema": PYTHON_COLLECTOR_SCHEMA,
            }
            rec["identity_digest"] = identity_digest(
                client_id=rec["client_id"],
                server_frame=rec["server_frame"],
                batch_round_id=rec["batch_round_id"],
                policy_version=rec["policy_version"],
                map_name=rec["map_name"],
                map_epoch=rec["map_epoch"],
                atlas_sha256=rec["atlas_sha256"],
                runtime_manifest_sha256=rec["runtime_manifest_sha256"],
            )
            if mode == "bad_identity" and index == 0:
                rec["identity_digest"] = hashlib.sha256(b"wrong-identity").hexdigest()
            if mode == "noncanonical_schema" and index == 0:
                rec["rust_provider_schema"] = "legacy-provider"
            records.append(rec)
        if mode == "extra_record":
            records.append(dict(records[-1]))
            records[-1]["index"] = len(records) - 1
            records[-1]["server_frame"] = records[-1]["server_frame"] + 99
            records[-1]["batch_round_id"] = records[-1]["batch_round_id"] + 99
            records[-1]["identity_digest"] = identity_digest(
                client_id=records[-1]["client_id"],
                server_frame=records[-1]["server_frame"],
                batch_round_id=records[-1]["batch_round_id"],
                policy_version=records[-1]["policy_version"],
                map_name=records[-1]["map_name"],
                map_epoch=records[-1]["map_epoch"],
                atlas_sha256=records[-1]["atlas_sha256"],
                runtime_manifest_sha256=records[-1]["runtime_manifest_sha256"],
            )
        if mode == "truncated_record" and records:
            # Truncate after assembly so trajectory hashing still runs.
            del records[0]["observation"]
        if mode == "missing_records":
            return None
        if mode == "empty_records":
            return []
        return records

    def trajectory_from_records(records):
        traj = hashlib.sha256()
        traj.update(TRAJECTORY_DOMAIN)
        traj.update(int(len(records)).to_bytes(8, "little", signed=False))
        for record in records:
            # Match TransitionRecord.to_mapping field set (no rust_features).
            if "observation" not in record:
                # Truncated fixtures still need a stable placeholder digest.
                traj.update(b"truncated\0")
                continue
            mapping = {
                "action": record["action"],
                "atlas_sha256": record["atlas_sha256"],
                "batch_round_id": record["batch_round_id"],
                "client_id": record["client_id"],
                "identity_digest": record["identity_digest"],
                "index": record["index"],
                "map_epoch": record["map_epoch"],
                "map_name": record["map_name"],
                "observation": record["observation"],
                "policy_version": record["policy_version"],
                "python_collector_schema": record["python_collector_schema"],
                "reward": record["reward"],
                "runtime_manifest_sha256": record["runtime_manifest_sha256"],
                "rust_feature_digest": record["rust_feature_digest"],
                "rust_provider_schema": record["rust_provider_schema"],
                "server_frame": record["server_frame"],
            }
            traj.update(canonical_bytes(mapping))
            traj.update(b"\0")
        return traj.hexdigest()

    def main(argv=None):
        mode = os.environ.get("FAKE_TRAINER_MODE", "ok")
        if mode == "timeout":
            time.sleep(3600)
            return 0
        if mode == "timeout_detached_child":
            # Spawn a detached sleeper (new session) so killpg on the trainer
            # alone would miss it; the proof must discover via /proc PPID walk.
            pid_file = os.environ.get("FAKE_TRAINER_CHILD_PID_FILE")
            child = sp.Popen(
                [sys.executable, "-c", "import time; time.sleep(3600)"],
                stdin=sp.DEVNULL,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
                start_new_session=True,
            )
            if pid_file:
                Path(pid_file).write_text(str(child.pid), encoding="utf-8")
            time.sleep(3600)
            return 0
        if mode == "legacy":
            print("legacy train.ppo models.policy", file=sys.stderr)
            return 3
        if mode == "crash":
            print("injected crash", file=sys.stderr)
            return 9

        args = parse_args(argv)
        reported_live_child = None
        if mode == "live_reported_detached":
            reported_live_child = sp.Popen(
                [sys.executable, "-c", "import time; time.sleep(3600)"],
                stdin=sp.DEVNULL,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
                start_new_session=True,
            )
            pid_file = os.environ.get("FAKE_TRAINER_CHILD_PID_FILE")
            if pid_file:
                Path(pid_file).write_text(
                    str(reported_live_child.pid), encoding="utf-8"
                )
        objectives = json.loads(Path(args.objectives).read_text(encoding="utf-8"))
        training = json.loads(Path(args.training_manifest).read_text(encoding="utf-8"))
        runtime_ev = json.loads(Path(args.runtime_evidence).read_text(encoding="utf-8"))
        objective_identity = objectives.get("objective_identity_sha256") or file_sha256(args.objectives)
        training_sha = training.get("sha256") or hashlib.sha256(canonical_bytes(training)).hexdigest()
        checkpoint_sha = file_sha256(args.checkpoint)
        atlas_sha = args.expected_atlas_sha256
        runtime_sha = args.expected_runtime_manifest_sha256

        count = int(args.transition_count)
        if mode == "count_499":
            count = 499

        if mode == "wrong_provider":
            provider = "LegacySpatialProvider"
            rust_schema = "legacy-provider"
            lattice = "old_lattice"
            collector = "MultiresSynchronousCollector"
            py_schema = "q2-multires-collected-rollout-v1"
        elif mode == "wrong_collector":
            provider = "RustAtlasSpatialProvider"
            rust_schema = "q2-multires-spatial-provider-v1"
            lattice = "q2_lattice"
            collector = "LegacyCollector"
            py_schema = "legacy-collector"
        else:
            provider = "RustAtlasSpatialProvider"
            rust_schema = "q2-multires-spatial-provider-v1"
            lattice = "q2_lattice"
            collector = "MultiresSynchronousCollector"
            py_schema = "q2-multires-collected-rollout-v1"

        received = {
            "seed": int(args.seed),
            "game_seed": int(args.game_seed),
            "q2ded": str(Path(args.q2ded).resolve()),
            "client_binary": str(Path(args.client_binary).resolve()),
            "runtime_root": str(Path(args.runtime_root).resolve()),
            "bundle_manifest": str(Path(args.bundle_manifest).resolve()),
            "objectives": str(Path(args.objectives).resolve()),
            "atlas_bin": str(Path(args.atlas_bin).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "training_manifest": str(Path(args.training_manifest).resolve()),
            "runtime_evidence": str(Path(args.runtime_evidence).resolve()),
            "transition_count": int(args.transition_count),
            "policy_version": int(args.policy_version),
            "map_epoch": int(args.map_epoch),
            "map_name": str(args.map_name),
            "out": str(Path(args.out).resolve()),
            "launch_id": str(args.launch_id),
            "expected_atlas_sha256": str(args.expected_atlas_sha256),
            "expected_runtime_manifest_sha256": str(args.expected_runtime_manifest_sha256),
        }
        if mode == "ignore_seed":
            received.pop("seed")
        if mode == "rewrite_game_seed":
            received["game_seed"] = int(args.game_seed) + 99

        # Digest-only modes intentionally omit/corrupt records.
        if mode == "digest_only":
            material = hashlib.sha256(
                f"seed={args.seed}:game={args.game_seed}:map={args.map_name}".encode()
            ).digest()
            traj = hashlib.sha256(TRAJECTORY_DOMAIN)
            traj.update(int(count).to_bytes(8, "little", signed=False))
            traj.update(material)
            traj.update(count.to_bytes(4, "little"))
            records = None
            trajectory = traj.hexdigest()
        else:
            records = build_records(args, atlas_sha, runtime_sha, count, mode)
            if records is None:
                trajectory = hashlib.sha256(b"digest-only-missing-records").hexdigest()
            elif len(records) == 0:
                trajectory = hashlib.sha256(b"digest-only-empty-records").hexdigest()
            else:
                # count_499 still emits 499 records; production rejects non-500.
                if mode == "count_499":
                    # Dense single-client-style indices already in build via 4-client
                    # layout only when count==500; for 499 use linear layout.
                    linear = []
                    material = hashlib.sha256(
                        f"seed={args.seed}:game={args.game_seed}:map={args.map_name}".encode()
                    ).digest()
                    for index in range(count):
                        step_material = hashlib.sha256(
                            material + index.to_bytes(4, "little")
                        ).digest()
                        obs = [
                            ((step_material[i % len(step_material)] / 255.0) * 2.0 - 1.0)
                            for i in range(OBS_DIM)
                        ]
                        act = [0.0] * ACTION_DIM
                        rust_features = [
                            float(step_material[4 + (i % 12)]) / 255.0 for i in range(24)
                        ]
                        obs[198:222] = rust_features
                        rec = {
                            "index": index,
                            "observation": obs,
                            "action": act,
                            "reward": float(step_material[3]) / 255.0,
                            "client_id": CLIENT_IDS[0],
                            "server_frame": BASE_SERVER_FRAME + index,
                            "batch_round_id": index,
                            "policy_version": int(args.policy_version),
                            "map_name": str(args.map_name),
                            "map_epoch": int(args.map_epoch),
                            "atlas_sha256": atlas_sha,
                            "runtime_manifest_sha256": runtime_sha,
                            "rust_provider_schema": RUST_PROVIDER_SCHEMA,
                            "rust_features": rust_features,
                            "rust_feature_digest": feature_digest(rust_features),
                            "python_collector_schema": PYTHON_COLLECTOR_SCHEMA,
                        }
                        rec["identity_digest"] = identity_digest(
                            client_id=rec["client_id"],
                            server_frame=rec["server_frame"],
                            batch_round_id=rec["batch_round_id"],
                            policy_version=rec["policy_version"],
                            map_name=rec["map_name"],
                            map_epoch=rec["map_epoch"],
                            atlas_sha256=rec["atlas_sha256"],
                            runtime_manifest_sha256=rec["runtime_manifest_sha256"],
                        )
                        linear.append(rec)
                    records = linear
                if mode == "wrong_traj" and records:
                    trajectory = hashlib.sha256(b"deliberately-wrong-trajectory").hexdigest()
                else:
                    trajectory = trajectory_from_records(records) if records else hashlib.sha256(b"none").hexdigest()

        try:
            pid_ceiling = int(Path("/proc/sys/kernel/pid_max").read_text().strip())
        except (OSError, ValueError):
            pid_ceiling = 4_194_304
        process_records = [
            {"role": role, "pid": pid_ceiling + 100 + index, "start_ticks": 100 + index}
            for index, role in enumerate(PROCESS_ROLES)
        ]
        if reported_live_child is not None:
            process_records[0] = {
                "role": "q2ded",
                "pid": reported_live_child.pid,
                "start_ticks": process_start_ticks(reported_live_child.pid),
            }
        process_ids = [record["pid"] for record in process_records]

        payload = {
            "schema": ONE_RUN_SCHEMA if mode != "bad_schema" else "not-a-one-run",
            "protocol_version": ONE_RUN_PROTOCOL_VERSION,
            "synthetic": mode == "synthetic",
            "legacy": mode == "legacy_flag",
            "collector": collector,
            "python_collector_schema": py_schema,
            "spatial_provider": provider,
            "rust_provider_schema": rust_schema,
            "lattice_crate": lattice,
            "transition_count": count if mode != "count_lie" else 500,
            "trajectory_sha256": trajectory,
            "seed": int(args.seed),
            "game_seed": int(args.game_seed),
            "policy_version": int(args.policy_version),
            "map_name": str(args.map_name),
            "map_epoch": int(args.map_epoch),
            "atlas_sha256": atlas_sha,
            "runtime_manifest_sha256": runtime_sha,
            "objective_identity_sha256": objective_identity,
            "training_manifest_sha256": training_sha,
            "checkpoint_sha256": checkpoint_sha,
            "b4_protocol_generation": int(runtime_ev.get("protocol_generation", 2)),
            "qm3c_causal_magic": int(runtime_ev.get("causal_magic", 0)),
            "client_wire_version": int(runtime_ev.get("client_wire_version", 0)),
            "policy_generation": "multires-atlas-policy-v1",
            "checkpoint_format": "q2-multires-attested-checkpoint-v1",
            "checkpoint_training_step": 0,
            "partial_admissions": 1 if mode == "partial" else 0,
            "stale_admissions": 1 if mode == "stale" else 0,
            "resync_admissions": 1 if mode == "resync" else 0,
            "process_records": process_records,
            "terminated_process_records": [dict(record) for record in process_records],
            "process_ids": process_ids,
            "launched_process_ids": list(process_ids),
            "terminated_process_ids": list(process_ids),
            "received_inputs": received,
        }
        if mode == "mixed_atlas":
            payload["atlas_sha256"] = hashlib.sha256(atlas_sha.encode() + b"x").hexdigest()
        if mode == "no_process_ids":
            payload["process_ids"] = []
        if mode == "missing_process_records":
            payload.pop("process_records")
        if mode == "malformed_process_records":
            payload["process_records"][0]["start_ticks"] = "not-an-integer"
        if mode == "duplicate_process_records":
            payload["process_records"][1]["pid"] = payload["process_records"][0]["pid"]
        if mode == "training_digest_lie":
            payload["training_manifest_sha256"] = hashlib.sha256(b"lie").hexdigest()
        if mode == "digest_only":
            # Explicitly omit records.
            pass
        elif mode == "missing_records":
            pass
        elif mode == "empty_records":
            payload["records"] = []
        else:
            payload["records"] = records
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(canonical_bytes(payload))
        return 0

    if __name__ == "__main__":
        sys.exit(main())
    '''
).lstrip()


def _write_fake_trainer(path: Path) -> Path:
    _write_executable(path, FAKE_TRAINER_SCRIPT)
    return path.resolve()


def _materialize_production_tree(
    root: Path,
    *,
    legacy_wire: bool = False,
    trainer_path: Path | None = None,
    include_trainer: bool = True,
) -> ProofConfig:
    root.mkdir(parents=True, exist_ok=True)
    runtime_root = root / "runtime"
    runtime_root.mkdir()
    optimizer_configuration = {
        "class": "torch.optim.Adam",
        "learning_rate": 3.0e-4,
        "kwargs": {},
    }
    (runtime_root / "multires-one-run.json").write_text(
        json.dumps({"optimizer": optimizer_configuration}, sort_keys=True),
        encoding="utf-8",
    )
    q2ded = root / "q2ded"
    client = root / "q2"
    _write_executable(q2ded)
    _write_executable(client)

    atlas_bytes = b"ATLAS-BIN-" + ATLAS.encode()
    atlas_bin = root / "mlstage_0001.atlas.bin"
    atlas_bin.write_bytes(atlas_bytes)
    atlas_sha256 = hashlib.sha256(atlas_bytes).hexdigest()

    bundle_manifest = root / "mlstage_0001.bundle.manifest.json"
    bundle_manifest.write_text(json.dumps({
        "bundle_version": 3,
        "artifact_state": "admitted",
        "name": "mlstage_0001",
        "atlas_sha256": atlas_sha256,
        "files": {},
        "analysis_files": {},
        "file_sizes": {},
        "analysis_file_sizes": {},
    }, sort_keys=True), encoding="utf-8")

    objectives = root / "mlstage_0001.objectives.json"
    objectives.write_text(json.dumps({
        "map_name": "mlstage_0001",
        "objective_identity_sha256": OBJECTIVE,
        "objectives": [{"class": "health", "id": 1}],
    }, sort_keys=True), encoding="utf-8")

    training_manifest = root / "training_manifest.json"
    training_configuration = MultiresTrainingConfiguration.create(
        reward={"version": 1},
        guide_dropout={"global_drop_probability": 0.0},
        ppo={"epochs": 2},
    )
    training_body = training_configuration.to_mapping()
    training_body["sha256"] = training_configuration.sha256
    training_manifest.write_text(json.dumps(training_body, sort_keys=True), encoding="utf-8")

    runtime_evidence = root / "runtime_evidence.json"
    evidence = _b4_runtime_evidence(atlas_sha256)
    if legacy_wire:
        evidence["client_wire_version"] = LEGACY_CLIENT_WIRE_VERSION
        evidence["observation_magic"] = LEGACY_OBSERVATION_MAGIC
        evidence["rollout_schema"] = LEGACY_ROLLOUT_SCHEMA
    runtime_evidence.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")

    checkpoint = root / "fresh.pt"
    if torch is not None:
        torch.manual_seed(17)
        checkpoint_policy = MultiresQ2BotPolicy()
        checkpoint_optimizer = torch.optim.Adam(
            checkpoint_policy.parameters(),
            lr=float(optimizer_configuration["learning_rate"]),
        )
        save_attested_checkpoint(
            checkpoint,
            checkpoint_policy,
            atlas_sha256=atlas_sha256,
            runtime_manifest_sha256=evidence["runtime_manifest_sha256"],
            training_config=training_configuration,
            initialization="random",
            training_step=0,
            optimizer=checkpoint_optimizer,
        )
    else:
        checkpoint.write_bytes(
            CHECKPOINT_FORMAT.encode("utf-8") + b"\0" + b"\x00" * 64
        )

    evidence_dir = root / "one_run_evidence"
    evidence_dir.mkdir(exist_ok=True)
    out_path = (root / "deterministic_transitions.json").resolve()

    trainer_executable = None
    if include_trainer:
        trainer_executable = (
            trainer_path.resolve()
            if trainer_path is not None
            else _write_fake_trainer(root / "fake_multires_one_run.py")
        )

    return ProofConfig(
        mode=MODE_PRODUCTION,
        seed=7142026,
        game_seed=4242,
        divergence_game_seed=4243,
        q2ded=q2ded.resolve(),
        client_binary=client.resolve(),
        runtime_root=runtime_root.resolve(),
        bundle_manifest=bundle_manifest.resolve(),
        objectives=objectives.resolve(),
        atlas_bin=atlas_bin.resolve(),
        checkpoint=checkpoint.resolve(),
        training_manifest=training_manifest.resolve(),
        runtime_evidence=runtime_evidence.resolve(),
        map_name="mlstage_0001",
        map_epoch=3,
        evidence_dir=evidence_dir.resolve(),
        out_path=out_path,
        trainer_executable=trainer_executable,
        timeout_seconds=15.0,
    )


def test_production_admission_requires_explicit_paths(tmp_path):
    config = ProofConfig(
        mode=MODE_PRODUCTION,
        seed=1,
        game_seed=2,
        divergence_game_seed=3,
    )
    with pytest.raises(Multires500ProofError, match="requires explicit|trainer"):
        admit_production_config(config)


def test_production_admission_requires_trainer_executable(tmp_path):
    config = _materialize_production_tree(tmp_path / "notrain", include_trainer=False)
    with pytest.raises(Multires500ProofError, match="trainer_executable|trainer_command"):
        admit_production_config(config)


def test_production_admission_accepts_admitted_bundle_and_b4_qm3c(tmp_path):
    config = _materialize_production_tree(tmp_path / "prod")
    admission = admit_production_config(config)
    assert admission.bundle_version == 3
    assert admission.artifact_state == "admitted"
    assert admission.b4_protocol_generation == B4_PROTOCOL_GENERATION
    assert admission.qm3c_causal_magic == B4_CAUSAL_MAGIC
    assert admission.client_wire_version == B4_CLIENT_WIRE_VERSION
    assert admission.policy_generation == POLICY_GENERATION
    assert admission.checkpoint_format == CHECKPOINT_FORMAT
    assert admission.checkpoint_initialization == "random"
    assert admission.checkpoint_training_step == 0
    assert len(admission.checkpoint_lineage_root_sha256) == 64
    assert len(admission.atlas_sha256) == 64
    assert admission.trainer_argv_prefix[0] == str(config.trainer_executable)


def test_production_rejects_training_manifest_digest_lie(tmp_path):
    config = _materialize_production_tree(tmp_path / "training-digest-lie")
    payload = json.loads(config.training_manifest.read_text(encoding="utf-8"))
    payload["sha256"] = hashlib.sha256(b"declared-training-lie").hexdigest()
    config.training_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Multires500ProofError, match="declared SHA-256"):
        admit_production_config(config)


@pytest.mark.skipif(torch is None, reason="real checkpoint validation requires torch")
def test_marker_and_sidecar_cannot_impersonate_real_checkpoint(tmp_path):
    config = _materialize_production_tree(tmp_path / "marker-sidecar-lie")
    config.checkpoint.write_bytes(
        CHECKPOINT_FORMAT.encode("utf-8") + b"\0" + b"marker-only"
    )
    config.checkpoint.with_suffix(".attestation.json").write_text(
        json.dumps({
            "checkpoint_format": CHECKPOINT_FORMAT,
            "initialization": "random",
            "training_step": 0,
            "lineage_root_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(Multires500ProofError, match="real attested weights-only"):
        admit_production_config(config)


@pytest.mark.skipif(torch is None, reason="real checkpoint validation requires torch")
def test_real_checkpoint_capability_is_loaded_not_inferred(tmp_path):
    config = _materialize_production_tree(tmp_path / "real-checkpoint")
    admission = admit_production_config(config)
    assert admission.checkpoint_initialization == "random"
    assert admission.checkpoint_training_step == 0
    assert admission.checkpoint_lineage_root_sha256 != "a" * 64


def test_production_rejects_legacy_wire_generations(tmp_path):
    config = _materialize_production_tree(tmp_path / "legacy", legacy_wire=True)
    with pytest.raises(Multires500ProofError, match="legacy"):
        admit_production_config(config)


def test_production_rejects_non_admitted_bundle(tmp_path):
    config = _materialize_production_tree(tmp_path / "built")
    payload = json.loads(config.bundle_manifest.read_text(encoding="utf-8"))
    payload["artifact_state"] = "built"
    config.bundle_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Multires500ProofError, match="only 'admitted'"):
        admit_production_config(config)


def test_production_rejects_legacy_selector_before_launch(tmp_path):
    config = _materialize_production_tree(tmp_path / "legacy-sel")
    config.legacy_selector = "train.ppo"
    with pytest.raises(Multires500ProofError, match="legacy"):
        admit_production_config(config)


def test_production_rejects_relative_trainer(tmp_path):
    config = _materialize_production_tree(tmp_path / "rel")
    config.trainer_executable = Path("relative_trainer")
    with pytest.raises(Multires500ProofError, match="absolute"):
        admit_production_config(config)


def test_production_rejects_missing_trainer_binary(tmp_path):
    config = _materialize_production_tree(tmp_path / "miss")
    config.trainer_executable = (tmp_path / "miss" / "does_not_exist").resolve()
    with pytest.raises(Multires500ProofError, match="does not exist"):
        admit_production_config(config)


def test_build_one_run_argv_handshake(tmp_path):
    config = _materialize_production_tree(tmp_path / "argv")
    admission = admit_production_config(config)
    out = (admission.evidence_dir / "one_run_test.json").resolve()
    request = type("Req", (), {
        "launch_id": "same_seed_run_a",
        "seed": config.seed,
        "game_seed": config.game_seed,
        "language": STACK_LANGUAGE,
        "transition_count": 500,
        "policy_version": 1,
        "map_name": admission.map_name,
        "map_epoch": config.map_epoch,
        "atlas_sha256": admission.atlas_sha256,
        "runtime_manifest_sha256": admission.runtime_manifest_sha256,
        "timeout_seconds": 5.0,
    })()
    argv = build_one_run_argv(
        trainer_argv_prefix=admission.trainer_argv_prefix,
        request=request,
        admission=admission,
        out_path=out,
    )
    assert argv[0] == str(config.trainer_executable)
    joined = " ".join(argv)
    for flag in (
        "--seed",
        "--game_seed",
        "--q2ded",
        "--client_binary",
        "--runtime_root",
        "--bundle_manifest",
        "--objectives",
        "--atlas_bin",
        "--checkpoint",
        "--training_manifest",
        "--runtime_evidence",
        "--transition_count",
        "--policy_version",
        "--map_epoch",
        "--map_name",
        "--out",
        "--launch_id",
        "--expected_atlas_sha256",
        "--expected_runtime_manifest_sha256",
    ):
        assert flag in argv
    assert "500" in argv
    assert str(admission.q2ded) in argv
    assert "shell=True" not in joined


def test_production_pass_with_fake_executable_subprocess(tmp_path):
    """End-to-end production path: real subprocess argv handshake + three launches."""
    config = _materialize_production_tree(tmp_path / "prod-pass")
    report = run_proof(config)
    assert report["mode"] == MODE_PRODUCTION
    assert report["admissible"] is True
    assert report["production_pass"] is True
    assert report["deterministic_ok"] is True
    assert report["same_seed_match"] is True
    assert report["different_seed_diverges"] is True
    assert report["runs"][0]["language"] == STACK_LANGUAGE
    assert report["runs"][1]["language"] == STACK_LANGUAGE
    assert report["collector"] == COLLECTOR_CLASS_NAME
    assert report["spatial_provider"] == SPATIAL_PROVIDER_CLASS_NAME
    assert report["lattice_crate"] == LATTICE_CRATE_NAME
    assert report["python_collector_schema"] == PYTHON_COLLECTOR_SCHEMA
    assert report["rust_provider_schema"] == RUST_PROVIDER_SCHEMA
    assert report["one_run_schema"] == ONE_RUN_SCHEMA
    assert report["one_run_protocol_version"] == ONE_RUN_PROTOCOL_VERSION
    _check_deterministic_transitions(report["verifier_evidence"], context={})
    assert config.out_path is not None and config.out_path.is_file()
    # Three per-launch artifacts from identical stack protocol.
    assert (config.evidence_dir / "one_run_same_seed_run_a.json").is_file()
    assert (config.evidence_dir / "one_run_same_seed_run_b.json").is_file()
    assert (config.evidence_dir / "one_run_divergence_game_seed.json").is_file()


def test_production_pass_writes_verifier_evidence_file(tmp_path):
    config = _materialize_production_tree(tmp_path / "prod-out")
    report = run_proof(config)
    assert config.out_path.is_file()
    payload = json.loads(config.out_path.read_text(encoding="utf-8"))
    assert payload["transition_count"] == 500
    assert payload == report["verifier_evidence"]
    _check_deterministic_transitions(payload, context={})


def test_production_rejects_unattested_trainer_schema(tmp_path):
    config = _materialize_production_tree(tmp_path / "bad-schema")
    env = os.environ.copy()
    env["FAKE_TRAINER_MODE"] = "bad_schema"
    # Run with env for the child subprocesses via monkeypatch of environ.
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "bad_schema"
    try:
        with pytest.raises(Multires500ProofError, match="one-run schema"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_synthetic_trainer_output(tmp_path):
    config = _materialize_production_tree(tmp_path / "synth")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "synthetic"
    try:
        with pytest.raises(Multires500ProofError, match="synthetic"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_499_from_trainer(tmp_path):
    config = _materialize_production_tree(tmp_path / "c499")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "count_499"
    try:
        with pytest.raises(Multires500ProofError, match="transition_count=499|proof failed"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_timeout_cleanup(tmp_path):
    config = _materialize_production_tree(tmp_path / "timeout")
    config.timeout_seconds = 0.3
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "timeout"
    try:
        with pytest.raises(Multires500ProofError, match="timed out"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old
    # Explicit cleanup accounting after failure.
    admission = admit_production_config(config)
    transport = ProductionTransport(admission)
    report = transport.cleanup()
    assert report.orphan_processes == 0


def test_production_rejects_wrong_provider(tmp_path):
    config = _materialize_production_tree(tmp_path / "prov")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "wrong_provider"
    try:
        with pytest.raises(Multires500ProofError, match="spatial_provider|lattice"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_wrong_collector(tmp_path):
    config = _materialize_production_tree(tmp_path / "coll")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "wrong_collector"
    try:
        with pytest.raises(Multires500ProofError, match="collector"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_ignored_required_input(tmp_path):
    config = _materialize_production_tree(tmp_path / "ignore")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "ignore_seed"
    try:
        with pytest.raises(Multires500ProofError, match="ignored required input"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_rewritten_input(tmp_path):
    config = _materialize_production_tree(tmp_path / "rewrite")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "rewrite_game_seed"
    try:
        with pytest.raises(Multires500ProofError, match="ignored or rewrote"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_partial_stale_resync(tmp_path):
    for mode, match in (
        ("partial", "partial"),
        ("stale", "stale"),
        ("resync", "resync"),
    ):
        config = _materialize_production_tree(tmp_path / f"adm-{mode}")
        old = os.environ.get("FAKE_TRAINER_MODE")
        os.environ["FAKE_TRAINER_MODE"] = mode
        try:
            with pytest.raises(Multires500ProofError, match=match):
                run_proof(config)
        finally:
            if old is None:
                os.environ.pop("FAKE_TRAINER_MODE", None)
            else:
                os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_legacy_trainer_exit(tmp_path):
    config = _materialize_production_tree(tmp_path / "legacy-exit")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "legacy"
    try:
        with pytest.raises(Multires500ProofError, match="trainer exit|legacy"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_mixed_atlas_digest_from_trainer(tmp_path):
    config = _materialize_production_tree(tmp_path / "mix-atlas")
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = "mixed_atlas"
    try:
        with pytest.raises(Multires500ProofError, match="atlas_sha256 digests differ"):
            run_proof(config)
    finally:
        if old is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old


def test_synthetic_matching_evidence_cannot_be_marked_production_pass():
    """Even perfect digests from injection stay non-admissible."""
    transport = _transport()
    report = run_proof(_synthetic_config(), transport=transport)
    assert report["same_seed_match"] is True
    assert report["different_seed_diverges"] is True
    assert report["failures"] == []
    assert report["production_pass"] is False
    assert report["admissible"] is False
    assert report["mode"] == MODE_SYNTHETIC
    assert report["records_required"] is True
    assert report["digest_only_admissible"] is False
    assert report["required_client_count"] == REQUIRED_CLIENT_COUNT
    assert report["transitions_per_client"] == TRANSITIONS_PER_CLIENT
    assert report["record_ordering"] == RECORD_ORDERING
    assert report["orphan_processes_after_teardown"] == 0
    assert report["teardown"]["orphan_processes_after_teardown"] == 0
    assert report["proof_client_ids"] == list(DEFAULT_SYNTHETIC_CLIENT_IDS)
    assert report["base_server_frame"] == DEFAULT_SYNTHETIC_BASE_SERVER_FRAME
    assert DEFAULT_SYNTHETIC_CLIENT_IDS[0] == "mrproof-00"
    assert DEFAULT_SYNTHETIC_BASE_SERVER_FRAME != 100


def test_synthetic_exact_500_layout_is_time_major_client_minor():
    transport = _transport()
    report = run_proof(_synthetic_config(), transport=transport)
    # Recover records by replaying the same transport seed path.
    result = transport.launch(
        type("Req", (), {
            "launch_id": "layout_check",
            "seed": 7142026,
            "game_seed": 4242,
            "language": STACK_LANGUAGE,
            "transition_count": REQUIRED_TRANSITION_COUNT,
            "policy_version": 1,
            "map_name": "mlstage_0001",
            "map_epoch": 3,
            "atlas_sha256": ATLAS,
            "runtime_manifest_sha256": RUNTIME,
            "timeout_seconds": 5.0,
        })()
    )
    assert len(result.records) == REQUIRED_TRANSITION_COUNT
    derived = client_ids_from_first_round(result.records)
    assert derived == DEFAULT_SYNTHETIC_CLIENT_IDS
    client_counts = {client_id: 0 for client_id in derived}
    for flat_index, record in enumerate(result.records):
        time_step, client_index, client_id = layout_for_flat_index(
            flat_index, derived
        )
        assert record.index == flat_index
        assert record.client_id == client_id
        assert record.batch_round_id == time_step
        assert record.server_frame == DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + time_step
        client_counts[record.client_id] += 1
    assert client_counts == {
        client_id: TRANSITIONS_PER_CLIENT for client_id in derived
    }
    assert report["orphan_processes_after_teardown"] == 0


def _env_mode(mode: str):
    old = os.environ.get("FAKE_TRAINER_MODE")
    os.environ["FAKE_TRAINER_MODE"] = mode
    return old


def _restore_mode(old: str | None) -> None:
    if old is None:
        os.environ.pop("FAKE_TRAINER_MODE", None)
    else:
        os.environ["FAKE_TRAINER_MODE"] = old


def test_production_rejects_digest_only_assertion(tmp_path):
    """M6c: a trajectory digest without records can never production-pass."""
    config = _materialize_production_tree(tmp_path / "digest-only")
    old = _env_mode("digest_only")
    try:
        with pytest.raises(Multires500ProofError, match="digest-only|records are missing"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_missing_records_key(tmp_path):
    config = _materialize_production_tree(tmp_path / "missing-recs")
    old = _env_mode("missing_records")
    try:
        with pytest.raises(Multires500ProofError, match="records|digest-only"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_empty_records(tmp_path):
    config = _materialize_production_tree(tmp_path / "empty-recs")
    old = _env_mode("empty_records")
    try:
        with pytest.raises(Multires500ProofError, match="empty|digest-only|records"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_extra_record(tmp_path):
    config = _materialize_production_tree(tmp_path / "extra")
    old = _env_mode("extra_record")
    try:
        with pytest.raises(Multires500ProofError, match="extra|length|records"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_reordered_client_minor(tmp_path):
    config = _materialize_production_tree(tmp_path / "reorder")
    old = _env_mode("reorder")
    try:
        with pytest.raises(Multires500ProofError, match="reordered|client_id|time-major"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_bad_identity_digest(tmp_path):
    config = _materialize_production_tree(tmp_path / "bad-id")
    old = _env_mode("bad_identity")
    try:
        with pytest.raises(Multires500ProofError, match="identity_digest"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_unbound_record_atlas(tmp_path):
    config = _materialize_production_tree(tmp_path / "unbound")
    old = _env_mode("unbound_record")
    try:
        with pytest.raises(Multires500ProofError, match="unbound atlas"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_truncated_record(tmp_path):
    config = _materialize_production_tree(tmp_path / "trunc")
    old = _env_mode("truncated_record")
    try:
        with pytest.raises(Multires500ProofError, match="truncated|missing fields"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_noncanonical_record_schema(tmp_path):
    config = _materialize_production_tree(tmp_path / "noncanon")
    old = _env_mode("noncanonical_schema")
    try:
        with pytest.raises(Multires500ProofError, match="noncanonical"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_wrong_trajectory_vs_records(tmp_path):
    config = _materialize_production_tree(tmp_path / "wrong-traj")
    old = _env_mode("wrong_traj")
    try:
        with pytest.raises(
            Multires500ProofError,
            match="trajectory_sha256 does not match recomputation",
        ):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_duplicate_layout_break(tmp_path):
    """Duplicate identity forced onto the wrong client slot fails closed."""
    config = _materialize_production_tree(tmp_path / "dup")
    old = _env_mode("duplicate")
    try:
        with pytest.raises(Multires500ProofError, match="duplicate|reordered|client_id"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_admit_canonical_records_rejects_duplicate_identity_fields():
    """Uniqueness of client/frame/round is enforced even off the exact-500 path."""
    records = []
    for index in range(3):
        records.append(
            make_transition_record(
                index=index,
                observation=_observation_with_dyn(float(index)),
                action=[0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                reward=0.1,
                client_id="mrproof-00",
                server_frame=DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + index,
                batch_round_id=index,
                policy_version=1,
                map_name="mlstage_0001",
                map_epoch=1,
                atlas_sha256=ATLAS,
                runtime_manifest_sha256=RUNTIME,
                rust_features=[0.1] * 24,
            ).to_mapping()
        )
    # Clone record 0's identity onto index 2 (still dense indices).
    records[2]["client_id"] = records[0]["client_id"]
    records[2]["server_frame"] = records[0]["server_frame"]
    records[2]["batch_round_id"] = records[0]["batch_round_id"]
    records[2]["identity_digest"] = records[0]["identity_digest"]
    with pytest.raises(Multires500ProofError, match="duplicate"):
        admit_canonical_records(
            records,
            expected_count=3,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def _one_canonical_record() -> dict:
    return make_transition_record(
        index=0,
        observation=_observation_with_dyn(),
        action=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        reward=0.0,
        client_id="mrproof-00",
        server_frame=DEFAULT_SYNTHETIC_BASE_SERVER_FRAME,
        batch_round_id=0,
        policy_version=1,
        map_name="mlstage_0001",
        map_epoch=1,
        atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        rust_features=[0.1] * 24,
    ).to_mapping()


def _admit_one(record: dict):
    return admit_canonical_records(
        [record],
        expected_count=1,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
        expected_map_name="mlstage_0001",
        expected_map_epoch=1,
        expected_policy_version=1,
    )


def test_rust_feature_digest_is_derived_from_observation_when_payload_omitted():
    record = _one_canonical_record()
    assert "rust_features" not in record
    admitted = _admit_one(record)
    assert admitted[0].rust_feature_digest == rust_feature_digest([0.1] * 24)

    record["rust_feature_digest"] = rust_feature_digest([0.2] * 24)
    with pytest.raises(
        Multires500ProofError,
        match=r"rust_feature_digest.*observation\[DYN\.slice\]",
    ):
        _admit_one(record)


def test_rust_feature_payload_cannot_override_observation_dyn_slice():
    record = _one_canonical_record()
    record["rust_features"] = [0.2] * 24
    record["rust_feature_digest"] = rust_feature_digest(record["rust_features"])
    with pytest.raises(
        Multires500ProofError,
        match=r"rust_features.*observation\[DYN\.slice\]",
    ):
        _admit_one(record)

    with pytest.raises(
        Multires500ProofError,
        match=r"rust_features.*observation\[DYN\.slice\]",
    ):
        make_transition_record(
            index=0,
            observation=_observation_with_dyn(),
            action=[0.0] * ACTION_DIM,
            reward=0.0,
            client_id="mrproof-00",
            server_frame=1,
            batch_round_id=0,
            policy_version=1,
            map_name="mlstage_0001",
            map_epoch=1,
            atlas_sha256=ATLAS,
            runtime_manifest_sha256=RUNTIME,
            rust_features=[0.2] * 24,
        )


def test_production_rejects_empty_process_ids(tmp_path):
    config = _materialize_production_tree(tmp_path / "nopids")
    old = _env_mode("no_process_ids")
    try:
        with pytest.raises(Multires500ProofError, match="process_ids"):
            run_proof(config)
    finally:
        _restore_mode(old)


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("missing_process_records", "process_records"),
        ("malformed_process_records", "process_records.*invalid"),
        ("duplicate_process_records", "process_records.*invalid"),
    ),
)
def test_production_rejects_unattested_process_records(tmp_path, mode, message):
    config = _materialize_production_tree(tmp_path / mode)
    old = _env_mode(mode)
    try:
        with pytest.raises(Multires500ProofError, match=message):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_training_digest_lie_from_one_run(tmp_path):
    config = _materialize_production_tree(tmp_path / "output-training-lie")
    old = _env_mode("training_digest_lie")
    try:
        with pytest.raises(Multires500ProofError, match="training_manifest_sha256"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_pass_emits_full_records_and_teardown(tmp_path):
    config = _materialize_production_tree(tmp_path / "prod-records")
    report = run_proof(config)
    assert report["production_pass"] is True
    assert report["orphan_processes_after_teardown"] == 0
    assert report["teardown"]["orphan_processes_after_teardown"] == 0
    assert report["records_required"] is True
    assert report["digest_only_admissible"] is False
    artifact = config.evidence_dir / "one_run_same_seed_run_a.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert isinstance(payload.get("records"), list)
    assert len(payload["records"]) == REQUIRED_TRANSITION_COUNT
    assert [record["role"] for record in payload["process_records"]] == [
        "q2ded",
        "network-client-00",
        "network-client-01",
        "network-client-02",
        "network-client-03",
    ]
    assert payload["process_ids"] == payload["launched_process_ids"]
    assert payload["process_ids"] == payload["terminated_process_ids"]
    assert payload["terminated_process_records"] == payload["process_records"]
    admitted = admit_canonical_records(
        payload["records"],
        expected_count=REQUIRED_TRANSITION_COUNT,
        expected_atlas_sha256=payload["atlas_sha256"],
        expected_runtime_manifest_sha256=payload["runtime_manifest_sha256"],
        expected_map_name=payload["map_name"],
        expected_map_epoch=int(payload["map_epoch"]),
        expected_policy_version=int(payload["policy_version"]),
        expected_trajectory_sha256=payload["trajectory_sha256"],
    )
    assert len(admitted) == REQUIRED_TRANSITION_COUNT
    assert admitted[0].client_id == "mrproof-00"
    assert admitted[1].client_id == "mrproof-01"
    assert admitted[2].client_id == "mrproof-02"
    assert admitted[3].client_id == "mrproof-03"
    assert admitted[4].client_id == "mrproof-00"
    assert admitted[4].batch_round_id == 1
    assert admitted[0].server_frame == DEFAULT_SYNTHETIC_BASE_SERVER_FRAME
    assert admitted[4].server_frame == DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + 1
    assert report["proof_client_ids"] == list(DEFAULT_SYNTHETIC_CLIENT_IDS)
    assert report["base_server_frame"] == DEFAULT_SYNTHETIC_BASE_SERVER_FRAME
    assert "verified_dead_process_ids" in report["teardown"]
    assert report["production_admission"]["checkpoint_initialization"] == "random"
    assert report["production_admission"]["checkpoint_training_step"] == 0


def test_admit_canonical_records_rejects_digest_only_none():
    with pytest.raises(Multires500ProofError, match="digest-only"):
        admit_canonical_records(
            None,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def _build_exact_500_raw_records(
    *,
    client_ids=DEFAULT_SYNTHETIC_CLIENT_IDS,
    base_frame=DEFAULT_SYNTHETIC_BASE_SERVER_FRAME,
    mutate=None,
):
    records = []
    for flat_index in range(REQUIRED_TRANSITION_COUNT):
        time_step = flat_index // REQUIRED_CLIENT_COUNT
        client_index = flat_index % REQUIRED_CLIENT_COUNT
        record = make_transition_record(
            index=flat_index,
            observation=_observation_with_dyn(float(flat_index)),
            action=[0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            reward=0.1,
            client_id=client_ids[client_index],
            server_frame=base_frame + time_step,
            batch_round_id=time_step,
            policy_version=1,
            map_name="mlstage_0001",
            map_epoch=1,
            atlas_sha256=ATLAS,
            runtime_manifest_sha256=RUNTIME,
            rust_features=[0.1] * 24,
        ).to_mapping()
        if mutate is not None:
            mutate(flat_index, record)
        records.append(record)
    return records


def test_admit_derives_client_ids_and_non_magic_base_frame():
    records = _build_exact_500_raw_records(
        client_ids=("mrproof-00", "mrproof-01", "mrproof-02", "mrproof-03"),
        base_frame=250,
    )
    admitted = admit_canonical_records(
        records,
        expected_count=REQUIRED_TRANSITION_COUNT,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
        expected_map_name="mlstage_0001",
        expected_map_epoch=1,
        expected_policy_version=1,
    )
    assert client_ids_from_first_round(admitted) == (
        "mrproof-00",
        "mrproof-01",
        "mrproof-02",
        "mrproof-03",
    )
    assert admitted[0].server_frame == 250
    assert admitted[4].server_frame == 251
    assert admitted[499].server_frame == 250 + 124
    assert admitted[499].batch_round_id == 124


def test_admit_rejects_cross_client_frame_skew():
    def mutate(flat_index, record):
        if flat_index == 5:
            record["server_frame"] = DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + 2
            record["identity_digest"] = make_transition_record(
                index=record["index"],
                observation=record["observation"],
                action=record["action"],
                reward=record["reward"],
                client_id=record["client_id"],
                server_frame=record["server_frame"],
                batch_round_id=record["batch_round_id"],
                policy_version=record["policy_version"],
                map_name=record["map_name"],
                map_epoch=record["map_epoch"],
                atlas_sha256=record["atlas_sha256"],
                runtime_manifest_sha256=record["runtime_manifest_sha256"],
                rust_features=[0.1] * 24,
            ).identity_digest

    records = _build_exact_500_raw_records(mutate=mutate)
    with pytest.raises(Multires500ProofError, match="skew|server_frame|non-unit"):
        admit_canonical_records(
            records,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def test_admit_rejects_non_unit_frame_jump():
    def mutate(flat_index, record):
        time_step = flat_index // REQUIRED_CLIENT_COUNT
        if time_step >= 3:
            record["server_frame"] = DEFAULT_SYNTHETIC_BASE_SERVER_FRAME + time_step + 1
            record["identity_digest"] = make_transition_record(
                index=record["index"],
                observation=record["observation"],
                action=record["action"],
                reward=record["reward"],
                client_id=record["client_id"],
                server_frame=record["server_frame"],
                batch_round_id=record["batch_round_id"],
                policy_version=record["policy_version"],
                map_name=record["map_name"],
                map_epoch=record["map_epoch"],
                atlas_sha256=record["atlas_sha256"],
                runtime_manifest_sha256=record["runtime_manifest_sha256"],
                rust_features=[0.1] * 24,
            ).identity_digest

    records = _build_exact_500_raw_records(mutate=mutate)
    with pytest.raises(Multires500ProofError, match="non-unit|server_frame"):
        admit_canonical_records(
            records,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def test_admit_rejects_client_reorder_after_first_round():
    def mutate(flat_index, record):
        if flat_index in (4, 5):
            # Round 1: swap slots 0 and 1 while keeping identities unique.
            record["client_id"] = "mrproof-01" if flat_index == 4 else "mrproof-00"
            record["identity_digest"] = make_transition_record(
                index=record["index"],
                observation=record["observation"],
                action=record["action"],
                reward=record["reward"],
                client_id=record["client_id"],
                server_frame=record["server_frame"],
                batch_round_id=record["batch_round_id"],
                policy_version=record["policy_version"],
                map_name=record["map_name"],
                map_epoch=record["map_epoch"],
                atlas_sha256=record["atlas_sha256"],
                runtime_manifest_sha256=record["runtime_manifest_sha256"],
                rust_features=[0.1] * 24,
            ).identity_digest

    records = _build_exact_500_raw_records(mutate=mutate)
    with pytest.raises(Multires500ProofError, match="reordered|client_id"):
        admit_canonical_records(
            records,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def test_admit_rejects_duplicate_client_ids_in_first_round():
    def mutate(flat_index, record):
        if flat_index == 1:
            record["client_id"] = "mrproof-00"
            record["identity_digest"] = make_transition_record(
                index=record["index"],
                observation=record["observation"],
                action=record["action"],
                reward=record["reward"],
                client_id=record["client_id"],
                server_frame=record["server_frame"],
                batch_round_id=record["batch_round_id"],
                policy_version=record["policy_version"],
                map_name=record["map_name"],
                map_epoch=record["map_epoch"],
                atlas_sha256=record["atlas_sha256"],
                runtime_manifest_sha256=record["runtime_manifest_sha256"],
                rust_features=[0.1] * 24,
            ).identity_digest

    records = _build_exact_500_raw_records(mutate=mutate)
    with pytest.raises(Multires500ProofError, match="unique client_ids|duplicate"):
        admit_canonical_records(
            records,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def test_admit_rejects_truncated_trajectory_length():
    records = _build_exact_500_raw_records()[:499]
    with pytest.raises(Multires500ProofError, match="length|truncated|missing"):
        admit_canonical_records(
            records,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def test_admit_rejects_non_positive_server_frame():
    def mutate(flat_index, record):
        if flat_index < 4:
            record["server_frame"] = 0
            record["identity_digest"] = make_transition_record(
                index=record["index"],
                observation=record["observation"],
                action=record["action"],
                reward=record["reward"],
                client_id=record["client_id"],
                server_frame=1,  # temporary valid for digest helper
                batch_round_id=record["batch_round_id"],
                policy_version=record["policy_version"],
                map_name=record["map_name"],
                map_epoch=record["map_epoch"],
                atlas_sha256=record["atlas_sha256"],
                runtime_manifest_sha256=record["runtime_manifest_sha256"],
                rust_features=[0.1] * 24,
            ).identity_digest
            # Force zero frame with a matching identity for the zero frame.
            from tools.run_multires_500_transition_proof import (
                transition_identity_digest,
            )
            record["identity_digest"] = transition_identity_digest(
                client_id=record["client_id"],
                server_frame=0,
                batch_round_id=record["batch_round_id"],
                policy_version=record["policy_version"],
                map_name=record["map_name"],
                map_epoch=record["map_epoch"],
                atlas_sha256=record["atlas_sha256"],
                runtime_manifest_sha256=record["runtime_manifest_sha256"],
            )

    records = _build_exact_500_raw_records(mutate=mutate)
    with pytest.raises(Multires500ProofError, match="positive"):
        admit_canonical_records(
            records,
            expected_count=REQUIRED_TRANSITION_COUNT,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_name="mlstage_0001",
            expected_map_epoch=1,
            expected_policy_version=1,
        )


def test_production_rejects_frame_skew(tmp_path):
    config = _materialize_production_tree(tmp_path / "frame-skew")
    old = _env_mode("frame_skew")
    try:
        with pytest.raises(Multires500ProofError, match="skew|server_frame|non-unit"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_frame_jump(tmp_path):
    config = _materialize_production_tree(tmp_path / "frame-jump")
    old = _env_mode("frame_jump")
    try:
        with pytest.raises(Multires500ProofError, match="non-unit|server_frame"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_rejects_duplicate_client_ids(tmp_path):
    config = _materialize_production_tree(tmp_path / "dup-clients")
    old = _env_mode("duplicate_client_ids")
    try:
        with pytest.raises(Multires500ProofError, match="unique client_ids|duplicate"):
            run_proof(config)
    finally:
        _restore_mode(old)


def test_production_timeout_kills_detached_descendant(tmp_path):
    """Timeout cleanup must discover setsid children via /proc and verify death."""
    config = _materialize_production_tree(tmp_path / "detached")
    config.timeout_seconds = 0.4
    child_pid_file = (tmp_path / "detached" / "child.pid").resolve()
    old_mode = os.environ.get("FAKE_TRAINER_MODE")
    old_pid_file = os.environ.get("FAKE_TRAINER_CHILD_PID_FILE")
    os.environ["FAKE_TRAINER_MODE"] = "timeout_detached_child"
    os.environ["FAKE_TRAINER_CHILD_PID_FILE"] = str(child_pid_file)
    try:
        with pytest.raises(Multires500ProofError, match="timed out"):
            run_proof(config)
    finally:
        if old_mode is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old_mode
        if old_pid_file is None:
            os.environ.pop("FAKE_TRAINER_CHILD_PID_FILE", None)
        else:
            os.environ["FAKE_TRAINER_CHILD_PID_FILE"] = old_pid_file

    assert child_pid_file.is_file(), "fake trainer did not record detached child pid"
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    # Give the OS a moment if SIGKILL was just delivered.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_production_kills_live_detached_process_reported_by_identity(tmp_path):
    """A post-exit same-instance survivor is rejected and scoped for cleanup."""
    config = _materialize_production_tree(tmp_path / "reported-detached")
    child_pid_file = (tmp_path / "reported-detached" / "child.pid").resolve()
    old_mode = os.environ.get("FAKE_TRAINER_MODE")
    old_pid_file = os.environ.get("FAKE_TRAINER_CHILD_PID_FILE")
    os.environ["FAKE_TRAINER_MODE"] = "live_reported_detached"
    os.environ["FAKE_TRAINER_CHILD_PID_FILE"] = str(child_pid_file)
    try:
        with pytest.raises(Multires500ProofError, match="same-identity processes"):
            run_proof(config)
    finally:
        if old_mode is None:
            os.environ.pop("FAKE_TRAINER_MODE", None)
        else:
            os.environ["FAKE_TRAINER_MODE"] = old_mode
        if old_pid_file is None:
            os.environ.pop("FAKE_TRAINER_CHILD_PID_FILE", None)
        else:
            os.environ["FAKE_TRAINER_CHILD_PID_FILE"] = old_pid_file

    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_discover_descendant_pids_finds_setsid_child():
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, subprocess, sys, time\n"
                "child = subprocess.Popen(\n"
                "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
                "    start_new_session=True,\n"
                ")\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(30)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = parent.stdout.readline().strip()
        child_pid = int(line)
        found = discover_descendant_pids(parent.pid)
        assert child_pid in found
    finally:
        for pid in (parent.pid,):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        try:
            os.kill(child_pid, 9)
        except Exception:
            pass
        parent.wait(timeout=2)


def test_prove_process_ids_dead_rejects_live_pid():
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with pytest.raises(Multires500ProofError, match="still alive"):
            prove_process_ids_dead([sleeper.pid], where="unit-test")
    finally:
        sleeper.kill()
        sleeper.wait(timeout=2)
    prove_process_ids_dead([sleeper.pid], where="unit-test-after-kill")


def test_process_identity_treats_reused_pid_as_dead_without_killing_it():
    """Same PID plus a different /proc start tick is a different process."""
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        raw = Path(f"/proc/{sleeper.pid}/stat").read_text(encoding="utf-8")
        actual_start_ticks = int(raw[raw.rfind(")") + 2:].split()[19])
        reused_identity = ProcessIdentity(
            role="q2ded",
            pid=sleeper.pid,
            start_ticks=actual_start_ticks + 1,
        )
        assert process_identity_alive(reused_identity) is False
        prove_process_records_dead([reused_identity], where="pid-reuse-test")
        assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait(timeout=2)
