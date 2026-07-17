#!/usr/bin/env python3
"""Fetch an exact policy generation and submit a deterministic rollout."""

import argparse
import io
import json
import os
import random
import socket
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.rollout_protocol import (
    CoordinatorClient,
    CoordinatorRequestError,
    CoordinatorTransportError,
    PPO_BEHAVIOR_METRIC_KEYS,
    PPO_ACTION_CARDINALITIES,
    PPO_EPISODE_SUMMARY_COLUMNS,
    PPO_TELEMETRY_SCHEMA,
    RolloutBatch,
    deterministic_synthetic_batch,
)
from harness.protocol import ML_PROTOCOL_GENERATION, OBS_DIM
from harness.distributed_runtime import (
    BackoffPolicy,
    ReconnectBackoff,
    RetryExhaustedError,
    is_retryable_http_status,
)


def _worker_runtime_config(expected, args, effective_device):
    if not isinstance(expected, Mapping):
        raise RuntimeError("runtime manifest runtime_config must be an object")
    actual = {
        "n_bots": int(args.n_bots),
        "n_ml": int(args.n_ml),
        "timescale": float(args.timescale),
        "max_ep_steps": int(args.max_ep_steps),
        "steps": int(args.steps),
        "deterministic": bool(args.deterministic),
        "deterministic_actions": bool(args.deterministic_actions),
        "device": str(effective_device),
    }
    required = ("n_bots", "n_ml", "timescale")
    missing = [name for name in required if name not in expected]
    if missing:
        raise RuntimeError(
            "runtime manifest is missing required worker config: "
            + ", ".join(missing)
        )
    # Preserve explicitly attested extension fields, but never copy a known
    # worker setting from the expected manifest: rebuild every known value
    # from the process that will actually collect the rollout.
    result = dict(expected)
    for name, value in actual.items():
        if name in expected or name in required:
            result[name] = value
    return result


def _retry_coordinator(operation, policy, jitter_key, wait=time.sleep):
    """Retry only transport/transient HTTP failures with a bounded budget."""
    backoff = ReconnectBackoff(policy, jitter_key=jitter_key)
    while True:
        try:
            result = operation()
            backoff.reset()
            return result
        except CoordinatorTransportError:
            pass
        except CoordinatorRequestError as error:
            if not is_retryable_http_status(error.status):
                raise
        delay = backoff.next_delay()
        wait(delay)


class _LeaseHeartbeat:
    def __init__(self, client, lease, interval, retry_policy):
        self.client = client
        self.lease = lease
        self.interval = max(0.1, float(interval))
        self.retry_policy = retry_policy
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.lease = _retry_coordinator(
                    lambda: self.client.heartbeat(self.lease),
                    self.retry_policy,
                    f"heartbeat:{self.lease.worker_id}",
                    wait=self._stop.wait,
                )
            except Exception as error:  # surfaced synchronously by check()
                self._error = error
                return

    def start(self):
        self._thread.start()
        return self

    def check(self):
        if self._error is not None:
            raise RuntimeError(f"assignment heartbeat failed: {self._error}") from self._error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval + 1.0))


def _prepare_deterministic_environment(args):
    if not args.deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "deterministic rollout workers require PYTHONHASHSEED=0 "
            "before interpreter startup"
        )


def _attest_worker_runtime(artifact, args, effective_device):
    if not artifact.runtime_manifest_sha256:
        raise RuntimeError("policy generation has no runtime manifest digest")
    if args.runtime_manifest is None:
        raise RuntimeError("real q2 workers require --runtime-manifest")
    from harness.runtime_attestation import (
        build_runtime_manifest,
        load_runtime_manifest,
        verify_runtime_manifest,
    )

    expected = load_runtime_manifest(args.runtime_manifest)
    key = (
        os.environ[args.attestation_key_env].encode()
        if args.attestation_key_env else None
    )
    expected_verification = verify_runtime_manifest(
        expected,
        hmac_key=key,
        require_signature=bool(args.attestation_key_env),
    )
    if not expected_verification.valid:
        raise RuntimeError(
            "invalid expected runtime manifest: "
            + "; ".join(expected_verification.errors)
        )
    if expected_verification.digest != artifact.runtime_manifest_sha256:
        raise RuntimeError("policy runtime digest does not match expected manifest")
    semantic = expected["semantic"]
    approved_maps = tuple(
        str(item.get("name", ""))
        for item in semantic.get("maps", ())
        if isinstance(item, dict) and item.get("name")
    )
    if not approved_maps:
        raise RuntimeError("runtime manifest has no approved map pool")
    if args.map_name not in approved_maps:
        raise RuntimeError(
            f"assigned map {args.map_name!r} is outside the approved manifest pool"
        )
    runtime_config = _worker_runtime_config(
        semantic.get("runtime_config"), args, effective_device
    )
    current = build_runtime_manifest(
        q2_root=Path(os.environ["Q2_ROOT"]),
        source_root=ROOT,
        map_names=approved_maps,
        source_revision=(
            os.environ.get("Q2_SOURCE_REVISION", "").strip() or None
        ),
        runtime_config=runtime_config,
        environment=dict(os.environ),
        hmac_key=key,
    )
    verification = verify_runtime_manifest(
        current,
        expected=expected,
        hmac_key=key,
        require_signature=bool(args.attestation_key_env),
    )
    if not verification.valid:
        detail = verification.errors[0] if verification.errors else "semantic mismatch"
        raise RuntimeError(f"runtime attestation failed: {detail}")
    return verification.digest


def _new_episode_accumulators(n_envs):
    return {
        "reward": np.zeros(n_envs, dtype=np.float64),
        "base_reward": np.zeros(n_envs, dtype=np.float64),
        "spatial_reward": np.zeros(n_envs, dtype=np.float64),
        "kills": np.zeros(n_envs, dtype=np.float64),
        "deaths": np.zeros(n_envs, dtype=np.float64),
        "length": np.zeros(n_envs, dtype=np.int64),
    }


def _new_batch_telemetry():
    return {
        "episode_summaries": [],
        "behavior_sums": np.zeros(len(PPO_BEHAVIOR_METRIC_KEYS), dtype=np.float64),
        "behavior_samples": 0,
    }


def _record_q2_telemetry(
    episode_accumulators,
    batch_telemetry,
    env_index,
    reward,
    done,
    info,
):
    """Accumulate one worker transition using the learner's local semantics."""
    spatial_reward = float(info.get("spatial_bonus", 0.0))
    episode_accumulators["reward"][env_index] += float(reward)
    episode_accumulators["base_reward"][env_index] += float(
        info.get("reward_base", float(reward) - spatial_reward)
    )
    episode_accumulators["spatial_reward"][env_index] += spatial_reward
    episode_accumulators["kills"][env_index] += float(info.get("kills", 0.0))
    episode_accumulators["deaths"][env_index] += float(info.get("deaths", 0.0))
    episode_accumulators["length"][env_index] += 1

    for metric_index, key in enumerate(PPO_BEHAVIOR_METRIC_KEYS):
        batch_telemetry["behavior_sums"][metric_index] += float(info.get(key, 0.0))
    batch_telemetry["behavior_samples"] += 1

    if not done:
        return
    batch_telemetry["episode_summaries"].append(tuple(
        float(episode_accumulators[column][env_index])
        for column in PPO_EPISODE_SUMMARY_COLUMNS
    ))
    for values in episode_accumulators.values():
        values[env_index] = 0


def _finalize_batch_telemetry(batch_telemetry):
    summaries = np.asarray(
        batch_telemetry["episode_summaries"], dtype=np.float64
    ).reshape(-1, len(PPO_EPISODE_SUMMARY_COLUMNS))
    return {
        "episode_summaries": summaries,
        "behavior_sums": batch_telemetry["behavior_sums"],
        "behavior_samples": np.array(
            [batch_telemetry["behavior_samples"]], dtype=np.int64
        ),
    }


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def collect_q2_batch(
    artifact,
    args,
    runtime=None,
    lease=None,
    recovery_lattice_payload=None,
):
    import numpy as np

    owns_runtime = runtime is None
    runtime = {} if runtime is None else runtime
    if not runtime:
        # These must describe the environment seen by attestation and must be
        # set before Torch creates a CUDA context. PYTHONHASHSEED=0 is a fixed
        # cross-worker prerequisite; rollout-specific randomness is seeded
        # explicitly below.
        _prepare_deterministic_environment(args)
    import torch
    from models.policy import ACTION_DIM, HIDDEN_DIM, OBS_DIM

    if not runtime:
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available() else
            "cpu" if args.device == "auto" else args.device
        )
        runtime_manifest_sha256 = _attest_worker_runtime(
            artifact, args, str(device)
        )
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        from harness.env import Q2MultiEnv
        from models.policy import Q2BotPolicy

        policy = Q2BotPolicy().to(device)
        os.environ["Q2_SV_PORT_BASE"] = str(args.sv_port_base)
        os.environ["Q2_ML_PORT_BASE"] = str(args.ml_port_base)
        env = Q2MultiEnv(
            server_id=0,
            map_name=args.map_name,
            map_pool=[args.map_name],
            map_seed=args.seed,
            game_seed=args.game_seed,
            spatial_seed=args.seed,
            n_bots=args.n_bots,
            num_ml_bots=args.n_ml,
            maxclients=max(8, args.n_bots, args.n_ml),
            max_ep_steps=args.max_ep_steps,
            timedemo=1,
            timescale=args.timescale,
            console_pipe=True,
        )
        n_envs = env.n_ml
        obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32)
        hidden = [policy.init_hidden(1, device) for _ in range(n_envs)]
        runtime.update({
            "device": device,
            "policy": policy,
            "env": env,
            "n_envs": n_envs,
            "obs": obs,
            "hidden": hidden,
            "started": False,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "episode_accumulators": _new_episode_accumulators(n_envs),
            "lane_index": lease.assignment.lane_index if lease else None,
            "lattice_artifact_sha256": "",
        })
    else:
        device = runtime["device"]
        policy = runtime["policy"]
        env = runtime["env"]
        n_envs = runtime["n_envs"]
        obs = runtime["obs"]
        hidden = runtime["hidden"]
        if artifact.runtime_manifest_sha256 != runtime["runtime_manifest_sha256"]:
            raise RuntimeError("policy generation changed runtime manifest")
        if lease is not None:
            if runtime.get("lane_index") != lease.assignment.lane_index:
                raise RuntimeError("persistent runtime cannot change assignment lanes")
            if (
                runtime.get("lattice_artifact_sha256", "")
                != lease.assignment.lattice_artifact_sha256
            ):
                raise RuntimeError(
                    "persistent runtime lattice does not match learner assignment"
                )
    episode_accumulators = runtime.setdefault(
        "episode_accumulators", _new_episode_accumulators(n_envs)
    )
    batch_telemetry = _new_batch_telemetry()
    state = torch.load(io.BytesIO(artifact.payload), map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    arrays = {
        "obs": np.empty((args.steps, n_envs, OBS_DIM), np.float32),
        "actions": np.empty((args.steps, n_envs, ACTION_DIM), np.float32),
        "rewards": np.empty((args.steps, n_envs), np.float32),
        "dones": np.empty((args.steps, n_envs), np.uint8),
        "values": np.empty((args.steps, n_envs), np.float32),
        "log_probs": np.empty((args.steps, n_envs), np.float32),
        "h_states": np.empty((args.steps, n_envs, HIDDEN_DIM), np.float32),
        "c_states": np.empty((args.steps, n_envs, HIDDEN_DIM), np.float32),
    }
    try:
        if not runtime["started"]:
            for index, value in enumerate(env.reset_all()):
                obs[index] = value
            runtime["started"] = True
            if lease is not None:
                expected_lattice = lease.assignment.lattice_artifact_sha256
                if bool(expected_lattice) != bool(recovery_lattice_payload):
                    raise RuntimeError("assignment lattice recovery payload is missing or unexpected")
                if recovery_lattice_payload:
                    from harness.spatial import load_lattice_state

                    incoming = (
                        args.lattice_dir
                        / f"incoming_{expected_lattice}.json.gz"
                    )
                    _atomic_write(incoming, recovery_lattice_payload)
                    load_lattice_state(env._spatial_rewards, incoming)
                    runtime["lattice_artifact_sha256"] = expected_lattice
                    for index, raw_obs in enumerate(env._last_obs):
                        obs[index] = env._obs_vector(index, raw_obs)
            elif args.lattice_dir:
                latest = args.lattice_dir / "lattice_latest.json.gz"
                if latest.is_file():
                    from harness.spatial import load_lattice_state

                    load_lattice_state(env._spatial_rewards, latest)
                    for index, raw_obs in enumerate(env._last_obs):
                        obs[index] = env._obs_vector(index, raw_obs)
        with torch.no_grad():
            for step in range(args.steps):
                arrays["obs"][step] = obs
                arrays["h_states"][step] = torch.cat(
                    [state[0] for state in hidden], dim=1
                ).squeeze(0).cpu().numpy()
                arrays["c_states"][step] = torch.cat(
                    [state[1] for state in hidden], dim=1
                ).squeeze(0).cpu().numpy()
                actions, values, log_probs, hidden = policy.act_batch(
                    obs, hidden, device, deterministic=args.deterministic_actions
                )
                results = env.step_all([actions[index] for index in range(n_envs)])
                arrays["actions"][step] = actions
                arrays["values"][step] = values
                arrays["log_probs"][step] = log_probs
                for index, (next_obs, reward, terminated, truncated, info) in enumerate(results):
                    done = bool(terminated or truncated)
                    arrays["rewards"][step, index] = reward
                    arrays["dones"][step, index] = done
                    _record_q2_telemetry(
                        episode_accumulators,
                        batch_telemetry,
                        index,
                        reward,
                        done,
                        info,
                    )
                    obs[index] = env.reset_slot(index) if done else next_obs
                    if done:
                        hidden[index] = policy.init_hidden(1, device)
        arrays["last_obs"] = obs.copy()
        arrays["last_h"] = torch.cat(
            [state[0] for state in hidden], dim=1
        ).squeeze(0).cpu().numpy()
        arrays["last_c"] = torch.cat(
            [state[1] for state in hidden], dim=1
        ).squeeze(0).cpu().numpy()
        arrays.update(_finalize_batch_telemetry(batch_telemetry))
        runtime["obs"] = obs
        runtime["hidden"] = hidden
    finally:
        if owns_runtime:
            env.close()

    determinism_key = (
        lease.assignment.determinism_key if lease is not None else
        f"q2:v{artifact.version}:{artifact.sha256}:cfg={artifact.config_hash}:"
        f"seed={args.seed}:game={args.game_seed}:rollout={args.rollout_index}:"
        f"map={args.map_name}:steps={args.steps}:envs={n_envs}"
    )
    metadata = {
        "worker_id": args.worker_id,
        "sequence": args.sequence,
        "policy_version": artifact.version,
        "policy_sha256": artifact.sha256,
        "config_hash": artifact.config_hash,
        "seed": args.seed,
        "game_seed": args.game_seed,
        "rollout_index": args.rollout_index,
        "determinism_key": determinism_key,
        "producer": "q2",
        "map_name": args.map_name,
        "n_envs": n_envs,
        "device": str(device),
        "deterministic_actions": bool(args.deterministic_actions),
        "telemetry_schema": PPO_TELEMETRY_SCHEMA,
        "protocol_generation": ML_PROTOCOL_GENERATION,
        "observation_dim": OBS_DIM,
        "action_cardinalities": dict(PPO_ACTION_CARDINALITIES),
        "runtime_manifest_sha256": runtime["runtime_manifest_sha256"],
        "lattice_mode": (
            "versioned_snapshot" if lease is not None else
            "fresh_worker_session" if owns_runtime else "persistent"
        ),
    }
    if lease is not None:
        metadata.update(lease.assignment.batch_contract())
        metadata.update({
            "worker_id": lease.worker_id,
            "lease_id": lease.lease_id,
            "lease_epoch": lease.epoch,
        })
    return RolloutBatch(metadata, arrays)


def _configure_leased_assignment(args, lease, runtime):
    assignment = lease.assignment
    if runtime:
        stable = (
            runtime.get("lane_index"),
            args.seed,
            args.game_seed,
            args.map_name,
            args.n_ml,
            args.steps,
        )
        expected = (
            assignment.lane_index,
            assignment.seed,
            assignment.game_seed,
            assignment.map_name,
            assignment.n_envs,
            assignment.steps,
        )
        if stable != expected:
            raise RuntimeError("persistent runtime assignment topology changed")
    args.seed = assignment.seed
    args.game_seed = assignment.game_seed
    args.rollout_index = assignment.rollout_index
    args.steps = assignment.steps
    args.n_ml = assignment.n_envs
    args.map_name = assignment.map_name


def _claim_next_assignment(client, args, preferred_lane, retry_policy):
    while True:
        try:
            return _retry_coordinator(
                lambda: client.claim_assignment(args.worker_id, preferred_lane),
                retry_policy,
                f"claim:{args.worker_id}",
            )
        except CoordinatorRequestError as error:
            if error.status != 409:
                raise
            time.sleep(max(0.1, args.poll_seconds))


def _snapshot_leased_lattice(batch, runtime, args, assignment):
    from harness.spatial import save_lattice_state

    args.lattice_dir.mkdir(parents=True, exist_ok=True)
    target = args.lattice_dir / f"outgoing_{assignment.assignment_id}.json.gz"
    save_lattice_state(
        runtime["env"]._spatial_rewards,
        target,
        total_env_steps=(
            assignment.policy_version
            + assignment.steps * assignment.n_envs
        ),
    )
    payload = target.read_bytes()
    batch.arrays["lattice_payload"] = np.frombuffer(
        payload, dtype=np.uint8
    ).copy()
    return payload


def _run_leased_worker(args):
    if args.mode != "q2" or not args.continuous:
        raise ValueError("--leased requires --mode q2 --continuous")
    if args.lattice_dir is None:
        raise ValueError("--leased requires --lattice-dir")
    retry_policy = BackoffPolicy(
        initial_delay=args.retry_initial_seconds,
        maximum_delay=args.retry_max_seconds,
        multiplier=2.0,
        jitter_fraction=0.2,
        max_attempts=args.retry_attempts,
    )
    client = CoordinatorClient(
        args.coordinator, token=args.token, timeout=args.request_timeout
    )
    runtime = {}
    preferred_lane = None
    generations = 0
    try:
        while True:
            lease = _claim_next_assignment(
                client, args, preferred_lane, retry_policy
            )
            _configure_leased_assignment(args, lease, runtime)
            artifact = _retry_coordinator(
                client.fetch_policy,
                retry_policy,
                f"policy:{args.worker_id}",
            )
            assignment = lease.assignment
            if (
                artifact.version != assignment.policy_version
                or artifact.sha256 != assignment.policy_sha256
                or artifact.config_hash != assignment.config_hash
            ):
                raise RuntimeError("claimed assignment does not match fetched policy")
            if args.policy_out:
                args.policy_out.parent.mkdir(parents=True, exist_ok=True)
                args.policy_out.write_bytes(artifact.payload)

            recovery_payload = None
            if not runtime and assignment.lattice_artifact_sha256:
                lattice = _retry_coordinator(
                    lambda: client.fetch_lattice(
                        assignment.lane_index,
                        assignment.lattice_artifact_sha256,
                    ),
                    retry_policy,
                    f"lattice:{args.worker_id}",
                )
                lattice.validate_recovery_for(assignment)
                recovery_payload = lattice.payload

            heartbeat = _LeaseHeartbeat(
                client, lease, args.heartbeat_seconds, retry_policy
            ).start()
            try:
                batch = collect_q2_batch(
                    artifact,
                    args,
                    runtime=runtime,
                    lease=lease,
                    recovery_lattice_payload=recovery_payload,
                )
                lattice_payload = _snapshot_leased_lattice(
                    batch, runtime, args, assignment
                )
                heartbeat.check()
                decision = _retry_coordinator(
                    lambda: client.submit(batch),
                    retry_policy,
                    f"submit:{args.worker_id}",
                )
            except Exception:
                try:
                    _retry_coordinator(
                        lambda: client.release_lease(
                            lease, "worker_collection_failed", True
                        ),
                        retry_policy,
                        f"release:{args.worker_id}",
                    )
                except Exception:
                    pass
                raise
            finally:
                heartbeat.close()

            print(json.dumps({
                "event": "batch_submission",
                "assignment_id": assignment.assignment_id,
                "rollout_hash": batch.rollout_hash(),
                **decision.as_dict(),
            }, sort_keys=True), flush=True)
            if not (decision.accepted or decision.status == "duplicate"):
                return 2
            if not decision.lattice_artifact_sha256:
                raise RuntimeError("learner did not adopt the submitted lattice")
            runtime["lattice_artifact_sha256"] = (
                decision.lattice_artifact_sha256
            )
            _atomic_write(
                args.lattice_dir / f"lattice_{artifact.version:08d}.json.gz",
                lattice_payload,
            )
            _atomic_write(
                args.lattice_dir / "lattice_latest.json.gz", lattice_payload
            )
            preferred_lane = assignment.lane_index
            args.sequence += 1
            generations += 1
            if args.max_generations and generations >= args.max_generations:
                return 0
    finally:
        if runtime.get("env") is not None:
            runtime["env"].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "q2"), default="synthetic")
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--game-seed", type=int, default=1)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--map-name", default="mltrain_00005208")
    parser.add_argument("--n-bots", type=int, default=4)
    parser.add_argument("--n-ml", type=int, default=4)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--timescale", type=float, default=10.0)
    parser.add_argument("--sv-port-base", type=int, default=36800)
    parser.add_argument("--ml-port-base", type=int, default=36900)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--policy-out", type=Path)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--max-generations", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--lattice-dir", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--attestation-key-env", default="")
    parser.add_argument("--leased", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--retry-attempts", type=int, default=8)
    parser.add_argument("--retry-initial-seconds", type=float, default=0.5)
    parser.add_argument("--retry-max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.continuous and args.verify_determinism:
        parser.error("--continuous and --verify-determinism are separate modes")
    if args.leased:
        try:
            return _run_leased_worker(args)
        except (RetryExhaustedError, RuntimeError, ValueError) as error:
            print(f"leased worker failed: {error}", file=sys.stderr, flush=True)
            return 2

    client = CoordinatorClient(args.coordinator, token=args.token)
    artifact = client.fetch_policy()
    if args.policy_out:
        args.policy_out.parent.mkdir(parents=True, exist_ok=True)
        args.policy_out.write_bytes(artifact.payload)
    runtime = {} if args.continuous and args.mode == "q2" else None
    if args.mode == "q2":
        batch = collect_q2_batch(artifact, args, runtime=runtime)
    else:
        batch = deterministic_synthetic_batch(
            artifact,
            args.worker_id,
            args.sequence,
            args.seed,
            args.game_seed,
            args.rollout_index,
            steps=args.steps,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
        )
    if args.verify_determinism:
        repeated = (
            collect_q2_batch(artifact, args)
            if args.mode == "q2"
            else deterministic_synthetic_batch(
                artifact,
                args.worker_id,
                args.sequence,
                args.seed,
                args.game_seed,
                args.rollout_index,
                steps=args.steps,
                obs_dim=args.obs_dim,
                action_dim=args.action_dim,
            )
        )
        if repeated.rollout_hash() != batch.rollout_hash():
            differences = {}
            for name in sorted(batch.arrays):
                first = batch.arrays[name]
                second = repeated.arrays[name]
                if not np.array_equal(first, second):
                    differences[name] = {
                        "shape": list(first.shape),
                        "max_abs": float(
                            np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))
                        ),
                    }
            raise RuntimeError(
                "local deterministic rollout validation failed: "
                + json.dumps(differences, sort_keys=True)
            )
    generations = 0
    try:
        while True:
            decision = client.submit(batch)
            print(json.dumps({
                "event": "batch_submission",
                "rollout_hash": batch.rollout_hash(),
                **decision.as_dict(),
            }, sort_keys=True), flush=True)
            if not (decision.accepted or decision.status == "duplicate"):
                return 2
            generations += 1
            if args.lattice_dir and runtime:
                from harness.spatial import save_lattice_state

                args.lattice_dir.mkdir(parents=True, exist_ok=True)
                instances = runtime["env"]._spatial_rewards
                save_lattice_state(
                    instances,
                    args.lattice_dir / f"lattice_{artifact.version:08d}.json.gz",
                    total_env_steps=(args.rollout_index + 1) * args.steps * runtime["n_envs"],
                )
                save_lattice_state(
                    instances,
                    args.lattice_dir / "lattice_latest.json.gz",
                    total_env_steps=(args.rollout_index + 1) * args.steps * runtime["n_envs"],
                )
            if not args.continuous or (
                args.max_generations and generations >= args.max_generations
            ):
                return 0
            while True:
                status = client.status()
                if int(status["policy_version"]) > artifact.version:
                    break
                time.sleep(max(0.1, args.poll_seconds))
            artifact = client.fetch_policy()
            args.sequence += 1
            args.rollout_index += 1
            batch = (
                collect_q2_batch(artifact, args, runtime=runtime)
                if args.mode == "q2"
                else deterministic_synthetic_batch(
                    artifact,
                    args.worker_id,
                    args.sequence,
                    args.seed,
                    args.game_seed,
                    args.rollout_index,
                    steps=args.steps,
                    obs_dim=args.obs_dim,
                    action_dim=args.action_dim,
                )
            )
    finally:
        if runtime and runtime.get("env") is not None:
            runtime["env"].close()


if __name__ == "__main__":
    raise SystemExit(main())
