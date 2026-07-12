#!/usr/bin/env python3
"""Fetch an exact policy generation and submit a deterministic rollout."""

import argparse
import io
import json
import os
import random
import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.rollout_protocol import (
    CoordinatorClient,
    RolloutBatch,
    deterministic_synthetic_batch,
)


def collect_q2_batch(artifact, args):
    import numpy as np
    import torch

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    from harness.env import Q2MultiEnv
    from models.policy import ACTION_DIM, HIDDEN_DIM, OBS_DIM, Q2BotPolicy

    policy = Q2BotPolicy().to(device)
    state = torch.load(io.BytesIO(artifact.payload), map_location=device)
    policy.load_state_dict(state)
    policy.eval()
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
        for index, value in enumerate(env.reset_all()):
            obs[index] = value
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
                for index, (next_obs, reward, terminated, truncated, _info) in enumerate(results):
                    done = bool(terminated or truncated)
                    arrays["rewards"][step, index] = reward
                    arrays["dones"][step, index] = done
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
    finally:
        env.close()

    determinism_key = (
        f"q2:v{artifact.version}:{artifact.sha256}:cfg={artifact.config_hash}:"
        f"seed={args.seed}:game={args.game_seed}:rollout={args.rollout_index}:"
        f"map={args.map_name}:steps={args.steps}:envs={n_envs}"
    )
    return RolloutBatch({
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
        "lattice_mode": "fresh_worker_session",
    }, arrays)


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
    args = parser.parse_args()

    client = CoordinatorClient(args.coordinator, token=args.token)
    artifact = client.fetch_policy()
    if args.policy_out:
        args.policy_out.parent.mkdir(parents=True, exist_ok=True)
        args.policy_out.write_bytes(artifact.payload)
    if args.mode == "q2":
        batch = collect_q2_batch(artifact, args)
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
    decision = client.submit(batch)
    print(json.dumps({
        "event": "batch_submission",
        "rollout_hash": batch.rollout_hash(),
        **decision.as_dict(),
    }, sort_keys=True), flush=True)
    return 0 if decision.accepted or decision.status == "duplicate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
