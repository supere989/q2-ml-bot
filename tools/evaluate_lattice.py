#!/usr/bin/env python3
"""Smoke-test lattice sidecars and gate a checkpoint's pull-vector response."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.spatial import VoxelSpatialReward


def _blank_raw_obs(pos=(0.0, 0.0, 0.0), tick=0):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[:3] = pos
    self_state[6] = 100.0
    self_state[7] = 50.0
    self_state[9] = 10.0
    return SimpleNamespace(
        tick=tick,
        self_state=self_state,
        rune_flags=np.zeros(5, dtype=np.float32),
        entity_count=0,
        entities=np.zeros((8, 9), dtype=np.float32),
        reward_item_pickup=0.0,
    )


def inspect_map(map_name: str, sidecar_dir: str = "") -> bool:
    if sidecar_dir:
        os.environ["Q2_LATTICE_DIR"] = sidecar_dir
    reward = VoxelSpatialReward.from_env(seed=0)
    obs = _blank_raw_obs()
    reward.reset(map_name, obs)
    sources = reward.sidecar_sources.get(map_name, {})
    memory = reward._memory_for_map(map_name)
    priors = sum(
        cell.prior_opportunity > 0.0 or cell.prior_threat > 0.0
        for cell in memory.values()
    )
    dynamic = len(reward.dynamic_cells.get(map_name, ()))
    print(f"map={map_name}")
    print(f"lattice={sources.get('lattice', 'MISSING')}")
    print(f"routes={sources.get('routes', 'MISSING')}")
    print(
        f"cells={len(memory)} priors={priors} dynamic={dynamic} "
        f"route={reward.selected_route or 'none'}"
    )
    return "lattice" in sources and "routes" in sources and priors > 0 and dynamic > 0


def evaluate_checkpoint(checkpoint: Path, warmup_steps: int = 4):
    try:
        import torch
        from models.policy import OBS_DIM, Q2BotPolicy
    except ModuleNotFoundError as exc:
        raise SystemExit(f"checkpoint evaluation requires PyTorch: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()
    cases = []
    channels = {
        "opportunity": (13, 16, 1.0),
        "engagement": (5, 8, 1.0),
        "threat": (9, 12, -1.0),
    }
    for channel, (vector_offset, score_offset, sign) in channels.items():
        for yaw_deg in (0.0, 90.0, 180.0, -90.0):
            for desired_world in ((1.0, 0.0), (0.0, 1.0)):
                signal_world = np.asarray(desired_world) * sign
                obs = np.zeros(OBS_DIM, dtype=np.float32)
                obs[6] = 0.5
                obs[9] = 0.5
                obs[183] = yaw_deg / 180.0
                memory = obs[-24:]
                memory[vector_offset] = signal_world[0]
                memory[vector_offset + 1] = signal_world[1]
                memory[score_offset] = 1.0
                obs_t = torch.from_numpy(obs).to(device).view(1, 1, -1)
                hx = policy.init_hidden(1, device)
                with torch.no_grad():
                    for _ in range(max(1, warmup_steps)):
                        params, _value, hx = policy(obs_t, hx)
                    local = params["cont_mean"][0, 0, :2].cpu().numpy()
                yaw = np.deg2rad(yaw_deg)
                world = np.asarray((
                    local[0] * np.cos(yaw) + local[1] * np.sin(yaw),
                    local[0] * np.sin(yaw) - local[1] * np.cos(yaw),
                ))
                denom = np.linalg.norm(world) * np.linalg.norm(desired_world)
                cosine = float(np.dot(world, desired_world) / denom) if denom > 1e-8 else 0.0
                cases.append((channel, yaw_deg, desired_world, cosine))
    values = np.asarray([case[3] for case in cases])
    by_channel = {}
    for channel in channels:
        channel_values = [case[3] for case in cases if case[0] == channel]
        by_channel[channel] = float(np.mean(channel_values))
    print(f"checkpoint={checkpoint}")
    for channel, value in by_channel.items():
        print(f"{channel}_cosine={value:+.3f}")
    print(f"mean_cosine={values.mean():+.3f} min_cosine={values.min():+.3f} cases={len(cases)}")
    return float(values.mean()), float(values.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="", help="map name whose sidecars should be smoke-tested")
    parser.add_argument("--sidecar_dir", default="", help="override sidecar search root")
    parser.add_argument("--checkpoint", type=Path, help="policy checkpoint to direction-gate")
    parser.add_argument("--warmup_steps", type=int, default=4)
    parser.add_argument(
        "--require_mean_cosine", type=float, default=None,
        help="exit nonzero unless checkpoint mean pull cosine reaches this value",
    )
    args = parser.parse_args()
    if not args.map and args.checkpoint is None:
        parser.error("provide --map, --checkpoint, or both")
    ok = True
    if args.map:
        ok = inspect_map(args.map, args.sidecar_dir) and ok
    if args.checkpoint is not None:
        mean_cosine, _minimum = evaluate_checkpoint(args.checkpoint, args.warmup_steps)
        if args.require_mean_cosine is not None:
            ok = mean_cosine >= args.require_mean_cosine and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
