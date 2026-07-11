#!/usr/bin/env python3
"""Measure deterministic aim/fire quality against real 3ZB2 opponents.

Entity positions from ``ml_obs.c`` are already expressed in the bot-local
forward/right/up basis. This evaluator therefore never subtracts the bot's
global yaw from ``atan2(local_y, local_x)`` (the old scratch evaluator did,
which invalidated its reported yaw errors).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from harness.protocol import ML_CONTROL_LEGACY_BOT
from models.policy import OBS_DIM, Q2BotPolicy


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _nearest_visible_enemy(obs) -> Optional[np.ndarray]:
    best = None
    count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
    for idx, ent in enumerate(obs.entities[:count]):
        if ent[7] <= 0.5 or ent[8] <= 0.5:
            continue
        if int(obs.entity_debug[idx, 2]) != ML_CONTROL_LEGACY_BOT:
            continue
        dist = float(np.linalg.norm(ent[:3]))
        if best is None or dist < best[0]:
            best = (dist, ent[:3].copy())
    return None if best is None else best[1]


def _desired_look_delta(obs, rel_xyz: np.ndarray) -> Tuple[float, float]:
    """Recover exact yaw/pitch deltas from AngleVectors-local coordinates."""
    x, y, z = (float(v) for v in rel_xyz)
    pitch_rad = math.radians(float(obs.pitch))
    horizontal_forward = math.cos(pitch_rad) * x + math.sin(pitch_rad) * z
    vertical = -math.sin(pitch_rad) * x + math.cos(pitch_rad) * z
    yaw_delta = -math.degrees(math.atan2(y, horizontal_forward))
    target_pitch = -math.degrees(
        math.atan2(vertical, max(math.hypot(horizontal_forward, y), 1e-3))
    )
    return _wrap_degrees(yaw_delta), target_pitch - float(obs.pitch)


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map_name", default="q2dm1")
    parser.add_argument("--map_glob", default="")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--min_visible", type=int, default=300)
    parser.add_argument("--n_bots", type=int, default=4)
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=11)
    parser.add_argument("--server_id", type=int, default=40)
    parser.add_argument("--port_offset", type=int, default=40)
    parser.add_argument("--timescale", type=float, default=4.0)
    parser.add_argument("--fraglimit", type=int, default=30)
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument(
        "--game_seed", type=int, default=-1,
        help="gameplay RNG seed; negative preserves normal game randomness",
    )
    parser.add_argument("--yaw_deg", type=float, default=12.0)
    parser.add_argument("--pitch_deg", type=float, default=14.0)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.n_bots == 2:
        parser.error("n_bots=2 uses the broken ml2sk1 botlist; use 4 or more")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute() and not checkpoint.exists():
        checkpoint = ROOT / checkpoint
    device = _pick_device(args.device)
    policy = Q2BotPolicy().to(device)
    try:
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not load {checkpoint} for OBS_DIM={OBS_DIM}; "
            "check Q2_EXT_OBS (current checkpoints generally require Q2_EXT_OBS=1)"
        ) from exc
    policy.eval()

    maps = (
        discover_map_pool(
            map_name=args.map_name,
            map_glob=args.map_glob,
            map_dir=args.map_dir or None,
        )
        if args.map_glob else [args.map_name]
    )
    if len(maps) != 1:
        parser.error(
            f"aim evaluation is single-map, but --map_glob matched {len(maps)} maps; "
            "pass one --map_name (or a glob that resolves to exactly one map)"
        )
    env = Q2MultiEnv(
        server_id=args.server_id,
        map_name=maps[0],
        map_pool=maps,
        map_change_episodes=0,
        n_bots=args.n_bots,
        num_ml_bots=1,
        port_offset=args.port_offset,
        maxclients=args.maxclients,
        ml_slot=args.ml_slot,
        game_seed=None if args.game_seed < 0 else args.game_seed,
        max_ep_steps=10**9,
        timedemo=0,
        timescale=args.timescale,
        fraglimit=args.fraglimit,
        timelimit=args.timelimit,
    )

    yaw_pre: List[float] = []
    pitch_pre: List[float] = []
    yaw_post: List[float] = []
    pitch_post: List[float] = []
    eligible_frames = no_target_frames = 0
    visible_frames = pre_aligned = post_aligned = 0
    yaw_improved = pitch_improved = 0
    fire_requests = visible_fires = hidden_fires = 0
    pre_aligned_fires = post_aligned_fires = 0
    kills = deaths = timeouts = 0
    damage_dealt = damage_taken = 0.0

    started = time.monotonic()
    try:
        obs_vec = env.reset_all()[0]
        hx = policy.init_hidden(1, device)
        for step in range(args.steps):
            raw_obs = env._last_obs[0]
            with torch.no_grad():
                action, _value, _log_prob, hx = policy.act(
                    obs_vec, hx, device=device, deterministic=True
                )
            action = np.asarray(action, dtype=np.float32)
            action[0:2] = np.clip(action[0:2], -1.0, 1.0)
            action[2] = np.clip(action[2], -45.0, 45.0)
            action[3] = np.clip(action[3], -30.0, 30.0)
            fire = bool(action[5] > 0.5)

            alive = bool(
                raw_obs is not None
                and not raw_obs.is_terminal
                and float(raw_obs.self_state[6]) > 0.0
            )
            rel_xyz = _nearest_visible_enemy(raw_obs) if alive else None
            sample = None
            if alive and rel_xyz is None:
                sample = {"visible": False, "fire": fire}
            elif alive:
                desired_yaw, desired_pitch = _desired_look_delta(raw_obs, rel_xyz)
                pre_yaw = abs(desired_yaw)
                pre_pitch = abs(desired_pitch)
                post_yaw = abs(_wrap_degrees(desired_yaw - float(action[2])))
                new_pitch = float(np.clip(float(raw_obs.pitch) + float(action[3]), -89.0, 89.0))
                effective_pitch_delta = new_pitch - float(raw_obs.pitch)
                post_pitch = abs(desired_pitch - effective_pitch_delta)
                is_pre_aligned = pre_yaw <= args.yaw_deg and pre_pitch <= args.pitch_deg
                is_post_aligned = post_yaw <= args.yaw_deg and post_pitch <= args.pitch_deg
                sample = {
                    "visible": True,
                    "fire": fire,
                    "pre_yaw": pre_yaw,
                    "pre_pitch": pre_pitch,
                    "post_yaw": post_yaw,
                    "post_pitch": post_pitch,
                    "pre_aligned": is_pre_aligned,
                    "post_aligned": is_post_aligned,
                }

            obs_vec, _reward, term, trunc, info = env.step_all([action])[0]
            timed_out = bool(info.get("timeout"))
            timeouts += int(timed_out)
            if sample is not None and not timed_out:
                eligible_frames += 1
                fire_requests += int(sample["fire"])
                if not sample["visible"]:
                    no_target_frames += 1
                    hidden_fires += int(sample["fire"])
                else:
                    visible_frames += 1
                    visible_fires += int(sample["fire"])
                    yaw_pre.append(sample["pre_yaw"])
                    pitch_pre.append(sample["pre_pitch"])
                    yaw_post.append(sample["post_yaw"])
                    pitch_post.append(sample["post_pitch"])
                    yaw_improved += int(sample["post_yaw"] < sample["pre_yaw"])
                    pitch_improved += int(sample["post_pitch"] < sample["pre_pitch"])
                    pre_aligned += int(sample["pre_aligned"])
                    post_aligned += int(sample["post_aligned"])
                    pre_aligned_fires += int(sample["fire"] and sample["pre_aligned"])
                    post_aligned_fires += int(sample["fire"] and sample["post_aligned"])

            kills += int(round(float(info.get("kills", 0.0))))
            deaths += int(round(float(info.get("deaths", 0.0))))
            damage_dealt += float(info.get("damage_dealt", 0.0))
            damage_taken += float(info.get("damage_taken", 0.0))
            if term or trunc:
                obs_vec = env.reset_slot(0)
                hx = policy.init_hidden(1, device)
            if (step + 1) % 500 == 0:
                print(
                    f"[{step + 1}/{args.steps}] visible={visible_frames} "
                    f"post_aligned={post_aligned} fires={fire_requests} "
                    f"damage={damage_dealt:.0f} kd={kills}/{deaths}",
                    flush=True,
                )
    finally:
        env.close()

    result = {
        "checkpoint": str(checkpoint),
        "obs_dim": OBS_DIM,
        "maps": [env.active_map],
        "n_bots": args.n_bots,
        "maxclients": args.maxclients,
        "ml_slot": args.ml_slot,
        "server_id": args.server_id,
        "server_port": env.sv_port,
        "ml_ports": env.ml_ports,
        "timescale": args.timescale,
        "game_seed": args.game_seed,
        "steps": args.steps,
        "eligible_alive_frames": eligible_frames,
        "visible_frames": visible_frames,
        "no_target_frames": no_target_frames,
        "timeouts": timeouts,
        "elapsed_seconds": time.monotonic() - started,
        "pre_yaw_deg": _summary(yaw_pre),
        "pre_pitch_deg": _summary(pitch_pre),
        "predicted_post_yaw_deg": _summary(yaw_post),
        "predicted_post_pitch_deg": _summary(pitch_post),
        "pre_aligned_rate": _rate(pre_aligned, visible_frames),
        "predicted_post_aligned_rate": _rate(post_aligned, visible_frames),
        "yaw_command_improvement_rate": _rate(yaw_improved, visible_frames),
        "pitch_command_improvement_rate": _rate(pitch_improved, visible_frames),
        "fire_requests": fire_requests,
        "visible_fire_rate": _rate(visible_fires, visible_frames),
        "hidden_fire_rate": _rate(hidden_fires, no_target_frames),
        "pre_aligned_fire_precision": _rate(pre_aligned_fires, fire_requests),
        "predicted_post_aligned_fire_precision": _rate(post_aligned_fires, fire_requests),
        "predicted_post_aligned_fire_recall": _rate(post_aligned_fires, post_aligned),
        "kills": kills,
        "deaths": deaths,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
    }
    print("AIM_EVAL=" + json.dumps(result, sort_keys=True), flush=True)

    if visible_frames < args.min_visible:
        print(
            f"insufficient visible samples: {visible_frames} < {args.min_visible}",
            file=sys.stderr,
        )
        return 2
    if timeouts > max(5, args.steps // 100):
        print(f"excessive timeouts: {timeouts}/{args.steps}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
