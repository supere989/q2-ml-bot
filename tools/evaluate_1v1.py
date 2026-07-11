#!/usr/bin/env python3
"""Evaluate the ML bot in a true 1v1 setup against one 3ZB2 opponent."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOCAL_Q2_ROOT = ROOT.parent / "q2_lithium_merge"
if LOCAL_Q2_ROOT.exists():
    os.environ.setdefault("Q2_ROOT", str(LOCAL_Q2_ROOT))
os.environ.setdefault("Q2_POLICY_STATEFUL", "0")
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from harness.tactical import (
    LIVE_MODEL,
    OllamaTacticalReasoner,
    apply_intent_to_action,
)
from models.policy import Q2BotPolicy

CONTROL_SOURCE_NAMES = {
    0: "unknown",
    1: "human",
    2: "ml_bot",
    3: "legacy_bot",
}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def _latest_checkpoint() -> Path:
    candidates = sorted((ROOT / "checkpoints").glob("policy_[0-9]*.pt"))
    if not candidates:
        raise FileNotFoundError("no checkpoints/policy_*.pt found")
    return candidates[-1]


def _safe_ratio(kills: float, deaths: float) -> float:
    return kills / max(deaths, 1.0)


def _source_name(value: int) -> str:
    return CONTROL_SOURCE_NAMES.get(int(value), f"source_{int(value)}")


def _debug_state(debug) -> Dict[str, object]:
    if debug is None:
        return {}
    flags = int(debug[3])
    return {
        "edict": int(debug[0]),
        "slot": int(debug[1]),
        "source": _source_name(int(debug[2])),
        "flags": flags,
        "bot": bool(flags & 0x02),
        "ml": bool(flags & 0x04),
        "observer": bool(flags & 0x20),
        "solid_not": bool(flags & 0x40),
        "noclip": bool(flags & 0x80),
        "noclient": bool(flags & 0x100),
        "spectator": bool(flags & 0x200),
        "fly": bool(flags & 0x400),
        "swim": bool(flags & 0x800),
        "pm_spectator": bool(flags & 0x1000),
        "pm_freeze": bool(flags & 0x2000),
        "grounded": bool(flags & 0x4000),
        "pm_on_ground": bool(flags & 0x8000),
    }


def _round_vec(values, digits: int = 1) -> List[float]:
    return [round(float(v), digits) for v in values]


def _action_snapshot(action: np.ndarray) -> Dict[str, object]:
    return {
        "forward": round(float(action[0]), 3),
        "right": round(float(action[1]), 3),
        "yaw": round(float(action[2]), 2),
        "pitch": round(float(action[3]), 2),
        "jump": bool(float(action[4]) > 0.5),
        "fire": bool(float(action[5]) > 0.5),
        "hook": int(round(float(action[6]))),
        "weapon": int(round(float(action[7]))),
    }


def _engine_action_snapshot(obs) -> Dict[str, object]:
    if obs is None or not hasattr(obs, "action_debug"):
        return {}
    dbg = obs.action_debug
    return {
        "tick": int(dbg[0]),
        "accepted": bool(int(dbg[1])),
        "timeout_count": int(dbg[2]),
        "weapon": int(dbg[3]),
        "forward": round(float(dbg[4]), 3),
        "right": round(float(dbg[5]), 3),
        "yaw": round(float(dbg[6]), 2),
        "pitch": round(float(dbg[7]), 2),
        "jump": bool(int(dbg[8])),
        "fire": bool(int(dbg[9])),
        "hook": int(dbg[10]),
    }


def _slot_snapshot(obs, action: np.ndarray | None, distance_xy: float) -> Dict[str, object]:
    if obs is None:
        return {}
    return {
        "slot": int(obs.bot_slot),
        "state": _debug_state(obs.self_debug) if hasattr(obs, "self_debug") else {},
        "pos": _round_vec(obs.self_state[:3]),
        "vel": _round_vec(obs.self_state[3:6]),
        "speed_xy": round(float(np.linalg.norm(obs.self_state[3:5])), 2),
        "distance_xy": round(float(distance_xy), 2),
        "action": _action_snapshot(action) if action is not None else {},
        "engine_action": _engine_action_snapshot(obs),
    }


def _scripted_opponent_action(step: int) -> np.ndarray:
    """Simple moving target for generated maps without 3ZB2 route files."""
    action = np.zeros(8, dtype=np.float32)
    action[0] = 0.75
    action[1] = 0.65 * math.sin(step * 0.11)
    action[2] = 22.0 * math.sin(step * 0.07)
    action[3] = 3.0 * math.sin(step * 0.03)
    action[4] = 1.0 if step % 37 == 0 else 0.0
    action[5] = 1.0
    action[6] = 0.0
    action[7] = 7.0
    return action


def _random_action(rng: random.Random) -> np.ndarray:
    """Exploration action for live human-in-the-loop sparring."""
    action = np.zeros(8, dtype=np.float32)
    action[0] = rng.uniform(-1.0, 1.0)
    action[1] = rng.uniform(-1.0, 1.0)
    action[2] = rng.uniform(-35.0, 35.0)
    action[3] = rng.uniform(-12.0, 12.0)
    action[4] = 1.0 if rng.random() < 0.18 else 0.0
    action[5] = 1.0 if rng.random() < 0.55 else 0.0
    action[6] = rng.choice([0.0, 0.0, 1.0, 2.0, 3.0])
    action[7] = rng.choice([0.0, 2.0, 3.0, 7.0, 8.0, 9.0])
    return action


def _apply_action_randomization(action: np.ndarray, rng: random.Random, args) -> tuple[np.ndarray, bool]:
    """Inject bounded randomness without changing the policy network."""
    if args.random_action_prob > 0.0 and rng.random() < args.random_action_prob:
        return _random_action(rng), True

    noise = max(0.0, float(args.action_noise))
    if noise <= 0.0:
        return action, False

    out = np.array(action, dtype=np.float32, copy=True)
    out[0] += rng.gauss(0.0, noise)
    out[1] += rng.gauss(0.0, noise)
    out[2] += rng.gauss(0.0, noise * 24.0)
    out[3] += rng.gauss(0.0, noise * 12.0)
    if rng.random() < noise * 0.20:
        out[4] = 1.0
    if rng.random() < noise * 0.10:
        out[5] = 1.0
    if rng.random() < noise * 0.05:
        out[6] = rng.choice([0.0, 1.0, 2.0, 3.0])
    if rng.random() < noise * 0.05:
        out[7] = rng.choice([0.0, 2.0, 3.0, 7.0, 8.0, 9.0])

    out[0] = np.clip(out[0], -1.0, 1.0)
    out[1] = np.clip(out[1], -1.0, 1.0)
    out[2] = np.clip(out[2], -45.0, 45.0)
    out[3] = np.clip(out[3], -30.0, 30.0)
    out[4] = 1.0 if out[4] > 0.5 else 0.0
    out[5] = 1.0 if out[5] > 0.5 else 0.0
    out[6] = np.clip(round(float(out[6])), 0.0, 3.0)
    out[7] = np.clip(round(float(out[7])), 0.0, 9.0)
    return out, False


def _enemy_snapshot(obs) -> Dict[str, object]:
    if obs is None:
        return {
            "visible_enemies": 0.0,
            "nearest_enemy": 0.0,
            "nearest_enemy_health": 0.0,
            "nearest_enemy_slot": -1,
            "nearest_enemy_source": "none",
            "entity_source_counts": {},
            "visible_source_counts": {},
            "visible_entities": [],
        }
    count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
    visible = 0
    nearest = 0.0
    nearest_health = 0.0
    nearest_slot = -1
    nearest_source = "none"
    best_dist = float("inf")
    source_counts: Dict[str, int] = {}
    visible_source_counts: Dict[str, int] = {}
    visible_entities = []
    for idx, ent in enumerate(obs.entities[:count]):
        if hasattr(obs, "entity_debug") and idx < obs.entity_debug.shape[0]:
            debug = obs.entity_debug[idx]
            slot = int(debug[1])
            source = _source_name(int(debug[2]))
            state = _debug_state(debug)
        else:
            slot = -1
            source = "unknown"
            state = {}
        source_counts[source] = source_counts.get(source, 0) + 1
        if not (ent[7] > 0.5 and ent[8] > 0.5):
            continue
        dist = float(np.linalg.norm(ent[:3]))
        visible += 1
        visible_source_counts[source] = visible_source_counts.get(source, 0) + 1
        if len(visible_entities) < 4:
            visible_entities.append({
                "slot": slot,
                "source": source,
                "state": state,
                "distance": round(dist, 1),
                "health": round(float(ent[6]), 1),
            })
        if dist < best_dist:
            best_dist = dist
            nearest = dist
            nearest_health = float(ent[6])
            nearest_slot = slot
            nearest_source = source
    return {
        "visible_enemies": float(visible),
        "nearest_enemy": float(nearest if visible else 0.0),
        "nearest_enemy_health": float(nearest_health if visible else 0.0),
        "nearest_enemy_slot": int(nearest_slot),
        "nearest_enemy_source": nearest_source,
        "entity_source_counts": source_counts,
        "visible_source_counts": visible_source_counts,
        "visible_entities": visible_entities,
    }


def evaluate_map(policy: Q2BotPolicy, device: torch.device, args, map_name: str, server_id: int) -> Dict[str, object]:
    opponent_mode = args.opponent
    if args.spectator_only and opponent_mode == "human":
        opponent_mode = "mirror"

    default_ml_bots = 2 if opponent_mode in {"mirror", "scripted"} else 1
    num_ml_bots = int(args.num_ml_bots) if int(args.num_ml_bots) > 0 else default_ml_bots
    if opponent_mode == "human":
        n_bots = num_ml_bots
    elif opponent_mode in {"mirror", "scripted"}:
        n_bots = num_ml_bots
    else:
        n_bots = max(2, num_ml_bots + 1)
    ml_slot = args.ml_slot
    if num_ml_bots > 1 and ml_slot > args.maxclients - num_ml_bots:
        ml_slot = args.maxclients - num_ml_bots

    env = Q2MultiEnv(
        server_id=server_id,
        map_name=map_name,
        map_pool=[map_name],
        n_bots=n_bots,
        port_offset=args.port_offset + server_id * 4,
        maxclients=args.maxclients,
        ml_slot=ml_slot,
        num_ml_bots=num_ml_bots,
        game_seed=(
            None if args.game_seed < 0
            else args.game_seed + server_id * 1009
        ),
        max_ep_steps=max(args.steps + 5, 100),
        timedemo=args.timedemo,
        timelimit=args.timelimit,
        fraglimit=args.fraglimit,
        harness_step_timeout_ms=args.harness_step_timeout_ms,
        ml_step_timeout_ms=args.ml_step_timeout_ms,
        start_observer=False,
        spectator_only=bool(args.spectator_only),
    )
    row = {
        "map": map_name,
        "steps": 0,
        "episodes": 0,
        "timeouts": 0,
        "entity_steps": 0,
        "visible_steps": 0,
        "aim_aligned_steps": 0,
        "fire_steps": 0,
        "rocket_steps": 0,
        "tactical_updates": 0,
        "tactical_errors": 0,
        "tactical_tactics": {},
        "random_actions": 0,
        "ml_distance_xy": 0.0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "kills": 0.0,
        "deaths": 0.0,
        "opponent": opponent_mode,
        "spectator_only": bool(args.spectator_only),
        "ml_slots": list(env.bot_slots),
        "stochastic": bool(args.stochastic),
        "random_action_prob": float(args.random_action_prob),
        "action_noise": float(args.action_noise),
    }
    tactical = None
    last_info: Dict[str, object] = {}
    rng = random.Random(args.seed + server_id * 1009)
    if args.tactical_model:
        tactical = OllamaTacticalReasoner(
            model=args.tactical_model,
            host=args.ollama_host,
            interval_steps=args.tactical_interval,
            timeout_s=args.tactical_timeout,
            keep_alive=args.tactical_keep_alive,
        )
    start = time.time()
    try:
        obs_list = env.reset_all()
        hx_list = [policy.init_hidden(1, device) for _ in obs_list]
        last_ml_pos = (
            np.array(env._last_obs[0].self_state[:3], dtype=np.float32)
            if env._last_obs and env._last_obs[0] is not None else None
        )
        last_slot_pos = [
            np.array(obs.self_state[:3], dtype=np.float32) if obs is not None else None
            for obs in env._last_obs
        ]
        slot_distance_xy = [0.0 for _ in env._last_obs]
        for step in range(args.steps):
            actions = []
            for idx, obs_vec in enumerate(obs_list):
                if opponent_mode == "scripted" and idx > 0:
                    action = _scripted_opponent_action(step + idx * 17)
                else:
                    action, _value, _logp, hx_list[idx] = policy.act(
                        obs_vec,
                        hx_list[idx],
                        device=device,
                        deterministic=not args.stochastic,
                    )
                actions.append(action)
            if not actions:
                break

            for action_idx in range(len(actions)):
                if action_idx == 0 or opponent_mode != "scripted":
                    actions[action_idx], fully_random = _apply_action_randomization(
                        actions[action_idx], rng, args
                    )
                    row["random_actions"] += int(fully_random)
            action = actions[0]

            obs_current = env._last_obs[0]
            if tactical and obs_current is not None:
                intent, updated = tactical.maybe_update(
                    obs_current, last_info, int(row["steps"])
                )
                if updated:
                    row["tactical_updates"] += 1
                tactic_counts = row["tactical_tactics"]
                tactic_counts[intent.tactic] = tactic_counts.get(intent.tactic, 0) + 1
                if args.tactical_apply:
                    action = apply_intent_to_action(action, intent, obs_current)
                    actions[0] = action
            results = env.step_all(actions)
            obs, _reward, term, trunc, info = results[0]
            obs_obj = env._last_obs[0]
            obs_list = [result[0] for result in results]
            last_info = info
            if obs_obj is not None:
                ml_pos = np.array(obs_obj.self_state[:3], dtype=np.float32)
                if last_ml_pos is not None:
                    row["ml_distance_xy"] += float(np.linalg.norm(ml_pos[:2] - last_ml_pos[:2]))
                last_ml_pos = ml_pos
            for idx, slot_obs in enumerate(env._last_obs):
                if slot_obs is None:
                    continue
                slot_pos = np.array(slot_obs.self_state[:3], dtype=np.float32)
                if idx >= len(last_slot_pos):
                    last_slot_pos.append(None)
                    slot_distance_xy.append(0.0)
                if last_slot_pos[idx] is not None:
                    slot_distance_xy[idx] += float(np.linalg.norm(slot_pos[:2] - last_slot_pos[idx][:2]))
                last_slot_pos[idx] = slot_pos
            row["steps"] += 1
            row["entity_steps"] += int(obs_obj is not None and obs_obj.entity_count > 0)
            row["visible_steps"] += int(float(info.get("enemy_visible_any", 0.0)) > 0.0)
            row["aim_aligned_steps"] += int(float(info.get("aim_aligned", 0.0)) > 0.0)
            row["fire_steps"] += int(float(action[5]) > 0.5)
            row["rocket_steps"] += int(round(float(action[7])) == 7)
            row["damage_dealt"] += float(info.get("damage_dealt", 0.0))
            row["damage_taken"] += float(info.get("damage_taken", 0.0))
            row["kills"] += float(info.get("kills", 0.0))
            row["deaths"] += float(info.get("deaths", 0.0))
            row["timeouts"] += int(bool(info.get("timeout")))
            if args.report_interval > 0 and int(row["steps"]) % args.report_interval == 0:
                ml_slot = int(obs_obj.bot_slot) if obs_obj is not None else -1
                ml_source = (
                    _source_name(int(obs_obj.self_debug[2]))
                    if obs_obj is not None and hasattr(obs_obj, "self_debug") else "unknown"
                )
                ml_state = (
                    _debug_state(obs_obj.self_debug)
                    if obs_obj is not None and hasattr(obs_obj, "self_debug") else {}
                )
                live = {
                    "map": map_name,
                    "step": int(row["steps"]),
                    "seconds": round(time.time() - start, 2),
                    "ml_slot": ml_slot,
                    "ml_source": ml_source,
                    "ml_state": ml_state,
                    "ml_pos": _round_vec(obs_obj.self_state[:3]) if obs_obj is not None else [],
                    "ml_vel": _round_vec(obs_obj.self_state[3:6]) if obs_obj is not None else [],
                    "ml_speed_xy": (
                        round(float(np.linalg.norm(obs_obj.self_state[3:5])), 2)
                        if obs_obj is not None else 0.0
                    ),
                    "ml_distance_xy": round(float(row["ml_distance_xy"]), 2),
                    "bot_slots": [
                        _slot_snapshot(
                            slot_obs,
                            actions[idx] if idx < len(actions) else None,
                            slot_distance_xy[idx] if idx < len(slot_distance_xy) else 0.0,
                        )
                        for idx, slot_obs in enumerate(env._last_obs)
                    ],
                    "last_action": _action_snapshot(action),
                    "engine_action": _engine_action_snapshot(obs_obj),
                    "timeout": bool(info.get("timeout")),
                    "health": float(obs_obj.self_state[6]) if obs_obj is not None else 0.0,
                    "armor": float(obs_obj.self_state[7]) if obs_obj is not None else 0.0,
                    "weapon": float(obs_obj.self_state[8]) if obs_obj is not None else 0.0,
                    "ammo": float(obs_obj.self_state[9]) if obs_obj is not None else 0.0,
                    "kills": float(row["kills"]),
                    "deaths": float(row["deaths"]),
                    "damage_dealt": round(float(row["damage_dealt"]), 2),
                    "damage_taken": round(float(row["damage_taken"]), 2),
                    "spatial_bonus": round(float(info.get("spatial_bonus", 0.0)), 4),
                    "voxel_visited": float(info.get("voxel_visited", 0.0)),
                    "aim_aligned": float(info.get("aim_aligned", 0.0)),
                    "random_actions": int(row["random_actions"]),
                    **_enemy_snapshot(obs_obj),
                }
                line = "LIVE " + json.dumps(live, sort_keys=True)
                print(line, flush=True)
                if args.report_file:
                    report_path = Path(args.report_file)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    with report_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(live, sort_keys=True) + "\n")
            if args.step_sleep > 0.0:
                time.sleep(args.step_sleep)
            for idx, result in enumerate(results):
                if result[2] or result[3]:
                    if idx == 0:
                        row["episodes"] += 1
                        last_info = {}
                        last_ml_pos = None
                    if idx < len(last_slot_pos):
                        last_slot_pos[idx] = None
                    obs_list[idx] = env.reset_slot(idx)
                    hx_list[idx] = policy.init_hidden(1, device)
    finally:
        if tactical:
            row["tactical_errors"] = tactical.error_count
            if tactical.last_error:
                row["tactical_last_error"] = tactical.last_error
        env.close()

    row["kd_ratio"] = round(_safe_ratio(float(row["kills"]), float(row["deaths"])), 4)
    row["visible_rate"] = round(float(row["visible_steps"]) / max(int(row["steps"]), 1), 4)
    row["aim_rate"] = round(float(row["aim_aligned_steps"]) / max(int(row["steps"]), 1), 4)
    row["seconds"] = round(time.time() - start, 2)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "policy_04916232.pt"))
    parser.add_argument("--map_name", default="q2dm1")
    parser.add_argument("--map_glob", default="mltrain_*.bsp")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--max_maps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--port_offset", type=int, default=460)
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=11)
    parser.add_argument(
        "--num_ml_bots",
        type=int,
        default=0,
        help="override ML-controlled bot count; 0 keeps the opponent-mode default",
    )
    parser.add_argument(
        "--opponent",
        choices=("3zb2", "mirror", "scripted", "human"),
        default="3zb2",
        help="opponent controller: 3zb2 route bot, same policy, scripted ML slot, or human client",
    )
    parser.add_argument(
        "--spectator_only",
        action="store_true",
        help="force connected human clients to observe only; human opponent mode becomes mirror bot 1v1",
    )
    parser.add_argument(
        "--timedemo",
        type=int,
        default=1,
        help="q2ded timedemo mode; use 0 for real-time watchable matches",
    )
    parser.add_argument(
        "--step_sleep",
        type=float,
        default=0.0,
        help="seconds to sleep after each policy step; useful for watch sessions",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=0.0,
        help="server timelimit in minutes; 0 disables in-game round timeout",
    )
    parser.add_argument(
        "--fraglimit",
        type=int,
        default=0,
        help="server fraglimit; 0 disables in-game frag limit",
    )
    parser.add_argument(
        "--harness_step_timeout_ms",
        type=int,
        default=int(os.environ.get("Q2_HARNESS_STEP_TIMEOUT_MS", "300")),
        help="Python observation wait budget per policy step",
    )
    parser.add_argument(
        "--ml_step_timeout_ms",
        type=int,
        default=int(os.environ.get("Q2_ML_STEP_TIMEOUT_MS", "250")),
        help="engine wait budget for an ML action matching the current tick",
    )
    parser.add_argument("--target_kd", type=float, default=15.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--game_seed", type=int, default=-1,
        help="gameplay RNG seed; negative preserves normal game randomness",
    )
    parser.add_argument(
        "--shuffle_maps",
        action="store_true",
        help="shuffle the discovered map list before evaluation",
    )
    parser.add_argument(
        "--random_action_prob",
        type=float,
        default=0.0,
        help="chance per step to replace the policy action with a random exploration action",
    )
    parser.add_argument(
        "--action_noise",
        type=float,
        default=0.0,
        help="bounded Gaussian perturbation applied to policy movement and aim",
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=0,
        help="print LIVE JSON status every N policy steps",
    )
    parser.add_argument(
        "--report_file",
        default="",
        help="optional JSONL path for live monitor rows",
    )
    parser.add_argument(
        "--tactical_model",
        default=os.environ.get("Q2_TACTICAL_MODEL", ""),
        help=f"optional Ollama tactical sidecar model, e.g. {LIVE_MODEL}",
    )
    parser.add_argument(
        "--ollama_host",
        default=os.environ.get("Q2_OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama API endpoint for tactical sidecar",
    )
    parser.add_argument(
        "--tactical_interval",
        type=int,
        default=int(os.environ.get("Q2_TACTICAL_INTERVAL", "20")),
        help="steps between tactical LLM updates",
    )
    parser.add_argument(
        "--tactical_timeout",
        type=float,
        default=float(os.environ.get("Q2_TACTICAL_TIMEOUT", "10.0")),
        help="seconds before a tactical LLM update is skipped",
    )
    parser.add_argument(
        "--tactical_keep_alive",
        default=os.environ.get("Q2_TACTICAL_KEEP_ALIVE", "10m"),
        help="Ollama keep_alive value for the tactical model",
    )
    parser.add_argument(
        "--tactical_apply",
        action="store_true",
        help="apply conservative tactical action modifiers; default is log-only",
    )
    args = parser.parse_args()

    device = _pick_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    maps = discover_map_pool(
        map_name=args.map_name,
        map_glob=args.map_glob,
        map_dir=args.map_dir or None,
    )
    if args.shuffle_maps:
        random.Random(args.seed).shuffle(maps)
    if args.max_maps > 0:
        maps = maps[:args.max_maps]

    print(f"checkpoint={ckpt}")
    print(f"device={device}")
    print(f"q2_root={os.environ.get('Q2_ROOT')}")
    print(f"maps={maps}")
    effective_opponent = "mirror" if args.spectator_only and args.opponent == "human" else args.opponent
    setup_ml_bots = (
        int(args.num_ml_bots)
        if int(args.num_ml_bots) > 0
        else (2 if effective_opponent in {"mirror", "scripted"} else 1)
    )
    if args.opponent == "human" and not args.spectator_only:
        print(f"setup={setup_ml_bots} ML bot(s) + human player")
    else:
        print(f"setup={setup_ml_bots} ML bot(s) + {effective_opponent} opponent mode")
    if args.spectator_only:
        print("spectator_only=1 connected clients remain observers")
    if args.stochastic or args.random_action_prob > 0.0 or args.action_noise > 0.0:
        print(
            "exploration="
            f"stochastic:{int(args.stochastic)} "
            f"random_action_prob:{args.random_action_prob} "
            f"action_noise:{args.action_noise}"
        )
    if args.timedemo == 0:
        print(f"watch_connect=127.0.0.1:{27910 + args.port_offset}")
    print(
        "timing="
        f"timelimit:{args.timelimit:g} "
        f"fraglimit:{args.fraglimit} "
        f"harness_step_timeout_ms:{args.harness_step_timeout_ms} "
        f"ml_step_timeout_ms:{args.ml_step_timeout_ms}"
    )
    if args.tactical_model:
        mode = "apply" if args.tactical_apply else "log-only"
        print(
            f"tactical_sidecar={args.tactical_model} host={args.ollama_host} "
            f"interval={args.tactical_interval} mode={mode}"
        )

    rows: List[Dict[str, object]] = []
    for idx, map_name in enumerate(maps):
        row = evaluate_map(policy, device, args, map_name, idx)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    kills = sum(float(row["kills"]) for row in rows)
    deaths = sum(float(row["deaths"]) for row in rows)
    kd_ratio = _safe_ratio(kills, deaths)
    summary = {
        "checkpoint": str(ckpt),
        "maps": len(rows),
        "steps_per_map": args.steps,
        "total_kills": kills,
        "total_deaths": deaths,
        "kd_ratio": round(kd_ratio, 4),
        "target_kd": args.target_kd,
        "target_met": kd_ratio >= args.target_kd,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
