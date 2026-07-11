#!/usr/bin/env python3
"""Low-latency live sparring harness for human vs ML bots.

Each ML slot owns an IO worker.  Workers respond to observations immediately
with an action for the same server tick, while the main thread samples telemetry.
Frame rows are indexed by round_id, slot, and server tick so dropped or late
frames can be aligned offline.
"""

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
LOCAL_Q2_ROOT = ROOT.parent / "q2_lithium_merge"
if LOCAL_Q2_ROOT.exists():
    os.environ.setdefault("Q2_ROOT", str(LOCAL_Q2_ROOT))
os.environ.setdefault("Q2_POLICY_STATEFUL", "0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_1v1 import (  # noqa: E402
    _action_snapshot,
    _apply_action_randomization,
    _debug_state,
    _enemy_snapshot,
    _engine_action_snapshot,
    _latest_checkpoint,
    _pick_device,
    _round_vec,
    _slot_snapshot,
)
from harness.env import Q2MultiEnv, _decode_action  # noqa: E402
from harness.protocol import OBS_SIZE, pack_action, parse_obs  # noqa: E402
from harness.spatial import VoxelSpatialReward  # noqa: E402
from models.policy import Q2BotPolicy  # noqa: E402


class JsonlWriter:
    def __init__(self, path: str):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Dict[str, object]) -> None:
        if not self.path:
            return
        line = json.dumps(row, sort_keys=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class SlotState:
    def __init__(self, slot: int):
        self.lock = threading.Lock()
        self.slot = slot
        self.latest_obs = None
        self.latest_action: Optional[np.ndarray] = None
        self.latest_info: Dict[str, object] = {}
        self.frames = 0
        self.actions = 0
        self.random_actions = 0
        self.fire_gate_suppressed = 0
        self.invalid_packets = 0
        self.recv_errors = 0
        self.last_recv_ns = 0
        self.last_action_ns = 0
        self.distance_xy = 0.0
        self.last_pos: Optional[np.ndarray] = None
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.kills = 0.0
        self.deaths = 0.0


class SlotWorker(threading.Thread):
    def __init__(
        self,
        *,
        slot_idx: int,
        slot: int,
        sock,
        policy: Q2BotPolicy,
        policy_lock: threading.Lock,
        device: torch.device,
        deterministic: bool,
        rng: random.Random,
        args,
        state: SlotState,
        stop_event: threading.Event,
        frame_writer: JsonlWriter,
        round_id: int,
    ):
        super().__init__(name=f"slot-{slot}", daemon=True)
        self.slot_idx = slot_idx
        self.slot = slot
        self.sock = sock
        self.policy = policy
        self.policy_lock = policy_lock
        self.device = device
        self.deterministic = deterministic
        self.rng = rng
        self.args = args
        self.state = state
        self.stop_event = stop_event
        self.frame_writer = frame_writer
        self.round_id = round_id
        self.hx = policy.init_hidden(1, device)
        self.spatial = VoxelSpatialReward.from_env()
        self.spatial_ready = False

    def run(self) -> None:
        self.sock.settimeout(0.2)
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                if not self.stop_event.is_set():
                    with self.state.lock:
                        self.state.recv_errors += 1
                continue

            recv_ns = time.time_ns()
            obs = parse_obs(data)
            if obs is None:
                with self.state.lock:
                    self.state.invalid_packets += 1
                self.frame_writer.write({
                    "round_id": self.round_id,
                    "slot": self.slot,
                    "recv_ns": recv_ns,
                    "invalid_len": len(data),
                    "expected_len": OBS_SIZE,
                })
                continue

            if not self.spatial_ready:
                self.spatial.reset(self.args.map_name, obs)
                self.spatial_ready = True

            if obs.is_terminal:
                self.hx = self.policy.init_hidden(1, self.device)
                self.spatial.reset(self.args.map_name, obs)

            self.spatial.update(obs)
            obs_vec = obs.to_vector(self.spatial.memory_features(obs))

            with self.policy_lock:
                action, _value, _logp, self.hx = self.policy.act(
                    obs_vec,
                    self.hx,
                    device=self.device,
                    deterministic=self.deterministic,
                )
            action, fully_random = _apply_action_randomization(action, self.rng, self.args)
            intended_weapon = int(round(float(action[7]))) if float(action[7]) > 0.0 else None
            fire_context = self.spatial.fire_context(obs, intended_weapon)
            fire_gate_suppressed = False
            if self.args.gate_fire and float(action[5]) > 0.5:
                can_fire = (
                    float(fire_context["enemy_visible_count"]) > 0.0 and
                    (
                        float(fire_context["aim_aligned"]) > 0.0 or
                        float(fire_context["splash_viable"]) > 0.0 or
                        (
                            float(fire_context["splash_weapon"]) > 0.0 and
                            float(fire_context["audio_contact"]) > 0.0
                        )
                    )
                )
                if not can_fire:
                    action = np.array(action, dtype=np.float32, copy=True)
                    action[5] = 0.0
                    fire_gate_suppressed = True

            try:
                self.sock.sendto(pack_action(_decode_action(action), obs.tick), addr)
                action_ns = time.time_ns()
            except OSError:
                action_ns = 0

            spatial_bonus, spatial_info = self.spatial.update(obs)
            pos = np.array(obs.self_state[:3], dtype=np.float32)
            with self.state.lock:
                if self.state.last_pos is not None:
                    self.state.distance_xy += float(
                        np.linalg.norm(pos[:2] - self.state.last_pos[:2])
                    )
                self.state.last_pos = pos
                self.state.latest_obs = obs
                self.state.latest_action = action
                self.state.latest_info = {
                    "spatial_bonus": float(spatial_bonus),
                    **spatial_info,
                    "fire_gate_suppressed": bool(fire_gate_suppressed),
                }
                self.state.frames += 1
                self.state.actions += int(action_ns > 0)
                self.state.random_actions += int(fully_random)
                self.state.fire_gate_suppressed += int(fire_gate_suppressed)
                self.state.last_recv_ns = recv_ns
                self.state.last_action_ns = action_ns
                self.state.damage_dealt += float(obs.reward_damage_dealt)
                self.state.damage_taken += float(obs.reward_damage_taken)
                self.state.kills += float(obs.reward_kill)
                self.state.deaths += float(obs.reward_death)

            enemy = _enemy_snapshot(obs)
            frame_row = {
                "round_id": self.round_id,
                "slot": self.slot,
                "tick": int(obs.tick),
                "recv_ns": recv_ns,
                "action_sent_ns": action_ns,
                "response_ms": round((action_ns - recv_ns) / 1_000_000.0, 3)
                if action_ns else None,
                "action": _action_snapshot(action),
                "engine_action": _engine_action_snapshot(obs),
                "fire_context": fire_context,
                "fire_gate_suppressed": bool(fire_gate_suppressed),
                "spatial": spatial_info,
                "state": _debug_state(obs.self_debug),
                "pos": _round_vec(obs.self_state[:3]),
                "vel": _round_vec(obs.self_state[3:6]),
                "health": float(obs.self_state[6]),
                "armor": float(obs.self_state[7]),
                "weapon_id": float(obs.self_state[8]),
                "ammo": float(obs.self_state[9]),
                "reward": round(float(obs.reward + spatial_bonus), 4),
                "terminal": bool(obs.is_terminal),
                "terminal_reason": int(obs.terminal_reason),
                "nearest_enemy_slot": enemy["nearest_enemy_slot"],
                "nearest_enemy_source": enemy["nearest_enemy_source"],
                "visible_source_counts": enemy["visible_source_counts"],
            }
            self.frame_writer.write(frame_row)


def _snapshot(states, start_time: float, round_id: int) -> Dict[str, object]:
    bot_slots = []
    human_seen = False
    visible_sources: Dict[str, int] = {}
    total_frames = 0
    total_actions = 0
    total_random = 0
    total_fire_gate_suppressed = 0
    total_damage_dealt = 0.0
    total_damage_taken = 0.0
    total_kills = 0.0
    total_deaths = 0.0

    for state in states:
        with state.lock:
            obs = state.latest_obs
            action = None if state.latest_action is None else state.latest_action.copy()
            distance_xy = state.distance_xy
            frames = state.frames
            actions = state.actions
            random_actions = state.random_actions
            fire_gate_suppressed = state.fire_gate_suppressed
            invalid_packets = state.invalid_packets
            recv_errors = state.recv_errors
            last_recv_ns = state.last_recv_ns
            last_action_ns = state.last_action_ns
            damage_dealt = state.damage_dealt
            damage_taken = state.damage_taken
            kills = state.kills
            deaths = state.deaths

        total_frames += frames
        total_actions += actions
        total_random += random_actions
        total_fire_gate_suppressed += fire_gate_suppressed
        total_damage_dealt += damage_dealt
        total_damage_taken += damage_taken
        total_kills += kills
        total_deaths += deaths

        slot_row = _slot_snapshot(obs, action, distance_xy) if obs is not None else {
            "slot": state.slot,
        }
        slot_row.update({
            "frames": frames,
            "actions_sent": actions,
            "fire_gate_suppressed": fire_gate_suppressed,
            "invalid_packets": invalid_packets,
            "recv_errors": recv_errors,
            "last_recv_ns": last_recv_ns,
            "last_action_ns": last_action_ns,
        })
        bot_slots.append(slot_row)

        if obs is not None:
            enemy = _enemy_snapshot(obs)
            human_seen = human_seen or "human" in enemy["entity_source_counts"]
            for source, count in enemy["visible_source_counts"].items():
                visible_sources[source] = visible_sources.get(source, 0) + int(count)

    elapsed = max(time.time() - start_time, 0.001)
    return {
        "round_id": round_id,
        "seconds": round(elapsed, 2),
        "frames": int(total_frames),
        "actions_sent": int(total_actions),
        "capture_fps": round(total_frames / elapsed, 2),
        "per_bot_capture_fps": round(total_frames / max(len(states), 1) / elapsed, 2),
        "random_actions": int(total_random),
        "fire_gate_suppressed": int(total_fire_gate_suppressed),
        "human_seen": bool(human_seen),
        "visible_source_counts": visible_sources,
        "damage_dealt": round(total_damage_dealt, 2),
        "damage_taken": round(total_damage_taken, 2),
        "kills": round(total_kills, 2),
        "deaths": round(total_deaths, 2),
        "bot_slots": bot_slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "policy_04916232.pt"))
    parser.add_argument("--map_name", default="mltrain_00000000")
    parser.add_argument("--port_offset", type=int, default=944)
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=10)
    parser.add_argument("--num_ml_bots", type=int, default=2)
    parser.add_argument(
        "--session_seconds",
        type=float,
        default=180.0,
        help="wall-clock runtime; 0 or negative runs until interrupted",
    )
    parser.add_argument("--timelimit", type=float, default=3.0)
    parser.add_argument("--fraglimit", type=int, default=10)
    parser.add_argument("--timedemo", type=int, default=0)
    parser.add_argument("--ml_step_timeout_ms", type=int, default=35)
    parser.add_argument("--harness_step_timeout_ms", type=int, default=80)
    parser.add_argument("--report_hz", type=float, default=1.0)
    parser.add_argument("--report_file", default="runs/live_async_summary.jsonl")
    parser.add_argument("--frame_file", default="runs/live_async_frames.jsonl")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--random_action_prob", type=float, default=0.02)
    parser.add_argument("--action_noise", type=float, default=0.04)
    parser.add_argument(
        "--gate_fire",
        action="store_true",
        help="suppress live fire unless the current observation has an aligned visible enemy",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = _pick_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    round_id = int(time.time())
    env = Q2MultiEnv(
        server_id=0,
        map_name=args.map_name,
        map_pool=[args.map_name],
        n_bots=args.num_ml_bots,
        port_offset=args.port_offset,
        maxclients=args.maxclients,
        ml_slot=args.ml_slot,
        num_ml_bots=args.num_ml_bots,
        max_ep_steps=max(int(args.session_seconds * 20), 100),
        timedemo=args.timedemo,
        timelimit=args.timelimit,
        fraglimit=args.fraglimit,
        harness_step_timeout_ms=args.harness_step_timeout_ms,
        ml_step_timeout_ms=args.ml_step_timeout_ms,
        console_pipe=True,
    )

    stop_event = threading.Event()

    def _stop(_signum=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    summary_writer = JsonlWriter(args.report_file)
    frame_writer = JsonlWriter(args.frame_file)
    policy_lock = threading.Lock()
    workers = []
    states = []

    print(f"checkpoint={ckpt}")
    print(f"device={device}")
    print(f"watch_connect=127.0.0.1:{27910 + args.port_offset}")
    print(
        "async_live="
        f"slots:{args.ml_slot}-{args.ml_slot + args.num_ml_bots - 1} "
        f"session_seconds:{args.session_seconds:g} "
        f"timelimit:{args.timelimit:g} fraglimit:{args.fraglimit} "
        f"ml_step_timeout_ms:{args.ml_step_timeout_ms} "
        f"gate_fire:{int(args.gate_fire)}"
    )

    try:
        env.reset_all()
        start_time = time.time()
        for idx, slot in enumerate(env.bot_slots):
            state = SlotState(slot)
            states.append(state)
            worker = SlotWorker(
                slot_idx=idx,
                slot=slot,
                sock=env._socks[idx],
                policy=policy,
                policy_lock=policy_lock,
                device=device,
                deterministic=not args.stochastic,
                rng=random.Random(args.seed + idx * 1009),
                args=args,
                state=state,
                stop_event=stop_event,
                frame_writer=frame_writer,
                round_id=round_id,
            )
            workers.append(worker)
            worker.start()

        report_period = 1.0 / max(float(args.report_hz), 0.1)
        deadline = (
            None if float(args.session_seconds) <= 0.0
            else start_time + float(args.session_seconds)
        )
        while not stop_event.is_set() and (
            deadline is None or time.time() < deadline
        ):
            time.sleep(report_period)
            row = _snapshot(states, start_time, round_id)
            line = "LIVE_ASYNC " + json.dumps(row, sort_keys=True)
            print(line, flush=True)
            summary_writer.write(row)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=1.0)
        env.close()

    row = _snapshot(states, start_time if "start_time" in locals() else time.time(), round_id)
    print("SUMMARY " + json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
