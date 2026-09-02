"""Profile the network-native collection loop on a live lane.

Runs Q2NetworkClientBatch against a running q2ded with stub (numpy) policy
inference and simulated rollout-worker buffer writes, instrumenting every
phase of the round cycle.  Optionally records the raw telemetry session for
deterministic replay (see harness/telemetry_replay.py).

Usage (server already running, token in /tmp/q2_local_token):

    python3 tools/collect_profile.py record \
        --rounds 300 --infer-ms 10 --seed-cells 4096 \
        --record /tmp/session.jsonl --report-out /tmp/profile.json

The report prints the per-round time split; the JSON keeps the raw samples.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import harness.client_env as client_env_mod
import harness.client_protocol as client_protocol_mod
from harness.client_batch import Q2NetworkClientBatch
from harness.client_env import Q2NetworkClientEnv
from harness.protocol import OBS_DIM, Action, Observation
from harness.spatial import SessionMemoryCell
from harness.telemetry_replay import SocketRecordingEnv

ACTION_DIM = 8
HIDDEN_DIM = 128


class PhaseStats:
    """Thread-safe wall-time accumulator for profiling wrappers."""

    def __init__(self):
        self._lock = threading.Lock()
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.samples: dict[str, list[float]] = {}

    def add(self, name: str, dt: float, keep: bool = False) -> None:
        with self._lock:
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            if keep:
                self.samples.setdefault(name, []).append(dt)

    @contextmanager
    def time(self, name: str, keep: bool = False):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started, keep=keep)


STATS = PhaseStats()


def _wrap(obj, attr: str, name: str, keep: bool = False):
    original = getattr(obj, attr)

    def timed(*args, **kwargs):
        with STATS.time(name, keep=keep):
            return original(*args, **kwargs)

    setattr(obj, attr, timed)
    return original


def instrument_env(env, spatial) -> None:
    _wrap(env, "drain_latest_telemetry", "drain")
    _wrap(env, "dispatch_action", "dispatch")
    _wrap(env, "transition_result", "post_transition")
    _wrap(env, "initial_result", "initial_result")
    _wrap(spatial, "update", "spatial_update", keep=True)
    _wrap(spatial, "_update_session_memory", "sm_deposit")
    _wrap(spatial, "_memory_features_internal", "sm_features", keep=True)
    _wrap(spatial, "_nearest_memory_signals", "sm_nearest", keep=True)
    _wrap(spatial, "_update_thermal_tracks", "sm_thermal")


def instrument_batch(batch) -> None:
    _wrap(batch, "_collect_echo", "echo_wait_validate", keep=True)
    _wrap(batch, "_collate_observations", "collate")


def instrument_parse() -> None:
    for module in (client_env_mod, client_protocol_mod):
        original = module.parse_client_telemetry

        def timed(data, _original=original):
            with STATS.time("parse_packet"):
                return _original(data)

        module.parse_client_telemetry = timed


def instrument_to_vector() -> None:
    original = Observation.to_vector

    def timed(self, session_memory=None):
        with STATS.time("to_vector"):
            return original(self, session_memory)

    Observation.to_vector = timed


class StubPolicy:
    """Deterministic numpy policy with a configurable fixed cost."""

    def __init__(self, seed: int = 0, infer_ms: float = 10.0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.05, size=(OBS_DIM, 64)).astype(np.float32)
        self.w2 = rng.normal(0, 0.1, size=(64, ACTION_DIM)).astype(np.float32)
        self.infer_s = max(0.0, float(infer_ms) / 1000.0)

    def act(self, obs_batch: np.ndarray) -> list[Action]:
        with STATS.time("infer", keep=True):
            hidden = np.tanh(obs_batch @ self.w1)
            raw = hidden @ self.w2
            # Fixed-cost spin to model the trainer's GPU inference latency.
            if self.infer_s > 0.0:
                deadline = time.perf_counter() + self.infer_s
                while time.perf_counter() < deadline:
                    pass
            actions = []
            for row in np.tanh(raw):
                actions.append(Action(
                    move_forward=float(row[0]),
                    move_right=float(row[1]),
                    # Small look deltas: large ones saturate the engine's
                    # ±89° pitch clamp, which breaks echo causality and
                    # forces nontrainable action-state resyncs every round.
                    look_yaw=float(3.0 * row[2]),
                    look_pitch=float(1.0 * row[3]),
                    jump=bool(row[4] > 0.5),
                    fire=bool(row[5] > 0.9),
                    hook=int(np.clip(round(row[6] * 2 + 2), 0, 3)),
                    weapon=0,
                ))
            return actions


def simulate_buffer_writes(arrays: dict, step: int, results) -> None:
    """Mirror the rollout_worker's per-step numpy buffer writes."""
    with STATS.time("buffer", keep=True):
        for index, result in enumerate(results):
            observation, reward, terminated, truncated, info = result
            arrays["obs"][step, index] = observation
            arrays["rewards"][step, index] = reward
            arrays["dones"][step, index] = bool(terminated or truncated)
            arrays["values"][step, index] = 0.0
            arrays["log_probs"][step, index] = 0.0


def seed_lattice_cells(batch: Q2NetworkClientBatch, n_cells: int,
                       map_name: str, seed: int = 1) -> None:
    """Pre-populate session memory so the profile sees a mature lattice."""
    rng = np.random.default_rng(seed)
    for env in batch.envs:
        spatial = env._spatial
        memory = spatial._memory_for_map(map_name)
        for _ in range(n_cells):
            cell = (
                int(rng.integers(-16, 16)),
                int(rng.integers(-16, 16)),
                int(rng.integers(-4, 4)),
            )
            entry = memory.setdefault(cell, SessionMemoryCell())
            entry.engagement_count += float(rng.random() * 4)
            entry.enemy_seen += float(rng.random() * 3)
            entry.damage_taken += float(rng.random() * 200)
            entry.damage_dealt += float(rng.random() * 200)
            entry.kills += float(rng.random() * 3)
            entry.deaths += float(rng.random() * 2)
            entry.hazard_damage += float(rng.random() * 50)


def build_envs(args, token: str) -> list[Q2NetworkClientEnv]:
    return [
        Q2NetworkClientEnv(
            server=args.server,
            telemetry_server=args.telemetry_server,
            telemetry_token=token,
            client_binary=args.client_binary,
            client_root=args.client_root,
            harness_port=args.harness_port_base + index,
            qport=args.qport_base + index,
            client_id=f"prof-{index:02d}",
            name=f"prof-{index:02d}",
            timeout=8.0,
            spatial_seed=args.spatial_seed + index * 1009,
        )
        for index in range(args.clients)
    ]


def cmd_record(args) -> int:
    token = Path(args.token_file).read_text().strip()
    envs = build_envs(args, token)
    record_handle = open(args.record, "w") if args.record else None
    if record_handle is not None:
        envs = [
            SocketRecordingEnv(env, index, record_handle)
            for index, env in enumerate(envs)
        ]
    batch = Q2NetworkClientBatch(envs, vector=True,
                                 round_timeout=args.round_timeout)
    policy = StubPolicy(seed=args.seed, infer_ms=args.infer_ms)
    n_envs = len(envs)
    arrays = {
        "obs": np.empty((args.rounds, n_envs, OBS_DIM), np.float32),
        "rewards": np.empty((args.rounds, n_envs), np.float32),
        "dones": np.empty((args.rounds, n_envs), np.uint8),
        "values": np.empty((args.rounds, n_envs), np.float32),
        "log_probs": np.empty((args.rounds, n_envs), np.float32),
    }

    instrument_parse()
    instrument_to_vector()
    instrument_batch(batch)
    for env in batch.envs:
        instrument_env(env, env._spatial)

    profiler = cProfile.Profile() if args.cprofile else None
    round_times: list[float] = []
    accepted_rounds = 0
    try:
        observations, _infos = batch.reset()
        if args.seed_cells > 0:
            map_name = str(_infos[0].get("map", "unknown"))
            seed_lattice_cells(batch, args.seed_cells, map_name)
        if profiler is not None:
            profiler.enable()
        wall_start = time.monotonic()
        if args.pipelined:
            emitted: list = []
            last_emit = [time.perf_counter()]

            def on_round(result):
                now = time.perf_counter()
                round_times.append(now - last_emit[0])
                last_emit[0] = now
                step = len(emitted)
                emitted.append(result)
                simulate_buffer_writes(arrays, step % args.rounds, list(zip(
                    list(result.observations),
                    list(result.rewards),
                    list(result.terminated),
                    list(result.truncated),
                    list(result.infos),
                )))

            batch.collect_rounds_pipelined(
                np.asarray(observations, dtype=np.float32),
                rounds=args.rounds,
                infer=lambda vectors, _round_id: policy.act(vectors),
                policy_version=0,
                on_round=on_round,
            )
            for result in emitted[args.warmup:]:
                if bool(result.infos[0].get("trainable_transition", False)):
                    accepted_rounds += 1
        else:
            for step in range(args.rounds):
                actions = policy.act(np.asarray(observations, dtype=np.float32))
                round_start = time.perf_counter()
                result = batch.collect_round(actions, policy_version=0)
                round_times.append(time.perf_counter() - round_start)
                simulate_buffer_writes(arrays, step % args.rounds, list(zip(
                    list(result.observations),
                    list(result.rewards),
                    list(result.terminated),
                    list(result.truncated),
                    list(result.infos),
                )))
                observations = result.observations
                if step >= args.warmup and bool(
                    result.infos[0].get("trainable_transition", False)
                ):
                    accepted_rounds += 1
        wall_total = time.monotonic() - wall_start
    finally:
        if profiler is not None:
            profiler.disable()
        batch.close()
        if record_handle is not None:
            record_handle.close()

    measured_rounds = max(0, args.rounds - args.warmup)
    round_samples = np.asarray(round_times[args.warmup:], dtype=np.float64)
    report = {
        "rounds": args.rounds,
        "warmup": args.warmup,
        "clients": args.clients,
        "infer_ms": args.infer_ms,
        "seed_cells": args.seed_cells,
        "wall_total_s": wall_total,
        "accepted_rounds": accepted_rounds,
        "round_wall_ms": {
            "mean": float(round_samples.mean() * 1000.0),
            "p50": float(np.percentile(round_samples, 50) * 1000.0),
            "p95": float(np.percentile(round_samples, 95) * 1000.0),
        },
        "phases_ms_per_round": {},
        "metrics": batch.metrics.as_dict(),
    }
    for name in sorted(STATS.totals):
        per_round = STATS.totals[name] / max(1, measured_rounds) * 1000.0
        entry = {"mean_ms_round": per_round, "calls_round":
                 STATS.counts[name] / max(1, measured_rounds)}
        if name in STATS.samples:
            samples = np.asarray(STATS.samples[name], dtype=np.float64)
            entry["p50_ms"] = float(np.percentile(samples, 50) * 1000.0)
            entry["p95_ms"] = float(np.percentile(samples, 95) * 1000.0)
        report["phases_ms_per_round"][name] = entry

    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=1) + "\n")
    print_report(report)
    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(
            "cumulative"
        ).print_stats(30)
        print(stream.getvalue())
    return 0


def print_report(report: dict) -> None:
    print(f"\n=== round profile: {report['clients']} clients, "
          f"infer={report['infer_ms']}ms, seed_cells={report['seed_cells']} ===")
    print(f"rounds={report['rounds']} warmup={report['warmup']} "
          f"wall={report['wall_total_s']:.1f}s "
          f"accepted={report['accepted_rounds']}")
    measured = max(1, report["rounds"] - report["warmup"])
    metrics = report["metrics"]
    print(f"rates: rounds/s={measured / report['wall_total_s']:.2f} "
          f"accepted_rounds/s={report['accepted_rounds'] / report['wall_total_s']:.2f} "
          f"accepted_transitions/s="
          f"{metrics.get('network_client/transitions_accepted', 0) / report['wall_total_s']:.1f} "
          f"catchups={metrics.get('network_client/realtime_catchup_resyncs', 0)} "
          f"echo_timeouts={metrics.get('network_client/echo_timeouts', 0)}")
    wall = report["round_wall_ms"]
    print(f"round wall: mean={wall['mean']:.2f}ms p50={wall['p50']:.2f}ms "
          f"p95={wall['p95']:.2f}ms")
    phases = report["phases_ms_per_round"]
    print(f"{'phase':<24}{'ms/round':>10}{'calls/round':>12}"
          f"{'p50 ms':>9}{'p95 ms':>9}")
    for name, entry in sorted(
        phases.items(), key=lambda item: -item[1]["mean_ms_round"]
    ):
        print(f"{name:<24}{entry['mean_ms_round']:>10.3f}"
              f"{entry['calls_round']:>12.2f}"
              f"{entry.get('p50_ms', 0.0):>9.3f}"
              f"{entry.get('p95_ms', 0.0):>9.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="live profiling run")
    rec.add_argument("--server", default="127.0.0.1:28200")
    rec.add_argument("--telemetry-server", default="127.0.0.1:28201")
    rec.add_argument("--client-binary",
                     default="/home/raymondj/q2-ml-client/release/quake2")
    rec.add_argument("--client-root",
                     default="/home/raymondj/q2-ml-client/release")
    rec.add_argument("--token-file", default="/tmp/q2_local_token")
    rec.add_argument("--clients", type=int, default=4)
    rec.add_argument("--harness-port-base", type=int, default=39100)
    rec.add_argument("--qport-base", type=int, default=49100)
    rec.add_argument("--rounds", type=int, default=300)
    rec.add_argument("--warmup", type=int, default=20)
    rec.add_argument("--infer-ms", type=float, default=10.0)
    rec.add_argument("--seed-cells", type=int, default=4096)
    rec.add_argument("--seed", type=int, default=0)
    rec.add_argument("--spatial-seed", type=int, default=7)
    rec.add_argument("--round-timeout", type=float, default=2.0)
    rec.add_argument("--record", default="")
    rec.add_argument("--report-out", default="")
    rec.add_argument("--cprofile", action="store_true")
    rec.add_argument("--pipelined", action="store_true",
                     help="drive rounds with collect_rounds_pipelined")
    rec.set_defaults(func=cmd_record)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
