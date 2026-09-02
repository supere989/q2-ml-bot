"""Prove serial/pipelined collector equivalence on a recorded live session.

Replays a telemetry session recorded by tools/collect_profile.py (raw wire
packets) through BOTH the serial collect_round loop and the pipelined
collect_rounds_pipelined driver, using real post-processing (spatial reward,
feature extraction) on byte-identical inputs, and asserts the emitted
BatchRound sequences are identical.

    python3 tools/collect_equivalence.py /tmp/session_v2.jsonl --rounds 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from harness.client_batch import Q2NetworkClientBatch
from harness.client_env import Q2NetworkClientEnv
from harness.protocol import Action
from harness.telemetry_replay import ReplayEnv, action_from_fields, read_recording


def build_replay_envs(events_by_client, port_base: int):
    envs = []
    for index, events in enumerate(events_by_client):
        detached = Q2NetworkClientEnv(
            server="127.0.0.1:28200",
            telemetry_server="127.0.0.1:28201",
            telemetry_token="replay",
            client_binary="/bin/false",
            client_root="/tmp",
            harness_port=port_base + index,
            qport=49500 + port_base + index,
            client_id=f"prof-{index:02d}",
            name=f"prof-{index:02d}",
            spatial_seed=7 + index * 1009,
        )
        env = ReplayEnv(detached, events)
        env._dispatched = 0
        envs.append(env)
    return envs


def action_script(events_by_client):
    """Per-client queue of recorded dispatch actions (peeked, not popped)."""
    return [
        [action_from_fields(event["action"]) for event in events
         if event["kind"] == "dispatch"]
        for events in events_by_client
    ]


class _counting_dispatch:
    """Count ReplayEnv dispatches so the action script can peek by index."""

    def __init__(self):
        self._original = ReplayEnv.dispatch_action

    def __enter__(self):
        original = self._original

        def counting(self, action):
            result = original(self, action)
            self._dispatched += 1
            return result

        ReplayEnv.dispatch_action = counting
        return self

    def __exit__(self, *exc):
        ReplayEnv.dispatch_action = self._original
        return False


def peek_actions(queues, envs):
    return [
        queues[index][env._dispatched]
        if env._dispatched < len(queues[index]) else Action()
        for index, env in enumerate(envs)
    ]


def run_serial(events_by_client, rounds: int):
    envs = build_replay_envs(events_by_client, 39600)
    queues = action_script(events_by_client)
    batch = Q2NetworkClientBatch(envs, vector=True, round_timeout=2.0)
    emitted = []
    try:
        batch.reset()
        with _counting_dispatch():
            for _ in range(rounds):
                emitted.append(batch.collect_round(
                    peek_actions(queues, envs), policy_version=0
                ))
    finally:
        metrics = batch.metrics
        batch.close()
    return emitted, metrics, envs


def run_pipelined(events_by_client, rounds: int):
    envs = build_replay_envs(events_by_client, 39700)
    queues = action_script(events_by_client)
    batch = Q2NetworkClientBatch(envs, vector=True, round_timeout=2.0)
    emitted = []
    try:
        observations, _infos = batch.reset()
        with _counting_dispatch():
            batch.collect_rounds_pipelined(
                observations,
                rounds=rounds,
                infer=lambda _vectors, _round_id: peek_actions(queues, envs),
                policy_version=0,
                on_round=emitted.append,
            )
    finally:
        metrics = batch.metrics
        batch.close()
    return emitted, metrics, envs


def compare_rounds(serial_rounds, pipelined_rounds) -> int:
    failures = 0
    if len(serial_rounds) != len(pipelined_rounds):
        print(f"FAIL: round count {len(serial_rounds)} != "
              f"{len(pipelined_rounds)}")
        return 1
    for index, (expected, actual) in enumerate(
        zip(serial_rounds, pipelined_rounds)
    ):
        problems = []
        if actual.round_id != expected.round_id:
            problems.append("round_id")
        if actual.policy_version != expected.policy_version:
            problems.append("policy_version")
        if actual.tags != expected.tags:
            problems.append("tags")
        if not np.array_equal(actual.observations, expected.observations):
            diff = np.abs(
                actual.observations - expected.observations
            )
            problems.append(f"observations(max|d|={diff.max():.3e})")
        if not np.array_equal(actual.rewards, expected.rewards):
            problems.append("rewards")
        if not np.array_equal(actual.terminated, expected.terminated):
            problems.append("terminated")
        if not np.array_equal(actual.truncated, expected.truncated):
            problems.append("truncated")
        if actual.infos != expected.infos:
            problems.append("infos")
        if problems:
            failures += 1
            print(f"FAIL round {index}: {', '.join(problems)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()

    events_by_client = read_recording(args.recording)
    print(f"recording: {len(events_by_client)} clients, "
          f"{[len(events) for events in events_by_client]} events")

    serial_rounds, serial_metrics, serial_envs = run_serial(
        events_by_client, args.rounds
    )
    pipelined_rounds, pipelined_metrics, pipelined_envs = run_pipelined(
        events_by_client, args.rounds
    )

    failures = compare_rounds(serial_rounds, pipelined_rounds)
    if serial_metrics != pipelined_metrics:
        failures += 1
        print(f"FAIL metrics:\n  serial    {serial_metrics}\n"
              f"  pipelined {pipelined_metrics}")
    for label, envs in (("serial", serial_envs),
                        ("pipelined", pipelined_envs)):
        leftovers = [
            len(env._events) - env._cursor for env in envs
        ]
        if any(leftovers):
            failures += 1
            print(f"FAIL {label}: unconsumed replay events {leftovers}")

    if failures:
        print(f"EQUIVALENCE FAIL ({failures} failures)")
        return 1
    print(f"EQUIVALENCE PASS: {len(serial_rounds)} rounds identical, "
          f"transitions_accepted={serial_metrics.transitions_accepted}, "
          f"catchup_resyncs={serial_metrics.realtime_catchup_resyncs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
