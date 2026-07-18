#!/usr/bin/env python3
"""Prove client_id isolation for multiple normal players on one q2ded."""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.client_env import Q2NetworkClientEnv
from harness.protocol import Action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:28200")
    parser.add_argument("--telemetry_server", default="127.0.0.1:28201")
    parser.add_argument("--telemetry_token", required=True)
    parser.add_argument("--client_binary", required=True)
    parser.add_argument("--client_root", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    common = dict(
        server=args.server,
        telemetry_server=args.telemetry_server,
        telemetry_token=args.telemetry_token,
        client_binary=args.client_binary,
        client_root=args.client_root,
    )
    envs = [
        Q2NetworkClientEnv(
            **common, client_id="scale-client-a", name="scale-a", harness_port=39010
        ),
        Q2NetworkClientEnv(
            **common, client_id="scale-client-b", name="scale-b", harness_port=39011
        ),
    ]
    try:
        first = [env.start() for env in envs]
        slots = {sample.client_slot for sample in first}
        assert slots == {first[0].client_slot, first[1].client_slot}
        assert len(slots) == 2, f"client IDs collided on slot {slots}"
        assert first[0].client_id == "scale-client-a"
        assert first[1].client_id == "scale-client-b"

        frames = [sample.server_frame for sample in first]
        initial_positions = [
            tuple(float(value) for value in sample.observation.self_state[:3])
            for sample in first
        ]
        max_displacements = [0.0, 0.0]
        saw_route_echo = [False, False]
        for step in range(args.steps):
            for index, env in enumerate(envs):
                obs, _reward, _terminal, _truncated, info = env.step(
                    Action(
                        move_forward=0.7,
                        move_right=0.3 if index == 0 else -0.3,
                        look_yaw=2.0 if index == 0 else -2.0,
                    )
                )
                expected = f"scale-client-{'a' if index == 0 else 'b'}"
                assert info["client_id"] == expected
                assert info["client_slot"] == first[index].client_slot
                assert info["server_frame"] > frames[index]
                frames[index] = info["server_frame"]
                position = tuple(float(value) for value in obs.self_state[:3])
                max_displacements[index] = max(
                    max_displacements[index],
                    math.dist(initial_positions[index], position),
                )
                expected_side = 0.3 if index == 0 else -0.3
                saw_route_echo[index] |= (
                    abs(float(obs.action_debug[4]) - 0.7) < 0.05
                    and abs(float(obs.action_debug[5]) - expected_side) < 0.05
                )
        assert all(distance > 16.0 for distance in max_displacements), (
            "one or more routed clients failed to move through usercmds: "
            f"displacements={max_displacements}"
        )
        assert all(saw_route_echo), (
            "per-client action echoes crossed or failed: "
            f"route_echo={saw_route_echo}"
        )
        print(
            "PASS independent_routes=2 "
            f"slots={sorted(slots)} frames={frames} "
            f"displacements={[round(value, 1) for value in max_displacements]} "
            f"steps_each={args.steps}"
        )
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
