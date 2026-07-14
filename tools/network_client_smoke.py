#!/usr/bin/env python3
"""Drive one normal-player client and print authoritative telemetry."""

import argparse
import math
from pathlib import Path
import sys
import time

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
    parser.add_argument("--harness_port", type=int, default=39000)
    parser.add_argument("--client_id", default="")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--vector", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    env = Q2NetworkClientEnv(
        server=args.server,
        telemetry_server=args.telemetry_server,
        telemetry_token=args.telemetry_token,
        client_binary=args.client_binary,
        client_root=args.client_root,
        harness_port=args.harness_port,
        client_id=args.client_id or None,
        debug=args.debug,
    )
    started = time.monotonic()
    try:
        if args.vector:
            vector, first_info = env.reset()
            print(
                f"connected client_id={first_info['client_id']} "
                f"slot={first_info['client_slot']} vector_dim={len(vector)}"
            )
        else:
            first = env.start()
            print(f"connected client_id={first.client_id} slot={first.client_slot}")
        initial_position = tuple(float(value) for value in env._last.observation.self_state[:3])
        max_displacement = 0.0
        saw_action_echo = False
        kills = damage = 0.0
        for step in range(args.steps):
            action = Action(
                move_forward=0.8,
                move_right=0.35 if (step // 20) % 2 == 0 else -0.35,
                look_yaw=3.0,
                fire=True,
            )
            if args.vector:
                vector, reward, terminal, _truncated, info = env.step_vector(action)
                obs = env._last.observation
            else:
                obs, reward, terminal, _truncated, info = env.step(action)
            kills += obs.reward_kill
            damage += obs.reward_damage_dealt
            position = tuple(float(value) for value in obs.self_state[:3])
            max_displacement = max(
                max_displacement, math.dist(initial_position, position)
            )
            saw_action_echo |= (
                abs(float(obs.action_debug[4]) - 0.8) < 0.05
                and abs(abs(float(obs.action_debug[5])) - 0.35) < 0.05
                and int(obs.action_debug[9]) == 1
            )
            if step % 10 == 0:
                print(
                    f"step={step} frame={info['server_frame']} "
                    f"pos={obs.self_state[:3].round(1).tolist()} "
                    f"vel={obs.self_state[3:6].round(1).tolist()} "
                    f"health={obs.self_state[6]:.0f} entities={obs.entity_count} "
                    f"echo=({obs.action_debug[4]:.2f},{obs.action_debug[5]:.2f},"
                    f"fire={int(obs.action_debug[9])}) reward={reward:.3f}"
                )
        elapsed = time.monotonic() - started
        assert max_displacement > 16.0, (
            "client received telemetry but did not move through normal usercmds: "
            f"max_displacement={max_displacement:.1f}"
        )
        assert saw_action_echo, "server never echoed the requested move/fire usercmd"
        print(
            f"PASS steps={args.steps} elapsed={elapsed:.2f}s "
            f"displacement={max_displacement:.1f} damage={damage:.1f} kills={kills:.0f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
