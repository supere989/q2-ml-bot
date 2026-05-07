"""
echo_bot.py — minimal UDP responder to verify the ml_bridge end-to-end.

Run BEFORE starting the q2ded server (so the bind succeeds first).  Listens
on ML_BASE_PORT+slot, decodes each ml_obs_t arriving from game.so, and
replies with a random ml_action_t.

Usage:
    python tools/echo_bot.py --slot 1
"""

import argparse
import socket
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.protocol import (
    ML_BASE_PORT, OBS_SIZE, ACT_SIZE,
    parse_obs, pack_action, Action,
)


def run(slot: int, mode: str = "random"):
    port = ML_BASE_PORT + slot
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(1.0)

    print(f"echo_bot listening on UDP {port} (slot {slot}, mode={mode})")

    n = 0
    last_print = time.time()
    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue

        if len(data) != OBS_SIZE:
            print(f"  warn: got {len(data)} bytes, expected {OBS_SIZE}")
            continue

        obs = parse_obs(data)
        if obs is None:
            print("  warn: bad magic")
            continue

        if mode == "random":
            act = Action(
                move_forward = float(np.random.uniform(-1, 1)),
                move_right   = float(np.random.uniform(-1, 1)),
                look_yaw     = float(np.random.uniform(-15, 15)),
                look_pitch   = 0.0,
                jump         = bool(np.random.random() < 0.05),
                fire         = bool(np.random.random() < 0.10),
                hook         = 0,
                weapon       = 0,
            )
        elif mode == "stand":
            act = Action()
        elif mode == "spin":
            act = Action(look_yaw=10.0)
        else:
            act = Action()

        sock.sendto(pack_action(act, obs.tick), addr)

        n += 1
        if time.time() - last_print > 2.0:
            pos = obs.self_state[0:3]
            hp  = obs.self_state[6]
            print(f"[{n:6d}] tick={obs.tick} pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) "
                  f"hp={hp:.0f} ents={obs.entity_count} reward={obs.reward:+.3f}")
            last_print = time.time()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slot", type=int, default=1)
    p.add_argument("--mode", choices=["random", "stand", "spin"], default="random")
    args = p.parse_args()
    try:
        run(args.slot, args.mode)
    except KeyboardInterrupt:
        print("\nbye")
