"""
monitor_stats.py — live dashboard for the ML bridge.

Listens on a bot's UDP slot, reads ml_obs_t packets, and prints a refreshing
terminal display showing:
  - data rate (obs/sec, EMA over 1s)
  - current bot state (pos, vel, hp, ammo)
  - reward signal flow (per-tick + running totals)
  - episode boundaries (terminal flags)

Sends zero actions back so the bot keeps reporting (use this INSTEAD of
dump_obs.py — it does the same thing plus stats).

Usage:
    python tools/monitor_stats.py --slot 7
"""

import argparse
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.protocol import (
    ML_BASE_PORT, OBS_SIZE, ML_OBS_MAGIC, ML_ACT_MAGIC,
    ACT_FMT, parse_obs,
)


def fmt_rate(n: int, dt: float) -> str:
    if dt <= 0:
        return "—"
    return f"{n / dt:>5.0f}/s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slot", type=int, default=7)
    p.add_argument("--no-action", action="store_true",
                   help="don't send action replies (bot will stall)")
    args = p.parse_args()

    port = ML_BASE_PORT + args.slot
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    s.settimeout(1.0)

    print(f"\033[2J\033[H", end="")  # clear screen
    print(f"monitor_stats listening UDP {port} (bot slot {args.slot})")
    print()

    # rolling window for rate
    pkt_times = deque(maxlen=2000)

    # totals
    n_pkts        = 0
    n_episodes    = 0
    ep_reward_sum = 0.0
    last_episode_reward = 0.0
    cumulative = {
        "damage_dealt": 0.0, "damage_taken": 0.0,
        "kill": 0.0, "death": 0.0,
        "item": 0.0, "hook": 0.0,
    }

    last_render = time.time()

    while True:
        try:
            data, addr = s.recvfrom(4096)
        except socket.timeout:
            continue

        now = time.time()
        if len(data) != OBS_SIZE:
            continue
        obs = parse_obs(data)
        if obs is None:
            continue

        n_pkts += 1
        pkt_times.append(now)

        # accumulate rewards into episode
        step_r = obs.reward
        ep_reward_sum += step_r
        cumulative["damage_dealt"] += obs.reward_damage_dealt
        cumulative["damage_taken"] += obs.reward_damage_taken
        cumulative["kill"]         += obs.reward_kill
        cumulative["death"]        += obs.reward_death
        cumulative["item"]         += obs.reward_item_pickup
        cumulative["hook"]         += obs.reward_hook_traversal

        if obs.is_terminal:
            n_episodes += 1
            last_episode_reward = ep_reward_sum
            ep_reward_sum = 0.0

        # send zero action so bot doesn't stall
        if not args.no_action:
            act = struct.pack(ACT_FMT, ML_ACT_MAGIC, obs.tick,
                              0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
            s.sendto(act, addr)

        # render at 5Hz max
        if now - last_render < 0.2:
            continue
        last_render = now

        # rate over last 1s
        cutoff = now - 1.0
        while pkt_times and pkt_times[0] < cutoff:
            pkt_times.popleft()
        rate_1s = len(pkt_times)

        # rate over last 5s (approximation: count in window)
        n5 = sum(1 for t in pkt_times if t >= now - 5.0)

        pos = obs.self_state[0:3]
        vel = obs.self_state[3:6]
        hp  = obs.self_state[6]
        wpn = obs.self_state[8]

        # render
        sys.stdout.write("\033[3;1H")  # cursor to row 3
        sys.stdout.write(f"\033[Kbot slot      : {obs.bot_slot}   tick: {obs.tick:>6}   addr: {addr[0]}:{addr[1]}\n")
        sys.stdout.write(f"\033[K\n")
        sys.stdout.write(f"\033[K── data rate ──────────────────────────────────────\n")
        sys.stdout.write(f"\033[K  packets received    : {n_pkts:>10}\n")
        sys.stdout.write(f"\033[K  rate (last 1s)      : {rate_1s:>5} obs/s\n")
        sys.stdout.write(f"\033[K  rate (last 5s avg)  : {n5/5:>5.0f} obs/s\n")
        sys.stdout.write(f"\033[K\n")
        sys.stdout.write(f"\033[K── bot state ──────────────────────────────────────\n")
        sys.stdout.write(f"\033[K  pos    : ({pos[0]:>+8.1f}, {pos[1]:>+8.1f}, {pos[2]:>+8.1f})\n")
        sys.stdout.write(f"\033[K  vel    : ({vel[0]:>+8.1f}, {vel[1]:>+8.1f}, {vel[2]:>+8.1f})    speed={((vel[0]**2+vel[1]**2)**0.5):>5.0f}\n")
        sys.stdout.write(f"\033[K  hp/wpn : {hp:>4.0f} / weapon_id={wpn:.0f}\n")
        sys.stdout.write(f"\033[K  facing : yaw={obs.yaw:>+6.1f}  pitch={obs.pitch:>+6.1f}\n")
        sys.stdout.write(f"\033[K  visible enemies: {obs.entity_count}\n")
        sys.stdout.write(f"\033[K\n")
        sys.stdout.write(f"\033[K── reward signal ──────────────────────────────────\n")
        sys.stdout.write(f"\033[K  step reward         : {step_r:>+7.3f}\n")
        sys.stdout.write(f"\033[K  this episode (so far): {ep_reward_sum:>+7.3f}\n")
        sys.stdout.write(f"\033[K  episodes completed   : {n_episodes}\n")
        sys.stdout.write(f"\033[K  last episode reward  : {last_episode_reward:>+7.3f}\n")
        sys.stdout.write(f"\033[K\n")
        sys.stdout.write(f"\033[K── reward components (cumulative) ─────────────────\n")
        sys.stdout.write(f"\033[K  damage dealt : {cumulative['damage_dealt']:>8.0f}    kills : {cumulative['kill']:>4.0f}\n")
        sys.stdout.write(f"\033[K  damage taken : {cumulative['damage_taken']:>8.0f}    deaths: {cumulative['death']:>4.0f}\n")
        sys.stdout.write(f"\033[K  item pickups : {cumulative['item']:>8.0f}    hooks : {cumulative['hook']:>4.0f}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nbye")
