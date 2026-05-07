"""
spectate.py — connect a Quake 2 client to a training server as a spectator.

Discovers running training envs (q2ded processes on UDP 27910..27916), shows
their state, and launches the local quake2 client with +connect + chasecam
prep so you immediately spectate the ML bot.

Usage:
    python tools/spectate.py             # list envs, prompt to pick one
    python tools/spectate.py --port 27910 --bot-slot 7
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.protocol import ML_BASE_PORT, OBS_SIZE, parse_obs

Q2_ROOT     = Path(os.environ.get("Q2_ROOT", "/home/raymond/q2_lithium_merge"))
QUAKE2      = Q2_ROOT / "quake2"
SCAN_PORTS  = list(range(27910, 27917))
SCAN_BOTS   = list(range(0, 8))


# ── env discovery ─────────────────────────────────────────────────

def find_q2ded_servers():
    """Return list of (pid, port) pairs for running q2ded processes."""
    out = subprocess.check_output(
        ["ss", "-anpu"], stderr=subprocess.DEVNULL
    ).decode()
    servers = []
    for line in out.splitlines():
        if "127.0.0.1:" not in line or "q2ded" not in line:
            continue
        for port in SCAN_PORTS:
            if f"127.0.0.1:{port}" in line:
                # extract pid from users:(("q2ded",pid=NNN,fd=...))
                if "pid=" in line:
                    pid = int(line.split("pid=")[1].split(",")[0])
                    servers.append((pid, port))
                break
    return servers


def probe_ml_bot(slot: int, timeout: float = 0.25):
    """Briefly bind to bot's UDP port and try to read one obs to detect activity.
    Returns Observation or None.  Will fail if the harness already owns the port."""
    port = ML_BASE_PORT + slot
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        # port in use → harness is listening, that's fine
        return "in_use"
    s.settimeout(timeout)
    try:
        data, _ = s.recvfrom(4096)
        return parse_obs(data)
    except socket.timeout:
        return None
    finally:
        s.close()


def list_envs():
    servers = find_q2ded_servers()
    if not servers:
        print("No q2ded training servers found on UDP 27910-27916.")
        return []

    print(f"{'#':>2}  {'PID':>6}  {'Port':>5}  {'ML slots':<20}")
    print("─" * 50)
    for i, (pid, port) in enumerate(servers):
        ml_slots = []
        for slot in SCAN_BOTS:
            r = probe_ml_bot(slot, timeout=0.25)
            if r == "in_use":
                ml_slots.append(f"{slot}*")     # harness already bound
            elif r is not None:
                ml_slots.append(f"{slot}!")     # we received a real obs
        ml_str = ",".join(ml_slots) if ml_slots else "—"
        print(f"{i:>2}  {pid:>6}  {port:>5}  {ml_str:<20}")
    print("(* = harness already listening; bot is sending obs there)")
    return servers


# ── client launcher ───────────────────────────────────────────────

def launch_spectator(port: int, bot_slot: int = None, fov: int = 110):
    """Start the local quake2 client connected as a spectator."""
    if not QUAKE2.exists():
        print(f"error: {QUAKE2} not found", file=sys.stderr)
        sys.exit(1)

    cmd = [
        str(QUAKE2),
        "+set", "name",      "spectator",
        "+set", "spectator", "1",          # join as spectator (Lithium userinfo)
        "+set", "fov",       str(fov),
        "+connect", f"127.0.0.1:{port}",
    ]

    print("Launching:", " ".join(cmd))
    print()
    print("─" * 60)
    print("Once connected (open console with ~):")
    print()
    print("  noclip          → free fly camera (best for watching bots)")
    print("  chasecam        → chase mode (Lithium counts only real")
    print("                    players; may fail with bot-only games)")
    print("  give all        → cheat-give weapons/items")
    print("  fov 110         → wider view")
    print("  cl_drawhud 0    → hide HUD")
    print()
    print("NOTE: noclip needs the server started with +set cheats 1")
    print("─" * 60)
    print()

    try:
        subprocess.run(cmd, cwd=str(Q2_ROOT))
    except KeyboardInterrupt:
        pass


# ── main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port",     type=int, default=None,
                   help="server port (default: prompt or first found)")
    p.add_argument("--bot-slot", type=int, default=None,
                   help="bot slot to chase after connect (default: none)")
    p.add_argument("--fov",      type=int, default=110)
    p.add_argument("--list",     action="store_true",
                   help="list servers and exit")
    args = p.parse_args()

    servers = list_envs()
    if args.list or not servers:
        return

    if args.port is None:
        if len(servers) == 1:
            args.port = servers[0][1]
        else:
            sel = input(f"Pick env # [0-{len(servers)-1}]: ").strip()
            args.port = servers[int(sel)][1]

    print()
    launch_spectator(args.port, args.bot_slot, args.fov)


if __name__ == "__main__":
    main()
