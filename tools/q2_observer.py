#!/usr/bin/env python3
"""
q2_observer.py — Claude's eyes, ears, and legs inside the Q2 engine.

Starts a Q2 client in a headless Xvfb display, connects it as a spectator
to a running training server, and provides:

  screenshot()   → PNG bytes (or saves to file)
  send_cmd(cmd)  → rcon command to the server (noclip, setpos, look, etc.)
  key(k)         → send keystroke to the client window
  get_obs()      → latest structured observation from ML bridge
  look(yaw,pit)  → point camera to absolute angles
  move(fwd,rt,z) → relative move in world space
  goto(x,y,z)    → teleport to world coordinates

Usage (standalone):
    python3 tools/q2_observer.py start [--port 27910] [--map mlmap_00000042]
    python3 tools/q2_observer.py screenshot [--out /tmp/view.png]
    python3 tools/q2_observer.py cmd "noclip"
    python3 tools/q2_observer.py goto 512 512 128
    python3 tools/q2_observer.py look 90 -20
    python3 tools/q2_observer.py obs
    python3 tools/q2_observer.py stop

The observer runs in a persistent background process so Claude can call
any sub-command without restarting Q2.
"""

import argparse
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from harness.protocol import ML_BASE_PORT, parse_obs

# ── Paths ─────────────────────────────────────────────────────────────────────

Q2ROOT   = Path(os.environ.get("Q2ROOT", "/home/raymond/q2_lithium_merge"))
Q2CLIENT = Q2ROOT / "quake2"
STATE    = Path("/tmp/q2_observer.json")   # persists pid, display, port, rcon_pw

DISPLAY  = ":99"
GEOMETRY = "1280x720x24"
RCON_PW  = "mlobs_rcon_2026"

# ── Xvfb ─────────────────────────────────────────────────────────────────────

def start_xvfb():
    existing = subprocess.run(["pgrep", "-f", f"Xvfb {DISPLAY}"],
                              capture_output=True).returncode == 0
    if existing:
        return
    subprocess.Popen(
        ["Xvfb", DISPLAY, "-screen", "0", GEOMETRY, "-ac"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)


def stop_xvfb():
    subprocess.run(["pkill", "-f", f"Xvfb {DISPLAY}"],
                   capture_output=True)


# ── Rcon ─────────────────────────────────────────────────────────────────────

def rcon(cmd: str, host="127.0.0.1", port=27910, password=RCON_PW, timeout=1.0):
    """Send an rcon command to q2ded and return the response."""
    pkt = b"\xff\xff\xff\xff" + f"rcon {password} {cmd}".encode() + b"\x00"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    s.sendto(pkt, (host, port))
    try:
        data, _ = s.recvfrom(4096)
        # Response: \xff\xff\xff\xff print\n<message>
        return data[10:].decode("latin-1", errors="replace").strip()
    except socket.timeout:
        return ""
    finally:
        s.close()


# ── Screenshot ────────────────────────────────────────────────────────────────

def screenshot(out_path: str = None) -> bytes:
    """Capture the Q2 window as PNG. Returns bytes; optionally saves to file."""
    tmp = out_path or f"/tmp/q2obs_{int(time.time())}.png"
    env = {**os.environ, "DISPLAY": DISPLAY}
    wid = _get_window_id()
    target = wid if wid else "root"
    subprocess.run(
        ["import", "-display", DISPLAY, "-window", target, tmp],
        env=env, capture_output=True,
    )
    data = Path(tmp).read_bytes() if Path(tmp).exists() else b""
    if not out_path:
        Path(tmp).unlink(missing_ok=True)
    return data


# ── Q2 client ─────────────────────────────────────────────────────────────────

def _get_window_id() -> str:
    env = {**os.environ, "DISPLAY": DISPLAY}
    result = subprocess.run(
        ["xdotool", "search", "--name", "Quake"],
        capture_output=True, text=True, env=env,
    )
    ids = [x for x in result.stdout.strip().split("\n") if x]
    return ids[0] if ids else ""


def _key(k: str, delay: float = 0.05):
    wid = _get_window_id()
    if not wid:
        return
    env = {**os.environ, "DISPLAY": DISPLAY}
    subprocess.run(["xdotool", "key", "--window", wid, k],
                   capture_output=True, env=env)
    time.sleep(delay)


def _console(cmd: str):
    """Type a console command into the Q2 client."""
    wid = _get_window_id()
    if not wid:
        return
    env = {**os.environ, "DISPLAY": DISPLAY}
    subprocess.run(["xdotool", "key", "--window", wid, "grave"],
                   capture_output=True, env=env)
    time.sleep(0.15)
    subprocess.run(["xdotool", "type", "--window", wid, "--clearmodifiers", cmd],
                   capture_output=True, env=env)
    subprocess.run(["xdotool", "key", "--window", wid, "Return"],
                   capture_output=True, env=env)
    time.sleep(0.1)
    subprocess.run(["xdotool", "key", "--window", wid, "grave"],
                   capture_output=True, env=env)
    time.sleep(0.3)


def start_client(sv_port: int, map_name: str = None):
    """Launch Q2 client into a running server as a spectator."""
    start_xvfb()

    cmd = [
        str(Q2CLIENT),
        "+set", "game",          "lithium",
        "+set", "name",          "MLObserver",
        "+set", "spectator",     "1",
        "+set", "rcon_password", RCON_PW,
        "+set", "r_mode",        "0",
        "+set", "r_customwidth", "1280",
        "+set", "r_customheight","720",
        "+set", "vid_fullscreen","0",
        "+set", "cl_gun",        "0",
        "+set", "fov",           "110",
        "+connect",              f"127.0.0.1:{sv_port}",
    ]

    env = {**os.environ, "DISPLAY": DISPLAY, "SDL_VIDEODRIVER": "x11"}
    proc = subprocess.Popen(
        cmd, cwd=str(Q2ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )

    # Wait for window then enable noclip
    print("Waiting for Q2 window...", end="", flush=True)
    for _ in range(30):
        time.sleep(1)
        if _get_window_id():
            break
        print(".", end="", flush=True)
    print()

    time.sleep(3)    # let connection complete and first frame render
    _console("noclip")
    _console("cl_drawhud 0")   # cleaner screenshots

    STATE.write_text(json.dumps({
        "client_pid": proc.pid,
        "sv_port":    sv_port,
    }))
    print(f"Observer ready — pid={proc.pid}  port={sv_port}  display={DISPLAY}")
    print(f"  screenshot: python3 tools/q2_observer.py screenshot --out /tmp/view.png")
    print(f"  move:       python3 tools/q2_observer.py move w 20   (w/a/s/d/up/down)")
    print(f"  turn:       python3 tools/q2_observer.py turn left 90")


def stop_client():
    if STATE.exists():
        info = json.loads(STATE.read_text())
        pid = info.get("client_pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Stopped client pid={pid}")
            except ProcessLookupError:
                print("Client already stopped")
        STATE.unlink(missing_ok=True)
    stop_xvfb()
    print("Observer stopped")


def _load_state():
    if not STATE.exists():
        print("Observer not running — run: python3 tools/q2_observer.py start")
        sys.exit(1)
    return json.loads(STATE.read_text())


# ── Movement / camera ─────────────────────────────────────────────────────────

def move(direction: str, steps: int = 10):
    """
    Move camera using keyboard input.
    direction: w (forward) | s (back) | a (left) | d (right) | up | down
    steps: how many keystrokes (each ~64 units at noclip speed)
    """
    key_map = {
        "w": "w", "forward": "w",
        "s": "s", "back": "s",
        "a": "a", "left_strafe": "a",
        "d": "d", "right_strafe": "d",
        "up": "space", "rise": "space",
        "down": "c", "fall": "c",
    }
    k = key_map.get(direction.lower(), direction)
    for _ in range(steps):
        _key(k, delay=0.04)


def turn(direction: str, degrees: int = 45):
    """
    Turn camera by repeating arrow key presses.
    direction: left | right | up | down
    degrees: approx degrees to turn (1 press ≈ 5°)
    """
    key_map = {"left": "Left", "right": "Right", "up": "Up", "down": "Down"}
    k = key_map.get(direction.lower(), direction)
    presses = max(1, degrees // 5)
    for _ in range(presses):
        _key(k, delay=0.03)


def look(yaw: float, pitch: float, sv_port: int):
    """Set camera angles via console command."""
    _console(f"angles {pitch:.1f} {yaw:.1f} 0")


def goto(x: float, y: float, z: float, sv_port: int):
    """Teleport observer via console setpos (requires cheats)."""
    _console(f"setpos {x:.0f} {y:.0f} {z:.0f}")


# ── Obs reader ────────────────────────────────────────────────────────────────

def get_obs(bot_slot: int = 7):
    """Read one observation packet from the ML bridge for the given bot slot."""
    port = ML_BASE_PORT + bot_slot
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        return {"error": f"port {port} in use (training harness owns it)"}
    s.settimeout(2.0)
    try:
        data, src = s.recvfrom(4096)
        obs = parse_obs(data)
        if not obs:
            return {"error": "bad packet"}
        return {
            "tick":          obs.tick,
            "bot_slot":      obs.bot_slot,
            "pos":           list(obs.self_state[:3]),
            "health":        float(obs.self_state[3]),
            "weapon_id":     float(obs.self_state[4]),
            "yaw":           obs.yaw,
            "pitch":         obs.pitch,
            "entity_count":  obs.entity_count,
            "hook_zones":    obs.hook_zone_count,
            "reward":        obs.reward,
            "is_terminal":   obs.is_terminal,
            "audio_age":     float(obs.audio[3]),
            "alert_level":   float(obs.audio[4]),
        }
    except socket.timeout:
        return {"error": "timeout — bot not sending (training running on this slot?)"}
    finally:
        s.close()


# ── Key injection ─────────────────────────────────────────────────────────────

def send_key(key: str):
    """Send a keystroke to the Q2 client window via xdotool."""
    env = {**os.environ, "DISPLAY": DISPLAY}
    wid = subprocess.run(
        ["xdotool", "search", "--name", "Quake"],
        capture_output=True, text=True, env=env,
    ).stdout.strip().split("\n")[0]
    if wid:
        subprocess.run(["xdotool", "key", "--window", wid, key],
                       env=env, capture_output=True)
    else:
        print("Q2 window not found")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="Launch observer client")
    s.add_argument("--port", type=int, default=27910, help="Training server port")
    s.add_argument("--map",  default=None, help="Map name (if no server running)")

    s = sub.add_parser("stop", help="Kill observer client and Xvfb")

    s = sub.add_parser("screenshot", help="Capture current view")
    s.add_argument("--out", default="/tmp/q2_observer_view.png")

    s = sub.add_parser("cmd", help="Send rcon command to server")
    s.add_argument("command", nargs="+")

    s = sub.add_parser("goto", help="Teleport to world coords")
    s.add_argument("x", type=float); s.add_argument("y", type=float)
    s.add_argument("z", type=float)

    s = sub.add_parser("look", help="Set camera yaw/pitch")
    s.add_argument("yaw",   type=float)
    s.add_argument("pitch", type=float, nargs="?", default=0.0)

    s = sub.add_parser("move", help="Move camera with keyboard (w/s/a/d/up/down)")
    s.add_argument("direction")
    s.add_argument("steps", type=int, nargs="?", default=10)

    s = sub.add_parser("turn", help="Turn camera left/right/up/down")
    s.add_argument("direction")
    s.add_argument("degrees", type=int, nargs="?", default=45)

    s = sub.add_parser("key", help="Send keystroke to Q2 window")
    s.add_argument("key")

    s = sub.add_parser("obs", help="Read one ML bridge observation packet")
    s.add_argument("--slot", type=int, default=7)

    s = sub.add_parser("noclip", help="Toggle noclip on observer")

    s = sub.add_parser("info", help="Show observer state")

    args = p.parse_args()

    if args.cmd == "start":
        start_client(args.port, args.map)

    elif args.cmd == "stop":
        stop_client()

    elif args.cmd == "screenshot":
        data = screenshot(args.out)
        if data:
            print(f"Saved {len(data):,} bytes → {args.out}")
        else:
            print("Screenshot failed — is observer running?")

    elif args.cmd == "cmd":
        info = _load_state()
        resp = rcon(" ".join(args.command), port=info["sv_port"])
        print(resp or "(no response)")

    elif args.cmd == "goto":
        info = _load_state()
        goto(args.x, args.y, args.z, info["sv_port"])
        print(f"Teleporting to ({args.x:.0f}, {args.y:.0f}, {args.z:.0f})")

    elif args.cmd == "look":
        info = _load_state()
        look(args.yaw, args.pitch, info["sv_port"])
        print(f"Looking: yaw={args.yaw:.1f} pitch={args.pitch:.1f}")

    elif args.cmd == "move":
        _load_state()  # verify running
        move(args.direction, args.steps)
        print(f"Moved {args.direction} × {args.steps}")

    elif args.cmd == "turn":
        _load_state()
        turn(args.direction, args.degrees)
        print(f"Turned {args.direction} {args.degrees}°")

    elif args.cmd == "key":
        send_key(args.key)

    elif args.cmd == "obs":
        result = get_obs(args.slot)
        print(json.dumps(result, indent=2))

    elif args.cmd == "noclip":
        info = _load_state()
        rcon("noclip", port=info["sv_port"])
        print("noclip toggled")

    elif args.cmd == "info":
        if STATE.exists():
            info = json.loads(STATE.read_text())
            print(json.dumps(info, indent=2))
        else:
            print("Not running")


if __name__ == "__main__":
    main()
