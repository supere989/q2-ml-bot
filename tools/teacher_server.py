#!/usr/bin/env python3
"""Run a dedicated legacy-3ZB2 teacher server with an interlaced rotation."""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.map_farm_client import (
    FarmMapGenerator,
    ShuffledStockRotation,
    query_live_mapname,
)

TEACHER_PREFIX = "mlteacher"
ML6_SECTION = r"""
[ml6sk1]
\\Evil Zeep	\male	\claymore	\1\2\3\1\3\0\3	\060\060\0\0	\0\0\0\1\0	\R\1
\\Claw Finger	\male	\claymore	\1\2\3\2\2\0\3	\060\060\0\0	\0\0\0\1\0	\R\1
\\Biohazard	\male	\claymore	\1\1\4\3\1\1\3	\060\060\0\0	\0\0\0\1\0	\R\1
\\Prong		\cyborg	\oni911		\1\2\3\1\3\0\3	\060\060\0\0	\0\0\0\1\0	\R\1
\\Sodom		\cyborg	\oni911		\1\2\3\2\2\0\3	\060\060\0\0	\0\0\0\1\0	\R\1
\\Korn		\cyborg	\oni911		\1\1\4\3\1\1\3	\060\060\0\0	\0\0\0\1\0	\R\1
"""
STOP = False


def _stop(_signum, _frame):
    global STOP
    STOP = True


def _server_command(proc: subprocess.Popen, command: str) -> None:
    if proc.stdin is None or proc.poll() is not None:
        raise RuntimeError("teacher q2ded console is unavailable")
    proc.stdin.write((command.rstrip() + "\n").encode())
    proc.stdin.flush()


def _ensure_botlist(q2_root: Path, name: str) -> None:
    if name != "ml6sk1":
        return
    config = q2_root / "3zb2" / "3ZBConfig.cfg"
    text = config.read_text(errors="replace")
    base = text.split("\n[ml6sk1]", 1)[0].rstrip()
    temporary = config.with_name(f".{config.name}.{os.getpid()}.tmp")
    temporary.write_text(base + "\n" + ML6_SECTION)
    shutil.copymode(config, temporary)
    os.replace(temporary, config)


def _write_config(q2_root: Path, args, first_map: str) -> Path:
    path = q2_root / "lithium" / f"ml_teacher_{args.port}.cfg"
    lines = [
        "set dedicated 1",
        "set deathmatch 1",
        "set cheats 1",
        f"set timelimit {args.timelimit:g}",
        f"set fraglimit {args.fraglimit}",
        "set use_mapqueue 0",
        "set mapqueue \"\"",
        "set map_random 0",
        "set autospawn 1",
        f"set botlist {args.botlist}",
        f"set maxclients {args.maxclients}",
        "set ml_enabled 1",
        "set ml_bot_slot 99",
        "set ml_teacher_enabled 1",
        f"set ml_teacher_addr {args.receiver_addr}",
        f"set ml_teacher_port {args.receiver_port}",
        f"set ml_teacher_stride {args.teacher_stride}",
        "set timedemo 0",
        "set timescale 1",
        "set use_hook 1",
        f"map {first_map}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_farm_url", default="http://100.86.206.50:32513")
    parser.add_argument("--stock_maps", default="q2dm2,q2dm4,q2dm6,q2dm8")
    parser.add_argument("--rotation_seed", type=int, default=2204)
    parser.add_argument("--port", type=int, default=28001)
    parser.add_argument("--receiver_addr", default="100.86.206.50")
    parser.add_argument("--receiver_port", type=int, default=32511)
    parser.add_argument("--teacher_stride", type=int, default=1)
    parser.add_argument("--botlist", default="ml6sk1")
    parser.add_argument("--maxclients", type=int, default=8)
    parser.add_argument("--timelimit", type=float, default=10.0)
    parser.add_argument("--fraglimit", type=int, default=30)
    args = parser.parse_args()

    stock_names = args.stock_maps.replace(",", " ").split()
    if "q2dm1" in stock_names:
        parser.error("q2dm1 is reserved for the public lane, not the teacher rotation")
    stock = ShuffledStockRotation(stock_names, args.rotation_seed)
    q2_root = Path(os.environ.get("Q2_ROOT", str(Path.home() / "q2_teacher_runtime")))
    q2ded = q2_root / "q2ded"
    if not q2ded.is_file():
        parser.error(f"q2ded not found at {q2ded}")
    _ensure_botlist(q2_root, args.botlist)

    mapgen = FarmMapGenerator(args.map_farm_url, prefix=TEACHER_PREFIX)
    first_map = stock.next()
    cfg = _write_config(q2_root, args, first_map)
    cmd = [
        "stdbuf", "-oL", "-eL", str(q2ded), "+set", "game", "lithium",
        "+set", "ip", os.environ.get("Q2_BIND_IP", "127.0.0.1"),
        "+set", "port", str(args.port), "+exec", cfg.name,
    ]
    proc = subprocess.Popen(
        cmd, cwd=q2_root, stdin=subprocess.PIPE, preexec_fn=os.setsid,
    )
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    current = first_map
    armed = None
    staged_generated = None
    last_check = 0.0
    mapgen.start()
    print(f"[teacher] pid={proc.pid} port={args.port} first={first_map} "
          f"stock={stock_names} generated={TEACHER_PREFIX}_*", flush=True)
    try:
        while not STOP and proc.poll() is None:
            if mapgen.busy:
                finished = mapgen.poll()
                if finished:
                    staged_generated = finished
            elif staged_generated is None:
                mapgen.start()

            if armed is None:
                if current.startswith(f"{TEACHER_PREFIX}_"):
                    armed = stock.next()
                elif staged_generated is not None:
                    armed = staged_generated
                    staged_generated = None
                if armed:
                    _server_command(proc, f'set sv_maplist "{current} {armed}"')
                    print(f"[teacher] armed {current} -> {armed}", flush=True)

            now = time.monotonic()
            if now - last_check >= 2.0:
                last_check = now
                live = query_live_mapname(args.port)
                if live and live != current:
                    current = live
                    armed = None
                    if staged_generated is None and not mapgen.busy:
                        mapgen.start()
                    print(f"[teacher] advanced to {current}", flush=True)
            time.sleep(0.1)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
