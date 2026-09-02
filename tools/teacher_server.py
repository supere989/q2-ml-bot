#!/usr/bin/env python3
"""Run a dedicated legacy-3ZB2 teacher server with an interlaced rotation.

Rotation is wrapper-driven (process restart per map), not in-engine: the
lithium/3ZB2 build never exits intermission without a human pressing attack,
and the rare in-engine ``gamemap`` it does attempt segfaults (observed
2026-09-02). The wrapper watches server stdout for the round-end markers,
restarts q2ded on the next map, and persists the stock-rotation draw count
so systemd restarts resume the rotation instead of resetting to the first
map (which is why the teacher corpus was 100% q2dm2).
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.map_farm_client import (
    FarmMapGenerator,
    ShuffledStockRotation,
)

TEACHER_PREFIX = "mlteacher"
ROUND_END_MARKERS = ("Timelimit hit.", "Fraglimit hit.")
FATAL_MARKERS = ("Can't find maps/",)
STOCK_DRAWS_FILE = ".teacher_stock_draws"
MIN_ROUND_SECONDS = 15.0
CRASH_BACKOFF_SECONDS = 5.0
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


def _watch_stdout(stream, hit_event: threading.Event,
                  fatal_event: threading.Event) -> None:
    """Tee server stdout; flag round end and unrecoverable map-load failures."""
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            print(line, flush=True)
            if any(marker in line for marker in ROUND_END_MARKERS):
                hit_event.set()
            if any(marker in line for marker in FATAL_MARKERS):
                fatal_event.set()
    except (OSError, ValueError):
        pass


def _load_stock_draws(q2_root: Path) -> int:
    try:
        return max(0, int((q2_root / STOCK_DRAWS_FILE).read_text().strip()))
    except (OSError, ValueError):
        return 0


def _save_stock_draws(q2_root: Path, draws: int) -> None:
    target = q2_root / STOCK_DRAWS_FILE
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{draws}\n")
    os.replace(temporary, target)


def _select_next_map(current: str, staged_generated: str | None,
                     draw_stock) -> tuple[str, str | None]:
    """Interleave generated maps with stock; fall back stock->stock on outage."""
    if current.startswith(f"{TEACHER_PREFIX}_"):
        return draw_stock(), staged_generated
    if staged_generated is not None:
        return staged_generated, None
    return draw_stock(), staged_generated


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


def _launch_server(q2_root: Path, args, map_name: str):
    cfg = _write_config(q2_root, args, map_name)
    cmd = [
        "stdbuf", "-oL", "-eL", str(q2_root / "q2ded"),
        "+set", "game", "lithium",
        "+set", "ip", os.environ.get("Q2_BIND_IP", "127.0.0.1"),
        "+set", "port", str(args.port), "+exec", cfg.name,
    ]
    proc = subprocess.Popen(
        cmd, cwd=q2_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )
    round_hit = threading.Event()
    fatal = threading.Event()
    threading.Thread(
        target=_watch_stdout, args=(proc.stdout, round_hit, fatal),
        name="q2ded-stdout-watch", daemon=True,
    ).start()
    return proc, round_hit, fatal, time.monotonic()


def _terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)


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
    if not (q2_root / "q2ded").is_file():
        parser.error(f"q2ded not found at {q2_root / 'q2ded'}")
    _ensure_botlist(q2_root, args.botlist)

    stock_draws = _load_stock_draws(q2_root)
    for _ in range(stock_draws):
        stock.next()

    def draw_stock() -> str:
        nonlocal stock_draws
        result = stock.next()
        stock_draws += 1
        _save_stock_draws(q2_root, stock_draws)
        return result

    mapgen = FarmMapGenerator(args.map_farm_url, prefix=TEACHER_PREFIX)
    mapgen.start()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    current = draw_stock()
    staged_generated = None
    proc, round_hit, fatal, launched_at = _launch_server(q2_root, args, current)
    print(f"[teacher] pid={proc.pid} port={args.port} first={current} "
          f"stock={stock_names} generated={TEACHER_PREFIX}_* "
          f"stock_draws={stock_draws}", flush=True)
    try:
        while not STOP:
            code = proc.poll()
            if fatal.is_set() and code is None:
                print(f"[teacher] {current} failed to load; skipping",
                      flush=True)
                _terminate(proc)
                code = proc.returncode
            if round_hit.is_set() or code is not None:
                reason = "round end" if round_hit.is_set() else f"exit={code}"
                uptime = time.monotonic() - launched_at
                if code is None:
                    _terminate(proc)
                next_map, staged_generated = _select_next_map(
                    current, staged_generated, draw_stock)
                print(f"[teacher] rotating {current} -> {next_map} ({reason}, "
                      f"after {uptime:.0f}s)", flush=True)
                if uptime < MIN_ROUND_SECONDS:
                    time.sleep(CRASH_BACKOFF_SECONDS)
                current = next_map
                proc, round_hit, fatal, launched_at = _launch_server(
                    q2_root, args, current)
                if staged_generated is None and not mapgen.busy:
                    mapgen.start()
                continue

            if mapgen.busy:
                finished = mapgen.poll()
                if finished:
                    staged_generated = finished
            elif staged_generated is None:
                mapgen.start()
            time.sleep(0.2)
    finally:
        if proc.poll() is None:
            _terminate(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
