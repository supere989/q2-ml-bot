#!/usr/bin/env python3
"""Run the public q2ded lane for normal-player ML training clients.

This launcher deliberately has no in-process policy or legacy bot bridge.  ML
actors connect through protocol 34 as ordinary players, while game.so sends
each actor its private authoritative observation on the separately protected
telemetry conduit.

The conduit secret is accepted only through the service environment.  It is
never a command-line argument or a log value.  q2ded must read it from a cfg,
so that cfg is written atomically with mode 0600 before the initial map command.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.map_farm_client import (  # noqa: E402
    FarmMapGenerator,
    ShuffledStockRotation,
    query_live_mapname,
)

LIVE_MAP_PREFIX = "mllive"
STOP = False
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]{32,63}\Z")
_MAP_PATTERN = re.compile(r"[A-Za-z0-9_]{1,63}\Z")
_SAFE_CVAR_PATTERN = re.compile(r"[^\s\"';\\]{1,255}\Z")


@dataclass(frozen=True)
class TelemetrySettings:
    port: int
    token: str = field(repr=False)


def _request_stop(_signum, _frame) -> None:
    global STOP
    STOP = True


def _telemetry_from_env(environ: dict[str, str] | None = None) -> TelemetrySettings:
    """Load the conduit contract without accepting a CLI/loggable secret."""
    source = os.environ if environ is None else environ
    if source.get("Q2_ML_CLIENT_TELEMETRY") != "1":
        raise ValueError("Q2_ML_CLIENT_TELEMETRY must be set to 1")
    try:
        port = int(source["Q2_ML_CLIENT_TELEMETRY_PORT"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Q2_ML_CLIENT_TELEMETRY_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("Q2_ML_CLIENT_TELEMETRY_PORT must be between 1 and 65535")
    token = source.get("Q2_ML_CLIENT_TELEMETRY_TOKEN", "")
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError(
            "Q2_ML_CLIENT_TELEMETRY_TOKEN must be 32-63 safe base64/URL characters"
        )
    return TelemetrySettings(port=port, token=token)


def _safe_map_name(value: str) -> str:
    if not _MAP_PATTERN.fullmatch(value):
        raise ValueError("map name contains unsafe cfg characters")
    return value


def _safe_cvar_value(label: str, value: str) -> str:
    if not _SAFE_CVAR_PATTERN.fullmatch(value):
        raise ValueError(f"{label} contains unsafe cfg characters")
    return value


def _write_config(
    q2_root: Path,
    args: argparse.Namespace,
    first_map: str,
    telemetry: TelemetrySettings,
) -> Path:
    """Write the only secret-bearing artifact atomically and mode 0600."""
    first_map = _safe_map_name(first_map)
    if not 1 <= args.maxclients <= 64:
        raise ValueError("maxclients must be between 1 and 64")
    if args.dlserver:
        _safe_cvar_value("download server", args.dlserver)

    path = q2_root / "lithium" / f"ml_network_public_{args.port}.cfg"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "set dedicated 1",
        "set deathmatch 1",
        "set cheats 1",
        f"set timelimit {args.timelimit:g}",
        f"set fraglimit {args.fraglimit}",
        "set use_mapqueue 0",
        'set mapqueue ""',
        "set map_random 0",
        # Real clients own all player slots.  No 3ZB2 or in-process ML bot may
        # be auto-created by the public network-native lane.
        "set autospawn 0",
        'set botlist ""',
        f"set maxclients {args.maxclients}",
        "set ml_enabled 0",
        "set ml_bot_slot 99",
        "set ml_teacher_enabled 0",
        "set ml_client_telemetry 1",
        f"set ml_client_telemetry_port {telemetry.port}",
        f'set ml_client_telemetry_token "{telemetry.token}"',
        "set timedemo 0",
        "set timescale 1",
        "set use_runes 1",
        "set use_startobserver 0",
        "set use_startchasecam 0",
        "set use_hook 1",
        "set hook_speed 1900",
        "set hook_pullspeed 1700",
        "set hook_pullspeed_max 2000",
        "set hook_pullscale 0.25",
        "set hook_gravity_comp 1.0",
        "set hook_min_lift 180",
        "set hook_maxtime 15.0",
        "set hook_damage 1",
        "set hook_initdamage 10",
        "set hook_maxdamage 20",
        "set hook_delay 0.2",
        "set rocket_speed_start 650",
        "set rocket_speed_max 2000",
        "set rocket_accel_time 0.75",
        "set rocket_accel_curve 12",
        "set rocket_haste_refire 0.36",
        "set energy_light_speed 1",
    ]
    if args.dlserver:
        lines.append(f'set sv_downloadserver "{args.dlserver}"')
    # Telemetry must be configured before this command loads the map and starts
    # accepting client frames.
    lines.extend([f"map {first_map}", ""])

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write("\n".join(lines))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _build_command(
    q2ded: Path,
    q2_root: Path,
    args: argparse.Namespace,
    config: Path,
) -> list[str]:
    del q2_root  # retained in the signature to make the launch boundary clear
    return [
        "stdbuf",
        "-oL",
        "-eL",
        str(q2ded),
        "+set",
        "game",
        "lithium",
        "+set",
        "ip",
        os.environ.get("Q2_BIND_IP", ""),
        "+set",
        "port",
        str(args.port),
        "+exec",
        config.name,
    ]


def _server_command(proc: subprocess.Popen, command: str) -> None:
    if proc.stdin is None or proc.poll() is not None:
        raise RuntimeError("public q2ded console is unavailable")
    proc.stdin.write((command.rstrip() + "\n").encode())
    proc.stdin.flush()


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_farm_url", default="http://100.86.206.50:32510")
    parser.add_argument("--stock_maps", default="q2dm1,q2dm3,q2dm5,q2dm7")
    parser.add_argument("--rotation_seed", type=int, default=1103)
    parser.add_argument("--port", type=int, default=28000)
    parser.add_argument("--maxclients", type=int, default=6)
    parser.add_argument("--timelimit", type=float, default=15.0)
    parser.add_argument("--fraglimit", type=int, default=20)
    parser.add_argument("--dlserver", default="http://5.78.204.86:32494")
    return parser


def main() -> int:
    global STOP
    STOP = False
    parser = _parser()
    args = parser.parse_args()
    try:
        telemetry = _telemetry_from_env()
    except ValueError as error:
        parser.error(str(error))
    if args.port == telemetry.port:
        parser.error("game and telemetry ports must be different")

    stock_names = args.stock_maps.replace(",", " ").split()
    try:
        for name in stock_names:
            _safe_map_name(name)
        stock = ShuffledStockRotation(stock_names, args.rotation_seed)
    except ValueError as error:
        parser.error(str(error))

    q2_root = Path(os.environ.get("Q2_ROOT", str(Path.home() / "q2_lithium_merge")))
    q2ded = q2_root / "q2ded"
    if not q2ded.is_file():
        parser.error(f"q2ded not found at {q2ded}")

    mapgen = FarmMapGenerator(args.map_farm_url, prefix=LIVE_MAP_PREFIX)
    first_map = stock.next()
    try:
        config = _write_config(q2_root, args, first_map, telemetry)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    command = _build_command(q2ded, q2_root, args, config)
    proc = subprocess.Popen(
        command,
        cwd=q2_root,
        stdin=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    current = first_map
    armed = None
    staged_generated = None
    last_check = 0.0
    mapgen.start()
    print(
        f"[network-public] pid={proc.pid} game_port={args.port} "
        f"telemetry_port={telemetry.port} first={first_map} "
        f"maxclients={args.maxclients} generated={LIVE_MAP_PREFIX}_*",
        flush=True,
    )
    try:
        while not STOP and proc.poll() is None:
            if mapgen.busy:
                finished = mapgen.poll()
                if finished:
                    staged_generated = finished
            elif staged_generated is None:
                mapgen.start()

            if armed is None:
                if current.startswith(f"{LIVE_MAP_PREFIX}_"):
                    armed = stock.next()
                elif staged_generated is not None:
                    armed = staged_generated
                    staged_generated = None
                if armed:
                    _server_command(proc, f'set sv_maplist "{current} {armed}"')
                    print(f"[network-public] armed {current} -> {armed}", flush=True)

            now = time.monotonic()
            if now - last_check >= 2.0:
                last_check = now
                live = query_live_mapname(args.port)
                if live and live != current:
                    current = live
                    armed = None
                    if staged_generated is None and not mapgen.busy:
                        mapgen.start()
                    print(f"[network-public] advanced to {current}", flush=True)
            time.sleep(0.1)
    finally:
        _stop_process(proc)

    if STOP:
        return 0
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
