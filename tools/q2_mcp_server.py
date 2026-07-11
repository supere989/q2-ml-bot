#!/usr/bin/env python3
"""Minimal MCP stdio server for Quake II ML audit/control workflows."""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
Q2_ROOT = Path(os.environ.get("Q2_ROOT", PROJECT_ROOT / "q2_lithium_merge"))
DEFAULT_REPORT = ROOT / "runs" / "human_sparring_projectile_physics.jsonl"
DEFAULT_CLIENT_LOG = Path("/tmp/q2client-restart.log")


def _json_text(data: Any) -> list[dict[str, str]]:
    return [{"type": "text", "text": json.dumps(data, indent=2, sort_keys=True)}]


def _tail(path: Path, lines: int = 40) -> list[str]:
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return data[-max(1, min(int(lines), 500)) :]


def _latest_jsonl(path: Path) -> dict[str, Any] | None:
    for line in reversed(_tail(path, 200)):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _run(args: list[str], cwd: Path = PROJECT_ROOT, timeout: float = 20.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def q2_status(args: dict[str, Any]) -> list[dict[str, str]]:
    report = Path(args.get("report_file") or DEFAULT_REPORT)
    client_log = Path(args.get("client_log") or DEFAULT_CLIENT_LOG)
    procs = _run(["pgrep", "-af", "q2ded|quake2|evaluate_1v1"], timeout=5.0)
    errors = [
        line for line in _tail(client_log, 80)
        if "GL3_SetSky" in line or "skip" in line or "ERROR" in line.upper()
    ]
    return _json_text({
        "q2_root": str(Q2_ROOT),
        "report_file": str(report),
        "latest_report": _latest_jsonl(report),
        "processes": procs["stdout"].splitlines(),
        "client_log": str(client_log),
        "client_log_alerts": errors[-20:],
    })


def q2_tail_report(args: dict[str, Any]) -> list[dict[str, str]]:
    report = Path(args.get("report_file") or DEFAULT_REPORT)
    return _json_text({"report_file": str(report), "lines": _tail(report, args.get("lines", 40))})


def q2_tail_client_log(args: dict[str, Any]) -> list[dict[str, str]]:
    log = Path(args.get("client_log") or DEFAULT_CLIENT_LOG)
    return _json_text({"client_log": str(log), "lines": _tail(log, args.get("lines", 80))})


def q2_capture_screen(args: dict[str, Any]) -> list[dict[str, str]]:
    out = Path(args.get("path") or (
        "/tmp/q2-audit-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
    ))
    result = _run(["spectacle", "-b", "-n", "-o", str(out)], timeout=15.0)
    return _json_text({
        "path": str(out),
        "exists": out.exists(),
        **result,
    })


def q2_validate_maps(args: dict[str, Any]) -> list[dict[str, str]]:
    generated_dir = str(args.get("generated_dir") or ROOT / "maps" / "generated")
    glob = str(args.get("glob") or "mltrain_*.map")
    result = _run([
        sys.executable,
        str(ROOT / "tools" / "validate_maps.py"),
        "--generated-dir",
        generated_dir,
        "--glob",
        glob,
    ], cwd=PROJECT_ROOT, timeout=30.0)
    return _json_text(result)


def q2_start_client(args: dict[str, Any]) -> list[dict[str, str]]:
    host = str(args.get("host") or "127.0.0.1")
    port = int(args.get("port") or 28846)
    log_path = Path(args.get("client_log") or DEFAULT_CLIENT_LOG)
    cmd = [
        str(Q2_ROOT / "quake2"),
        "-datadir",
        str(Q2_ROOT),
        "+set",
        "game",
        "lithium",
        "+set",
        "vid_fullscreen",
        "0",
        "+set",
        "s_initsound",
        "1",
        "+connect",
        f"{host}:{port}",
    ]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return _json_text({"pid": proc.pid, "client_log": str(log_path), "command": cmd})


TOOLS: dict[str, tuple[str, dict[str, Any], Callable[[dict[str, Any]], list[dict[str, str]]]]] = {
    "q2_status": (
        "Return live Quake II process state, latest ML telemetry, and client log alerts.",
        {
            "type": "object",
            "properties": {
                "report_file": {"type": "string"},
                "client_log": {"type": "string"},
            },
        },
        q2_status,
    ),
    "q2_tail_report": (
        "Return the last N JSONL telemetry rows from the live evaluation report.",
        {
            "type": "object",
            "properties": {
                "report_file": {"type": "string"},
                "lines": {"type": "integer", "minimum": 1, "maximum": 500},
            },
        },
        q2_tail_report,
    ),
    "q2_tail_client_log": (
        "Return the last N lines from the Quake II client log.",
        {
            "type": "object",
            "properties": {
                "client_log": {"type": "string"},
                "lines": {"type": "integer", "minimum": 1, "maximum": 500},
            },
        },
        q2_tail_client_log,
    ),
    "q2_capture_screen": (
        "Capture the current desktop through KDE Spectacle for visual audit.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
        },
        q2_capture_screen,
    ),
    "q2_validate_maps": (
        "Run generated-map validation for spawn safety, sky config, pickups, and hook zones.",
        {
            "type": "object",
            "properties": {
                "generated_dir": {"type": "string"},
                "glob": {"type": "string"},
            },
        },
        q2_validate_maps,
    ),
    "q2_start_client": (
        "Start a Yamagi Quake II client connected to a running local test server.",
        {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "client_log": {"type": "string"},
            },
        },
        q2_start_client,
    ),
}


def _tool_list() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": desc, "inputSchema": schema}
        for name, (desc, schema, _handler) in TOOLS.items()
    ]


def _reply(msg_id: Any, result: Any = None, error: Any = None) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    print(json.dumps(payload), flush=True)


def _handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        _reply(msg_id, {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "q2-ml-audit", "version": "0.1.0"},
        })
    elif method == "tools/list":
        _reply(msg_id, {"tools": _tool_list()})
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in TOOLS:
            _reply(msg_id, error={"code": -32602, "message": f"unknown tool: {name}"})
            return
        try:
            content = TOOLS[name][2](arguments)
            _reply(msg_id, {"content": content})
        except Exception as exc:
            _reply(msg_id, error={"code": -32000, "message": str(exc)})
    elif method and method.startswith("notifications/"):
        return
    else:
        _reply(msg_id, error={"code": -32601, "message": f"unknown method: {method}"})


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            _handle(json.loads(line))
        except Exception as exc:
            _reply(None, error={"code": -32700, "message": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
