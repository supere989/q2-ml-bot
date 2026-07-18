#!/usr/bin/env python3
"""Run the signed public pre/one-run/public post B6 sequence.

This controller is the sole creator of the unpredictable run nonce.  It binds
that nonce to both remote Ed25519 probes and to the one-run ``launch_id``.  Its
local monotonic intervals establish ordering without comparing clocks across
the public host and WSL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.capture_b6_public_state import (  # noqa: E402
    CAMPAIGN_ID,
    canonical_bytes,
    canonical_sha256,
    load_authority,
    verify_probe,
)


SCHEMA = "q2-multires-b6-attested-campaign-ledger-v1"
PLAN_SCHEMA = "q2-multires-b6-attested-campaign-plan-v1"
TOOL = "run_b6_attested_campaign"
_NONCE = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")
_SAFE_TARGET = re.compile(r"[A-Za-z0-9_.@-]{1,128}\Z")


class CampaignControllerError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignControllerError(message)


def _record(path: Path, *, newline: bool | None = None) -> dict[str, Any]:
    source = Path(os.path.abspath(path.expanduser()))
    _require(source.is_file() and not source.is_symlink(), f"missing raw record: {source}")
    data = source.read_bytes()
    _require(bool(data), f"empty raw record: {source}")
    if newline is True:
        _require(data.endswith(b"\n"), f"raw record lacks final newline: {source}")
    if newline is False:
        _require(not data.endswith(b"\n"), f"raw record has unexpected final newline: {source}")
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _load_canonical(path: Path, label: str, *, newline: bool) -> dict[str, Any]:
    raw = path.read_bytes()
    encoded = raw[:-1] if newline and raw.endswith(b"\n") else raw
    _require((not newline or raw == encoded + b"\n") and b"\n" not in encoded,
             f"{label} must be one canonical JSON record")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignControllerError(f"{label} is not JSON") from error
    _require(isinstance(value, dict) and canonical_bytes(value) == encoded,
             f"{label} is not canonical JSON")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    plan = _load_canonical(path, "B6 controller plan", newline=True)
    expected = {
        "schema", "campaign_id", "ssh_argv", "remote_python",
        "remote_capture_tool", "remote_authority", "remote_signing_key",
        "remote_output_root", "one_run_argv", "plan_sha256",
    }
    _require(set(plan) == expected and plan["schema"] == PLAN_SCHEMA
             and plan["campaign_id"] == CAMPAIGN_ID,
             "B6 controller plan identity/fields differ")
    supplied = plan["plan_sha256"]
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    _require(isinstance(supplied, str) and supplied == canonical_sha256(unsigned),
             "B6 controller plan seal differs")
    ssh_argv = plan["ssh_argv"]
    _require(isinstance(ssh_argv, list) and len(ssh_argv) >= 2
             and ssh_argv[0] == "ssh"
             and all(isinstance(item, str) and item for item in ssh_argv)
             and _SAFE_TARGET.fullmatch(ssh_argv[-1]) is not None,
             "B6 controller SSH argv differs")
    for name in (
        "remote_python", "remote_capture_tool", "remote_authority",
        "remote_signing_key", "remote_output_root",
    ):
        value = plan[name]
        _require(isinstance(value, str) and value.startswith("/")
                 and os.path.abspath(value) == value,
                 f"B6 controller {name} must be absolute and normalized")
    one_run = plan["one_run_argv"]
    _require(isinstance(one_run, list) and len(one_run) >= 4
             and all(isinstance(item, str) and item for item in one_run),
             "B6 one-run argv differs")
    _require(one_run[1:3] == ["-m", "train.multires_one_run"],
             "B6 controller only admits train.multires_one_run")
    _require("--launch_id" not in one_run and "--out" not in one_run,
             "controller owns one-run launch_id and output")
    _require(
        "--campaign_mode" in one_run
        and one_run[one_run.index("--campaign_mode") + 1] == CAMPAIGN_ID,
        "one-run argv does not select the exact B6 campaign mode",
    )
    return plan


def _new_root(path: Path) -> Path:
    value = Path(os.path.abspath(path.expanduser()))
    _require(value.is_absolute() and not value.exists() and not value.is_symlink(),
             "B6 controller output root must be a new absolute path")
    _require(value.parent.is_dir() and not value.parent.is_symlink(),
             "B6 controller output parent differs")
    value.mkdir(mode=0o700)
    return value


def _run_command(argv: Sequence[str], stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    _require(not stdout_path.exists() and not stderr_path.exists(),
             "subprocess raw output path already exists")
    started = time.monotonic_ns()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(list(argv), stdout=stdout, stderr=stderr, check=False)
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    completed = time.monotonic_ns()
    return {
        "argv": list(argv),
        "argv_sha256": hashlib.sha256(canonical_bytes(list(argv))).hexdigest(),
        "started_monotonic_ns": started,
        "completed_monotonic_ns": completed,
        "returncode": result.returncode,
        "stdout": _record_allow_empty(stdout_path),
        "stderr": _record_allow_empty(stderr_path),
    }


def _record_allow_empty(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _remote_probe_argv(
    plan: Mapping[str, Any], *, run_nonce: str, phase: str,
    predecessor_evidence_sha256: str,
) -> list[str]:
    remote_output = (
        f"{plan['remote_output_root']}/{run_nonce}-{phase}-public-probe.json"
    )
    return [
        *plan["ssh_argv"], "--", plan["remote_python"],
        plan["remote_capture_tool"], "--authority", plan["remote_authority"],
        "--campaign-id", CAMPAIGN_ID, "--run-nonce", run_nonce,
        "--phase", phase, "--predecessor-evidence-sha256",
        predecessor_evidence_sha256,
        "--signing-key", plan["remote_signing_key"],
        "--output", remote_output,
    ]


def _extract_probe_stdout(stdout_path: Path, destination: Path) -> dict[str, Any]:
    raw = stdout_path.read_bytes()
    _require(raw.endswith(b"\n") and raw.count(b"\n") == 1,
             "remote public probe stdout is not one record")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignControllerError("remote public probe stdout is not JSON") from error
    _require(isinstance(value, dict) and raw == canonical_bytes(value) + b"\n",
             "remote public probe stdout is not canonical")
    _publish(destination, value)
    return value


def launch_id_for(run_nonce: str, pre_evidence_sha256: str) -> str:
    _require(_NONCE.fullmatch(run_nonce) is not None,
             "B6 launch nonce is malformed")
    _require(re.fullmatch(r"(?!0{64})[0-9a-f]{64}", pre_evidence_sha256) is not None,
             "B6 pre-probe evidence identity is malformed")
    digest = hashlib.sha256(
        b"q2-b6-attested-launch-v1\x00"
        + bytes.fromhex(run_nonce) + bytes.fromhex(pre_evidence_sha256)
    ).hexdigest()
    return "b6-" + digest[:60]


def _one_run_value(path: Path, launch_id: str) -> dict[str, Any]:
    value = _load_canonical(path, "B6 one-run output", newline=False)
    supplied = value.get("evidence_sha256")
    unsigned = dict(value)
    unsigned.pop("evidence_sha256", None)
    received = value.get("received_inputs")
    _require(
        isinstance(supplied, str) and supplied == canonical_sha256(unsigned)
        and value.get("campaign_mode") == CAMPAIGN_ID
        and isinstance(received, Mapping)
        and received.get("launch_id") == launch_id
        and received.get("campaign_mode") == CAMPAIGN_ID,
        "one-run output is not sealed to the controller nonce/campaign",
    )
    return value


def run_campaign(plan_path: Path, output_root: Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    root = _new_root(output_root)
    run_nonce = secrets.token_hex(32)
    _require(_NONCE.fullmatch(run_nonce) is not None, "controller nonce generation failed")
    # The same canonical authority must exist in the staged local repository.
    local_authority = ROOT / "docs/multires/B6-PUBLIC-HOST-AUTHORITY.json"
    _require(local_authority.is_file(), "local B6 public authority is absent")
    authority = load_authority(local_authority.resolve())

    stages: dict[str, Any] = {}
    pre_stdout, pre_stderr = root / "pre.stdout", root / "pre.stderr"
    stages["public_pre"] = _run_command(
        _remote_probe_argv(
            plan, run_nonce=run_nonce, phase="pre",
            predecessor_evidence_sha256=authority["authority_sha256"],
        ),
        pre_stdout, pre_stderr,
    )
    _require(stages["public_pre"]["returncode"] == 0,
             "signed public pre-probe failed")
    pre_path = root / "public-pre.json"
    pre = _extract_probe_stdout(pre_stdout, pre_path)
    pre_payload = verify_probe(
        pre, authority_path=local_authority, expected_campaign_id=CAMPAIGN_ID,
        expected_run_nonce=run_nonce, expected_phase="pre",
        expected_predecessor_evidence_sha256=authority["authority_sha256"],
    )
    stages["public_pre"]["output"] = _record(pre_path, newline=True)
    stages["public_pre"]["evidence_sha256"] = pre["evidence_sha256"]

    launch_id = launch_id_for(run_nonce, pre["evidence_sha256"])
    one_run_path = root / "one-run.json"
    one_stdout, one_stderr = root / "one-run.stdout", root / "one-run.stderr"
    one_argv = [
        *plan["one_run_argv"], "--launch_id", launch_id, "--out", str(one_run_path)
    ]
    stages["one_run"] = _run_command(one_argv, one_stdout, one_stderr)
    _require(stages["one_run"]["returncode"] == 0, "B6 one-run failed")
    one_run = _one_run_value(one_run_path, launch_id)
    stages["one_run"]["output"] = _record(one_run_path, newline=False)
    stages["one_run"]["evidence_sha256"] = one_run["evidence_sha256"]

    post_stdout, post_stderr = root / "post.stdout", root / "post.stderr"
    stages["public_post"] = _run_command(
        _remote_probe_argv(
            plan, run_nonce=run_nonce, phase="post",
            predecessor_evidence_sha256=one_run["evidence_sha256"],
        ),
        post_stdout, post_stderr,
    )
    _require(stages["public_post"]["returncode"] == 0,
             "signed public post-probe failed")
    post_path = root / "public-post.json"
    post = _extract_probe_stdout(post_stdout, post_path)
    post_payload = verify_probe(
        post, authority_path=local_authority, expected_campaign_id=CAMPAIGN_ID,
        expected_run_nonce=run_nonce, expected_phase="post",
        expected_predecessor_evidence_sha256=one_run["evidence_sha256"],
    )
    stages["public_post"]["output"] = _record(post_path, newline=True)
    stages["public_post"]["evidence_sha256"] = post["evidence_sha256"]

    _require(
        stages["public_pre"]["completed_monotonic_ns"]
        <= stages["one_run"]["started_monotonic_ns"]
        <= stages["one_run"]["completed_monotonic_ns"]
        <= stages["public_post"]["started_monotonic_ns"],
        "controller monotonic stage order differs",
    )
    _require(
        pre_payload["run_binding"]["capture_nonce"]
        != post_payload["run_binding"]["capture_nonce"],
        "public pre/post reused a capture nonce",
    )
    _require(pre_payload["state"] == post_payload["state"],
             "public service/runtime/map/credential state changed")
    ledger = {
        "schema": SCHEMA,
        "tool": TOOL,
        "status": "green",
        "campaign_id": CAMPAIGN_ID,
        "run_nonce": run_nonce,
        "launch_id": launch_id,
        "controller_host": {
            "hostname": platform.node(),
            "kernel_release": platform.release(),
            "architecture": platform.machine(),
        },
        "controller_tool": _record(Path(__file__).resolve()),
        "plan": _record(plan_path, newline=True),
        "authority": _record(local_authority, newline=True),
        "stages": stages,
        "ordering_basis": "single-controller-monotonic-ns-v1",
        "ledger_sha256": "",
    }
    unsigned = dict(ledger)
    unsigned.pop("ledger_sha256")
    ledger["ledger_sha256"] = canonical_sha256(unsigned)
    _publish(root / "campaign-ledger.json", ledger)
    return ledger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_campaign(args.plan, args.output_root)
    except Exception as error:
        print(f"B6 attested campaign refused: {error}", file=sys.stderr, flush=True)
        return 2
    print(canonical_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
