from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

import tools.assemble_b6_wsl_g1_campaign as b6
import tools.capture_b6_public_state as probe
from tools.provision_b6_public_probe_key import ProvisioningError, provision
import tools.run_b6_attested_campaign as controller
from tools.run_b6_attested_campaign import launch_id_for


RUN_NONCE = hashlib.sha256(b"run-nonce").hexdigest()
PREDECESSOR = hashlib.sha256(b"predecessor").hexdigest()


def _file(path: str, ordinal: int) -> dict:
    return {
        "path": path,
        "kind": "file",
        "device": 1,
        "inode": 100 + ordinal,
        "bytes": 10 + ordinal,
        "uid": 1000,
        "gid": 1000,
        "mode": 0o755,
        "mtime_ns": 1000 + ordinal,
        "ctime_ns": 2000 + ordinal,
        "sha256": hashlib.sha256(f"file-{ordinal}".encode()).hexdigest(),
    }


def _directory(path: str, ordinal: int) -> dict:
    row = _file(path, ordinal)
    row.pop("bytes")
    row.pop("sha256")
    row["kind"] = "directory"
    return row


def _authority(tmp_path: Path) -> tuple[Path, Path, dict]:
    key_path = tmp_path / "probe-key.pem"
    activation_path = tmp_path / "activation.json"
    activation = provision(key_path.resolve(), activation_path.resolve())
    tool_path = Path(probe.__file__).resolve()
    tool_bytes = tool_path.read_bytes()
    services = [
        {
            "name": "q2-teacher-server.service",
            "fragment_path": "/etc/systemd/system/q2-teacher-server.service",
            "drop_in_paths": [],
            "working_directory": "/srv/q2",
            "root_directory": "",
            "environment_files": "",
            "exec_start": "teacher exact argv",
            "executable_path": "/usr/bin/python3",
        },
        {
            "name": "q2mlbot.service",
            "fragment_path": "/etc/systemd/system/q2mlbot.service",
            "drop_in_paths": [],
            "working_directory": "/srv/q2",
            "root_directory": "",
            "environment_files": "/etc/q2.env (ignore_errors=no)",
            "exec_start": "public exact argv",
            "executable_path": "/usr/bin/python3",
        },
    ]
    authority = {
        "schema": probe.AUTHORITY_SCHEMA,
        "hostname": "public-fixture",
        "architecture": "x86_64",
        "machine_identity_sha256": hashlib.sha256(b"machine").hexdigest(),
        "public_ipv4": "192.0.2.1",
        "services": services,
        "required_ports": [28000, 28001, 28049],
        "queue_paths": [],
        "map_roots": ["/srv/public/maps", "/srv/teacher/maps"],
        "credential_paths": ["/etc/q2.env"],
        "runtime_artifact_paths": ["/srv/q2/game.so"],
        "capture_tool": {
            "path": probe.TOOL_SOURCE,
            "bytes": len(tool_bytes),
            "sha256": hashlib.sha256(tool_bytes).hexdigest(),
        },
        "probe_attestation": {
            "status": "active",
            "algorithm": "ed25519",
            "key_id": activation["key_id"],
            "public_key_base64": activation["public_key_base64"],
            "private_key_path": str(key_path.resolve()),
        },
    }
    authority["authority_sha256"] = probe.canonical_sha256(authority)
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(probe.canonical_bytes(authority) + b"\n")
    return authority_path.resolve(), key_path.resolve(), authority


def _state(authority: dict) -> dict:
    services = []
    for index, expected in enumerate(authority["services"]):
        services.append({
            "name": expected["name"],
            "load_state": "loaded",
            "active_state": "active",
            "sub_state": "running",
            "fragment_path": expected["fragment_path"],
            "drop_in_paths": [],
            "working_directory": expected["working_directory"],
            "root_directory": expected["root_directory"],
            "environment_files": expected["environment_files"],
            "exec_start": expected["exec_start"],
            "main_pid": 200 + index,
            "main_pid_start_ticks": 2000 + index,
            "active_enter_monotonic_usec": 3000 + index,
            "exec_main_start_monotonic_usec": 2900 + index,
            "unit_artifacts": [_file(expected["fragment_path"], 10 + index)],
            "main_executable": _file(expected["executable_path"], 20 + index),
        })
    credential = _file(authority["credential_paths"][0], 40)
    credential.pop("sha256")
    credential["opaque_hmac_sha256"] = hashlib.sha256(b"credential").hexdigest()
    return {
        "services": services,
        "ports": [
            {
                "transport": "udp4",
                "local_address_hex": "00000000",
                "port": port,
                "socket_inode": 500 + index,
                "pid": 200,
                "pid_start_ticks": 2000,
            }
            for index, port in enumerate(authority["required_ports"])
        ],
        "queues": [],
        "maps": [
            _directory(path, 50 + index)
            for index, path in enumerate(authority["map_roots"])
        ],
        "runtime_artifacts": [_file(authority["runtime_artifact_paths"][0], 60)],
        "credentials": [credential],
    }


def _signed_probe(
    authority_path: Path, key_path: Path, authority: dict, *,
    phase: str = "pre", predecessor: str = PREDECESSOR,
) -> dict:
    state = _state(authority)
    payload = {
        "schema": probe.PAYLOAD_SCHEMA,
        "tool": probe.TOOL,
        "synthetic": False,
        "run_binding": {
            "campaign_id": probe.CAMPAIGN_ID,
            "run_nonce": RUN_NONCE,
            "phase": phase,
            "capture_nonce": hashlib.sha256(f"capture-{phase}".encode()).hexdigest(),
            "predecessor_evidence_sha256": predecessor,
        },
        "captured_at_unix_ns": 100,
        "authority_sha256": authority["authority_sha256"],
        "capture_tool": authority["capture_tool"],
        "scope": probe._scope(authority),
        "host": {
            "hostname": authority["hostname"],
            "kernel_release": "6.18-test",
            "architecture": authority["architecture"],
            "machine_identity_sha256": authority["machine_identity_sha256"],
        },
        "state": state,
        "state_sha256": probe.canonical_sha256(state),
    }
    signature = probe._sign(payload, key_path)
    result = {
        "schema": probe.SCHEMA,
        "payload": payload,
        "attestation": {
            "schema": probe.ATTESTATION_SCHEMA,
            "algorithm": "ed25519",
            "key_id": authority["probe_attestation"]["key_id"],
            "payload_sha256": probe.canonical_sha256(payload),
            "signature_base64": base64.b64encode(signature).decode("ascii"),
        },
    }
    result["evidence_sha256"] = probe.canonical_sha256(result)
    return result


def _resign(raw: dict, key_path: Path) -> None:
    raw["payload"]["state_sha256"] = probe.canonical_sha256(raw["payload"]["state"])
    signature = probe._sign(raw["payload"], key_path)
    raw["attestation"]["payload_sha256"] = probe.canonical_sha256(raw["payload"])
    raw["attestation"]["signature_base64"] = base64.b64encode(signature).decode("ascii")
    raw.pop("evidence_sha256", None)
    raw["evidence_sha256"] = probe.canonical_sha256(raw)


def _write_json(path: Path, value: dict, *, newline: bool) -> None:
    path.write_bytes(probe.canonical_bytes(value) + (b"\n" if newline else b""))


def _controller_fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    authority_path, key_path, authority = _authority(tmp_path)
    monkeypatch.setattr(b6, "PUBLIC_HOST_AUTHORITY_PATH", authority_path)
    plan = {
        "schema": controller.PLAN_SCHEMA,
        "campaign_id": probe.CAMPAIGN_ID,
        "ssh_argv": ["ssh", "root@public-fixture"],
        "remote_python": "/usr/bin/python3",
        "remote_capture_tool": "/srv/b6/capture_b6_public_state.py",
        "remote_authority": "/srv/b6/authority.json",
        "remote_signing_key": "/etc/b6/probe.pem",
        "remote_output_root": "/srv/b6/evidence",
        "one_run_argv": [
            "/usr/bin/python3", "-m", "train.multires_one_run",
            "--campaign_mode", probe.CAMPAIGN_ID,
        ],
    }
    plan["plan_sha256"] = probe.canonical_sha256(plan)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan, newline=True)
    pre = _signed_probe(
        authority_path, key_path, authority,
        predecessor=authority["authority_sha256"],
    )
    one = {"evidence_sha256": hashlib.sha256(b"one-run").hexdigest()}
    post = _signed_probe(
        authority_path, key_path, authority, phase="post",
        predecessor=one["evidence_sha256"],
    )
    pre_path = tmp_path / "public-pre.json"
    one_path = tmp_path / "one-run.json"
    post_path = tmp_path / "public-post.json"
    _write_json(pre_path, pre, newline=True)
    _write_json(one_path, one, newline=False)
    _write_json(post_path, post, newline=True)
    launch_id = launch_id_for(RUN_NONCE, pre["evidence_sha256"])
    argvs = {
        "public_pre": controller._remote_probe_argv(
            plan, run_nonce=RUN_NONCE, phase="pre",
            predecessor_evidence_sha256=authority["authority_sha256"],
        ),
        "one_run": [
            *plan["one_run_argv"], "--launch_id", launch_id,
            "--out", str(one_path.resolve()),
        ],
        "public_post": controller._remote_probe_argv(
            plan, run_nonce=RUN_NONCE, phase="post",
            predecessor_evidence_sha256=one["evidence_sha256"],
        ),
    }
    outputs = {
        "public_pre": (pre_path, pre["evidence_sha256"]),
        "one_run": (one_path, one["evidence_sha256"]),
        "public_post": (post_path, post["evidence_sha256"]),
    }
    stages = {}
    for index, name in enumerate(("public_pre", "one_run", "public_post")):
        started = 100 + index * 20
        argv = argvs[name]
        output_path, evidence_sha256 = outputs[name]
        stages[name] = {
            "argv": argv,
            "argv_sha256": hashlib.sha256(
                probe.canonical_bytes(argv)
            ).hexdigest(),
            "started_monotonic_ns": started,
            "completed_monotonic_ns": started + 10,
            "returncode": 0,
            "stdout": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            "stderr": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            "output": b6.file_record(output_path),
            "evidence_sha256": evidence_sha256,
        }
    expected_host = {
        "hostname": "DESKTOP-RTX2080",
        "kernel_release": "6.6.0-microsoft-standard-WSL2",
        "architecture": "x86_64",
        "machine_identity_sha256": hashlib.sha256(b"wsl-machine").hexdigest(),
    }
    ledger = {
        "schema": controller.SCHEMA,
        "tool": controller.TOOL,
        "status": "green",
        "campaign_id": probe.CAMPAIGN_ID,
        "run_nonce": RUN_NONCE,
        "launch_id": launch_id,
        "controller_host": {
            name: expected_host[name]
            for name in ("hostname", "kernel_release", "architecture")
        },
        "controller_tool": b6.file_record(
            Path(controller.__file__).resolve()
        ),
        "plan": b6.file_record(plan_path),
        "authority": b6.file_record(authority_path),
        "stages": stages,
        "ordering_basis": "single-controller-monotonic-ns-v1",
    }
    ledger["ledger_sha256"] = probe.canonical_sha256(ledger)
    paths = {
        "plan": plan_path,
        "one": one_path,
        "pre": pre_path,
        "post": post_path,
        "host": expected_host,
        "authority": authority,
    }
    return ledger, paths


def test_signed_probe_verifies_and_b6_consumes_exact_authority(tmp_path, monkeypatch):
    authority_path, key_path, authority = _authority(tmp_path)
    raw = _signed_probe(authority_path, key_path, authority)
    payload = probe.verify_probe(
        raw, authority_path=authority_path,
        expected_campaign_id=probe.CAMPAIGN_ID,
        expected_run_nonce=RUN_NONCE, expected_phase="pre",
        expected_predecessor_evidence_sha256=PREDECESSOR,
    )
    assert payload["state_sha256"] == probe.canonical_sha256(payload["state"])
    monkeypatch.setattr(b6, "PUBLIC_HOST_AUTHORITY_PATH", authority_path)
    admitted = b6._validate_public_probe(
        raw, "pre", run_nonce=RUN_NONCE, phase="pre",
        predecessor_evidence_sha256=PREDECESSOR,
    )
    assert admitted["evidence_sha256"] == raw["evidence_sha256"]


def test_resealed_payload_without_private_signature_is_rejected(tmp_path):
    authority_path, key_path, authority = _authority(tmp_path)
    raw = _signed_probe(authority_path, key_path, authority)
    raw["payload"]["state"]["services"][0]["main_pid"] += 1
    raw["payload"]["state_sha256"] = probe.canonical_sha256(raw["payload"]["state"])
    raw["attestation"]["payload_sha256"] = probe.canonical_sha256(raw["payload"])
    raw.pop("evidence_sha256")
    raw["evidence_sha256"] = probe.canonical_sha256(raw)
    with pytest.raises(probe.PublicProbeError, match="signature is invalid"):
        probe.verify_probe(
            raw, authority_path=authority_path,
            expected_campaign_id=probe.CAMPAIGN_ID,
            expected_run_nonce=RUN_NONCE, expected_phase="pre",
            expected_predecessor_evidence_sha256=PREDECESSOR,
        )


@pytest.mark.parametrize("field,value,match", [
    ("phase", "post", "replayed"),
    ("run_nonce", hashlib.sha256(b"other").hexdigest(), "replayed"),
    ("predecessor_evidence_sha256", hashlib.sha256(b"other").hexdigest(), "replayed"),
])
def test_signed_probe_replay_or_chain_substitution_is_rejected(
    tmp_path, field, value, match,
):
    authority_path, key_path, authority = _authority(tmp_path)
    raw = _signed_probe(authority_path, key_path, authority)
    raw["payload"]["run_binding"][field] = value
    _resign(raw, key_path)
    with pytest.raises(probe.PublicProbeError, match=match):
        probe.verify_probe(
            raw, authority_path=authority_path,
            expected_campaign_id=probe.CAMPAIGN_ID,
            expected_run_nonce=RUN_NONCE, expected_phase="pre",
            expected_predecessor_evidence_sha256=PREDECESSOR,
        )


def test_signed_inactive_service_and_capture_tool_drift_are_rejected(tmp_path):
    authority_path, key_path, authority = _authority(tmp_path)
    inactive = _signed_probe(authority_path, key_path, authority)
    inactive["payload"]["state"]["services"][0]["active_state"] = "inactive"
    _resign(inactive, key_path)
    with pytest.raises(probe.PublicProbeError, match="service authority/state differs"):
        probe.verify_probe(
            inactive, authority_path=authority_path,
            expected_campaign_id=probe.CAMPAIGN_ID,
            expected_run_nonce=RUN_NONCE, expected_phase="pre",
            expected_predecessor_evidence_sha256=PREDECESSOR,
        )
    drift = _signed_probe(authority_path, key_path, authority)
    drift["payload"]["capture_tool"]["bytes"] += 1
    _resign(drift, key_path)
    with pytest.raises(probe.PublicProbeError, match="capture-tool identity differs"):
        probe.verify_probe(
            drift, authority_path=authority_path,
            expected_campaign_id=probe.CAMPAIGN_ID,
            expected_run_nonce=RUN_NONCE, expected_phase="pre",
            expected_predecessor_evidence_sha256=PREDECESSOR,
        )


def test_capture_rejects_symlink_and_provisioning_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"bytes")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(probe.PublicProbeError, match="symlink"):
        probe._stable_regular_file(link.resolve().parent / link.name, "fixture")
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    authority_path, key_path, _authority_value = _authority(key_dir)
    assert authority_path.is_file() and key_path.is_file()
    with pytest.raises(ProvisioningError, match="already exists"):
        provision(key_path, tmp_path / "second-activation.json")


def test_launch_id_cryptographically_depends_on_nonce_and_signed_pre_probe():
    first = launch_id_for(RUN_NONCE, PREDECESSOR)
    assert first.startswith("b6-") and len(first) == 63
    assert first != launch_id_for(hashlib.sha256(b"other nonce").hexdigest(), PREDECESSOR)
    assert first != launch_id_for(RUN_NONCE, hashlib.sha256(b"other pre").hexdigest())


def test_controller_ledger_validates_exact_invocations_and_monotonic_order(
    tmp_path, monkeypatch,
):
    ledger, paths = _controller_fixture(tmp_path, monkeypatch)
    admitted = b6._validate_controller_ledger(
        ledger, plan_path=paths["plan"], one_run_path=paths["one"],
        public_pre_path=paths["pre"], public_post_path=paths["post"],
        expected_host=paths["host"], authority=paths["authority"],
    )
    assert admitted["launch_id"] == ledger["launch_id"]
    assert admitted["ledger_sha256"] == ledger["ledger_sha256"]


@pytest.mark.parametrize("mutation,match", [
    (
        lambda raw: raw["stages"]["public_post"]["argv"].__setitem__(
            raw["stages"]["public_post"]["argv"].index(
                "--predecessor-evidence-sha256"
            ) + 1,
            hashlib.sha256(b"wrong predecessor").hexdigest(),
        ),
        "public_post invocation differs",
    ),
    (
        lambda raw: raw["stages"]["one_run"].__setitem__(
            "started_monotonic_ns", 50
        ),
        "monotonic stage order differs",
    ),
])
def test_controller_ledger_rejects_resealed_chain_or_order_tampering(
    tmp_path, monkeypatch, mutation, match,
):
    ledger, paths = _controller_fixture(tmp_path, monkeypatch)
    mutation(ledger)
    for stage in ledger["stages"].values():
        argv = stage["argv"]
        stage["argv_sha256"] = hashlib.sha256(
            probe.canonical_bytes(argv)
        ).hexdigest()
    ledger.pop("ledger_sha256")
    ledger["ledger_sha256"] = probe.canonical_sha256(ledger)
    with pytest.raises(b6.B6CampaignError, match=match):
        b6._validate_controller_ledger(
            ledger, plan_path=paths["plan"], one_run_path=paths["one"],
            public_pre_path=paths["pre"], public_post_path=paths["post"],
            expected_host=paths["host"], authority=paths["authority"],
        )


def test_public_probe_v2_schema_accepts_signed_fixture(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    authority_path, key_path, authority = _authority(tmp_path)
    raw = _signed_probe(authority_path, key_path, authority)
    schema = json.loads(
        (Path(__file__).resolve().parents[1]
         / "schemas/q2-multires-b6-public-state-probe-v2.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(raw)
