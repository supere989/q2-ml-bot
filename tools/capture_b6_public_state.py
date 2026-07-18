#!/usr/bin/env python3
"""Capture and authenticate the exact B6 public-server state.

The probe is an Ed25519-signed canonical payload.  Its repository authority
fixes the host, scope, capture-tool bytes, and verification key.  File reads
use no-follow descriptors and stat/read/stat identity checks; directories are
walked through directory descriptors so a renamed or substituted path cannot
silently enter the inventory.

Credential contents are never emitted.  Equality is represented by an opaque,
domain-separated HMAC derived from the dedicated probe signing key.  Neither
credential bytes nor private-key bytes enter evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "q2-multires-b6-public-state-probe-v2"
PAYLOAD_SCHEMA = "q2-multires-b6-public-state-payload-v2"
ATTESTATION_SCHEMA = "q2-ed25519-attestation-v1"
AUTHORITY_SCHEMA = "q2-multires-b6-public-host-authority-v2"
TOOL = "capture_b6_public_state"
TOOL_SOURCE = "tools/capture_b6_public_state.py"
CAMPAIGN_ID = "b6-wsl-g1-no-update"
SIGNATURE_DOMAIN = b"q2-b6-public-state-probe-v2\x00"

_SHA256 = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")
_NONCE = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")
_KEY_ID = re.compile(r"[a-z0-9][a-z0-9._-]{7,63}\Z")
_SOCKET = re.compile(r"socket:\[(\d+)\]\Z")


class PublicProbeError(RuntimeError):
    """The public state could not be captured or authenticated exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicProbeError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PublicProbeError("value is not canonical JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{label} keys differ; missing={sorted(expected-actual)} "
        f"extra={sorted(actual-expected)}",
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} must be a non-placeholder SHA-256")
    return str(value)


def _regular_metadata(info: os.stat_result, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "file",
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _directory_metadata(info: os.stat_result, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "directory",
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _stable_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _absolute_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    _require(expanded.is_absolute(), f"{label} must be absolute")
    normalized = Path(os.path.abspath(expanded))
    _require(str(normalized) == str(expanded), f"{label} must be normalized")
    current = Path("/")
    for part in normalized.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError as error:
            raise PublicProbeError(f"{label} is unavailable: {normalized}") from error
        _require(not stat.S_ISLNK(info.st_mode),
                 f"{label} contains a symlink component: {current}")
    return normalized


def _read_descriptor(fd: int) -> bytes:
    pieces: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(pieces)
        pieces.append(chunk)


def _stable_regular_file(
    path: Path,
    label: str,
    *,
    digest_key: bytes | None = None,
) -> tuple[dict[str, Any], bytes]:
    source = _absolute_path(path, label)
    try:
        before_path = os.lstat(source)
    except OSError as error:
        raise PublicProbeError(f"cannot stat {label}: {source}") from error
    _require(stat.S_ISREG(before_path.st_mode),
             f"{label} is not a regular file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError as error:
        raise PublicProbeError(f"cannot open {label}: {source}") from error
    try:
        before_fd = os.fstat(fd)
        _require(_stable_identity(before_path) == _stable_identity(before_fd),
                 f"{label} identity changed before read: {source}")
        data = _read_descriptor(fd)
        after_fd = os.fstat(fd)
        try:
            after_path = os.lstat(source)
        except OSError as error:
            raise PublicProbeError(f"{label} disappeared after read: {source}") from error
        identity = _stable_identity(before_fd)
        _require(
            identity == _stable_identity(after_fd) == _stable_identity(after_path),
            f"{label} changed during stat/read/stat capture: {source}",
        )
    finally:
        os.close(fd)
    row = _regular_metadata(before_fd, str(source))
    if digest_key is None:
        row["sha256"] = hashlib.sha256(data).hexdigest()
    else:
        row["opaque_hmac_sha256"] = hmac.new(
            digest_key, data, hashlib.sha256
        ).hexdigest()
    return row, data


def _open_child(parent_fd: int, name: str, flags: int, label: str) -> int:
    _require(name not in {"", ".", ".."} and "/" not in name,
             f"{label} has an unsafe directory entry")
    try:
        return os.open(
            name,
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise PublicProbeError(f"cannot open {label} entry {name!r}") from error


def _inventory_directory(root: Path, label: str) -> list[dict[str, Any]]:
    source = _absolute_path(root, label)
    before_path = os.lstat(source)
    _require(stat.S_ISDIR(before_path.st_mode), f"{label} root is not a directory")
    root_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    rows: list[dict[str, Any]] = []

    def walk(directory_fd: int, display: Path) -> None:
        initial = os.fstat(directory_fd)
        _require(stat.S_ISDIR(initial.st_mode), f"{label} entry is not a directory")
        try:
            entries = sorted(os.scandir(directory_fd), key=lambda entry: entry.name)
        except OSError as error:
            raise PublicProbeError(f"cannot enumerate {label}: {display}") from error
        rows.append(_directory_metadata(initial, str(display)))
        for entry in entries:
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PublicProbeError(
                    f"cannot stat {label} entry: {display / entry.name}"
                ) from error
            child_path = display / entry.name
            _require(not stat.S_ISLNK(entry_info.st_mode),
                     f"{label} inventory contains a symlink: {child_path}")
            if stat.S_ISDIR(entry_info.st_mode):
                child_fd = _open_child(
                    directory_fd, entry.name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), label,
                )
                try:
                    opened = os.fstat(child_fd)
                    _require(_stable_identity(entry_info) == _stable_identity(opened),
                             f"{label} directory changed before traversal: {child_path}")
                    walk(child_fd, child_path)
                    final = os.fstat(child_fd)
                    _require(_stable_identity(opened) == _stable_identity(final),
                             f"{label} directory changed during traversal: {child_path}")
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry_info.st_mode):
                child_fd = _open_child(directory_fd, entry.name, os.O_RDONLY, label)
                try:
                    opened = os.fstat(child_fd)
                    _require(_stable_identity(entry_info) == _stable_identity(opened),
                             f"{label} file changed before read: {child_path}")
                    data = _read_descriptor(child_fd)
                    final = os.fstat(child_fd)
                    _require(_stable_identity(opened) == _stable_identity(final),
                             f"{label} file changed during read: {child_path}")
                finally:
                    os.close(child_fd)
                row = _regular_metadata(opened, str(child_path))
                row["sha256"] = hashlib.sha256(data).hexdigest()
                rows.append(row)
            else:
                raise PublicProbeError(
                    f"{label} inventory contains a special file: {child_path}"
                )
        final = os.fstat(directory_fd)
        _require(_stable_identity(initial) == _stable_identity(final),
                 f"{label} directory changed during traversal: {display}")

    try:
        opened_root = os.fstat(root_fd)
        _require(_stable_identity(before_path) == _stable_identity(opened_root),
                 f"{label} root changed before traversal")
        walk(root_fd, source)
        final_root = os.fstat(root_fd)
        after_path = os.lstat(source)
        _require(
            _stable_identity(opened_root)
            == _stable_identity(final_root)
            == _stable_identity(after_path),
            f"{label} root changed during traversal",
        )
    finally:
        os.close(root_fd)
    _require(rows, f"{label} inventory is empty")
    return sorted(rows, key=lambda row: row["path"])


def _machine_identity_sha256(path: Path = Path("/etc/machine-id")) -> str:
    _row, raw = _stable_regular_file(path, "machine identity")
    identity = raw[:-1] if raw.endswith(b"\n") else raw
    _require(
        len(raw) in (32, 33)
        and (len(raw) == 32 or raw[-1:] == b"\n")
        and re.fullmatch(rb"[0-9a-f]{32}", identity) is not None,
        "public host machine identity is malformed",
    )
    return hashlib.sha256(identity).hexdigest()


def _start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        _require(close > 0, f"process {pid} stat comm terminator is absent")
        value = int(raw[close + 1:].split()[19])
    except (OSError, ValueError, IndexError) as error:
        raise PublicProbeError(f"cannot bind process {pid} to /proc start ticks") from error
    _require(value > 0, f"process {pid} has invalid start ticks")
    return value


def _systemctl_properties(name: str) -> dict[str, str]:
    properties = (
        "Id", "LoadState", "ActiveState", "SubState", "FragmentPath",
        "DropInPaths", "MainPID", "ActiveEnterTimestampMonotonic",
        "ExecMainStartTimestampMonotonic", "ExecStart", "WorkingDirectory",
        "RootDirectory", "EnvironmentFiles",
    )
    command = ["systemctl", "show", name, "--no-pager"]
    for prop in properties:
        command.extend(["--property", prop])
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise PublicProbeError(f"cannot inspect service {name!r}") from error
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            _require(key not in values, f"service {name!r} repeated property {key}")
            values[key] = value
    # systemd omits some empty structured properties even when explicitly
    # selected.  Treat only the three documented optional path properties as
    # canonical empty strings; every other requested property must be present.
    for prop in ("DropInPaths", "RootDirectory", "EnvironmentFiles"):
        values.setdefault(prop, "")
    _require(set(values) == set(properties), f"service {name!r} properties differ")
    return values


def _service(authority: Mapping[str, Any]) -> dict[str, Any]:
    name = str(authority["name"])
    values = _systemctl_properties(name)
    try:
        pid = int(values["MainPID"])
        active_enter = int(values["ActiveEnterTimestampMonotonic"] or "0")
        exec_start = int(values["ExecMainStartTimestampMonotonic"] or "0")
    except ValueError as error:
        raise PublicProbeError(f"service {name!r} timestamps/PID are malformed") from error
    _require(
        values["Id"] == name
        and values["LoadState"] == "loaded"
        and values["ActiveState"] == "active"
        and values["SubState"] in {"running", "listening"}
        and pid >= 2 and active_enter > 0 and exec_start > 0,
        f"service {name!r} is not an active live service",
    )
    drop_ins = shlex.split(values["DropInPaths"])
    expected_drop_ins = list(authority["drop_in_paths"])
    _require(
        values["FragmentPath"] == authority["fragment_path"]
        and drop_ins == expected_drop_ins
        and values["WorkingDirectory"] == authority["working_directory"]
        and values["RootDirectory"] == authority["root_directory"]
        and values["EnvironmentFiles"] == authority["environment_files"]
        and values["ExecStart"] == authority["exec_start"],
        f"service {name!r} unit/runtime authority differs",
    )
    unit_paths = [Path(values["FragmentPath"]), *(Path(path) for path in drop_ins)]
    unit_artifacts = [
        _stable_regular_file(path, f"service {name} unit artifact")[0]
        for path in unit_paths
    ]
    try:
        executable_path = os.readlink(f"/proc/{pid}/exe")
    except OSError as error:
        raise PublicProbeError(f"cannot resolve service {name!r} executable") from error
    _require(executable_path == authority["executable_path"],
             f"service {name!r} running executable differs")
    executable = _stable_regular_file(
        Path(executable_path), f"service {name} running executable"
    )[0]
    return {
        "name": name,
        "load_state": values["LoadState"],
        "active_state": values["ActiveState"],
        "sub_state": values["SubState"],
        "fragment_path": values["FragmentPath"],
        "drop_in_paths": drop_ins,
        "working_directory": values["WorkingDirectory"],
        "root_directory": values["RootDirectory"],
        "environment_files": values["EnvironmentFiles"],
        "exec_start": values["ExecStart"],
        "main_pid": pid,
        "main_pid_start_ticks": _start_ticks(pid),
        "active_enter_monotonic_usec": active_enter,
        "exec_main_start_monotonic_usec": exec_start,
        "unit_artifacts": unit_artifacts,
        "main_executable": executable,
    }


def _pid_socket_inodes() -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            start = _start_ticks(pid)
            links = list((proc / "fd").iterdir())
        except (OSError, PublicProbeError):
            continue
        for link in links:
            try:
                target = os.readlink(link)
            except OSError:
                continue
            match = _SOCKET.fullmatch(target)
            if match:
                result[int(match.group(1))] = (pid, start)
    return result


def _socket_rows(ports: set[int]) -> list[dict[str, Any]]:
    owners = _pid_socket_inodes()
    rows = []
    tables = (
        ("tcp4", Path("/proc/net/tcp")), ("tcp6", Path("/proc/net/tcp6")),
        ("udp4", Path("/proc/net/udp")), ("udp6", Path("/proc/net/udp6")),
    )
    for transport, path in tables:
        try:
            lines = path.read_text(encoding="ascii").splitlines()[1:]
        except OSError as error:
            raise PublicProbeError(f"cannot inspect {path}") from error
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                local_hex, port_hex = fields[1].split(":", 1)
                port = int(port_hex, 16)
                inode = int(fields[9])
            except (ValueError, IndexError):
                continue
            if port not in ports:
                continue
            socket_state = fields[3]
            if transport.startswith("tcp") and socket_state != "0A":
                continue
            if transport.startswith("udp") and socket_state != "07":
                continue
            owner = owners.get(inode)
            _require(owner is not None,
                     f"configured public port {port} has no visible process owner")
            rows.append({
                "transport": transport,
                "local_address_hex": local_hex,
                "port": port,
                "socket_inode": inode,
                "pid": owner[0],
                "pid_start_ticks": owner[1],
            })
    _require({row["port"] for row in rows} == ports,
             "one or more configured public ports are not bound")
    return sorted(rows, key=lambda row: (row["port"], row["transport"], row["socket_inode"]))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _row, raw = _stable_regular_file(path, label)
    _require(raw.endswith(b"\n") and raw.count(b"\n") == 1,
             f"{label} must be one canonical newline-terminated JSON record")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicProbeError(f"{label} is not JSON") from error
    _require(isinstance(value, dict) and raw == canonical_bytes(value) + b"\n",
             f"{label} is not canonical JSON")
    return value


def load_authority(path: Path, *, require_active: bool = True) -> dict[str, Any]:
    value = _load_json(path, "B6 public host authority")
    expected = {
        "schema", "hostname", "architecture", "machine_identity_sha256",
        "public_ipv4", "services", "required_ports", "queue_paths",
        "map_roots", "credential_paths", "runtime_artifact_paths",
        "capture_tool", "probe_attestation", "authority_sha256",
    }
    _exact_keys(value, expected, "B6 public host authority")
    supplied = _digest(value["authority_sha256"], "B6 public host authority seal")
    unsigned = dict(value)
    unsigned.pop("authority_sha256")
    _require(
        value["schema"] == AUTHORITY_SCHEMA
        and canonical_sha256(unsigned) == supplied,
        "B6 public host authority identity/seal differs",
    )
    attestation = _mapping(value["probe_attestation"], "probe attestation authority")
    _exact_keys(attestation, {
        "status", "algorithm", "key_id", "public_key_base64", "private_key_path"
    }, "probe attestation authority")
    _require(attestation["algorithm"] == "ed25519"
             and isinstance(attestation["key_id"], str)
             and _KEY_ID.fullmatch(attestation["key_id"]) is not None,
             "probe attestation algorithm/key ID differs")
    if require_active:
        _require(attestation["status"] == "active",
                 "public probe signing key is not operationally provisioned")
    public_text = attestation["public_key_base64"]
    _require(isinstance(public_text, str), "public probe key is absent")
    try:
        public_key = base64.b64decode(public_text, validate=True)
    except (ValueError, TypeError) as error:
        raise PublicProbeError("public probe key is malformed") from error
    _require(len(public_key) == 32 and any(public_key),
             "public probe Ed25519 key is malformed")
    private_path = Path(str(attestation["private_key_path"]))
    _require(
        private_path.is_absolute()
        and str(Path(os.path.abspath(private_path))) == str(private_path),
        "probe private-key path must be absolute and normalized",
    )
    capture_tool = _mapping(value["capture_tool"], "capture-tool authority")
    _exact_keys(capture_tool, {"path", "bytes", "sha256"}, "capture-tool authority")
    _require(capture_tool["path"] == TOOL_SOURCE
             and type(capture_tool["bytes"]) is int and capture_tool["bytes"] > 0,
             "capture-tool authority identity differs")
    _digest(capture_tool["sha256"], "capture-tool authority digest")
    services = value["services"]
    _require(isinstance(services, list) and len(services) == 2,
             "public service authority cardinality differs")
    service_names: list[str] = []
    for item in services:
        service = _mapping(item, "public service authority")
        _exact_keys(service, {
            "name", "fragment_path", "drop_in_paths", "working_directory",
            "root_directory", "environment_files", "exec_start",
            "executable_path",
        }, "public service authority")
        _require(
            isinstance(service["name"], str) and service["name"].endswith(".service")
            and isinstance(service["drop_in_paths"], list)
            and all(isinstance(path, str) and path.startswith("/")
                    for path in service["drop_in_paths"])
            and all(isinstance(service[name], str) for name in (
                "fragment_path", "working_directory", "root_directory",
                "environment_files", "exec_start", "executable_path",
            ))
            and service["fragment_path"].startswith("/")
            and service["working_directory"].startswith("/")
            and service["executable_path"].startswith("/"),
            "public service authority fields differ",
        )
        service_names.append(service["name"])
    _require(
        service_names == ["q2-teacher-server.service", "q2mlbot.service"],
        "public service authority order differs",
    )
    _require(
        isinstance(value["required_ports"], list)
        and value["required_ports"] == [28000, 28001, 28049]
        and value["queue_paths"] == [],
        "public port/queue authority differs",
    )
    for name in ("map_roots", "credential_paths", "runtime_artifact_paths"):
        paths = value[name]
        _require(
            isinstance(paths, list) and bool(paths)
            and paths == sorted(set(paths))
            and all(isinstance(path, str) and path.startswith("/") for path in paths),
            f"public {name} authority differs",
        )
    return value


def _capture_tool_record(authority: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(os.path.abspath(__file__))
    row, _raw = _stable_regular_file(source, "capture tool source")
    record = {
        "path": TOOL_SOURCE,
        "bytes": row["bytes"],
        "sha256": row["sha256"],
    }
    _require(record == authority["capture_tool"],
             "executed capture-tool bytes differ from host authority")
    return record


def _scope(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "services": [item["name"] for item in authority["services"]],
        "ports": authority["required_ports"],
        "queue_paths": authority["queue_paths"],
        "map_roots": authority["map_roots"],
        "credential_paths": authority["credential_paths"],
        "runtime_artifact_paths": authority["runtime_artifact_paths"],
    }


def _credential_rows(paths: Iterable[str], key: bytes) -> list[dict[str, Any]]:
    _require(len(key) == 32 and any(key), "credential HMAC derivation failed")
    return [
        _stable_regular_file(Path(path), "credential", digest_key=key)[0]
        for path in paths
    ]


def _runtime_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = [_stable_regular_file(Path(path), "runtime artifact")[0] for path in paths]
    return sorted(rows, key=lambda row: row["path"])


def _map_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_inventory_directory(Path(path), "map artifact root"))
    paths_seen = [row["path"] for row in rows]
    _require(len(paths_seen) == len(set(paths_seen)), "map inventories overlap")
    return sorted(rows, key=lambda row: row["path"])


def _private_key(path: Path, authority: Mapping[str, Any]) -> tuple[Path, bytes]:
    expected = Path(authority["probe_attestation"]["private_key_path"])
    _require(path == expected, "signing-key path differs from public authority")
    row, raw = _stable_regular_file(path, "public probe signing key")
    _require(row["mode"] & 0o077 == 0, "public probe signing key is group/world accessible")
    _require(
        raw.startswith(b"-----BEGIN PRIVATE KEY-----\n")
        and raw.endswith(b"-----END PRIVATE KEY-----\n"),
        "public probe signing key is not a private PEM",
    )
    credential_hmac_key = hashlib.sha256(
        b"q2-b6-credential-hmac-v1\x00" + raw
    ).digest()
    return path, credential_hmac_key


def _run_openssl(command: Sequence[str], *, stdin: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            list(command), input=stdin, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
    except FileNotFoundError as error:
        raise PublicProbeError("OpenSSL is required for Ed25519 attestation") from error
    _require(completed.returncode == 0,
             f"OpenSSL Ed25519 operation failed with status {completed.returncode}")
    return completed.stdout


def _sign(payload: Mapping[str, Any], private_key: Path) -> bytes:
    # OpenSSL's Ed25519 implementation is one-shot and refuses pipe input
    # because it must know the complete message size before the operation.
    with tempfile.TemporaryDirectory(prefix="q2-b6-ed25519-sign-") as directory:
        message_path = Path(directory) / "message.bin"
        message_path.write_bytes(SIGNATURE_DOMAIN + canonical_bytes(payload))
        signature = _run_openssl(
            (
                "openssl", "pkeyutl", "-sign", "-rawin", "-inkey",
                str(private_key), "-in", str(message_path),
            )
        )
    _require(len(signature) == 64, "OpenSSL returned a malformed Ed25519 signature")
    return signature


def _verify_signature(payload: Mapping[str, Any], signature: bytes, public_key: bytes) -> None:
    _require(len(signature) == 64 and len(public_key) == 32,
             "Ed25519 signature/public key length differs")
    # RFC 8410 SubjectPublicKeyInfo prefix for an Ed25519 raw public key.
    public_der = bytes.fromhex("302a300506032b6570032100") + public_key
    with tempfile.TemporaryDirectory(prefix="q2-b6-ed25519-") as directory:
        root = Path(directory)
        public_path = root / "public.der"
        signature_path = root / "signature.bin"
        message_path = root / "message.bin"
        public_path.write_bytes(public_der)
        signature_path.write_bytes(signature)
        message_path.write_bytes(SIGNATURE_DOMAIN + canonical_bytes(payload))
        completed = subprocess.run(
            [
                "openssl", "pkeyutl", "-verify", "-rawin", "-pubin",
                "-keyform", "DER", "-inkey", str(public_path),
                "-sigfile", str(signature_path), "-in", str(message_path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    _require(completed.returncode == 0, "public probe Ed25519 signature is invalid")


def capture(
    *, authority_path: Path, campaign_id: str, run_nonce: str, phase: str,
    predecessor_evidence_sha256: str, signing_key: Path,
) -> dict[str, Any]:
    authority = load_authority(authority_path)
    _require(campaign_id == CAMPAIGN_ID, "public probe campaign identity differs")
    _require(_NONCE.fullmatch(run_nonce) is not None, "run nonce must be fresh 256-bit hex")
    _require(phase in {"pre", "post"}, "public probe phase differs")
    _digest(predecessor_evidence_sha256, "public probe predecessor evidence")
    services_authority = authority["services"]
    _require(isinstance(services_authority, list) and len(services_authority) == 2,
             "public service authority cardinality differs")
    ports = authority["required_ports"]
    _require(isinstance(ports, list) and len(ports) == len(set(ports)) and ports,
             "public port authority differs")
    private_key, credential_hmac_key = _private_key(signing_key, authority)
    state = {
        "services": [_service(item) for item in services_authority],
        "ports": _socket_rows(set(ports)),
        "queues": [],
        "maps": _map_rows(authority["map_roots"]),
        "runtime_artifacts": _runtime_rows(authority["runtime_artifact_paths"]),
        "credentials": _credential_rows(
            authority["credential_paths"], credential_hmac_key
        ),
    }
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "tool": TOOL,
        "synthetic": False,
        "run_binding": {
            "campaign_id": campaign_id,
            "run_nonce": run_nonce,
            "phase": phase,
            "capture_nonce": secrets_token_hex(),
            "predecessor_evidence_sha256": predecessor_evidence_sha256,
        },
        "captured_at_unix_ns": time.time_ns(),
        "authority_sha256": authority["authority_sha256"],
        "capture_tool": _capture_tool_record(authority),
        "scope": _scope(authority),
        "host": {
            "hostname": os.uname().nodename,
            "kernel_release": os.uname().release,
            "architecture": os.uname().machine,
            "machine_identity_sha256": _machine_identity_sha256(),
        },
        "state": state,
        "state_sha256": canonical_sha256(state),
    }
    signature = _sign(payload, private_key)
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "algorithm": "ed25519",
        "key_id": authority["probe_attestation"]["key_id"],
        "payload_sha256": canonical_sha256(payload),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }
    result = {"schema": SCHEMA, "payload": payload, "attestation": attestation}
    result["evidence_sha256"] = canonical_sha256(result)
    verify_probe(
        result, authority_path=authority_path, expected_campaign_id=campaign_id,
        expected_run_nonce=run_nonce, expected_phase=phase,
        expected_predecessor_evidence_sha256=predecessor_evidence_sha256,
    )
    return result


def secrets_token_hex() -> str:
    # Kept in a named wrapper so tests can make capture nonces deterministic.
    import secrets
    return secrets.token_hex(32)


def verify_probe(
    value: Mapping[str, Any], *, authority_path: Path,
    expected_campaign_id: str, expected_run_nonce: str, expected_phase: str,
    expected_predecessor_evidence_sha256: str,
) -> dict[str, Any]:
    authority = load_authority(authority_path)
    _exact_keys(value, {"schema", "payload", "attestation", "evidence_sha256"},
                "public probe envelope")
    _require(value["schema"] == SCHEMA, "public probe envelope schema differs")
    supplied_seal = _digest(value["evidence_sha256"], "public probe evidence seal")
    unsigned = dict(value)
    unsigned.pop("evidence_sha256")
    _require(canonical_sha256(unsigned) == supplied_seal,
             "public probe evidence seal differs")
    payload = _mapping(value["payload"], "public probe payload")
    _exact_keys(payload, {
        "schema", "tool", "synthetic", "run_binding", "captured_at_unix_ns",
        "authority_sha256", "capture_tool", "scope", "host", "state",
        "state_sha256",
    }, "public probe payload")
    _require(payload["schema"] == PAYLOAD_SCHEMA and payload["tool"] == TOOL
             and payload["synthetic"] is False,
             "public probe payload identity differs")
    _require(payload["authority_sha256"] == authority["authority_sha256"],
             "public probe authority binding differs")
    _require(payload["capture_tool"] == authority["capture_tool"],
             "public probe capture-tool identity differs")
    _require(payload["scope"] == _scope(authority), "public probe scope differs")
    run_binding = _mapping(payload["run_binding"], "public probe run binding")
    _exact_keys(run_binding, {
        "campaign_id", "run_nonce", "phase", "capture_nonce",
        "predecessor_evidence_sha256",
    },
                "public probe run binding")
    _require(
        run_binding["campaign_id"] == expected_campaign_id
        and run_binding["run_nonce"] == expected_run_nonce
        and run_binding["phase"] == expected_phase
        and run_binding["predecessor_evidence_sha256"]
        == expected_predecessor_evidence_sha256
        and isinstance(run_binding["capture_nonce"], str)
        and _NONCE.fullmatch(run_binding["capture_nonce"]) is not None,
        "public probe is replayed or bound to a different campaign phase",
    )
    state = _mapping(payload["state"], "public probe state")
    _exact_keys(state, {
        "services", "ports", "queues", "maps", "runtime_artifacts", "credentials"
    }, "public probe state")
    _require(payload["state_sha256"] == canonical_sha256(state),
             "public probe state seal differs")
    host = _mapping(payload["host"], "public probe host")
    _exact_keys(host, {
        "hostname", "kernel_release", "architecture", "machine_identity_sha256"
    }, "public probe host")
    _require(
        host["hostname"] == authority["hostname"]
        and host["architecture"] == authority["architecture"]
        and host["machine_identity_sha256"] == authority["machine_identity_sha256"],
        "public probe host differs from authority",
    )
    _digest(host["machine_identity_sha256"], "public probe machine identity")
    _require(type(payload["captured_at_unix_ns"]) is int
             and payload["captured_at_unix_ns"] > 0,
             "public probe capture time differs")
    _validate_state_shape(state, authority)
    attestation = _mapping(value["attestation"], "public probe attestation")
    _exact_keys(attestation, {
        "schema", "algorithm", "key_id", "payload_sha256", "signature_base64"
    }, "public probe attestation")
    _require(
        attestation["schema"] == ATTESTATION_SCHEMA
        and attestation["algorithm"] == "ed25519"
        and attestation["key_id"] == authority["probe_attestation"]["key_id"]
        and attestation["payload_sha256"] == canonical_sha256(payload),
        "public probe attestation identity/payload differs",
    )
    try:
        signature = base64.b64decode(attestation["signature_base64"], validate=True)
        public_key = base64.b64decode(
            authority["probe_attestation"]["public_key_base64"], validate=True
        )
    except (TypeError, ValueError) as error:
        raise PublicProbeError("public probe attestation encoding is malformed") from error
    _verify_signature(payload, signature, public_key)
    return dict(payload)


def _validate_artifact_rows(
    value: Any, label: str, *, allow_directories: bool,
) -> list[Mapping[str, Any]]:
    _require(isinstance(value, list) and value, f"{label} inventory is empty")
    rows = [_mapping(item, f"{label} row") for item in value]
    paths: list[str] = []
    for row in rows:
        kind = row.get("kind")
        common = {
            "path", "kind", "device", "inode", "uid", "gid", "mode",
            "mtime_ns", "ctime_ns",
        }
        if kind == "file":
            _exact_keys(row, common | {"bytes", "sha256"}, f"{label} file")
            _digest(row["sha256"], f"{label} file digest")
            _require(type(row["bytes"]) is int and row["bytes"] >= 0,
                     f"{label} file byte count differs")
        elif kind == "directory" and allow_directories:
            _exact_keys(row, common, f"{label} directory")
        else:
            raise PublicProbeError(f"{label} contains a non-regular artifact")
        for field in ("device", "inode", "uid", "gid", "mode", "mtime_ns", "ctime_ns"):
            _require(type(row[field]) is int and row[field] >= 0,
                     f"{label} {field} differs")
        _require(isinstance(row["path"], str) and row["path"].startswith("/"),
                 f"{label} path differs")
        paths.append(row["path"])
    _require(paths == sorted(paths) and len(paths) == len(set(paths)),
             f"{label} inventory is not exact sorted unique")
    return rows


def _validate_state_shape(state: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    services = state["services"]
    _require(isinstance(services, list) and len(services) == len(authority["services"]),
             "public service inventory differs")
    _require([item.get("name") for item in services if isinstance(item, Mapping)]
             == [item["name"] for item in authority["services"]],
             "public service order/identity differs")
    for row, expected in zip(services, authority["services"]):
        service = _mapping(row, "public service")
        _exact_keys(service, {
            "name", "load_state", "active_state", "sub_state", "fragment_path",
            "drop_in_paths", "working_directory", "root_directory",
            "environment_files", "exec_start", "main_pid", "main_pid_start_ticks",
            "active_enter_monotonic_usec", "exec_main_start_monotonic_usec",
            "unit_artifacts", "main_executable",
        }, "public service")
        _require(
            service["name"] == expected["name"]
            and service["fragment_path"] == expected["fragment_path"]
            and service["drop_in_paths"] == expected["drop_in_paths"]
            and service["working_directory"] == expected["working_directory"]
            and service["root_directory"] == expected["root_directory"]
            and service["environment_files"] == expected["environment_files"]
            and service["exec_start"] == expected["exec_start"]
            and service["main_executable"].get("path") == expected["executable_path"]
            and service["load_state"] == "loaded"
            and service["active_state"] == "active"
            and service["sub_state"] in {"running", "listening"}
            and type(service["main_pid"]) is int and service["main_pid"] >= 2
            and type(service["main_pid_start_ticks"]) is int
            and service["main_pid_start_ticks"] > 0
            and type(service["active_enter_monotonic_usec"]) is int
            and service["active_enter_monotonic_usec"] > 0
            and type(service["exec_main_start_monotonic_usec"]) is int
            and service["exec_main_start_monotonic_usec"] > 0,
            "public service authority/state differs",
        )
        units = _validate_artifact_rows(
            service["unit_artifacts"], "service unit artifact", allow_directories=False
        )
        _require([item["path"] for item in units]
                 == [expected["fragment_path"], *expected["drop_in_paths"]],
                 "service unit/drop-in inventory differs")
        executable = _mapping(service["main_executable"], "service main executable")
        _validate_artifact_rows([executable], "service main executable", allow_directories=False)
    ports = state["ports"]
    _require(isinstance(ports, list) and ports
             and {item.get("port") for item in ports if isinstance(item, Mapping)}
             == set(authority["required_ports"]),
             "public socket inventory differs")
    for item in ports:
        row = _mapping(item, "public socket")
        _exact_keys(row, {
            "transport", "local_address_hex", "port", "socket_inode", "pid",
            "pid_start_ticks",
        }, "public socket")
        _require(
            row["transport"] in {"tcp4", "tcp6", "udp4", "udp6"}
            and isinstance(row["local_address_hex"], str)
            and type(row["port"]) is int and row["port"] in authority["required_ports"]
            and type(row["socket_inode"]) is int and row["socket_inode"] > 0
            and type(row["pid"]) is int and row["pid"] >= 2
            and type(row["pid_start_ticks"]) is int and row["pid_start_ticks"] > 0,
            "public socket ownership differs",
        )
    _require(state["queues"] == [], "public VPS queue inventory must be explicitly empty")
    maps = _validate_artifact_rows(state["maps"], "map artifact", allow_directories=True)
    for root in authority["map_roots"]:
        _require(any(item["path"] == root for item in maps),
                 f"map inventory omits authority root {root}")
    _require(all(any(item["path"] == root or item["path"].startswith(root + "/")
                     for root in authority["map_roots"]) for item in maps),
             "map inventory escapes authority roots")
    runtime = _validate_artifact_rows(
        state["runtime_artifacts"], "runtime artifact", allow_directories=False
    )
    _require([item["path"] for item in runtime] == authority["runtime_artifact_paths"],
             "runtime artifact inventory differs")
    credentials = state["credentials"]
    _require(isinstance(credentials, list)
             and [item.get("path") for item in credentials if isinstance(item, Mapping)]
             == authority["credential_paths"],
             "credential inventory differs")
    for item in credentials:
        row = _mapping(item, "credential artifact")
        _exact_keys(row, {
            "path", "kind", "device", "inode", "bytes", "uid", "gid", "mode",
            "mtime_ns", "ctime_ns", "opaque_hmac_sha256",
        }, "credential artifact")
        _require(row["kind"] == "file", "credential is not a regular file")
        for field in (
            "device", "inode", "bytes", "uid", "gid", "mode", "mtime_ns",
            "ctime_ns",
        ):
            _require(type(row[field]) is int and row[field] >= 0,
                     f"credential {field} differs")
        _digest(row["opaque_hmac_sha256"], "credential opaque HMAC")


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(path.expanduser()))
    _require(destination.is_absolute() and not destination.exists() and not destination.is_symlink(),
             "public-state output must be a new absolute path")
    _absolute_path(destination.parent, "public-state output parent")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--phase", required=True, choices=("pre", "post"))
    parser.add_argument("--predecessor-evidence-sha256", required=True)
    parser.add_argument("--signing-key", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = capture(
            authority_path=args.authority, campaign_id=args.campaign_id,
            run_nonce=args.run_nonce, phase=args.phase,
            predecessor_evidence_sha256=args.predecessor_evidence_sha256,
            signing_key=args.signing_key,
        )
        _publish(args.output, value)
    except Exception as error:
        print(f"B6 public-state probe refused: {error}", file=sys.stderr)
        return 1
    print(canonical_bytes(value).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
