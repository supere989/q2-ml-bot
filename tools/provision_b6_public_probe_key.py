#!/usr/bin/env python3
"""Generate one dedicated Ed25519 key for B6 public-probe signing.

Run this tool on the public host as the unprivileged account that will execute
the capture.  The private PEM is created once with mode 0600 and is never
printed or placed in evidence.  The canonical activation record contains only
the public key and the authority fields that must be reviewed and pinned in
``B6-PUBLIC-HOST-AUTHORITY.json``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "q2-multires-b6-public-probe-key-activation-v1"
DER_PREFIX = bytes.fromhex("302a300506032b6570032100")


class ProvisioningError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvisioningError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _new_absolute(path: Path, label: str) -> Path:
    value = path.expanduser()
    _require(value.is_absolute() and str(Path(os.path.abspath(value))) == str(value),
             f"{label} must be absolute and normalized")
    _require(not value.exists() and not value.is_symlink(), f"{label} already exists")
    parent = value.parent
    _require(parent.is_dir() and not parent.is_symlink(),
             f"{label} parent must be a real directory")
    return value


def _openssl(command: Sequence[str], *, stdin: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            list(command), input=stdin, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
    except FileNotFoundError as error:
        raise ProvisioningError("OpenSSL is required") from error
    _require(result.returncode == 0,
             f"OpenSSL provisioning command failed with status {result.returncode}")
    return result.stdout


def _publish(path: Path, content: bytes, mode: int) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def provision(private_key_path: Path, activation_path: Path) -> dict[str, Any]:
    private_path = _new_absolute(private_key_path, "private-key destination")
    public_path = _new_absolute(activation_path, "activation-record destination")
    private_pem = _openssl(("openssl", "genpkey", "-algorithm", "ED25519"))
    _require(private_pem.startswith(b"-----BEGIN PRIVATE KEY-----\n")
             and private_pem.endswith(b"-----END PRIVATE KEY-----\n"),
             "OpenSSL returned a malformed private key")
    public_der = _openssl(
        ("openssl", "pkey", "-pubout", "-outform", "DER"), stdin=private_pem
    )
    _require(len(public_der) == len(DER_PREFIX) + 32
             and public_der.startswith(DER_PREFIX),
             "OpenSSL returned a malformed Ed25519 public key")
    public_raw = public_der[len(DER_PREFIX):]
    public_sha256 = hashlib.sha256(public_raw).hexdigest()
    key_id = f"b6-public-probe-{public_sha256[:16]}"
    activation = {
        "schema": SCHEMA,
        "algorithm": "ed25519",
        "key_id": key_id,
        "public_key_base64": base64.b64encode(public_raw).decode("ascii"),
        "public_key_sha256": public_sha256,
        "private_key_path": str(private_path),
        "generated_at_unix_ns": time.time_ns(),
        "activation_sha256": "",
    }
    unsigned = dict(activation)
    unsigned.pop("activation_sha256")
    activation["activation_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    # Publish the secret first; if public-record publication fails, the key is
    # retained rather than silently regenerated under the same intended ID.
    _publish(private_path, private_pem, 0o600)
    info = os.lstat(private_path)
    _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
             "private key did not publish as a mode-0600 regular file")
    _publish(public_path, _canonical(activation) + b"\n", 0o644)
    return activation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--activation-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        activation = provision(args.private_key, args.activation_output)
    except Exception as error:
        print(f"B6 public-probe key provisioning refused: {error}", file=sys.stderr)
        return 1
    # This contains public material only.
    print(_canonical(activation).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
