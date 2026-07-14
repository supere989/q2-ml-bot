"""Versioned, checksum-attested procedural-map bundle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path


MAP_BUNDLE_VERSION = 2
MAP_ARTIFACT_SUFFIXES = (
    ".bsp",
    ".json",
    ".lattice.json",
    ".routes.json",
)
FARM_MAP_PREFIXES = ("mllive_", "mlteacher_")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def artifact_names(name: str) -> tuple[str, ...]:
    return tuple(f"{name}{suffix}" for suffix in MAP_ARTIFACT_SUFFIXES)


def installed_manifest_name(name: str) -> str:
    return f"{name}.bundle.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_manifest(name: str, generator: dict, payloads: dict[str, bytes]) -> dict:
    expected = set(artifact_names(name))
    if set(payloads) != expected:
        missing = sorted(expected - set(payloads))
        extra = sorted(set(payloads) - expected)
        raise ValueError(f"invalid map artifacts missing={missing} extra={extra}")
    return {
        "bundle_version": MAP_BUNDLE_VERSION,
        "name": name,
        "generator": generator,
        "files": {
            filename: sha256_bytes(payload)
            for filename, payload in sorted(payloads.items())
        },
        # Kept separate so a rolling-upgrade v1 consumer can continue reading
        # the legacy files[name] digest while the v2 consumer requires sizes.
        "file_sizes": {
            filename: len(payload)
            for filename, payload in sorted(payloads.items())
        },
        # Item timing is derived from respawn_s fields in the route graph.
        # Naming that relationship here prevents a consumer from treating the
        # route graph as optional navigation-only metadata.
        "sidecars": {
            "hook_zones": f"{name}.json",
            "lattice_priors": f"{name}.lattice.json",
            "routes": f"{name}.routes.json",
            "item_timing": f"{name}.routes.json",
        },
    }


def encode_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_manifest(manifest: dict, name: str | None = None) -> str:
    if not isinstance(manifest, dict):
        raise ValueError("map bundle manifest must be an object")
    if int(manifest.get("bundle_version", 0)) != MAP_BUNDLE_VERSION:
        raise ValueError(
            f"unsupported map bundle version {manifest.get('bundle_version')!r}"
        )
    manifest_name = str(manifest.get("name", ""))
    if name is not None and manifest_name != name:
        raise ValueError("map bundle manifest name mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(artifact_names(manifest_name)):
        raise ValueError("map bundle is missing required lattice/runtime artifacts")
    sizes = manifest.get("file_sizes")
    if not isinstance(sizes, dict) or set(sizes) != set(files):
        raise ValueError("map bundle artifact-size table mismatch")
    for filename, digest in files.items():
        size = sizes[filename]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid artifact checksum for {filename}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid artifact size for {filename}")
    sidecars = manifest.get("sidecars")
    expected_sidecars = {
        "hook_zones": f"{manifest_name}.json",
        "lattice_priors": f"{manifest_name}.lattice.json",
        "routes": f"{manifest_name}.routes.json",
        "item_timing": f"{manifest_name}.routes.json",
    }
    if sidecars != expected_sidecars:
        raise ValueError("map bundle sidecar contract mismatch")
    return manifest_name


def verify_payloads(manifest: dict, payloads: dict[str, bytes]) -> None:
    name = validate_manifest(manifest)
    expected = set(artifact_names(name))
    if set(payloads) != expected:
        raise ValueError("map bundle payload set does not match manifest")
    for filename, payload in payloads.items():
        if len(payload) != manifest["file_sizes"][filename]:
            raise ValueError(f"size mismatch for {filename}")
        if sha256_bytes(payload) != manifest["files"][filename]:
            raise ValueError(f"checksum mismatch for {filename}")


def verify_installed_artifact(path: Path, manifest: dict) -> bytes:
    name = validate_manifest(manifest)
    if path.name not in artifact_names(name):
        raise ValueError(f"artifact {path.name} is not declared by the map bundle")
    payload = path.read_bytes()
    if len(payload) != manifest["file_sizes"][path.name]:
        raise ValueError(f"size mismatch for {path.name}")
    if sha256_bytes(payload) != manifest["files"][path.name]:
        raise ValueError(f"checksum mismatch for {path.name}")
    return payload


def install_bundle(
    install_dir: Path,
    manifest: dict,
    manifest_payload: bytes,
    payloads: dict[str, bytes],
) -> None:
    """Publish one verified bundle, using the BSP rename as the commit marker."""
    name = validate_manifest(manifest)
    verify_payloads(manifest, payloads)
    if json.loads(manifest_payload) != manifest:
        raise ValueError("encoded map manifest does not match verified manifest")
    install_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    publication = [
        f"{name}.json",
        f"{name}.lattice.json",
        f"{name}.routes.json",
        installed_manifest_name(name),
        f"{name}.bsp",
    ]
    try:
        for filename in publication:
            payload = (
                manifest_payload
                if filename == installed_manifest_name(name)
                else payloads[filename]
            )
            temporary = install_dir / f".{filename}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            temporary.write_bytes(payload)
            staged[filename] = temporary
        # q2ded only sees the map after every policy-side sidecar and its
        # attestation have reached their final names.
        for filename in publication:
            os.replace(staged.pop(filename), install_dir / filename)
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def farm_map_requires_attestation(map_name: str) -> bool:
    return map_name.startswith(FARM_MAP_PREFIXES)
