"""Versioned, checksum-attested procedural-map bundle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path


MAP_BUNDLE_VERSION = 2
MAP_BUNDLE_V3 = 3
SUPPORTED_MAP_BUNDLE_VERSIONS = (MAP_BUNDLE_VERSION, MAP_BUNDLE_V3)
MAP_ARTIFACT_SUFFIXES = (
    ".bsp",
    ".json",
    ".lattice.json",
    ".routes.json",
)
# Atlas objectives have a sole versioned artifact. The legacy .routes.json
# remains only the generator route/item-timing sidecar, so the two authorities
# cannot overwrite each other during bundle assembly or installation.
MAP_ANALYSIS_ARTIFACT_SUFFIXES = (
    ".analysis.manifest.json",
    ".atlas.manifest.json",
    ".atlas.bin.zst",
    ".navigation.bin.zst",
    ".visibility.bin.zst",
    ".design-signature.json",
    ".objectives.json",
)
FARM_MAP_PREFIXES = ("mllive_", "mlteacher_")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def artifact_names(name: str) -> tuple[str, ...]:
    return tuple(f"{name}{suffix}" for suffix in MAP_ARTIFACT_SUFFIXES)


def analysis_artifact_names(name: str) -> tuple[str, ...]:
    return tuple(f"{name}{suffix}" for suffix in MAP_ANALYSIS_ARTIFACT_SUFFIXES)


def declared_artifact_names(manifest: dict) -> tuple[str, ...]:
    """Return the exact payload set after validating the manifest."""
    name = validate_manifest(manifest)
    analysis = tuple(sorted(manifest.get("analysis_files", {})))
    return (*artifact_names(name), *analysis)


def installed_manifest_name(name: str) -> str:
    return f"{name}.bundle.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    name: str,
    generator: dict,
    payloads: dict[str, bytes],
    *,
    analysis_payloads: dict[str, bytes] | None = None,
    bundle_version: int = MAP_BUNDLE_VERSION,
) -> dict:
    expected = set(artifact_names(name))
    if set(payloads) != expected:
        missing = sorted(expected - set(payloads))
        extra = sorted(set(payloads) - expected)
        raise ValueError(f"invalid map artifacts missing={missing} extra={extra}")
    if bundle_version not in SUPPORTED_MAP_BUNDLE_VERSIONS:
        raise ValueError(f"unsupported map bundle version {bundle_version!r}")
    analysis_payloads = analysis_payloads or {}
    expected_analysis = set(analysis_artifact_names(name))
    if analysis_payloads and set(analysis_payloads) != expected_analysis:
        missing = sorted(expected_analysis - set(analysis_payloads))
        extra = sorted(set(analysis_payloads) - expected_analysis)
        raise ValueError(
            f"invalid analysis artifacts missing={missing} extra={extra}"
        )
    if bundle_version == MAP_BUNDLE_V3 and not analysis_payloads:
        raise ValueError("bundle v3 requires the complete Atlas analysis artifact set")
    manifest = {
        "bundle_version": bundle_version,
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
    if analysis_payloads:
        manifest["analysis_files"] = {
            filename: sha256_bytes(payload)
            for filename, payload in sorted(analysis_payloads.items())
        }
        manifest["analysis_file_sizes"] = {
            filename: len(payload)
            for filename, payload in sorted(analysis_payloads.items())
        }
        manifest["analysis_sidecars"] = _analysis_sidecars(name)
    return manifest


def encode_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_manifest(manifest: dict, name: str | None = None) -> str:
    if not isinstance(manifest, dict):
        raise ValueError("map bundle manifest must be an object")
    bundle_version = manifest.get("bundle_version")
    if (
        not isinstance(bundle_version, int)
        or isinstance(bundle_version, bool)
        or bundle_version not in SUPPORTED_MAP_BUNDLE_VERSIONS
    ):
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
    analysis_files = manifest.get("analysis_files")
    analysis_sizes = manifest.get("analysis_file_sizes")
    analysis_sidecars = manifest.get("analysis_sidecars")
    if analysis_files is None and analysis_sizes is None and analysis_sidecars is None:
        if bundle_version == MAP_BUNDLE_V3:
            raise ValueError("bundle v3 is missing mandatory Atlas analysis artifacts")
    else:
        expected_analysis = set(analysis_artifact_names(manifest_name))
        if not isinstance(analysis_files, dict) or set(analysis_files) != expected_analysis:
            raise ValueError("map bundle analysis-artifact table mismatch")
        if not isinstance(analysis_sizes, dict) or set(analysis_sizes) != expected_analysis:
            raise ValueError("map bundle analysis-artifact-size table mismatch")
        for filename, digest in analysis_files.items():
            size = analysis_sizes[filename]
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"invalid analysis artifact checksum for {filename}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"invalid analysis artifact size for {filename}")
        if analysis_sidecars != _analysis_sidecars(manifest_name):
            raise ValueError("map bundle analysis sidecar contract mismatch")
    return manifest_name


def verify_payloads(manifest: dict, payloads: dict[str, bytes]) -> None:
    name = validate_manifest(manifest)
    expected = set(artifact_names(name)) | set(manifest.get("analysis_files", {}))
    if set(payloads) != expected:
        raise ValueError("map bundle payload set does not match manifest")
    for filename, payload in payloads.items():
        digest_table, size_table = _artifact_tables(manifest, filename)
        if len(payload) != size_table[filename]:
            raise ValueError(f"size mismatch for {filename}")
        if sha256_bytes(payload) != digest_table[filename]:
            raise ValueError(f"checksum mismatch for {filename}")


def verify_installed_artifact(path: Path, manifest: dict) -> bytes:
    name = validate_manifest(manifest)
    if path.name not in declared_artifact_names(manifest):
        raise ValueError(f"artifact {path.name} is not declared by the map bundle")
    payload = path.read_bytes()
    digest_table, size_table = _artifact_tables(manifest, path.name)
    if len(payload) != size_table[path.name]:
        raise ValueError(f"size mismatch for {path.name}")
    if sha256_bytes(payload) != digest_table[path.name]:
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
        *sorted(manifest.get("analysis_files", {})),
        f"{name}.json",
        f"{name}.lattice.json",
        f"{name}.routes.json",
        installed_manifest_name(name),
        f"{name}.bsp",
    ]
    backups: dict[str, Path] = {}
    published: list[str] = []
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
            destination = install_dir / filename
            if destination.exists():
                backup = install_dir / (
                    f".{filename}.{os.getpid()}.{uuid.uuid4().hex}.bak"
                )
                backup.write_bytes(destination.read_bytes())
                backups[filename] = backup
        # q2ded only sees the map after every policy-side sidecar and its
        # attestation have reached their final names.
        for filename in publication:
            os.replace(staged[filename], install_dir / filename)
            staged.pop(filename)
            published.append(filename)
    except Exception:
        # Restore the complete prior generation if publication fails before
        # the BSP commit marker. Readers holding an old file descriptor remain
        # valid while new opens return the restored generation.
        for filename in reversed(published):
            destination = install_dir / filename
            backup = backups.pop(filename, None)
            if backup is None:
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            else:
                os.replace(backup, destination)
        raise
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass


def farm_map_requires_attestation(map_name: str) -> bool:
    return map_name.startswith(FARM_MAP_PREFIXES)


def _analysis_sidecars(name: str) -> dict[str, str]:
    return {
        "analysis_manifest": f"{name}.analysis.manifest.json",
        "atlas_manifest": f"{name}.atlas.manifest.json",
        "atlas_transport": f"{name}.atlas.bin.zst",
        "navigation": f"{name}.navigation.bin.zst",
        "visibility": f"{name}.visibility.bin.zst",
        "design_signature": f"{name}.design-signature.json",
        "objectives": f"{name}.objectives.json",
    }


def _artifact_tables(manifest: dict, filename: str) -> tuple[dict, dict]:
    if filename in manifest["files"]:
        return manifest["files"], manifest["file_sizes"]
    return manifest["analysis_files"], manifest["analysis_file_sizes"]
