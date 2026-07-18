"""Immutable multi-map Atlas catalog for the fresh v2 policy lineage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ATLAS_CATALOG_SCHEMA = "q2-multires-atlas-catalog-v1"
ATLAS_CATALOG_DOMAIN = "q2-multires-atlas-catalog-content-v1"
ATLAS_MAP_SPEC_SCHEMA = "q2-multires-atlas-catalog-map-spec-v1"
CLIENT_COUNT = 4
_SHA256 = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")
_MAP_NAME = re.compile(r"[A-Za-z0-9_.-]{1,63}\Z")
_RELATIVE_PATH = re.compile(r"(?!/)(?!.*\.\.)[A-Za-z0-9_./-]+\Z")


class AtlasCatalogError(ValueError):
    """Raised before any catalog byte is admitted or used."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AtlasCatalogError("Atlas catalog is not canonical JSON") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path, label: str) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise AtlasCatalogError(
            f"{label} must be an absolute regular non-symlink file"
        )
    return source.resolve()


def _portable_path(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise AtlasCatalogError(
            f"{label} must be within the portable catalog root {root}"
        ) from error
    if (
        not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or not _RELATIVE_PATH.fullmatch(relative.as_posix())
    ):
        raise AtlasCatalogError(f"{label} has an unsafe catalog-relative path")
    return relative.as_posix()


def artifact_record(path: Path, root: Path, label: str) -> dict[str, str]:
    source = _file(path, label)
    return {
        "path": _portable_path(source, root, label),
        "sha256": sha256_file(source),
    }


def _validate_record(value: object, root: Path, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise AtlasCatalogError(f"{label} record is malformed")
    relative = Path(str(value["path"]))
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or not _RELATIVE_PATH.fullmatch(relative.as_posix())
    ):
        raise AtlasCatalogError(f"{label} path is not portable catalog-relative")
    source = _file(root / relative, label)
    if _portable_path(source, root, label) != relative.as_posix():
        raise AtlasCatalogError(f"{label} path escapes or aliases the catalog root")
    if not isinstance(value["sha256"], str) or not _SHA256.fullmatch(
        value["sha256"]
    ) or sha256_file(source) != value["sha256"]:
        raise AtlasCatalogError(f"{label} record digest differs")
    return source


@dataclass(frozen=True)
class AtlasCatalogMap:
    map_name: str
    bsp: Path
    atlas: Path
    atlas_sha256: str
    atlas_manifest: Path
    bundle_manifest: Path
    objectives: Path


@dataclass(frozen=True)
class AtlasCatalog:
    path: Path
    file_sha256: str
    atlas_catalog_sha256: str
    rust_extension: Path
    maps: tuple[AtlasCatalogMap, ...]

    def by_name(self) -> dict[str, AtlasCatalogMap]:
        return {record.map_name: record for record in self.maps}

    def provider_artifacts_for_client(self, client_index: int) -> dict[str, Any]:
        """Build the exact full-catalog provider map for one client.

        Dyn is created empty by the catalog-bound Rust authority for every
        observed map epoch.  No snapshot/checkpoint or previous epoch is
        exposed through this conversion.
        """
        if type(client_index) is not int or not 0 <= client_index < CLIENT_COUNT:
            raise AtlasCatalogError("catalog client index must be within 0..3")
        from .rust_multires_provider import RustMapArtifacts

        return {
            record.map_name: RustMapArtifacts(
                bundle_manifest_path=record.bundle_manifest,
                uncompressed_atlas_path=record.atlas,
                expected_atlas_sha256=record.atlas_sha256,
                dyn_snapshot_path=None,
                environment_steps_base=0,
                dyn_checkpoint_path=None,
                dyn_checkpoint_sha256="",
                checkpoint_client_life_epoch=0,
                checkpoint_server_frame=0,
            )
            for record in self.maps
        }


def _verify_empty_dyn_authority(
    extension_module: Any,
    *,
    label: str,
    atlas_sha256: str,
    map_sha256: str,
    origin: tuple[int, int, int],
    client_index: int,
) -> None:
    dyn_type = getattr(extension_module, "DynRuntime", None)
    empty = getattr(dyn_type, "empty", None)
    if not callable(empty):
        raise AtlasCatalogError("Rust DynRuntime.empty authority is required")
    try:
        runtime = empty(
            atlas_sha256, map_sha256, list(origin), 0,
            client_index, CLIENT_COUNT, 0,
        )
    except Exception as error:
        raise AtlasCatalogError(f"{label} Rust empty-Dyn creation failed") from error
    identities = {
        "atlas_sha256": atlas_sha256,
        "map_sha256": map_sha256,
        "origin": tuple(origin),
        "map_epoch": 0,
        "client_id": client_index,
        "client_count": CLIENT_COUNT,
        "environment_steps": 0,
        "client_life_epoch": 0,
        "server_frame": 0,
        "last_event_id": 0,
        "accepted_event_count": 0,
        "cell_count": 0,
    }
    for field, expected_value in identities.items():
        actual = getattr(runtime, field, None)
        if field == "origin":
            actual = tuple(actual or ())
        if actual != expected_value:
            raise AtlasCatalogError(f"{label} Rust {field} identity differs")


def _map_payload(
    spec: Mapping[str, Any], label: str, *, root: Path, extension_module: Any
) -> dict[str, Any]:
    required = {
        "schema", "map_name", "bsp", "atlas", "atlas_manifest",
        "bundle_manifest", "objectives",
    }
    if set(spec) != required or spec.get("schema") != ATLAS_MAP_SPEC_SCHEMA:
        raise AtlasCatalogError(f"{label} fields/schema differ")
    name = spec["map_name"]
    if not isinstance(name, str) or not _MAP_NAME.fullmatch(name):
        raise AtlasCatalogError(f"{label} map_name is invalid")
    paths = {
        field: _file(Path(str(spec[field])), f"{label}.{field}")
        for field in (
            "bsp", "atlas", "atlas_manifest", "bundle_manifest", "objectives"
        )
    }
    atlas_sha256 = sha256_file(paths["atlas"])
    map_sha256 = sha256_file(paths["bsp"])
    try:
        bundle = json.loads(paths["bundle_manifest"].read_text(encoding="utf-8"))
        atlas_manifest = json.loads(
            paths["atlas_manifest"].read_text(encoding="utf-8")
        )
        objectives = json.loads(paths["objectives"].read_text(encoding="utf-8"))
        atlas_record = atlas_manifest["artifacts"][f"{name}.atlas.bin"]
        origin = tuple(int(value) for value in atlas_manifest["grid"]["origin"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise AtlasCatalogError(f"{label} map documents are invalid") from error
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("bundle_version") != 3
        or bundle.get("name") != name
        or bundle.get("files", {}).get(f"{name}.bsp") != map_sha256
        or bundle.get("analysis_files", {}).get(f"{name}.atlas.manifest.json")
        != sha256_file(paths["atlas_manifest"])
        or bundle.get("analysis_files", {}).get(f"{name}.objectives.json")
        != sha256_file(paths["objectives"])
        or atlas_record.get("sha256_uncompressed") != atlas_sha256
        or atlas_manifest.get("bsp", {}).get("canonical_map_id") != name
        or atlas_manifest.get("bsp", {}).get("sha256")
        != map_sha256
        or objectives.get("map_name", objectives.get("canonical_map_id")) != name
        or objectives.get("atlas_sha256") != atlas_sha256
        or len(origin) != 3
    ):
        raise AtlasCatalogError(f"{label} bundle/Atlas/objective identities differ")
    for index in range(CLIENT_COUNT):
        _verify_empty_dyn_authority(
            extension_module, label=f"{label}.client[{index}]",
            atlas_sha256=atlas_sha256, map_sha256=map_sha256,
            origin=origin, client_index=index,
        )
    return {
        "map_name": name,
        "bsp": artifact_record(paths["bsp"], root, f"{label}.bsp"),
        "atlas": artifact_record(paths["atlas"], root, f"{label}.atlas"),
        "atlas_sha256": atlas_sha256,
        "atlas_manifest": artifact_record(
            paths["atlas_manifest"], root, f"{label}.atlas_manifest"
        ),
        "bundle_manifest": artifact_record(
            paths["bundle_manifest"], root, f"{label}.bundle_manifest"
        ),
        "objectives": artifact_record(paths["objectives"], root, f"{label}.objectives"),
    }


def author_catalog(
    map_specs: Sequence[Mapping[str, Any]], *, catalog_path: Path,
    rust_extension_path: Path, extension_module: Any,
) -> dict[str, Any]:
    if not map_specs:
        raise AtlasCatalogError("Atlas catalog requires at least one map")
    destination = Path(catalog_path).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise AtlasCatalogError("catalog path must be absolute and non-symlink")
    root = destination.parent.resolve()
    rust_extension = artifact_record(
        rust_extension_path, root, "Rust Dyn creation authority"
    )
    module_path = getattr(extension_module, "__file__", None)
    if module_path is None or Path(str(module_path)).resolve() != _file(
        rust_extension_path, "Rust Dyn creation authority"
    ):
        raise AtlasCatalogError("loaded Rust extension differs from catalog authority")
    maps = [
        _map_payload(
            spec, f"map_spec[{index}]", root=root,
            extension_module=extension_module,
        )
        for index, spec in enumerate(map_specs)
    ]
    names = [record["map_name"] for record in maps]
    if len(names) != len(set(names)):
        raise AtlasCatalogError("Atlas catalog map names repeat")
    maps.sort(key=lambda record: record["map_name"])
    dyn_creation = {
        "mode": "rust-empty-per-map-epoch-v1",
        "snapshot_schema": "Q2LAT002",
        "client_count": CLIENT_COUNT,
        "environment_steps": 0,
        "thermal_checkpoint_fields": 0,
    }
    content = {
        "domain": ATLAS_CATALOG_DOMAIN,
        "rust_extension": rust_extension,
        "dyn_creation": dyn_creation,
        "maps": maps,
    }
    return {
        "schema": ATLAS_CATALOG_SCHEMA,
        "domain": ATLAS_CATALOG_DOMAIN,
        "map_count": len(maps),
        "rust_extension": rust_extension,
        "dyn_creation": dyn_creation,
        "maps": maps,
        "atlas_catalog_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
    }


def load_atlas_catalog(
    path: Path, *, expected_sha256: str | None = None,
    extension_module: Any,
) -> AtlasCatalog:
    source = _file(path, "Atlas catalog")
    root = source.parent.resolve()
    raw = source.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AtlasCatalogError("Atlas catalog is invalid JSON") from error
    if not isinstance(document, Mapping) or set(document) != {
        "schema", "domain", "map_count", "rust_extension", "dyn_creation",
        "maps", "atlas_catalog_sha256"
    } or document.get("schema") != ATLAS_CATALOG_SCHEMA or document.get(
        "domain"
    ) != ATLAS_CATALOG_DOMAIN:
        raise AtlasCatalogError("Atlas catalog fields/schema/domain differ")
    if raw != canonical_bytes(document) + b"\n":
        raise AtlasCatalogError("Atlas catalog is not canonical JSON plus LF")
    maps = document["maps"]
    if not isinstance(maps, list) or not maps or document["map_count"] != len(maps):
        raise AtlasCatalogError("Atlas catalog map count differs")
    names = [item.get("map_name") if isinstance(item, Mapping) else None for item in maps]
    if names != sorted(names) or len(names) != len(set(names)):
        raise AtlasCatalogError("Atlas catalog map names are not unique sorted")
    dyn_creation = document["dyn_creation"]
    if dyn_creation != {
        "mode": "rust-empty-per-map-epoch-v1",
        "snapshot_schema": "Q2LAT002",
        "client_count": CLIENT_COUNT,
        "environment_steps": 0,
        "thermal_checkpoint_fields": 0,
    }:
        raise AtlasCatalogError("Atlas catalog Dyn creation authority differs")
    rust_extension = _validate_record(
        document["rust_extension"], root, "Rust Dyn creation authority"
    )
    module_path = getattr(extension_module, "__file__", None)
    if module_path is None or Path(str(module_path)).resolve() != rust_extension:
        raise AtlasCatalogError("loaded Rust extension differs from catalog authority")
    content = {
        "domain": ATLAS_CATALOG_DOMAIN,
        "rust_extension": document["rust_extension"],
        "dyn_creation": dyn_creation,
        "maps": maps,
    }
    seal = hashlib.sha256(canonical_bytes(content)).hexdigest()
    if document["atlas_catalog_sha256"] != seal or (
        expected_sha256 is not None and expected_sha256 != seal
    ):
        raise AtlasCatalogError("Atlas catalog content seal differs")
    records: list[AtlasCatalogMap] = []
    for index, item in enumerate(maps):
        if not isinstance(item, Mapping) or set(item) != {
            "map_name", "bsp", "atlas", "atlas_sha256", "atlas_manifest",
            "bundle_manifest", "objectives",
        }:
            raise AtlasCatalogError(f"Atlas catalog maps[{index}] fields differ")
        name = item["map_name"]
        if not isinstance(name, str) or not _MAP_NAME.fullmatch(name):
            raise AtlasCatalogError(f"Atlas catalog maps[{index}] map_name is invalid")
        atlas = _validate_record(item["atlas"], root, f"maps[{index}].atlas")
        if item["atlas_sha256"] != sha256_file(atlas):
            raise AtlasCatalogError(f"maps[{index}] Atlas digest differs")
        bsp = _validate_record(item["bsp"], root, f"maps[{index}].bsp")
        atlas_manifest = _validate_record(
            item["atlas_manifest"], root, f"maps[{index}].atlas_manifest"
        )
        bundle_manifest = _validate_record(
            item["bundle_manifest"], root, f"maps[{index}].bundle_manifest"
        )
        objectives = _validate_record(
            item["objectives"], root, f"maps[{index}].objectives"
        )
        # Rebuild the semantic record from the referenced bytes on every load.
        # The catalog seal alone only proves the catalog document; it must not
        # make a hand-authored or subsequently rebound map tuple admissible.
        reconstructed = _map_payload({
            "schema": ATLAS_MAP_SPEC_SCHEMA,
            "map_name": name,
            "bsp": str(bsp),
            "atlas": str(atlas),
            "atlas_manifest": str(atlas_manifest),
            "bundle_manifest": str(bundle_manifest),
            "objectives": str(objectives),
        }, f"maps[{index}]", root=root, extension_module=extension_module)
        if reconstructed != dict(item):
            raise AtlasCatalogError(
                f"Atlas catalog maps[{index}] semantic record differs"
            )
        records.append(AtlasCatalogMap(
            map_name=name,
            bsp=bsp,
            atlas=atlas,
            atlas_sha256=str(item["atlas_sha256"]),
            atlas_manifest=atlas_manifest,
            bundle_manifest=bundle_manifest,
            objectives=objectives,
        ))
    return AtlasCatalog(
        path=source, file_sha256=sha256_file(source),
        atlas_catalog_sha256=seal, rust_extension=rust_extension,
        maps=tuple(records),
    )
