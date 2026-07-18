import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from harness.atlas_catalog import (
    ATLAS_MAP_SPEC_SCHEMA, AtlasCatalogError, author_catalog,
    canonical_bytes, load_atlas_catalog,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extension(root: Path, *, corrupt: bool = False):
    path = root / "q2_lattice_rs.so"
    path.write_bytes(b"catalog-bound-rust-extension")

    class Runtime:
        @staticmethod
        def empty(atlas, map_sha, origin, epoch, client, count, steps):
            values = {
                "atlas_sha256": atlas, "map_sha256": map_sha,
                "origin": tuple(origin), "map_epoch": epoch,
                "client_id": client, "client_count": count,
                "environment_steps": steps, "client_life_epoch": 0,
                "server_frame": 0, "last_event_id": 0,
                "accepted_event_count": 0, "cell_count": 0,
            }
            if corrupt:
                values["client_id"] = client + 1
            return SimpleNamespace(**values)

    return SimpleNamespace(__file__=str(path.resolve()), DynRuntime=Runtime), path


def emit_map(root: Path, name: str) -> dict:
    root.mkdir(parents=True)
    bsp = root / f"{name}.bsp"
    atlas = root / f"{name}.atlas.bin"
    bsp.write_bytes(f"IBSP-{name}".encode())
    atlas.write_bytes(f"ATLAS-{name}".encode())
    objectives = root / f"{name}.objectives.json"
    objectives.write_text(json.dumps({
        "map_name": name, "atlas_sha256": sha(atlas),
    }, sort_keys=True))
    atlas_manifest = root / f"{name}.atlas.manifest.json"
    atlas_manifest.write_text(json.dumps({
        "grid": {"origin": [0, 0, 0]},
        "bsp": {"canonical_map_id": name, "sha256": sha(bsp),
                "size_bytes": bsp.stat().st_size},
        "artifacts": {f"{name}.atlas.bin": {
            "sha256_uncompressed": sha(atlas),
            "uncompressed_size": atlas.stat().st_size,
        }},
    }, sort_keys=True))
    bundle = root / f"{name}.bundle.json"
    bundle.write_text(json.dumps({
        "bundle_version": 3, "name": name,
        "files": {f"{name}.bsp": sha(bsp)},
        "analysis_files": {
            f"{name}.atlas.manifest.json": sha(atlas_manifest),
            f"{name}.objectives.json": sha(objectives),
        },
    }, sort_keys=True))
    return {
        "schema": ATLAS_MAP_SPEC_SCHEMA, "map_name": name,
        "bsp": str(bsp.resolve()), "atlas": str(atlas.resolve()),
        "atlas_manifest": str(atlas_manifest.resolve()),
        "bundle_manifest": str(bundle.resolve()),
        "objectives": str(objectives.resolve()),
    }


def publish(tmp_path, specs):
    path = tmp_path / "catalog.json"
    module, rust_path = extension(tmp_path)
    document = author_catalog(
        specs, catalog_path=path, rust_extension_path=rust_path,
        extension_module=module,
    )
    path.write_bytes(canonical_bytes(document) + b"\n")
    return path, document, module


def test_catalog_is_unique_sorted_content_sealed_and_builds_full_client_view(tmp_path):
    second = emit_map(tmp_path / "second", "q2dm1")
    first = emit_map(tmp_path / "first", "gen_arena")
    path, document, module = publish(tmp_path, [second, first])
    catalog = load_atlas_catalog(
        path, expected_sha256=document["atlas_catalog_sha256"],
        extension_module=module,
    )
    assert [record.map_name for record in catalog.maps] == ["gen_arena", "q2dm1"]
    artifacts = catalog.provider_artifacts_for_client(2)
    assert set(artifacts) == {"gen_arena", "q2dm1"}
    assert all(item.dyn_checkpoint_path is None for item in artifacts.values())
    assert all(item.dyn_snapshot_path is None for item in artifacts.values())


def test_catalog_rejects_duplicate_unknown_and_rebound_artifacts(tmp_path):
    spec = emit_map(tmp_path / "map", "q2dm1")
    with pytest.raises(AtlasCatalogError, match="repeat"):
        module, rust_path = extension(tmp_path)
        author_catalog(
            [spec, spec], catalog_path=tmp_path / "duplicate.json",
            rust_extension_path=rust_path, extension_module=module,
        )
    path, _document, module = publish(tmp_path, [spec])
    Path(spec["atlas"]).write_bytes(b"rebound")
    with pytest.raises(AtlasCatalogError, match="digest"):
        load_atlas_catalog(path, extension_module=module)


def test_catalog_rejects_bad_dyn_authority_and_noncanonical_file(tmp_path):
    spec = emit_map(tmp_path / "map", "q2dm1")
    module, rust_path = extension(tmp_path, corrupt=True)
    with pytest.raises(AtlasCatalogError, match="client_id"):
        author_catalog(
            [spec], catalog_path=tmp_path / "bad.json",
            rust_extension_path=rust_path, extension_module=module,
        )
    spec = emit_map(tmp_path / "other", "q2dm2")
    module, rust_path = extension(tmp_path)
    path = tmp_path / "pretty.json"
    document = author_catalog(
        [spec], catalog_path=path, rust_extension_path=rust_path,
        extension_module=module,
    )
    path.write_text(json.dumps(document, indent=2))
    with pytest.raises(AtlasCatalogError, match="canonical"):
        load_atlas_catalog(path, extension_module=module)


def test_catalog_identity_survives_portable_tree_copy(tmp_path):
    source = tmp_path / "source"
    spec = emit_map(source / "maps/q2dm1", "q2dm1")
    path, document, _module = publish(source, [spec])
    destination = tmp_path / "destination"
    shutil.copytree(source, destination)
    copied_path = destination / path.relative_to(source)
    copied_module = SimpleNamespace(
        __file__=str(destination / "q2_lattice_rs.so"),
        DynRuntime=extension(destination)[0].DynRuntime,
    )
    catalog = load_atlas_catalog(
        copied_path, expected_sha256=document["atlas_catalog_sha256"],
        extension_module=copied_module,
    )
    assert catalog.atlas_catalog_sha256 == document["atlas_catalog_sha256"]
