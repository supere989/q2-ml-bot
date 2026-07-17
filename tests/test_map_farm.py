import hashlib
from importlib.machinery import ModuleSpec
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    onnxruntime = types.ModuleType("onnxruntime")
    onnxruntime.__spec__ = ModuleSpec("onnxruntime", loader=None)
    sys.modules["onnxruntime"] = onnxruntime

from tools.map_farm_client import FarmMapGenerator, ShuffledStockRotation
from tools.map_farm_worker import MapFarm
from tools.map_bundle import (
    analysis_artifact_names,
    build_manifest,
    declared_artifact_names,
    installed_manifest_name,
)


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def _bundle(
    name="mllive_12345678",
    corrupt=False,
    style=None,
    missing=None,
    *,
    bundle_version=2,
    analysis=False,
    corrupt_analysis=False,
    missing_analysis=None,
):
    payloads = {
        f"{name}.bsp": b"compiled-bsp",
        f"{name}.json": b'{"spawn": []}',
        f"{name}.lattice.json": b'{"cell_size": 256, "objectives": []}',
        f"{name}.routes.json": b'{"version": 1, "nodes": [], "routes": []}',
    }
    if missing is not None:
        payloads.pop(f"{name}.{missing}.json")
    manifest = {
        "bundle_version": bundle_version,
        "name": name,
        "files": {
            key: hashlib.sha256(value).hexdigest()
            for key, value in payloads.items()
        },
        "file_sizes": {key: len(value) for key, value in payloads.items()},
        "sidecars": {
            "hook_zones": f"{name}.json",
            "lattice_priors": f"{name}.lattice.json",
            "routes": f"{name}.routes.json",
            "item_timing": f"{name}.routes.json",
        },
    }
    if style:
        manifest["generator"] = {"style": style}
    if corrupt:
        manifest["files"][f"{name}.bsp"] = "0" * 64
    if analysis or bundle_version == 3:
        analysis_payloads = {
            f"{name}.analysis.manifest.json": b'{"status":"passed"}\n',
            f"{name}.atlas.manifest.json": b'{"schema_version":1}\n',
            f"{name}.atlas.bin.zst": b"Q2AZST01fixture-atlas",
            f"{name}.navigation.bin.zst": b"fixture-navigation",
            f"{name}.visibility.bin.zst": b"fixture-visibility",
            f"{name}.design-signature.json": b'{"coordinate_free":true}\n',
            f"{name}.objectives.json": b'{"schema":"q2-atlas-objectives-v1"}\n',
        }
        payloads.update(analysis_payloads)
        manifest["analysis_files"] = {
            key: hashlib.sha256(value).hexdigest()
            for key, value in analysis_payloads.items()
        }
        manifest["analysis_file_sizes"] = {
            key: len(value) for key, value in analysis_payloads.items()
        }
        manifest["analysis_sidecars"] = {
            "analysis_manifest": f"{name}.analysis.manifest.json",
            "atlas_manifest": f"{name}.atlas.manifest.json",
            "atlas_transport": f"{name}.atlas.bin.zst",
            "navigation": f"{name}.navigation.bin.zst",
            "visibility": f"{name}.visibility.bin.zst",
            "design_signature": f"{name}.design-signature.json",
            "objectives": f"{name}.objectives.json",
        }
        if corrupt_analysis:
            manifest["analysis_files"][f"{name}.atlas.bin.zst"] = "0" * 64
        if missing_analysis is not None:
            payloads.pop(f"{name}.{missing_analysis}")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for filename, payload in payloads.items():
            bundle.writestr(filename, payload)
        bundle.writestr("manifest.json", json.dumps(manifest))
    return stream.getvalue()


def _legacy_bundle(name="mllive_12345678"):
    payloads = {
        f"{name}.bsp": b"compiled-bsp",
        f"{name}.json": b'{"spawn": []}',
    }
    manifest = {
        "name": name,
        "files": {
            key: hashlib.sha256(value).hexdigest()
            for key, value in payloads.items()
        },
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for filename, payload in payloads.items():
            bundle.writestr(filename, payload)
        bundle.writestr("manifest.json", json.dumps(manifest))
    return stream.getvalue()


def test_farm_bundle_is_verified_and_installed_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(_bundle()))
    generator = FarmMapGenerator("http://map-farm.test:32510")

    generator._fetch()

    assert generator._result == "mllive_12345678"
    install = tmp_path / "runtime" / "baseq2" / "maps"
    assert (install / "mllive_12345678.bsp").read_bytes() == b"compiled-bsp"
    assert (install / "mllive_12345678.lattice.json").is_file()
    assert (install / "mllive_12345678.routes.json").is_file()
    installed_manifest = json.loads(
        (install / installed_manifest_name("mllive_12345678")).read_text()
    )
    assert installed_manifest["sidecars"]["item_timing"].endswith(".routes.json")
    assert not list(install.glob("*.tmp"))


def test_v2_install_accepts_optional_analysis_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(_bundle(analysis=True)),
    )
    generator = FarmMapGenerator("http://map-farm.test:32510")

    generator._fetch()

    assert generator._result == "mllive_12345678"
    install = tmp_path / "runtime" / "baseq2" / "maps"
    for filename in analysis_artifact_names("mllive_12345678"):
        assert (install / filename).is_file()


def test_bundle_v3_is_isolated_by_default_and_explicitly_admitted(tmp_path, monkeypatch):
    encoded = _bundle(bundle_version=3)
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "default-runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response(encoded),
    )
    default_generator = FarmMapGenerator("http://map-farm.test:32510")
    default_generator._fetch()
    assert default_generator._result is None
    assert "bundle v3 is disabled" in default_generator._error
    assert not (tmp_path / "default-runtime" / "baseq2" / "maps").exists()

    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "isolated-runtime"))
    isolated_generator = FarmMapGenerator(
        "http://map-farm.test:32510", allow_bundle_v3=True,
    )
    isolated_generator._fetch()
    assert isolated_generator._result == "mllive_12345678"
    install = tmp_path / "isolated-runtime" / "baseq2" / "maps"
    with zipfile.ZipFile(io.BytesIO(encoded)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    for filename in declared_artifact_names(manifest):
        assert (install / filename).is_file()
    # Generator timing routes and Atlas objectives are distinct authorities;
    # v3 installation must preserve both byte identities without collision.
    assert (install / "mllive_12345678.routes.json").read_bytes() == (
        b'{"version": 1, "nodes": [], "routes": []}'
    )
    assert (install / "mllive_12345678.objectives.json").read_bytes() == (
        b'{"schema":"q2-atlas-objectives-v1"}\n'
    )
    assert manifest["sidecars"]["item_timing"].endswith(".routes.json")
    assert manifest["analysis_sidecars"]["objectives"].endswith(
        ".objectives.json"
    )


def test_bundle_v3_rejects_missing_or_mismatched_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    generator = FarmMapGenerator(
        "http://map-farm.test:32510", allow_bundle_v3=True,
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(
            _bundle(bundle_version=3, missing_analysis="visibility.bin.zst")
        ),
    )
    generator._fetch()
    assert generator._result is None
    assert "archive member set is invalid" in generator._error

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(
            _bundle(bundle_version=3, corrupt_analysis=True)
        ),
    )
    generator._fetch()
    assert generator._result is None
    assert "checksum mismatch" in generator._error
    assert not (tmp_path / "runtime" / "baseq2" / "maps").exists()


def test_bundle_v3_mid_publication_failure_restores_prior_generation(
    tmp_path, monkeypatch,
):
    encoded = _bundle(bundle_version=3)
    with zipfile.ZipFile(io.BytesIO(encoded)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    install = tmp_path / "runtime" / "baseq2" / "maps"
    install.mkdir(parents=True)
    expected_prior = {
        filename: f"prior:{filename}".encode()
        for filename in declared_artifact_names(manifest)
    }
    expected_prior[installed_manifest_name(manifest["name"])] = b"prior-manifest"
    for filename, payload in expected_prior.items():
        (install / filename).write_bytes(payload)

    real_replace = os.replace
    failed = False

    def fail_during_analysis_publication(source, destination):
        nonlocal failed
        if not failed and Path(destination).name.endswith("navigation.bin.zst"):
            failed = True
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr("tools.map_bundle.os.replace", fail_during_analysis_publication)
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response(encoded),
    )
    generator = FarmMapGenerator(
        "http://map-farm.test:32510", allow_bundle_v3=True,
    )
    generator._fetch()

    assert generator._result is None
    assert "injected publication failure" in generator._error
    for filename, payload in expected_prior.items():
        assert (install / filename).read_bytes() == payload
    assert not list(install.glob(".*.tmp"))
    assert not list(install.glob(".*.bak"))


def test_farm_bundle_rejects_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response(_bundle(corrupt=True)),
    )
    generator = FarmMapGenerator("http://map-farm.test:32510")

    generator._fetch()

    assert generator._result is None
    assert "checksum mismatch" in generator._error
    assert not (tmp_path / "runtime" / "baseq2" / "maps").exists()


def test_farm_bundle_rejects_incomplete_v2_bundle_without_lattice_sidecars(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(_bundle(missing="lattice")),
    )
    generator = FarmMapGenerator("http://map-farm.test:32510")

    generator._fetch()

    assert generator._result is None
    assert "missing required lattice/runtime artifacts" in generator._error
    assert not (tmp_path / "runtime" / "baseq2" / "maps").exists()


def test_farm_bundle_rejects_v1_bundle_for_safe_queue_migration(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(_legacy_bundle()),
    )
    generator = FarmMapGenerator("http://map-farm.test:32510")

    generator._fetch()

    assert generator._result is None
    assert "unsupported map bundle version" in generator._error
    assert not (tmp_path / "runtime" / "baseq2" / "maps").exists()


def test_v2_bundle_keeps_v1_digest_shape_for_worker_first_rollout():
    encoded = _bundle()

    with zipfile.ZipFile(io.BytesIO(encoded)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        for suffix in ("bsp", "json"):
            filename = f"mllive_12345678.{suffix}"
            assert isinstance(manifest["files"][filename], str)
            assert hashlib.sha256(bundle.read(filename)).hexdigest() == (
                manifest["files"][filename]
            )


def test_manifest_builder_requires_complete_analysis_only_for_v3():
    name = "mllive_12345678"
    core = {
        f"{name}.bsp": b"bsp",
        f"{name}.json": b"hook",
        f"{name}.lattice.json": b"lattice",
        f"{name}.routes.json": b"routes",
    }
    analysis = {
        filename: f"analysis:{filename}".encode()
        for filename in analysis_artifact_names(name)
    }

    plain_v2 = build_manifest(name, {}, core)
    optional_v2 = build_manifest(name, {}, core, analysis_payloads=analysis)
    complete_v3 = build_manifest(
        name, {}, core, analysis_payloads=analysis, bundle_version=3,
    )
    assert "analysis_files" not in plain_v2
    assert set(optional_v2["analysis_files"]) == set(analysis)
    assert set(declared_artifact_names(complete_v3)) == set(core) | set(analysis)
    with pytest.raises(ValueError, match="complete Atlas analysis artifact set"):
        build_manifest(name, {}, core, bundle_version=3)


def test_worker_claim_is_atomic_and_wakes_replenishment(tmp_path):
    farm = MapFarm(tmp_path / "queue", tmp_path / "runtime", depth=2)
    ready = farm.queue_dir / "mllive_12345678.zip"
    ready.write_bytes(_bundle())

    name, claimed = farm.claim()

    assert name == "mllive_12345678"
    assert claimed.exists()
    assert not ready.exists()
    assert farm.claim() is None
    farm.restore(name, claimed)
    assert ready.exists()


def test_worker_health_reports_queued_generator_styles(tmp_path):
    farm = MapFarm(tmp_path / "queue", tmp_path / "runtime", depth=2)
    ready = farm.queue_dir / "mllive_12345678.zip"
    ready.write_bytes(_bundle(style="arena_lanes"))

    status = farm.status()

    assert status["ready_styles"] == {"mllive_12345678": "arena_lanes"}
    assert status["arena_target"] == 1


def test_worker_arena_target_scales_with_queue_depth(tmp_path):
    live = MapFarm(tmp_path / "live", tmp_path / "runtime", depth=2)
    teacher = MapFarm(tmp_path / "teacher", tmp_path / "runtime", depth=4)

    assert live.arena_target == 1
    assert teacher.arena_target == 2


def test_worker_publishes_complete_attested_bundle_and_wsl_mirror(
    tmp_path, monkeypatch,
):
    root = tmp_path / "source"
    generated = root / "maps" / "generated"
    generated.mkdir(parents=True)
    monkeypatch.setattr("tools.map_farm_worker.ROOT", root)

    def fake_run(command, **_kwargs):
        if Path(command[1]).name == "generator.py":
            name = command[command.index("--name") + 1]
            (generated / f"{name}.map").write_text("map-source")
            (generated / f"{name}.json").write_text("# hook zones\n")
            (generated / f"{name}.meta.json").write_text(
                json.dumps({"style": command[command.index("--style") + 1]})
            )
            (generated / f"{name}.lattice.json").write_text(
                '{"cell_size": 256, "objectives": []}'
            )
            (generated / f"{name}.routes.json").write_text(
                '{"version": 1, "nodes": [], "routes": []}'
            )
        else:
            Path(command[-1]).with_suffix(".bsp").write_bytes(b"compiled-bsp")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tools.map_farm_worker.subprocess.run", fake_run)
    farm = MapFarm(tmp_path / "queue", tmp_path / "runtime", depth=1)

    farm._build_one()

    [ready] = farm.ready()
    name = ready.stem
    with zipfile.ZipFile(ready) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert set(bundle.namelist()) == {
            "manifest.json",
            f"{name}.bsp",
            f"{name}.json",
            f"{name}.lattice.json",
            f"{name}.routes.json",
        }
    assert manifest["bundle_version"] == 2
    assert manifest["sidecars"]["item_timing"] == f"{name}.routes.json"
    mirror = tmp_path / "runtime" / "baseq2" / "maps"
    assert (mirror / f"{name}.lattice.json").is_file()
    assert (mirror / f"{name}.routes.json").is_file()
    assert (mirror / installed_manifest_name(name)).is_file()


def test_teacher_queue_cannot_consume_live_maps(tmp_path):
    farm = MapFarm(
        tmp_path / "queue", tmp_path / "runtime", depth=4, prefix="mlteacher",
    )
    live = farm.queue_dir / "mllive_12345678.zip"
    teacher = farm.queue_dir / "mlteacher_56781234.zip"
    live.write_bytes(_bundle("mllive_12345678"))
    teacher.write_bytes(_bundle("mlteacher_56781234"))

    name, _claimed = farm.claim()

    assert name == "mlteacher_56781234"
    assert live.exists()


def test_teacher_consumer_rejects_live_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response(_bundle()),
    )
    generator = FarmMapGenerator("http://map-farm.test:32513", prefix="mlteacher")

    generator._fetch()

    assert generator._result is None
    assert "invalid farm map name" in generator._error


def test_stock_rotation_shuffles_without_repeats_in_a_cycle():
    rotation = ShuffledStockRotation(["q2dm2", "q2dm4", "q2dm6", "q2dm8"], seed=2204)

    first = [rotation.next() for _ in range(4)]
    second = [rotation.next() for _ in range(4)]

    assert set(first) == {"q2dm2", "q2dm4", "q2dm6", "q2dm8"}
    assert set(second) == set(first)
    assert first[-1] != second[0]
    assert "q2dm1" not in first + second
