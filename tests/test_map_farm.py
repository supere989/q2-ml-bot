import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from tools.map_farm_client import FarmMapGenerator, ShuffledStockRotation
from tools.map_farm_worker import MapFarm
from tools.map_bundle import installed_manifest_name


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


def _bundle(name="mllive_12345678", corrupt=False, style=None, missing=None):
    payloads = {
        f"{name}.bsp": b"compiled-bsp",
        f"{name}.json": b'{"spawn": []}',
        f"{name}.lattice.json": b'{"cell_size": 256, "objectives": []}',
        f"{name}.routes.json": b'{"version": 1, "nodes": [], "routes": []}',
    }
    if missing is not None:
        payloads.pop(f"{name}.{missing}.json")
    manifest = {
        "bundle_version": 2,
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
