import hashlib
import io
import json
import sys
import types
import zipfile

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from tools.map_farm_client import FarmMapGenerator, ShuffledStockRotation
from tools.map_farm_worker import MapFarm


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


def _bundle(name="mllive_12345678", corrupt=False, style=None):
    payloads = {
        f"{name}.bsp": b"compiled-bsp",
        f"{name}.json": b'{"spawn": []}',
    }
    manifest = {
        "name": name,
        "files": {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
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


def test_farm_bundle_is_verified_and_installed_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(_bundle()))
    generator = FarmMapGenerator("http://map-farm.test:32510")

    generator._fetch()

    assert generator._result == "mllive_12345678"
    install = tmp_path / "runtime" / "baseq2" / "maps"
    assert (install / "mllive_12345678.bsp").read_bytes() == b"compiled-bsp"
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
