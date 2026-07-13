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

from tools.live_match_onnx import FarmMapGenerator
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


def _bundle(name="mllive_12345678", corrupt=False):
    payloads = {
        f"{name}.bsp": b"compiled-bsp",
        f"{name}.json": b'{"spawn": []}',
    }
    manifest = {
        "name": name,
        "files": {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    }
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
