"""Dependency-free map-farm client and stock-rotation helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import socket
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class FarmMapGenerator:
    """Download and atomically install checksum-attested farm maps."""

    def __init__(self, base_url: str, prefix: str = "mllive"):
        self.url = f"{base_url.rstrip('/')}/next.zip"
        self.prefix = prefix
        self.q2_root = Path(os.environ.get("Q2_ROOT", str(ROOT.parent / "q2_lithium_merge")))
        self._thread = None
        self._result = None
        self._error = ""
        self._last_attempt = 0.0

    def _valid_name(self, name: str) -> bool:
        return (
            name.startswith(f"{self.prefix}_")
            and len(name) == len(self.prefix) + 9
            and name[len(self.prefix) + 1:].isdigit()
        )

    def _fetch(self):
        try:
            with urllib.request.urlopen(self.url, timeout=15) as response:
                if response.status == 204:
                    raise RuntimeError("remote map queue is empty")
                bundle_data = response.read()
            with zipfile.ZipFile(io.BytesIO(bundle_data)) as bundle:
                manifest = json.loads(bundle.read("manifest.json"))
                name = str(manifest["name"])
                if not self._valid_name(name):
                    raise ValueError(f"invalid farm map name {name!r}")
                expected = {
                    f"{name}.bsp": manifest["files"][f"{name}.bsp"],
                    f"{name}.json": manifest["files"][f"{name}.json"],
                }
                payloads = {filename: bundle.read(filename) for filename in expected}
                for filename, digest in expected.items():
                    actual = hashlib.sha256(payloads[filename]).hexdigest()
                    if actual != digest:
                        raise ValueError(f"checksum mismatch for {filename}")
            install_dir = self.q2_root / "baseq2" / "maps"
            install_dir.mkdir(parents=True, exist_ok=True)
            for filename, payload in payloads.items():
                temporary = install_dir / f".{filename}.{os.getpid()}.tmp"
                temporary.write_bytes(payload)
                os.replace(temporary, install_dir / filename)
            self._result = name
        except Exception as error:
            self._error = str(error)

    def start(self):
        if self._thread is not None:
            return
        now = time.monotonic()
        if now - self._last_attempt < 5.0:
            return
        self._last_attempt = now
        self._result = None
        self._error = ""
        self._thread = threading.Thread(target=self._fetch, name="map-farm-fetch", daemon=True)
        self._thread.start()

    def poll(self) -> str | None:
        if self._thread is None or self._thread.is_alive():
            return None
        self._thread.join()
        self._thread = None
        result, error = self._result, self._error
        self._result = None
        self._error = ""
        if result:
            print(f"[mapfarm] installed remote map: {result}", flush=True)
            return result
        if error:
            print(f"[mapfarm] fetch deferred: {error}", flush=True)
        return None

    @property
    def busy(self) -> bool:
        return self._thread is not None


class ShuffledStockRotation:
    """Shuffle without replacement, avoiding a cycle-boundary repeat."""

    def __init__(self, maps: list[str], seed: int):
        if not maps:
            raise ValueError("stock rotation requires at least one map")
        self.maps = list(dict.fromkeys(maps))
        self.rng = random.Random(seed)
        self._bag: list[str] = []
        self._last = ""

    def next(self) -> str:
        if not self._bag:
            self._bag = self.maps.copy()
            self.rng.shuffle(self._bag)
            if len(self._bag) > 1 and self._bag[-1] == self._last:
                self._bag[0], self._bag[-1] = self._bag[-1], self._bag[0]
        result = self._bag.pop()
        self._last = result
        return result


def query_live_mapname(port: int) -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2.0)
            sock.sendto(b"\xff\xff\xff\xffstatus\n", ("127.0.0.1", port))
            data, _ = sock.recvfrom(4096)
        text = data.decode(errors="replace")
        for line in text.split("\n"):
            if "\\mapname\\" in line:
                parts = line.split("\\")
                return parts[parts.index("mapname") + 1]
    except (OSError, ValueError, IndexError):
        pass
    return ""
