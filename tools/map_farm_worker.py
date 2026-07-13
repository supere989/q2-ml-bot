#!/usr/bin/env python3
"""Precompile lit live maps and serve an atomic ready queue over HTTP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE_MAP_PREFIX = "mllive"
SEED_RANGES = {
    "mllive": (10_000_000, 49_999_999),
    "mlteacher": (50_000_000, 99_999_999),
}


class MapFarm:
    def __init__(self, queue_dir: Path, q2_root: Path, depth: int, threads: int = 8,
                 prefix: str = LIVE_MAP_PREFIX):
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", prefix):
            raise ValueError(f"invalid map prefix {prefix!r}")
        self.queue_dir = queue_dir
        self.q2_root = q2_root
        self.depth = max(1, depth)
        self.threads = max(1, threads)
        self.prefix = prefix
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.building = ""
        self.last_error = ""

    def ready(self) -> list[Path]:
        return sorted(self.queue_dir.glob(f"{self.prefix}_*.zip"))

    def status(self) -> dict:
        with self.lock:
            paths = self.ready()
            ready = [path.stem for path in paths]
            ready_styles = {}
            for path in paths:
                try:
                    with zipfile.ZipFile(path) as bundle:
                        manifest = json.loads(bundle.read("manifest.json"))
                    ready_styles[path.stem] = str(
                        manifest.get("generator", {}).get("style", "unknown")
                    )
                except (OSError, KeyError, ValueError, zipfile.BadZipFile):
                    ready_styles[path.stem] = "unknown"
            return {
                "ready": ready,
                "ready_styles": ready_styles,
                "ready_count": len(ready),
                "target_depth": self.depth,
                "prefix": self.prefix,
                "building": self.building,
                "last_error": self.last_error,
            }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_one(self) -> None:
        low, high = SEED_RANGES.get(self.prefix, (10_000_000, 99_999_999))
        seed = random.SystemRandom().randint(low, high)
        name = f"{self.prefix}_{seed:08d}"
        generated = ROOT / "maps" / "generated"
        map_path = generated / f"{name}.map"
        bsp_path = generated / f"{name}.bsp"
        json_path = generated / f"{name}.json"
        meta_path = generated / f"{name}.meta.json"
        q2tool = ROOT / "maps" / "q2tools" / "bin" / "q2tool"
        with self.lock:
            self.building = name
            self.last_error = ""
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "maps" / "generator.py"),
                 "--seed", str(seed), "--name", name],
                cwd=ROOT, check=True, timeout=30,
            )
            subprocess.run(
                [str(q2tool), "-bsp", "-vis", "-fast", "-rad",
                 "-threads", str(self.threads),
                 "-moddir", str(self.q2_root / "baseq2"), str(map_path)],
                cwd=ROOT, check=True,
            )
            if not bsp_path.is_file() or not json_path.is_file() or not meta_path.is_file():
                raise RuntimeError(f"compiler did not produce map artifacts for {name}")
            generator_meta = json.loads(meta_path.read_text())
            manifest = {
                "name": name,
                "generator": generator_meta,
                "files": {
                    bsp_path.name: self._sha256(bsp_path),
                    json_path.name: self._sha256(json_path),
                },
            }
            temporary = self.queue_dir / f".{name}.{uuid.uuid4().hex}.tmp"
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
                bundle.write(bsp_path, bsp_path.name)
                bundle.write(json_path, json_path.name)
                bundle.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            os.replace(temporary, self.queue_dir / f"{name}.zip")
            print(f"[farm] ready {name}", flush=True)
        except Exception as error:
            with self.lock:
                self.last_error = str(error)
            print(f"[farm] build failed for {name}: {error}", flush=True)
            time.sleep(5)
        finally:
            for suffix in (
                ".map", ".prt", ".bsp", ".json", ".meta.json",
                ".lattice.json", ".routes.json",
            ):
                try:
                    (generated / f"{name}{suffix}").unlink()
                except FileNotFoundError:
                    pass
            with self.lock:
                self.building = ""

    def build_loop(self) -> None:
        while not self.stop.is_set():
            if len(self.ready()) < self.depth:
                self._build_one()
                continue
            self.wake.wait(5)
            self.wake.clear()

    def claim(self) -> tuple[str, Path] | None:
        with self.lock:
            ready = self.ready()
            if not ready:
                return None
            # Do not make queue filename order become the map rotation.
            source = random.SystemRandom().choice(ready)
            claimed = source.with_name(f".{source.stem}.{uuid.uuid4().hex}.claimed")
            os.replace(source, claimed)
            return source.stem, claimed

    def restore(self, name: str, claimed: Path) -> None:
        if claimed.exists():
            os.replace(claimed, self.queue_dir / f"{name}.zip")


def handler_for(farm: MapFarm):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = json.dumps(farm.status(), sort_keys=True).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/next.zip":
                self.send_error(404)
                return
            claimed = farm.claim()
            if claimed is None:
                self.send_response(204)
                self.end_headers()
                return
            name, path = claimed
            try:
                size = path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Q2-Map-Name", name)
                self.end_headers()
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                path.unlink()
                farm.wake.set()
                print(f"[farm] delivered {name}", flush=True)
            except (BrokenPipeError, ConnectionResetError, OSError):
                farm.restore(name, path)

        def log_message(self, fmt, *args):
            print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="100.86.206.50")
    parser.add_argument("--port", type=int, default=32510)
    parser.add_argument("--queue_depth", type=int, default=2)
    parser.add_argument("--prefix", default=LIVE_MAP_PREFIX)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--queue_dir", default=str(ROOT / "maps" / "farm_queue"))
    parser.add_argument("--q2_root", default=os.environ.get("Q2_ROOT", str(ROOT.parent / "runtime")))
    args = parser.parse_args()

    farm = MapFarm(Path(args.queue_dir), Path(args.q2_root), args.queue_depth,
                   args.threads, args.prefix)
    builder = threading.Thread(target=farm.build_loop, name="map-builder", daemon=True)
    builder.start()
    server = ThreadingHTTPServer((args.bind, args.port), handler_for(farm))
    print(f"[farm] listening on http://{args.bind}:{args.port} "
          f"prefix={farm.prefix} depth={farm.depth}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        farm.stop.set()
        farm.wake.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
