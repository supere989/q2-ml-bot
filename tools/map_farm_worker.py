#!/usr/bin/env python3
"""Precompile lit live maps and serve an atomic ready queue over HTTP."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))

from tools.map_bundle import (
    artifact_names,
    build_manifest,
    encode_manifest,
    install_bundle,
)

LIVE_MAP_PREFIX = "mllive"
ARENA_STYLES = ("arena_open", "arena_vertical", "arena_lanes")
LEGACY_STYLES = ("open", "towers", "canyon", "pits")
SEED_RANGES = {
    "mllive": (10_000_000, 49_999_999),
    "mlteacher": (50_000_000, 99_999_999),
}
VPS_COMPILATION_FALLBACK_ENABLED = False


class MapFarm:
    def __init__(self, queue_dir: Path, q2_root: Path, depth: int, threads: int = 8,
                 prefix: str = LIVE_MAP_PREFIX, arena_fraction: float = 0.5,
                 install_dir: Path | None = None):
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", prefix):
            raise ValueError(f"invalid map prefix {prefix!r}")
        self.queue_dir = queue_dir
        self.q2_root = q2_root
        self.depth = max(1, depth)
        self.threads = max(1, threads)
        self.prefix = prefix
        # This WSL mirror is the network trainer's authoritative sidecar root.
        # It is published before the corresponding bundle can be claimed.
        self.install_dir = (
            Path(install_dir)
            if install_dir is not None
            else self.q2_root / "baseq2" / "maps"
        )
        self.arena_target = min(
            self.depth,
            max(0, int(round(self.depth * max(0.0, min(1.0, arena_fraction))))),
        )
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.building = ""
        self.last_error = ""
        self.analysis_status = "not_attempted"
        self.analysis_error = ""

    def report_analysis_failure(self, error: str) -> None:
        """Expose isolated analysis failure without enabling local fallback."""

        if not isinstance(error, str) or not error.strip():
            raise ValueError("analysis failure must include a nonempty error")
        with self.lock:
            self.analysis_status = "failed"
            self.analysis_error = error.strip()

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
                "arena_target": self.arena_target,
                "prefix": self.prefix,
                "building": self.building,
                "last_error": self.last_error,
                "analysis_status": self.analysis_status,
                "analysis_error": self.analysis_error,
                "vps_compilation_fallback_enabled": VPS_COMPILATION_FALLBACK_ENABLED,
            }

    def _build_one(self) -> None:
        low, high = SEED_RANGES.get(self.prefix, (10_000_000, 99_999_999))
        seed = random.SystemRandom().randint(low, high)
        name = f"{self.prefix}_{seed:08d}"
        generated = ROOT / "maps" / "generated"
        map_path = generated / f"{name}.map"
        bsp_path = generated / f"{name}.bsp"
        json_path = generated / f"{name}.json"
        meta_path = generated / f"{name}.meta.json"
        lattice_path = generated / f"{name}.lattice.json"
        routes_path = generated / f"{name}.routes.json"
        q2tool = ROOT / "maps" / "q2tools" / "bin" / "q2tool"
        ready_styles = self.status()["ready_styles"]
        arena_ready = sum(style in ARENA_STYLES for style in ready_styles.values())
        style_pool = ARENA_STYLES if arena_ready < self.arena_target else LEGACY_STYLES
        style = random.SystemRandom().choice(style_pool)
        with self.lock:
            self.building = name
            self.last_error = ""
        temporary = None
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "maps" / "generator.py"),
                 "--seed", str(seed), "--name", name, "--style", style],
                cwd=ROOT, check=True, timeout=30,
            )
            subprocess.run(
                [str(q2tool), "-bsp", "-vis", "-fast", "-rad",
                 "-threads", str(self.threads),
                 "-moddir", str(self.q2_root / "baseq2"), str(map_path)],
                cwd=ROOT, check=True,
            )
            artifact_paths = {
                bsp_path.name: bsp_path,
                json_path.name: json_path,
                lattice_path.name: lattice_path,
                routes_path.name: routes_path,
            }
            missing = [
                filename for filename, path in artifact_paths.items()
                if not path.is_file()
            ]
            if missing or not meta_path.is_file():
                raise RuntimeError(
                    f"compiler did not produce required artifacts for {name}: {missing}"
                )
            generator_meta = json.loads(meta_path.read_text())
            payloads = {
                filename: path.read_bytes()
                for filename, path in artifact_paths.items()
            }
            manifest = build_manifest(name, generator_meta, payloads)
            manifest_payload = encode_manifest(manifest)
            temporary = self.queue_dir / f".{name}.{uuid.uuid4().hex}.tmp"
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
                for filename in artifact_names(name):
                    bundle.writestr(filename, payloads[filename])
                bundle.writestr("manifest.json", manifest_payload)
            # Make lattice priors and the route/item clock available to the
            # WSL network trainer before the public server can claim this map.
            install_bundle(
                self.install_dir, manifest, manifest_payload, payloads,
            )
            os.replace(temporary, self.queue_dir / f"{name}.zip")
            print(f"[farm] ready {name}", flush=True)
        except Exception as error:
            with self.lock:
                self.last_error = str(error)
            print(f"[farm] build failed for {name}: {error}", flush=True)
            time.sleep(5)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
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
    parser.add_argument("--arena_fraction", type=float, default=0.5)
    parser.add_argument("--queue_dir", default=str(ROOT / "maps" / "farm_queue"))
    parser.add_argument("--q2_root", default=os.environ.get("Q2_ROOT", str(ROOT.parent / "runtime")))
    parser.add_argument(
        "--install_dir", default="",
        help="persistent WSL bundle mirror (default: Q2_ROOT/baseq2/maps)",
    )
    args = parser.parse_args()

    farm = MapFarm(Path(args.queue_dir), Path(args.q2_root), args.queue_depth,
                   args.threads, args.prefix, args.arena_fraction,
                   Path(args.install_dir) if args.install_dir else None)
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
