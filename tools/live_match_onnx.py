#!/usr/bin/env python3
"""
Same as live_match.py, but inference runs on ONNX Runtime instead of
PyTorch/CUDA. No torch dependency at all -- built for resource-constrained
deployment (a small VPS with no GPU and little spare RAM), where a full
PyTorch+CUDA process (~2GB RSS observed) doesn't fit.

The .onnx file must be exported via models.policy.export_onnx() from the
exact checkpoint you want to serve (see README/tools for the exporter);
this script only runs inference, it doesn't need training code or torch.

Decode logic mirrors Q2BotPolicy.act(deterministic=True) exactly (verified
to match within float32 noise): cont_mean used directly for the 4 continuous
dims, argmax for jump/fire/hook/weapon. fire_logits from the export are
already conditioned on the argmax-selected weapon (autoregressive head).

Live map generation (--live_maps): instead of playing a fixed map or a
static pre-baked pool, consume fresh procedural maps. With --stock_maps,
production alternates stock and generated maps, leaving an entire stock-map
round for the remote farm to replenish its generated-map queue. Production
combines this with --map_farm_url so generation and radiosity run on the WSL
compute host; local compilation remains available for standalone use. With
use_mapqueue=0 (our config), EndDMLevel() falls through to the vanilla
sv_maplist cvar -- a cycling list where, if the current map is present,
the server advances to the next entry. So we keep a rolling 2-entry list
("current next"), and each time we detect the live server actually
transitioned (polled via the standard out-of-band `status` query -- the
harness's own bookkeeping doesn't see engine-driven transitions), we
generate a new next map and re-arm the list.

Lighting: training's compile.sh deliberately skips the -rad (radiosity)
pass -- the bot doesn't render pixels, so unlit/fullbright geometry is fine
and much faster to iterate on. For a human-facing live match that looks
badly overbright/fullbright, so this script runs its own compile with -rad
included. A full rad pass takes ~50-100s (vs <1s for bsp+vis alone) --
far too slow to share the live VPS without pacing impact. The WSL map farm
keeps a ready queue of fully compiled bundles; the VPS only downloads,
checksum-verifies, and atomically installs them.
"""

import argparse
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from tools.map_farm_client import (
    FarmMapGenerator,
    ShuffledStockRotation,
    query_live_mapname,
)

HIDDEN_DIM = 256  # must match models.policy.HIDDEN_DIM
LIVE_MAP_PREFIX = "mllive"

_STOP = False


def _handle_stop(_signum, _frame):
    global _STOP
    _STOP = True


class OnnxPolicy:
    """Minimal drop-in replacement for Q2BotPolicy.act(), ONNX-backed."""

    def __init__(self, onnx_path: Path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def init_hidden(self):
        h = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
        c = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
        return h, c

    def act(self, obs_vec: np.ndarray, hx):
        h_in, c_in = hx
        outs = self.sess.run(None, {
            "obs": obs_vec.reshape(1, 1, -1).astype(np.float32),
            "h_in": h_in, "c_in": c_in,
        })
        (cont_mean, _cont_log_std, jump_logits, fire_logits,
         hook_logits, weapon_logits, value, h_out, c_out) = outs
        cont = cont_mean.reshape(-1)
        action = np.array([
            cont[0], cont[1], cont[2], cont[3],
            float(np.argmax(jump_logits)),
            float(np.argmax(fire_logits)),
            float(np.argmax(hook_logits)),
            float(np.argmax(weapon_logits)),
        ], dtype=np.float32)
        return action, float(value.reshape(-1)[0]), (h_out, c_out)


class AsyncMapGenerator:
    """Generates+compiles (with lighting) a fresh procedural map without
    blocking the game loop. The bsp+vis+rad compile takes ~50-100s; start()
    right after a round begins so it's ready long before that round ends."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.q2_root = os.environ.get("Q2_ROOT", str(ROOT.parent / "q2_lithium_merge"))
        self._proc = None
        self._name = None

    def start(self):
        if self._proc is not None:
            return  # already have one in flight
        seed = self.rng.randint(10_000_000, 99_999_999)
        name = f"{LIVE_MAP_PREFIX}_{seed:08d}"
        gen = subprocess.run(
            [sys.executable, str(ROOT / "maps" / "generator.py"),
             "--seed", str(seed), "--name", name],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
        if gen.returncode != 0:
            print(f"[mapgen] generate failed: {gen.stderr[-500:]}", flush=True)
            return
        q2tool = ROOT / "maps" / "q2tools" / "bin" / "q2tool"
        map_path = ROOT / "maps" / "generated" / f"{name}.map"
        self._proc = subprocess.Popen(
            [str(q2tool), "-bsp", "-vis", "-fast", "-rad",
             "-moddir", str(Path(self.q2_root) / "baseq2"), str(map_path)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self._name = name
        print(f"[mapgen] started background compile: {name}", flush=True)

    def poll(self) -> str:
        """Returns the finished map name once ready, else None. Non-blocking."""
        if self._proc is None:
            return None
        rc = self._proc.poll()
        if rc is None:
            return None  # still running
        name, proc = self._name, self._proc
        self._proc = None
        self._name = None
        # q2tool writes the compiled .bsp next to the .map source, NOT into
        # the server's baseq2/maps/ -- that copy is normally compile.sh's job,
        # which we bypass by calling q2tool directly for the combined
        # bsp+vis+rad pass. Install it ourselves.
        src_bsp = ROOT / "maps" / "generated" / f"{name}.bsp"
        src_json = ROOT / "maps" / "generated" / f"{name}.json"
        if rc != 0 or not src_bsp.exists():
            out = proc.stdout.read()[-800:] if proc.stdout else ""
            print(f"[mapgen] compile failed for {name} (rc={rc}): {out}", flush=True)
            return None
        install_dir = Path(self.q2_root) / "baseq2" / "maps"
        install_dir.mkdir(parents=True, exist_ok=True)
        (install_dir / f"{name}.bsp").write_bytes(src_bsp.read_bytes())
        if src_json.exists():
            (install_dir / f"{name}.json").write_bytes(src_json.read_bytes())
        print(f"[mapgen] fresh lit map ready: {name}", flush=True)
        return name

    @property
    def busy(self) -> bool:
        return self._proc is not None


def generate_fresh_map_blocking(mapgen) -> str:
    """Synchronous variant -- only used once, for the very first map at
    startup (no round is in progress yet to be blocked)."""
    mapgen.start()
    if not mapgen.busy:
        return None
    while mapgen.busy:
        name = mapgen.poll()
        if name:
            return name
        time.sleep(0.1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="exported policy .onnx path")
    parser.add_argument("--map_name", default="mltrain_00005207")
    parser.add_argument("--map_glob", default="")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--server_id", type=int, default=90)
    parser.add_argument("--port_offset", type=int, default=90)
    parser.add_argument("--maxclients", type=int, default=4)
    parser.add_argument("--num_ml_bots", type=int, default=1)
    parser.add_argument("--fraglimit", type=int, default=20)
    parser.add_argument("--timelimit", type=float, default=15)
    parser.add_argument("--live_maps", action="store_true",
                         help="generate a fresh procedural map every round instead of a fixed/pooled map")
    parser.add_argument("--map_farm_url", default="",
                         help="private map-farm base URL; consume compiled maps instead of compiling locally")
    parser.add_argument("--stock_maps", default="",
                         help="comma/space separated stock maps to alternate with generated maps")
    parser.add_argument("--rotation_seed", type=int, default=20260712,
                         help="independent shuffle stream for the stock rotation")
    parser.add_argument("--dlserver", default="",
                         help="HTTP URL serving the game data root (sv_downloadserver) -- "
                              "fast client downloads instead of the legacy in-game transfer")
    args = parser.parse_args()

    policy = OnnxPolicy(Path(args.onnx))
    rng = random.Random()
    mapgen = (
        FarmMapGenerator(args.map_farm_url)
        if args.live_maps and args.map_farm_url
        else AsyncMapGenerator(rng) if args.live_maps else None
    )
    stock_names = args.stock_maps.replace(",", " ").split()
    stock_rotation = (
        ShuffledStockRotation(stock_names, args.rotation_seed) if stock_names else None
    )

    if args.live_maps:
        # Interlaced mode deliberately starts on stock while the first fresh
        # map is fetched. It never substitutes --map_name into that rotation.
        if stock_rotation:
            start_map = stock_rotation.next()
        else:
            start_map = generate_fresh_map_blocking(mapgen)
            if not start_map:
                print("[mapgen] initial generation failed, falling back to --map_name", flush=True)
                start_map = args.map_name
        maps = [start_map]
    else:
        maps = discover_map_pool(
            map_name=args.map_name, map_glob=args.map_glob, map_dir=args.map_dir or None,
        ) if args.map_glob else [args.map_name]

    ml_slot = args.maxclients - args.num_ml_bots
    sv_port = 27910 + args.port_offset

    print(f"onnx={args.onnx}")
    print(f"maps={maps}  live_maps={args.live_maps}")
    print(f"maxclients={args.maxclients}  ml_slot={ml_slot} (human joins any slot below this)")
    print(f"sv_port={sv_port}  server_id={args.server_id}")

    env = Q2MultiEnv(
        server_id=args.server_id,
        map_name=maps[0],
        map_pool=maps,
        map_change_episodes=0,
        n_bots=args.num_ml_bots,
        num_ml_bots=args.num_ml_bots,
        port_offset=args.port_offset,
        maxclients=args.maxclients,
        ml_slot=ml_slot,
        max_ep_steps=10**9,
        timedemo=0,
        timescale=1.0,
        fraglimit=args.fraglimit,
        timelimit=args.timelimit,
        console_pipe=args.live_maps or bool(args.dlserver),
    )

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    kills = deaths = 0
    current_map = maps[0]
    next_map = None  # map currently armed in sv_maplist
    staged_generated = None
    try:
        obs = env.reset_all()[0]
        hx = policy.init_hidden()

        if args.dlserver:
            env.set_cvar("sv_downloadserver", args.dlserver)
            print(f"[mapgen] sv_downloadserver = {args.dlserver}", flush=True)

        if args.live_maps:
            mapgen.start()  # begin generating the NEXT map in the background now

        print("ML bot is live. Waiting for a human to join...", flush=True)
        last_report = time.monotonic()
        last_map_check = time.monotonic()
        while not _STOP:
            action, _value, hx = policy.act(obs, hx)
            obs, _reward, term, trunc, info = env.step_all([action])[0]
            kills += int(info.get("kills", 0.0) > 0)
            deaths += int(info.get("deaths", 0.0) > 0)
            if term or trunc:
                obs = env.reset_slot(0)
                hx = policy.init_hidden()

            now = time.monotonic()

            if args.live_maps:
                # 1. Stage a finished farm map. In interlaced mode it is only
                # armed after a stock round; generated rounds arm stock next.
                if mapgen.busy:
                    finished = mapgen.poll()
                    if finished:
                        staged_generated = finished
                elif staged_generated is None:
                    mapgen.start()

                if next_map is None:
                    if stock_rotation and current_map.startswith(f"{LIVE_MAP_PREFIX}_"):
                        next_map = stock_rotation.next()
                    elif staged_generated is not None:
                        next_map = staged_generated
                        staged_generated = None
                    if next_map:
                        env.set_cvar("sv_maplist", f"{current_map} {next_map}")
                        print(f"[mapgen] armed sv_maplist: {current_map} -> {next_map}", flush=True)

                # 2. did the round actually transition? (poll infrequently --
                #    it's just a UDP status query, but no need to hammer it)
                if now - last_map_check >= 5.0:
                    last_map_check = now
                    live_map = query_live_mapname(sv_port)
                    if live_map and live_map != current_map:
                        # engine-driven transition happened (to next_map if it
                        # was ready in time, else it looped back to current_map
                        # per EndDMLevel's fallback -- either way, sync up and
                        # start generating the one after)
                        current_map = live_map
                        next_map = None
                        if staged_generated is None and not mapgen.busy:
                            mapgen.start()
                        print(f"[mapgen] round advanced to {current_map}; "
                              f"generating the next one now", flush=True)

            if now - last_report >= 30.0:
                print(f"[live] bot kills={kills} deaths={deaths} map={info.get('map')}", flush=True)
                last_report = now
    finally:
        print(f"shutting down. final: kills={kills} deaths={deaths}")
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
