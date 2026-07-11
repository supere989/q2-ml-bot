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
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool

HIDDEN_DIM = 256  # must match models.policy.HIDDEN_DIM

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
    args = parser.parse_args()

    policy = OnnxPolicy(Path(args.onnx))

    maps = discover_map_pool(
        map_name=args.map_name, map_glob=args.map_glob, map_dir=args.map_dir or None,
    ) if args.map_glob else [args.map_name]

    ml_slot = args.maxclients - args.num_ml_bots

    print(f"onnx={args.onnx}")
    print(f"maps={maps}")
    print(f"maxclients={args.maxclients}  ml_slot={ml_slot} (human joins any slot below this)")
    print(f"sv_port={27910 + args.port_offset}  server_id={args.server_id}")

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
    )

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    kills = deaths = 0
    try:
        obs = env.reset_all()[0]
        hx = policy.init_hidden()
        print("ML bot is live. Waiting for a human to join...", flush=True)
        last_report = time.monotonic()
        while not _STOP:
            action, _value, hx = policy.act(obs, hx)
            obs, _reward, term, trunc, info = env.step_all([action])[0]
            kills += int(info.get("kills", 0.0) > 0)
            deaths += int(info.get("deaths", 0.0) > 0)
            if term or trunc:
                obs = env.reset_slot(0)
                hx = policy.init_hidden()
            now = time.monotonic()
            if now - last_report >= 30.0:
                print(f"[live] bot kills={kills} deaths={deaths} map={info.get('map')}", flush=True)
                last_report = now
    finally:
        print(f"shutting down. final: kills={kills} deaths={deaths}")
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
