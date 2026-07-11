#!/usr/bin/env python3
"""ML bot vs ML bot: pit N instances of the same (or different) ONNX
checkpoint against each other, zero 3ZB2 AI opponents.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool

HIDDEN_DIM = 256


class OnnxPolicy:
    def __init__(self, onnx_path: Path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def init_hidden(self):
        h = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
        c = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
        return h, c

    def act(self, obs_vec, hx):
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, help="checkpoint for slot A (and B, unless --onnx_b given)")
    p.add_argument("--onnx_b", default="", help="optional different checkpoint for slot B")
    p.add_argument("--map_name", default="mltrain_00005207")
    p.add_argument("--map_glob", default="")
    p.add_argument("--server_id", type=int, default=50)
    p.add_argument("--port_offset", type=int, default=50)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--fraglimit", type=int, default=20)
    p.add_argument("--timelimit", type=float, default=10)
    p.add_argument(
        "--game_seed", type=int, default=-1,
        help="gameplay RNG seed; negative preserves normal game randomness",
    )
    args = p.parse_args()

    policy_a = OnnxPolicy(Path(args.onnx))
    policy_b = OnnxPolicy(Path(args.onnx_b)) if args.onnx_b else policy_a

    maps = discover_map_pool(map_name=args.map_name, map_glob=args.map_glob) if args.map_glob else [args.map_name]

    print(f"onnx_a={args.onnx}")
    print(f"onnx_b={args.onnx_b or args.onnx}")
    print(f"maps={maps}")

    env = Q2MultiEnv(
        server_id=args.server_id,
        map_name=maps[0],
        map_pool=maps,
        map_change_episodes=0,
        n_bots=4,
        num_ml_bots=2,
        port_offset=args.port_offset,
        maxclients=4,
        ml_slot=2,
        game_seed=None if args.game_seed < 0 else args.game_seed,
        max_ep_steps=10**9,
        timedemo=0,
        timescale=1.0,
        fraglimit=args.fraglimit,
        timelimit=args.timelimit,
    )

    kills = [0, 0]
    deaths = [0, 0]
    damage_dealt = [0.0, 0.0]

    try:
        obs = env.reset_all()
        hx = [policy_a.init_hidden(), policy_b.init_hidden()]
        policies = [policy_a, policy_b]
        print("both ML bots live. running...", flush=True)
        last_report = time.monotonic()
        for step in range(args.steps):
            actions = []
            for i in range(2):
                action, _v, hx[i] = policies[i].act(obs[i], hx[i])
                actions.append(action)
            results = env.step_all(actions)
            new_obs = [None, None]
            for i in range(2):
                o, _r, term, trunc, info = results[i]
                new_obs[i] = o
                kills[i] += int(info.get("kills", 0.0) > 0)
                deaths[i] += int(info.get("deaths", 0.0) > 0)
                damage_dealt[i] += float(info.get("damage_dealt", 0.0))
                if term or trunc:
                    new_obs[i] = env.reset_slot(i)
                    hx[i] = policies[i].init_hidden()
            obs = new_obs
            now = time.monotonic()
            if now - last_report >= 15.0:
                print(f"[{step}/{args.steps}] A: kills={kills[0]} deaths={deaths[0]} dmg={damage_dealt[0]:.0f}  "
                      f"B: kills={kills[1]} deaths={deaths[1]} dmg={damage_dealt[1]:.0f}", flush=True)
                last_report = now
    finally:
        env.close()

    print(f"FINAL A: kills={kills[0]} deaths={deaths[0]} damage_dealt={damage_dealt[0]:.0f}")
    print(f"FINAL B: kills={kills[1]} deaths={deaths[1]} damage_dealt={damage_dealt[1]:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
