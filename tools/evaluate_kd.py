#!/usr/bin/env python3
"""
Evaluate a Q2BotPolicy checkpoint against three Lithium/3ZB2 opponents.

The target for the current project goal is a 15:1 kill/death ratio with one ML
bot in a four-player generated-map setup.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from models.policy import Q2BotPolicy


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def _latest_checkpoint() -> Path:
    candidates = sorted((ROOT / "checkpoints").glob("policy_[0-9]*.pt"))
    if not candidates:
        raise FileNotFoundError("no checkpoints/policy_*.pt found")
    return candidates[-1]


def _safe_ratio(kills: float, deaths: float) -> float:
    return kills / max(deaths, 1.0)


def evaluate_map(
    policy: Q2BotPolicy,
    device: torch.device,
    map_name: str,
    steps: int,
    n_bots: int,
    server_id: int,
    port_offset: int,
    maxclients: int,
    ml_slot: int,
    deterministic: bool,
) -> Dict[str, object]:
    env = Q2MultiEnv(
        server_id=server_id,
        map_name=map_name,
        map_pool=[map_name],
        n_bots=n_bots,
        port_offset=port_offset,
        maxclients=maxclients,
        ml_slot=ml_slot,
        max_ep_steps=max(steps + 5, 100),
    )
    kills = 0.0
    deaths = 0.0
    damage_dealt = 0.0
    damage_taken = 0.0
    items = 0.0
    timeouts = 0
    episodes = 0
    start = time.time()

    try:
        obs = env.reset_all()[0]
        hx = policy.init_hidden(1, device)
        for _ in range(steps):
            action, _value, _logp, hx = policy.act(
                obs, hx, device=device, deterministic=deterministic
            )
            obs, _reward, term, trunc, info = env.step_all([action])[0]
            kills += float(info.get("kills", 0.0))
            deaths += float(info.get("deaths", 0.0))
            damage_dealt += float(info.get("damage_dealt", 0.0))
            damage_taken += float(info.get("damage_taken", 0.0))
            items += float(info.get("items", 0.0))
            timeouts += int(bool(info.get("timeout", False)))

            if term or trunc:
                episodes += 1
                obs = env.reset_slot(0)
                hx = policy.init_hidden(1, device)
    finally:
        env.close()

    return {
        "map": map_name,
        "steps": steps,
        "kills": kills,
        "deaths": deaths,
        "kd_ratio": round(_safe_ratio(kills, deaths), 4),
        "damage_dealt": round(damage_dealt, 2),
        "damage_taken": round(damage_taken, 2),
        "items": items,
        "episodes": episodes,
        "timeouts": timeouts,
        "seconds": round(time.time() - start, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="policy .pt checkpoint")
    parser.add_argument("--map_name", default="q2dm1")
    parser.add_argument("--map_glob", default="mltrain_*.bsp")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--max_maps", type=int, default=0)
    parser.add_argument("--n_bots", type=int, default=4)
    parser.add_argument(
        "--server_id",
        type=int,
        default=0,
        help="server id for eval RNG/slot defaults",
    )
    parser.add_argument(
        "--port_offset",
        type=int,
        default=20,
        help="q2ded port offset; default uses 27930 to avoid live training",
    )
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=11)
    parser.add_argument("--target_kd", type=float, default=15.0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    device = _pick_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    maps = discover_map_pool(
        map_name=args.map_name,
        map_glob=args.map_glob,
        map_dir=args.map_dir or None,
    )
    if args.max_maps > 0:
        maps = maps[:args.max_maps]

    print(f"checkpoint={ckpt}")
    print(f"device={device}")
    print(f"maps={maps}")
    print(f"setup=1 ML bot + {args.n_bots - 1} opponents")
    print(f"eval_server_id={args.server_id} q2_port={27910 + args.port_offset} "
          f"ml_slot={args.ml_slot}")

    rows: List[Dict[str, object]] = []
    for map_name in maps:
        row = evaluate_map(
            policy=policy,
            device=device,
            map_name=map_name,
            steps=args.steps,
            n_bots=args.n_bots,
            server_id=args.server_id,
            port_offset=args.port_offset,
            maxclients=args.maxclients,
            ml_slot=args.ml_slot,
            deterministic=not args.stochastic,
        )
        rows.append(row)
        print(json.dumps(row))

    total_kills = sum(float(row["kills"]) for row in rows)
    total_deaths = sum(float(row["deaths"]) for row in rows)
    kd_ratio = _safe_ratio(total_kills, total_deaths)
    summary = {
        "checkpoint": str(ckpt),
        "maps": len(rows),
        "steps_per_map": args.steps,
        "total_kills": total_kills,
        "total_deaths": total_deaths,
        "kd_ratio": round(kd_ratio, 4),
        "target_kd": args.target_kd,
        "target_met": kd_ratio >= args.target_kd,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
