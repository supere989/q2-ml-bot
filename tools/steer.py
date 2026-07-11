#!/usr/bin/env python3
"""
steer.py — push LLM-coach directives into the live training run.

Directives compile to lattice deposits in every bot's session memory; the
existing memory gradients (death aversion, opportunity pull, engagement pull)
change behavior without retraining. The trainer polls directives.json once
per PPO update (~near-realtime).

Actions:
  avoid   "DO NOT GO THERE"      -> death-aversion repulsion at (x,y,z)
  seek    "value/target is here" -> opportunity pull
  engage  "expect combat here"   -> engagement pull
  danger  "risky, not lethal"    -> threat shading

Usage:
  python3 tools/steer.py avoid 1280 1400 96 --map mltrain_00005103 \
      --strength 4 --note "camper death zone"
  python3 tools/steer.py --clear
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRECTIVES = ROOT / "directives.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?",
                   choices=["avoid", "seek", "engage", "danger"])
    p.add_argument("x", nargs="?", type=float)
    p.add_argument("y", nargs="?", type=float)
    p.add_argument("z", nargs="?", type=float)
    p.add_argument("--map", default="", help="map name ('' = bot's current map)")
    p.add_argument("--strength", type=float, default=3.0)
    p.add_argument("--note", default="")
    p.add_argument("--server", type=int, default=None,
                   help="address one server's channel (directives/server_N.json)")
    p.add_argument("--clear", action="store_true", help="drop pending directives")
    args = p.parse_args()

    target = DIRECTIVES
    if args.server is not None:
        target = ROOT / "directives" / f"server_{args.server}.json"
        target.parent.mkdir(exist_ok=True)
    data = {"seq": 0, "directives": []}
    if target.exists():
        try:
            data = json.loads(target.read_text())
        except Exception:
            pass

    if args.clear:
        data = {"seq": int(data.get("seq", 0)) + 1, "directives": []}
    else:
        if not args.action or args.x is None:
            p.error("need: action x y z (or --clear)")
        data["seq"] = int(data.get("seq", 0)) + 1
        data.setdefault("directives", []).append({
            "action": args.action, "x": args.x, "y": args.y, "z": args.z,
            "map": args.map, "strength": args.strength, "note": args.note,
        })

    target.write_text(json.dumps(data, indent=1) + "\n")
    print(f"seq={data['seq']} pending={len(data['directives'])} → {target}")


if __name__ == "__main__":
    main()
