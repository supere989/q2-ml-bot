#!/usr/bin/env python3
"""
map_judge.py — Claude-driven map coherence judge.

The "by Claude-driven influence" arm of the generator's sanity check: reads a
generated map's .meta.json + .lattice.json (and observed-heat data when the
map has been played), computes an item-economy dossier, and asks Claude to
judge competitive coherence. Writes <map>.judgment.json next to the sidecars.

The judgment carries template_adjustments — deltas to the heat-engine pull
weights — so design critique feeds directly back into generation parameters.

Usage:
    python3 tools/map_judge.py mltrain_00005100              # one map
    python3 tools/map_judge.py --glob "mltrain_000051*"      # batch
    python3 tools/map_judge.py --glob "mlv2*" --model claude-haiku-4-5
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "maps" / "generated"
HEAT_DIR = ROOT / "observed_heat"

MODEL_DEFAULT = "claude-opus-4-8"

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {"type": "number",
                      "description": "0..1 overall competitive coherence"},
        "verdict": {"type": "string", "enum": ["pass", "revise", "regenerate"]},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string",
                                 "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["severity", "description", "suggestion"],
                "additionalProperties": False,
            },
        },
        "template_adjustments": {
            "type": "array",
            "description": "Deltas to heat-engine pull weights",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "channel": {"type": "string"},
                    "delta": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["item", "channel", "delta", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["coherence", "verdict", "strengths", "issues",
                 "template_adjustments"],
    "additionalProperties": False,
}

SYSTEM = """You are a competitive arena-FPS map design judge for a Quake 2 \
deathmatch training-map generator. Maps are procedurally generated: items are \
placed by a multi-channel heat-field engine where each item radiates heat and \
carries attraction/repulsion templates (armour seeks weapon heat so counters \
live near threats; ammo trails its weapon; supplies avoid spawns).

Judge the map dossier for competitive coherence:
- Item economy: are power weapons contested (not spawn-adjacent)? Do counters \
(armour) answer threats at the right distance — reachable but not co-located? \
Is ammo paired with its weapons? Are supplies distributed, not clumped?
- Spawn fairness: does each spawn have comparable access to a usable weapon \
and escape options? No spawn should be in a power weapon's control zone.
- Risk/reward: high value should carry exposure or hazard cost (towers, lava \
rims). Free value is incoherent.
- Flow: engagement variety (verticality, chokepoints via lane walls, hook \
routes) without dead zones.
- When observed play data is present, weigh it heavily: intent (where the \
generator put value) should correlate with reality (where fights happened). \
Dead zones with loot, or fights clustering away from the economy, indicate \
dysfunctional displacement.

Be specific and quantitative; cite the dossier numbers. template_adjustments \
should be small deltas (±0.1..±0.5) to named item/channel pull weights from \
the dossier's template list, only where the evidence supports them."""


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def build_dossier(name: str) -> dict:
    meta = json.loads((GENERATED / f"{name}.meta.json").read_text())
    lattice = json.loads((GENERATED / f"{name}.lattice.json").read_text())
    items = lattice.get("items", [])
    objectives = lattice.get("objectives", [])
    danger = lattice.get("danger", [])

    by_class: dict = {}
    for it in items:
        by_class.setdefault(it["class"], []).append((it["x"], it["y"], it["z"]))

    power = [p for c in ("weapon_railgun", "weapon_rocketlauncher")
             for p in by_class.get(c, [])]
    armor = [p for c in ("item_armor_body", "item_armor_combat")
             for p in by_class.get(c, [])]
    weapons = [p for c, pts in by_class.items()
               if c.startswith("weapon_") for p in pts]

    # Counter-play distances: every power weapon's nearest armour
    counter = [round(min(_dist(w, a) for a in armor))
               for w in power if armor]
    # Power-weapon spread
    spread = [round(_dist(p1, p2))
              for i, p1 in enumerate(power) for p2 in power[i + 1:]]
    # Spawn fairness: each spawn's nearest weapon, and distance into the
    # nearest power weapon's control zone
    spawns = [(s["x"], s["y"], s["z"]) for s in lattice.get("spawns", [])]
    spawn_fairness = [{
        "nearest_weapon": round(min(_dist(s, w) for w in weapons)),
        "nearest_power": round(min(_dist(s, p) for p in power)),
    } for s in spawns if weapons and power]

    dossier = {
        "meta": {k: meta[k] for k in (
            "style", "rooms", "arenas", "terrace_levels", "max_elevation",
            "stairs", "towers", "lane_walls", "lava_pools", "lava_area",
            "spawns", "platforms", "hook_zones", "hook_required",
            "heat_placed_items", "structure_loot_sites", "armor_total",
            "relax_moves") if k in meta},
        "item_counts": {c: len(v) for c, v in sorted(by_class.items())},
        "power_weapon_count": len(power),
        "power_to_nearest_armor_dist": counter,
        "power_weapon_spread": spread,
        "spawn_fairness": spawn_fairness,
        "objectives": objectives,
        "danger_volumes": len(danger),
        "world_scale_note": "distances in game units; 384=melee range, "
                            "900=mid-range duel, 2500=cross-map",
    }

    heat_file = HEAT_DIR / f"{name}.heat.json"
    if heat_file.exists():
        heat = json.loads(heat_file.read_text())
        deps = heat.get("deposits", [])
        dossier["observed_play"] = {
            "env_steps": heat.get("env_steps"),
            "deposit_count": len(deps),
            "by_channel": {
                ch: len([d for d in deps if d["channel"] == ch])
                for ch in ("weapon", "danger", "objective")
            },
            "hottest": sorted(deps, key=lambda d: -d["amount"])[:8],
        }
    return dossier


def resolve_client():
    import os
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        key_file = Path.home() / "Claude-API.key"
        if key_file.exists():
            os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
    # Falls back to ANTHROPIC_AUTH_TOKEN / `ant auth login` profile if no key
    return anthropic.Anthropic()


def judge_map(client, name: str, model: str) -> dict:
    dossier = build_dossier(name)
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        output_config={"format": {"type": "json_schema",
                                  "schema": JUDGMENT_SCHEMA}},
        messages=[{
            "role": "user",
            "content": ("Judge this generated map's competitive coherence.\n\n"
                        + json.dumps(dossier, indent=1)),
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"model refused judging {name}")
    text = next(b.text for b in response.content if b.type == "text")
    judgment = json.loads(text)
    judgment["_model"] = model
    judgment["_usage"] = {"in": response.usage.input_tokens,
                          "out": response.usage.output_tokens}
    return judgment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map", nargs="?", help="map name (without extension)")
    p.add_argument("--glob", default=None, help="judge all maps matching glob")
    p.add_argument("--model", default=MODEL_DEFAULT)
    args = p.parse_args()

    if args.glob:
        names = sorted(f.name.replace(".meta.json", "")
                       for f in GENERATED.glob(f"{args.glob}.meta.json"))
    elif args.map:
        names = [args.map]
    else:
        p.error("give a map name or --glob")

    client = resolve_client()
    for name in names:
        try:
            j = judge_map(client, name, args.model)
        except FileNotFoundError as e:
            print(f"{name}: missing sidecar ({e})")
            continue
        out = GENERATED / f"{name}.judgment.json"
        out.write_text(json.dumps(j, indent=2) + "\n")
        worst = max((i["severity"] for i in j["issues"]),
                    key=["low", "medium", "high"].index, default="none")
        print(f"{name}: coherence={j['coherence']:.2f} verdict={j['verdict']} "
              f"issues={len(j['issues'])} (worst: {worst}) "
              f"adjustments={len(j['template_adjustments'])} "
              f"[{j['_usage']['in']}+{j['_usage']['out']} tok]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
