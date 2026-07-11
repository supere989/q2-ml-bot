#!/usr/bin/env python3
"""map_cards.py — per-map learnability cards + generator grading system.

A map's worth as training ground isn't its static coherence (that's the judge,
tools/map_judge.py) — it's whether the bot actually LEARNS on it. This reads a
training run's TensorBoard per-map telemetry and writes a persistent "data card"
per map that accumulates across runs, so a generator-algorithm change can be
graded by how learnable its output turned out to be.

Card (maps/cards/<seed>.json) holds the map's identity + features + judge score
(if any) + one entry per run (final reward, improvement Δ, episode churn, K/D,
learnability) + the rolling learnability mean. Cards survive runs and are
version-controlled, so `grade` can compare generator versions over time.

Learnability blends three signals:
  improvement  — did training raise reward on this map? (the core signal:
                 a learnable map is one the bot gets BETTER at)
  competence   — the reward ceiling actually reached
  survival     — episode churn: fewer deaths per unit play = longer lives

Usage:
  python3 tools/map_cards.py ingest --runs 'runs/ppo_D_*' --run-tag D
  python3 tools/map_cards.py list
  python3 tools/map_cards.py grade            # grade generators from all cards
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "maps" / "generated"
CARDS = ROOT / "maps" / "cards"


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def learnability(final_reward, delta, episodes, total_steps):
    # improvement: reward rise early→late (the core "can it learn here")
    improvement = _clip01((delta / 150.0 + 1.0) / 2.0)   # -150→0, 0→.5, +150→1
    # competence: how close to break-even the reward ceiling got
    competence = _clip01(1.0 + final_reward / 300.0)     # -300→0, 0→1
    # survival: lower episode churn per step = longer lives = better
    if total_steps and episodes:
        steps_per_ep = total_steps / max(1, episodes)
        survival = _clip01(steps_per_ep / 120.0)          # ~120 steps/life ≈ good
    else:
        survival = 0.5
    score = 0.45 * improvement + 0.35 * competence + 0.20 * survival
    return round(score, 3), {"improvement": round(improvement, 3),
                             "competence": round(competence, 3),
                             "survival": round(survival, 3)}


def extract_map_metrics(event_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    total_steps = 0
    if "train/sps" in tags:
        total_steps = int(ea.Scalars("train/sps")[-1].step)
    kd = ([e.value for e in ea.Scalars("combat/kd_ratio")]
          if "combat/kd_ratio" in tags else [])
    kd_recent = float(np.mean(kd[-20:])) if kd else 0.0
    out = {}
    for t in tags:
        m = re.match(r"maps/(mltrain_\d+)/reward_mean", t)
        if not m:
            continue
        name = m.group(1)
        rv = [e.value for e in ea.Scalars(t)]
        ep_t = f"maps/{name}/episodes"
        episodes = int(np.sum([e.value for e in ea.Scalars(ep_t)])) if ep_t in tags else 0
        if len(rv) < 4:
            continue
        final_r = float(np.mean(rv[-30:]))
        delta = float(np.mean(rv[-30:]) - np.mean(rv[:30])) if len(rv) > 40 else 0.0
        score, parts = learnability(final_r, delta, episodes, total_steps)
        out[name] = {"final_reward": round(final_r, 1), "delta": round(delta, 1),
                     "episodes": episodes, "kd_recent": round(kd_recent, 3),
                     "learnability": score, "parts": parts}
    return out


def _features(name):
    p = GENERATED / f"{name}.meta.json"
    if not p.exists():
        return {}, None
    m = json.loads(p.read_text())
    feats = {k: m.get(k) for k in ("style", "rooms", "arenas", "towers",
             "lane_walls", "lava_pools", "platforms", "hook_required",
             "terrace_levels", "armor_total")}
    return feats, m.get("generator")


def _judge(name):
    p = GENERATED / f"{name}.judgment.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("coherence")
        except Exception:
            return None
    return None


def ingest(run_glob, run_tag, stamp):
    CARDS.mkdir(parents=True, exist_ok=True)
    dirs = sorted(glob.glob(str(ROOT / run_glob)))
    if not dirs:
        print(f"no run dirs match {run_glob}")
        return
    event_dir = dirs[-1]
    metrics = extract_map_metrics(event_dir)
    print(f"ingest {Path(event_dir).name}: {len(metrics)} maps")
    for name, mt in metrics.items():
        seed = name.split("_")[-1]
        card_path = CARDS / f"{seed}.json"
        if card_path.exists():
            card = json.loads(card_path.read_text())
        else:
            feats, gen = _features(name)
            card = {"map": name, "seed": int(seed), "generator": gen,
                    "features": feats, "judge_coherence": _judge(name),
                    "runs": []}
        # replace any prior entry for this run_tag (idempotent re-ingest)
        card["runs"] = [r for r in card["runs"] if r.get("run_tag") != run_tag]
        card["runs"].append({"run_tag": run_tag, "t": stamp, **mt})
        ls = [r["learnability"] for r in card["runs"]]
        card["learnability_mean"] = round(float(np.mean(ls)), 3)
        card["learnability_n"] = len(ls)
        card_path.write_text(json.dumps(card, indent=2) + "\n")
    print(f"updated {len(metrics)} cards in {CARDS}")


def _load_cards():
    return [json.loads(p.read_text()) for p in sorted(CARDS.glob("*.json"))]


def list_cards():
    cards = _load_cards()
    if not cards:
        print("no cards yet — run `ingest` first")
        return
    cards.sort(key=lambda c: -c.get("learnability_mean", 0))
    print(f"{'seed':>6} {'gen':>4} {'style':<8} {'learn':>6} {'n':>2} "
          f"{'judge':>6} {'lava':>4} {'tow':>3} {'reward':>8}")
    for c in cards:
        f = c.get("features", {})
        last = c["runs"][-1] if c["runs"] else {}
        print(f"{c['seed']:>6} {str(c.get('generator')):>4} "
              f"{str(f.get('style','?')):<8} {c.get('learnability_mean',0):>6.3f} "
              f"{c.get('learnability_n',0):>2} "
              f"{(c.get('judge_coherence') or 0):>6.2f} "
              f"{str(f.get('lava_pools','?')):>4} {str(f.get('towers','?')):>3} "
              f"{last.get('final_reward',0):>8.1f}")


def grade():
    """Aggregate cards into a per-generator-version grade."""
    cards = _load_cards()
    if not cards:
        print("no cards yet")
        return
    by_gen = {}
    for c in cards:
        by_gen.setdefault(c.get("generator") or "?", []).append(c)
    print(f"{'generator':>10} {'maps':>5} {'learn_mean':>10} {'learn_std':>9} "
          f"{'unlearnable':>11} {'judge_mean':>10}")
    for gen, cs in sorted(by_gen.items()):
        ls = [c.get("learnability_mean", 0) for c in cs]
        jd = [c["judge_coherence"] for c in cs if c.get("judge_coherence")]
        unlearnable = sum(1 for x in ls if x < 0.35)   # bot can't learn these
        # generator grade: high mean learnability, low spread, few unlearnable
        grade_val = (np.mean(ls) - 0.5 * np.std(ls)
                     - 0.3 * unlearnable / max(1, len(ls)))
        print(f"{gen:>10} {len(cs):>5} {np.mean(ls):>10.3f} {np.std(ls):>9.3f} "
              f"{unlearnable:>3}/{len(cs):<7} {(np.mean(jd) if jd else 0):>10.2f}"
              f"   GRADE={grade_val:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("--runs", required=True, help="glob of run dirs, e.g. 'runs/ppo_D_*'")
    pi.add_argument("--run-tag", required=True)
    pi.add_argument("--stamp", default="", help="timestamp label (caller supplies)")
    sub.add_parser("list")
    sub.add_parser("grade")
    args = p.parse_args()
    if args.cmd == "ingest":
        ingest(args.runs, args.run_tag, args.stamp)
    elif args.cmd == "list":
        list_cards()
    elif args.cmd == "grade":
        grade()


if __name__ == "__main__":
    main()
