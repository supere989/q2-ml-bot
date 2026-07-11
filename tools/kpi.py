"""
kpi.py — read TensorBoard event files and compute experiment KPIs.

Used by karpathy_loop.py to evaluate experiments. We read the most recent
events from a run, compute summary stats over the trailing portion of the
run, and return both raw metrics and a composite score.
"""
import math
from pathlib import Path
from typing import Dict, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRIC_TAGS = [
    "episode/reward_mean",
    "episode/length_mean",
    "episode/count",
    "train/loss",
    "train/sps",
    "train/value_mean",
    "train/return_mean",
]


def read_run_metrics(run_dir: Path, last_frac: float = 0.3) -> Dict[str, float]:
    """
    Load scalar metrics from a TensorBoard run, average over the last `last_frac`.

    Returns a dict {tag: mean_value}, omitting tags with no events.
    """
    ea = EventAccumulator(str(run_dir),
                          size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    out: Dict[str, float] = {}
    for tag in METRIC_TAGS:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        n = len(events)
        tail_start = int(n * (1.0 - last_frac))
        tail = events[tail_start:] or events
        values = [e.value for e in tail
                  if e.value is not None and not math.isnan(e.value)]
        if values:
            out[tag] = sum(values) / len(values)
    out["_n_updates"] = float(n) if METRIC_TAGS[0] in available else 0.0
    return out


def composite_score(m: Dict[str, float]) -> float:
    """
    Composite score for ranking experiments. Higher is better.

    reward_mean dominates once it's flowing. length_mean kicks in as a
    backstop when reward is near zero — surviving longer is weakly good.
    Loss is penalized only if it explodes (>10) — otherwise ignored.
    """
    reward = m.get("episode/reward_mean", 0.0)
    if math.isnan(reward):
        reward = 0.0

    length_norm = m.get("episode/length_mean", 0.0) / 1000.0   # ~unit scale

    loss = m.get("train/loss", 0.0)
    loss_penalty = max(0.0, loss - 10.0) * 0.01                # only if exploding

    return reward + 0.3 * length_norm - loss_penalty


def latest_run_dir(runs_dir: Path) -> Optional[Path]:
    runs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_")]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    import sys, json
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if p is None:
        p = latest_run_dir(Path("runs"))
    if p is None:
        print("No runs found"); sys.exit(1)
    print(f"Reading {p}")
    m = read_run_metrics(p)
    print(json.dumps(m, indent=2))
    print(f"composite: {composite_score(m):.4f}")
