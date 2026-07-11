#!/usr/bin/env python3
"""
karpathy_loop.py — automated time-budgeted ML iteration loop.

Each iteration:
  1. Pick a hypothesis from tools/hypotheses.py
  2. Start training with that hypothesis's overrides
  3. Wait BUDGET seconds
  4. Stop training, read KPI from TensorBoard events
  5. Compare to baseline composite score
  6. If better → commit (becomes new baseline)
     If worse  → revert (next hypothesis tries a different angle)
  7. Log every iteration to runs/loop_log.jsonl

Usage:
    python3 tools/karpathy_loop.py                # 5-min budget, all phase-1
    python3 tools/karpathy_loop.py --budget 300   # 5 minutes
    python3 tools/karpathy_loop.py --phase 1 2    # multiple phases
    python3 tools/karpathy_loop.py --resume       # continue from last log entry

Run under nohup or screen — survives SSH disconnects:
    nohup python3 tools/karpathy_loop.py > /tmp/loop.log 2>&1 &
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.kpi import (
    read_run_metrics, composite_score, latest_run_dir
)
from tools.hypotheses import HYPOTHESES, by_phase


LOG_FILE = PROJECT_ROOT / "runs" / "loop_log.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs"
WARMUP_SECONDS = 90       # let q2ded spawn and PPO collect first rollouts
COOLDOWN_SECONDS = 3      # let processes die cleanly


# ── process management ──────────────────────────────────────────────────────

def _kill_train():
    """Hard-kill any running training stack."""
    for pat in ("train\\.ppo", "q2ded"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(COOLDOWN_SECONDS)


# Base configuration applied to every experiment. Hypotheses override these.
LOOP_DEFAULTS = {
    "n_servers":         4,
    "n_bots_per_server": 4,   # 1 ML + 3 AI opponents
    "max_ep_steps":      5000,
    "map_glob":          "mltrain_*.bsp",
    "map_change_episodes": 1,
}

def _start_train(overrides: dict, log_path: Path, env_vars: dict = None) -> subprocess.Popen:
    """Launch train.ppo with LOOP_DEFAULTS merged with hypothesis overrides.

    env_vars: optional dict of additional environment variables (e.g., reward weights).
    """
    merged = {**LOOP_DEFAULTS, **overrides}
    args = ["python3", "-u", "-m", "train.ppo"]
    for k, v in merged.items():
        args += [f"--{k}", str(v)]

    env = os.environ.copy()
    if env_vars:
        for k, v in env_vars.items():
            env[k] = str(v)
    # NVIDIA WSL CUDA needs nvidia-smi/libcuda on path
    if Path("/usr/lib/wsl/lib").is_dir():
        env["PATH"] = "/usr/lib/wsl/lib:" + env.get("PATH", "")
    # AMD ROCm guard — harmless to set if no AMD GPU
    if Path("/sys/class/drm/card1/device/power_dpm_force_performance_level").exists():
        env["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        args, cwd=str(PROJECT_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT, env=env,
    )
    return proc


def _wait_for_first_update(log_path: Path, max_wait: int) -> bool:
    """Return True once PPO has emitted its first 'steps=' line."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            if log_path.exists() and "steps=" in log_path.read_text():
                return True
        except OSError:
            pass
        time.sleep(2)
    return False


# ── experiment runner ──────────────────────────────────────────────────────

def run_experiment(hyp: dict, budget_sec: int, verbose: bool = True) -> dict:
    """
    Run one experiment for budget_sec seconds with hyp['overrides'] and
    hyp['env_vars'] applied. Returns a dict with metrics + composite score + meta.
    """
    name = hyp["name"]
    overrides = hyp.get("overrides", {})
    env_vars  = hyp.get("env_vars", {})
    started_at = time.time()
    log_path = Path(f"/tmp/loop_{name}_{int(started_at)}.log")

    _kill_train()
    if verbose:
        extras = f" env={env_vars}" if env_vars else ""
        print(f"  ▶ launching '{name}' overrides={overrides}{extras}")
    proc = _start_train(overrides, log_path, env_vars=env_vars)

    # Wait for actual training to start; counts against the budget
    if not _wait_for_first_update(log_path, max_wait=WARMUP_SECONDS + 30):
        _kill_train()
        return {
            "name": name, "ok": False, "reason": "no PPO output within warmup window",
            "log": str(log_path), "duration": time.time() - started_at,
        }

    if verbose:
        print(f"    warmup done at +{time.time()-started_at:.0f}s, running for {budget_sec}s...")

    time.sleep(budget_sec)
    _kill_train()

    # Read KPI from latest run directory
    run = latest_run_dir(RUNS_DIR)
    if run is None:
        return {"name": name, "ok": False, "reason": "no run dir produced"}

    metrics = read_run_metrics(run, last_frac=0.5)
    score = composite_score(metrics)

    return {
        "name":       name,
        "ok":         True,
        "hypothesis": hyp.get("hypothesis", ""),
        "overrides":  overrides,
        "metrics":    metrics,
        "score":      score,
        "run_dir":    str(run.relative_to(PROJECT_ROOT)),
        "log":        str(log_path),
        "duration":   time.time() - started_at,
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
    }


# ── decision rule ──────────────────────────────────────────────────────────

def better(candidate: float, baseline: float, tolerance: float = 0.02) -> bool:
    """A candidate must beat baseline by at least `tolerance` to win."""
    return candidate > baseline + tolerance


# ── persistence ────────────────────────────────────────────────────────────

def append_log(record: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_log() -> list:
    if not LOG_FILE.exists():
        return []
    out = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except Exception: pass
    return out


# ── main loop ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=300,
                    help="seconds per experiment (default 300 = 5 min)")
    ap.add_argument("--phase",  type=int, nargs="+", default=[1],
                    help="which hypothesis phase(s) to test (default: 1)")
    ap.add_argument("--limit",  type=int, default=None,
                    help="max number of experiments to run")
    ap.add_argument("--names",  nargs="+", default=None,
                    help="only run specific hypothesis names")
    args = ap.parse_args()

    # Pick which hypotheses to run
    if args.names:
        from tools.hypotheses import by_name
        candidates = [by_name(n) for n in args.names if by_name(n)]
    else:
        candidates = []
        for p in args.phase:
            candidates.extend(by_phase(p))

    if args.limit:
        candidates = candidates[: args.limit]

    print(f"╔════════════════════════════════════════════════════════════╗")
    print(f"║  Karpathy Loop — {len(candidates)} experiments × {args.budget}s ≈ {(len(candidates)*args.budget)//60} min  ║")
    print(f"╚════════════════════════════════════════════════════════════╝")

    # Establish baseline from the first hypothesis (assumed to be 'baseline')
    baseline_score = float("-inf")
    baseline_name = None
    committed = []

    for i, hyp in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] {hyp['name']}: {hyp.get('hypothesis','')}")
        result = run_experiment(hyp, args.budget)
        if not result.get("ok"):
            print(f"    ✗ failed: {result.get('reason')}")
            result["decision"] = "failed"
            append_log(result)
            continue

        score = result["score"]
        rwd = result["metrics"].get("episode/reward_mean", 0)
        sps = result["metrics"].get("train/sps", 0)
        n_ep = result["metrics"].get("episode/count", 0)
        print(f"    score={score:+.4f}  reward={rwd:+.4f}  sps={sps:.0f}  episodes={n_ep:.0f}")

        if baseline_name is None:
            baseline_score = score
            baseline_name = hyp["name"]
            result["decision"] = "baseline"
            print(f"    ★ baseline established: score={score:+.4f}")
        elif better(score, baseline_score):
            old = baseline_score
            baseline_score = score
            baseline_name = hyp["name"]
            committed.append(hyp["name"])
            result["decision"] = "kept"
            print(f"    ✓ KEPT (improved {old:+.4f} → {score:+.4f})")
        else:
            result["decision"] = "rejected"
            print(f"    ✗ rejected (score {score:+.4f} ≤ baseline {baseline_score:+.4f})")

        append_log(result)

    # Final summary
    print(f"\n╔════════════════════════════════════════════════════════════╗")
    print(f"║ SUMMARY                                                    ║")
    print(f"╠════════════════════════════════════════════════════════════╣")
    if baseline_name is None:
        print("║ no successful experiments — check /tmp/loop_*.log          ║")
        print("╚════════════════════════════════════════════════════════════╝")
        return
    print(f"║ best: {baseline_name:<20s}  score={baseline_score:+.4f}            ║")
    print(f"║ kept improvements: {len(committed)}                                 ║")
    for c in committed:
        print(f"║   • {c}")
    print(f"╚════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
