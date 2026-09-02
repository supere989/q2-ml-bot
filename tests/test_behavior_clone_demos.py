import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "behavior_clone_demos.py"

OBS_DIM = 219  # real teacher schema (Q2_EXT_OBS=1)
GATE_KEYS = (
    "yaw_mae_deg",
    "pitch_mae_deg",
    "move_drift_mae",
    "fire_precision",
    "fire_recall",
    "hidden_fire_rate",
    "weapon_top1_accuracy",
)


def _write_corpus(batches_dir: Path, n_replicates: int = 4, ticks: int = 64) -> None:
    """Write npz batches in the teacher schema with a learnable mapping.

    Teacher labels are deterministic functions of the observation so a few
    epochs measurably reduce yaw MAE: yaw = 10*obs[0], pitch = 5*obs[3],
    fire = obs[1] > 0, plus constant weapon 7. Multiple files share (map,
    slot) keys, so the loader merges them into four longer episodes.
    """
    rng = np.random.default_rng(20260901)
    for rep in range(n_replicates):
        obs_rows, act_rows = [], []
        seq_rows, tick_rows, slot_rows, map_rows = [], [], [], []
        for map_name in ("q2dm2", "q2dm4"):
            for slot in (0, 1):
                obs = rng.uniform(-1.0, 1.0, size=(ticks, OBS_DIM)).astype(np.float32)
                actions = np.zeros((ticks, 8), dtype=np.float32)
                actions[:, 0] = 0.5 * obs[:, 2]
                actions[:, 1] = -0.25 * obs[:, 5]
                actions[:, 2] = np.clip(10.0 * obs[:, 0], -45.0, 45.0)
                actions[:, 3] = np.clip(5.0 * obs[:, 3], -30.0, 30.0)
                actions[:, 4] = (obs[:, 4] > 0.5).astype(np.float32)
                actions[:, 5] = (obs[:, 1] > 0.0).astype(np.float32)
                actions[:, 6] = 0.0
                actions[:, 7] = 7.0
                obs_rows.append(obs)
                act_rows.append(actions)
                seq_rows.append(np.arange(rep * 100_000, rep * 100_000 + ticks,
                                          dtype=np.uint32))
                tick_rows.append(np.arange(rep * 10_000, rep * 10_000 + ticks,
                                           dtype=np.uint32))
                slot_rows.append(np.full(ticks, slot, dtype=np.uint16))
                map_rows.append(np.full(ticks, map_name, dtype="<U32"))
        np.savez(
            batches_dir / f"batch_{rep:03d}.npz",
            obs=np.concatenate(obs_rows),
            actions=np.concatenate(act_rows),
            sequence=np.concatenate(seq_rows),
            ticks=np.concatenate(tick_rows),
            slots=np.concatenate(slot_rows),
            maps=np.concatenate(map_rows),
        )


def test_behavior_clone_demos_learns_synthetic_teacher(tmp_path):
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_corpus(batches_dir)
    output_dir = tmp_path / "out"

    # Run the CLI in a fresh interpreter: OBS_DIM is fixed at import time
    # from Q2_EXT_OBS, and earlier test modules in a full-suite run may
    # already have imported harness.protocol with the default (209).
    env = dict(os.environ)
    env["Q2_EXT_OBS"] = "1"
    proc = subprocess.run(
        [
            sys.executable, str(TOOL),
            "--batches_dir", str(batches_dir),
            "--output_dir", str(output_dir),
            "--epochs", "2",
            "--batch_seqs", "8",
            "--seq_len", "16",
            "--lr", "1e-2",
            "--holdout_frac", "0.25",
            "--device", "cpu",
            "--seed", "7142026",
        ],
        capture_output=True, text=True, timeout=300, cwd=ROOT, env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout

    assert (output_dir / "policy_bc_final.pt").is_file()
    metrics = json.loads((output_dir / "metrics.json").read_text())

    gates = metrics["gates"]
    for key in GATE_KEYS:
        assert key in gates, f"missing gate key {key}"
        assert math.isfinite(gates[key]), f"gate {key} not finite"

    train_eps = set(metrics["episodes"]["train"])
    holdout_eps = set(metrics["episodes"]["holdout"])
    assert train_eps and holdout_eps
    assert train_eps.isdisjoint(holdout_eps)

    history = metrics["history"]
    assert len(history) == 2
    assert history[-1]["yaw_mae_deg"] < history[0]["yaw_mae_deg"], (
        f"yaw MAE did not improve: {history[0]['yaw_mae_deg']:.3f} -> "
        f"{history[-1]['yaw_mae_deg']:.3f}"
    )
