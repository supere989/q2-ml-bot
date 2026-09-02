#!/usr/bin/env python3
"""Behavior-clone the policy from recorded 3ZB2 teacher demos.

The corpus is a directory of ``*.npz`` batches written by the teacher
receiver (``~/q2-rollout/live-3zb2/teacher_batches`` on the training box).
Rows are grouped into episodes by (map, slot), sorted by (tick, sequence),
and chunked into fixed-length segments with a zeroed LSTM state per segment,
matching the PPO trainer's recurrent chunks.

Reference gates from the accepted bc_live_v2 clone (report header only, not
enforced here): yaw MAE 2.23 deg, pitch MAE 1.69 deg, movement drift 0.0041,
aligned-fire precision 92.7%, hidden-fire zero.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# OBS_DIM is fixed at import time from Q2_EXT_OBS (harness/protocol.py);
# teacher demos are recorded with the extended observation block (219 dims).
os.environ["Q2_EXT_OBS"] = "1"

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.policy import OBS_DIM, Q2BotPolicy

ACTION_DIM = 8  # [fwd, right, yaw, pitch, jump, fire, hook, weapon]


def load_corpus(batches_dir: Path) -> Tuple[dict, int]:
    """Concatenate every npz batch into single column arrays."""
    columns: Dict[str, List[np.ndarray]] = {
        "obs": [], "actions": [], "sequence": [], "ticks": [],
        "slots": [], "maps": [],
    }
    files = sorted(batches_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no *.npz batches under {batches_dir}")
    for path in files:
        with np.load(path) as batch:
            for key in columns:
                columns[key].append(np.asarray(batch[key]))
    corpus = {key: np.concatenate(parts, axis=0) for key, parts in columns.items()}
    n_rows = corpus["obs"].shape[0]
    if corpus["obs"].shape[1] != OBS_DIM:
        raise ValueError(
            f"obs width {corpus['obs'].shape[1]} != OBS_DIM={OBS_DIM}; "
            "teacher demos require Q2_EXT_OBS=1 (219-wide observations)"
        )
    if corpus["actions"].shape != (n_rows, ACTION_DIM):
        raise ValueError(
            f"actions shape {corpus['actions'].shape} != ({n_rows}, {ACTION_DIM})"
        )
    for key in ("sequence", "ticks", "slots", "maps"):
        if corpus[key].shape != (n_rows,):
            raise ValueError(f"{key} shape {corpus[key].shape} != ({n_rows},)")
    return corpus, len(files)


def build_episodes(corpus: dict) -> List[dict]:
    """Group rows into (map, slot) trajectories sorted by (tick, sequence)."""
    order = np.lexsort(
        (corpus["sequence"], corpus["ticks"], corpus["slots"], corpus["maps"])
    )
    maps = corpus["maps"][order]
    slots = corpus["slots"][order]
    change = np.zeros(len(order), dtype=bool)
    change[0] = True
    change[1:] = (maps[1:] != maps[:-1]) | (slots[1:] != slots[:-1])
    starts = np.flatnonzero(change)

    episodes = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(order)
        idx = order[start:end]
        episodes.append({
            "key": f"{maps[start]}/slot{int(slots[start])}",
            "obs": corpus["obs"][idx].astype(np.float32),
            "actions": corpus["actions"][idx].astype(np.float32),
        })
    return episodes


def split_episodes(
    episodes: List[dict], holdout_frac: float, rng: np.random.Generator
) -> Tuple[List[dict], List[dict]]:
    """Hold out whole episodes; train/holdout are disjoint when n >= 2."""
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")
    perm = rng.permutation(len(episodes))
    n_holdout = int(round(holdout_frac * len(episodes)))
    if len(episodes) >= 2:
        n_holdout = min(max(1, n_holdout), len(episodes) - 1)
    else:
        n_holdout = 1  # degenerate single-episode corpus: eval on the train episode
    holdout_idx = set(perm[:n_holdout].tolist())
    train = [ep for i, ep in enumerate(episodes) if i not in holdout_idx]
    holdout = [ep for i, ep in enumerate(episodes) if i in holdout_idx]
    return train, holdout


def episode_segments(
    episodes: List[dict], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunk episodes into (S, seq_len, ...) segments, dropping ragged tails."""
    obs_parts: List[np.ndarray] = []
    act_parts: List[np.ndarray] = []
    for ep in episodes:
        n_seg = len(ep["obs"]) // seq_len
        if n_seg == 0:
            continue
        obs_parts.append(
            ep["obs"][: n_seg * seq_len].reshape(n_seg, seq_len, OBS_DIM)
        )
        act_parts.append(
            ep["actions"][: n_seg * seq_len].reshape(n_seg, seq_len, ACTION_DIM)
        )
    if not obs_parts:
        return (
            np.zeros((0, seq_len, OBS_DIM), dtype=np.float32),
            np.zeros((0, seq_len, ACTION_DIM), dtype=np.float32),
        )
    return np.concatenate(obs_parts, axis=0), np.concatenate(act_parts, axis=0)


def _teacher_targets(actions: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "cont": actions[..., :4],
        "jump": actions[..., 4].round().long().clamp(0, 1),
        "fire": actions[..., 5].round().long().clamp(0, 1),
        "hook": actions[..., 6].round().long().clamp(0, 3),
        "weapon": actions[..., 7].round().long().clamp(0, 9),
    }


@torch.no_grad()
def evaluate(
    policy: Q2BotPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    batch_seqs: int = 256,
) -> Dict[str, float]:
    """Holdout gate metrics; fire is conditioned on the teacher's weapon."""
    policy.eval()
    yaw_err: List[np.ndarray] = []
    pitch_err: List[np.ndarray] = []
    move_err: List[np.ndarray] = []
    fire_pred: List[np.ndarray] = []
    fire_true: List[np.ndarray] = []
    weapon_hit: List[np.ndarray] = []
    for start in range(0, len(obs), batch_seqs):
        obs_t = torch.from_numpy(obs[start:start + batch_seqs]).to(device)
        act_t = torch.from_numpy(actions[start:start + batch_seqs]).to(device)
        targets = _teacher_targets(act_t)
        hx = policy.init_hidden(obs_t.shape[0], device)
        params, _value, _hx = policy(obs_t, hx)
        cont = params["cont_mean"]
        yaw_err.append((cont[..., 2] - targets["cont"][..., 2]).abs().cpu().numpy())
        pitch_err.append((cont[..., 3] - targets["cont"][..., 3]).abs().cpu().numpy())
        move_err.append((cont[..., :2] - targets["cont"][..., :2]).abs().cpu().numpy())
        fire_logits = policy.fire_logits_for(params["feat"], targets["weapon"])
        fire_pred.append((fire_logits.argmax(dim=-1) > 0).cpu().numpy())
        fire_true.append((targets["fire"] > 0).cpu().numpy())
        weapon_hit.append(
            (params["weapon_logits"].argmax(dim=-1) == targets["weapon"]).cpu().numpy()
        )

    pred = np.concatenate([a.ravel() for a in fire_pred])
    true = np.concatenate([a.ravel() for a in fire_true])
    tp = int(np.count_nonzero(pred & true))
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))
    non_fire = int(np.count_nonzero(~true))
    return {
        "yaw_mae_deg": float(np.concatenate([a.ravel() for a in yaw_err]).mean()),
        "pitch_mae_deg": float(np.concatenate([a.ravel() for a in pitch_err]).mean()),
        "move_drift_mae": float(np.concatenate([a.ravel() for a in move_err]).mean()),
        "fire_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "fire_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "hidden_fire_rate": float(fp / non_fire) if non_fire else 0.0,
        "weapon_top1_accuracy": float(
            np.concatenate([a.ravel() for a in weapon_hit]).mean()
        ),
    }


def train(
    policy: Q2BotPolicy,
    train_segments: Tuple[np.ndarray, np.ndarray],
    holdout_segments: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    args,
) -> List[dict]:
    obs, actions = train_segments
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    history: List[dict] = []

    policy.train()
    for epoch in range(args.epochs):
        perm = rng.permutation(len(obs))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(obs), args.batch_seqs):
            idx = perm[start:start + args.batch_seqs]
            obs_t = torch.from_numpy(obs[idx]).to(device)
            act_t = torch.from_numpy(actions[idx]).to(device)
            targets = _teacher_targets(act_t)

            hx = policy.init_hidden(obs_t.shape[0], device)
            params, _value, _hx = policy(obs_t, hx)
            fire_logits = policy.fire_logits_for(params["feat"], targets["weapon"])

            cont_loss = F.mse_loss(params["cont_mean"], targets["cont"])
            jump_loss = F.cross_entropy(
                params["jump_logits"].reshape(-1, 2), targets["jump"].reshape(-1)
            )
            fire_loss = F.cross_entropy(
                fire_logits.reshape(-1, 2), targets["fire"].reshape(-1)
            )
            hook_loss = F.cross_entropy(
                params["hook_logits"].reshape(-1, 4), targets["hook"].reshape(-1)
            )
            weapon_loss = F.cross_entropy(
                params["weapon_logits"].reshape(-1, 10), targets["weapon"].reshape(-1)
            )
            loss = (
                args.w_cont * cont_loss
                + args.w_jump * jump_loss
                + args.w_fire * fire_loss
                + args.w_hook * hook_loss
                + args.w_weapon * weapon_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            batches += 1

        metrics = evaluate(policy, *holdout_segments, device)
        policy.train()
        entry = {"epoch": epoch + 1, "train_loss": total_loss / max(1, batches)}
        entry.update(metrics)
        history.append(entry)
        print(
            f"epoch={epoch + 1} train_loss={entry['train_loss']:.4f} "
            f"yaw_mae_deg={metrics['yaw_mae_deg']:.3f} "
            f"pitch_mae_deg={metrics['pitch_mae_deg']:.3f} "
            f"fire_p={metrics['fire_precision']:.3f} "
            f"fire_r={metrics['fire_recall']:.3f}",
            flush=True,
        )
    return history


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/bc_demos_v1")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_seqs", type=int, default=64,
                        help="segments per training batch")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="segment length, matching the trainer's chunk_len")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--holdout_frac", type=float, default=0.1,
                        help="fraction of whole (map, slot) episodes held out")
    parser.add_argument("--seed", type=int, default=7142026)
    parser.add_argument("--w_cont", type=float, default=1.0)
    parser.add_argument("--w_jump", type=float, default=0.5)
    parser.add_argument("--w_fire", type=float, default=1.0)
    parser.add_argument("--w_hook", type=float, default=0.5)
    parser.add_argument("--w_weapon", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _pick_device(args.device)

    batches_dir = Path(args.batches_dir).expanduser()
    corpus, n_files = load_corpus(batches_dir)
    episodes = build_episodes(corpus)
    n_rows = int(corpus["obs"].shape[0])
    print(
        f"corpus: files={n_files} rows={n_rows} episodes={len(episodes)} "
        f"obs_dim={OBS_DIM} device={device}"
    )

    split_rng = np.random.default_rng(args.seed)
    train_eps, holdout_eps = split_episodes(episodes, args.holdout_frac, split_rng)
    train_segments = episode_segments(train_eps, args.seq_len)
    holdout_segments = episode_segments(holdout_eps, args.seq_len)
    if len(train_segments[0]) == 0:
        raise ValueError(
            f"no train segments: {len(train_eps)} episodes all shorter "
            f"than seq_len={args.seq_len}"
        )
    if len(holdout_segments[0]) == 0:
        raise ValueError(
            f"no holdout segments: {len(holdout_eps)} episodes all shorter "
            f"than seq_len={args.seq_len}"
        )
    print(
        f"episodes: train={len(train_eps)} holdout={len(holdout_eps)} "
        f"segments: train={len(train_segments[0])} "
        f"holdout={len(holdout_segments[0])}"
    )

    policy = Q2BotPolicy().to(device)
    history = train(policy, train_segments, holdout_segments, device, args)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "policy_bc_final.pt"
    torch.save(policy.state_dict(), checkpoint)

    report = {
        "gates": history[-1] if history else {},
        "history": history,
        "corpus": {
            "files": n_files,
            "rows": n_rows,
            "episodes": len(episodes),
            "train_segments": int(len(train_segments[0])),
            "holdout_segments": int(len(holdout_segments[0])),
        },
        "episodes": {
            "train": [ep["key"] for ep in train_eps],
            "holdout": [ep["key"] for ep in holdout_eps],
        },
        "config": {
            "batches_dir": str(batches_dir),
            "epochs": args.epochs,
            "batch_seqs": args.batch_seqs,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "holdout_frac": args.holdout_frac,
            "seed": args.seed,
            "loss_weights": {
                "cont": args.w_cont, "jump": args.w_jump, "fire": args.w_fire,
                "hook": args.w_hook, "weapon": args.w_weapon,
            },
            "device": str(device),
            "obs_dim": OBS_DIM,
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved_checkpoint={checkpoint}")
    print(f"saved_metrics={metrics_path}")
    print("gates=" + json.dumps(report["gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
