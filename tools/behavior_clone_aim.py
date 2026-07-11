#!/usr/bin/env python3
"""
Warm-start the policy with a simple local-frame aim/fire teacher.

This is a curriculum bootstrap, not the final controller: it teaches the
network to turn toward visible enemies and fire so PPO can optimize combat
from a policy that can produce damage.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from models.policy import OBS_DIM, Q2BotPolicy, export_onnx

TEACHER_WEAPON = 7  # Rocket Launcher; ML training spawn carries rockets.


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


def _teacher_action(obs) -> np.ndarray:
    action = np.array([0.35, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, TEACHER_WEAPON], dtype=np.float32)
    best = None
    count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
    for ent in obs.entities[:count]:
        if ent[7] > 0.5 and ent[8] > 0.5:
            dist = float(np.linalg.norm(ent[:3]))
            if best is None or dist < best[0]:
                best = (dist, ent[:3].copy())

    if best is None:
        return action

    x, y, z = best[1]
    yaw = math.degrees(math.atan2(float(y), float(x)))
    pitch = math.degrees(math.atan2(float(z), max(math.hypot(float(x), float(y)), 1e-3)))
    action[0] = 0.0
    action[1] = 0.0
    action[2] = float(np.clip(-yaw, -45.0, 45.0))
    action[3] = float(np.clip(-pitch, -30.0, 30.0))
    action[5] = 1.0
    return action


def collect(args) -> Tuple[np.ndarray, np.ndarray, dict]:
    maps = discover_map_pool(map_glob=args.map_glob, map_dir=args.map_dir or None)
    obs_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    stats = {
        "visible_samples": 0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "items": 0.0,
        "episodes": 0,
    }

    map_idx = 0
    while (
        (len(obs_rows) < args.samples or stats["visible_samples"] < args.min_visible_samples)
        and len(obs_rows) < args.max_collected
    ):
        map_name = maps[map_idx % len(maps)]
        map_idx += 1
        env = Q2MultiEnv(
            map_pool=[map_name],
            n_bots=args.n_bots,
            port_offset=args.port_offset,
            maxclients=args.maxclients,
            ml_slot=args.ml_slot,
            max_ep_steps=args.episode_steps,
        )
        try:
            env.reset_all()
            for _ in range(args.episode_steps):
                obs = env._last_obs[0]
                action = _teacher_action(obs)
                obs_rows.append(obs.to_vector())
                act_rows.append(action)
                stats["visible_samples"] += int(action[5] > 0.5)

                _vec, _reward, term, trunc, info = env.step_all([action])[0]
                stats["damage_dealt"] += float(info.get("damage_dealt", 0.0))
                stats["damage_taken"] += float(info.get("damage_taken", 0.0))
                stats["items"] += float(info.get("items", 0.0))
                if term or trunc:
                    stats["episodes"] += 1
                    break
                reached_targets = (
                    len(obs_rows) >= args.samples and
                    stats["visible_samples"] >= args.min_visible_samples
                )
                if reached_targets or len(obs_rows) >= args.max_collected:
                    break
        finally:
            env.close()

    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.float32), stats


def synthetic_aim(args) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(args.seed)
    obs = np.zeros((args.synthetic_samples, OBS_DIM), dtype=np.float32)
    actions = np.zeros((args.synthetic_samples, 8), dtype=np.float32)
    visible = 0

    for i in range(args.synthetic_samples):
        obs[i, 0:3] = np.array([
            rng.uniform(-1200.0, 2200.0),
            rng.uniform(-1200.0, 2200.0),
            rng.uniform(-64.0, 512.0),
        ], dtype=np.float32)
        obs[i, 3:6] = rng.uniform(-320.0, 320.0, size=3).astype(np.float32)
        obs[i, 6] = rng.uniform(40.0, 100.0)   # health
        obs[i, 8] = rng.choice([8.0, 15.0, 37.0])
        obs[i, 9] = rng.choice([0.0, 1.0, 10.0, 50.0, 100.0])
        action = np.array([0.35, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, TEACHER_WEAPON], dtype=np.float32)

        visible_slot = -1
        if rng.random() < args.synthetic_visible_rate:
            visible_slot = 0

        for slot in range(3):
            dist = rng.uniform(180.0, 1200.0)
            yaw = rng.uniform(-args.synthetic_yaw_abs, args.synthetic_yaw_abs)
            pitch = rng.uniform(-24.0, 24.0)
            x = dist * math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
            y = dist * math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
            z = dist * math.sin(math.radians(pitch))

            ent0 = 10 + slot * 9
            obs[i, ent0 + 0] = x
            obs[i, ent0 + 1] = y
            obs[i, ent0 + 2] = z
            obs[i, ent0 + 6] = 100.0
            obs[i, ent0 + 7] = 1.0
            obs[i, ent0 + 8] = 1.0 if slot == visible_slot else 0.0

            if slot != visible_slot:
                continue

            action[0] = 0.0
            action[1] = 0.0
            action[2] = float(np.clip(-yaw, -45.0, 45.0))
            action[3] = float(np.clip(-pitch, -30.0, 30.0))
            action[5] = 1.0
            visible += 1

        ray0 = 10 + 8 * 9
        for ray in range(16):
            off = ray0 + ray * 4
            angle = (2.0 * math.pi * ray) / 16.0
            obs[i, off + 0] = math.cos(angle)
            obs[i, off + 1] = math.sin(angle)
            obs[i, off + 2] = rng.uniform(-0.15, 0.15)
            obs[i, off + 3] = rng.uniform(96.0, 2400.0)

        actions[i] = action

    return obs, actions, {
        "visible_samples": visible,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "items": 0.0,
        "episodes": 0,
    }


def train_bc(policy: Q2BotPolicy, obs: np.ndarray, actions: np.ndarray, device: torch.device, args) -> None:
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scale = torch.tensor([1.0, 1.0, 45.0, 30.0], device=device)

    policy.train()
    n = obs_t.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            params, _value, _hx = policy(obs_t[idx].unsqueeze(1))
            cont_mean = params["cont_mean"].squeeze(1)
            visible_w = torch.where(
                act_t[idx, 5] > 0.5,
                torch.full_like(act_t[idx, 5], args.visible_weight),
                torch.ones_like(act_t[idx, 5]),
            )
            cont_error = (cont_mean[:, :4] - act_t[idx, :4]) / scale
            cont_weights = torch.tensor(
                [args.move_weight, args.move_weight, args.look_weight, args.look_weight],
                device=device,
            )
            cont_per = (cont_error.pow(2) * cont_weights).mean(dim=1)
            cont_loss = (cont_per * visible_w).sum() / visible_w.sum().clamp_min(1.0)
            jump_loss = F.cross_entropy(params["jump_logits"].squeeze(1), act_t[idx, 4].long().clamp(0, 1))
            fire_loss = F.cross_entropy(
                params["fire_logits"].squeeze(1),
                act_t[idx, 5].long().clamp(0, 1),
                weight=torch.tensor([1.0, args.fire_weight], device=device),
            )
            hook_loss = F.cross_entropy(params["hook_logits"].squeeze(1), act_t[idx, 6].long().clamp(0, 3))
            weapon_loss = F.cross_entropy(params["weapon_logits"].squeeze(1), act_t[idx, 7].long().clamp(0, 9))
            loss = (
                cont_loss
                + 0.1 * jump_loss
                + fire_loss
                + 0.1 * hook_loss
                + args.weapon_weight * weapon_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total / max(1, (n + args.batch_size - 1) // args.batch_size):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--map_glob", default="mltrain_*.bsp")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--min_visible_samples", type=int, default=500)
    parser.add_argument("--max_collected", type=int, default=30000)
    parser.add_argument("--episode_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--visible_weight", type=float, default=8.0)
    parser.add_argument("--fire_weight", type=float, default=20.0)
    parser.add_argument("--weapon_weight", type=float, default=0.25)
    parser.add_argument("--move_weight", type=float, default=1.0)
    parser.add_argument("--look_weight", type=float, default=16.0)
    parser.add_argument("--n_bots", type=int, default=4)
    parser.add_argument("--port_offset", type=int, default=20)
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=11)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic_samples", type=int, default=20000)
    parser.add_argument("--synthetic_visible_rate", type=float, default=0.65)
    parser.add_argument("--synthetic_yaw_abs", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = _pick_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"loaded={ckpt} device={device}")

    obs, actions, stats = synthetic_aim(args) if args.synthetic else collect(args)
    print(f"collected={len(obs)} visible_samples={stats['visible_samples']} "
          f"damage={stats['damage_dealt']:.0f}/{stats['damage_taken']:.0f} "
          f"items={stats['items']:.0f}")
    train_bc(policy, obs, actions, device, args)

    try:
        step = int(ckpt.stem.split("_")[-1]) + 1
    except ValueError:
        step = 1
    out = ROOT / "checkpoints" / f"policy_{step:08d}.pt"
    torch.save(policy.state_dict(), out)
    try:
        export_onnx(policy, str(ROOT / "checkpoints" / "policy_latest.onnx"), device)
    except Exception as exc:
        print(f"onnx export failed: {exc}")
    print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
