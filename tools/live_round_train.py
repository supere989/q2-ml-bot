#!/usr/bin/env python3
"""Live deathmatch sparring trainer with per-round PPO updates.

Runs standard deathmatch rules, collects ML bot rollouts during the round,
shows an in-game training progress bar during intermission, updates the policy,
then advances to the next round.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
LOCAL_Q2_ROOT = ROOT.parent / "q2_lithium_merge"
if LOCAL_Q2_ROOT.exists():
    os.environ.setdefault("Q2_ROOT", str(LOCAL_Q2_ROOT))
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from models.policy import ACTION_DIM, OBS_DIM, Q2BotPolicy, export_onnx


class RoundBuffer:
    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, obs, actions, rewards, dones, values, log_probs) -> None:
        self.obs.append(np.array(obs, dtype=np.float32, copy=True))
        self.actions.append(np.array(actions, dtype=np.float32, copy=True))
        self.rewards.append(np.array(rewards, dtype=np.float32, copy=True))
        self.dones.append(np.array(dones, dtype=np.float32, copy=True))
        self.values.append(np.array(values, dtype=np.float32, copy=True))
        self.log_probs.append(np.array(log_probs, dtype=np.float32, copy=True))

    def tensors(self, device: torch.device, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        obs = torch.tensor(np.stack(self.obs), device=device)
        actions = torch.tensor(np.stack(self.actions), device=device)
        rewards = torch.tensor(np.stack(self.rewards), device=device)
        dones = torch.tensor(np.stack(self.dones), device=device)
        values = torch.tensor(np.stack(self.values), device=device)
        old_log_probs = torch.tensor(np.stack(self.log_probs), device=device)

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[1], device=device)
        for t in reversed(range(rewards.shape[0])):
            next_value = last_values if t == rewards.shape[0] - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return obs, actions, returns, advantages, old_log_probs


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


def _save_policy(policy: Q2BotPolicy, device: torch.device, round_idx: int, steps: int) -> Path:
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"policy_live_round_{round_idx:04d}_{steps:08d}.pt"
    torch.save(policy.state_dict(), ckpt)
    torch.save(policy.state_dict(), ckpt_dir / "policy_live_latest.pt")
    try:
        export_onnx(policy, str(ckpt_dir / "policy_live_latest.onnx"), device)
    except Exception as exc:
        print(f"! onnx export failed: {exc}", flush=True)
    return ckpt


def _ppo_update(
    policy: Q2BotPolicy,
    optimizer: optim.Optimizer,
    env: Q2MultiEnv,
    buf: RoundBuffer,
    device: torch.device,
    args: argparse.Namespace,
    round_idx: int,
) -> Dict[str, float]:
    last_values = torch.zeros(args.num_ml_bots, device=device)
    obs, actions, returns, advantages, old_log_probs = buf.tensors(
        device, last_values, args.gamma, args.gae_lambda
    )

    obs_flat = obs.reshape(-1, OBS_DIM)
    act_flat = actions.reshape(-1, ACTION_DIM)
    ret_flat = returns.reshape(-1)
    adv_flat = advantages.reshape(-1)
    old_lp_flat = old_log_probs.reshape(-1)

    policy.train()
    losses = []
    kls = []
    total_batches = max(1, args.n_epochs * ((obs_flat.shape[0] + args.batch_size - 1) // args.batch_size))
    batch_n = 0

    for epoch in range(1, args.n_epochs + 1):
        idx = torch.randperm(obs_flat.shape[0], device=device)
        for start in range(0, obs_flat.shape[0], args.batch_size):
            batch_n += 1
            progress = 100.0 * batch_n / total_batches
            env.set_training_progress(
                True,
                epoch=epoch,
                epochs=args.n_epochs,
                progress=progress,
                status=f"round {round_idx} PPO update",
            )

            b = idx[start:start + args.batch_size]
            obs_b = obs_flat[b].unsqueeze(1)
            act_b = act_flat[b].unsqueeze(1)

            act_params, value_b, _ = policy(obs_b)
            log_prob, entropy = policy.action_log_prob_entropy(act_params, act_b)
            log_prob = log_prob.reshape(-1)
            entropy = entropy.reshape(-1)
            value = value_b.reshape(-1)

            ratio = (log_prob - old_lp_flat[b]).exp()
            adv = adv_flat[b]
            pg_loss = -torch.min(
                ratio * adv,
                ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv,
            ).mean()
            vf_loss = ((value - ret_flat[b]) ** 2).mean()
            ent_loss = -entropy.mean()
            loss = pg_loss + args.vf_coef * vf_loss + args.ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            approx_kl = (old_lp_flat[b] - log_prob).mean().detach().abs().item()
            losses.append(float(loss.item()))
            kls.append(approx_kl)
            if args.target_kl > 0.0 and approx_kl > args.target_kl:
                break

    env.set_training_progress(True, args.n_epochs, args.n_epochs, 100.0, "checkpointing")
    policy.eval()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
    }


def _round_map(maps: List[str], round_idx: int, args: argparse.Namespace) -> str:
    if not maps:
        return args.map_name
    if args.map_change_rounds <= 0:
        return maps[0]
    return maps[((round_idx - 1) // args.map_change_rounds) % len(maps)]


def run(args: argparse.Namespace) -> int:
    device = _pick_device()
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    maps = discover_map_pool(args.map_name, args.map_glob, args.map_dir or None)
    if args.shuffle_maps:
        random.Random(args.map_seed).shuffle(maps)

    print(f"checkpoint={ckpt}")
    print(f"device={device}")
    print(f"maps={maps}")
    print(f"rules=deathmatch timelimit={args.timelimit} fraglimit={args.fraglimit}")
    print(f"watch_connect=127.0.0.1:{27910 + args.port_offset}")

    env = Q2MultiEnv(
        server_id=0,
        map_name=_round_map(maps, 1, args),
        map_pool=[_round_map(maps, 1, args)],
        map_seed=args.map_seed,
        n_bots=args.num_ml_bots,
        port_offset=args.port_offset,
        maxclients=args.maxclients,
        ml_slot=args.ml_slot,
        num_ml_bots=args.num_ml_bots,
        max_ep_steps=max(args.max_round_steps + 5, 100),
        timedemo=0,
        timelimit=args.timelimit,
        fraglimit=args.fraglimit,
        intermission_time=args.intermission_time,
        intermission_maxtime=args.intermission_maxtime,
        console_pipe=True,
        start_observer=False,
        spectator_only=False,
    )

    report_path = Path(args.report_file) if args.report_file else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    try:
        obs_list = env.reset_all()
        obs_np = np.stack(obs_list).astype(np.float32)
        hx_list = [policy.init_hidden(1, device) for _ in range(args.num_ml_bots)]

        for round_idx in range(1, args.rounds + 1):
            map_name = _round_map(maps, round_idx, args)
            if env.active_map != map_name:
                obs_np = np.stack(env.trigger_next_round(map_name)).astype(np.float32)
                hx_list = [policy.init_hidden(1, device) for _ in range(args.num_ml_bots)]

            env.set_training_progress(False)
            buf = RoundBuffer()
            stats = {
                "round": round_idx,
                "map": map_name,
                "steps": 0,
                "kills": 0.0,
                "deaths": 0.0,
                "damage_dealt": 0.0,
                "damage_taken": 0.0,
                "intermission": False,
            }
            start = time.time()

            for _ in range(args.max_round_steps):
                with torch.no_grad():
                    actions_np, values_np, log_probs_np, hx_list = policy.act_batch(
                        obs_np, hx_list, device=device, deterministic=False
                    )

                results = env.step_all(list(actions_np))
                rewards = np.zeros(args.num_ml_bots, dtype=np.float32)
                dones = np.zeros(args.num_ml_bots, dtype=np.float32)
                next_obs = np.zeros_like(obs_np)
                round_over = False

                for idx, (obs_vec, reward, term, trunc, info) in enumerate(results):
                    obs_obj = env._last_obs[idx]
                    rewards[idx] = float(reward)
                    dones[idx] = float(term or trunc)
                    next_obs[idx] = obs_vec
                    stats["kills"] += float(info.get("kills", 0.0))
                    stats["deaths"] += float(info.get("deaths", 0.0))
                    stats["damage_dealt"] += float(info.get("damage_dealt", 0.0))
                    stats["damage_taken"] += float(info.get("damage_taken", 0.0))
                    if obs_obj is not None and getattr(obs_obj, "round_intermission", False):
                        round_over = True

                buf.add(obs_np, actions_np, rewards, dones, values_np, log_probs_np)
                obs_np = next_obs
                stats["steps"] += 1
                total_steps += args.num_ml_bots

                if round_over:
                    stats["intermission"] = True
                    break

                for idx, result in enumerate(results):
                    if result[2] or result[3]:
                        obs_np[idx] = env.reset_slot(idx)
                        hx_list[idx] = policy.init_hidden(1, device)

                if args.step_sleep > 0.0:
                    time.sleep(args.step_sleep)

            if len(buf) == 0:
                print(f"round={round_idx} no rollout samples; stopping", flush=True)
                break

            env.set_training_progress(True, 0, args.n_epochs, 0.0, "preparing batch")
            metrics = _ppo_update(policy, optimizer, env, buf, device, args, round_idx)
            ckpt_path = _save_policy(policy, device, round_idx, total_steps)
            stats.update(metrics)
            stats["checkpoint"] = str(ckpt_path)
            stats["seconds"] = round(time.time() - start, 2)
            stats["samples"] = len(buf) * args.num_ml_bots
            line = json.dumps(stats, sort_keys=True)
            print("ROUND " + line, flush=True)
            if report_path:
                with report_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

            if args.intermission_hold_after > 0.0:
                time.sleep(args.intermission_hold_after)
            if round_idx < args.rounds:
                next_map = _round_map(maps, round_idx + 1, args)
                obs_np = np.stack(env.trigger_next_round(next_map)).astype(np.float32)
                hx_list = [policy.init_hidden(1, device) for _ in range(args.num_ml_bots)]
    finally:
        env.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--map_name", default="mltrain_00000000")
    parser.add_argument("--map_glob", default="")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--map_seed", type=int, default=0)
    parser.add_argument("--shuffle_maps", action="store_true")
    parser.add_argument("--map_change_rounds", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--timelimit", type=float, default=20.0)
    parser.add_argument("--fraglimit", type=int, default=50)
    parser.add_argument("--port_offset", type=int, default=944)
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=10)
    parser.add_argument("--num_ml_bots", type=int, default=2)
    parser.add_argument("--max_round_steps", type=int, default=16000)
    parser.add_argument("--step_sleep", type=float, default=0.08)
    parser.add_argument("--intermission_time", type=float, default=999999.0)
    parser.add_argument("--intermission_maxtime", type=float, default=-1.0)
    parser.add_argument("--intermission_hold_after", type=float, default=2.0)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.12)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--report_file", default="runs/live_round_train.jsonl")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
