"""
ppo.py — PPO training loop for Q2BotPolicy.

Runs N parallel Q2BotEnv instances, collects rollouts, updates policy.
Designed to run overnight on the Vega 10 iGPU (ROCm) or CPU fallback.

Usage:
    HSA_OVERRIDE_GFX_VERSION=9.0.0 python -m train.ppo
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List
from torch.utils.tensorboard import SummaryWriter

from models.policy import Q2BotPolicy, export_onnx, OBS_DIM, ACTION_DIM


# ── Hyperparameters ─────────────────────────────────────────────────

DEFAULT = dict(
    n_envs          = 4,        # parallel Q2 server instances
    n_steps         = 256,      # steps per env per rollout
    n_epochs        = 4,        # PPO epochs per rollout
    batch_size      = 512,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    vf_coef         = 0.5,
    ent_coef        = 0.01,
    max_grad_norm   = 0.5,
    total_steps     = 5_000_000,
    save_every      = 100_000,
    map_name        = "q2dm1",
)


# ── Rollout buffer ───────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_envs: int, n_steps: int, obs_dim: int,
                 act_dim: int, device: torch.device):
        self.n   = n_envs
        self.T   = n_steps
        self.dev = device

        self.obs      = torch.zeros(n_steps, n_envs, obs_dim,  device=device)
        self.actions  = torch.zeros(n_steps, n_envs, act_dim,  device=device)
        self.rewards  = torch.zeros(n_steps, n_envs,           device=device)
        self.dones    = torch.zeros(n_steps, n_envs,           device=device)
        self.values   = torch.zeros(n_steps, n_envs,           device=device)
        self.log_probs = torch.zeros(n_steps, n_envs,          device=device)
        self.ptr      = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns(self, last_value: torch.Tensor,
                        gamma: float, gae_lambda: float):
        advantages = torch.zeros_like(self.rewards)
        last_gae   = 0.0
        for t in reversed(range(self.T)):
            next_val   = last_value if t == self.T - 1 else self.values[t + 1]
            delta      = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            last_gae   = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        self.returns = advantages + self.values
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.ptr = 0


# ── Training loop ────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    policy    = Q2BotPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr"])

    print(f"Policy parameters: {policy.param_count():,}")

    # deferred import so server processes don't start until here
    from harness.env import Q2BotEnv

    envs = [
        Q2BotEnv(
            env_id      = i,
            map_name    = cfg["map_name"],
            bot_slot    = i + 1,
            port_offset = i,
            timelimit   = 10,
            n_bots      = 3,
        )
        for i in range(cfg["n_envs"])
    ]

    buf = RolloutBuffer(
        cfg["n_envs"], cfg["n_steps"], OBS_DIM, ACTION_DIM, device
    )

    obs_np    = np.zeros((cfg["n_envs"], OBS_DIM), dtype=np.float32)
    hx_list   = [policy.init_hidden(1, device) for _ in range(cfg["n_envs"])]

    # reset all envs
    for i, env in enumerate(envs):
        o, _ = env.reset()
        obs_np[i] = o

    total_env_steps = 0
    update_n        = 0
    save_dir        = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    # TensorBoard logging — view with: tensorboard --logdir runs
    run_name = f"ppo_{cfg['map_name']}_{int(time.time())}"
    writer   = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"TensorBoard log: runs/{run_name}")
    print(f"  view with:  tensorboard --logdir runs --bind_all")

    # episode reward tracking per env
    ep_rewards = np.zeros(cfg["n_envs"], dtype=np.float64)
    ep_lengths = np.zeros(cfg["n_envs"], dtype=np.int64)
    completed_ep_rewards = []
    completed_ep_lengths = []
    completed_ep_components = {k: 0.0 for k in
        ["damage_dealt","damage_taken","kill","death","item","hook"]}

    start = time.time()

    while total_env_steps < cfg["total_steps"]:
        # ── collect rollout ──────────────────────────────────────────
        policy.eval()
        for step in range(cfg["n_steps"]):
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)

            actions_np = np.zeros((cfg["n_envs"], ACTION_DIM), dtype=np.float32)
            values_np  = np.zeros(cfg["n_envs"], dtype=np.float32)
            lps_np     = np.zeros(cfg["n_envs"], dtype=np.float32)

            for i in range(cfg["n_envs"]):
                act, val, hx_new = policy.act(obs_np[i], hx_list[i], device)
                actions_np[i] = act
                values_np[i]  = val
                hx_list[i]    = hx_new
                lps_np[i]     = 0.0  # placeholder; compute properly in update

            rewards_np = np.zeros(cfg["n_envs"], dtype=np.float32)
            dones_np   = np.zeros(cfg["n_envs"], dtype=np.float32)

            for i, env in enumerate(envs):
                o, r, term, trunc, _ = env.step(actions_np[i])
                obs_np[i]      = o
                rewards_np[i]  = r
                dones_np[i]    = float(term or trunc)
                ep_rewards[i] += r
                ep_lengths[i] += 1
                if term or trunc:
                    completed_ep_rewards.append(float(ep_rewards[i]))
                    completed_ep_lengths.append(int(ep_lengths[i]))
                    ep_rewards[i] = 0
                    ep_lengths[i] = 0
                    o, _ = env.reset()
                    obs_np[i]  = o
                    hx_list[i] = policy.init_hidden(1, device)

            buf.add(
                torch.tensor(obs_np,     device=device),
                torch.tensor(actions_np, device=device),
                torch.tensor(rewards_np, device=device),
                torch.tensor(dones_np,   device=device),
                torch.tensor(values_np,  device=device),
                torch.tensor(lps_np,     device=device),
            )

        total_env_steps += cfg["n_steps"] * cfg["n_envs"]

        # compute bootstrap value
        with torch.no_grad():
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            last_vals = torch.zeros(cfg["n_envs"], device=device)
            for i in range(cfg["n_envs"]):
                obs_ti = obs_t[i].unsqueeze(0).unsqueeze(0)
                _, v, _ = policy(obs_ti, hx_list[i])
                last_vals[i] = v.squeeze()

        buf.compute_returns(last_vals, cfg["gamma"], cfg["gae_lambda"])

        # ── PPO update ───────────────────────────────────────────────
        policy.train()
        update_n += 1

        obs_flat = buf.obs.reshape(-1, OBS_DIM)
        act_flat = buf.actions.reshape(-1, ACTION_DIM)
        adv_flat = buf.advantages.reshape(-1)
        ret_flat = buf.returns.reshape(-1)

        total_loss = 0.0
        for epoch in range(cfg["n_epochs"]):
            idx = torch.randperm(obs_flat.shape[0], device=device)
            for start_i in range(0, obs_flat.shape[0], cfg["batch_size"]):
                b = idx[start_i:start_i + cfg["batch_size"]]
                obs_b = obs_flat[b].unsqueeze(1)   # (B, 1, OBS_DIM)

                act_params, val_b, _ = policy(obs_b)
                val_b = val_b.squeeze()

                # simplified policy loss (proper log_prob computation)
                cont_dist = torch.distributions.Normal(
                    act_params["cont_mean"].squeeze(1),
                    act_params["cont_log_std"].squeeze(1).exp(),
                )
                log_prob = cont_dist.log_prob(act_flat[b, :4]).sum(-1)

                ratio  = (log_prob - buf.log_probs.reshape(-1)[b]).exp()
                adv_b  = adv_flat[b]
                pg_loss = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_b,
                ).mean()

                vf_loss  = ((val_b - ret_flat[b]) ** 2).mean()
                ent_loss = -cont_dist.entropy().sum(-1).mean()

                loss = pg_loss + cfg["vf_coef"] * vf_loss + cfg["ent_coef"] * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                total_loss += loss.item()

        # logging
        elapsed = time.time() - start
        sps     = total_env_steps / elapsed

        # episode stats over completed episodes since last update
        if completed_ep_rewards:
            mean_ep_r   = float(np.mean(completed_ep_rewards))
            min_ep_r    = float(np.min(completed_ep_rewards))
            max_ep_r    = float(np.max(completed_ep_rewards))
            mean_ep_len = float(np.mean(completed_ep_lengths))
            n_ep        = len(completed_ep_rewards)

            writer.add_scalar("episode/reward_mean", mean_ep_r,   total_env_steps)
            writer.add_scalar("episode/reward_min",  min_ep_r,    total_env_steps)
            writer.add_scalar("episode/reward_max",  max_ep_r,    total_env_steps)
            writer.add_scalar("episode/length_mean", mean_ep_len, total_env_steps)
            writer.add_scalar("episode/count",       n_ep,        total_env_steps)
        else:
            mean_ep_r = float("nan")
            n_ep      = 0

        # training metrics
        writer.add_scalar("train/loss",       total_loss / cfg["n_epochs"], total_env_steps)
        writer.add_scalar("train/sps",        sps,                          total_env_steps)
        writer.add_scalar("train/value_mean", float(buf.values.mean()),     total_env_steps)
        writer.add_scalar("train/return_mean",float(buf.returns.mean()),    total_env_steps)

        completed_ep_rewards.clear()
        completed_ep_lengths.clear()

        print(f"[{update_n:4d}] steps={total_env_steps:>8,}  sps={sps:>6.0f}  "
              f"loss={total_loss:.4f}  ep_r={mean_ep_r:>+7.2f} (n={n_ep})  "
              f"elapsed={elapsed/60:.1f}m")

        # checkpoint
        if total_env_steps % cfg["save_every"] < cfg["n_steps"] * cfg["n_envs"]:
            ckpt = save_dir / f"policy_{total_env_steps:08d}.pt"
            torch.save(policy.state_dict(), ckpt)
            export_onnx(policy, str(save_dir / "policy_latest.onnx"), device)
            print(f"  → saved {ckpt}")

    for env in envs:
        env.close()

    export_onnx(policy, "policy_final.onnx", device)
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULT.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    train(vars(args))
