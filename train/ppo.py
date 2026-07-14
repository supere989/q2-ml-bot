"""
ppo.py — PPO training loop for Q2BotPolicy.

Runs N parallel Q2BotEnv instances, collects rollouts, updates policy.
Designed to run overnight on the Vega 10 iGPU (ROCm) or CPU fallback.

Usage:
    HSA_OVERRIDE_GFX_VERSION=9.0.0 python -m train.ppo
"""

import atexit
import hashlib
import io
import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
from harness.rollout_protocol import (
    PPO_BEHAVIOR_METRIC_KEYS,
    PPO_EPISODE_SUMMARY_COLUMNS,
)
from models.policy import (
    ACTION_DIM,
    ENT_CNT,
    ENT_DIM,
    ENT_OFF,
    HIDDEN_DIM,
    OBS_DIM,
    WEAPON_CLASSES,
    Q2BotPolicy,
    export_onnx,
)

print("=== PPO MODULE LOADED ===")

# ── GPU thermal guard ─────────────────────────────────────────────────

_TEMP_PATH = Path("/sys/class/drm/card1/device/hwmon/hwmon6/temp1_input")  # AMD only
_TEMP_WARN  = 82.0   # °C — log warning
_TEMP_PAUSE = 88.0   # °C — pause training until cooled

def _gpu_temp() -> Optional[float]:
    try:
        return int(_TEMP_PATH.read_text()) / 1000.0
    except Exception:
        return None

def _thermal_check(writer, step: int):
    """Pause training if GPU is too hot. Returns temp for logging."""
    temp = _gpu_temp()
    if temp is None:
        return None
    if temp >= _TEMP_PAUSE:
        print(f"[thermal] GPU {temp:.0f}°C ≥ {_TEMP_PAUSE}°C — pausing until ≤ {_TEMP_WARN}°C")
        while True:
            time.sleep(10)
            temp = _gpu_temp()
            if temp is None or temp <= _TEMP_WARN:
                print(f"[thermal] GPU {temp:.0f}°C — resuming")
                break
    elif temp >= _TEMP_WARN:
        print(f"[thermal] GPU {temp:.0f}°C — warm, watching")
    if writer:
        writer.add_scalar("system/gpu_temp_c", temp, step)
    return temp


# ── Hyperparameters ─────────────────────────────────────────────────

DEFAULT = dict(
    seed                = 0,
    game_seed           = -1,        # >=0 enables deterministic game.so rand() per server
    deterministic       = 0,         # opt-in deterministic Torch/CUDA kernels for A/B runs
    n_servers           = 2,         # parallel q2ded instances
    n_bots_per_server   = 4,         # total bots in server (ML + 3ZB2 opponents)
    n_ml_bots           = 1,         # ML-controlled slots per server (venvs each)
    n_steps             = 256,       # steps per virtual-env per rollout
    n_epochs            = 4,
    batch_size          = 512,
    lr                  = 3e-4,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    clip_eps            = 0.2,
    vf_coef             = 0.5,
    ent_coef            = 0.01,
    max_grad_norm       = 0.5,
    total_steps         = 20_000_000,
    save_every          = 100_000,
    map_name            = "q2dm1",
    map_glob            = "",
    map_dir             = "",
    map_change_episodes = 0,
    map_seed            = 0,
    max_ep_steps        = 1000,
    timelimit           = 0.0,
    fraglimit           = 0,
    chunk_len           = 16,
    timescale           = 8.0,   # wall-clock compression on q2ded (game time unchanged)
    aux_coef            = 0.05,  # auxiliary next-obs prediction loss weight
    aim_anchor_coef     = 0.0,   # opt-in recurrent geometric aim/fire supervision
    aim_anchor_look_weight = 16.0,
    aim_anchor_fire_weight = 1.0,
    aim_anchor_yaw_deg  = 12.0,
    aim_anchor_pitch_deg = 14.0,
    target_fire_gate    = 1,     # network-native only: mask blind/unaligned fire
    lattice_direction_coef = 0.02,
)


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"


def _lattice_direction_loss(act_params: dict, obs: torch.Tensor) -> dict:
    """Teach movement to follow lattice pulls and convert contact to pursuit.

    The 24-d memory block is always the tail of the observation, independent
    of Q2_EXT_OBS. Away from contact, engagement/opportunity attract and threat
    repels. During live contact a combat-ready bot follows the nearest visible
    enemy in its already bot-local frame instead of dropping lattice guidance
    at precisely the point where actionability is required.
    """
    memory = obs[..., -24:]
    engagement = memory[..., 5:8] * memory[..., 8:9] * 0.5
    threat = memory[..., 9:12] * memory[..., 12:13]
    opportunity = memory[..., 13:16] * memory[..., 16:17]
    desired_world = engagement + opportunity - threat
    desired_xy = desired_world[..., :2]
    strength = torch.linalg.vector_norm(desired_xy, dim=-1)

    # Quake yaw: forward=(cos,sin), right=(sin,-cos). The lateral sign is the
    # engine's actual AngleVectors convention, not the usual 2-D basis.
    yaw = obs[..., 183] * torch.pi
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    memory_target = torch.stack((
        desired_xy[..., 0] * cos_yaw + desired_xy[..., 1] * sin_yaw,
        desired_xy[..., 0] * sin_yaw - desired_xy[..., 1] * cos_yaw,
    ), dim=-1)
    memory_target = memory_target / torch.linalg.vector_norm(
        memory_target, dim=-1, keepdim=True
    ).clamp_min(1e-6)

    entities = obs[..., ENT_OFF:ENT_OFF + ENT_CNT * ENT_DIM].reshape(
        *obs.shape[:-1], ENT_CNT, ENT_DIM
    )
    candidates = (
        (entities[..., 6] > 0.0)
        & (entities[..., 7] > 0.5)
        & (entities[..., 8] > 0.5)
    )
    visible = candidates.any(dim=-1)
    distance_sq = entities[..., :3].square().sum(dim=-1).masked_fill(
        ~candidates, torch.inf
    )
    nearest_index = distance_sq.argmin(dim=-1)
    gather_index = nearest_index.unsqueeze(-1).unsqueeze(-1).expand(
        *nearest_index.shape, 1, 2
    )
    pursuit_target = entities[..., :2].gather(-2, gather_index).squeeze(-2)
    pursuit_target = pursuit_target / torch.linalg.vector_norm(
        pursuit_target, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    alive = obs[..., 6] > 0.0
    combat_ready = alive & (obs[..., 6] > 0.175) & (obs[..., 9] > 0.0)
    committed = visible & combat_ready
    target = torch.where(committed.unsqueeze(-1), pursuit_target, memory_target)
    mask = alive & (committed | ((strength > 0.03) & ~visible))
    predicted = act_params["cont_mean"][..., :2]
    predicted = predicted / torch.linalg.vector_norm(
        predicted, dim=-1, keepdim=True
    ).clamp_min(0.25)
    cosine = (predicted * target).sum(dim=-1).clamp(-1.0, 1.0)
    memory_weight = strength.clamp(max=1.0)
    # Engagement/opportunity history raises commitment; threat only repels when
    # no live, winnable target is present.
    combat_weight = (
        0.5 + 0.25 * memory[..., 0] + 0.15 * memory[..., 2]
        + 0.10 * memory[..., 21].clamp(min=0.0)
    ).clamp(max=1.0)
    weights = torch.where(committed, combat_weight, memory_weight) * mask.float()
    denominator = weights.sum()
    loss = ((1.0 - cosine) * weights).sum() / denominator.clamp_min(1.0)
    mean_cosine = (cosine * weights).sum() / denominator.clamp_min(1.0)
    return {
        "loss": loss,
        "cosine": mean_cosine,
        "samples": mask.float().sum(),
    }


# ── Rollout buffer ───────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_envs: int, n_steps: int, obs_dim: int,
                 act_dim: int, hidden_dim: int, device: torch.device):
        self.n   = n_envs
        self.T   = n_steps
        self.dev = device

        self.obs      = torch.zeros(n_steps, n_envs, obs_dim,  device=device)
        self.actions  = torch.zeros(n_steps, n_envs, act_dim,  device=device)
        self.rewards  = torch.zeros(n_steps, n_envs,           device=device)
        self.dones    = torch.zeros(n_steps, n_envs,           device=device)
        self.values   = torch.zeros(n_steps, n_envs,           device=device)
        self.log_probs = torch.zeros(n_steps, n_envs,          device=device)
        # A hard action mask is part of the behavior distribution. Preserve
        # the rollout-time decision so PPO evaluates the same distribution.
        self.fire_allowed = torch.ones(
            n_steps, n_envs, dtype=torch.bool, device=device
        )
        self.h_states = torch.zeros(n_steps, n_envs, hidden_dim, device=device)
        self.c_states = torch.zeros(n_steps, n_envs, hidden_dim, device=device)
        self.ptr      = 0

    def add(
        self,
        obs,
        action,
        reward,
        done,
        value,
        log_prob,
        h_state,
        c_state,
        fire_allowed=None,
    ):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        if fire_allowed is None:
            self.fire_allowed[self.ptr] = True
        else:
            self.fire_allowed[self.ptr] = fire_allowed
        self.h_states[self.ptr]  = h_state
        self.c_states[self.ptr]  = c_state
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


def _reconcile_server_fire_suppressions(
    actions: np.ndarray,
    log_probs: np.ndarray,
    fire_allowed: np.ndarray,
    fire_metadata: dict | None,
    step_results,
    *,
    n_ml: int,
) -> int:
    """Replace server-suppressed fire with its exact hard-mask likelihood.

    The network server may invalidate a shot after collection-time inference
    because protection or target state changed. The applied action is no-fire,
    whose log-probability under the resulting closed gate is zero. Remove the
    sampled fire log-probability from the recorded joint likelihood and store
    the closed mask so every later PPO evaluation uses that same distribution.
    """
    suppressions = 0
    for server_index, results in step_results:
        base = server_index * n_ml
        for bot_index, (_obs, _reward, _term, _trunc, info) in enumerate(results):
            if not info.get("fire_gate_suppressed", False):
                continue
            vector_index = base + bot_index
            if (
                fire_metadata is None
                or actions[vector_index, 5] <= 0.5
                or not fire_allowed[vector_index]
            ):
                raise RuntimeError(
                    "server suppressed fire outside the recorded network "
                    "target-gate distribution"
                )
            fire_log_probability = float(
                fire_metadata["raw_fire_log_probability"][vector_index]
            )
            if not np.isfinite(fire_log_probability):
                raise RuntimeError(
                    "server-suppressed fire has a non-finite behavior "
                    "log-probability"
                )
            log_probs[vector_index] -= fire_log_probability
            actions[vector_index, 5] = 0.0
            fire_allowed[vector_index] = False
            suppressions += 1
    return suppressions


def _forward_sequence_with_done_masks(
    policy: Q2BotPolicy,
    obs_b: torch.Tensor,
    hx_b,
    dones_b: torch.Tensor,
):
    """Replay a recurrent chunk while matching rollout-side hidden resets."""
    params_by_key = None
    values = []
    hx = hx_b

    for t in range(obs_b.shape[1]):
        act_params_t, value_t, hx = policy(obs_b[:, t:t + 1, :], hx)
        if params_by_key is None:
            params_by_key = {key: [] for key in act_params_t}
        for key, val in act_params_t.items():
            params_by_key[key].append(val)
        values.append(value_t)

        keep = (1.0 - dones_b[:, t]).view(1, -1, 1)
        hx = (hx[0] * keep, hx[1] * keep)

    act_params = {key: torch.cat(vals, dim=1) for key, vals in params_by_key.items()}
    return act_params, torch.cat(values, dim=1)


def _ppo_parameter_groups(policy: Q2BotPolicy):
    """Return non-overlapping parameter groups used by PPO diagnostics."""
    modules = {
        "shared": (policy.encoder, policy.lstm),
        "actor": (
            policy.actor_cont,
            policy.log_std_head,
            policy.actor_jump,
            policy.actor_hook,
            policy.actor_weapon,
            policy.weapon_embed,
            policy.actor_fire,
        ),
        "critic": (policy.critic,),
        "aux": (policy.predict_next,),
    }
    return {
        name: tuple(param for module in group for param in module.parameters())
        for name, group in modules.items()
    }


def _tensor_group_norm(tensors, device: torch.device) -> torch.Tensor:
    """L2 norm across tensors, ignoring absent gradients."""
    squared = [tensor.detach().float().pow(2).sum()
               for tensor in tensors if tensor is not None]
    if not squared:
        return torch.zeros((), device=device)
    return torch.stack(squared).sum().sqrt()


def _gradient_cosine(grads_a, grads_b, device: torch.device) -> torch.Tensor:
    """Cosine similarity across matching optional gradient tensors."""
    pairs = [(a.detach().float(), b.detach().float())
             for a, b in zip(grads_a, grads_b)
             if a is not None and b is not None]
    if not pairs:
        return torch.zeros((), device=device)
    dot = torch.stack([(a * b).sum() for a, b in pairs]).sum()
    norm_a = torch.stack([a.pow(2).sum() for a, _ in pairs]).sum().sqrt()
    norm_b = torch.stack([b.pow(2).sum() for _, b in pairs]).sum().sqrt()
    denom = norm_a * norm_b
    if float(denom.detach().cpu()) <= 1e-12:
        return torch.zeros((), device=device)
    return dot / denom


def _explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Population explained variance, returning zero for constant targets."""
    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    target_var = target.var(unbiased=False)
    if float(target_var.detach().cpu()) <= 1e-12:
        return torch.zeros((), device=target.device)
    residual_var = (target - prediction).var(unbiased=False)
    return 1.0 - residual_var / target_var


# Observation layout after entities: rays[16*4], hook zones[4*8], audio[5],
# yaw, pitch. Extended observations are appended later and do not shift this.
_AIM_PITCH_OBS_INDEX = ENT_OFF + ENT_CNT * ENT_DIM + 16 * 4 + 4 * 8 + 5 + 1
_POSTURE_DOWNLOOK_DEG = 15.0


def _wrap_degrees_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


@torch.no_grad()
def _aim_anchor_targets(
    obs: torch.Tensor,
    yaw_threshold_deg: float = 12.0,
    pitch_threshold_deg: float = 14.0,
) -> Dict[str, torch.Tensor]:
    """Build exact geometric aim/fire labels from normalized live observations.

    Entity positions are already in Quake's bot-local forward/right/up basis.
    The inverse below matches tools/behavior_clone_aim.py, including Quake's
    yaw/right sign, current-pitch basis, per-tick look limits, and engine pitch
    clamp. Dead terminal observations are excluded from both losses.
    """
    if yaw_threshold_deg < 0 or pitch_threshold_deg < 0:
        raise ValueError("aim anchor alignment thresholds must be nonnegative")

    entities = obs[..., ENT_OFF:ENT_OFF + ENT_CNT * ENT_DIM].reshape(
        *obs.shape[:-1], ENT_CNT, ENT_DIM
    )
    xyz = entities[..., :3] * 4096.0
    candidates = (
        (entities[..., 6] > 0.0)
        & (entities[..., 7] > 0.5)
        & (entities[..., 8] > 0.5)
    )
    has_target = candidates.any(dim=-1)
    distance_sq = xyz.square().sum(dim=-1).masked_fill(
        ~candidates, torch.inf
    )
    nearest_index = distance_sq.argmin(dim=-1)
    gather_index = nearest_index.unsqueeze(-1).unsqueeze(-1).expand(
        *nearest_index.shape, 1, 3
    )
    nearest_xyz = xyz.gather(-2, gather_index).squeeze(-2)
    nearest_xyz = torch.where(
        has_target.unsqueeze(-1), nearest_xyz, torch.zeros_like(nearest_xyz)
    )

    current_pitch = obs[..., _AIM_PITCH_OBS_INDEX] * 90.0
    pitch_rad = current_pitch * (torch.pi / 180.0)
    x, y, z = nearest_xyz.unbind(dim=-1)
    horizontal_forward = torch.cos(pitch_rad) * x + torch.sin(pitch_rad) * z
    vertical = -torch.sin(pitch_rad) * x + torch.cos(pitch_rad) * z
    horizontal_distance = torch.hypot(horizontal_forward, y).clamp_min(1e-3)
    desired_yaw = _wrap_degrees_tensor(
        -torch.atan2(y, horizontal_forward) * (180.0 / torch.pi)
    )
    target_pitch = -torch.atan2(vertical, horizontal_distance) * (
        180.0 / torch.pi
    )
    desired_pitch = target_pitch - current_pitch

    yaw_command = desired_yaw.clamp(-45.0, 45.0)
    pitch_command = desired_pitch.clamp(-30.0, 30.0)
    look_target = torch.stack((yaw_command, pitch_command), dim=-1)
    look_target = torch.where(
        has_target.unsqueeze(-1), look_target, torch.zeros_like(look_target)
    )

    self_alive = obs[..., 6] > 0.0
    look_mask = self_alive & has_target
    new_pitch = (current_pitch + pitch_command).clamp(-89.0, 89.0)
    effective_pitch_command = new_pitch - current_pitch
    yaw_residual = _wrap_degrees_tensor(-desired_yaw + yaw_command)
    pitch_residual = -desired_pitch + effective_pitch_command
    aligned_after_teacher = (
        (yaw_residual.abs() <= float(yaw_threshold_deg))
        & (pitch_residual.abs() <= float(pitch_threshold_deg))
    )
    fire_target = (look_mask & aligned_after_teacher).long()

    return {
        "look_target": look_target,
        "look_mask": look_mask,
        "fire_target": fire_target,
        "fire_mask": self_alive,
        "has_target": has_target,
    }


def _balanced_binary_class_weights(
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Inverse-frequency binary weights, finite for one/no-class batches."""
    valid_targets = targets[mask].long()
    weights = torch.zeros(2, device=targets.device, dtype=torch.float32)
    if valid_targets.numel() == 0:
        return weights
    # torch.bincount has no deterministic CUDA implementation in the PyTorch
    # version on the training box. Binary equality reductions are exact here.
    counts = torch.stack(
        ((valid_targets == 0).sum(), (valid_targets == 1).sum())
    ).to(dtype=torch.float32)
    present = counts > 0
    n_present = present.sum().to(dtype=torch.float32)
    weights[present] = float(valid_targets.numel()) / (
        n_present * counts[present]
    )
    return weights


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _aim_anchor_loss(
    policy: Q2BotPolicy,
    act_params: Dict[str, torch.Tensor],
    recorded_actions: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    fire_class_weights: torch.Tensor,
    look_weight: float,
    fire_weight: float,
) -> Dict[str, torch.Tensor]:
    """Aim/fire loss on the same recurrent features used by PPO."""
    look_mask = targets["look_mask"]
    look_error = act_params["cont_mean"][..., 2:4] - targets["look_target"]
    look_scaled_sq = (look_error / look_error.new_tensor([45.0, 30.0])).square()
    look_loss = _masked_mean(look_scaled_sq, look_mask.unsqueeze(-1).expand_as(look_scaled_sq))

    weapon_idx = recorded_actions[..., 7].round().long().clamp(
        0, WEAPON_CLASSES - 1
    )
    fire_logits = policy.fire_logits_for(act_params["feat"], weapon_idx)
    fire_ce = F.cross_entropy(
        fire_logits.reshape(-1, 2),
        targets["fire_target"].reshape(-1),
        reduction="none",
    ).reshape_as(targets["fire_target"])
    target_weights = fire_class_weights[targets["fire_target"]]
    fire_mask = targets["fire_mask"]
    fire_loss = _masked_mean(fire_ce * target_weights, fire_mask)

    inner = float(look_weight) * look_loss + float(fire_weight) * fire_loss
    with torch.no_grad():
        yaw_mae = _masked_mean(look_error[..., 0].abs(), look_mask)
        pitch_mae = _masked_mean(look_error[..., 1].abs(), look_mask)
        fire_probability = fire_logits.softmax(dim=-1)[..., 1]
        positive = fire_mask & (targets["fire_target"] > 0)
        negative = fire_mask & ~positive
        hidden_negative = negative & ~targets["has_target"]
        fire_accuracy = _masked_mean(
            (fire_logits.argmax(dim=-1) == targets["fire_target"]).float(),
            fire_mask,
        )
        positive_probability = _masked_mean(fire_probability, positive)
        negative_probability = _masked_mean(fire_probability, negative)
        hidden_probability = _masked_mean(fire_probability, hidden_negative)

    return {
        "inner": inner,
        "look": look_loss,
        "fire": fire_loss,
        "yaw_mae_deg": yaw_mae,
        "pitch_mae_deg": pitch_mae,
        "fire_accuracy": fire_accuracy,
        "fire_positive_probability": positive_probability,
        "fire_negative_probability": negative_probability,
        "hidden_fire_probability": hidden_probability,
    }


# ── Device selection ─────────────────────────────────────────────────

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"[warn] CUDA/HIP probe failed ({e!s:.80}), falling back to CPU")
    return torch.device("cpu")


# ── Training loop ────────────────────────────────────────────────────

def train(cfg: dict):
    seed = int(cfg.get("seed", 0))
    deterministic = bool(int(cfg.get("deterministic", 0)))
    if deterministic:
        # Must be set before the first CUDA context is created. This is
        # required by deterministic cuBLAS matrix multiplies on CUDA 10.2+.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _pick_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"Training on: {device}")
    print(f"Random seed: {seed}")
    print(f"Deterministic kernels: {'ON' if deterministic else 'off'}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    policy    = Q2BotPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr"])
    grad_diagnostics = os.environ.get("Q2_PPO_GRAD_DIAGNOSTICS", "0") == "1"
    stateful = os.environ.get("Q2_POLICY_STATEFUL", "1").lower() in {
        "1", "true", "yes", "on"
    }
    aim_anchor_coef = float(cfg.get("aim_anchor_coef", 0.0) or 0.0)
    aim_anchor_look_weight = float(cfg.get("aim_anchor_look_weight", 16.0))
    aim_anchor_fire_weight = float(cfg.get("aim_anchor_fire_weight", 1.0))
    aim_anchor_yaw_deg = float(cfg.get("aim_anchor_yaw_deg", 12.0))
    aim_anchor_pitch_deg = float(cfg.get("aim_anchor_pitch_deg", 14.0))
    target_fire_gate_requested = bool(int(cfg.get("target_fire_gate", 1)))
    lattice_direction_coef = float(cfg.get("lattice_direction_coef", 0.02) or 0.0)
    for name, value in (
        ("aim_anchor_coef", aim_anchor_coef),
        ("aim_anchor_look_weight", aim_anchor_look_weight),
        ("aim_anchor_fire_weight", aim_anchor_fire_weight),
        ("aim_anchor_yaw_deg", aim_anchor_yaw_deg),
        ("aim_anchor_pitch_deg", aim_anchor_pitch_deg),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}")
    if aim_anchor_coef > 0 and not stateful:
        raise ValueError("aim_anchor_coef > 0 requires Q2_POLICY_STATEFUL=1")
    if lattice_direction_coef < 0:
        raise ValueError(
            f"lattice_direction_coef must be nonnegative, got {lattice_direction_coef}"
        )

    print(f"Policy parameters: {policy.param_count():,}")
    if grad_diagnostics:
        print("PPO gradient diagnostics: ON")
    if aim_anchor_coef > 0:
        print(
            "Recurrent aim anchor: ON "
            f"coef={aim_anchor_coef:g} "
            f"look={aim_anchor_look_weight:g} fire={aim_anchor_fire_weight:g}"
        )
    if lattice_direction_coef > 0:
        print(f"Lattice direction objective: ON coef={lattice_direction_coef:g}")

    # Per-run tag isolates parallel runs: separate checkpoint dir + TB run
    # name so stacked ablation trainers never overwrite each other or the
    # resume run. Resume reads from Q2_RESUME_DIR (default = this run's own
    # ckpt dir) so a fresh ablation can warm-start from a shared checkpoint.
    run_tag  = os.environ.get("Q2_RUN_TAG", "").strip()
    ckpt_dir = Path(os.environ.get("Q2_CKPT_DIR",
                    f"checkpoints/{run_tag}" if run_tag else "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_steps = 0
    resume_dir = None
    if cfg.get("resume"):
        resume_dir = Path(os.environ.get("Q2_RESUME_DIR", str(ckpt_dir)))
        candidates = sorted(resume_dir.glob("policy_[0-9]*.pt"))
        if not candidates:
            raise FileNotFoundError(f"--resume requested but no policy_*.pt in {resume_dir}")
        latest = candidates[-1]
        policy.load_state_dict(torch.load(latest, map_location=device))
        # filename is policy_00101376.pt → 101376 trained env steps
        try:
            resume_steps = int(latest.stem.split("_")[-1])
        except ValueError:
            resume_steps = 0
        optimizer_checkpoint = resume_dir / f"optimizer_{resume_steps:08d}.pt"
        if optimizer_checkpoint.is_file():
            optimizer.load_state_dict(torch.load(
                optimizer_checkpoint, map_location=device
            ))
            print(f"Restored optimizer from {optimizer_checkpoint}")
        print(f"Resumed from {latest} (env_steps={resume_steps:,})")

    from harness.env import Q2MultiEnv, discover_map_pool

    try:
        network_client_count = int(os.environ.get("Q2_NETWORK_CLIENTS", "0"))
    except ValueError as error:
        raise ValueError("Q2_NETWORK_CLIENTS must be an integer") from error
    if network_client_count < 0:
        raise ValueError("Q2_NETWORK_CLIENTS cannot be negative")
    network_native = network_client_count > 0
    target_fire_gate = network_native and target_fire_gate_requested

    n_servers         = 1 if network_native else cfg["n_servers"]
    n_bots            = (
        network_client_count if network_native else cfg["n_bots_per_server"]
    )
    n_ml              = (
        network_client_count if network_native else
        max(1, min(int(cfg.get("n_ml_bots", 1)), n_bots))
    )
    total_venvs       = n_servers * n_ml
    map_pool = discover_map_pool(
        map_name=cfg["map_name"],
        map_glob=cfg.get("map_glob", ""),
        map_dir=cfg.get("map_dir", "") or None,
    )
    map_label = cfg["map_name"] if len(map_pool) == 1 else "curriculum"
    print(f"Map pool ({len(map_pool)}): {', '.join(map_pool[:8])}"
          f"{' ...' if len(map_pool) > 8 else ''}")

    distributed = os.environ.get("Q2_DISTRIBUTED_LEARNER", "0") == "1"
    if network_native and distributed:
        raise ValueError(
            "Q2_NETWORK_CLIENTS and Q2_DISTRIBUTED_LEARNER are mutually exclusive"
        )
    rollout_coordinator = None
    rollout_server = None
    publish_distributed_policy = None
    if distributed:
        from harness.rollout_protocol import (
            CoordinatorServer,
            CoordinatorRecoveryConfig,
            PolicyArtifact,
            RolloutCoordinator,
        )

        quorum = max(1, int(os.environ.get("Q2_ROLLOUT_QUORUM", "1")))
        remote_envs = max(1, int(os.environ.get("Q2_ROLLOUT_ENVS_PER_WORKER", "1")))
        total_venvs = quorum * remote_envs
        from harness.runtime_attestation import (
            load_runtime_manifest,
            verify_runtime_manifest,
        )

        runtime_manifest_path = os.environ.get(
            "Q2_ROLLOUT_RUNTIME_MANIFEST", ""
        ).strip()
        if not runtime_manifest_path:
            raise ValueError(
                "Q2_DISTRIBUTED_LEARNER=1 requires "
                "Q2_ROLLOUT_RUNTIME_MANIFEST"
            )
        runtime_manifest = load_runtime_manifest(Path(runtime_manifest_path))
        attestation_key_name = os.environ.get(
            "Q2_ROLLOUT_ATTESTATION_KEY_ENV", ""
        ).strip()
        attestation_key = (
            os.environ[attestation_key_name].encode()
            if attestation_key_name else None
        )
        runtime_verification = verify_runtime_manifest(
            runtime_manifest,
            hmac_key=attestation_key,
            require_signature=bool(attestation_key_name),
        )
        if not runtime_verification.valid:
            raise ValueError(
                "invalid rollout runtime manifest: "
                + "; ".join(runtime_verification.errors)
            )
        runtime_manifest_sha256 = runtime_verification.digest
        recovery_enabled = os.environ.get("Q2_ROLLOUT_RECOVERY", "0") == "1"
        config_payload = json.dumps({
            "ppo": cfg,
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": HIDDEN_DIM,
            "ext_obs": os.environ.get("Q2_EXT_OBS", "0"),
            "rust_lattice": os.environ.get("Q2_RUST_LATTICE", "0"),
            "worker_envs": remote_envs,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "recovery_enabled": recovery_enabled,
        }, sort_keys=True, separators=(",", ":")).encode()
        distributed_config_hash = hashlib.sha256(config_payload).hexdigest()
        recovery_config = None
        if recovery_enabled:
            from harness.distributed_runtime import LearnerLatticeStore

            learner_id = os.environ.get(
                "Q2_ROLLOUT_LEARNER_ID", "q2-ppo-shadow"
            )
            lattice_root = Path(os.environ.get(
                "Q2_ROLLOUT_LATTICE_DIR",
                str(ckpt_dir / "distributed_lattice"),
            ))
            lattice_store = LearnerLatticeStore(
                lattice_root, learner_id, distributed_config_hash
            )
            configured_game_seed = int(cfg.get("game_seed", -1))
            recovery_config = CoordinatorRecoveryConfig(
                learner_id=learner_id,
                steps=int(cfg["n_steps"]),
                n_envs=remote_envs,
                maps=tuple(map_pool),
                base_seed=seed,
                base_game_seed=(
                    configured_game_seed
                    if configured_game_seed >= 0 else seed
                ),
                lease_ttl=float(os.environ.get(
                    "Q2_ROLLOUT_LEASE_TTL", "45"
                )),
                max_attempts=int(os.environ.get(
                    "Q2_ROLLOUT_MAX_ATTEMPTS", "3"
                )),
                lattice_store=lattice_store,
            )
        rollout_coordinator = RolloutCoordinator(
            quorum=quorum,
            schema="ppo",
            expected_runtime_manifest_sha256=runtime_manifest_sha256,
            recovery=recovery_config,
        )
        rollout_server = CoordinatorServer(
            rollout_coordinator,
            os.environ.get("Q2_ROLLOUT_BIND", "0.0.0.0"),
            int(os.environ.get("Q2_ROLLOUT_PORT", "38888")),
            token=os.environ.get("Q2_ROLLOUT_TOKEN", ""),
        ).start()

        def _publish_distributed_policy(version: int) -> None:
            buffer = io.BytesIO()
            cpu_state = {
                name: tensor.detach().cpu()
                for name, tensor in policy.state_dict().items()
            }
            torch.save(cpu_state, buffer)
            rollout_coordinator.publish(PolicyArtifact.create(
                int(version), buffer.getvalue(), distributed_config_hash,
                runtime_manifest_sha256,
            ))

        publish_distributed_policy = _publish_distributed_policy
        publish_distributed_policy(resume_steps)
        print(
            "Distributed learner: "
            f"{rollout_server.address[0]}:{rollout_server.address[1]} "
            f"quorum={quorum} envs/worker={remote_envs} "
            f"policy_version={resume_steps} config={distributed_config_hash[:12]} "
            f"recovery={'on' if recovery_enabled else 'off'}"
        )

    if distributed:
        servers = []
    elif network_native:
        from harness.client_batch import build_network_client_multi_env

        telemetry_token = os.environ.get(
            "Q2_ML_CLIENT_TELEMETRY_TOKEN", ""
        )
        if not telemetry_token:
            raise ValueError(
                "network-native training requires "
                "Q2_ML_CLIENT_TELEMETRY_TOKEN"
            )
        required_network_paths = {
            "Q2_NETWORK_CLIENT_BINARY": os.environ.get(
                "Q2_NETWORK_CLIENT_BINARY", ""
            ),
            "Q2_NETWORK_CLIENT_ROOT": os.environ.get(
                "Q2_NETWORK_CLIENT_ROOT", ""
            ),
        }
        missing_network_paths = [
            name for name, value in required_network_paths.items() if not value
        ]
        if missing_network_paths:
            raise ValueError(
                "network-native training requires "
                + ", ".join(missing_network_paths)
            )
        network_server = os.environ.get(
            "Q2_NETWORK_SERVER", "100.101.57.114:28000"
        )
        telemetry_server = os.environ.get(
            "Q2_NETWORK_TELEMETRY_SERVER", "100.101.57.114:28049"
        )
        client_data_root = os.environ.get(
            "Q2_NETWORK_CLIENT_DATA_ROOT",
            str(Path(required_network_paths["Q2_NETWORK_CLIENT_ROOT"]) / ".ml-clients"),
        )
        servers = [build_network_client_multi_env(
            n_clients=network_client_count,
            server=network_server,
            telemetry_server=telemetry_server,
            telemetry_token=telemetry_token,
            client_binary=required_network_paths["Q2_NETWORK_CLIENT_BINARY"],
            client_root=required_network_paths["Q2_NETWORK_CLIENT_ROOT"],
            client_data_root=client_data_root,
            harness_host=os.environ.get("Q2_NETWORK_HARNESS_HOST", "127.0.0.1"),
            harness_port_base=int(os.environ.get(
                "Q2_NETWORK_HARNESS_PORT_BASE", "39000"
            )),
            qport_base=int(os.environ.get("Q2_NETWORK_QPORT_BASE", "49000")),
            client_id_prefix=os.environ.get(
                "Q2_NETWORK_CLIENT_ID_PREFIX", "public-trainer"
            ),
            name_prefix=os.environ.get("Q2_NETWORK_CLIENT_NAME_PREFIX", "Lattice"),
            client_timeout=float(os.environ.get(
                "Q2_NETWORK_CLIENT_TIMEOUT", "30"
            )),
            round_timeout=float(os.environ.get(
                "Q2_NETWORK_ROUND_TIMEOUT", "2"
            )),
            max_rejected_echoes=int(os.environ.get(
                "Q2_NETWORK_MAX_REJECTED_ECHOES", "16"
            )),
            max_ep_steps=int(cfg["max_ep_steps"]),
            initial_policy_version=resume_steps,
            spatial_seed=seed,
        )]
        print(
            "Network-native clients: "
            f"count={network_client_count} game={network_server} "
            f"telemetry={telemetry_server} policy_version={resume_steps}"
        )
        print(
            "Network target-fire gate: "
            f"{'ON' if target_fire_gate else 'off'} "
            f"yaw<={aim_anchor_yaw_deg:g}deg "
            f"pitch<={aim_anchor_pitch_deg:g}deg"
        )
    else:
        servers = [
            Q2MultiEnv(
                server_id    = i,
                map_name     = cfg["map_name"],
                map_pool     = map_pool,
                map_seed     = int(cfg["map_seed"]),
                game_seed    = (
                    None if int(cfg.get("game_seed", -1)) < 0
                    else int(cfg["game_seed"]) + i * 1009
                ),
                spatial_seed = seed + i * 104729,
                map_change_episodes = int(cfg["map_change_episodes"]),
                n_bots       = n_bots,
                num_ml_bots  = n_ml,
                port_offset  = i,
                max_ep_steps = cfg["max_ep_steps"],
                timelimit    = cfg["timelimit"],
                fraglimit    = cfg["fraglimit"],
                timescale    = float(cfg.get("timescale", 1.0)),
                console_pipe = True,   # live gamemap rotation instead of restarts
            )
            for i in range(n_servers)
        ]
    lattice_instances = [
        sr for server in servers for sr in server._spatial_rewards
    ]
    if resume_dir is not None:
        from harness.spatial import load_lattice_state
        lattice_candidates = (
            resume_dir / f"lattice_{resume_steps:08d}.json.gz",
            resume_dir / "lattice_latest.json.gz",
        )
        lattice_path = next((path for path in lattice_candidates if path.is_file()), None)
        if lattice_path is not None:
            stats = load_lattice_state(lattice_instances, lattice_path)
            print(
                f"Resumed lattice from {lattice_path} "
                f"({stats['cells']:,} cells / {stats['instances']} instances)"
            )
        else:
            print("No lattice checkpoint found; maps will start from sidecar priors")

    def _close_all_servers() -> None:
        for server in servers:
            server.close()
        if rollout_server is not None:
            rollout_server.close()

    # An optimizer/assertion failure used to orphan q2ded children and leave
    # their UDP ports occupied for the next experiment. Normal completion
    # unregisters this after performing the same cleanup explicitly.
    atexit.register(_close_all_servers)

    buf = RolloutBuffer(total_venvs, cfg["n_steps"], OBS_DIM, ACTION_DIM, HIDDEN_DIM, device)

    obs_np  = np.zeros((total_venvs, OBS_DIM), dtype=np.float32)
    hx_list = [policy.init_hidden(1, device) for _ in range(total_venvs)]

    # reset all servers in parallel and populate initial obs (sequential boot
    # cost ~50s/server; each env owns its own ports, sockets, and process)
    if not distributed:
        with ThreadPoolExecutor(max_workers=n_servers) as ex:
            all_initial = list(ex.map(lambda srv: srv.reset_all(), servers))
        for si, obs_list in enumerate(all_initial):
            for bi, o in enumerate(obs_list):
                obs_np[si * n_ml + bi] = o

    total_env_steps    = resume_steps
    _directive_seqs    = {}
    update_n           = 0
    next_save_at       = ((resume_steps // cfg["save_every"]) + 1) * cfg["save_every"]
    save_dir           = ckpt_dir

    # Entropy-guarded curriculum evolution (pool-level churn). Opt-in per run;
    # isolated by map prefix so it never touches maps other runs glob. See
    # train/curriculum.py for the absorb-for-entropy-not-convergence rationale.
    evolver = None
    if (not distributed and not network_native and
            os.environ.get("Q2_CURRICULUM_EVOLVE") == "1"):
        try:
            from train.curriculum import CurriculumEvolver
            _pref = os.environ.get("Q2_CURRICULUM_PREFIX") or (
                map_pool[0].rsplit("_", 1)[0] if map_pool else "mlcur")
            evolver = CurriculumEvolver(
                map_prefix=_pref,
                max_churn=int(os.environ.get("Q2_CURRICULUM_MAX_CHURN", "1")),
                mastery_kd=float(os.environ.get("Q2_CURRICULUM_MASTERY_KD", "2.5")),
                min_samples=float(os.environ.get("Q2_CURRICULUM_MIN_SAMPLES", "40")),
                min_pool=int(os.environ.get("Q2_CURRICULUM_MIN_POOL", "8")))
            print(f"  curriculum evolution ON (prefix={_pref}, "
                  f"churn<={evolver.max_churn}/cycle, mastery_kd>={evolver.mastery_kd})")
        except Exception as e:
            print(f"  ! curriculum init failed (continuing without): {e}")

    run_name = (f"ppo_{run_tag}_{int(time.time())}" if run_tag
                else f"ppo_{_safe_tag(map_label)}_{int(time.time())}")
    runs_dir = Path(os.environ.get("Q2_RUNS_DIR", "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=str(runs_dir / run_name))
    writer.add_scalar("config/seed", seed, resume_steps)
    print(f"TensorBoard log: {runs_dir / run_name}")
    print(f"  view with:  tensorboard --logdir runs --bind_all")
    if distributed:
        print(f"  virtual envs: {total_venvs} from distributed rollout quorum")
    elif network_native:
        print(
            f"  virtual envs: {total_venvs} normal-player clients on the "
            "public network-native lane"
        )
    else:
        print(
            f"  virtual envs: {total_venvs}  "
            f"({n_servers} servers × {n_ml} ML bot(s) + "
            f"{n_bots - n_ml} AI opponents each)"
        )

    ep_rewards           = np.zeros(total_venvs, dtype=np.float64)
    ep_base_rewards      = np.zeros(total_venvs, dtype=np.float64)
    ep_spatial_rewards   = np.zeros(total_venvs, dtype=np.float64)
    ep_kills             = np.zeros(total_venvs, dtype=np.float64)
    ep_deaths            = np.zeros(total_venvs, dtype=np.float64)
    ep_lengths           = np.zeros(total_venvs, dtype=np.int64)
    completed_ep_rewards = []
    completed_ep_base_rewards = []
    completed_ep_spatial_rewards = []
    completed_ep_kills = []
    completed_ep_deaths = []
    completed_ep_lengths = []
    completed_map_rewards: Dict[str, List[float]] = {}

    start = time.time()
    behavior_metric_keys = PPO_BEHAVIOR_METRIC_KEYS

    while total_env_steps < cfg["total_steps"]:
        # ── collect rollout ──────────────────────────────────────────
        policy.eval()
        rollout_behavior = {key: 0.0 for key in behavior_metric_keys}
        rollout_behavior_samples = 0
        rollout_fire_gate_samples = 0
        rollout_fire_gate_allowed = 0
        rollout_fire_gate_executed = 0
        rollout_fire_gate_blocked_probability = 0.0
        rollout_posture_samples = 0
        rollout_signed_forward = 0.0
        rollout_forward_commands = 0
        rollout_backward_commands = 0
        rollout_look_pitch = 0.0
        rollout_look_pitch_abs = 0.0
        rollout_view_pitch = 0.0
        rollout_view_pitch_abs = 0.0
        rollout_downlook_frames = 0

        accepted_rollout_steps = 0
        target_rollout_steps = 0 if distributed else int(cfg["n_steps"])
        while accepted_rollout_steps < target_rollout_steps:
            # Store the current observation which produced the actions!
            current_obs = obs_np.copy()

            # Stack current hidden states before act_batch updates them
            if stateful:
                h_step = torch.cat([hx[0] for hx in hx_list], dim=1).squeeze(0) # (total_venvs, HIDDEN_DIM)
                c_step = torch.cat([hx[1] for hx in hx_list], dim=1).squeeze(0) # (total_venvs, HIDDEN_DIM)
            else:
                h_step = torch.zeros(total_venvs, HIDDEN_DIM, device=device)
                c_step = torch.zeros(total_venvs, HIDDEN_DIM, device=device)

            # Batched policy inference — 1 GPU call instead of total_venvs calls
            fire_metadata = None
            if network_native:
                (
                    actions_np,
                    values_np,
                    lps_np,
                    hx_list,
                    fire_metadata,
                ) = policy.act_batch(
                    obs_np,
                    hx_list,
                    device,
                    gate_fire=target_fire_gate,
                    fire_gate_yaw_deg=aim_anchor_yaw_deg,
                    fire_gate_pitch_deg=aim_anchor_pitch_deg,
                    return_fire_metadata=True,
                )
                fire_allowed_np = fire_metadata["fire_allowed"]
            else:
                actions_np, values_np, lps_np, hx_list = policy.act_batch(
                    obs_np, hx_list, device
                )
                fire_allowed_np = np.ones(total_venvs, dtype=np.bool_)

            rewards_np = np.zeros(total_venvs, dtype=np.float32)
            dones_np   = np.zeros(total_venvs, dtype=np.float32)

            # Parallel server stepping — UDP recv releases the GIL
            def _step_server(si_srv):
                si, srv = si_srv
                base = si * n_ml
                actions = [actions_np[base + k] for k in range(srv.n_ml)]
                if network_native:
                    return si, srv.step_all(
                        actions, policy_version=total_env_steps
                    )
                return si, srv.step_all(actions)

            with ThreadPoolExecutor(max_workers=n_servers) as ex:
                step_results = list(ex.map(_step_server, enumerate(servers)))

            # A live gamemap resets the authoritative server-frame epoch.  The
            # client batch returns the first observation from the new map as a
            # synchronization boundary, not as a transition from the old map.
            # Reset recurrent/episode memory and recollect this rollout slot so
            # cross-map rewards and actions can never enter PPO.
            if network_native:
                trainable_flags = [
                    bool(info.get("trainable_transition", False))
                    for _si, results in step_results
                    for _o, _r, _term, _trunc, info in results
                ]
                if not all(trainable_flags):
                    if any(trainable_flags):
                        raise RuntimeError(
                            "network collector returned a partially trainable "
                            "map-epoch round"
                        )
                    target_maps = set()
                    preflight_packets_drained = 0
                    map_epoch_resync = False
                    for si, results in step_results:
                        base = si * n_ml
                        for bi, (o, _r, _term, _trunc, info) in enumerate(results):
                            vi = base + bi
                            obs_np[vi] = o
                            hx_list[vi] = policy.init_hidden(1, device)
                            target_maps.add(str(info.get("map", "unknown")))
                            preflight_packets_drained += int(
                                info.get("preflight_packets_drained", 0)
                            )
                            map_epoch_resync |= bool(
                                info.get("map_epoch_resync", False)
                            )
                            ep_rewards[vi] = 0.0
                            ep_base_rewards[vi] = 0.0
                            ep_spatial_rewards[vi] = 0.0
                            ep_kills[vi] = 0.0
                            ep_deaths[vi] = 0.0
                            ep_lengths[vi] = 0
                    print(
                        "Network client synchronization boundary: "
                        f"kind={'map-epoch' if map_epoch_resync else 'realtime-catchup'} "
                        f"maps={','.join(sorted(target_maps))} "
                        f"drained={preflight_packets_drained}"
                    )
                    continue

                # The server independently rejects a shot when protection or
                # last-moment world-state changes invalidate the policy mask.
                # Reconcile that authoritative shield into the stored action
                # and behavior log-probability so PPO never learns from an
                # action the game did not execute. This remains required when
                # the proactive policy gate is disabled for an ablation.
                _reconcile_server_fire_suppressions(
                    actions_np,
                    lps_np,
                    fire_allowed_np,
                    fire_metadata,
                    step_results,
                    n_ml=n_ml,
                )

                # Explicit signed posture telemetry. The historical
                # forward-intent metric used abs(move_forward), which hid a
                # backwards moonwalk behind a healthy-looking value.
                effective_forward = np.clip(actions_np[:, 0], -1.0, 1.0)
                effective_look_pitch = np.clip(actions_np[:, 3], -30.0, 30.0)
                view_pitch = current_obs[:, _AIM_PITCH_OBS_INDEX] * 90.0
                rollout_posture_samples += int(effective_forward.size)
                rollout_signed_forward += float(effective_forward.sum())
                rollout_forward_commands += int((effective_forward > 0.15).sum())
                rollout_backward_commands += int((effective_forward < -0.15).sum())
                rollout_look_pitch += float(effective_look_pitch.sum())
                rollout_look_pitch_abs += float(
                    np.abs(effective_look_pitch).sum()
                )
                rollout_view_pitch += float(view_pitch.sum())
                rollout_view_pitch_abs += float(np.abs(view_pitch).sum())
                # Quake pitch is positive when looking down.
                rollout_downlook_frames += int(
                    (view_pitch > _POSTURE_DOWNLOOK_DEG).sum()
                )

            if target_fire_gate:
                rollout_fire_gate_samples += int(fire_allowed_np.size)
                rollout_fire_gate_allowed += int(fire_allowed_np.sum())
                rollout_fire_gate_executed += int(
                    (actions_np[:, 5] > 0.5).sum()
                )
                closed = ~fire_allowed_np
                rollout_fire_gate_blocked_probability += float(
                    fire_metadata["raw_fire_probability"][closed].sum()
                )

            for si, results in step_results:
                base = si * n_ml
                srv = servers[si]
                for bi, (o, r, term, trunc, info) in enumerate(results):
                    vi = base + bi
                    if network_native and (
                        not info.get("trainable_transition", False)
                        or int(info.get("policy_version", -1)) != total_env_steps
                    ):
                        raise RuntimeError(
                            "network collector returned an unversioned or "
                            "non-trainable transition"
                        )
                    spatial_bonus = float(info.get("spatial_bonus", 0.0))
                    base_reward = float(info.get("reward_base", r - spatial_bonus))
                    kills = float(info.get("kills", 0.0))
                    deaths = float(info.get("deaths", 0.0))
                    for key in behavior_metric_keys:
                        rollout_behavior[key] += float(info.get(key, 0.0))
                    rollout_behavior_samples += 1
                    obs_np[vi]      = o
                    rewards_np[vi]  = r
                    dones_np[vi]    = float(term or trunc)
                    ep_rewards[vi] += r
                    ep_base_rewards[vi] += base_reward
                    ep_spatial_rewards[vi] += spatial_bonus
                    ep_kills[vi] += kills
                    ep_deaths[vi] += deaths
                    ep_lengths[vi] += 1
                    if term or trunc:
                        completed_ep_rewards.append(float(ep_rewards[vi]))
                        completed_ep_base_rewards.append(float(ep_base_rewards[vi]))
                        completed_ep_spatial_rewards.append(float(ep_spatial_rewards[vi]))
                        completed_ep_kills.append(float(ep_kills[vi]))
                        completed_ep_deaths.append(float(ep_deaths[vi]))
                        completed_ep_lengths.append(int(ep_lengths[vi]))
                        completed_map_rewards.setdefault(
                            str(info.get("map", "unknown")), []
                        ).append(float(ep_rewards[vi]))
                        ep_rewards[vi] = 0.0
                        ep_base_rewards[vi] = 0.0
                        ep_spatial_rewards[vi] = 0.0
                        ep_kills[vi] = 0.0
                        ep_deaths[vi] = 0.0
                        ep_lengths[vi] = 0
                        obs_np[vi]  = srv.reset_slot(bi)
                        hx_list[vi] = policy.init_hidden(1, device)

            # Use torch.from_numpy to avoid CPU-GPU transfer bottlenecks and copies
            buf.add(
                torch.from_numpy(current_obs).to(device, dtype=torch.float32),
                torch.from_numpy(actions_np).to(device, dtype=torch.float32),
                torch.from_numpy(rewards_np).to(device, dtype=torch.float32),
                torch.from_numpy(dones_np).to(device, dtype=torch.float32),
                torch.from_numpy(values_np).to(device, dtype=torch.float32),
                torch.from_numpy(lps_np).to(device, dtype=torch.float32),
                h_step,
                c_step,
                torch.from_numpy(fire_allowed_np).to(
                    device, dtype=torch.bool
                ),
            )
            accepted_rollout_steps += 1

        if distributed:
            from harness.rollout_protocol import merge_ppo_batches

            policy_version = rollout_coordinator.policy().version
            batches = rollout_coordinator.wait_for_quorum(
                policy_version,
                float(os.environ.get("Q2_ROLLOUT_TIMEOUT", "600")),
            )
            if not batches:
                raise TimeoutError(
                    f"distributed rollout quorum timed out for policy {policy_version}"
                )
            merged = merge_ppo_batches(batches)
            expected_obs = (cfg["n_steps"], total_venvs, OBS_DIM)
            if merged["obs"].shape != expected_obs:
                raise ValueError(
                    f"distributed rollout shape {merged['obs'].shape} != {expected_obs}"
                )
            for target, name in (
                (buf.obs, "obs"),
                (buf.actions, "actions"),
                (buf.rewards, "rewards"),
                (buf.dones, "dones"),
                (buf.values, "values"),
                (buf.log_probs, "log_probs"),
                (buf.h_states, "h_states"),
                (buf.c_states, "c_states"),
            ):
                target.copy_(torch.from_numpy(merged[name]).to(
                    device, dtype=target.dtype
                ))
            episode_columns = {
                name: index
                for index, name in enumerate(PPO_EPISODE_SUMMARY_COLUMNS)
            }
            episode_summaries = merged["episode_summaries"]
            completed_ep_rewards.extend(
                episode_summaries[:, episode_columns["reward"]].tolist()
            )
            completed_ep_base_rewards.extend(
                episode_summaries[:, episode_columns["base_reward"]].tolist()
            )
            completed_ep_spatial_rewards.extend(
                episode_summaries[:, episode_columns["spatial_reward"]].tolist()
            )
            completed_ep_kills.extend(
                episode_summaries[:, episode_columns["kills"]].tolist()
            )
            completed_ep_deaths.extend(
                episode_summaries[:, episode_columns["deaths"]].tolist()
            )
            completed_ep_lengths.extend(
                int(value)
                for value in episode_summaries[:, episode_columns["length"]]
            )
            for map_name, reward in zip(
                merged["episode_map_names"],
                episode_summaries[:, episode_columns["reward"]],
            ):
                completed_map_rewards.setdefault(str(map_name), []).append(
                    float(reward)
                )
            rollout_behavior.update(zip(
                behavior_metric_keys,
                merged["behavior_sums"].tolist(),
            ))
            rollout_behavior_samples = int(merged["behavior_samples"][0])
            total_env_steps += cfg["n_steps"] * total_venvs
            with torch.no_grad():
                obs_t = torch.from_numpy(merged["last_obs"]).to(
                    device, dtype=torch.float32
                ).unsqueeze(1)
                h_stack = torch.from_numpy(merged["last_h"]).to(
                    device, dtype=torch.float32
                ).unsqueeze(0)
                c_stack = torch.from_numpy(merged["last_c"]).to(
                    device, dtype=torch.float32
                ).unsqueeze(0)
                _, values_t, _ = policy(obs_t, (h_stack, c_stack))
                last_vals = values_t.squeeze(-1).squeeze(-1)
        else:
            total_env_steps += cfg["n_steps"] * total_venvs

            # compute bootstrap value
            with torch.no_grad():
                # Batched bootstrap computation to run in a single GPU pass
                if os.environ.get("Q2_POLICY_STATEFUL", "1").lower() in {"1", "true", "yes", "on"}:
                    h_stack = torch.cat([hx[0] for hx in hx_list], dim=1)
                    c_stack = torch.cat([hx[1] for hx in hx_list], dim=1)
                else:
                    h_stack, c_stack = policy.init_hidden(total_venvs, device)
                obs_t = torch.from_numpy(obs_np).to(
                    device, dtype=torch.float32
                ).unsqueeze(1)
                _, values_t, _ = policy(obs_t, (h_stack, c_stack))
                last_vals = values_t.squeeze(-1).squeeze(-1)

        buf.compute_returns(last_vals, cfg["gamma"], cfg["gae_lambda"])

        # ── PPO update ───────────────────────────────────────────────
        policy.train()
        update_n += 1

        # Reshape rollout buffers into contiguous chunks to preserve temporal structure
        T_steps, N_envs, _ = buf.obs.shape
        chunk_len = cfg.get("chunk_len", 16) if stateful else 1
        assert T_steps % chunk_len == 0, f"n_steps ({T_steps}) must be divisible by chunk_len ({chunk_len})"
        num_chunks_per_env = T_steps // chunk_len
        total_chunks = num_chunks_per_env * N_envs

        # Rearrange from (T, N, ...) -> (num_chunks, chunk_len, N, ...) -> (num_chunks, N, chunk_len, ...) -> (total_chunks, chunk_len, ...)
        obs_chunked = buf.obs.reshape(num_chunks_per_env, chunk_len, N_envs, OBS_DIM).permute(0, 2, 1, 3).reshape(total_chunks, chunk_len, OBS_DIM)
        act_chunked = buf.actions.reshape(num_chunks_per_env, chunk_len, N_envs, ACTION_DIM).permute(0, 2, 1, 3).reshape(total_chunks, chunk_len, ACTION_DIM)
        adv_chunked = buf.advantages.reshape(num_chunks_per_env, chunk_len, N_envs).permute(0, 2, 1).reshape(total_chunks, chunk_len)
        ret_chunked = buf.returns.reshape(num_chunks_per_env, chunk_len, N_envs).permute(0, 2, 1).reshape(total_chunks, chunk_len)
        log_probs_chunked = buf.log_probs.reshape(num_chunks_per_env, chunk_len, N_envs).permute(0, 2, 1).reshape(total_chunks, chunk_len)
        fire_allowed_chunked = buf.fire_allowed.reshape(num_chunks_per_env, chunk_len, N_envs).permute(0, 2, 1).reshape(total_chunks, chunk_len)
        dones_chunked = buf.dones.reshape(num_chunks_per_env, chunk_len, N_envs).permute(0, 2, 1).reshape(total_chunks, chunk_len)

        aim_anchor_targets = None
        aim_anchor_fire_class_weights = torch.zeros(2, device=device)
        aim_anchor_target_stats = None
        if aim_anchor_coef > 0:
            aim_anchor_targets = _aim_anchor_targets(
                obs_chunked,
                yaw_threshold_deg=aim_anchor_yaw_deg,
                pitch_threshold_deg=aim_anchor_pitch_deg,
            )
            aim_anchor_fire_class_weights = _balanced_binary_class_weights(
                aim_anchor_targets["fire_target"],
                aim_anchor_targets["fire_mask"],
            )
            fire_valid = aim_anchor_targets["fire_mask"].float()
            fire_denom = fire_valid.sum().clamp_min(1.0)
            aim_anchor_target_stats = {
                "eligible_alive_rate": fire_valid.mean(),
                "visible_alive_rate": (
                    aim_anchor_targets["look_mask"].float().sum() / fire_denom
                ),
                "fire_positive_rate": (
                    aim_anchor_targets["fire_target"].float().sum() / fire_denom
                ),
                "fire_class_weight_negative": aim_anchor_fire_class_weights[0],
                "fire_class_weight_positive": aim_anchor_fire_class_weights[1],
            }

        # Slice the recorded hidden states to get the initial state for each chunk
        chunk_starts = np.arange(0, T_steps, chunk_len)
        h_chunk_starts = buf.h_states[chunk_starts].reshape(total_chunks, HIDDEN_DIM)
        c_chunk_starts = buf.c_states[chunk_starts].reshape(total_chunks, HIDDEN_DIM)

        chunks_per_batch = max(1, cfg["batch_size"] // chunk_len)
        aux_coef = float(cfg.get("aux_coef", 0.05) or 0.0)
        if aim_anchor_coef > 0 and chunks_per_batch < total_chunks:
            raise ValueError(
                "recurrent aim anchor currently requires one full-rollout "
                "minibatch so masked look/fire losses retain exact weighting; "
                f"need batch_size >= {total_chunks * chunk_len}"
            )

        diagnostic_metric_names = (
            "component_grad_norm_policy_weighted",
            "component_grad_norm_vf_weighted",
            "component_grad_norm_aux_weighted",
            "component_grad_norm_aim_anchor_weighted",
            "shared_grad_cos_policy_vf",
            "shared_grad_cos_policy_aim_anchor",
            "grad_norm_shared_pre_clip",
            "grad_norm_actor_pre_clip",
            "grad_norm_critic_pre_clip",
            "grad_norm_aux_pre_clip",
            "grad_clip_scale",
        )
        diagnostic_totals = None
        diagnostic_parameter_groups = None
        diagnostic_parameter_snapshots = None
        diagnostic_all_parameters = None
        diagnostic_shared_indices = None
        explained_variance_pre = None
        if grad_diagnostics:
            diagnostic_totals = torch.zeros(
                len(diagnostic_metric_names), device=device, dtype=torch.float64
            )
            diagnostic_parameter_groups = _ppo_parameter_groups(policy)
            diagnostic_all_parameters = tuple(policy.parameters())
            parameter_indices = {
                id(param): index
                for index, param in enumerate(diagnostic_all_parameters)
            }
            diagnostic_shared_indices = tuple(
                parameter_indices[id(param)]
                for param in diagnostic_parameter_groups["shared"]
            )
            diagnostic_parameter_snapshots = {
                name: tuple(param.detach().clone() for param in params)
                for name, params in diagnostic_parameter_groups.items()
            }
            explained_variance_pre = _explained_variance(
                buf.values, buf.returns
            )

        optimization_metric_names = (
            "loss",
            "pg_loss",
            "vf_loss",
            "entropy",
            "entropy_loss",
            "aux_loss",
            "lattice_direction_loss",
            "lattice_direction_weighted_loss",
            "lattice_direction_cosine",
            "lattice_direction_samples",
            "aim_anchor_inner_loss",
            "aim_anchor_weighted_loss",
            "aim_anchor_look_loss",
            "aim_anchor_fire_loss",
            "aim_anchor_yaw_mae_deg",
            "aim_anchor_pitch_mae_deg",
            "aim_anchor_fire_accuracy",
            "aim_anchor_fire_positive_probability",
            "aim_anchor_fire_negative_probability",
            "aim_anchor_hidden_fire_probability",
            "approx_kl",
            "clip_fraction",
            "grad_norm_pre_clip",
        )
        # Accumulate diagnostics on-device and synchronize once per PPO
        # update. Calling .item() for every metric in every minibatch adds
        # enough GPU barriers to measurably slow the main training loop.
        optimization_totals = torch.zeros(
            len(optimization_metric_names), device=device, dtype=torch.float64
        )
        optimization_minibatches = 0
        for epoch in range(cfg["n_epochs"]):
            idx = torch.randperm(total_chunks, device=device)
            for start_i in range(0, total_chunks, chunks_per_batch):
                b = idx[start_i:start_i + chunks_per_batch]
                obs_b = obs_chunked[b]   # (B_chunks, chunk_len, OBS_DIM)
                act_b = act_chunked[b]   # (B_chunks, chunk_len, ACTION_DIM)

                if stateful:
                    h_b = h_chunk_starts[b] # (B_chunks, HIDDEN_DIM)
                    c_b = c_chunk_starts[b] # (B_chunks, HIDDEN_DIM)
                    hx_b = (h_b.unsqueeze(0), c_b.unsqueeze(0)) # (1, B_chunks, HIDDEN_DIM)
                    act_params, val_b = _forward_sequence_with_done_masks(
                        policy, obs_b, hx_b, dones_chunked[b]
                    )
                else:
                    act_params, val_b, _ = policy(obs_b)
                val_b = val_b.squeeze(-1) # (B_chunks, chunk_len)

                # Compute mixed-action log probabilities and entropy using policy's built-in function
                log_prob, entropy = policy.action_log_prob_entropy(
                    act_params,
                    act_b,
                    fire_allowed=fire_allowed_chunked[b],
                )
                log_prob = log_prob.reshape(-1)
                entropy = entropy.reshape(-1)

                old_log_probs = log_probs_chunked[b].reshape(-1)
                log_ratio = log_prob - old_log_probs
                ratio = log_ratio.exp()

                # Diagnostics only: these tensors are detached from the
                # optimization graph and do not change the PPO objective.
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > cfg["clip_eps"]
                    ).float().mean()
                
                adv_flat_b = adv_chunked[b].reshape(-1)
                pg_loss = -torch.min(
                    ratio * adv_flat_b,
                    ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_flat_b,
                ).mean()

                ret_flat_b = ret_chunked[b].reshape(-1)
                vf_loss  = ((val_b.reshape(-1) - ret_flat_b) ** 2).mean()
                ent_loss = -entropy.mean()

                # auxiliary world-model loss: predict the next obs from the
                # LSTM features. Masked at episode boundaries; needs temporal
                # chunks (chunk_len > 1) to have a "next" step to predict.
                aux_loss = torch.zeros((), device=device)
                if chunk_len > 1:
                    pred   = policy.predict_next(act_params["feat"][:, :-1])
                    with torch.no_grad():
                        target = policy.encoder(obs_b[:, 1:])
                    keep   = (1.0 - dones_chunked[b][:, :-1]).unsqueeze(-1)
                    denom  = keep.sum() * HIDDEN_DIM
                    if denom > 0:
                        aux_loss = (((pred - target) ** 2) * keep).sum() / denom

                anchor_zero = torch.zeros((), device=device)
                aim_anchor_metrics = {
                    "inner": anchor_zero,
                    "look": anchor_zero,
                    "fire": anchor_zero,
                    "yaw_mae_deg": anchor_zero,
                    "pitch_mae_deg": anchor_zero,
                    "fire_accuracy": anchor_zero,
                    "fire_positive_probability": anchor_zero,
                    "fire_negative_probability": anchor_zero,
                    "hidden_fire_probability": anchor_zero,
                }
                if aim_anchor_coef > 0:
                    aim_anchor_metrics = _aim_anchor_loss(
                        policy,
                        act_params,
                        act_b,
                        {key: value[b] for key, value in aim_anchor_targets.items()},
                        aim_anchor_fire_class_weights,
                        look_weight=aim_anchor_look_weight,
                        fire_weight=aim_anchor_fire_weight,
                    )
                aim_anchor_component = aim_anchor_coef * aim_anchor_metrics["inner"]
                lattice_direction_metrics = _lattice_direction_loss(act_params, obs_b)
                lattice_direction_component = (
                    lattice_direction_coef * lattice_direction_metrics["loss"]
                )

                loss = (pg_loss + cfg["vf_coef"] * vf_loss + cfg["ent_coef"] * ent_loss
                        + aux_coef * aux_loss + aim_anchor_component
                        + lattice_direction_component)

                diagnostic_component_values = None
                if grad_diagnostics:
                    policy_component = pg_loss + cfg["ent_coef"] * ent_loss
                    vf_component = cfg["vf_coef"] * vf_loss
                    aux_component = aux_coef * aux_loss
                    policy_component_grads = torch.autograd.grad(
                        policy_component,
                        diagnostic_all_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    vf_component_grads = torch.autograd.grad(
                        vf_component,
                        diagnostic_all_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if aux_component.requires_grad:
                        aux_component_grads = torch.autograd.grad(
                            aux_component,
                            diagnostic_all_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    else:
                        aux_component_grads = (None,) * len(
                            diagnostic_all_parameters
                        )
                    if aim_anchor_component.requires_grad:
                        aim_anchor_component_grads = torch.autograd.grad(
                            aim_anchor_component,
                            diagnostic_all_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    else:
                        aim_anchor_component_grads = (None,) * len(
                            diagnostic_all_parameters
                        )
                    policy_shared_grads = tuple(
                        policy_component_grads[index]
                        for index in diagnostic_shared_indices
                    )
                    vf_shared_grads = tuple(
                        vf_component_grads[index]
                        for index in diagnostic_shared_indices
                    )
                    aim_anchor_shared_grads = tuple(
                        aim_anchor_component_grads[index]
                        for index in diagnostic_shared_indices
                    )
                    diagnostic_component_values = (
                        _tensor_group_norm(policy_component_grads, device),
                        _tensor_group_norm(vf_component_grads, device),
                        _tensor_group_norm(aux_component_grads, device),
                        _tensor_group_norm(aim_anchor_component_grads, device),
                        _gradient_cosine(
                            policy_shared_grads, vf_shared_grads, device
                        ),
                        _gradient_cosine(
                            policy_shared_grads,
                            aim_anchor_shared_grads,
                            device,
                        ),
                    )

                optimizer.zero_grad()
                loss.backward()
                diagnostic_group_norms = None
                if grad_diagnostics:
                    diagnostic_group_norms = tuple(
                        _tensor_group_norm(
                            (param.grad for param in
                             diagnostic_parameter_groups[name]),
                            device,
                        )
                        for name in ("shared", "actor", "critic", "aux")
                    )
                grad_norm_pre_clip = nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg["max_grad_norm"]
                )
                grad_norm_pre_clip_tensor = torch.as_tensor(
                    grad_norm_pre_clip, device=device
                ).detach()
                grad_clip_scale = None
                if grad_diagnostics:
                    grad_clip_scale = torch.clamp(
                        float(cfg["max_grad_norm"])
                        / (grad_norm_pre_clip_tensor + 1e-6),
                        max=1.0,
                    )
                optimizer.step()
                if grad_diagnostics:
                    diagnostic_totals += torch.stack(
                        diagnostic_component_values
                        + diagnostic_group_norms
                        + (grad_clip_scale,)
                    ).to(dtype=diagnostic_totals.dtype)
                with torch.no_grad():
                    optimization_totals += torch.stack(
                        (
                            loss.detach(),
                            pg_loss.detach(),
                            vf_loss.detach(),
                            entropy.mean().detach(),
                            ent_loss.detach(),
                            aux_loss.detach(),
                            lattice_direction_metrics["loss"].detach(),
                            lattice_direction_component.detach(),
                            lattice_direction_metrics["cosine"].detach(),
                            lattice_direction_metrics["samples"].detach(),
                            aim_anchor_metrics["inner"].detach(),
                            aim_anchor_component.detach(),
                            aim_anchor_metrics["look"].detach(),
                            aim_anchor_metrics["fire"].detach(),
                            aim_anchor_metrics["yaw_mae_deg"].detach(),
                            aim_anchor_metrics["pitch_mae_deg"].detach(),
                            aim_anchor_metrics["fire_accuracy"].detach(),
                            aim_anchor_metrics[
                                "fire_positive_probability"
                            ].detach(),
                            aim_anchor_metrics[
                                "fire_negative_probability"
                            ].detach(),
                            aim_anchor_metrics[
                                "hidden_fire_probability"
                            ].detach(),
                            approx_kl.detach(),
                            clip_fraction.detach(),
                            grad_norm_pre_clip_tensor,
                        )
                    ).to(dtype=optimization_totals.dtype)
                optimization_minibatches += 1

        optimization_denom = float(max(1, optimization_minibatches))
        optimization_values = (
            optimization_totals / optimization_denom
        ).detach().cpu().tolist()
        optimization_means = dict(zip(optimization_metric_names, optimization_values))

        diagnostic_values = {}
        if grad_diagnostics:
            post_kl_sum = torch.zeros((), device=device)
            post_clip_count = torch.zeros((), device=device)
            post_sample_count = 0
            post_values = []
            post_returns = []
            was_training = policy.training
            policy.eval()
            with torch.no_grad():
                for start_i in range(0, total_chunks, chunks_per_batch):
                    end_i = min(start_i + chunks_per_batch, total_chunks)
                    obs_post = obs_chunked[start_i:end_i]
                    act_post = act_chunked[start_i:end_i]
                    if stateful:
                        h_post = h_chunk_starts[start_i:end_i]
                        c_post = c_chunk_starts[start_i:end_i]
                        params_post, values_post = _forward_sequence_with_done_masks(
                            policy,
                            obs_post,
                            (h_post.unsqueeze(0), c_post.unsqueeze(0)),
                            dones_chunked[start_i:end_i],
                        )
                    else:
                        params_post, values_post, _ = policy(obs_post)
                    log_prob_post, _entropy_post = policy.action_log_prob_entropy(
                        params_post,
                        act_post,
                        fire_allowed=fire_allowed_chunked[start_i:end_i],
                    )
                    log_ratio_post = (
                        log_prob_post.reshape(-1)
                        - log_probs_chunked[start_i:end_i].reshape(-1)
                    )
                    ratio_post = log_ratio_post.exp()
                    post_kl_sum += (
                        (ratio_post - 1.0) - log_ratio_post
                    ).sum()
                    post_clip_count += (
                        (ratio_post - 1.0).abs() > cfg["clip_eps"]
                    ).float().sum()
                    post_sample_count += int(log_ratio_post.numel())
                    post_values.append(values_post.squeeze(-1).reshape(-1))
                    post_returns.append(
                        ret_chunked[start_i:end_i].reshape(-1)
                    )
            if was_training:
                policy.train()

            post_denom = float(max(1, post_sample_count))
            post_update_approx_kl = post_kl_sum / post_denom
            post_update_clip_fraction = post_clip_count / post_denom
            post_value_tensor = torch.cat(post_values)
            post_return_tensor = torch.cat(post_returns)
            explained_variance_post = _explained_variance(
                post_value_tensor, post_return_tensor
            )

            parameter_delta_names = tuple(
                f"param_delta_{name}"
                for name in ("shared", "actor", "critic", "aux")
            )
            parameter_delta_values = tuple(
                _tensor_group_norm(
                    (
                        param.detach() - before
                        for param, before in zip(
                            diagnostic_parameter_groups[name],
                            diagnostic_parameter_snapshots[name],
                        )
                    ),
                    device,
                )
                for name in ("shared", "actor", "critic", "aux")
            )
            diagnostic_single_names = (
                "post_update_approx_kl",
                "post_update_clip_fraction",
                "explained_variance_pre",
                "explained_variance_post",
            ) + parameter_delta_names
            diagnostic_single_values = (
                post_update_approx_kl,
                post_update_clip_fraction,
                explained_variance_pre,
                explained_variance_post,
            ) + parameter_delta_values
            diagnostic_all_names = (
                diagnostic_metric_names + diagnostic_single_names
            )
            diagnostic_all_values = torch.cat(
                (
                    diagnostic_totals / optimization_denom,
                    torch.stack(diagnostic_single_values).to(
                        dtype=diagnostic_totals.dtype
                    ),
                )
            ).detach().cpu().tolist()
            diagnostic_values = dict(zip(
                diagnostic_all_names, diagnostic_all_values
            ))

        if distributed:
            # The next policy must be durable before workers can observe it.
            # Together with the accepted-batch journal this prevents a crash
            # from replaying an optimizer update or losing Adam state.
            policy_checkpoint = ckpt_dir / f"policy_{total_env_steps:08d}.pt"
            optimizer_checkpoint = ckpt_dir / f"optimizer_{total_env_steps:08d}.pt"
            for target, value in (
                (policy_checkpoint, policy.state_dict()),
                (optimizer_checkpoint, optimizer.state_dict()),
            ):
                temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
                with temporary.open("wb") as handle:
                    torch.save(value, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            checkpoint_directory = os.open(ckpt_dir, os.O_RDONLY)
            try:
                os.fsync(checkpoint_directory)
            finally:
                os.close(checkpoint_directory)
            publish_distributed_policy(total_env_steps)

        # logging
        elapsed = time.time() - start
        # A resumed checkpoint can already carry tens of millions of steps;
        # throughput is work completed by this process, not lifetime steps
        # divided by this process's short wall clock.
        run_env_steps = total_env_steps - resume_steps
        sps = run_env_steps / elapsed

        # episode stats over completed episodes since last update
        if completed_ep_rewards:
            mean_ep_r   = float(np.mean(completed_ep_rewards))
            min_ep_r    = float(np.min(completed_ep_rewards))
            max_ep_r    = float(np.max(completed_ep_rewards))
            mean_ep_len = float(np.mean(completed_ep_lengths))
            n_ep        = len(completed_ep_rewards)
            mean_base_r = float(np.mean(completed_ep_base_rewards))
            mean_spatial_r = float(np.mean(completed_ep_spatial_rewards))
            total_kills = float(np.sum(completed_ep_kills))
            total_deaths = float(np.sum(completed_ep_deaths))
            kd_ratio = total_kills / max(total_deaths, 1.0)

            writer.add_scalar("episode/reward_mean", mean_ep_r,   total_env_steps)
            writer.add_scalar("episode/base_reward_mean", mean_base_r, total_env_steps)
            writer.add_scalar("episode/spatial_reward_mean", mean_spatial_r, total_env_steps)
            writer.add_scalar("combat/kills", total_kills, total_env_steps)
            writer.add_scalar("combat/deaths", total_deaths, total_env_steps)
            writer.add_scalar("combat/kd_ratio", kd_ratio, total_env_steps)
            writer.add_scalar("episode/reward_min",  min_ep_r,    total_env_steps)
            writer.add_scalar("episode/reward_max",  max_ep_r,    total_env_steps)
            writer.add_scalar("episode/length_mean", mean_ep_len, total_env_steps)
            writer.add_scalar("episode/count",       n_ep,        total_env_steps)
            for map_name, rewards in completed_map_rewards.items():
                writer.add_scalar(f"maps/{_safe_tag(map_name)}/reward_mean",
                                  float(np.mean(rewards)), total_env_steps)
                writer.add_scalar(f"maps/{_safe_tag(map_name)}/episodes",
                                  len(rewards), total_env_steps)
        else:
            mean_ep_r = float("nan")
            mean_base_r = float("nan")
            mean_spatial_r = float("nan")
            total_kills = 0.0
            total_deaths = 0.0
            kd_ratio = 0.0
            n_ep      = 0

        # training metrics
        writer.add_scalar("train/loss", optimization_means["loss"], total_env_steps)
        writer.add_scalar("train/pg_loss", optimization_means["pg_loss"], total_env_steps)
        writer.add_scalar("train/vf_loss", optimization_means["vf_loss"], total_env_steps)
        writer.add_scalar("train/entropy", optimization_means["entropy"], total_env_steps)
        writer.add_scalar(
            "train/entropy_loss", optimization_means["entropy_loss"], total_env_steps
        )
        writer.add_scalar("train/aux_loss", optimization_means["aux_loss"], total_env_steps)
        for metric_name in (
            "lattice_direction_loss",
            "lattice_direction_weighted_loss",
            "lattice_direction_cosine",
            "lattice_direction_samples",
        ):
            writer.add_scalar(
                f"lattice/{metric_name.removeprefix('lattice_direction_')}",
                optimization_means[metric_name],
                total_env_steps,
            )
        if aim_anchor_coef > 0:
            for metric_name in (
                "aim_anchor_inner_loss",
                "aim_anchor_weighted_loss",
                "aim_anchor_look_loss",
                "aim_anchor_fire_loss",
                "aim_anchor_yaw_mae_deg",
                "aim_anchor_pitch_mae_deg",
                "aim_anchor_fire_accuracy",
                "aim_anchor_fire_positive_probability",
                "aim_anchor_fire_negative_probability",
                "aim_anchor_hidden_fire_probability",
            ):
                writer.add_scalar(
                    f"anchor/{metric_name.removeprefix('aim_anchor_')}",
                    optimization_means[metric_name],
                    total_env_steps,
                )
            for name, value in aim_anchor_target_stats.items():
                writer.add_scalar(
                    f"anchor/{name}", float(value.detach().cpu()), total_env_steps
                )
        writer.add_scalar("train/approx_kl", optimization_means["approx_kl"], total_env_steps)
        writer.add_scalar(
            "train/clip_fraction", optimization_means["clip_fraction"], total_env_steps
        )
        writer.add_scalar(
            "train/grad_norm_pre_clip",
            optimization_means["grad_norm_pre_clip"],
            total_env_steps,
        )
        if grad_diagnostics:
            for name, value in diagnostic_values.items():
                writer.add_scalar(
                    f"diagnostics/{name}", value, total_env_steps
                )
        writer.add_scalar("train/sps", sps, total_env_steps)
        if target_fire_gate and rollout_fire_gate_samples > 0:
            gate_denom = float(rollout_fire_gate_samples)
            allowed_rate = rollout_fire_gate_allowed / gate_denom
            writer.add_scalar(
                "targeting/fire_gate_allowed_rate",
                allowed_rate,
                total_env_steps,
            )
            writer.add_scalar(
                "targeting/fire_gate_closed_rate",
                1.0 - allowed_rate,
                total_env_steps,
            )
            writer.add_scalar(
                "targeting/executed_fire_rate",
                rollout_fire_gate_executed / gate_denom,
                total_env_steps,
            )
            closed_count = rollout_fire_gate_samples - rollout_fire_gate_allowed
            writer.add_scalar(
                "targeting/blocked_fire_probability_mean",
                rollout_fire_gate_blocked_probability
                / float(max(1, closed_count)),
                total_env_steps,
            )
        if rollout_posture_samples > 0:
            posture_denom = float(rollout_posture_samples)
            for tag, value in (
                ("movement/signed_forward_mean", rollout_signed_forward),
                ("movement/forward_command_rate", rollout_forward_commands),
                ("movement/backward_command_rate", rollout_backward_commands),
                ("aim/look_pitch_command_mean", rollout_look_pitch),
                ("aim/look_pitch_command_abs_mean", rollout_look_pitch_abs),
                ("aim/view_pitch_mean", rollout_view_pitch),
                ("aim/view_pitch_abs_mean", rollout_view_pitch_abs),
                ("aim/downlook_rate", rollout_downlook_frames),
            ):
                writer.add_scalar(tag, value / posture_denom, total_env_steps)

        rollout_tensors = {
            "reward": buf.rewards,
            "return": buf.returns,
            "value": buf.values,
        }
        for name, values in rollout_tensors.items():
            writer.add_scalar(
                f"rollout/{name}_mean", float(values.mean().item()), total_env_steps
            )
            writer.add_scalar(
                f"rollout/{name}_min", float(values.min().item()), total_env_steps
            )
            writer.add_scalar(
                f"rollout/{name}_max", float(values.max().item()), total_env_steps
            )
            writer.add_scalar(
                f"rollout/{name}_std",
                float(values.std(unbiased=False).item()),
                total_env_steps,
            )

        # Preserve the original mean tags for existing dashboards.
        writer.add_scalar("train/value_mean", float(buf.values.mean()), total_env_steps)
        writer.add_scalar("train/return_mean", float(buf.returns.mean()), total_env_steps)
        if rollout_behavior_samples > 0:
            denom = float(rollout_behavior_samples)
            writer.add_scalar(
                "behavior/ammo_depleted_rate",
                rollout_behavior["ammo_depleted"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "behavior/requested_ammo_weapon_unavailable_rate",
                rollout_behavior["requested_ammo_weapon_unavailable"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "behavior/fire_no_ammo_rate",
                rollout_behavior["fire_no_ammo"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "targeting/acquisition_bonus_mean",
                rollout_behavior["target_alignment_bonus"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "targeting/acquisition_rate",
                rollout_behavior["target_acquired"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "targeting/aligned_rate",
                rollout_behavior["target_aligned"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "behavior/hook_action_rate",
                rollout_behavior["hook_action"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "behavior/hook_enemy_rate",
                rollout_behavior["hook_enemy"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "behavior/hook_no_ammo_melee_rate",
                rollout_behavior["hook_no_ammo_melee"] / denom,
                total_env_steps,
            )
            for key, tag in (
                ("movement_speed", "movement/speed_mean"),
                ("movement_intent", "movement/intent_mean"),
                ("forward_intent", "movement/forward_intent_mean"),
                ("movement_nominal", "movement/nominal_rate"),
                ("movement_slow", "movement/slow_rate"),
                ("movement_overspeed", "movement/overspeed_rate"),
                ("movement_discipline", "movement/reward_mean"),
                ("jump_action", "behavior/jump_action_rate"),
                ("jump_slow", "behavior/jump_slow_rate"),
                ("hook_overspeed", "behavior/hook_overspeed_rate"),
                ("hook_fire_action", "behavior/hook_fire_rate"),
                ("hook_noop_action", "behavior/hook_noop_rate"),
                ("hook_release_action", "behavior/hook_release_rate"),
                ("hook_release_overspeed", "behavior/hook_release_overspeed_rate"),
                ("hook_correction_available", "hook/target_available_rate"),
                ("hook_correction_needed", "hook/correction_needed_rate"),
                ("hook_correction_active", "hook/correction_active_rate"),
                ("hook_correction_started", "hook/correction_started_rate"),
                ("hook_correction_escape", "hook/escape_correction_rate"),
                ("hook_correction_progress", "hook/progress_units_mean"),
                ("hook_correction_progress_reward", "hook/progress_reward_mean"),
                ("hook_correction_success", "hook/correction_success_rate"),
                ("hook_correction_timeout", "hook/correction_timeout_rate"),
                ("hook_correction_heat", "hook/target_heat_mean"),
            ):
                writer.add_scalar(tag, rollout_behavior[key] / denom, total_env_steps)
            writer.add_scalar(
                "memory/bonus_mean",
                rollout_behavior["session_memory_bonus"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/cells_mean",
                rollout_behavior["session_memory_cells"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/current_engagement",
                rollout_behavior["session_current_engagement"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/current_threat",
                rollout_behavior["session_current_threat"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/current_opportunity",
                rollout_behavior["session_current_opportunity"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/current_self_fire",
                rollout_behavior["session_current_self_fire"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/nearest_engagement",
                rollout_behavior["session_nearest_engagement"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/nearest_threat",
                rollout_behavior["session_nearest_threat"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "memory/nearest_opportunity",
                rollout_behavior["session_nearest_opportunity"] / denom,
                total_env_steps,
            )
            for key, tag in (
                ("lattice_prior_loaded", "lattice/prior_loaded_rate"),
                ("lattice_routes_loaded", "lattice/routes_loaded_rate"),
                ("lattice_dynamic_cells", "lattice/dynamic_cells_mean"),
                ("lattice_route_active", "lattice/route_active_rate"),
            ):
                writer.add_scalar(tag, rollout_behavior[key] / denom, total_env_steps)
            writer.add_scalar(
                "threat/bonus_mean",
                rollout_behavior["threat_bonus"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "threat/in_range_rate",
                rollout_behavior["threat_in_range"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "threat/active_rate",
                rollout_behavior["threat_active"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "threat/ignored_rate",
                rollout_behavior["threat_ignored"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "threat/damage_margin_step",
                rollout_behavior["damage_margin_step"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "survival/low_health_rate",
                rollout_behavior["survival_low_health"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "survival/contact_rate",
                rollout_behavior["survival_contact"] / denom,
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/bonus_mean",
                rollout_behavior["outcome_bonus"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/win_rate",
                rollout_behavior["outcome_win"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/survival_rate",
                rollout_behavior["outcome_survival"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/loss_rate",
                rollout_behavior["outcome_loss"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/idle_rate",
                rollout_behavior["outcome_idle"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/damage_margin",
                rollout_behavior["episode_damage_margin"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/frag_margin",
                rollout_behavior["episode_frag_margin"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/contact_events",
                rollout_behavior["episode_contact_events"] /
                max(1.0, rollout_behavior["outcome_sample"]),
                total_env_steps,
            )
            writer.add_scalar(
                "outcome/count",
                rollout_behavior["outcome_sample"],
                total_env_steps,
            )
            # ext channels + exchange/chicken + rune observability — so we can
            # actually SEE whether the obs dims are being exercised.
            for key, tag in (
                ("offense", "ext/offense_mean"),
                ("survival", "ext/survival_mean"),
                ("damage_prox", "ext/damage_prox_mean"),
                ("exchange_ratio", "exchange/ratio_mean"),
                ("exchange_dominating", "exchange/dominating_rate"),
                ("exchange_even", "exchange/even_rate"),
                ("exchange_losing", "exchange/losing_rate"),
                ("rune_held", "ext/rune_held_rate"),
                ("rune_switch", "ext/rune_switch_mean"),
                ("win_margin", "exchange/win_margin_mean"),
            ):
                writer.add_scalar(tag, rollout_behavior[key] / denom, total_env_steps)

        if network_native:
            for tag, value in servers[0].metrics.as_dict().items():
                writer.add_scalar(tag, value, total_env_steps)

        completed_ep_rewards.clear()
        completed_ep_base_rewards.clear()
        completed_ep_spatial_rewards.clear()
        completed_ep_kills.clear()
        completed_ep_deaths.clear()
        completed_ep_lengths.clear()
        completed_map_rewards.clear()

        temp = _thermal_check(writer, total_env_steps)
        temp_str = f"  gpu={temp:.0f}°C" if temp else ""
        print(f"[{update_n:4d}] steps={total_env_steps:>8,}  sps={sps:>6.0f}  "
              f"loss={optimization_means['loss']:.4f}  ep_r={mean_ep_r:>+7.2f} "
              f"base={mean_base_r:>+7.2f} spatial={mean_spatial_r:>+6.2f} "
              f"kd={kd_ratio:>5.2f} ({total_kills:.0f}/{total_deaths:.0f}) "
              f"(n={n_ep})  "
              f"elapsed={elapsed/60:.1f}m{temp_str}")

        # ── LLM-coach directives: external steering via lattice deposits ──
        # Channels: directives.json broadcasts (optional per-entry "server"
        # filter); directives/server_<i>.json addresses one server's bots.
        try:
            _paths = [(Path("directives.json"), None)]
            _ddir = Path("directives")
            if _ddir.is_dir():
                for _f in sorted(_ddir.glob("server_*.json")):
                    try:
                        _paths.append((_f, int(_f.stem.split("_")[1])))
                    except ValueError:
                        pass
            for _dir_path, _chan_si in _paths:
                if not _dir_path.exists():
                    continue
                _data = json.loads(_dir_path.read_text())
                _seq = int(_data.get("seq", 0))
                if _seq <= _directive_seqs.get(str(_dir_path), 0):
                    continue
                _directive_seqs[str(_dir_path)] = _seq
                _applied = 0
                for d in _data.get("directives", []):
                    _tgt = _chan_si if _chan_si is not None else d.get("server")
                    for _si, srv in enumerate(servers):
                        if _tgt is not None and int(_tgt) != _si:
                            continue
                        for sr in srv._spatial_rewards:
                            if sr.apply_directive(
                                    d.get("map", ""), d.get("action", ""),
                                    d.get("x", 0), d.get("y", 0), d.get("z", 0),
                                    float(d.get("strength", 3.0))):
                                _applied += 1
                print(f"  → coach[{_dir_path.name}] seq={_seq}: "
                      f"{len(_data.get('directives', []))} directives → {_applied} bot-memories")
        except Exception as e:
            print(f"  ! directive apply failed (continuing): {e}")

        # checkpoint
        if total_env_steps >= next_save_at:
            ckpt = save_dir / f"policy_{total_env_steps:08d}.pt"
            torch.save(policy.state_dict(), ckpt)
            torch.save(
                optimizer.state_dict(),
                save_dir / f"optimizer_{total_env_steps:08d}.pt",
            )
            try:
                from harness.spatial import save_lattice_state
                lattice_ckpt = save_lattice_state(
                    lattice_instances,
                    save_dir / f"lattice_{total_env_steps:08d}.json.gz",
                    total_env_steps=total_env_steps,
                )
                save_lattice_state(
                    lattice_instances,
                    save_dir / "lattice_latest.json.gz",
                    total_env_steps=total_env_steps,
                )
                print(f"  → saved {lattice_ckpt}")
            except Exception as e:
                print(f"  ! lattice checkpoint failed (continuing): {e}")
            try:
                export_onnx(policy, str(save_dir / "policy_latest.onnx"), device)
            except Exception as e:
                print(f"  ! onnx export failed (continuing): {e}")
            try:
                from harness.spatial import export_observed_heat
                n_maps = export_observed_heat(
                    [sr for srv in servers for sr in srv._spatial_rewards],
                    os.environ.get("Q2_OBSERVED_HEAT_DIR", "observed_heat"),
                    total_env_steps=total_env_steps)
                print(f"  → observed heat exported for {n_maps} maps")
            except Exception as e:
                print(f"  ! observed-heat export failed (continuing): {e}")
            if evolver is not None:
                try:
                    evolver.step(servers, total_env_steps)
                    writer.add_scalar("curriculum/pool_size",
                                      len(servers[0].map_pool), total_env_steps)
                    writer.add_scalar("curriculum/retired_total",
                                      len(evolver.retired), total_env_steps)
                    writer.add_scalar("curriculum/compiling",
                                      len(evolver.pending), total_env_steps)
                except Exception as e:
                    print(f"  ! curriculum step failed (continuing): {e}")
            print(f"  → saved {ckpt}")
            next_save_at = total_env_steps + cfg["save_every"]

    _close_all_servers()
    atexit.unregister(_close_all_servers)

    final_ckpt = save_dir / f"policy_{total_env_steps:08d}.pt"
    torch.save(policy.state_dict(), final_ckpt)
    torch.save(
        optimizer.state_dict(),
        save_dir / f"optimizer_{total_env_steps:08d}.pt",
    )
    try:
        from harness.spatial import save_lattice_state
        save_lattice_state(
            lattice_instances,
            save_dir / f"lattice_{total_env_steps:08d}.json.gz",
            total_env_steps=total_env_steps,
        )
        save_lattice_state(
            lattice_instances,
            save_dir / "lattice_latest.json.gz",
            total_env_steps=total_env_steps,
        )
    except Exception as e:
        print(f"! final lattice checkpoint failed: {e}")
    try:
        export_onnx(policy, str(save_dir / "policy_final.onnx"), device)
    except Exception as e:
        print(f"! final onnx export failed: {e}")
    writer.close()
    print(f"  → saved final {final_ckpt}")
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULT.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--resume", action="store_true",
                   help="Load latest checkpoints/policy_*.pt and continue from its step count")
    args = p.parse_args()
    train(vars(args))
