#!/usr/bin/env python3
"""
Warm-start the policy with a simple local-frame aim/fire teacher.

This is a curriculum bootstrap, not the final controller: it teaches the
network to turn toward visible enemies and fire so PPO can optimize combat
from a policy that can produce damage.
"""

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import Q2MultiEnv, discover_map_pool
from harness.protocol import ENT_SCALE, SELF_SCALE
from models.policy import (
    ENT_CNT,
    ENT_DIM,
    ENT_OFF,
    OBS_DIM,
    Q2BotPolicy,
    export_onnx,
)

TEACHER_WEAPON = 7  # Rocket Launcher; ML training spawn carries rockets.
DEFAULT_AIM_YAW_DEG = float(os.environ.get("Q2_AIM_YAW_DEG", "12.0"))
DEFAULT_AIM_PITCH_DEG = float(os.environ.get("Q2_AIM_PITCH_DEG", "14.0"))


def _next_policy_name(checkpoint: Path) -> str:
    match = re.fullmatch(r"policy_(\d+)", checkpoint.stem)
    if match is None:
        raise ValueError(
            f"checkpoint name must be policy_<steps>.pt for PPO resume accounting: {checkpoint}"
        )
    return f"policy_{int(match.group(1)) + 1:08d}"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def _nearest_visible_enemy(obs) -> Optional[np.ndarray]:
    """Return the nearest visible enemy's bot-local xyz vector."""
    best = None
    count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
    for ent in obs.entities[:count]:
        if ent[7] <= 0.5 or ent[8] <= 0.5:
            continue
        dist = float(np.linalg.norm(ent[:3]))
        if best is None or dist < best[0]:
            best = (dist, ent[:3].copy())
    return None if best is None else best[1]


def _local_aim_target(
    rel_xyz: np.ndarray,
    current_pitch: float = 0.0,
) -> Tuple[float, float]:
    """Return yaw/pitch deltas that turn toward a bot-local target.

    ``ml_obs.c`` projects entities onto Quake's ``forward/right/up`` basis.
    Quake's right vector has the opposite sign from the conventional
    mathematical +Y axis, and positive pitch looks down, so both local
    angles must be negated before they are applied to ``ent->s.angles``.
    """
    x, y, z = (float(v) for v in rel_xyz)
    pitch_rad = math.radians(current_pitch)
    # Undo the pitch component of AngleVectors' local forward/up basis to
    # recover horizontal-forward and world-up components.
    horizontal_forward = math.cos(pitch_rad) * x + math.sin(pitch_rad) * z
    vertical = -math.sin(pitch_rad) * x + math.cos(pitch_rad) * z
    yaw_delta = -math.degrees(math.atan2(y, horizontal_forward))
    target_pitch = -math.degrees(
        math.atan2(vertical, max(math.hypot(horizontal_forward, y), 1e-3))
    )
    return _wrap_degrees(yaw_delta), target_pitch - current_pitch


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _fire_after_command(
    yaw_delta: float,
    pitch_delta: float,
    yaw_command: float,
    pitch_command: float,
    aim_yaw_deg: float,
    aim_pitch_deg: float,
    current_pitch: float = 0.0,
) -> bool:
    """Whether the view will be aligned after this tick's turn command."""
    # ``yaw_delta``/``pitch_delta`` are the desired engine commands. Their
    # negations are the target's current local-frame angles. ML_ApplyAction
    # applies the look deltas before BUTTON_ATTACK reaches Think_Weapon, so
    # simultaneous turn+fire is valid when the post-command residual aligns.
    yaw_residual = _wrap_degrees(-yaw_delta + yaw_command)
    new_pitch = float(np.clip(current_pitch + pitch_command, -89.0, 89.0))
    effective_pitch_command = new_pitch - current_pitch
    pitch_residual = -pitch_delta + effective_pitch_command
    return abs(yaw_residual) <= aim_yaw_deg and abs(pitch_residual) <= aim_pitch_deg


def _teacher_action(
    obs,
    aim_yaw_deg: float = DEFAULT_AIM_YAW_DEG,
    aim_pitch_deg: float = DEFAULT_AIM_PITCH_DEG,
) -> np.ndarray:
    action = np.array([0.35, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, TEACHER_WEAPON], dtype=np.float32)
    rel_xyz = _nearest_visible_enemy(obs)
    if rel_xyz is None:
        return action

    yaw_delta, pitch_delta = _local_aim_target(rel_xyz, float(obs.pitch))
    action[0] = 0.0
    action[1] = 0.0
    yaw_command = float(np.clip(yaw_delta, -45.0, 45.0))
    pitch_command = float(np.clip(pitch_delta, -30.0, 30.0))
    action[2] = yaw_command
    action[3] = pitch_command
    # Do not teach the policy to fire while it is still turning. The old
    # labels marked every visible target as fire=1, including targets 180
    # degrees behind the bot, which explicitly trained an always-fire policy.
    action[5] = float(
        _fire_after_command(
            yaw_delta,
            pitch_delta,
            yaw_command,
            pitch_command,
            aim_yaw_deg,
            aim_pitch_deg,
            current_pitch=float(obs.pitch),
        )
    )
    return action


def collect(args) -> Tuple[np.ndarray, np.ndarray, dict]:
    maps = discover_map_pool(map_glob=args.map_glob, map_dir=args.map_dir or None)
    obs_rows: List[np.ndarray] = []
    act_rows: List[np.ndarray] = []
    stats = {
        "visible_samples": 0,
        "aligned_samples": 0,
        "fire_samples": 0,
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
            server_id=args.server_id,
            map_pool=[map_name],
            n_bots=args.n_bots,
            port_offset=args.port_offset,
            maxclients=args.maxclients,
            ml_slot=args.ml_slot,
            game_seed=(
                None if args.game_seed < 0
                else args.game_seed + (map_idx - 1) * 1009
            ),
            max_ep_steps=args.episode_steps,
        )
        try:
            obs_vec = env.reset_all()[0]
            for _ in range(args.episode_steps):
                obs = env._last_obs[0]
                rel_xyz = _nearest_visible_enemy(obs)
                action = _teacher_action(
                    obs,
                    aim_yaw_deg=args.aim_yaw_deg,
                    aim_pitch_deg=args.aim_pitch_deg,
                )
                # Keep the exact vector the environment supplied, including
                # its 24-dim live session-memory block. Calling
                # obs.to_vector() here silently replaced that block with
                # zeros and made "real" collection partly out-of-distribution.
                obs_rows.append(obs_vec.copy())
                act_rows.append(action)
                stats["visible_samples"] += int(rel_xyz is not None)
                stats["aligned_samples"] += int(rel_xyz is not None and action[5] > 0.5)
                stats["fire_samples"] += int(action[5] > 0.5)

                obs_vec, _reward, term, trunc, info = env.step_all([action])[0]
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


def synthetic_aim(
    args,
    sample_count: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate normalized policy vectors with local-frame aim labels.

    This function produces the same representation as
    ``Observation.to_vector()``. The previous implementation wrote raw
    positions, velocities, health, entity coordinates, and ray distances
    directly into an already-normalized policy vector. That made the first
    BC run train on inputs thousands of times outside the live domain.
    """
    rng = np.random.default_rng(args.seed if seed is None else seed)
    n_samples = args.synthetic_samples if sample_count is None else sample_count
    obs = np.zeros((n_samples, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n_samples, 8), dtype=np.float32)
    visible = 0
    aligned = 0

    for i in range(n_samples):
        self_raw = np.zeros(10, dtype=np.float32)
        self_raw[0:3] = np.array([
            rng.uniform(-1200.0, 2200.0),
            rng.uniform(-1200.0, 2200.0),
            rng.uniform(-64.0, 512.0),
        ], dtype=np.float32)
        self_raw[3:6] = rng.uniform(-320.0, 320.0, size=3).astype(np.float32)
        self_raw[6] = rng.uniform(40.0, 100.0)   # health
        self_raw[7] = rng.uniform(0.0, 200.0)    # armor
        self_raw[8] = rng.choice([8.0, 15.0, 37.0])
        self_raw[9] = rng.choice([0.0, 1.0, 10.0, 50.0, 100.0])
        obs[i, :10] = self_raw / SELF_SCALE
        action = np.array([0.35, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, TEACHER_WEAPON], dtype=np.float32)
        current_pitch = rng.uniform(-30.0, 30.0)

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
            ent_raw = np.array(
                [x, y, z, 0.0, 0.0, 0.0, 100.0, 1.0,
                 1.0 if slot == visible_slot else 0.0],
                dtype=np.float32,
            )
            obs[i, ent0:ent0 + 9] = ent_raw / ENT_SCALE

            if slot != visible_slot:
                continue

            yaw_delta, pitch_delta = _local_aim_target(
                np.array([x, y, z], dtype=np.float32), current_pitch
            )
            action[0] = 0.0
            action[1] = 0.0
            yaw_command = float(np.clip(yaw_delta, -45.0, 45.0))
            pitch_command = float(np.clip(pitch_delta, -30.0, 30.0))
            action[2] = yaw_command
            action[3] = pitch_command
            action[5] = float(
                _fire_after_command(
                    yaw_delta,
                    pitch_delta,
                    yaw_command,
                    pitch_command,
                    args.aim_yaw_deg,
                    args.aim_pitch_deg,
                    current_pitch=current_pitch,
                )
            )
            visible += 1
            aligned += int(action[5] > 0.5)

        ray0 = 10 + 8 * 9
        world_yaw = rng.uniform(-180.0, 180.0)
        for ray in range(16):
            off = ray0 + ray * 4
            angle = math.radians(world_yaw) + (2.0 * math.pi * ray) / 16.0
            obs[i, off + 0] = math.cos(angle)
            obs[i, off + 1] = math.sin(angle)
            obs[i, off + 2] = rng.uniform(-0.15, 0.15)
            ray_distance = -1.0 if rng.random() < 0.1 else rng.uniform(96.0, 2048.0)
            obs[i, off + 3] = ray_distance / 4096.0

        # Facing is at the end of the 185-dim base observation. Extended
        # observations (when enabled) and session memory remain zero.
        obs[i, 183] = world_yaw / 180.0
        obs[i, 184] = current_pitch / 90.0
        if OBS_DIM > 209:
            # Extended-observation "no recent inbound damage" sentinel.
            # Zero would mean an attacker at zero normalized distance.
            obs[i, 193] = -1.0

        actions[i] = action

    return obs, actions, {
        "visible_samples": visible,
        "aligned_samples": aligned,
        "fire_samples": aligned,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "items": 0.0,
        "episodes": 0,
    }


def _visible_mask(obs_t: torch.Tensor) -> torch.Tensor:
    ents = obs_t[:, ENT_OFF:ENT_OFF + ENT_CNT * ENT_DIM].reshape(-1, ENT_CNT, ENT_DIM)
    return ((ents[:, :, 7] > 0.5) & (ents[:, :, 8] > 0.5)).any(dim=1)


def _distill_kl(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(current, dim=-1),
        F.softmax(reference, dim=-1),
        reduction="batchmean",
    )


def _policy_outputs(
    policy: Q2BotPolicy,
    obs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    rows: Dict[str, List[np.ndarray]] = {
        "cont": [], "jump": [], "fire": [], "hook": [], "weapon": [], "value": [],
    }
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            obs_t = torch.from_numpy(obs[start:start + batch_size]).to(device)
            params, value, _hx = policy(obs_t.unsqueeze(1))
            cont = params["cont_mean"].squeeze(1)
            jump = params["jump_logits"].squeeze(1).argmax(dim=-1)
            hook = params["hook_logits"].squeeze(1).argmax(dim=-1)
            weapon = params["weapon_logits"].squeeze(1).argmax(dim=-1)
            fire = policy.fire_logits_for(
                params["feat"], weapon.unsqueeze(1)
            ).squeeze(1).argmax(dim=-1)
            for key, tensor in (
                ("cont", cont), ("jump", jump), ("fire", fire),
                ("hook", hook), ("weapon", weapon), ("value", value.squeeze(-1).squeeze(-1)),
            ):
                rows[key].append(tensor.detach().cpu().numpy())
    return {key: np.concatenate(parts, axis=0) for key, parts in rows.items()}


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_policy(
    policy: Q2BotPolicy,
    reference: Q2BotPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    args,
) -> Dict[str, float]:
    pred = _policy_outputs(policy, obs, device, args.eval_batch_size)
    ref = _policy_outputs(reference, obs, device, args.eval_batch_size)
    ents = obs[:, ENT_OFF:ENT_OFF + ENT_CNT * ENT_DIM].reshape(-1, ENT_CNT, ENT_DIM)
    visible = ((ents[:, :, 7] > 0.5) & (ents[:, :, 8] > 0.5)).any(axis=1)
    fire_target = actions[:, 5] > 0.5

    pred_cont = pred["cont"].copy()
    pred_cont[:, 0:2] = np.clip(pred_cont[:, 0:2], -1.0, 1.0)
    pred_cont[:, 2] = np.clip(pred_cont[:, 2], -45.0, 45.0)
    pred_cont[:, 3] = np.clip(pred_cont[:, 3], -30.0, 30.0)
    ref_cont = ref["cont"].copy()
    ref_cont[:, 0:2] = np.clip(ref_cont[:, 0:2], -1.0, 1.0)
    ref_cont[:, 2] = np.clip(ref_cont[:, 2], -45.0, 45.0)
    ref_cont[:, 3] = np.clip(ref_cont[:, 3], -30.0, 30.0)

    predicted_post_aligned = np.zeros(len(obs), dtype=bool)
    for row in np.flatnonzero(visible):
        candidates = ents[row][
            (ents[row, :, 7] > 0.5) & (ents[row, :, 8] > 0.5)
        ]
        rel_xyz = min(candidates, key=lambda ent: float(np.linalg.norm(ent[:3])))[:3]
        current_pitch = float(obs[row, 184] * 90.0)
        desired_yaw, desired_pitch = _local_aim_target(rel_xyz, current_pitch)
        post_yaw = abs(_wrap_degrees(desired_yaw - float(pred_cont[row, 2])))
        new_pitch = float(np.clip(current_pitch + pred_cont[row, 3], -89.0, 89.0))
        effective_pitch_delta = new_pitch - current_pitch
        post_pitch = abs(desired_pitch - effective_pitch_delta)
        predicted_post_aligned[row] = (
            post_yaw <= args.aim_yaw_deg and post_pitch <= args.aim_pitch_deg
        )

    visible_yaw = pred_cont[visible, 2]
    target_yaw = actions[visible, 2]
    visible_pitch = pred_cont[visible, 3]
    target_pitch = actions[visible, 3]
    pred_fire = pred["fire"] > 0
    tp = int(np.count_nonzero(pred_fire & fire_target))
    predicted_positive = int(np.count_nonzero(pred_fire))
    actual_positive = int(np.count_nonzero(fire_target))
    aligned_fires = int(np.count_nonzero(pred_fire & predicted_post_aligned))
    predicted_aligned = int(np.count_nonzero(predicted_post_aligned))
    visible_misaligned = visible & ~predicted_post_aligned

    metrics = {
        "yaw_mae_deg": float(np.mean(np.abs(visible_yaw - target_yaw))),
        "yaw_corr": _safe_corr(visible_yaw, target_yaw),
        "pitch_mae_deg": float(np.mean(np.abs(visible_pitch - target_pitch))),
        "visible_fire_rate": float(np.mean(pred_fire[visible])) if visible.any() else 0.0,
        "hidden_fire_rate": float(np.mean(pred_fire[~visible])) if (~visible).any() else 0.0,
        "teacher_fire_precision": float(tp / predicted_positive) if predicted_positive else 0.0,
        "teacher_fire_recall": float(tp / actual_positive) if actual_positive else 0.0,
        "predicted_post_align_rate": (
            float(np.mean(predicted_post_aligned[visible])) if visible.any() else 0.0
        ),
        "aligned_fire_precision": (
            float(aligned_fires / predicted_positive) if predicted_positive else 0.0
        ),
        "aligned_fire_recall": (
            float(aligned_fires / predicted_aligned) if predicted_aligned else 0.0
        ),
        "visible_misaligned_fire_rate": (
            float(np.mean(pred_fire[visible_misaligned])) if visible_misaligned.any() else 0.0
        ),
        "move_drift_mae": float(np.mean(np.abs(pred_cont[:, :2] - ref_cont[:, :2]))),
        "hidden_look_drift_mae": float(
            np.mean(np.abs(pred_cont[~visible, 2:4] - ref_cont[~visible, 2:4]))
        ),
        "jump_agreement": float(np.mean(pred["jump"] == ref["jump"])),
        "hook_agreement": float(np.mean(pred["hook"] == ref["hook"])),
        "weapon_agreement": float(np.mean(pred["weapon"] == ref["weapon"])),
        "value_drift_mae": float(np.mean(np.abs(pred["value"] - ref["value"]))),
    }
    return metrics


def train_bc(policy: Q2BotPolicy, obs: np.ndarray, actions: np.ndarray, device: torch.device, args) -> Q2BotPolicy:
    """Train aim/fire while preserving the already-working policy.

    By default only the continuous actor head (whose yaw/pitch rows are
    independent from its movement rows) and the autoregressive fire head
    are trainable. The old implementation optimized every model parameter
    against fabricated movement/jump/hook/weapon labels; eight epochs moved
    the LSTM and non-aim heads substantially, so simply adding epochs risked
    erasing the navigation skill we are trying to retain.
    """
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    reference = copy.deepcopy(policy).eval().to(device)
    for param in reference.parameters():
        param.requires_grad_(False)

    for param in policy.parameters():
        param.requires_grad_(False)
    for module in (policy.actor_cont, policy.actor_fire):
        for param in module.parameters():
            param.requires_grad_(True)
    if args.train_backbone:
        for module in (policy.encoder, policy.lstm):
            for param in module.parameters():
                param.requires_grad_(True)

    trainable = [param for param in policy.parameters() if param.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)
    look_scale = torch.tensor([45.0, 30.0], device=device)
    visible_all = _visible_mask(obs_t)
    fire_targets_all = act_t[:, 5].long().clamp(0, 1)
    positive = int(fire_targets_all.sum().item())
    negative = int(len(fire_targets_all) - positive)
    if positive == 0 or negative == 0:
        raise ValueError(f"fire labels need both classes, got positive={positive} negative={negative}")
    fire_weight = args.fire_weight if args.fire_weight > 0 else negative / positive
    fire_class_weights = torch.tensor([1.0, fire_weight], device=device)
    print(
        f"trainable_params={sum(p.numel() for p in trainable):,} "
        f"train_backbone={args.train_backbone} fire_positive={positive} "
        f"fire_negative={negative} fire_weight={fire_weight:.3f}"
    )

    policy.train()
    n = obs_t.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        totals = {"loss": 0.0, "look": 0.0, "fire": 0.0, "preserve": 0.0}
        batches = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch_obs = obs_t[idx]
            params, value, _hx = policy(batch_obs.unsqueeze(1))
            cont_mean = params["cont_mean"].squeeze(1)
            visible = visible_all[idx]
            if visible.any():
                look_loss = (((
                    cont_mean[visible, 2:4] - act_t[idx][visible, 2:4]
                ) / look_scale).pow(2)).mean()
            else:
                look_loss = cont_mean.sum() * 0.0

            with torch.no_grad():
                ref_params, ref_value, _ref_hx = reference(batch_obs.unsqueeze(1))
                ref_aux = reference.predict_next(ref_params["feat"])

            # Condition fire on the frozen reference policy's selected
            # weapon: this is the deployment path, while retaining the
            # checkpoint's learned weapon-specific firing behavior. Training
            # the same label through every weapon embedding erased that
            # autoregressive distinction.
            weapon_idx = ref_params["weapon_logits"].argmax(dim=-1)
            fire_logits = policy.fire_logits_for(params["feat"], weapon_idx)
            fire_targets = fire_targets_all[idx].unsqueeze(1)
            fire_loss = F.cross_entropy(
                fire_logits.squeeze(1),
                fire_targets.squeeze(1),
                weight=fire_class_weights,
            )

            hidden = ~visible
            keep_move = F.mse_loss(
                cont_mean[:, :2], ref_params["cont_mean"].squeeze(1)[:, :2]
            )
            if hidden.any():
                keep_scan = (((
                    cont_mean[hidden, 2:4]
                    - ref_params["cont_mean"].squeeze(1)[hidden, 2:4]
                ) / look_scale).pow(2)).mean()
            else:
                keep_scan = cont_mean.sum() * 0.0

            keep_discrete = cont_mean.sum() * 0.0
            keep_std = cont_mean.sum() * 0.0
            keep_value = cont_mean.sum() * 0.0
            keep_aux = cont_mean.sum() * 0.0
            if args.train_backbone:
                keep_discrete = (
                    _distill_kl(params["jump_logits"], ref_params["jump_logits"])
                    + _distill_kl(params["hook_logits"], ref_params["hook_logits"])
                    + _distill_kl(params["weapon_logits"], ref_params["weapon_logits"])
                )
                keep_std = F.mse_loss(params["cont_log_std"], ref_params["cont_log_std"])
                keep_value = F.mse_loss(value, ref_value)
                keep_aux = F.mse_loss(policy.predict_next(params["feat"]), ref_aux)

            preserve_loss = (
                args.keep_move_weight * keep_move
                + args.keep_scan_weight * keep_scan
                + args.keep_discrete_weight * keep_discrete
                + args.keep_std_weight * keep_std
                + args.keep_value_weight * keep_value
                + args.keep_aux_weight * keep_aux
            )
            loss = args.look_weight * look_loss + fire_loss + preserve_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            totals["loss"] += float(loss.item())
            totals["look"] += float(look_loss.item())
            totals["fire"] += float(fire_loss.item())
            totals["preserve"] += float(preserve_loss.item())
            batches += 1
        print(
            f"epoch={epoch + 1} loss={totals['loss'] / batches:.4f} "
            f"look={totals['look'] / batches:.4f} "
            f"fire={totals['fire'] / batches:.4f} "
            f"preserve={totals['preserve'] / batches:.4f}"
        )

    for param in policy.parameters():
        param.requires_grad_(True)
    policy.eval()
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval_only", action="store_true",
        help="evaluate a checkpoint on the fixed synthetic holdout without training/writes",
    )
    parser.add_argument(
        "--reference_checkpoint", default="",
        help="optional preservation baseline for --eval_only drift/agreement metrics",
    )
    parser.add_argument("--output_dir", default="checkpoints/bc_aim")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--map_glob", default="mltrain_*.bsp")
    parser.add_argument("--map_dir", default="")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--min_visible_samples", type=int, default=500)
    parser.add_argument("--max_collected", type=int, default=30000)
    parser.add_argument("--episode_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--look_weight", type=float, default=16.0)
    parser.add_argument(
        "--fire_weight", type=float, default=0.0,
        help="positive-class weight; <=0 balances from collected labels",
    )
    parser.add_argument("--keep_move_weight", type=float, default=8.0)
    parser.add_argument("--keep_scan_weight", type=float, default=8.0)
    parser.add_argument("--keep_discrete_weight", type=float, default=4.0)
    parser.add_argument("--keep_std_weight", type=float, default=0.5)
    parser.add_argument("--keep_value_weight", type=float, default=0.05)
    parser.add_argument("--keep_aux_weight", type=float, default=0.2)
    parser.add_argument(
        "--train_backbone", action="store_true",
        help="EXPERIMENTAL: also tune encoder/LSTM with one-step distillation; canary first",
    )
    parser.add_argument("--n_bots", type=int, default=4)
    parser.add_argument("--server_id", type=int, default=30)
    parser.add_argument("--port_offset", type=int, default=30)
    parser.add_argument("--maxclients", type=int, default=12)
    parser.add_argument("--ml_slot", type=int, default=11)
    parser.add_argument(
        "--game_seed", type=int, default=-1,
        help="gameplay RNG seed for real collection; negative preserves normal randomness",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic_samples", type=int, default=20000)
    parser.add_argument(
        "--synthetic_replay_samples", type=int, default=0,
        help="when collecting real rollouts, mix this many normalized synthetic rows",
    )
    parser.add_argument("--synthetic_visible_rate", type=float, default=0.65)
    parser.add_argument("--synthetic_yaw_abs", type=float, default=180.0)
    parser.add_argument("--eval_samples", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--aim_yaw_deg", type=float, default=DEFAULT_AIM_YAW_DEG)
    parser.add_argument("--aim_pitch_deg", type=float, default=DEFAULT_AIM_PITCH_DEG)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.reference_checkpoint and not args.eval_only:
        parser.error("--reference_checkpoint is only valid with --eval_only")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _pick_device()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = ROOT / ckpt

    policy = Q2BotPolicy().to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"loaded={ckpt} device={device}")

    if args.eval_only:
        reference = policy
        if args.reference_checkpoint:
            reference_path = Path(args.reference_checkpoint)
            if not reference_path.is_absolute() and not reference_path.exists():
                reference_path = ROOT / reference_path
            reference = Q2BotPolicy().to(device)
            reference.load_state_dict(torch.load(reference_path, map_location=device))
            reference.eval()
            print(f"reference={reference_path}")
        eval_obs, eval_actions, _eval_stats = synthetic_aim(
            args, sample_count=args.eval_samples, seed=args.seed + 1
        )
        metrics = evaluate_policy(
            policy, reference, eval_obs, eval_actions, device, args
        )
        print("eval=" + json.dumps(metrics, sort_keys=True))
        return 0

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run_name = args.run_name or _next_policy_name(ckpt)
    if re.fullmatch(r"policy_\d+", run_name) is None:
        raise ValueError(
            f"run_name must be policy_<steps> so train.ppo can resume it: {run_name!r}"
        )
    out = output_dir / f"{run_name}.pt"
    onnx_out = output_dir / f"{run_name}.onnx"
    if not args.overwrite:
        existing = [path for path in (out, onnx_out) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite BC artifact(s): "
                + ", ".join(str(path) for path in existing)
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    obs, actions, stats = synthetic_aim(args) if args.synthetic else collect(args)
    print(f"collected={len(obs)} visible_samples={stats['visible_samples']} "
          f"aligned_samples={stats['aligned_samples']} fire_samples={stats['fire_samples']} "
          f"damage={stats['damage_dealt']:.0f}/{stats['damage_taken']:.0f} "
          f"items={stats['items']:.0f}")
    if not args.synthetic and args.synthetic_replay_samples > 0:
        replay_obs, replay_actions, replay_stats = synthetic_aim(
            args,
            sample_count=args.synthetic_replay_samples,
            seed=args.seed + 100_000,
        )
        obs = np.concatenate([obs, replay_obs], axis=0)
        actions = np.concatenate([actions, replay_actions], axis=0)
        print(
            f"synthetic_replay={len(replay_obs)} "
            f"visible_samples={replay_stats['visible_samples']} "
            f"fire_samples={replay_stats['fire_samples']} "
            f"combined_training_rows={len(obs)}"
        )
    eval_obs, eval_actions, _eval_stats = synthetic_aim(
        args, sample_count=args.eval_samples, seed=args.seed + 1
    )
    reference_before = copy.deepcopy(policy).eval().to(device)
    before = evaluate_policy(
        policy, reference_before, eval_obs, eval_actions, device, args
    )
    print("eval_before=" + json.dumps(before, sort_keys=True))

    reference = train_bc(policy, obs, actions, device, args)
    after = evaluate_policy(policy, reference, eval_obs, eval_actions, device, args)
    print("eval_after=" + json.dumps(after, sort_keys=True))

    temp_tag = f".tmp-{os.getpid()}"
    temp_out = out.with_name(out.name + temp_tag)
    temp_onnx = onnx_out.with_name(onnx_out.name + temp_tag)
    try:
        torch.save(policy.state_dict(), temp_out)
        export_onnx(policy, str(temp_onnx), device)
        os.replace(temp_out, out)
        os.replace(temp_onnx, onnx_out)
    except Exception as exc:
        temp_out.unlink(missing_ok=True)
        temp_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"atomic checkpoint/ONNX save failed: {exc}") from exc
    print(f"saved_checkpoint={out}")
    print(f"saved_onnx={onnx_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
