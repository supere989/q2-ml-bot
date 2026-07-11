"""
ppo.py — PPO training loop for Q2BotPolicy.

Runs N parallel Q2BotEnv instances, collects rollouts, updates policy.
Designed to run overnight on the Vega 10 iGPU (ROCm) or CPU fallback.

Usage:
    HSA_OVERRIDE_GFX_VERSION=9.0.0 python -m train.ppo
"""

import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
from models.policy import Q2BotPolicy, export_onnx, OBS_DIM, ACTION_DIM, HIDDEN_DIM

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
)


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"


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
        self.h_states = torch.zeros(n_steps, n_envs, hidden_dim, device=device)
        self.c_states = torch.zeros(n_steps, n_envs, hidden_dim, device=device)
        self.ptr      = 0

    def add(self, obs, action, reward, done, value, log_prob, h_state, c_state):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
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

    print(f"Policy parameters: {policy.param_count():,}")
    if grad_diagnostics:
        print("PPO gradient diagnostics: ON")

    # Per-run tag isolates parallel runs: separate checkpoint dir + TB run
    # name so stacked ablation trainers never overwrite each other or the
    # resume run. Resume reads from Q2_RESUME_DIR (default = this run's own
    # ckpt dir) so a fresh ablation can warm-start from a shared checkpoint.
    run_tag  = os.environ.get("Q2_RUN_TAG", "").strip()
    ckpt_dir = Path(os.environ.get("Q2_CKPT_DIR",
                    f"checkpoints/{run_tag}" if run_tag else "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_steps = 0
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
        print(f"Resumed from {latest} (env_steps={resume_steps:,})")

    from harness.env import Q2MultiEnv, discover_map_pool

    n_servers         = cfg["n_servers"]
    n_bots            = cfg["n_bots_per_server"]    # bots IN server (ML + AI)
    n_ml              = max(1, min(int(cfg.get("n_ml_bots", 1)), n_bots))
    total_venvs       = n_servers * n_ml
    map_pool = discover_map_pool(
        map_name=cfg["map_name"],
        map_glob=cfg.get("map_glob", ""),
        map_dir=cfg.get("map_dir", "") or None,
    )
    map_label = cfg["map_name"] if len(map_pool) == 1 else "curriculum"
    print(f"Map pool ({len(map_pool)}): {', '.join(map_pool[:8])}"
          f"{' ...' if len(map_pool) > 8 else ''}")

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

    buf = RolloutBuffer(total_venvs, cfg["n_steps"], OBS_DIM, ACTION_DIM, HIDDEN_DIM, device)

    obs_np  = np.zeros((total_venvs, OBS_DIM), dtype=np.float32)
    hx_list = [policy.init_hidden(1, device) for _ in range(total_venvs)]

    # reset all servers in parallel and populate initial obs (sequential boot
    # cost ~50s/server; each env owns its own ports, sockets, and process)
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
    if os.environ.get("Q2_CURRICULUM_EVOLVE") == "1":
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
    writer   = SummaryWriter(log_dir=f"runs/{run_name}")
    writer.add_scalar("config/seed", seed, resume_steps)
    print(f"TensorBoard log: runs/{run_name}")
    print(f"  view with:  tensorboard --logdir runs --bind_all")
    print(f"  virtual envs: {total_venvs}  ({n_servers} servers × 1 ML bot + {n_bots-1} AI opponents)")

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
    behavior_metric_keys = (
        "ammo_depleted",
        "requested_ammo_weapon_unavailable",
        "fire_no_ammo",
        "hook_action",
        "hook_enemy",
        "hook_no_ammo_melee",
        "session_memory_bonus",
        "session_memory_cells",
        "session_current_engagement",
        "session_current_threat",
        "session_current_opportunity",
        "session_current_self_fire",
        "session_nearest_engagement",
        "session_nearest_threat",
        "session_nearest_opportunity",
        "threat_bonus",
        "threat_in_range",
        "threat_active",
        "threat_ignored",
        "survival_low_health",
        "survival_contact",
        "damage_margin_step",
        "outcome_sample",
        "outcome_bonus",
        "outcome_win",
        "outcome_survival",
        "outcome_loss",
        "outcome_idle",
        "episode_damage_margin",
        "episode_frag_margin",
        "episode_contact_events",
        # ext channels + exchange/chicken + rune observability
        "offense",
        "survival",
        "damage_prox",
        "exchange_ratio",
        "exchange_dominating",
        "exchange_even",
        "exchange_losing",
        "rune_held",
        "rune_switch",
        "win_margin",
    )

    while total_env_steps < cfg["total_steps"]:
        # ── collect rollout ──────────────────────────────────────────
        policy.eval()
        rollout_behavior = {key: 0.0 for key in behavior_metric_keys}
        rollout_behavior_samples = 0

        stateful = os.environ.get("Q2_POLICY_STATEFUL", "1").lower() in {"1", "true", "yes", "on"}

        for step in range(cfg["n_steps"]):
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
            actions_np, values_np, lps_np, hx_list = policy.act_batch(
                obs_np, hx_list, device)

            rewards_np = np.zeros(total_venvs, dtype=np.float32)
            dones_np   = np.zeros(total_venvs, dtype=np.float32)

            # Parallel server stepping — UDP recv releases the GIL
            def _step_server(si_srv):
                si, srv = si_srv
                base = si * n_ml
                return si, srv.step_all([actions_np[base + k] for k in range(srv.n_ml)])

            with ThreadPoolExecutor(max_workers=n_servers) as ex:
                step_results = list(ex.map(_step_server, enumerate(servers)))

            for si, results in step_results:
                base = si * n_ml
                srv = servers[si]
                for bi, (o, r, term, trunc, info) in enumerate(results):
                    vi = base + bi
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
            )

        total_env_steps += cfg["n_steps"] * total_venvs

        # compute bootstrap value
        with torch.no_grad():
            # Batched bootstrap computation to run in a single GPU pass
            if os.environ.get("Q2_POLICY_STATEFUL", "1").lower() in {"1", "true", "yes", "on"}:
                h_stack = torch.cat([hx[0] for hx in hx_list], dim=1)
                c_stack = torch.cat([hx[1] for hx in hx_list], dim=1)
            else:
                h_stack, c_stack = policy.init_hidden(total_venvs, device)
            
            obs_t = torch.from_numpy(obs_np).to(device, dtype=torch.float32).unsqueeze(1) # (N, 1, OBS_DIM)
            _, values_t, _ = policy(obs_t, (h_stack, c_stack))
            last_vals = values_t.squeeze(-1).squeeze(-1) # (total_venvs,)

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
        dones_chunked = buf.dones.reshape(num_chunks_per_env, chunk_len, N_envs).permute(0, 2, 1).reshape(total_chunks, chunk_len)

        # Slice the recorded hidden states to get the initial state for each chunk
        chunk_starts = np.arange(0, T_steps, chunk_len)
        h_chunk_starts = buf.h_states[chunk_starts].reshape(total_chunks, HIDDEN_DIM)
        c_chunk_starts = buf.c_states[chunk_starts].reshape(total_chunks, HIDDEN_DIM)

        chunks_per_batch = max(1, cfg["batch_size"] // chunk_len)
        aux_coef = float(cfg.get("aux_coef", 0.05) or 0.0)

        diagnostic_metric_names = (
            "component_grad_norm_policy_weighted",
            "component_grad_norm_vf_weighted",
            "component_grad_norm_aux_weighted",
            "shared_grad_cos_policy_vf",
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
                log_prob, entropy = policy.action_log_prob_entropy(act_params, act_b)
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

                loss = (pg_loss + cfg["vf_coef"] * vf_loss + cfg["ent_coef"] * ent_loss
                        + aux_coef * aux_loss)

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
                    policy_shared_grads = tuple(
                        policy_component_grads[index]
                        for index in diagnostic_shared_indices
                    )
                    vf_shared_grads = tuple(
                        vf_component_grads[index]
                        for index in diagnostic_shared_indices
                    )
                    diagnostic_component_values = (
                        _tensor_group_norm(policy_component_grads, device),
                        _tensor_group_norm(vf_component_grads, device),
                        _tensor_group_norm(aux_component_grads, device),
                        _gradient_cosine(
                            policy_shared_grads, vf_shared_grads, device
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
                        params_post, act_post
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

    for srv in servers:
        srv.close()

    final_ckpt = save_dir / f"policy_{total_env_steps:08d}.pt"
    torch.save(policy.state_dict(), final_ckpt)
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
