"""
policy.py — recurrent attention policy network for the Q2 ML bot.

Architecture (v2):
  obs_vector (206,)
    ├─ entity block (8 × 9) → shared embed → attention + max pool ─┐
    └─ rest (134,)          → MLP ─────────────────────────────────┴→ combine → (256,)
    → LSTM (hidden=256, layers=1) → (256,)
    → actor heads (cont mean + state-dependent log_std, jump, hook, weapon;
       fire is autoregressive — conditioned on the chosen weapon)
    → critic head → value scalar
    → predict_next head → next-obs prediction (auxiliary loss only)

v2 changes vs the original flat-MLP LSTM:
  1. Stateful memory is the DEFAULT (Q2_POLICY_STATEFUL=0 to opt out).
  2. Attention pooling over the 8 entity slots (permutation-invariant).
  3. Auxiliary next-obs prediction head (world-model gradient shaping).
  4. State-dependent log_std head instead of a global constant.
  5. Fire head conditioned on the sampled weapon (autoregressive).

Checkpoints from the v1 architecture are NOT loadable into v2.
Exports to ONNX for in-game inference via ONNX Runtime.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, Optional

# OBS_DIM follows protocol.py: 206 (Run A) or 216 when Q2_EXT_OBS=1 (Run B).
# The extended block is appended at the very END of the vector (after session
# memory), so the entity-attention offsets below are unaffected either way.
from harness.protocol import OBS_DIM as OBS_DIM  # noqa: E402
HIDDEN_DIM = 256
ACTION_DIM = 8   # [fwd, right, yaw, pitch, jump, fire, hook, weapon]

# obs layout (must match harness/protocol.py to_vector)
ENT_OFF = 10     # self_state occupies [0:10]
ENT_CNT = 8
ENT_DIM = 9      # floats per entity slot
ENT_LEN = ENT_CNT * ENT_DIM          # 72, entities occupy [10:82]
REST_DIM = OBS_DIM - ENT_LEN         # 134 (A) / 144 (B)

WEAPON_CLASSES = 10
WEAPON_EMB_DIM = 16

LOG_STD_MIN, LOG_STD_MAX = -3.0, 1.0


def _stateful_enabled() -> bool:
    return os.environ.get("Q2_POLICY_STATEFUL", "1").lower() in {"1", "true", "yes", "on"}


class ObsEncoder(nn.Module):
    """Entity-attention encoder.

    The 8 entity slots are embedded with a shared MLP and pooled with
    softmax attention + max pooling, so the policy sees enemies as a set
    rather than as fixed positions in the obs vector. Everything else
    (self state, rays, hook zones, audio, facing, session memory) goes
    through a plain MLP and is fused with the pooled entity features.
    """

    def __init__(self, obs_dim: int = OBS_DIM, out_dim: int = HIDDEN_DIM):
        super().__init__()
        self.ent_embed = nn.Sequential(
            nn.Linear(ENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )
        self.ent_score = nn.Linear(64, 1)
        self.rest = nn.Sequential(
            nn.Linear(REST_DIM, 192), nn.LayerNorm(192), nn.ELU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 + 64 + 192, out_dim), nn.LayerNorm(out_dim), nn.ELU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        ents = x[..., ENT_OFF:ENT_OFF + ENT_LEN].reshape(*lead, ENT_CNT, ENT_DIM)
        emb  = self.ent_embed(ents)                          # (..., 8, 64)
        w    = torch.softmax(self.ent_score(emb), dim=-2)    # (..., 8, 1)
        pooled_attn = (w * emb).sum(dim=-2)                  # (..., 64)
        pooled_max  = emb.max(dim=-2).values                 # (..., 64)
        rest = torch.cat([x[..., :ENT_OFF], x[..., ENT_OFF + ENT_LEN:]], dim=-1)
        r    = self.rest(rest)                               # (..., 192)
        return self.combine(torch.cat([pooled_attn, pooled_max, r], dim=-1))


class Q2BotPolicy(nn.Module):
    """
    Actor-critic LSTM policy (v2).

    forward() accepts a sequence of observations and optional LSTM state.
    Returns action distribution parameters (including the LSTM features
    under "feat", needed for the autoregressive fire head and the
    auxiliary prediction loss), value estimate, and new hidden state.

    For inference (single step, batch=1):
        obs: (1, 1, OBS_DIM)
        hx:  tuple of (1, 1, HIDDEN_DIM) tensors

    For training (batched sequences):
        obs: (batch, seq_len, OBS_DIM)
        hx:  None (zero-initialized)
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        hidden_dim: int = HIDDEN_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = ObsEncoder(obs_dim, hidden_dim)
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim,
                               num_layers=1, batch_first=True)

        # actor: continuous actions for [fwd, right, yaw, pitch]
        self.actor_cont   = nn.Linear(hidden_dim, 4)
        self.log_std_head = nn.Linear(hidden_dim, 4)   # state-dependent noise

        # actor: discrete actions for [jump, fire, hook, weapon]
        self.actor_jump   = nn.Linear(hidden_dim, 2)   # binary
        self.actor_hook   = nn.Linear(hidden_dim, 4)   # 0-3
        self.actor_weapon = nn.Linear(hidden_dim, WEAPON_CLASSES)
        # fire is autoregressive: conditioned on the chosen weapon
        self.weapon_embed = nn.Embedding(WEAPON_CLASSES, WEAPON_EMB_DIM)
        self.actor_fire   = nn.Linear(hidden_dim + WEAPON_EMB_DIM, 2)

        # critic
        self.critic = nn.Linear(hidden_dim, 1)

        # auxiliary world-model head: predict the next ENCODED obs (latent,
        # unit-scale) from feat. Raw obs are unnormalised (positions in the
        # thousands) and would dominate the joint loss. Never used for
        # acting — only its gradient shapes the LSTM state.
        self.predict_next = nn.Linear(hidden_dim, hidden_dim)

        # Stateless hidden state caches to avoid VRAM/RAM allocation churn
        self._cached_hx_device = None
        self._cached_hx = None

        self._cached_hx_batch_device = None
        self._cached_hx_batch_size = 0
        self._cached_hx_batch = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_cont.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        # start with the same exploration noise everywhere (log_std == 0),
        # matching the old constant-parameter behaviour at init
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hx:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[dict, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            act_params: dict of distribution parameters (+ "feat")
            value:      (batch, seq, 1)
            hx_new:     new LSTM state

        Note: fire logits are NOT in act_params — fire is conditioned on
        the chosen weapon. Use fire_logits_for(feat, weapon_idx).
        """
        enc = self.encoder(obs)
        lstm_out, hx_new = self.lstm(enc, hx)
        feat = lstm_out  # (B, T, H)

        act_params = {
            "feat":          feat,
            "cont_mean":     self.actor_cont(feat),                    # (B, T, 4)
            "cont_log_std":  self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX),
            "jump_logits":   self.actor_jump(feat),                    # (B, T, 2)
            "hook_logits":   self.actor_hook(feat),                    # (B, T, 4)
            "weapon_logits": self.actor_weapon(feat),                  # (B, T, 10)
        }
        value = self.critic(feat)                                      # (B, T, 1)

        return act_params, value, hx_new

    def fire_logits_for(self, feat: torch.Tensor, weapon_idx: torch.Tensor) -> torch.Tensor:
        """Fire logits conditioned on the (sampled or taken) weapon index."""
        emb = self.weapon_embed(weapon_idx)                            # (..., EMB)
        return self.actor_fire(torch.cat([feat, emb], dim=-1))

    def init_hidden(
        self, batch: int = 1, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        c = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h, c

    def action_log_prob_entropy(
        self,
        act_params: dict,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-probability and entropy for the full mixed action vector."""
        cont_dist = torch.distributions.Normal(
            act_params["cont_mean"],
            act_params["cont_log_std"].exp(),
        )
        log_prob = cont_dist.log_prob(actions[..., :4]).sum(-1)
        entropy = cont_dist.entropy().sum(-1)

        weapon_idx = actions[..., 7].round().long().clamp(0, WEAPON_CLASSES - 1)
        fire_logits = self.fire_logits_for(act_params["feat"], weapon_idx)

        for logits, action_idx, max_class in (
            (act_params["jump_logits"],   4, 1),
            (fire_logits,                 5, 1),
            (act_params["hook_logits"],   6, 3),
            (act_params["weapon_logits"], 7, WEAPON_CLASSES - 1),
        ):
            dist = torch.distributions.Categorical(logits=logits)
            idx = actions[..., action_idx].round().long().clamp(0, max_class)
            log_prob = log_prob + dist.log_prob(idx)
            entropy = entropy + dist.entropy()

        return log_prob, entropy

    @torch.no_grad()
    def act(
        self,
        obs_vec: np.ndarray,
        hx: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step inference for in-game use.

        Returns: (action_array (8,), value_scalar, log_prob_scalar, new_hx)
        """
        obs_t = torch.from_numpy(obs_vec).to(device=device, dtype=torch.float32)
        obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # (1, 1, OBS_DIM)
        if not _stateful_enabled():
            if self._cached_hx_device != device or self._cached_hx is None:
                self._cached_hx = self.init_hidden(1, device)
                self._cached_hx_device = device
            hx = self._cached_hx

        act_params, value, hx_new = self.forward(obs_t, hx)

        cont_mean = act_params["cont_mean"].squeeze()
        cont_std  = act_params["cont_log_std"].exp().squeeze()
        cont_dist = torch.distributions.Normal(cont_mean, cont_std)

        cont = cont_mean if deterministic else cont_dist.sample()
        log_prob_t = cont_dist.log_prob(cont).sum()

        def sample_cat(logits, det):
            d = torch.distributions.Categorical(logits=logits.reshape(-1))
            sample = d.probs.argmax() if det else d.sample()
            return sample, d.log_prob(sample)

        jump, jump_lp     = sample_cat(act_params["jump_logits"],   deterministic)
        hook, hook_lp     = sample_cat(act_params["hook_logits"],   deterministic)
        weapon, weapon_lp = sample_cat(act_params["weapon_logits"], deterministic)
        fire_logits = self.fire_logits_for(
            act_params["feat"], weapon.view(1, 1)
        )
        fire, fire_lp     = sample_cat(fire_logits, deterministic)
        log_prob_t = log_prob_t + jump_lp + fire_lp + hook_lp + weapon_lp

        action = np.array([
            cont[0].item(), cont[1].item(),
            cont[2].item(), cont[3].item(),
            float(jump.item()), float(fire.item()),
            float(hook.item()), float(weapon.item()),
        ], dtype=np.float32)

        return action, value.item(), log_prob_t.item(), hx_new

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch:    np.ndarray,                                   # (N, OBS_DIM)
        hx_list:      list,                                          # list of N (h, c) tuples
        device:       torch.device = torch.device("cpu"),
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Vectorized inference across N envs in a single GPU call.

        Returns:
            actions:    (N, 8) action array
            values:     (N,) value estimates
            log_probs:  (N,) joint action log probabilities
            hx_new:     list of N new (h, c) tuples
        """
        N = obs_batch.shape[0]
        if N == 0:
            return (np.zeros((0, 8), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.float32),
                    [])

        # Stack hidden states: each is (1, 1, H); we need (1, N, H)
        if _stateful_enabled():
            h_stack = torch.cat([hx[0] for hx in hx_list], dim=1)
            c_stack = torch.cat([hx[1] for hx in hx_list], dim=1)
        else:
            if (self._cached_hx_batch_device != device or
                self._cached_hx_batch_size != N or
                self._cached_hx_batch is None):
                self._cached_hx_batch = self.init_hidden(N, device)
                self._cached_hx_batch_device = device
                self._cached_hx_batch_size = N
            h_stack, c_stack = self._cached_hx_batch

        # Obs: (N, 1, OBS_DIM) — one timestep per env, N envs in the batch
        obs_t = torch.from_numpy(obs_batch).to(device, dtype=torch.float32).unsqueeze(1)

        act_params, value, hx_new = self.forward(obs_t, (h_stack, c_stack))

        cont_mean = act_params["cont_mean"].squeeze(1)              # (N, 4)
        cont_std  = act_params["cont_log_std"].squeeze(1).exp()     # (N, 4)
        cont_dist = torch.distributions.Normal(cont_mean, cont_std)
        cont = cont_mean if deterministic else cont_dist.sample()   # (N, 4)
        log_probs = cont_dist.log_prob(cont).sum(-1)                # (N,)

        def sample_cat(logits):
            logits = logits.squeeze(1)
            d = torch.distributions.Categorical(logits=logits)
            sample = d.probs.argmax(-1) if deterministic else d.sample()
            return sample, d.log_prob(sample)

        jump, jump_lp     = sample_cat(act_params["jump_logits"])
        hook, hook_lp     = sample_cat(act_params["hook_logits"])
        weapon, weapon_lp = sample_cat(act_params["weapon_logits"])
        fire_logits = self.fire_logits_for(
            act_params["feat"], weapon.unsqueeze(-1)
        )
        fire, fire_lp     = sample_cat(fire_logits)
        log_probs = log_probs + jump_lp + fire_lp + hook_lp + weapon_lp

        actions = torch.stack([
            cont[:, 0], cont[:, 1], cont[:, 2], cont[:, 3],
            jump.float(), fire.float(), hook.float(), weapon.float(),
        ], dim=-1)                                                  # (N, 8)

        # Split hidden state back into per-env list
        h_new, c_new = hx_new
        new_hx = [(h_new[:, i:i+1, :].clone(), c_new[:, i:i+1, :].clone())
                  for i in range(N)]

        return (actions.cpu().numpy().astype(np.float32),
                value.squeeze().cpu().numpy().astype(np.float32),
                log_probs.cpu().numpy().astype(np.float32),
                new_hx)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _OnnxWrapper(nn.Module):
    """Deterministic single-pass export: weapon = argmax, fire conditioned on it."""

    def __init__(self, policy: Q2BotPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, h_in, c_in):
        act_params, value, (h_out, c_out) = self.policy(obs, (h_in, c_in))
        weapon_logits = act_params["weapon_logits"]
        weapon = weapon_logits.argmax(-1)
        fire_logits = self.policy.fire_logits_for(act_params["feat"], weapon)
        return (act_params["cont_mean"], act_params["cont_log_std"],
                act_params["jump_logits"], fire_logits,
                act_params["hook_logits"], weapon_logits,
                value, h_out, c_out)


def export_onnx(policy: Q2BotPolicy, path: str, device: torch.device):
    """Export policy to ONNX for deployment via ONNX Runtime in game.so."""
    import torch.onnx

    policy.eval().to(device)
    wrapper = _OnnxWrapper(policy).eval().to(device)

    dummy_obs = torch.zeros(1, 1, OBS_DIM, device=device)
    dummy_h   = torch.zeros(1, 1, HIDDEN_DIM, device=device)
    dummy_c   = torch.zeros(1, 1, HIDDEN_DIM, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_h, dummy_c),
        path,
        input_names  = ["obs", "h_in", "c_in"],
        output_names = ["cont_mean", "cont_log_std",
                        "jump_logits", "fire_logits",
                        "hook_logits", "weapon_logits",
                        "value", "h_out", "c_out"],
        dynamic_axes = {
            "obs": {0: "batch", 1: "seq"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"},
        },
        opset_version = 17,
        do_constant_folding = True,
    )
    print(f"Exported ONNX policy → {path}")
    print(f"Parameters: {policy.param_count():,}")
