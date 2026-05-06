"""
policy.py — LSTM policy network for the Q2 ML bot.

Architecture:
  obs_vector (185,)
    → encoder MLP → (256,)
    → LSTM (hidden=256, layers=1) → (256,)
    → actor head → action logits / means
    → critic head → value scalar

Designed to be small enough to train on the Vega 10 iGPU overnight
and export to ONNX for in-game inference via ONNX Runtime.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

OBS_DIM    = 185
HIDDEN_DIM = 256
ACTION_DIM = 8   # [fwd, right, yaw, pitch, jump, fire, hook, weapon]


class ObsEncoder(nn.Module):
    """Projects raw obs vector into a compact feature embedding."""

    def __init__(self, obs_dim: int = OBS_DIM, out_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ELU(),
            nn.Linear(256, out_dim), nn.LayerNorm(out_dim), nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Q2BotPolicy(nn.Module):
    """
    Actor-critic LSTM policy.

    forward() accepts a sequence of observations and optional LSTM state.
    Returns action distribution parameters, value estimate, and new hidden state.

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
        self.actor_cont  = nn.Linear(hidden_dim, 4)
        self.log_std     = nn.Parameter(torch.zeros(4))

        # actor: discrete actions for [jump, fire, hook, weapon]
        self.actor_jump  = nn.Linear(hidden_dim, 2)   # binary
        self.actor_fire  = nn.Linear(hidden_dim, 2)
        self.actor_hook  = nn.Linear(hidden_dim, 4)   # 0-3
        self.actor_weapon = nn.Linear(hidden_dim, 10) # 0-9

        # critic
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_cont.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        hx:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[dict, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            act_params: dict of distribution parameters
            value:      (batch, seq, 1)
            hx_new:     new LSTM state
        """
        B, T, _ = obs.shape

        # encode each timestep independently
        enc = self.encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)

        # temporal context
        lstm_out, hx_new = self.lstm(enc, hx)

        feat = lstm_out  # (B, T, H)

        act_params = {
            "cont_mean": self.actor_cont(feat),                     # (B, T, 4)
            "cont_log_std": self.log_std.expand(B, T, -1),
            "jump_logits":  self.actor_jump(feat),                  # (B, T, 2)
            "fire_logits":  self.actor_fire(feat),
            "hook_logits":  self.actor_hook(feat),                  # (B, T, 4)
            "weapon_logits": self.actor_weapon(feat),               # (B, T, 10)
        }
        value = self.critic(feat)                                   # (B, T, 1)

        return act_params, value, hx_new

    def init_hidden(
        self, batch: int = 1, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        c = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h, c

    @torch.no_grad()
    def act(
        self,
        obs_vec: np.ndarray,
        hx: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step inference for in-game use.

        Returns: (action_array (8,), value_scalar, new_hx)
        """
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device)
        obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # (1, 1, OBS_DIM)

        act_params, value, hx_new = self.forward(obs_t, hx)

        # sample continuous actions
        if deterministic:
            cont = act_params["cont_mean"].squeeze()
        else:
            std  = act_params["cont_log_std"].exp().squeeze()
            cont = torch.distributions.Normal(
                act_params["cont_mean"].squeeze(), std
            ).sample()

        # sample discrete actions
        def sample_cat(logits, det):
            d = torch.distributions.Categorical(logits=logits.squeeze(0).squeeze(0))
            return d.probs.argmax() if det else d.sample()

        jump   = sample_cat(act_params["jump_logits"],   deterministic).item()
        fire   = sample_cat(act_params["fire_logits"],   deterministic).item()
        hook   = sample_cat(act_params["hook_logits"],   deterministic).item()
        weapon = sample_cat(act_params["weapon_logits"], deterministic).item()

        action = np.array([
            cont[0].item(), cont[1].item(),
            cont[2].item(), cont[3].item(),
            float(jump), float(fire), float(hook), float(weapon),
        ], dtype=np.float32)

        return action, value.item(), hx_new

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def export_onnx(policy: Q2BotPolicy, path: str, device: torch.device):
    """Export policy to ONNX for deployment via ONNX Runtime in game.so."""
    import torch.onnx

    policy.eval().to(device)

    dummy_obs = torch.zeros(1, 1, OBS_DIM, device=device)
    dummy_h   = torch.zeros(1, 1, HIDDEN_DIM, device=device)
    dummy_c   = torch.zeros(1, 1, HIDDEN_DIM, device=device)

    torch.onnx.export(
        policy,
        (dummy_obs, (dummy_h, dummy_c)),
        path,
        input_names  = ["obs", "h_in", "c_in"],
        output_names = ["cont_mean", "cont_log_std",
                        "jump_logits", "fire_logits",
                        "hook_logits", "weapon_logits",
                        "value", "h_out", "c_out"],
        dynamic_axes = {"obs": {0: "batch", 1: "seq"}},
        opset_version = 17,
        do_constant_folding = True,
    )
    print(f"Exported ONNX policy → {path}")
    print(f"Parameters: {policy.param_count():,}")
