"""Fresh 298-input recurrent policy for the multires Atlas generation.

This is a new module graph, not a compatibility mode of ``models.policy``.
Its posture head is categorical ``down | neutral | up`` and every observation
slice comes from the frozen named contract in ``harness.multires_contract``.
Legacy state dictionaries cannot load strictly into this graph.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from harness.multires_contract import (
    ACTION_DIM,
    DYN,
    ENTITIES,
    FACTUAL_DIM,
    GUIDES,
    HOOK_CLASSES,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
    POSTURE_NEUTRAL,
    RECOVERY,
    WEAPON_CLASSES,
)


HIDDEN_DIM = 256
ENTITY_COUNT = 8
ENTITY_DIM = 9
WEAPON_EMBED_DIM = 16
LOG_STD_MIN = -3.0
LOG_STD_MAX = 1.0
CONTINUOUS_LOW = (-1.0, -1.0, -45.0, -30.0)
CONTINUOUS_HIGH = (1.0, 1.0, 45.0, 30.0)

# view is the last two values in the 195-value pre-posture factual prefix.
VIEW_YAW_INDEX = 183
VIEW_PITCH_INDEX = 184


def _continuous_bounds(
    reference: torch.Tensor,
    obs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = reference.new_tensor(CONTINUOUS_LOW).expand_as(reference).clone()
    high = reference.new_tensor(CONTINUOUS_HIGH).expand_as(reference).clone()
    if obs is not None:
        current_pitch = obs[..., VIEW_PITCH_INDEX] * 90.0
        low[..., 3] = torch.maximum(low[..., 3], -89.0 - current_pitch)
        high[..., 3] = torch.minimum(high[..., 3], 89.0 - current_pitch)
    return low, high


def _censored_normal_log_prob(
    distribution: torch.distributions.Normal,
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    standard_low = (low - distribution.loc) / distribution.scale
    standard_high_survival = (distribution.loc - high) / distribution.scale
    lower_mass = torch.special.log_ndtr(standard_low)
    upper_mass = torch.special.log_ndtr(standard_high_survival)
    density = distribution.log_prob(actions)
    return torch.where(
        actions <= low,
        lower_mass,
        torch.where(actions >= high, upper_mass, density),
    )


def _sample_continuous(
    distribution: torch.distributions.Normal,
    deterministic: bool,
    low: torch.Tensor,
    high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = distribution.loc if deterministic else distribution.sample()
    action = torch.maximum(torch.minimum(raw, high), low)
    log_prob = _censored_normal_log_prob(
        distribution, action, low, high
    ).sum(dim=-1)
    return action, log_prob


def _masked_fire_logits(
    fire_logits: torch.Tensor,
    fire_allowed: Optional[torch.Tensor],
) -> torch.Tensor:
    if fire_allowed is None:
        return fire_logits
    allowed = fire_allowed.to(device=fire_logits.device, dtype=torch.bool)
    if allowed.shape != fire_logits.shape[:-1]:
        raise ValueError("fire_allowed shape does not match fire logits")
    closed = torch.zeros_like(fire_logits)
    closed[..., 1] = torch.finfo(fire_logits.dtype).min
    return torch.where(allowed.unsqueeze(-1), fire_logits, closed)


def _wrap_degrees(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def target_fire_allowed(
    obs: torch.Tensor,
    look_action: torch.Tensor,
    *,
    yaw_threshold_deg: float = 12.0,
    pitch_threshold_deg: float = 14.0,
) -> torch.Tensor:
    """Derive a conservative post-command fire gate from factual visibility.

    No Atlas, Dyn, recovery, guide, or teacher-only field participates.  Final
    permission and suppression remain authoritative in game.so.
    """
    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"multires observation must have width {OBS_DIM}")
    if obs.shape[:-1] != look_action.shape[:-1] or look_action.shape[-1] != 2:
        raise ValueError("look_action leading dimensions must match observation")
    if yaw_threshold_deg < 0.0 or pitch_threshold_deg < 0.0:
        raise ValueError("fire-gate thresholds must be nonnegative")

    entities = obs[..., ENTITIES.slice].reshape(
        *obs.shape[:-1], ENTITY_COUNT, ENTITY_DIM
    )
    candidates = (
        (entities[..., 6] > 0.0)
        & (entities[..., 7] > 0.5)
        & (entities[..., 8] > 0.0)
    )
    xyz = entities[..., :3]
    current_pitch = obs[..., VIEW_PITCH_INDEX] * 90.0
    pitch_radians = current_pitch * (torch.pi / 180.0)
    x, y, z = xyz.unbind(dim=-1)
    horizontal_forward = (
        torch.cos(pitch_radians).unsqueeze(-1) * x
        + torch.sin(pitch_radians).unsqueeze(-1) * z
    )
    vertical = (
        -torch.sin(pitch_radians).unsqueeze(-1) * x
        + torch.cos(pitch_radians).unsqueeze(-1) * z
    )
    horizontal_distance = torch.hypot(horizontal_forward, y).clamp_min(1e-6)
    desired_yaw = _wrap_degrees(
        -torch.atan2(y, horizontal_forward) * (180.0 / torch.pi)
    )
    target_pitch = -torch.atan2(vertical, horizontal_distance) * (
        180.0 / torch.pi
    )
    desired_pitch = target_pitch - current_pitch.unsqueeze(-1)

    yaw_command = look_action[..., 0].clamp(-45.0, 45.0)
    pitch_command = look_action[..., 1].clamp(-30.0, 30.0)
    new_pitch = (current_pitch + pitch_command).clamp(-89.0, 89.0)
    effective_pitch = new_pitch - current_pitch
    yaw_residual = _wrap_degrees(-desired_yaw + yaw_command.unsqueeze(-1))
    pitch_residual = -desired_pitch + effective_pitch.unsqueeze(-1)
    aligned = (
        candidates
        & (yaw_residual.abs() <= float(yaw_threshold_deg))
        & (pitch_residual.abs() <= float(pitch_threshold_deg))
    )
    alive = obs[..., 6] > 0.0
    return alive & aligned.any(dim=-1)


class MultiresObservationEncoder(nn.Module):
    """Encode factual, dynamic, recovery, and advisory blocks separately."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        factual_rest_dim = FACTUAL_DIM - ENTITIES.width
        self.entity_embed = nn.Sequential(
            nn.Linear(ENTITY_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )
        self.entity_score = nn.Linear(64, 1)
        self.factual = nn.Sequential(
            nn.Linear(factual_rest_dim, 160), nn.LayerNorm(160), nn.ELU(),
        )
        self.dyn = nn.Sequential(
            nn.Linear(DYN.width, 64), nn.LayerNorm(64), nn.ELU(),
        )
        self.recovery = nn.Sequential(
            nn.Linear(RECOVERY.width, 48), nn.LayerNorm(48), nn.ELU(),
        )
        self.guides = nn.Sequential(
            nn.Linear(GUIDES.width, 96), nn.LayerNorm(96), nn.ELU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 + 64 + 160 + 64 + 48 + 96, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != OBS_DIM:
            raise ValueError(f"multires policy requires exactly {OBS_DIM} inputs")
        leading = observation.shape[:-1]
        entities = observation[..., ENTITIES.slice].reshape(
            *leading, ENTITY_COUNT, ENTITY_DIM
        )
        embedded = self.entity_embed(entities)
        weights = torch.softmax(self.entity_score(embedded), dim=-2)
        entity_attention = (weights * embedded).sum(dim=-2)
        entity_maximum = embedded.max(dim=-2).values
        factual = torch.cat((
            observation[..., :ENTITIES.start],
            observation[..., ENTITIES.stop:FACTUAL_DIM],
        ), dim=-1)
        return self.combine(torch.cat((
            entity_attention,
            entity_maximum,
            self.factual(factual),
            self.dyn(observation[..., DYN.slice]),
            self.recovery(observation[..., RECOVERY.slice]),
            self.guides(observation[..., GUIDES.slice]),
        ), dim=-1))


class MultiresQ2BotPolicy(nn.Module):
    """Fresh recurrent actor/critic with a three-way posture head."""

    policy_generation = POLICY_GENERATION
    observation_dim = OBS_DIM
    action_dim = ACTION_DIM
    categorical_cardinalities = {
        "posture": POSTURE_CLASSES,
        "fire": 2,
        "hook": HOOK_CLASSES,
        "weapon": WEAPON_CLASSES,
    }

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = MultiresObservationEncoder(self.hidden_dim)
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True
        )
        self.actor_cont = nn.Linear(self.hidden_dim, 4)
        self.log_std_head = nn.Linear(self.hidden_dim, 4)
        self.actor_posture = nn.Linear(self.hidden_dim, POSTURE_CLASSES)
        self.actor_hook = nn.Linear(self.hidden_dim, HOOK_CLASSES)
        self.actor_weapon = nn.Linear(self.hidden_dim, WEAPON_CLASSES)
        self.weapon_embed = nn.Embedding(WEAPON_CLASSES, WEAPON_EMBED_DIM)
        self.actor_fire = nn.Linear(self.hidden_dim + WEAPON_EMBED_DIM, 2)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.predict_next = nn.Linear(self.hidden_dim, self.hidden_dim)
        self._initialize_fresh()

    def _initialize_fresh(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_cont.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)
        with torch.no_grad():
            posture_bias = torch.full((POSTURE_CLASSES,), -1.0)
            posture_bias[POSTURE_NEUTRAL] = 2.0
            self.actor_posture.bias.copy_(posture_bias)
            self.actor_hook.bias.copy_(torch.tensor([2.0, -1.0, -3.0, -1.0]))

    def forward(
        self,
        obs: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        encoded = self.encoder(obs)
        features, hx_new = self.lstm(encoded, hx)
        parameters = {
            "feat": features,
            "cont_mean": self.actor_cont(features),
            "cont_log_std": self.log_std_head(features).clamp(
                LOG_STD_MIN, LOG_STD_MAX
            ),
            "posture_logits": self.actor_posture(features),
            "hook_logits": self.actor_hook(features),
            "weapon_logits": self.actor_weapon(features),
        }
        return parameters, self.critic(features), hx_new

    def fire_logits_for(
        self, features: torch.Tensor, weapon_index: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.weapon_embed(weapon_index)
        return self.actor_fire(torch.cat((features, embedding), dim=-1))

    def init_hidden(
        self, batch: int = 1, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch, self.hidden_dim, device=device),
            torch.zeros(1, batch, self.hidden_dim, device=device),
        )

    def action_log_prob_entropy(
        self,
        parameters: dict[str, torch.Tensor],
        actions: torch.Tensor,
        *,
        fire_allowed: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"multires action must have width {ACTION_DIM}")
        continuous_distribution = torch.distributions.Normal(
            parameters["cont_mean"], parameters["cont_log_std"].exp()
        )
        low, high = _continuous_bounds(actions[..., :4], obs)
        log_probability = _censored_normal_log_prob(
            continuous_distribution, actions[..., :4], low, high
        ).sum(dim=-1)
        entropy = torch.zeros_like(log_probability)
        weapon_index = actions[..., 7].round().long().clamp(0, WEAPON_CLASSES - 1)
        fire_logits = _masked_fire_logits(
            self.fire_logits_for(parameters["feat"], weapon_index), fire_allowed
        )
        categorical = (
            (parameters["posture_logits"], 4, POSTURE_CLASSES - 1),
            (fire_logits, 5, 1),
            (parameters["hook_logits"], 6, HOOK_CLASSES - 1),
            (parameters["weapon_logits"], 7, WEAPON_CLASSES - 1),
        )
        for logits, action_index, maximum in categorical:
            distribution = torch.distributions.Categorical(logits=logits)
            index = actions[..., action_index].round().long().clamp(0, maximum)
            log_probability = log_probability + distribution.log_prob(index)
            entropy = entropy + distribution.entropy()
        return log_probability, entropy

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch: np.ndarray,
        hx_list: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        device: torch.device = torch.device("cpu"),
        deterministic: bool = False,
        gate_fire: bool = False,
        fire_gate_yaw_deg: float = 12.0,
        fire_gate_pitch_deg: float = 14.0,
        return_fire_metadata: bool = False,
    ) -> tuple:
        source = np.asarray(obs_batch, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != OBS_DIM:
            raise ValueError(f"obs_batch must have shape (N, {OBS_DIM})")
        count = source.shape[0]
        if count == 0:
            result = (
                np.zeros((0, ACTION_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                [],
            )
            if return_fire_metadata:
                return (*result, {
                    "fire_allowed": np.ones(0, dtype=np.bool_),
                    "raw_fire_probability": np.zeros(0, dtype=np.float32),
                    "raw_fire_log_probability": np.zeros(0, dtype=np.float32),
                })
            return result
        if len(hx_list) != count:
            raise ValueError("one recurrent state is required per observation")
        hidden = torch.cat([state[0] for state in hx_list], dim=1).to(device)
        cell = torch.cat([state[1] for state in hx_list], dim=1).to(device)
        obs = torch.from_numpy(source).to(device=device).unsqueeze(1)
        parameters, values, hx_new = self.forward(obs, (hidden, cell))
        continuous_distribution = torch.distributions.Normal(
            parameters["cont_mean"].squeeze(1),
            parameters["cont_log_std"].squeeze(1).exp(),
        )
        low, high = _continuous_bounds(
            continuous_distribution.loc, obs.squeeze(1)
        )
        continuous, log_probability = _sample_continuous(
            continuous_distribution, deterministic, low, high
        )

        def sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            distribution = torch.distributions.Categorical(logits=logits.squeeze(1))
            value = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            return value, distribution.log_prob(value)

        posture, posture_log_probability = sample(parameters["posture_logits"])
        hook, hook_log_probability = sample(parameters["hook_logits"])
        weapon, weapon_log_probability = sample(parameters["weapon_logits"])
        raw_fire_logits = self.fire_logits_for(
            parameters["feat"], weapon.unsqueeze(-1)
        )
        fire_allowed = torch.ones(count, dtype=torch.bool, device=device)
        if gate_fire:
            fire_allowed = target_fire_allowed(
                obs.squeeze(1),
                continuous[:, 2:4],
                yaw_threshold_deg=fire_gate_yaw_deg,
                pitch_threshold_deg=fire_gate_pitch_deg,
            )
        fire, fire_log_probability = sample(
            _masked_fire_logits(raw_fire_logits, fire_allowed.unsqueeze(-1))
        )
        log_probability = (
            log_probability
            + posture_log_probability
            + fire_log_probability
            + hook_log_probability
            + weapon_log_probability
        )
        actions = torch.stack((
            continuous[:, 0],
            continuous[:, 1],
            continuous[:, 2],
            continuous[:, 3],
            posture.float(),
            fire.float(),
            hook.float(),
            weapon.float(),
        ), dim=-1)
        hidden_new, cell_new = hx_new
        states = [
            (
                hidden_new[:, index:index + 1].clone(),
                cell_new[:, index:index + 1].clone(),
            )
            for index in range(count)
        ]
        result = (
            actions.cpu().numpy().astype(np.float32),
            values.squeeze(1).squeeze(-1).cpu().numpy().astype(np.float32),
            log_probability.cpu().numpy().astype(np.float32),
            states,
        )
        if not return_fire_metadata:
            return result
        metadata = {
            "fire_allowed": fire_allowed.cpu().numpy().astype(np.bool_),
            "raw_fire_probability": raw_fire_logits.softmax(dim=-1)[
                ..., 1
            ].squeeze(-1).cpu().numpy().astype(np.float32),
            "raw_fire_log_probability": raw_fire_logits.log_softmax(dim=-1)[
                ..., 1
            ].squeeze(-1).cpu().numpy().astype(np.float32),
        }
        return (*result, metadata)

    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MultiresOnnxWrapper(nn.Module):
    def __init__(self, policy: MultiresQ2BotPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, hidden, cell):
        parameters, value, (hidden_out, cell_out) = self.policy(
            obs, (hidden, cell)
        )
        weapon_logits = parameters["weapon_logits"]
        weapon = weapon_logits.argmax(dim=-1)
        fire_logits = self.policy.fire_logits_for(parameters["feat"], weapon)
        return (
            parameters["cont_mean"],
            parameters["cont_log_std"],
            parameters["posture_logits"],
            fire_logits,
            parameters["hook_logits"],
            weapon_logits,
            value,
            hidden_out,
            cell_out,
        )
