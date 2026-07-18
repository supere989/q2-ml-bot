"""PPO update kernel for the fresh multires policy graph.

Collection remains owned by the B4 synchronous client batch.  This module
admits only exact 298-observation/eight-action tensors plus the authoritative
effective-fire mask needed to replay server suppression causally.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from harness.multires_contract import ACTION_DIM, OBS_DIM, POSTURE_CLASSES
from models.multires_policy import MultiresQ2BotPolicy


@dataclass(frozen=True)
class MultiresPPOConfig:
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    normalize_advantage: bool = True

    def validate(self) -> None:
        for name in ("clip_coef", "value_coef", "entropy_coef", "max_grad_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


@dataclass
class MultiresRolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    fire_allowed: torch.Tensor
    valid: torch.Tensor
    recurrent_reset: torch.Tensor
    initial_hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def validate(self) -> None:
        if self.observations.ndim != 3 or self.observations.shape[-1] != OBS_DIM:
            raise ValueError(f"observations must have shape (batch, time, {OBS_DIM})")
        leading = self.observations.shape[:2]
        if self.actions.shape != (*leading, ACTION_DIM):
            raise ValueError(f"actions must have shape {leading + (ACTION_DIM,)}")
        for name in (
            "old_log_probabilities", "old_values", "advantages", "returns",
            "fire_allowed", "valid", "recurrent_reset",
        ):
            value = getattr(self, name)
            if value.shape != leading:
                raise ValueError(f"{name} must have shape {leading}")
        for name in ("fire_allowed", "valid", "recurrent_reset"):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean tensor")
        floating = (
            self.observations,
            self.actions,
            self.old_log_probabilities,
            self.old_values,
            self.advantages,
            self.returns,
        )
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("rollout batch contains a non-finite tensor")
        if not bool(self.valid.any()):
            raise ValueError("rollout batch has no trainable transitions")
        posture = self.actions[..., 4]
        if not torch.equal(posture, posture.round()) or bool(
            ((posture < 0) | (posture >= POSTURE_CLASSES)).any()
        ):
            raise ValueError("posture actions must be categorical values 0, 1, or 2")
        for action_index, maximum, name in (
            (5, 1, "fire"), (6, 3, "hook"), (7, 9, "weapon")
        ):
            values = self.actions[..., action_index]
            if not torch.equal(values, values.round()) or bool(
                ((values < 0) | (values > maximum)).any()
            ):
                raise ValueError(f"{name} action is outside its categorical range")
        if self.initial_hidden is not None:
            hidden, cell = self.initial_hidden
            expected = (1, leading[0])
            if hidden.ndim != 3 or cell.ndim != 3:
                raise ValueError("initial recurrent state must be rank three")
            if hidden.shape[:2] != expected or cell.shape != hidden.shape:
                raise ValueError("initial recurrent state does not match batch size")
            if not torch.isfinite(hidden).all() or not torch.isfinite(cell).all():
                raise ValueError("initial recurrent state contains non-finite values")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    if selected.numel() == 0:
        raise ValueError("masked PPO reduction has no samples")
    return selected.mean()


class MultiresPPOTrainer:
    def __init__(
        self,
        policy: MultiresQ2BotPolicy,
        optimizer: torch.optim.Optimizer,
        config: MultiresPPOConfig = MultiresPPOConfig(),
    ):
        if getattr(policy, "observation_dim", None) != OBS_DIM:
            raise ValueError("trainer policy is not the fresh 298-input generation")
        config.validate()
        self.policy = policy
        self.optimizer = optimizer
        self.config = config

    def _forward_recurrent_batch(
        self, batch: MultiresRolloutBatch
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch_size, time_steps = batch.observations.shape[:2]
        if batch.initial_hidden is None:
            hidden = self.policy.init_hidden(
                batch_size, device=batch.observations.device
            )
        else:
            hidden = batch.initial_hidden
        parameter_steps: dict[str, list[torch.Tensor]] = {}
        value_steps: list[torch.Tensor] = []
        for time_index in range(time_steps):
            reset = batch.recurrent_reset[:, time_index].view(1, batch_size, 1)
            keep = (~reset).to(dtype=hidden[0].dtype)
            hidden = (hidden[0] * keep, hidden[1] * keep)
            parameters, values, hidden = self.policy(
                batch.observations[:, time_index:time_index + 1], hidden
            )
            if not parameter_steps:
                parameter_steps = {name: [] for name in parameters}
            for name, value in parameters.items():
                parameter_steps[name].append(value)
            value_steps.append(values)
        return (
            {
                name: torch.cat(values, dim=1)
                for name, values in parameter_steps.items()
            },
            torch.cat(value_steps, dim=1),
        )

    def update(self, batch: MultiresRolloutBatch) -> dict[str, float | int]:
        batch.validate()
        mask = batch.valid.to(dtype=torch.bool)
        advantages = batch.advantages
        if self.config.normalize_advantage:
            selected = advantages[mask]
            advantages = advantages.clone()
            advantages[mask] = (
                selected - selected.mean()
            ) / selected.std(unbiased=False).clamp_min(1e-8)

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        optimizer_steps = 0
        self.policy.train()
        for _epoch in range(self.config.epochs):
            parameters, values = self._forward_recurrent_batch(batch)
            values = values.squeeze(-1)
            log_probabilities, entropy = self.policy.action_log_prob_entropy(
                parameters,
                batch.actions,
                fire_allowed=batch.fire_allowed.to(dtype=torch.bool),
                obs=batch.observations,
            )
            log_ratio = log_probabilities - batch.old_log_probabilities
            ratio = log_ratio.exp()
            unclipped = -advantages * ratio
            clipped = -advantages * ratio.clamp(
                1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
            )
            policy_loss = _masked_mean(torch.maximum(unclipped, clipped), mask)
            value_loss = 0.5 * _masked_mean(
                (values - batch.returns).square(), mask
            )
            entropy_mean = _masked_mean(entropy, mask)
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy_mean
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite multires PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite multires PPO gradient norm")
            self.optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)
                clip_fraction = _masked_mean(
                    (torch.abs(ratio - 1.0) > self.config.clip_coef).float(), mask
                )
            for name, value in (
                ("loss", loss),
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy", entropy_mean),
                ("approx_kl", approx_kl),
                ("clip_fraction", clip_fraction),
                ("grad_norm", grad_norm),
            ):
                totals[name] += float(value.detach().cpu())
        averaged: dict[str, float | int] = {
            name: value / self.config.epochs for name, value in totals.items()
        }
        # This is a transaction fact, not an averaged PPO metric.  Keep it an
        # exact integer so the outer evidence loop cannot confuse one policy
        # publication with the multiple optimizer.step() calls that produced
        # it.
        averaged["optimizer_steps"] = optimizer_steps
        return averaged
