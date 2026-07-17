"""Live composition root: public client batch -> causal collector -> PPO.

No symbol from ``train.ppo``, ``harness.spatial``, or the retired model graph is
imported here.  Selecting this trainer therefore has one executable lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from harness.multires_admission import SPATIAL_ATTESTATION_INFO_KEY
from harness.multires_collector import (
    CollectedMultiresRollout,
    CollectorConfig,
    MultiresSynchronousCollector,
    PolicyDecision,
    TransitionObserver,
)
from harness.multires_contract import GuideDropoutIdentity, OBS_DIM
from .multires_ppo import MultiresPPOTrainer, MultiresRolloutBatch
from .multires_runtime import MultiresTrainerRuntime


class TorchMultiresCollectorPolicy:
    """Thin recurrent adapter around the fresh policy's batched actor API."""

    def __init__(
        self,
        runtime: MultiresTrainerRuntime,
        *,
        device: torch.device,
        deterministic: bool = False,
    ):
        self.runtime = runtime
        self.policy = runtime.policy
        self.device = device
        self.deterministic = bool(deterministic)

    def initial_state(self, client_count: int):
        return self.policy.init_hidden(client_count, device=self.device)

    def _reset_state(self, state: Any, reset: np.ndarray):
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("collector recurrent state is malformed")
        mask = torch.as_tensor(
            ~np.asarray(reset, dtype=np.bool_),
            dtype=state[0].dtype,
            device=self.device,
        ).view(1, -1, 1)
        return state[0].to(self.device) * mask, state[1].to(self.device) * mask

    def act(self, observations, state, recurrent_reset) -> PolicyDecision:
        state = self._reset_state(state, recurrent_reset)
        count = observations.shape[0]
        state_list = [
            (
                state[0][:, index:index + 1],
                state[1][:, index:index + 1],
            )
            for index in range(count)
        ]
        actions, values, log_probabilities, next_states, metadata = (
            self.policy.act_batch(
                observations,
                state_list,
                device=self.device,
                deterministic=self.deterministic,
                gate_fire=True,
                return_fire_metadata=True,
            )
        )
        next_state = (
            torch.cat([item[0] for item in next_states], dim=1),
            torch.cat([item[1] for item in next_states], dim=1),
        )
        return PolicyDecision(
            actions=actions,
            values=values,
            log_probabilities=log_probabilities,
            next_state=next_state,
            fire_allowed=metadata["fire_allowed"],
            raw_fire_log_probability=metadata["raw_fire_log_probability"],
        )

    @torch.no_grad()
    def values(self, observations, state, recurrent_reset):
        state = self._reset_state(state, recurrent_reset)
        source = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        _parameters, values, _next_state = self.policy(source, state)
        return values[:, 0, 0].detach().cpu().numpy().astype(np.float32)


def _tensor_batch(
    rollout: CollectedMultiresRollout,
    *,
    device: torch.device,
) -> MultiresRolloutBatch:
    rollout.validate()
    initial_hidden = rollout.initial_state
    if initial_hidden is not None:
        initial_hidden = tuple(value.detach().to(device) for value in initial_hidden)

    def tensor(value: np.ndarray, dtype: torch.dtype = torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    return MultiresRolloutBatch(
        observations=tensor(rollout.observations),
        actions=tensor(rollout.actions),
        old_log_probabilities=tensor(rollout.old_log_probabilities),
        old_values=tensor(rollout.old_values),
        advantages=tensor(rollout.advantages),
        returns=tensor(rollout.returns),
        fire_allowed=tensor(rollout.fire_allowed, torch.bool),
        valid=tensor(rollout.valid, torch.bool),
        recurrent_reset=tensor(rollout.recurrent_reset, torch.bool),
        initial_hidden=initial_hidden,
    )


@dataclass(frozen=True)
class MultiresTrainingUpdate:
    rollout: CollectedMultiresRollout
    ppo_metrics: Mapping[str, float | int]


class MultiresLiveTrainer:
    """One stateful live trainer with exact policy-version batch boundaries."""

    def __init__(
        self,
        runtime: MultiresTrainerRuntime,
        network_batch: Any,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        collector_config: CollectorConfig = CollectorConfig(),
        deterministic_collection: bool = False,
        transition_observer: TransitionObserver | None = None,
    ):
        if not runtime.runtime.runtime_manifest_sha256:
            raise ValueError("live multires trainer requires sealed runtime evidence")
        self.runtime = runtime
        self.device = device
        self.policy_adapter = TorchMultiresCollectorPolicy(
            runtime, device=device, deterministic=deterministic_collection
        )
        self.collector = MultiresSynchronousCollector(
            network_batch,
            self.policy_adapter,
            observation_transform=self._transform_observations,
            reward_function=runtime.reward,
            reset_reward_stream=runtime.reset_reward_stream,
            transition_observer=transition_observer,
            config=collector_config,
        )
        self.optimizer = optimizer
        self.ppo = MultiresPPOTrainer(
            runtime.policy, optimizer, runtime.ppo_config
        )

    def _transform_observations(
        self,
        observations: np.ndarray,
        infos: Sequence[Mapping[str, Any]],
        policy_version: int,
    ) -> np.ndarray:
        if observations.shape != (len(infos), OBS_DIM):
            raise ValueError("live multires observation/info cardinality differs")
        transformed = np.empty_like(observations, dtype=np.float32)
        for index, (vector, info) in enumerate(zip(observations, infos)):
            attestation = info.get(SPATIAL_ATTESTATION_INFO_KEY)
            if not isinstance(attestation, Mapping):
                raise ValueError("observation lacks spatial provider attestation")
            expected = {
                "atlas_sha256": self.runtime.runtime.atlas_sha256,
                "runtime_manifest_sha256": (
                    self.runtime.runtime.runtime_manifest_sha256
                ),
            }
            for name, wanted in expected.items():
                if attestation.get(name) != wanted:
                    raise ValueError(
                        f"observation {name} differs from admitted runtime"
                    )
            transformed[index], _dropped = self.runtime.prepare_policy_vector(
                vector,
                dropout_identity=GuideDropoutIdentity(
                    map_name=str(info["map"]),
                    policy_version=policy_version,
                    client_id=str(info["client_id"]),
                    tick=int(info["server_frame"]),
                ),
                training=True,
            )
        return transformed

    def train_update(self, *, policy_version: int) -> MultiresTrainingUpdate:
        rollout = self.collector.collect(policy_version=policy_version)
        metrics = self.ppo.update(_tensor_batch(rollout, device=self.device))
        return MultiresTrainingUpdate(rollout=rollout, ppo_metrics=metrics)
