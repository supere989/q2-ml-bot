"""Deterministic synchronous collector for the network-native multires path.

This module deliberately has no PyTorch dependency.  Echo/admission behavior
can therefore be qualified in-process on deployment hosts before a GPU trainer
is imported.  A small adapter in ``train.multires_live`` connects the same
records to the recurrent policy and PPO tensor batch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .causal_protocol import CausalTelemetry
from .client_batch import BatchRound, decode_policy_action
from .multires_admission import (
    CausalRewardAdmission,
    PRIVATE_CAUSAL_INFO_KEY,
    PRIVATE_SPATIAL_REWARD_INFO_KEY,
    SPATIAL_ATTESTATION_INFO_KEY,
    PrivateSpatialRewardEvidence,
    detach_private_causal,
    detach_private_spatial_reward,
)
from .multires_contract import ACTION_DIM, OBS_DIM
from .multires_reward import (
    CausalRewardFrame,
    CausalRewardResult,
)


class CollectorAdmissionError(RuntimeError):
    """Raised before a non-causal or mixed-version sample can enter PPO."""


@dataclass(frozen=True)
class PolicyDecision:
    actions: np.ndarray
    values: np.ndarray
    log_probabilities: np.ndarray
    next_state: Any
    fire_allowed: np.ndarray
    raw_fire_log_probability: np.ndarray


class CollectorPolicy(Protocol):
    def initial_state(self, client_count: int) -> Any:
        ...

    def act(
        self,
        observations: np.ndarray,
        state: Any,
        recurrent_reset: np.ndarray,
    ) -> PolicyDecision:
        ...

    def values(
        self,
        observations: np.ndarray,
        state: Any,
        recurrent_reset: np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class CollectorConfig:
    transitions_per_client: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    maximum_boundary_rounds: int = 64

    def validate(self) -> None:
        if self.transitions_per_client < 1:
            raise ValueError("transitions_per_client must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be within [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        if self.maximum_boundary_rounds < 0:
            raise ValueError("maximum_boundary_rounds cannot be negative")


@dataclass(frozen=True)
class CollectedMultiresRollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    fire_allowed: np.ndarray
    valid: np.ndarray
    recurrent_reset: np.ndarray
    terminated: np.ndarray
    initial_state: Any
    policy_version: int
    boundary_rounds: int
    infos: tuple[tuple[Mapping[str, Any], ...], ...]

    def validate(self) -> None:
        if self.observations.ndim != 3 or self.observations.shape[-1] != OBS_DIM:
            raise CollectorAdmissionError(
                f"rollout observations must have shape (clients, time, {OBS_DIM})"
            )
        leading = self.observations.shape[:2]
        if self.actions.shape != (*leading, ACTION_DIM):
            raise CollectorAdmissionError("rollout actions have the wrong shape")
        for name in (
            "old_log_probabilities", "old_values", "rewards", "advantages",
            "returns", "fire_allowed", "valid", "recurrent_reset", "terminated",
        ):
            if getattr(self, name).shape != leading:
                raise CollectorAdmissionError(f"rollout {name} has the wrong shape")
        for name in ("fire_allowed", "valid", "recurrent_reset", "terminated"):
            if getattr(self, name).dtype != np.bool_:
                raise CollectorAdmissionError(f"rollout {name} must be boolean")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.observations, self.actions, self.old_log_probabilities,
                self.old_values, self.rewards, self.advantages, self.returns,
            )
        ):
            raise CollectorAdmissionError("rollout contains NaN or infinity")
        if not self.valid.all():
            raise CollectorAdmissionError(
                "collector may store only fully admitted trainable transitions"
            )

    def deterministic_sha256(self) -> str:
        """Hash training tensors and provenance, excluding object identities."""
        self.validate()
        digest = hashlib.sha256()
        digest.update(b"q2-multires-collected-rollout-v1\0")
        digest.update(int(self.policy_version).to_bytes(8, "little", signed=False))
        for value in (
            self.observations, self.actions, self.old_log_probabilities,
            self.old_values, self.rewards, self.advantages, self.returns,
            self.fire_allowed, self.valid, self.recurrent_reset, self.terminated,
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
        for time_infos in self.infos:
            for info in time_infos:
                identity = (
                    str(info["client_id"]), int(info["server_frame"]),
                    int(info["batch_round_id"]), int(info["policy_version"]),
                )
                digest.update(repr(identity).encode("utf-8") + b"\0")
        return digest.hexdigest()


ObservationTransform = Callable[[np.ndarray, Sequence[Mapping[str, Any]], int], np.ndarray]
RewardFunction = Callable[[str, CausalRewardFrame], CausalRewardResult]
RewardStreamReset = Callable[[str], None]
TransitionObserver = Callable[
    [str, CausalRewardFrame, CausalRewardResult, Mapping[str, Any]], None
]


def _detached_public_info(info: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an observer-only copy with no private causal object reachable."""
    private_types = (CausalTelemetry, PrivateSpatialRewardEvidence)

    def reject_private(value: Any, label: str) -> None:
        if isinstance(value, private_types):
            raise CollectorAdmissionError(
                f"observer public info contains private causal payload at {label}"
            )
        if isinstance(value, Mapping):
            for key, item in value.items():
                reject_private(item, f"{label}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                reject_private(item, f"{label}[{index}]")

    if PRIVATE_CAUSAL_INFO_KEY in info or PRIVATE_SPATIAL_REWARD_INFO_KEY in info:
        raise CollectorAdmissionError(
            "observer public info still contains private causal keys"
        )
    detached = copy.deepcopy(dict(info))
    reject_private(detached, "info")
    return MappingProxyType(detached)


_POLICY_INFO_REQUIRED = (
    "client_id",
    "map",
    "server_frame",
    SPATIAL_ATTESTATION_INFO_KEY,
)
_POLICY_INFO_OPTIONAL = (
    "client_slot",
    "map_epoch",
)
_POLICY_ATTESTATION_KEYS = frozenset({
    "schema",
    "feature_schema_sha256",
    "atlas_sha256",
    "runtime_manifest_sha256",
    "map_epoch",
})


def _freeze_policy_value(value: Any, label: str) -> Any:
    """Copy a public policy identity value into an immutable value graph."""
    if isinstance(value, (CausalTelemetry, PrivateSpatialRewardEvidence)):
        raise CollectorAdmissionError(
            f"policy info contains private causal payload at {label}"
        )
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        frozen = {
            str(key): _freeze_policy_value(item, f"{label}.{key}")
            for key, item in value.items()
        }
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_policy_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise CollectorAdmissionError(
        f"policy info contains unsupported value at {label}: "
        f"{type(value).__name__}"
    )


def _policy_info_projection(
    infos: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Expose only immutable public identity needed for policy composition.

    Rich batch diagnostics intentionally retain causal/lifecycle facts.  This
    separate allowlist prevents those facts, private reward objects, or future
    diagnostic additions from becoming reachable by an observation transform.
    """
    projected = []
    for index, info in enumerate(infos):
        missing = [name for name in _POLICY_INFO_REQUIRED if name not in info]
        if missing:
            raise CollectorAdmissionError(
                f"policy info {index} omits public identity fields: {missing!r}"
            )
        attestation = info[SPATIAL_ATTESTATION_INFO_KEY]
        if not isinstance(attestation, Mapping) or set(attestation) != set(
            _POLICY_ATTESTATION_KEYS
        ):
            raise CollectorAdmissionError(
                f"policy info {index} spatial attestation schema differs"
            )
        selected = {}
        for name in (*_POLICY_INFO_REQUIRED, *_POLICY_INFO_OPTIONAL):
            if name in info:
                selected[name] = _freeze_policy_value(
                    info[name], f"info[{index}].{name}"
                )
        projected.append(MappingProxyType(selected))
    return tuple(projected)


def _validate_observations(value: Any, client_count: int) -> np.ndarray:
    observations = np.asarray(value, dtype=np.float32)
    if observations.shape != (client_count, OBS_DIM):
        raise CollectorAdmissionError(
            f"collector expected {(client_count, OBS_DIM)} observations, "
            f"received {observations.shape}"
        )
    if not np.isfinite(observations).all():
        raise CollectorAdmissionError("collector observations contain NaN or infinity")
    return observations


def _validate_decision(decision: PolicyDecision, client_count: int) -> None:
    expected = {
        "actions": (client_count, ACTION_DIM),
        "values": (client_count,),
        "log_probabilities": (client_count,),
        "fire_allowed": (client_count,),
        "raw_fire_log_probability": (client_count,),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(decision, name))
        if value.shape != shape:
            raise CollectorAdmissionError(
                f"policy decision {name} must have shape {shape}"
            )
        if name not in ("fire_allowed",) and not np.isfinite(value).all():
            raise CollectorAdmissionError(f"policy decision {name} is non-finite")
    if np.asarray(decision.fire_allowed).dtype != np.bool_:
        raise CollectorAdmissionError("policy fire_allowed must be boolean")


class MultiresSynchronousCollector:
    def __init__(
        self,
        batch: Any,
        policy: CollectorPolicy,
        *,
        observation_transform: ObservationTransform,
        reward_function: RewardFunction,
        reset_reward_stream: RewardStreamReset | None = None,
        transition_observer: TransitionObserver | None = None,
        config: CollectorConfig = CollectorConfig(),
    ):
        config.validate()
        if not bool(getattr(batch, "vector", True)):
            raise ValueError("multires collector requires vector network batching")
        if not callable(observation_transform) or not callable(reward_function):
            raise TypeError("multires collector requires transform and reward admission")
        if transition_observer is not None and not callable(transition_observer):
            raise TypeError("multires transition observer must be callable")
        self.batch = batch
        self.policy = policy
        self.observation_transform = observation_transform
        self.reward_function = reward_function
        self.reset_reward_stream = reset_reward_stream
        self.transition_observer = transition_observer
        self.config = config
        self._started = False
        self._current_raw: np.ndarray | None = None
        self._current_infos: tuple[Mapping[str, Any], ...] | None = None
        self._causal_admission = CausalRewardAdmission()
        self._failed = False

    def prime(
        self,
        observations: Any,
        infos: Sequence[Mapping[str, Any]],
    ) -> None:
        """Adopt one already-started direct-batch reset transaction.

        Operational launchers use this to capture every client PID immediately
        after ``Q2NetworkClientBatch.reset`` without performing a second reset
        or reaching into collector state.
        """
        if self._started:
            raise CollectorAdmissionError("collector is already started")
        client_count = len(infos)
        if client_count < 1:
            raise CollectorAdmissionError("collector batch has no clients")
        current = _validate_observations(observations, client_count)
        copied = tuple(dict(value) for value in infos)
        self._reset_causal_streams(copied)
        self._current_raw = current.copy()
        self._current_infos = copied
        self._started = True

    def _reset_causal_streams(
        self, infos: Sequence[Mapping[str, Any]]
    ) -> None:
        self._causal_admission.reset()
        if self.reset_reward_stream is None:
            return
        client_ids = {
            str(info.get("client_id", "")) for info in infos
            if info.get("client_id")
        }
        for client_id in sorted(client_ids):
            self.reset_reward_stream(client_id)

    def collect(self, *, policy_version: int) -> CollectedMultiresRollout:
        if self._failed:
            raise CollectorAdmissionError(
                "collector is failed; construct a new synchronized collector"
            )
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if not self._started:
            reset_observations, reset_infos = self.batch.reset()
            client_count = len(reset_infos)
            current_raw = _validate_observations(reset_observations, client_count)
            current_infos = tuple(dict(value) for value in reset_infos)
            self._reset_causal_streams(current_infos)
            self._started = True
        else:
            if self._current_raw is None or self._current_infos is None:
                raise CollectorAdmissionError("collector continuation state is missing")
            client_count = len(self._current_infos)
            current_raw = self._current_raw.copy()
            current_infos = tuple(dict(value) for value in self._current_infos)
        if client_count < 1:
            raise CollectorAdmissionError("collector batch has no clients")
        state = self.policy.initial_state(client_count)
        initial_state = state
        reset_mask = np.ones(client_count, dtype=np.bool_)

        observation_steps: list[np.ndarray] = []
        action_steps: list[np.ndarray] = []
        log_probability_steps: list[np.ndarray] = []
        value_steps: list[np.ndarray] = []
        reward_steps: list[np.ndarray] = []
        fire_allowed_steps: list[np.ndarray] = []
        reset_steps: list[np.ndarray] = []
        terminal_steps: list[np.ndarray] = []
        info_steps: list[tuple[Mapping[str, Any], ...]] = []
        pending_observer_events: list[
            tuple[
                str,
                CausalRewardFrame,
                CausalRewardResult,
                Mapping[str, Any],
            ]
        ] = []
        boundary_rounds = 0

        while len(observation_steps) < self.config.transitions_per_client:
            policy_infos = _policy_info_projection(current_infos)
            policy_observations = self.observation_transform(
                current_raw.copy(), policy_infos, policy_version
            )
            policy_observations = _validate_observations(
                policy_observations, client_count
            )
            decision = self.policy.act(
                policy_observations, state, reset_mask.copy()
            )
            _validate_decision(decision, client_count)
            sampled_actions = np.asarray(decision.actions, dtype=np.float32).copy()
            recorded_actions = sampled_actions.copy()
            recorded_log_probabilities = np.asarray(
                decision.log_probabilities, dtype=np.float32
            ).copy()
            recorded_fire_allowed = np.asarray(
                decision.fire_allowed, dtype=np.bool_
            ).copy()
            raw_fire_log_probability = np.asarray(
                decision.raw_fire_log_probability, dtype=np.float32
            )
            game_actions = [decode_policy_action(row) for row in sampled_actions]
            result: BatchRound = self.batch.collect_round(
                game_actions, policy_version=policy_version
            )
            if result.policy_version != policy_version:
                raise CollectorAdmissionError("batch returned a mixed policy version")
            trainable = [info.get("trainable_transition") is True for info in result.infos]
            if not all(trainable):
                if any(trainable):
                    raise CollectorAdmissionError(
                        "partial batch admission violates synchronous collection"
                    )
                boundary_rounds += 1
                if boundary_rounds > self.config.maximum_boundary_rounds:
                    raise CollectorAdmissionError(
                        "too many non-trainable boundary rounds"
                    )
                current_raw = _validate_observations(
                    result.observations, client_count
                )
                current_infos = tuple(dict(value) for value in result.infos)
                self._reset_causal_streams(current_infos)
                state = decision.next_state
                reset_mask = np.ones(client_count, dtype=np.bool_)
                continue

            sanitized_infos: list[Mapping[str, Any]] = []
            round_observer_events: list[
                tuple[
                    str,
                    CausalRewardFrame,
                    CausalRewardResult,
                    Mapping[str, Any],
                ]
            ] = []
            rewards = np.empty(client_count, dtype=np.float32)
            for index, (raw_info, tag) in enumerate(zip(result.infos, result.tags)):
                info = dict(raw_info)
                if (
                    tag.policy_version != policy_version
                    or tag.client_id != info.get("client_id")
                    or tag.client_index != index
                ):
                    raise CollectorAdmissionError("batch action tag identity differs")
                sampled_fire = bool(sampled_actions[index, 5])
                if info.get("requested_action_fire") is not sampled_fire:
                    raise CollectorAdmissionError(
                        "batch fire request differs from sampled policy action"
                    )
                effective_fire = info.get("effective_action_fire")
                suppressed = info.get("fire_gate_suppressed") is True
                if suppressed:
                    if not sampled_fire or not recorded_fire_allowed[index]:
                        raise CollectorAdmissionError(
                            "server suppression occurred outside sampled open fire gate"
                        )
                    recorded_log_probabilities[index] -= raw_fire_log_probability[index]
                    recorded_actions[index, 5] = 0.0
                    recorded_fire_allowed[index] = False
                elif effective_fire is not sampled_fire:
                    raise CollectorAdmissionError(
                        "effective fire differs without authoritative suppression"
                    )
                private = detach_private_causal(info)
                spatial_reward = detach_private_spatial_reward(info)
                frame = self._causal_admission.admit(
                    private, spatial_reward, info
                )
                reward_result = self.reward_function(str(info["client_id"]), frame)
                if not isinstance(reward_result, CausalRewardResult):
                    raise CollectorAdmissionError(
                        "reward reducer must return CausalRewardResult"
                    )
                reward_value = reward_result.reward
                if type(reward_value) not in (int, float) or not np.isfinite(reward_value):
                    raise CollectorAdmissionError("reward reducer returned a non-finite value")
                if not isinstance(reward_result.metrics, Mapping) or any(
                    not isinstance(name, str)
                    or type(value) not in (int, float)
                    or not np.isfinite(value)
                    for name, value in reward_result.metrics.items()
                ):
                    raise CollectorAdmissionError(
                        "reward reducer returned invalid causal metrics"
                    )
                rewards[index] = float(reward_value)
                sanitized_infos.append(info)
                round_observer_events.append((
                    str(info["client_id"]),
                    frame,
                    CausalRewardResult(
                        reward=float(reward_result.reward),
                        metrics={
                            name: float(value)
                            for name, value in reward_result.metrics.items()
                        },
                    ),
                    _detached_public_info(info),
                ))

            observation_steps.append(policy_observations.copy())
            action_steps.append(recorded_actions)
            log_probability_steps.append(recorded_log_probabilities)
            value_steps.append(np.asarray(decision.values, dtype=np.float32).copy())
            reward_steps.append(rewards)
            fire_allowed_steps.append(recorded_fire_allowed)
            reset_steps.append(reset_mask.copy())
            terminals = np.asarray(result.terminated, dtype=np.bool_)
            truncated = np.asarray(result.truncated, dtype=np.bool_)
            if terminals.shape != (client_count,):
                raise CollectorAdmissionError("batch terminal shape differs")
            if truncated.shape != (client_count,):
                raise CollectorAdmissionError("batch truncated shape differs")
            terminals = terminals | truncated
            pending_observer_events.extend(round_observer_events)
            for index, terminal in enumerate(terminals):
                if terminal:
                    client_id = str(sanitized_infos[index]["client_id"])
                    self._causal_admission.reset_client(client_id)
                    if self.reset_reward_stream is not None:
                        self.reset_reward_stream(client_id)
            terminal_steps.append(terminals)
            info_steps.append(tuple(sanitized_infos))
            current_raw = _validate_observations(result.observations, client_count)
            current_infos = tuple(dict(value) for value in sanitized_infos)
            state = decision.next_state
            reset_mask = terminals.copy()

        final_observations = self.observation_transform(
            current_raw.copy(), _policy_info_projection(current_infos),
            policy_version,
        )
        final_observations = _validate_observations(final_observations, client_count)
        bootstrap = np.asarray(
            self.policy.values(final_observations, state, reset_mask.copy()),
            dtype=np.float32,
        )
        if bootstrap.shape != (client_count,) or not np.isfinite(bootstrap).all():
            raise CollectorAdmissionError("policy bootstrap values are invalid")

        def stack(values: Sequence[np.ndarray], dtype: Any = None) -> np.ndarray:
            result = np.stack(values, axis=1)
            return result.astype(dtype, copy=False) if dtype is not None else result

        observations = stack(observation_steps, np.float32)
        actions = stack(action_steps, np.float32)
        log_probabilities = stack(log_probability_steps, np.float32)
        values = stack(value_steps, np.float32)
        rewards = stack(reward_steps, np.float32)
        fire_allowed = stack(fire_allowed_steps, np.bool_)
        recurrent_reset = stack(reset_steps, np.bool_)
        terminated = stack(terminal_steps, np.bool_)
        advantages = np.zeros_like(rewards)
        last_advantage = np.zeros(client_count, dtype=np.float32)
        next_value = bootstrap
        for time_index in range(rewards.shape[1] - 1, -1, -1):
            nonterminal = (~terminated[:, time_index]).astype(np.float32)
            delta = (
                rewards[:, time_index]
                + self.config.gamma * next_value * nonterminal
                - values[:, time_index]
            )
            last_advantage = delta + (
                self.config.gamma * self.config.gae_lambda
                * nonterminal * last_advantage
            )
            advantages[:, time_index] = last_advantage
            next_value = values[:, time_index]
        returns = advantages + values
        rollout = CollectedMultiresRollout(
            observations=observations,
            actions=actions,
            old_log_probabilities=log_probabilities,
            old_values=values,
            rewards=rewards,
            advantages=advantages,
            returns=returns,
            fire_allowed=fire_allowed,
            valid=np.ones_like(terminated, dtype=np.bool_),
            recurrent_reset=recurrent_reset,
            terminated=terminated,
            initial_state=initial_state,
            policy_version=policy_version,
            boundary_rounds=boundary_rounds,
            infos=tuple(info_steps),
        )
        rollout.validate()
        # Publish metrics only at the complete rollout commit point.  A bad
        # bootstrap, GAE result, or rollout shape must never produce observer
        # evidence for transitions that cannot be passed to PPO.
        if self.transition_observer is not None:
            for event in pending_observer_events:
                try:
                    self.transition_observer(*event)
                except Exception as error:
                    self._failed = True
                    raise CollectorAdmissionError(
                        "causal transition observer failed"
                    ) from error
        self._current_raw = current_raw.copy()
        self._current_infos = tuple(dict(value) for value in current_infos)
        return rollout
