import numpy as np
import pytest
from types import MappingProxyType

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_batch import BatchActionTag, BatchRound
from harness.multires_admission import (
    PRIVATE_CAUSAL_INFO_KEY,
    PRIVATE_SPATIAL_REWARD_INFO_KEY,
    SPATIAL_ATTESTATION_INFO_KEY,
    SPATIAL_REWARD_EVIDENCE_SCHEMA,
    PrivateSpatialRewardEvidence,
)
from harness.multires_collector import (
    CollectorAdmissionError,
    CollectorConfig,
    MultiresSynchronousCollector,
    PolicyObservationBatch,
    PolicyDecision,
)
from harness.multires_contract import OBS_DIM
from harness.multires_reward import (
    CausalRewardFrame,
    CausalRewardReducer,
    CausalRewardResult,
)


class FakePolicy:
    def initial_state(self, client_count):
        return np.zeros(client_count, dtype=np.float32)

    def act(self, observations, state, recurrent_reset):
        state = np.where(recurrent_reset, 0.0, state)
        count = len(observations)
        actions = np.zeros((count, 8), dtype=np.float32)
        actions[:, 4] = 1.0
        return PolicyDecision(
            actions=actions,
            values=state.copy(),
            log_probabilities=np.full(count, -0.5, dtype=np.float32),
            next_state=state + 1.0,
            fire_allowed=np.zeros(count, dtype=np.bool_),
            raw_fire_log_probability=np.full(count, -1.0, dtype=np.float32),
        )

    def values(self, observations, state, recurrent_reset):
        return np.where(recurrent_reset, 0.0, state).astype(np.float32)


class FailingBootstrapPolicy(FakePolicy):
    def values(self, observations, state, recurrent_reset):
        raise RuntimeError("bootstrap failed")


class FakeBatch:
    vector = True

    def __init__(self, *, boundary_first=True):
        self.clients = ("c0", "c1")
        self.frame = 10
        self.round = 0
        self.boundary_first = boundary_first
        self.reset_called = False

    @staticmethod
    def _observations(frame):
        values = np.zeros((2, OBS_DIM), dtype=np.float32)
        values[:, 0] = frame / 100.0
        return values

    @staticmethod
    def _attestation():
        return {
            "schema": "q2-multires-spatial-provider-v1",
            "feature_schema_sha256": "1" * 64,
            "atlas_sha256": "2" * 64,
            "runtime_manifest_sha256": "3" * 64,
            "map_epoch": 0,
        }

    def reset(self):
        assert not self.reset_called
        self.reset_called = True
        infos = [
            {
                "client_id": client,
                "client_slot": index + 1,
                "map": "q2dm1",
                "server_frame": self.frame,
                SPATIAL_ATTESTATION_INFO_KEY: self._attestation(),
            }
            for index, client in enumerate(self.clients)
        ]
        return self._observations(self.frame), infos

    def collect_round(self, actions, *, policy_version):
        round_id = self.round
        self.round += 1
        if self.boundary_first and round_id == 0:
            infos = tuple({
                "client_id": client,
                "client_slot": index + 1,
                "map": "q2dm1",
                "server_frame": self.frame,
                "trainable_transition": False,
                SPATIAL_ATTESTATION_INFO_KEY: self._attestation(),
            } for index, client in enumerate(self.clients))
            return BatchRound(
                round_id=round_id,
                policy_version=policy_version,
                observations=self._observations(self.frame),
                rewards=np.zeros(2, dtype=np.float32),
                terminated=np.ones(2, dtype=np.bool_),
                truncated=np.zeros(2, dtype=np.bool_),
                infos=infos,
                tags=tuple(BatchActionTag(
                    round_id, policy_version, index, client, index + 1,
                    self.frame - 1,
                ) for index, client in enumerate(self.clients)),
            )

        self.frame += 1
        echo_tick = self.frame - 1
        generation = 2
        infos = []
        tags = []
        for index, client in enumerate(self.clients):
            causal = CausalTelemetry(
                tick=self.frame,
                client_life_epoch=1,
                target_id=0,
                target_epoch=0,
                environmental_source_id=0,
                environmental_source_epoch=0,
                environmental_mod=0,
                environmental_damage=0,
                crouch_edge_id=0,
                crouch_edge_epoch=0,
                echo_tick=echo_tick,
                action_generation=generation,
                hook_zone_id=0,
                hook_attempt_tick=0,
                hook_action_generation=0,
                flags=(
                    CausalFlags.ECHO_VALID
                    | CausalFlags.FACTS_COMPLETE
                    | CausalFlags.TRANSITION_TRAINABLE
                ),
            )
            spatial_reward = PrivateSpatialRewardEvidence(
                schema=SPATIAL_REWARD_EVIDENCE_SCHEMA,
                client_id=client,
                client_slot=index + 1,
                map_name="q2dm1",
                map_epoch=0,
                server_frame=self.frame,
                client_epoch=1,
                l1_index=(3, 4, 5),
                cost_to_safety_q8=0xFFFFFFFF,
                signed_safe_clearance_q8=256,
                hazard_types=0,
                hazard_severity=0,
                atlas_region_id=91,
                confidence=0xFFFF,
                hazard_component_id=0,
                hazard_component_epoch=0,
            )
            infos.append({
                "client_id": client,
                "client_slot": index + 1,
                "map": "q2dm1",
                "server_frame": self.frame,
                "batch_round_id": round_id,
                "policy_version": policy_version,
                "action_tick": echo_tick,
                "authoritative_echo_tick": echo_tick,
                "authoritative_action_generation": generation,
                "authoritative_map_epoch": 0,
                "authoritative_echo_valid": True,
                "trainable_transition": True,
                "causal_echo_valid": True,
                "causal_facts_complete": True,
                "causal_transition_trainable": True,
                "fire_gate_target": False,
                "fire_gate_protected": False,
                "fire_gate_suppressed": False,
                "requested_action_fire": False,
                "effective_action_fire": False,
                "action_debug_fire": False,
                "action_debug_hook": 0,
                "damage_dealt": 0.0,
                "action_debug_vertical_intent": 1,
                "action_debug_actual_ducked": False,
                "standing_blocked": False,
                "action_debug_water_vertical_mode": False,
                SPATIAL_ATTESTATION_INFO_KEY: self._attestation(),
                PRIVATE_CAUSAL_INFO_KEY: causal,
                PRIVATE_SPATIAL_REWARD_INFO_KEY: spatial_reward,
            })
            tags.append(BatchActionTag(
                round_id, policy_version, index, client, index + 1, echo_tick
            ))
        return BatchRound(
            round_id=round_id,
            policy_version=policy_version,
            observations=self._observations(self.frame),
            rewards=np.zeros(2, dtype=np.float32),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
            infos=tuple(infos),
            tags=tuple(tags),
        )


def collect_once():
    reducers = {}

    def reward(client_id, frame):
        reducers.setdefault(client_id, CausalRewardReducer())
        return reducers[client_id].step(frame)

    collector = MultiresSynchronousCollector(
        FakeBatch(),
        FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        config=CollectorConfig(transitions_per_client=4),
    )
    return collector.collect(policy_version=7)


def test_in_process_collector_is_deterministic_and_excludes_boundaries():
    first = collect_once()
    second = collect_once()
    assert first.boundary_rounds == 1
    assert first.observations.shape == (2, 4, OBS_DIM)
    assert first.valid.all()
    assert first.recurrent_reset[:, 0].all()
    assert first.deterministic_sha256() == second.deterministic_sha256()
    assert all(
        PRIVATE_CAUSAL_INFO_KEY not in info
        and PRIVATE_SPATIAL_REWARD_INFO_KEY not in info
        for step in first.infos for info in step
    )


def test_repeated_whole_batch_boundaries_fail_closed_at_livelock_cap():
    class AlwaysBoundaryBatch(FakeBatch):
        def collect_round(self, actions, *, policy_version):
            self.round = 0
            return super().collect_round(actions, policy_version=policy_version)

    collector = MultiresSynchronousCollector(
        AlwaysBoundaryBatch(), FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=lambda _client_id, _frame: CausalRewardResult(
            reward=0.0, metrics={}
        ),
        config=CollectorConfig(
            transitions_per_client=1, maximum_boundary_rounds=2
        ),
    )
    with pytest.raises(
        CollectorAdmissionError, match="too many non-trainable boundary rounds"
    ):
        collector.collect(policy_version=7)


def test_collector_continues_without_resetting_live_clients():
    batch = FakeBatch(boundary_first=False)
    reducers = {}

    def reward(client_id, frame):
        reducers.setdefault(client_id, CausalRewardReducer())
        return reducers[client_id].step(frame)

    collector = MultiresSynchronousCollector(
        batch, FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        config=CollectorConfig(transitions_per_client=2),
    )
    collector.collect(policy_version=1)
    collector.collect(policy_version=2)
    assert batch.reset_called
    assert batch.round == 4


def test_start_and_nontrainable_boundary_reset_all_causal_streams():
    resets = []
    reducers = {}

    def reward(client_id, frame):
        reducers.setdefault(client_id, CausalRewardReducer())
        return reducers[client_id].step(frame)

    def reset(client_id):
        resets.append(client_id)
        reducers.pop(client_id, None)

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=True), FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        reset_reward_stream=reset,
        config=CollectorConfig(transitions_per_client=1),
    )
    collector.collect(policy_version=1)
    # One reset at client start and one at the whole-batch resync boundary.
    assert resets.count("c0") == 2
    assert resets.count("c1") == 2


def test_transition_observer_receives_only_detached_post_admission_facts():
    events = []

    def reward(_client_id, frame):
        return CausalRewardResult(
            reward=float(frame.tick), metrics={"causal_tick": float(frame.tick)}
        )

    def observe(client_id, frame, result, info):
        assert isinstance(frame, CausalRewardFrame)
        assert isinstance(result, CausalRewardResult)
        assert isinstance(info, MappingProxyType)
        assert PRIVATE_CAUSAL_INFO_KEY not in info
        assert PRIVATE_SPATIAL_REWARD_INFO_KEY not in info
        with pytest.raises(TypeError):
            info["observer_write"] = True
        # The result and nested public values are detached observer copies.
        result.metrics["observer_write"] = 1.0
        info[SPATIAL_ATTESTATION_INFO_KEY]["map_epoch"] = 99
        events.append((client_id, frame, result, info))

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=False),
        FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        transition_observer=observe,
        config=CollectorConfig(transitions_per_client=2),
    )
    rollout = collector.collect(policy_version=11)

    assert [event[0] for event in events] == ["c0", "c1", "c0", "c1"]
    assert all(event[2].metrics["observer_write"] == 1.0 for event in events)
    assert all(
        info[SPATIAL_ATTESTATION_INFO_KEY]["map_epoch"] == 0
        for step in rollout.infos
        for info in step
    )
    assert all(
        PRIVATE_CAUSAL_INFO_KEY not in info
        and PRIVATE_SPATIAL_REWARD_INFO_KEY not in info
        for step in rollout.infos
        for info in step
    )


def test_policy_transform_guide_audit_is_identity_fenced_and_published():
    events = []

    def transform(observations, infos, _version):
        return PolicyObservationBatch(
            observations=observations,
            transition_info=tuple({
                "client_id": info["client_id"],
                "server_frame": info["server_frame"],
                "guide_dropped": (True, False, False, False),
                "guide_classes": (0, 1, None, None),
                "global_guide_drop": False,
            } for info in infos),
        )

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=False), FakePolicy(),
        observation_transform=transform,
        reward_function=lambda _client, _frame: CausalRewardResult(
            reward=0.0, metrics={}
        ),
        transition_observer=lambda *_args: events.append(_args[-1]),
        config=CollectorConfig(transitions_per_client=1),
    )
    rollout = collector.collect(policy_version=4)
    assert len(events) == 2
    assert all(info["guide_dropped"] == (True, False, False, False)
               for info in events)
    assert all(info["guide_classes"] == (0, 1, None, None) for info in events)
    assert all(step["global_guide_drop"] is False for step in rollout.infos[0])


def test_policy_transform_rejects_teacher_or_unbound_telemetry_fields():
    def transform(observations, infos, _version):
        return PolicyObservationBatch(
            observations=observations,
            transition_info=tuple({
                "client_id": info["client_id"],
                "server_frame": info["server_frame"],
                "guide_dropped": (False,) * 4,
                "guide_classes": (None,) * 4,
                "global_guide_drop": False,
                "teacher_action": 7,
            } for info in infos),
        )

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=False), FakePolicy(),
        observation_transform=transform,
        reward_function=lambda _client, _frame: CausalRewardResult(
            reward=0.0, metrics={}
        ),
        config=CollectorConfig(transitions_per_client=1),
    )
    with pytest.raises(CollectorAdmissionError, match="fields differ"):
        collector.collect(policy_version=4)


def test_transition_observer_is_withheld_until_entire_round_admits():
    events = []

    def reward(client_id, _frame):
        if client_id == "c1":
            raise RuntimeError("reject second client")
        return CausalRewardResult(reward=0.0, metrics={})

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=False),
        FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        transition_observer=lambda *event: events.append(event),
        config=CollectorConfig(transitions_per_client=1),
    )
    with pytest.raises(RuntimeError, match="reject second client"):
        collector.collect(policy_version=1)
    assert events == []


def test_transition_observer_is_withheld_until_complete_rollout_validates():
    events = []

    def reward(_client_id, _frame):
        return CausalRewardResult(reward=0.0, metrics={})

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=False),
        FailingBootstrapPolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        transition_observer=lambda *event: events.append(event),
        config=CollectorConfig(transitions_per_client=1),
    )
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        collector.collect(policy_version=1)
    assert events == []


def test_transition_observer_failure_is_fatal_to_collection():
    def reward(_client_id, _frame):
        return CausalRewardResult(reward=0.0, metrics={})

    def reject(*_event):
        raise RuntimeError("observer sink unavailable")

    collector = MultiresSynchronousCollector(
        FakeBatch(boundary_first=False),
        FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=reward,
        transition_observer=reject,
        config=CollectorConfig(transitions_per_client=1),
    )
    with pytest.raises(CollectorAdmissionError, match="observer failed"):
        collector.collect(policy_version=3)
    with pytest.raises(CollectorAdmissionError, match="collector is failed"):
        collector.collect(policy_version=3)


def test_policy_transform_gets_only_frozen_whitelisted_info_across_boundaries():
    class PrivacyBatch(FakeBatch):
        def __init__(self):
            super().__init__(boundary_first=False)
            self.boundary_phases = (
                "corpse",
                "new_life_teleport_settling",
                "new_life_actionable_prime",
            )

        @staticmethod
        def _causal(frame):
            return CausalTelemetry(
                tick=frame, client_life_epoch=1, target_id=0, target_epoch=0,
                environmental_source_id=0, environmental_source_epoch=0,
                environmental_mod=0, environmental_damage=0,
                crouch_edge_id=0, crouch_edge_epoch=0, echo_tick=frame - 1,
                action_generation=2, hook_zone_id=0, hook_attempt_tick=0,
                hook_action_generation=0,
                flags=CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE,
            )

        def reset(self):
            observations, infos = super().reset()
            for info in infos:
                info[PRIVATE_CAUSAL_INFO_KEY] = self._causal(self.frame)
                info["causal_reset_sentinel"] = "must-not-reach-policy"
            return observations, infos

        def collect_round(self, actions, *, policy_version):
            if self.round >= len(self.boundary_phases):
                return super().collect_round(
                    actions, policy_version=policy_version
                )
            round_id = self.round
            phase = self.boundary_phases[round_id]
            self.round += 1
            self.frame += 1
            infos = tuple({
                "client_id": client,
                "client_slot": index + 1,
                "map": "q2dm1",
                "server_frame": self.frame,
                "trainable_transition": False,
                "death_lifecycle_phase": phase,
                "causal_echo_valid": True,
                "causal_facts_complete": True,
                "causal_transition_trainable": phase.endswith("prime"),
                SPATIAL_ATTESTATION_INFO_KEY: self._attestation(),
                PRIVATE_CAUSAL_INFO_KEY: self._causal(self.frame),
            } for index, client in enumerate(self.clients))
            return BatchRound(
                round_id=round_id,
                policy_version=policy_version,
                observations=self._observations(self.frame),
                rewards=np.zeros(2, dtype=np.float32),
                terminated=np.ones(2, dtype=np.bool_),
                truncated=np.zeros(2, dtype=np.bool_),
                infos=infos,
                tags=tuple(BatchActionTag(
                    round_id, policy_version, index, client, index + 1,
                    self.frame - 1,
                ) for index, client in enumerate(self.clients)),
            )

    calls = []
    outer_keys = {
        "client_id", "client_slot", "map", "server_frame",
        SPATIAL_ATTESTATION_INFO_KEY,
    }
    attestation_keys = {
        "schema", "feature_schema_sha256", "atlas_sha256",
        "runtime_manifest_sha256", "map_epoch",
    }

    def spy(observations, infos, _version):
        calls.append(tuple(int(info["server_frame"]) for info in infos))
        for info in infos:
            assert isinstance(info, MappingProxyType)
            assert set(info) == outer_keys
            assert not any(
                "causal" in key or "lifecycle" in key or "action_state" in key
                for key in info
            )
            assert not any(
                isinstance(value, (CausalTelemetry, PrivateSpatialRewardEvidence))
                for value in info.values()
            )
            attestation = info[SPATIAL_ATTESTATION_INFO_KEY]
            assert isinstance(attestation, MappingProxyType)
            assert set(attestation) == attestation_keys
            with pytest.raises(TypeError):
                info["write"] = True
            with pytest.raises(TypeError):
                attestation["map_epoch"] = 99
        return observations

    collector = MultiresSynchronousCollector(
        PrivacyBatch(), FakePolicy(), observation_transform=spy,
        reward_function=lambda _client_id, _frame: CausalRewardResult(
            reward=0.0, metrics={}
        ),
        config=CollectorConfig(
            transitions_per_client=1, maximum_boundary_rounds=3
        ),
    )
    rollout = collector.collect(policy_version=17)
    assert rollout.boundary_rounds == 3
    # Reset, corpse, settle, prime, admitted action, and bootstrap.
    assert len(calls) == 5


def test_policy_projection_rejects_extra_nested_attestation_key():
    class ExtraAttestationBatch(FakeBatch):
        @staticmethod
        def _attestation():
            value = FakeBatch._attestation()
            value["private_future_sentinel"] = "must-fail-closed"
            return value

    collector = MultiresSynchronousCollector(
        ExtraAttestationBatch(boundary_first=False), FakePolicy(),
        observation_transform=lambda observations, _infos, _version: observations,
        reward_function=lambda _client_id, _frame: CausalRewardResult(
            reward=0.0, metrics={}
        ),
        config=CollectorConfig(transitions_per_client=1),
    )
    with pytest.raises(
        CollectorAdmissionError, match="spatial attestation schema differs"
    ):
        collector.collect(policy_version=1)
