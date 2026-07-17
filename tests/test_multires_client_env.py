import numpy as np
import pytest

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_batch import Q2NetworkClientBatch
from harness.client_env import ClientTelemetryDrain, Q2NetworkClientEnv
from harness.client_protocol import ClientTelemetry
from harness.multires_admission import (
    PRIVATE_CAUSAL_INFO_KEY,
    PRIVATE_SPATIAL_REWARD_INFO_KEY,
    SPATIAL_PROVIDER_SCHEMA,
    SPATIAL_REWARD_EVIDENCE_SCHEMA,
    PrivateSpatialRewardEvidence,
    SpatialProviderFrame,
    SpatialProviderBinding,
)
from harness.multires_contract import FEATURE_SCHEMA_SHA256, OBS_DIM
from harness.protocol import Action, ML_TERMINAL_INTERMISSION, SpatialPolicyFeatures


ATLAS = "a" * 64
RUNTIME = "b" * 64


class Observation:
    def __init__(self):
        self.action_debug = np.zeros(15, dtype=np.float32)
        self.action_debug[0] = 9
        self.action_debug[1] = 1
        self.action_debug[8] = 1
        self.self_state = np.zeros(10, dtype=np.float32)
        self.self_state[6] = 100.0
        self.is_terminal = False
        self.terminal_reason = 0
        self.actual_ducked = False
        self.standing_blocked = False
        self.water_vertical_mode = False
        self.reward_damage_dealt = 0.0

    def to_vector(self, spatial):
        assert isinstance(spatial, SpatialPolicyFeatures)
        result = np.zeros(OBS_DIM, dtype=np.float32)
        result[-100:] = spatial.to_vector()
        return result


def causal():
    return CausalTelemetry(
        tick=10, client_life_epoch=1, target_id=0, target_epoch=0,
        environmental_source_id=0, environmental_source_epoch=0,
        environmental_mod=0,
        environmental_damage=0, crouch_edge_id=0, crouch_edge_epoch=0,
        echo_tick=9, action_generation=2, hook_zone_id=0,
        hook_attempt_tick=0, hook_action_generation=0,
        flags=(CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
               | CausalFlags.TRANSITION_TRAINABLE
               | CausalFlags.ROLE_PLAYING
               | CausalFlags.ROLE_PUBLIC_PM_NORMAL),
    )


class Provider:
    def __init__(self, map_epoch=0):
        self.closed = False
        self.close_calls = 0
        self.map_epoch = map_epoch
        self.samples = 0

    def sample(self, telemetry, *, episode_projection):
        self.samples += 1
        evidence = None
        if not episode_projection:
            evidence = PrivateSpatialRewardEvidence(
                schema=SPATIAL_REWARD_EVIDENCE_SCHEMA,
                client_id=telemetry.client_id,
                client_slot=telemetry.client_slot,
                map_name=telemetry.map_name,
                map_epoch=self.map_epoch,
                server_frame=telemetry.server_frame,
                client_epoch=telemetry.causal.client_life_epoch,
                l1_index=(2, 3, 4),
                cost_to_safety_q8=0xFFFFFFFF,
                signed_safe_clearance_q8=256,
                hazard_types=0,
                hazard_severity=0,
                atlas_region_id=91,
                confidence=0xFFFF,
                hazard_component_id=0,
                hazard_component_epoch=0,
            )
        return SpatialProviderFrame(
            schema=SPATIAL_PROVIDER_SCHEMA,
            feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            atlas_sha256=ATLAS,
            runtime_manifest_sha256=RUNTIME,
            client_id=telemetry.client_id,
            client_slot=telemetry.client_slot,
            map_name=telemetry.map_name,
            map_epoch=self.map_epoch,
            server_frame=telemetry.server_frame,
            spatial=SpatialPolicyFeatures(
                dyn=np.zeros(24, dtype=np.float32),
                recovery=np.zeros(16, dtype=np.float32),
                objectives=np.zeros((4, 15), dtype=np.float32),
            ),
            private_reward_evidence=evidence,
        )

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_multires_selection_never_constructs_or_rewards_through_legacy(monkeypatch):
    from harness import spatial

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy VoxelSpatialReward was reached")

    monkeypatch.setattr(spatial.VoxelSpatialReward, "from_env", forbidden)
    provider = Provider()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="token",
        client_binary="/tmp/yquake2",
        client_root="/tmp/q2",
        multires_spatial_provider=provider,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
    )
    sample = ClientTelemetry(
        sequence=1,
        client_slot=1,
        server_frame=10,
        client_id=env.client_id,
        map_name="q2dm1",
        observation=Observation(),
        causal=causal(),
    )
    initial, _info = env.initial_result(sample, vector=True)
    transition, reward, _terminated, _truncated, info = env.transition_result(
        sample, vector=True
    )
    assert initial.shape == transition.shape == (OBS_DIM,)
    assert reward == 0.0
    assert "reward_base" not in info and "spatial_bonus" not in info
    assert PRIVATE_CAUSAL_INFO_KEY in info
    assert PRIVATE_SPATIAL_REWARD_INFO_KEY in info
    env.close()
    assert provider.closed
    assert provider.close_calls == 1
    assert env._multires_spatial_provider is None
    env.close()
    assert provider.close_calls == 1


def test_provider_factory_rebuilds_atomically_on_map_rotation():
    class Factory:
        def __init__(self):
            self.providers = []

        def create(self, telemetry, *, map_epoch):
            provider = Provider(map_epoch)
            self.providers.append((telemetry.map_name, map_epoch, provider))
            return SpatialProviderBinding(provider=provider, atlas_sha256=ATLAS)

    factory = Factory()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="token",
        client_binary="/tmp/yquake2",
        client_root="/tmp/q2",
        multires_spatial_provider_factory=factory,
        expected_runtime_manifest_sha256=RUNTIME,
    )
    first = ClientTelemetry(
        sequence=1, client_slot=1, server_frame=10, client_id=env.client_id,
        map_name="q2dm1", observation=Observation(), causal=causal(),
    )
    env.initial_result(first, vector=True)
    second = ClientTelemetry(
        sequence=2, client_slot=1, server_frame=10, client_id=env.client_id,
        map_name="q2dm2", observation=Observation(), causal=causal(),
    )
    vector, reward, *_ = env.transition_result(second, vector=True)
    assert vector.shape == (OBS_DIM,) and reward == 0.0
    assert [(name, epoch) for name, epoch, _provider in factory.providers] == [
        ("q2dm1", 0), ("q2dm2", 1),
    ]
    assert factory.providers[0][2].closed
    env.close()
    assert factory.providers[1][2].closed


def test_episode_reset_reuses_cached_vector_without_mutating_provider_again():
    provider = Provider()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="token",
        client_binary="/tmp/yquake2",
        client_root="/tmp/q2",
        multires_spatial_provider=provider,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
    )
    sample = ClientTelemetry(
        sequence=1, client_slot=1, server_frame=10, client_id=env.client_id,
        map_name="q2dm1", observation=Observation(), causal=causal(),
    )
    env._last = sample
    initial, _ = env.initial_result(sample, vector=True)
    reset = env.reset_episode_vector()
    assert np.array_equal(reset, initial)
    assert provider.samples == 1
    env.close()


def test_stalled_boundary_polls_reuse_projection_until_new_telemetry_once():
    provider = Provider()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="token",
        client_binary="/tmp/yquake2",
        client_root="/tmp/q2",
        multires_spatial_provider=provider,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
    )
    boundary = ClientTelemetry(
        sequence=7, client_slot=1, server_frame=41, client_id=env.client_id,
        map_name="q2dm1", observation=Observation(), causal=causal(),
    )

    admitted, reward, _terminated, _truncated, admitted_info = (
        env.transition_result(boundary, vector=True)
    )
    assert reward == 0.0
    assert provider.samples == 1
    assert PRIVATE_CAUSAL_INFO_KEY in admitted_info
    assert PRIVATE_SPATIAL_REWARD_INFO_KEY in admitted_info

    # Two consecutive stalled barrier polls over the exact same datagram must
    # neither re-enter the source/provider nor replay private reward evidence.
    first_poll, first_info = env.initial_result(boundary, vector=True)
    first_poll[:] = -99.0
    first_info["map"] = "mutated-by-caller"
    second_poll, second_info = env.initial_result(boundary, vector=True)
    assert provider.samples == 1
    assert np.array_equal(second_poll, admitted)
    assert second_info["map"] == "q2dm1"
    assert PRIVATE_CAUSAL_INFO_KEY not in first_info
    assert PRIVATE_SPATIAL_REWARD_INFO_KEY not in first_info
    assert PRIVATE_CAUSAL_INFO_KEY not in second_info
    assert PRIVATE_SPATIAL_REWARD_INFO_KEY not in second_info

    advanced = ClientTelemetry(
        sequence=8, client_slot=1, server_frame=42, client_id=env.client_id,
        map_name="q2dm1", observation=Observation(), causal=causal(),
    )
    advanced_vector, _ = env.initial_result(advanced, vector=True)
    replayed_vector, _ = env.initial_result(advanced, vector=True)
    assert provider.samples == 2
    assert np.array_equal(replayed_vector, advanced_vector)
    env.close()


def test_batch_map_barrier_polls_do_not_replay_unchanged_provider_frame():
    provider = Provider()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="token",
        client_binary="/tmp/yquake2",
        client_root="/tmp/q2",
        multires_spatial_provider=provider,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
    )

    def sample(sequence, frame, *, intermission=False):
        observation = Observation()
        observation.is_terminal = intermission
        observation.terminal_reason = (
            ML_TERMINAL_INTERMISSION if intermission else 0
        )
        return ClientTelemetry(
            sequence=sequence,
            client_slot=1,
            server_frame=frame,
            client_id=env.client_id,
            map_name="q2dm1",
            observation=observation,
            causal=causal(),
        )

    initial = sample(1, 10)
    boundary = sample(2, 11, intermission=True)
    preflight = [boundary]
    progress = []

    def start():
        env._last = initial
        return initial

    def drain_latest_telemetry():
        previous = env._last
        if not preflight:
            return ClientTelemetryDrain(previous, previous, 0, ())
        latest = preflight.pop(0)
        env._last = latest
        return ClientTelemetryDrain(previous, latest, 1, (latest.map_name,))

    def receive_telemetry(*, after_sequence, timeout=None):
        if not progress:
            raise TimeoutError("stalled map download")
        latest = progress.pop(0)
        assert latest.sequence > after_sequence
        env._last = latest
        return latest

    def forbidden_dispatch(_action):
        raise AssertionError("map barrier dispatched an action")

    env.start = start
    env.drain_latest_telemetry = drain_latest_telemetry
    env.receive_telemetry = receive_telemetry
    env.dispatch_action = forbidden_dispatch

    batch = Q2NetworkClientBatch([env], round_timeout=0.001)
    try:
        batch.reset()
        assert provider.samples == 1
        first = batch.collect_round([Action()], policy_version=1)
        assert first.infos[0]["map_epoch_pending"] is True
        assert provider.samples == 2

        batch.collect_round([Action()], policy_version=2)
        batch.collect_round([Action()], policy_version=3)
        assert provider.samples == 2

        progress.append(sample(3, 12, intermission=True))
        advanced = batch.collect_round([Action()], policy_version=4)
        assert advanced.infos[0]["preflight_to_frame"] == 12
        assert provider.samples == 3
    finally:
        batch.close()


def test_failed_rotation_is_retryable_and_close_is_terminal_and_idempotent():
    class Factory:
        def __init__(self):
            self.providers = []
            self.fail_once = True
            self.resets = 0

        def create(self, telemetry, *, map_epoch):
            if telemetry.map_name == "q2dm2" and self.fail_once:
                self.fail_once = False
                raise RuntimeError("injected rotation failure")
            provider = Provider(map_epoch)
            self.providers.append(provider)
            return SpatialProviderBinding(provider=provider, atlas_sha256=ATLAS)

        def reset_session(self):
            self.resets += 1

    factory = Factory()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="token",
        client_binary="/tmp/yquake2",
        client_root="/tmp/q2",
        multires_spatial_provider_factory=factory,
        expected_runtime_manifest_sha256=RUNTIME,
    )
    first = ClientTelemetry(
        sequence=1, client_slot=1, server_frame=10, client_id=env.client_id,
        map_name="q2dm1", observation=Observation(), causal=causal(),
    )
    env.initial_result(first, vector=True)
    original = factory.providers[-1]
    second = ClientTelemetry(
        sequence=2, client_slot=1, server_frame=11, client_id=env.client_id,
        map_name="q2dm2", observation=Observation(), causal=causal(),
    )
    with pytest.raises(RuntimeError, match="injected"):
        env.transition_result(second, vector=True)
    assert not original.closed
    assert env._multires_map_name == "q2dm1"
    assert env._multires_map_epoch == 0
    env.transition_result(second, vector=True)
    assert original.closed and env._multires_map_epoch == 1

    env.close()
    assert factory.resets == 1
    assert env._multires_spatial_provider is None
    env.close()
    assert factory.resets == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.initial_result(first, vector=True)
    with pytest.raises(RuntimeError, match="fresh environment"):
        env.start()
