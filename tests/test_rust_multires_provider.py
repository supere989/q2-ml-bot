from types import SimpleNamespace

import numpy as np
import pytest

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_protocol import ClientTelemetry
from harness.rust_multires_provider import (
    RUST_RECOVERY_EVIDENCE_SCHEMA,
    RustAtlasSpatialProvider,
    RustSpatialProviderError,
    SpatialQueryInputs,
)


ATLAS = "a" * 64
RUNTIME = "b" * 64
MAP_SHA = "c" * 64


class DynRuntime:
    atlas_sha256 = ATLAS
    map_epoch = 4
    client_id = 17
    map_sha256 = MAP_SHA
    origin = (0, 0, 0)
    cell_count = 5
    resident_bytes = 1024
    snapshot_size = len(b"dyn-snapshot")

    def __init__(self):
        self.environment_steps = 99
        self.last_event_id = 0
        self.snapshot_sha256 = "d" * 64
        self.ingested = False

    def ingest_events(
        self, atlas_sha256, map_sha256, origin, map_epoch, client_id,
        client_life_epoch, server_frame, expected_environment_steps,
        environment_steps, events,
    ):
        assert (
            atlas_sha256, map_sha256, tuple(origin), map_epoch, client_id,
            client_life_epoch, server_frame, expected_environment_steps,
            environment_steps, events,
        ) == (ATLAS, MAP_SHA, (0, 0, 0), 4, 17, 3, 10, 99, 100, [])
        self.environment_steps = environment_steps
        self.ingested = True
        return {
            "schema": "q2-dyn-named-event-v1",
            "events_applied": 0,
            "cells_updated": 0,
            "decay_intervals": 0,
            "environment_steps": 100,
            "client_life_epoch": 3,
            "server_frame": 10,
            "last_event_id": 0,
            "snapshot_sha256": self.snapshot_sha256,
        }

    def feature_block(
        self, position, yaw, map_epoch, client_id, client_life_epoch,
        environment_steps, server_frame, survivability, thermal=None,
        search_radius=2048.0, score_scale=8.0,
    ):
        assert self.ingested
        assert (map_epoch, client_id, client_life_epoch) == (4, 17, 3)
        assert (environment_steps, server_frame) == (100, 10)
        return np.arange(24, dtype=np.float32)

    def commit_frame(
        self, atlas_sha256, map_sha256, origin, map_epoch, client_id,
        client_life_epoch, server_frame, expected_environment_steps,
        environment_steps, events, position, yaw, survivability, thermal=None,
        search_radius=2048.0, score_scale=8.0,
    ):
        report = self.ingest_events(
            atlas_sha256, map_sha256, origin, map_epoch, client_id,
            client_life_epoch, server_frame, expected_environment_steps,
            environment_steps, events,
        )
        block = self.feature_block(
            position, yaw, map_epoch, client_id, client_life_epoch,
            environment_steps, server_frame, survivability, thermal,
            search_radius, score_scale,
        )
        return report, block, 0.0

    def snapshot_bytes(self):
        return b"dyn-snapshot"


class AtlasRuntime:
    atlas_sha256 = ATLAS
    map_epoch = 4
    map_id = "q2dm1"
    resident_bytes = 2048

    def __init__(self, *, omit_component=False):
        self.omit_component = omit_component

    def advisory_spatial_features_with_evidence(self, *args):
        evidence = {
            "schema": RUST_RECOVERY_EVIDENCE_SCHEMA,
            "atlas_sha256": ATLAS,
            "map_epoch": 4,
            "client_id": 17,
            "client_epoch": 3,
            "server_frame": 10,
            "l1_index": [1, 2, 3],
            "cost_to_safety_q8": 512,
            "signed_safe_clearance_q8": -256,
            "hazard_types": 1,
            "hazard_severity": 2,
            "atlas_region_id": 91,
            "confidence": 65535,
            "hazard_component_id": 44,
            "hazard_component_epoch": 4,
        }
        if self.omit_component:
            evidence.pop("hazard_component_id")
        assert args[6] == [(10, 0.75)]
        return np.arange(76, dtype=np.float32), evidence, {
            "atlas_lookup_us": 1.0,
            "recovery_query_us": 2.0,
            "guide_query_us": 3.0,
        }


class Inputs:
    class Stage:
        def __init__(self, inputs):
            self.inputs = inputs
            self._active = True

        def commit(self):
            self._active = False

        def rollback(self):
            self._active = False

    def sample(self, telemetry, *, map_epoch):
        return SpatialQueryInputs(
            client_id=telemetry.client_id,
            client_epoch=telemetry.causal.client_life_epoch,
            map_name=telemetry.map_name,
            map_epoch=map_epoch,
            server_frame=telemetry.server_frame,
            expected_environment_steps=99,
            environment_steps=100,
            survivability=(0.0, 0.5, 0.25),
            objective_beliefs=((10, 0.75),),
        )

    def stage(self, telemetry, *, map_epoch, emit_dyn_events):
        return self.Stage(self.sample(telemetry, map_epoch=map_epoch))

    def public_thermal_metrics(self):
        return 1, 0


def telemetry():
    causal = CausalTelemetry(
        tick=10,
        client_life_epoch=3,
        target_id=0,
        target_epoch=0,
        environmental_source_id=0,
        environmental_source_epoch=0,
        environmental_mod=0,
        environmental_damage=0,
        crouch_edge_id=0,
        crouch_edge_epoch=0,
        echo_tick=9,
        action_generation=2,
        hook_zone_id=0,
        hook_attempt_tick=0,
        hook_action_generation=0,
        flags=(CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
               | CausalFlags.TRANSITION_TRAINABLE),
    )
    observation = SimpleNamespace(
        self_state=np.array([1, 2, 3, 0, 0, 0, 100, 0, 1, 10], dtype=np.float32),
        yaw=20.0,
    )
    return ClientTelemetry(
        sequence=1, client_slot=2, server_frame=10, client_id="client-17",
        map_name="q2dm1", observation=observation, causal=causal,
    )


def provider(atlas=None):
    return RustAtlasSpatialProvider(
        atlas_runtime=atlas or AtlasRuntime(),
        dyn_runtime=DynRuntime(),
        input_source=Inputs(),
        expected_atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        map_name="q2dm1",
        map_epoch=4,
        rust_client_id=17,
        map_sha256=MAP_SHA,
        atlas_origin=(0, 0, 0),
        atlas_runtime_metadata={
            "atlas_build_peak_rss_bytes": 4096,
            "atlas_cell_count": 12,
            "atlas_chunk_count": 2,
            "atlas_deserialize_ms": 0.25,
        },
    )


def test_real_rust_adapter_composes_exact_24_16_60_and_private_evidence():
    result = provider().sample(telemetry(), episode_projection=False)
    assert result.spatial.dyn.shape == (24,)
    assert result.spatial.recovery.shape == (16,)
    assert result.spatial.objectives.shape == (4, 15)
    assert result.private_reward_evidence.hazard_component_id == 44
    assert result.private_reward_evidence.hazard_component_epoch == 4
    assert result.private_reward_evidence.atlas_region_id == 91
    snapshots = provider().drain_runtime_snapshots()
    assert snapshots == ()


def test_real_runtime_snapshot_uses_accepted_public_provider_path():
    runtime = provider()
    runtime.sample(telemetry(), episode_projection=False)
    snapshot, = runtime.drain_runtime_snapshots()
    assert snapshot["client_id"] == "client-17"
    assert snapshot["map_name"] == "q2dm1"
    assert snapshot["map_epoch"] == 4
    assert snapshot["server_frame"] == 10
    assert snapshot["atlas_loaded"] is True
    assert snapshot["atlas_hash_match"] is True
    assert snapshot["dyn_cell_count"] == 5
    assert snapshot["live_thermal_tracks"] == 1
    assert snapshot["thermal_checkpoint_fields"] == 0
    assert set(snapshot["query_timings_us"]) == {
        "dyn_query_us", "atlas_lookup_us", "recovery_query_us", "guide_query_us"
    }


def test_real_rust_adapter_fails_closed_until_component_authority_is_present():
    with pytest.raises(RustSpatialProviderError, match="missing=.*hazard_component_id"):
        provider(AtlasRuntime(omit_component=True)).sample(
            telemetry(), episode_projection=False
        )


def test_real_rust_adapter_rejects_map_rotation_without_new_attested_runtime():
    sample = telemetry()
    sample = ClientTelemetry(
        sequence=sample.sequence, client_slot=sample.client_slot,
        server_frame=sample.server_frame, client_id=sample.client_id,
        map_name="q2dm2", observation=sample.observation, causal=sample.causal,
    )
    with pytest.raises(RustSpatialProviderError, match="map rotation"):
        provider().sample(sample, episode_projection=False)
