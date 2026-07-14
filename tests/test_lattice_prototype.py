import json
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from harness.spatial import (
    SessionMemoryCell,
    VoxelSpatialReward,
    load_lattice_state,
    save_lattice_state,
)
from tools.map_bundle import build_manifest, encode_manifest, installed_manifest_name
if torch is not None:
    from train.ppo import _lattice_direction_loss


def _obs(*, pos=(0.0, 0.0, 0.0), tick=0, pickup=0.0):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[:3] = pos
    self_state[6] = 100.0
    self_state[7] = 50.0
    self_state[9] = 10.0
    return SimpleNamespace(
        tick=tick,
        self_state=self_state,
        rune_flags=np.zeros(5, dtype=np.float32),
        entity_count=0,
        entities=np.zeros((8, 9), dtype=np.float32),
        reward_item_pickup=pickup,
    )


def _write_sidecars(root, map_name="prototype"):
    (root / f"{map_name}.lattice.json").write_text(json.dumps({
        "cell_size": 256,
        "objectives": [
            {"item": "item_quad", "x": 640, "y": 128, "z": 128, "value": 1.0}
        ],
        "danger": [[-256, -256, -256, 0, 0, 0]],
        "spawns": [],
        "items": [],
    }))
    (root / f"{map_name}.routes.json").write_text(json.dumps({
        "version": 1,
        "nodes": [{
            "id": 0, "type": "item", "class": "item_quad",
            "x": 384, "y": 128, "z": 128, "respawn_s": 60,
            "axis": "offense", "value": 1.5,
        }],
        "routes": [{
            "id": 0, "archetype": "offense", "node_ids": [0],
        }],
    }))


def _attest_sidecars(root, map_name):
    payloads = {
        f"{map_name}.bsp": b"compiled-bsp",
        f"{map_name}.json": b"# hook zones\n",
        f"{map_name}.lattice.json": (
            root / f"{map_name}.lattice.json"
        ).read_bytes(),
        f"{map_name}.routes.json": (
            root / f"{map_name}.routes.json"
        ).read_bytes(),
    }
    for filename, payload in payloads.items():
        (root / filename).write_bytes(payload)
    manifest = build_manifest(map_name, {"generator": "test"}, payloads)
    (root / installed_manifest_name(map_name)).write_bytes(
        encode_manifest(manifest)
    )


def test_preload_and_route_timing_reach_live_memory(tmp_path, monkeypatch):
    _write_sidecars(tmp_path)
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    reward = VoxelSpatialReward.from_env(seed=7)
    obs = _obs(pos=(128, 128, 128), tick=10)

    reward.reset("prototype", obs)
    features = reward.memory_features(obs)

    assert reward.sidecar_sources["prototype"]["lattice"].endswith(
        "prototype.lattice.json"
    )
    assert reward.sidecar_sources["prototype"]["routes"].endswith(
        "prototype.routes.json"
    )
    assert features[12] > 0.0  # nearby generated danger prior
    assert features[16] > 0.0  # objective/readiness opportunity pull
    assert reward.dynamic_cells["prototype"]
    assert reward.selected_route == "offense"


def test_live_farm_preload_requires_and_records_bundle_attestation(
    tmp_path, monkeypatch,
):
    map_name = "mllive_12345678"
    _write_sidecars(tmp_path, map_name)
    _attest_sidecars(tmp_path, map_name)
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    reward = VoxelSpatialReward.from_env(seed=7)

    reward.reset(map_name, _obs(pos=(128, 128, 128), tick=10))

    sources = reward.sidecar_sources[map_name]
    assert sources["bundle"].endswith(f"{map_name}.bundle.json")
    assert len(sources["bundle_sha256"]) == 64
    assert sources["item_timing"].endswith(f"{map_name}.routes.json")
    assert map_name in reward.item_timings


def test_live_farm_preload_rejects_corrupt_attested_route_graph(
    tmp_path, monkeypatch,
):
    map_name = "mllive_12345678"
    _write_sidecars(tmp_path, map_name)
    _attest_sidecars(tmp_path, map_name)
    (tmp_path / f"{map_name}.routes.json").write_text("{}")
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    reward = VoxelSpatialReward.from_env(seed=7)

    reward.reset(map_name, _obs(pos=(128, 128, 128), tick=10))

    sources = reward.sidecar_sources[map_name]
    assert "routes" not in sources
    assert "mismatch" in sources["routes_error"]
    assert map_name not in reward.item_timings


def test_live_farm_preload_rejects_unattested_sidecars(tmp_path, monkeypatch):
    map_name = "mllive_12345678"
    _write_sidecars(tmp_path, map_name)
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    reward = VoxelSpatialReward.from_env(seed=7)

    reward.reset(map_name, _obs(pos=(128, 128, 128), tick=10))

    sources = reward.sidecar_sources[map_name]
    assert "lattice" not in sources
    assert "no installed bundle manifest" in sources["lattice_error"]
    assert "routes" not in sources
    assert "no installed bundle manifest" in sources["routes_error"]


def test_own_pickup_rephases_item_readiness(tmp_path, monkeypatch):
    _write_sidecars(tmp_path)
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    reward = VoxelSpatialReward.from_env(seed=7)
    reward.reset("prototype", _obs(pos=(384, 128, 128), tick=10))

    reward._refresh_route_heat(
        _obs(pos=(384, 128, 128), tick=20, pickup=1.0)
    )

    row = reward.item_timings["prototype"].rows[0]
    assert not row.present
    assert row.taken_by_me
    assert row.predicted_available_t == pytest.approx(62.0)


def test_route_heat_refresh_is_throttled_but_pickups_force_it(
    tmp_path, monkeypatch
):
    _write_sidecars(tmp_path)
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    reward = VoxelSpatialReward.from_env(seed=7)
    reward.reset("prototype", _obs(pos=(128, 128, 128), tick=10))

    assert not reward._refresh_route_heat(
        _obs(pos=(128, 128, 128), tick=11)
    )
    assert reward._route_heat_last_tick["prototype"] == 10
    assert reward._refresh_route_heat(
        _obs(pos=(128, 128, 128), tick=15)
    )
    assert reward._route_heat_last_tick["prototype"] == 15
    assert reward._refresh_route_heat(
        _obs(pos=(384, 128, 128), tick=16, pickup=1.0)
    )
    assert reward._route_heat_last_tick["prototype"] == 16


def test_memory_features_cache_one_nearest_pass_per_tick(monkeypatch):
    reward = VoxelSpatialReward.from_env(seed=3)
    reward.map_name = "prototype"
    reward.session_memories["prototype"] = {
        (1, 0, 0): SessionMemoryCell(kills=2.0, enemy_seen=1.0),
        (-1, 0, 0): SessionMemoryCell(deaths=2.0, damage_taken=30.0),
    }
    obs = _obs(pos=(0, 0, 0), tick=20)
    original = reward._nearest_memory_signals
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reward, "_nearest_memory_signals", counting)
    first = reward.memory_features(obs)
    second = reward.memory_features(obs)
    assert calls == 1
    assert first is second

    reward._invalidate_feature_cache()
    reward.memory_features(obs)
    assert calls == 2


def test_batched_nearest_signals_match_single_channel_queries():
    reward = VoxelSpatialReward.from_env(seed=4)
    reward.map_name = "prototype"
    reward.session_memories["prototype"] = {
        (2, 0, 0): SessionMemoryCell(kills=2.0, enemy_seen=2.0),
        (-2, 0, 0): SessionMemoryCell(deaths=2.0, damage_taken=50.0),
        (0, 2, 0): SessionMemoryCell(self_fire=3.0),
    }
    obs = _obs(pos=(0, 0, 0), tick=30)
    kinds = ("engagement", "threat", "opportunity", "self_fire", "deaths")

    together = reward._nearest_memory_signals(obs, kinds)

    for kind in kinds:
        assert together[kind] == reward._nearest_memory_signal(obs, kind)


def test_lattice_checkpoint_round_trip_excludes_dynamic_heat(tmp_path):
    original = VoxelSpatialReward.from_env(seed=1)
    original.map_name = "prototype"
    original.session_memories["prototype"] = {
        (1, 2, 3): SessionMemoryCell(
            kills=2.0,
            deaths=1.0,
            prior_opportunity=8.0,
            readiness=3.0,
            route_bias=1.0,
        )
    }
    path = save_lattice_state([original], tmp_path / "lattice.json.gz", 1234)
    restored = VoxelSpatialReward.from_env(seed=2)

    stats = load_lattice_state([restored], path)
    cell = restored.session_memories["prototype"][(1, 2, 3)]

    assert stats == {"env_steps": 1234, "instances": 1, "cells": 1}
    assert cell.kills == 2.0
    assert cell.deaths == 1.0
    assert cell.prior_opportunity == 8.0
    assert cell.readiness == 0.0
    assert cell.route_bias == 0.0


@pytest.mark.skipif(torch is None, reason="direction objective requires PyTorch")
def test_direction_objective_rewards_correct_local_movement():
    obs = torch.zeros((1, 1, 219), dtype=torch.float32)
    obs[..., 6] = 0.5
    # Opportunity vector +X with full strength; yaw=0 makes +X local forward.
    obs[..., -24 + 13] = 1.0
    obs[..., -24 + 16] = 1.0
    forward = {"cont_mean": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])}
    backward = {"cont_mean": torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]])}

    good = _lattice_direction_loss(forward, obs)
    bad = _lattice_direction_loss(backward, obs)

    assert good["samples"].item() == 1.0
    assert good["cosine"].item() == pytest.approx(1.0)
    assert good["loss"].item() == pytest.approx(0.0)
    assert bad["cosine"].item() == pytest.approx(-1.0)
    assert bad["loss"].item() == pytest.approx(2.0)


@pytest.mark.skipif(torch is None, reason="direction objective requires PyTorch")
def test_direction_objective_converts_visible_combat_to_pursuit():
    obs = torch.zeros((1, 1, 219), dtype=torch.float32)
    obs[..., 6] = 0.5
    obs[..., -24 + 13] = 1.0
    obs[..., -24 + 16] = 1.0
    obs[..., 9] = 0.5  # ammo
    obs[..., 10:13] = torch.tensor([0.25, 0.0, 0.0])
    obs[..., 10 + 6] = 0.5  # enemy health
    obs[..., 10 + 7] = 1.0  # is_enemy
    obs[..., 10 + 8] = 1.0  # visible
    params = {"cont_mean": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])}

    result = _lattice_direction_loss(params, obs)

    assert result["samples"].item() == 1.0
    assert result["cosine"].item() == pytest.approx(0.5)
    assert result["loss"].item() == pytest.approx(0.0)


@pytest.mark.skipif(torch is None, reason="direction objective requires PyTorch")
def test_direction_objective_uses_quake_right_vector_sign():
    obs = torch.zeros((1, 1, 219), dtype=torch.float32)
    obs[..., 6] = 0.5
    # At yaw zero, world +Y is local LEFT because Quake right points -Y.
    obs[..., -24 + 14] = 1.0
    obs[..., -24 + 16] = 1.0
    local_left = {"cont_mean": torch.tensor([[[0.0, -1.0, 0.0, 0.0]]])}

    result = _lattice_direction_loss(local_left, obs)

    assert result["cosine"].item() == pytest.approx(1.0)
    assert result["loss"].item() == pytest.approx(0.0)


@pytest.mark.skipif(torch is None, reason="direction objective requires PyTorch")
def test_direction_objective_ignores_dead_transitions():
    obs = torch.zeros((1, 1, 219), dtype=torch.float32)
    obs[..., -24 + 13] = 1.0
    obs[..., -24 + 16] = 1.0
    params = {"cont_mean": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])}

    result = _lattice_direction_loss(params, obs)

    assert result["samples"].item() == 0.0
    assert result["loss"].item() == 0.0
