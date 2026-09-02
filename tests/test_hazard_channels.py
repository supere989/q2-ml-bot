"""Wire v5 survival-pack tests: MOD-typed hazard channels, threat
de-conflation, per-target thermal attribution, and self-exposure shaping.

Design contract (docs/SURVIVAL-SUCCESS-CORPUS-2026-07-24.md, P1):
environmental damage/deaths (MOD water/slime/lava/crush/falling) deposit
into the hazard channel, never into combat threat; a missing/stale MOD
falls back to the combat channel so no event is ever lost.
"""

import gzip
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from harness.client_protocol import (
    CLIENT_TELEMETRY_SIZE,
    ML_CLIENT_WIRE_VERSION,
)
from harness.protocol import (
    ENVIRONMENTAL_MODS,
    ML_OBS_MAGIC,
    MOD_FALLING,
    MOD_LAVA,
    MOD_RAILGUN,
    OBS_FMT,
    OBS_SIZE,
    parse_obs,
)
from harness.spatial import (
    SessionMemoryCell,
    VoxelSpatialReward,
    load_lattice_state,
)

# Number of packed values preceding the wire v5 block in OBS_FMT:
# III ff 10f 9f*8 I 4f*16 8f*4 I 5f 6f 3f 5f 3f 2f 4B
_V5_INDEX = 3 + 2 + 10 + 72 + 1 + 64 + 32 + 1 + 5 + 6 + 3 + 5 + 3 + 2 + 4


def _obs(*, pos=(0.0, 0.0, 0.0), tick=0, health=100.0, entities=None,
         entity_count=0, entity_debug=None, damage_taken=0.0, deaths=0.0,
         kills=0.0, last_damage_mod=0, last_death_mod=0,
         last_hit_target_edict=0, last_hit_target_epoch=0,
         self_exposure=0.0):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[:3] = pos
    self_state[6] = health
    self_state[7] = 50.0
    self_state[9] = 10.0
    if entities is None:
        entities = np.zeros((8, 9), dtype=np.float32)
    if entity_debug is None:
        entity_debug = np.zeros((8, 4), dtype=np.uint32)
    return SimpleNamespace(
        tick=tick,
        yaw=0.0,
        pitch=0.0,
        self_state=self_state,
        entities=entities,
        entity_count=entity_count,
        entity_debug=entity_debug,
        rune_flags=np.zeros(5, dtype=np.float32),
        audio=np.zeros(5, dtype=np.float32),
        action_debug=np.zeros(12, dtype=np.float32),
        hook_zones=np.zeros((4, 8), dtype=np.float32),
        hook_zone_count=0,
        inbound_dmg_dist=-1.0,
        inbound_dmg_recency=0.0,
        is_terminal=False,
        reward_damage_dealt=0.0,
        reward_damage_taken=damage_taken,
        reward_kill=kills,
        reward_death=deaths,
        reward_item_pickup=0.0,
        reward_hook_traversal=0.0,
        reward_damage_taken_prox=0.0,
        reward_offense=0.0,
        reward_survival=0.0,
        last_damage_mod=last_damage_mod,
        last_death_mod=last_death_mod,
        last_hit_target_edict=last_hit_target_edict,
        last_hit_target_epoch=last_hit_target_epoch,
        self_exposure=self_exposure,
    )


def _visible_enemy(index=0, rel=(300.0, 0.0, 0.0), exposure=0.5):
    entities = np.zeros((8, 9), dtype=np.float32)
    entities[index] = (*rel, 0.0, 0.0, 0.0, 100.0, 1.0, exposure)
    return entities


def _cell_of(reward, pos=(0.0, 0.0, 0.0)):
    return reward.cell_for(_obs(pos=pos))


# ── wire sizes / parse round-trip ────────────────────────────────────────

def test_wire_v5_sizes_match_c_static_asserts():
    # Guards the coordinated bump: these numbers are the _Static_assert
    # values in ml_client_wire.h (server) — ml_obs_t 1060, telemetry 1156.
    assert struct.calcsize(OBS_FMT) == 1060
    assert OBS_SIZE == 1060
    assert CLIENT_TELEMETRY_SIZE == 96 + 1060
    assert ML_CLIENT_WIRE_VERSION == 5


def test_parse_obs_reads_v5_fields():
    values = [0] * len(struct.unpack(OBS_FMT, bytes(OBS_SIZE)))
    values[0] = ML_OBS_MAGIC
    values[1] = 77
    values[2] = 2
    (values[_V5_INDEX], values[_V5_INDEX + 1],
     values[_V5_INDEX + 2], values[_V5_INDEX + 3],
     values[_V5_INDEX + 4], values[_V5_INDEX + 5],
     values[_V5_INDEX + 6], values[_V5_INDEX + 7]) = (
        MOD_LAVA, MOD_RAILGUN, 5, 0x123, 0.75, 3.0, 7.0, 42.0,
    )
    obs = parse_obs(struct.pack(OBS_FMT, *values))
    assert obs is not None
    assert obs.last_damage_mod == MOD_LAVA
    assert obs.last_death_mod == MOD_RAILGUN
    assert obs.last_hit_target_edict == 5
    assert obs.last_hit_target_epoch == 0x123
    assert obs.self_exposure == pytest.approx(0.75)
    assert obs.score_self == pytest.approx(3.0)
    assert obs.score_leader == pytest.approx(7.0)
    assert obs.time_remaining == pytest.approx(42.0)


# ── MOD classification + threat de-conflation ────────────────────────────

def test_environmental_damage_deposits_to_hazard_not_threat():
    reward = VoxelSpatialReward()
    reward.reset("hazmap", _obs(tick=0))
    reward.update(_obs(tick=1, damage_taken=50.0, last_damage_mod=MOD_LAVA))
    entry = reward._memory_for_map("hazmap")[_cell_of(reward)]
    assert entry.hazard_damage == 50.0
    assert entry.damage_taken == 0.0
    assert reward._threat_score(entry) == 0.0


def test_combat_damage_still_feeds_threat():
    reward = VoxelSpatialReward()
    reward.reset("hazmap", _obs(tick=0))
    reward.update(_obs(tick=1, damage_taken=50.0,
                       last_damage_mod=MOD_RAILGUN))
    entry = reward._memory_for_map("hazmap")[_cell_of(reward)]
    assert entry.hazard_damage == 0.0
    assert entry.damage_taken == 50.0
    assert reward._threat_score(entry) == pytest.approx(0.03 * 50.0)


def test_environmental_death_skips_combat_death_counter():
    reward = VoxelSpatialReward()
    reward.reset("hazmap", _obs(tick=0))
    reward.update(_obs(tick=1, deaths=1.0, last_death_mod=MOD_FALLING))
    entry = reward._memory_for_map("hazmap")[_cell_of(reward)]
    assert entry.hazard_deaths == 1.0
    assert entry.deaths == 0.0
    assert reward._threat_score(entry) == 0.0


def test_stale_mod_falls_back_to_combat_and_never_loses_death():
    reward = VoxelSpatialReward()
    reward.reset("hazmap", _obs(tick=0))
    # MOD 0 (stale/unknown) must keep the historical combat behavior.
    reward.update(_obs(tick=1, deaths=1.0, last_death_mod=0))
    entry = reward._memory_for_map("hazmap")[_cell_of(reward)]
    assert entry.deaths == 1.0
    assert entry.hazard_deaths == 0.0


def test_environmental_mod_set_matches_server_taxonomy():
    # g_local.h: WATER=17 SLIME=18 LAVA=19 CRUSH=20 FALLING=22
    assert ENVIRONMENTAL_MODS == frozenset({17, 18, 19, 20, 22})
    assert MOD_RAILGUN not in ENVIRONMENTAL_MODS


# ── hazard aversion reward term ──────────────────────────────────────────

def _seed_hazard_cell(reward, map_name="hazmap"):
    reward.reset(map_name, _obs(tick=0))
    cell = _cell_of(reward)
    entry = SessionMemoryCell(hazard_damage=100.0, hazard_deaths=1.0)
    reward._memory_for_map(map_name)[cell] = entry
    return cell


def test_hazard_aversion_fires_from_typed_channel():
    reward = VoxelSpatialReward()
    _seed_hazard_cell(reward)
    _, info = reward.update(_obs(tick=1))
    assert info["memory_hazard_aversion"] > 0.0
    expected = -reward.session_memory_hazard_aversion * info[
        "memory_hazard_aversion"
    ]
    # No engagement/opportunity/death/self-fire memory exists, so the
    # hazard term is the whole session-memory delta.
    assert info["session_memory_bonus"] == pytest.approx(expected)


def test_hazard_aversion_respects_lattice_reward_gate():
    reward = VoxelSpatialReward(lattice_reward_enabled=False)
    _seed_hazard_cell(reward)
    _, info = reward.update(_obs(tick=1))
    # The metric stays observable; the reward contribution gates off.
    assert info["memory_hazard_aversion"] > 0.0
    assert info["session_memory_bonus"] == 0.0


def test_checkpoint_load_tolerates_cells_without_hazard_fields(tmp_path):
    path = tmp_path / "lattice.json.gz"
    payload = {
        "version": 1,
        "env_steps": 10,
        "instances": [{
            "maps": {
                "hazmap": [
                    {"cell": [0, 0, 0], "deaths": 2.0, "damage_taken": 40.0},
                ],
            },
        }],
    }
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(payload))
    reward = VoxelSpatialReward()
    result = load_lattice_state([reward], path)
    assert result["cells"] == 1
    entry = reward._memory_for_map("hazmap")[(0, 0, 0)]
    assert entry.deaths == 2.0
    assert entry.hazard_damage == 0.0
    assert entry.hazard_deaths == 0.0


# ── thermal per-target attribution ───────────────────────────────────────

def _thermal_obs(tick, epoch_a=0x123, epoch_b=0x321):
    entities = _visible_enemy(0, rel=(300.0, 0.0, 0.0))
    entities[1] = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0, 0.5)
    entity_debug = np.zeros((8, 4), dtype=np.uint32)
    entity_debug[0] = (5, 4, 2, epoch_a << 18)
    entity_debug[1] = (6, 5, 2, epoch_b << 18)
    return _obs(tick=tick, entities=entities, entity_count=2,
                entity_debug=entity_debug)


def test_kill_clears_only_the_attributed_track():
    reward = VoxelSpatialReward()
    reward._update_thermal_tracks(_thermal_obs(tick=1))
    track_a = (5 << 14) | 0x123
    track_b = (6 << 14) | 0x321
    assert track_a in reward._thermal_tracks
    assert track_b in reward._thermal_tracks

    reward._update_thermal_tracks(_obs(
        tick=2, kills=1.0,
        last_hit_target_edict=5, last_hit_target_epoch=0x123,
    ))
    assert track_a not in reward._thermal_tracks
    assert track_b in reward._thermal_tracks


def test_kill_without_attribution_clears_all_tracks():
    reward = VoxelSpatialReward()
    reward._update_thermal_tracks(_thermal_obs(tick=1))
    assert len(reward._thermal_tracks) == 2
    reward._update_thermal_tracks(_obs(tick=2, kills=1.0))
    assert len(reward._thermal_tracks) == 0


# ── self-exposure shaping ────────────────────────────────────────────────

def test_exposed_while_visible_enemy_penalty():
    reward = VoxelSpatialReward()
    reward.reset("hazmap", _obs(tick=0))
    _, info = reward.update(_obs(
        tick=1, entities=_visible_enemy(), entity_count=1,
        self_exposure=0.8,
    ))
    assert info["self_exposure"] == pytest.approx(0.8)
    assert info["exposed_while_visible"] == 1.0
    assert info["exposure_shaping"] == pytest.approx(
        -reward.exposure_penalty * 0.8
    )


def test_covered_reward_when_recently_damaged_and_unseen():
    reward = VoxelSpatialReward()
    reward.reset("hazmap", _obs(tick=0))
    reward.last_damage_tick = 1  # recent damage, now out of sight
    _, info = reward.update(_obs(tick=2, self_exposure=0.25))
    assert info["covered_after_damage"] == 1.0
    assert info["exposure_shaping"] == pytest.approx(
        reward.cover_reward * (1.0 - 0.25)
    )


def test_exposure_shaping_respects_lattice_reward_gate():
    reward = VoxelSpatialReward(lattice_reward_enabled=False)
    reward.reset("hazmap", _obs(tick=0))
    _, info = reward.update(_obs(
        tick=1, entities=_visible_enemy(), entity_count=1,
        self_exposure=0.8,
    ))
    assert info["self_exposure"] == pytest.approx(0.8)
    assert info["exposed_while_visible"] == 0.0
    assert info["exposure_shaping"] == 0.0
