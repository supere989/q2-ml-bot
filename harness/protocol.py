"""
protocol.py — Python-side serialization matching ml_bridge.h structs.

Both sides use plain C struct layout (little-endian, no padding beyond
natural alignment).  struct.calcsize must match sizeof() in C.
"""

import os
import struct
import hashlib
import json
import numpy as np

# Reward weights — overridable via env vars so the Karpathy loop can sweep them.
# Defaults match the original hardcoded values.
def _w(name: str, default: float) -> float:
    try: return float(os.environ.get(name, default))
    except (TypeError, ValueError): return default

R_DAMAGE_DEALT  = _w("R_DAMAGE_DEALT",  0.003)
R_KILL          = _w("R_KILL",          1.0)
R_DAMAGE_TAKEN  = _w("R_DAMAGE_TAKEN",  0.001)
R_DEATH         = _w("R_DEATH",         0.5)
R_ITEM          = _w("R_ITEM",          0.1)
# Deprecated: the engine channel cannot prove that grapple displacement moved
# toward an intended destination, so treating it as a rate/value signal made
# policies spam hook.  Positive hook shaping now lives in
# VoxelSpatialReward's lattice-target correction contract.
R_HOOK          = 0.0
from dataclasses import dataclass
from enum import IntEnum

ML_BASE_PORT   = 27950
ML_PROTOCOL_GENERATION = 2
ML_OBS_MAGIC   = 0x514D324F   # "QM2O": multires/Atlas observation generation
ML_ACT_MAGIC   = 0x514D3241   # "QM2A": multires/Atlas action generation
ML_MAX_ENTITIES = 8
ML_RAY_COUNT    = 16
ML_HOOK_ZONES   = 4
ML_ENTITY_DUCKED = 0x20000
ML_ENTITY_EPOCH_SHIFT = 18
ML_ENTITY_EPOCH_MASK = 0xFFFC0000

ML_CONTROL_UNKNOWN    = 0
ML_CONTROL_HUMAN      = 1
ML_CONTROL_ML_BOT     = 2
ML_CONTROL_LEGACY_BOT = 3

ML_TERMINAL_NONE         = 0
ML_TERMINAL_DEATH        = 1
ML_TERMINAL_INTERMISSION = 2

# action_debug flags emitted by game.so. These are debug/control-plane
# metadata and intentionally remain outside the policy observation vector.
ML_FIRE_GATE_PROTECTED  = 0x01
ML_FIRE_GATE_TARGET     = 0x02
ML_FIRE_GATE_SUPPRESSED = 0x04
ML_HIT_STREAK_SHIFT = 8
ML_HIT_STREAK_MASK = 0x0000FF00
ML_ACTION_GENERATION_SHIFT = 16
ML_ACTION_GENERATION_MASK = 0x00FF0000
ML_ACTION_GENERATION_COUNT = 192


class ActionDebugIndex(IntEnum):
    """Frozen indices for the authoritative action echo outside policy input."""

    TICK = 0
    ACCEPTED = 1
    TIMEOUT_COUNT = 2
    WEAPON = 3
    MOVE_FORWARD = 4
    MOVE_RIGHT = 5
    LOOK_YAW = 6
    LOOK_PITCH = 7
    VERTICAL_INTENT = 8
    APPLIED_UPMOVE = 9
    ACTUAL_DUCKED = 10
    WATER_VERTICAL_MODE = 11
    FIRE = 12
    HOOK = 13
    FLAGS = 14

# ── Observation ────────────────────────────────────────────────────────

# ml_self_t:  pos[3] vel[3] health armor weapon_id ammo  → 10 floats
_SELF_FMT = "10f"

# ml_entity_t: eye-to-damage-point[3], local relative velocity[3], health,
# is_enemy, signed exact exposure → 9 floats
_ENT_FMT  = "9f"

# ml_ray_t: direction[3] distance → 4 floats
_RAY_FMT  = "4f"

# ml_hook_zone_t: anchor[3] landing[3] distance flags → 8 floats
_HOOK_FMT = "8f"

# ml_audio_t: sound_dir[3] sound_age alert_level → 5 floats
_AUDIO_FMT = "5f"

# ml_entity_debug_t: edict_index client_slot control_source flags → 4 uint32
_DEBUG_FMT = "4I"

# ml_action_debug_t: provenance, movement/look, requested vertical enum,
# signed applied upmove, resulting stance/water state, buttons, and flags.
_ACTION_DEBUG_FMT = "4I4fIi5I"

OBS_FMT = (
    "<"                                     # little-endian
    "III"                                   # magic tick bot_slot
    "ff"                                    # yaw pitch
    + _SELF_FMT                             # self
    + _ENT_FMT * ML_MAX_ENTITIES            # entities[8]
    + "I"                                   # entity_count
    + _RAY_FMT  * ML_RAY_COUNT              # rays[16]
    + _HOOK_FMT * ML_HOOK_ZONES             # hook_zones[4]
    + "I"                                   # hook_zone_count
    + _AUDIO_FMT                            # audio
    + "6f"                                  # reward components x6
    + "3f"                                  # ext reward: dmg_prox, offense, survival
    + "5f"                                  # rune_flags[5]
    + "3f"                                  # inbound_dmg_dir[3]
    + "2f"                                  # inbound_dmg_dist, inbound_dmg_recency
    + "3f"                                  # actual_ducked, standing_blocked, water mode
    + "4B"                                  # is_terminal, terminal_reason, 2 pad bytes
    + _DEBUG_FMT                            # self_debug
    + _DEBUG_FMT * ML_MAX_ENTITIES          # entity_debug[8]
    + _ACTION_DEBUG_FMT                     # action_debug
)

OBS_SIZE = struct.calcsize(OBS_FMT)

OBS_ENGINE_SENSOR_DIM = 185
OBS_ENGINE_EXTENSION_DIM = 10
OBS_STANCE_DIM = 3
OBS_FACTUAL_DIM = 198
OBS_DYN_DIM = 24
OBS_RECOVERY_DIM = 16
OBS_OBJECTIVE_COUNT = 4
OBS_OBJECTIVE_DIM = 15
OBS_OBJECTIVES_DIM = OBS_OBJECTIVE_COUNT * OBS_OBJECTIVE_DIM
OBS_DIM = OBS_FACTUAL_DIM + OBS_DYN_DIM + OBS_RECOVERY_DIM + OBS_OBJECTIVES_DIM
# Source-level bridge for the existing Dyn producer while B3 replaces its
# storage/runtime. This is a fixed width alias, not an observation selector.
OBS_SESSION_MEMORY_DIM = OBS_DYN_DIM

FACTUAL_SLICE = slice(0, OBS_FACTUAL_DIM)
DYN_SLICE = slice(OBS_FACTUAL_DIM, OBS_FACTUAL_DIM + OBS_DYN_DIM)
RECOVERY_SLICE = slice(DYN_SLICE.stop, DYN_SLICE.stop + OBS_RECOVERY_DIM)
OBJECTIVES_SLICE = slice(RECOVERY_SLICE.stop, OBS_DIM)
OBJECTIVE_SLICES = tuple(
    slice(
        OBJECTIVES_SLICE.start + index * OBS_OBJECTIVE_DIM,
        OBJECTIVES_SLICE.start + (index + 1) * OBS_OBJECTIVE_DIM,
    )
    for index in range(OBS_OBJECTIVE_COUNT)
)

# Input normalization so every obs feature lands ~[-1,1] instead of health
# (0-200) competing with position (0-4096) in the same linear layer. The bot
# couldn't cleanly perceive HP/ammo before this — and the survivability
# projection needs clean HP/armor/ammo + enemy-HP to be computable at all.
# self_state: [pos.xyz, vel.xyz, health, armor, weapon_id, ammo]
SELF_SCALE = np.array([4096, 4096, 4096, 1000, 1000, 1000,
                       200, 200, 40, 100], dtype=np.float32)
# entity: [aim_point_local.xyz, relative_velocity_local.xyz, health,
#          is_enemy(0/1), signed_exposure(-1..1)]. Exposure magnitude is the
# exact clear-probe fraction; positive is fire-actionable, negative is sensed
# while the shooter is protected (track/aim only).
ENT_SCALE = np.array([4096, 4096, 4096, 1000, 1000, 1000, 200, 1, 1],
                     dtype=np.float32)
_SELF_FEATURE_NAMES = (
    "self_pos_x", "self_pos_y", "self_pos_z",
    "self_vel_x", "self_vel_y", "self_vel_z",
    "self_health", "self_armor", "self_weapon_id", "self_ammo",
)
_ENTITY_FIELDS = (
    "aim_forward", "aim_quake_right", "aim_up",
    "relative_velocity_forward", "relative_velocity_quake_right",
    "relative_velocity_up", "health", "is_enemy", "signed_exposure",
)
_RAY_FIELDS = ("dir_x", "dir_y", "dir_z", "distance")
_HOOK_FIELDS = (
    "anchor_x", "anchor_y", "anchor_z", "landing_x", "landing_y",
    "landing_z", "distance", "flags",
)
_AUDIO_FEATURE_NAMES = (
    "audio_direction_x", "audio_direction_y", "audio_direction_z",
    "audio_age", "audio_alert_level",
)
_DYN_FEATURE_NAMES = (
    "dyn_l2_current_engagement", "dyn_l2_current_threat",
    "dyn_l2_current_opportunity", "dyn_l2_current_self_fire",
    "dyn_l2_current_confidence", "dyn_immediate_thermal_forward",
    "dyn_immediate_thermal_quake_right", "dyn_immediate_thermal_up",
    "dyn_immediate_thermal_heat", "dyn_combat_threat_forward",
    "dyn_combat_threat_quake_right", "dyn_combat_threat_up",
    "dyn_combat_threat_score", "dyn_opportunity_forward",
    "dyn_opportunity_quake_right", "dyn_opportunity_up",
    "dyn_opportunity_score", "dyn_self_fire_forward",
    "dyn_self_fire_quake_right", "dyn_self_fire_up", "dyn_self_fire_score",
    "dyn_win_margin", "dyn_effective_health_norm", "dyn_own_dps_share",
)
_RECOVERY_FEATURE_NAMES = (
    "recovery_hazard_lava", "recovery_hazard_slime",
    "recovery_hazard_hurt", "recovery_hazard_void_or_lethal_drop",
    "recovery_hazard_crush_or_current", "recovery_hazard_strength",
    "recovery_hull_clearance", "recovery_cost_to_safety", "recovery_confidence",
    "recovery_primary_forward", "recovery_primary_quake_right",
    "recovery_primary_up", "recovery_alternate_forward",
    "recovery_alternate_quake_right", "recovery_alternate_up",
    "recovery_time_to_impact",
)
_OBJECTIVE_FIELDS = (
    "forward", "quake_right", "up", "cost",
    "risk", "confidence", "availability_belief", "class_weapon",
    "class_ammunition", "class_health", "class_armor", "class_powerup",
    "class_rune", "class_control", "class_spawn_egress",
)

FACTUAL_FEATURE_NAMES = (
    _SELF_FEATURE_NAMES
    + tuple(
        f"entity_{slot}_{field}"
        for slot in range(ML_MAX_ENTITIES)
        for field in _ENTITY_FIELDS
    )
    + tuple(
        f"ray_{ray}_{field}"
        for ray in range(ML_RAY_COUNT)
        for field in _RAY_FIELDS
    )
    + tuple(
        f"hook_zone_{zone}_{field}"
        for zone in range(ML_HOOK_ZONES)
        for field in _HOOK_FIELDS
    )
    + _AUDIO_FEATURE_NAMES
    + ("view_yaw", "view_pitch")
    + (
        "rune_resist", "rune_strength", "rune_haste", "rune_regen",
        "rune_vampire", "inbound_damage_direction_x",
        "inbound_damage_direction_y", "inbound_damage_direction_z",
        "inbound_damage_distance", "inbound_damage_recency",
        "actual_ducked", "standing_blocked", "water_vertical_mode",
    )
)
DYN_FEATURE_NAMES = _DYN_FEATURE_NAMES
RECOVERY_FEATURE_NAMES = _RECOVERY_FEATURE_NAMES
OBJECTIVE_FEATURE_NAMES = tuple(
    f"guide_{slot}_{field}"
    for slot in range(OBS_OBJECTIVE_COUNT)
    for field in _OBJECTIVE_FIELDS
)
POLICY_FEATURE_NAMES = (
    FACTUAL_FEATURE_NAMES + DYN_FEATURE_NAMES + RECOVERY_FEATURE_NAMES
    + OBJECTIVE_FEATURE_NAMES
)
POLICY_FEATURE_INDEX = {
    name: index for index, name in enumerate(POLICY_FEATURE_NAMES)
}
POLICY_FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(
    POLICY_FEATURE_NAMES,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")).hexdigest()
SPATIAL_FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(
    DYN_FEATURE_NAMES + RECOVERY_FEATURE_NAMES + OBJECTIVE_FEATURE_NAMES,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")).hexdigest()

if len(FACTUAL_FEATURE_NAMES) != OBS_FACTUAL_DIM:
    raise RuntimeError("factual feature-name layout is not 198 floats")
if len(POLICY_FEATURE_NAMES) != OBS_DIM or len(POLICY_FEATURE_INDEX) != OBS_DIM:
    raise RuntimeError("policy feature-name layout is not unique and 298 floats")

# ── Action ─────────────────────────────────────────────────────────────

ACT_FMT  = "<IIffff4B"          # magic tick fwd right yaw pitch vertical fire hook weapon
ACT_SIZE = struct.calcsize(ACT_FMT)


class VerticalIntent(IntEnum):
    CROUCH_OR_DOWN = 0
    NEUTRAL = 1
    JUMP_OR_UP = 2


def _strict_vector(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, received {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(vector)


@dataclass(frozen=True)
class SpatialPolicyFeatures:
    """Atlas-bound, per-client policy additions produced outside game telemetry."""

    dyn: np.ndarray
    recovery: np.ndarray
    objectives: np.ndarray

    def to_vector(self) -> np.ndarray:
        dyn = _strict_vector(self.dyn, (OBS_DYN_DIM,), "Dyn features")
        recovery = _strict_vector(
            self.recovery, (OBS_RECOVERY_DIM,), "recovery features"
        )
        objectives = _strict_vector(
            self.objectives,
            (OBS_OBJECTIVE_COUNT, OBS_OBJECTIVE_DIM),
            "objective features",
        )
        return np.concatenate((dyn, recovery, objectives.reshape(-1))).astype(
            np.float32, copy=False
        )


@dataclass
class Observation:
    tick:           int
    bot_slot:       int
    yaw:            float
    pitch:          float

    # self state as flat numpy array [pos_xyz, vel_xyz, health, armor, weapon_id, ammo]
    self_state:     np.ndarray  # shape (10,)

    # entities: [eye-to-best-damage-point local xyz, local relative velocity,
    #            health, is_enemy, signed exact exposure] × N
    entities:       np.ndarray  # shape (ML_MAX_ENTITIES, 9)
    entity_count:   int

    # lidar-style rays [dir_xyz, distance] × 16
    rays:           np.ndarray  # shape (ML_RAY_COUNT, 4)

    # hook zones [anchor_xyz, landing_xyz, distance, flags] × 4
    hook_zones:     np.ndarray  # shape (ML_HOOK_ZONES, 8)
    hook_zone_count: int

    # audio [sound_dir_xyz, sound_age, alert_level]
    audio:          np.ndarray  # shape (5,)

    # reward components
    reward_damage_dealt:    float
    reward_damage_taken:    float
    reward_kill:            float
    reward_death:           float
    reward_item_pickup:     float
    reward_hook_traversal:  float

    # extended reward channels (always present; consumed by both runs)
    reward_damage_taken_prox: float
    reward_offense:           float
    reward_survival:          float

    # Always-on factual observation block.
    rune_flags:          np.ndarray  # shape (5,) resist,strength,haste,regen,vampire
    inbound_dmg_dir:     np.ndarray  # shape (3,) unit vector toward attacker
    inbound_dmg_dist:    float       # units, -1 if none recent
    inbound_dmg_recency: float       # 1=fresh → 0 by ~1s
    actual_ducked:       bool        # resulting PMF_DUCKED state
    standing_blocked:    bool        # standing player hull is blocked at origin
    water_vertical_mode: bool        # signed upmove means swim down/up

    is_terminal:    bool
    terminal_reason: int

    # debug-only identity metadata; not included in policy input vector
    self_debug:     np.ndarray  # shape (4,) [edict_index, client_slot, source, flags]
    entity_debug:   np.ndarray  # shape (ML_MAX_ENTITIES, 4)
    action_debug:   np.ndarray  # shape (15,) engine-applied action echo

    @property
    def round_intermission(self) -> bool:
        return int(self.terminal_reason) == ML_TERMINAL_INTERMISSION

    @property
    def death_terminal(self) -> bool:
        return int(self.terminal_reason) == ML_TERMINAL_DEATH

    @property
    def reward(self) -> float:
        """Composite reward signal (weights from env, see top of file)."""
        return (
              self.reward_damage_dealt   * R_DAMAGE_DEALT
            + self.reward_kill           * R_KILL
            - self.reward_damage_taken   * R_DAMAGE_TAKEN
            - self.reward_death          * R_DEATH
            + self.reward_item_pickup    * R_ITEM
            + self.reward_hook_traversal * R_HOOK
        )

    def factual_vector(self) -> np.ndarray:
        """Return the frozen 198-float engine/client factual block."""
        entities = self._policy_entities()
        # rays: [dir.xyz (unit), distance] → scale only the distance column.
        rays_n = self.rays.copy()
        rays_n[:, 3] /= 4096.0
        # hook zones: [anchor.xyz, landing.xyz, distance, flags] → scale coords.
        hooks_n = self.hook_zones / np.array(
            [4096, 4096, 4096, 4096, 4096, 4096, 4096, 8], dtype=np.float32)
        parts = [
            self.self_state / SELF_SCALE,           # 10 — normalised
            (entities / ENT_SCALE).flatten(),       # 72 — normalised
            rays_n.flatten(),                       # 64 — distance normalised
            hooks_n.flatten(),                      # 32 — coords normalised
            self.audio,                             # 5  (dir unit, age/alert bounded)
            [self.yaw / 180.0, self.pitch / 90.0], # 2 — normalised facing
            self.rune_flags,                        # 5 — always on
            self.inbound_dmg_dir,                   # 3
        ]                                           # subtotal: 193 dims
        dist_n = (
            self.inbound_dmg_dist / 1500.0
            if self.inbound_dmg_dist >= 0.0 else -1.0
        )
        parts.extend((
            [dist_n, self.inbound_dmg_recency],     # 2
            [
                float(self.actual_ducked),
                float(self.standing_blocked),
                float(self.water_vertical_mode),
            ],                                      # 3
        ))
        factual = np.concatenate(parts).astype(np.float32)
        if factual.shape != (OBS_FACTUAL_DIM,) or not np.isfinite(factual).all():
            raise ValueError("engine/client factual observation is not finite 198-float data")
        return factual

    def to_vector(self, spatial: SpatialPolicyFeatures) -> np.ndarray:
        """Assemble the one accepted 298-float policy generation.

        Spatial additions are mandatory and exact. There is deliberately no
        zero-fill, truncation, legacy-width toggle, or optional-tail path.
        """
        if not isinstance(spatial, SpatialPolicyFeatures):
            raise TypeError("spatial must be an Atlas-bound SpatialPolicyFeatures instance")
        vector = np.concatenate((self.factual_vector(), spatial.to_vector())).astype(
            np.float32, copy=False
        )
        if vector.shape != (OBS_DIM,) or not np.isfinite(vector).all():
            raise ValueError("policy observation is not finite 298-float data")
        return vector

    BASE_OBS_DIM = OBS_FACTUAL_DIM
    OBS_DIM = OBS_DIM

    def _policy_entities(self) -> np.ndarray:
        """Put the nearest visible enemy first for policy input stability."""
        entities = self.entities.copy()
        count = max(0, min(int(self.entity_count), entities.shape[0]))
        if count <= 1:
            return entities

        def key(idx: int):
            ent = entities[idx]
            is_enemy = ent[7] > 0.5
            visible = abs(float(ent[8])) > 0.0
            dist = float(np.linalg.norm(ent[:3]))
            return (
                0 if (is_enemy and visible) else 1,
                dist if (is_enemy and visible) else 0.0,
                0 if is_enemy else 1,
                dist,
            )

        order = sorted(range(count), key=key)
        entities[:count] = entities[order]
        return entities


@dataclass
class Action:
    move_forward: float = 0.0   # [-1, 1]
    move_right:   float = 0.0   # [-1, 1]
    look_yaw:     float = 0.0   # degrees delta
    look_pitch:   float = 0.0   # degrees delta
    vertical_intent: VerticalIntent = VerticalIntent.NEUTRAL
    fire:         bool  = False
    hook:         int   = 0     # 0=idle 1=fire 2=hold 3=release
    weapon:       int   = 0     # 0=no-change, 1-9=select


def parse_obs(data: bytes) -> Observation | None:
    if len(data) != OBS_SIZE:
        return None
    v = struct.unpack(OBS_FMT, data)
    i = 0

    magic, tick, bot_slot = v[i], v[i+1], v[i+2]; i += 3
    if magic != ML_OBS_MAGIC:
        return None

    yaw, pitch = v[i], v[i+1]; i += 2
    self_state = np.array(v[i:i+10], dtype=np.float32); i += 10

    ents_flat = v[i:i + ML_MAX_ENTITIES * 9]; i += ML_MAX_ENTITIES * 9
    entity_count = v[i]; i += 1
    if entity_count > ML_MAX_ENTITIES:
        return None
    entities = np.array(ents_flat, dtype=np.float32).reshape(ML_MAX_ENTITIES, 9)

    rays_flat = v[i:i + ML_RAY_COUNT * 4]; i += ML_RAY_COUNT * 4
    rays = np.array(rays_flat, dtype=np.float32).reshape(ML_RAY_COUNT, 4)

    hooks_flat = v[i:i + ML_HOOK_ZONES * 8]; i += ML_HOOK_ZONES * 8
    hook_zone_count = v[i]; i += 1
    if hook_zone_count > ML_HOOK_ZONES:
        return None
    hook_zones = np.array(hooks_flat, dtype=np.float32).reshape(ML_HOOK_ZONES, 8)

    audio = np.array(v[i:i+5], dtype=np.float32); i += 5

    r_dd, r_dt, r_k, r_d, r_ip, r_ht = v[i:i+6]; i += 6
    r_prox, r_off, r_surv = v[i:i+3]; i += 3
    rune_flags = np.array(v[i:i+5], dtype=np.float32); i += 5
    inbound_dmg_dir = np.array(v[i:i+3], dtype=np.float32); i += 3
    inbound_dmg_dist, inbound_dmg_recency = v[i], v[i+1]; i += 2
    if any(value not in (0.0, 1.0) for value in v[i:i+3]):
        return None
    actual_ducked = bool(v[i])
    standing_blocked = bool(v[i+1])
    water_vertical_mode = bool(v[i+2])
    i += 3
    if v[i] not in (0, 1) or v[i+1] not in (
        ML_TERMINAL_NONE, ML_TERMINAL_DEATH, ML_TERMINAL_INTERMISSION
    ):
        return None
    is_terminal = bool(v[i])
    terminal_reason = int(v[i+1])
    i += 4
    self_debug = np.array(v[i:i+4], dtype=np.uint32); i += 4
    entity_debug_flat = v[i:i + ML_MAX_ENTITIES * 4]; i += ML_MAX_ENTITIES * 4
    entity_debug = np.array(entity_debug_flat, dtype=np.uint32).reshape(ML_MAX_ENTITIES, 4)
    action_debug = np.array(v[i:i+15], dtype=np.float64); i += 15
    if (
        action_debug[ActionDebugIndex.ACCEPTED] not in (0, 1)
        or action_debug[ActionDebugIndex.VERTICAL_INTENT] not in (0, 1, 2)
        or action_debug[ActionDebugIndex.APPLIED_UPMOVE] not in (-320, 0, 320)
        or action_debug[ActionDebugIndex.ACTUAL_DUCKED] not in (0, 1)
        or action_debug[ActionDebugIndex.WATER_VERTICAL_MODE] not in (0, 1)
        or action_debug[ActionDebugIndex.FIRE] not in (0, 1)
        or not 0 <= action_debug[ActionDebugIndex.HOOK] <= 3
        or not 0 <= action_debug[ActionDebugIndex.WEAPON] <= 9
    ):
        return None
    if action_debug[ActionDebugIndex.ACCEPTED]:
        expected_upmove = (-320, 0, 320)[
            int(action_debug[ActionDebugIndex.VERTICAL_INTENT])
        ]
        if action_debug[ActionDebugIndex.APPLIED_UPMOVE] != expected_upmove:
            return None
    factual_values = np.concatenate((
        np.asarray([yaw, pitch], dtype=np.float64),
        self_state.astype(np.float64, copy=False),
        entities.astype(np.float64, copy=False).reshape(-1),
        rays.astype(np.float64, copy=False).reshape(-1),
        hook_zones.astype(np.float64, copy=False).reshape(-1),
        audio.astype(np.float64, copy=False),
        np.asarray([
            r_dd, r_dt, r_k, r_d, r_ip, r_ht, r_prox, r_off, r_surv,
            *rune_flags, *inbound_dmg_dir, inbound_dmg_dist,
            inbound_dmg_recency,
        ], dtype=np.float64),
        action_debug,
    ))
    if not np.isfinite(factual_values).all():
        return None

    return Observation(
        tick=tick, bot_slot=bot_slot, yaw=yaw, pitch=pitch,
        self_state=self_state,
        entities=entities, entity_count=entity_count,
        rays=rays,
        hook_zones=hook_zones, hook_zone_count=hook_zone_count,
        audio=audio,
        reward_damage_dealt=r_dd, reward_damage_taken=r_dt,
        reward_kill=r_k, reward_death=r_d,
        reward_item_pickup=r_ip, reward_hook_traversal=r_ht,
        reward_damage_taken_prox=r_prox,
        reward_offense=r_off, reward_survival=r_surv,
        rune_flags=rune_flags,
        inbound_dmg_dir=inbound_dmg_dir,
        inbound_dmg_dist=inbound_dmg_dist,
        inbound_dmg_recency=inbound_dmg_recency,
        actual_ducked=actual_ducked,
        standing_blocked=standing_blocked,
        water_vertical_mode=water_vertical_mode,
        is_terminal=is_terminal,
        terminal_reason=terminal_reason,
        self_debug=self_debug,
        entity_debug=entity_debug,
        action_debug=action_debug,
    )


def pack_action(act: Action, tick: int) -> bytes:
    tick_value = int(tick)
    if tick_value != tick or not 0 <= tick_value <= 0xFFFFFFFF:
        raise ValueError("tick must be a uint32")
    try:
        vertical_raw = int(act.vertical_intent)
        if float(act.vertical_intent) != float(vertical_raw):
            raise ValueError
        vertical = VerticalIntent(vertical_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("vertical_intent must be one of the three VerticalIntent values") from error
    continuous = (
        float(act.move_forward), float(act.move_right),
        float(act.look_yaw), float(act.look_pitch),
    )
    if not np.isfinite(continuous).all():
        raise ValueError("action contains NaN or infinity")
    if not (-1.0 <= continuous[0] <= 1.0 and -1.0 <= continuous[1] <= 1.0):
        raise ValueError("movement action is outside [-1, 1]")
    if not (-45.0 <= continuous[2] <= 45.0 and -30.0 <= continuous[3] <= 30.0):
        raise ValueError("look action exceeds the per-decision bounds")
    hook = int(act.hook)
    weapon = int(act.weapon)
    fire = int(act.fire)
    if fire != act.fire or fire not in (0, 1):
        raise ValueError("fire must be a binary categorical value")
    if hook != act.hook or not 0 <= hook <= 3:
        raise ValueError("hook must be an integer in [0, 3]")
    if weapon != act.weapon or not 0 <= weapon <= 9:
        raise ValueError("weapon must be an integer in [0, 9]")
    return struct.pack(
        ACT_FMT,
        ML_ACT_MAGIC, tick_value,
        *continuous,
        int(vertical), fire, hook, weapon,
    )
