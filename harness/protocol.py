"""
protocol.py — Python-side serialization matching ml_bridge.h structs.

Both sides use plain C struct layout (little-endian, no padding beyond
natural alignment).  struct.calcsize must match sizeof() in C.
"""

import os
import struct
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
from dataclasses import dataclass, field
from typing import Optional

ML_BASE_PORT   = 27950
ML_OBS_MAGIC   = 0x514D4C4F   # "QMLO"
ML_ACT_MAGIC   = 0x514D4C41   # "QMLA"
ML_MAX_ENTITIES = 8
ML_RAY_COUNT    = 16
ML_HOOK_ZONES   = 4

ML_CONTROL_UNKNOWN    = 0
ML_CONTROL_HUMAN      = 1
ML_CONTROL_ML_BOT     = 2
ML_CONTROL_LEGACY_BOT = 3

ML_TERMINAL_NONE         = 0
ML_TERMINAL_DEATH        = 1
ML_TERMINAL_INTERMISSION = 2

# action_debug[11] flags emitted by game.so. These are debug/control-plane
# metadata and intentionally remain outside the policy observation vector.
ML_FIRE_GATE_PROTECTED  = 0x01
ML_FIRE_GATE_TARGET     = 0x02
ML_FIRE_GATE_SUPPRESSED = 0x04

# ── Observation ────────────────────────────────────────────────────────

# ml_self_t:  pos[3] vel[3] health armor weapon_id ammo  → 10 floats
_SELF_FMT = "10f"

# ml_entity_t: rel_pos[3] vel[3] health is_enemy visible → 9 floats
_ENT_FMT  = "9f"

# ml_ray_t: direction[3] distance → 4 floats
_RAY_FMT  = "4f"

# ml_hook_zone_t: anchor[3] landing[3] distance flags → 8 floats
_HOOK_FMT = "8f"

# ml_audio_t: sound_dir[3] sound_age alert_level → 5 floats
_AUDIO_FMT = "5f"

# ml_entity_debug_t: edict_index client_slot control_source flags → 4 uint32
_DEBUG_FMT = "4I"

# ml_action_debug_t: tick accepted timeout_count weapon, movement, buttons
_ACTION_DEBUG_FMT = "4I4f4I"

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
    + "4B"                                  # is_terminal, terminal_reason, 2 pad bytes
    + _DEBUG_FMT                            # self_debug
    + _DEBUG_FMT * ML_MAX_ENTITIES          # entity_debug[8]
    + _ACTION_DEBUG_FMT                     # action_debug
)

OBS_SIZE = struct.calcsize(OBS_FMT)

OBS_BASE_DIM = 185
# 21 session-memory features + 3 survivability-projection features
# (win_margin, effective-HP, my-DPS-share) so the bot perceives "will I win
# this race" directly, not just raw HP it has to re-derive.
OBS_SESSION_MEMORY_DIM = 24

# Input normalization so every obs feature lands ~[-1,1] instead of health
# (0-200) competing with position (0-4096) in the same linear layer. The bot
# couldn't cleanly perceive HP/ammo before this — and the survivability
# projection needs clean HP/armor/ammo + enemy-HP to be computable at all.
# self_state: [pos.xyz, vel.xyz, health, armor, weapon_id, ammo]
SELF_SCALE = np.array([4096, 4096, 4096, 1000, 1000, 1000,
                       200, 200, 40, 100], dtype=np.float32)
# entity: [rel_pos.xyz, vel.xyz, health, is_enemy(0/1), visible(0/1)]
ENT_SCALE = np.array([4096, 4096, 4096, 1000, 1000, 1000, 200, 1, 1],
                     dtype=np.float32)
# Extended observation block (rune_flags[5] + inbound_dmg_dir[3] + dist +
# recency = 10). Appended to the policy input only when Q2_EXT_OBS=1 (Run B);
# Run A leaves it off so the 206-dim checkpoint keeps resuming.
OBS_EXT_DIM = 10
EXT_OBS = os.environ.get("Q2_EXT_OBS", "0") == "1"
OBS_DIM = OBS_BASE_DIM + OBS_SESSION_MEMORY_DIM + (OBS_EXT_DIM if EXT_OBS else 0)

# ── Action ─────────────────────────────────────────────────────────────

ACT_FMT  = "<IIffff4B"          # magic tick fwd right yaw pitch jump fire hook weapon
ACT_SIZE = struct.calcsize(ACT_FMT)


@dataclass
class Observation:
    tick:           int
    bot_slot:       int
    yaw:            float
    pitch:          float

    # self state as flat numpy array [pos_xyz, vel_xyz, health, armor, weapon_id, ammo]
    self_state:     np.ndarray  # shape (10,)

    # visible entities as array [rel_pos_xyz, vel_xyz, health, is_enemy, visible] × N
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

    # extended observation block (policy input only when Q2_EXT_OBS)
    rune_flags:          np.ndarray  # shape (5,) resist,strength,haste,regen,vampire
    inbound_dmg_dir:     np.ndarray  # shape (3,) unit vector toward attacker
    inbound_dmg_dist:    float       # units, -1 if none recent
    inbound_dmg_recency: float       # 1=fresh → 0 by ~1s

    is_terminal:    bool
    terminal_reason: int

    # debug-only identity metadata; not included in policy input vector
    self_debug:     np.ndarray  # shape (4,) [edict_index, client_slot, source, flags]
    entity_debug:   np.ndarray  # shape (ML_MAX_ENTITIES, 4)
    action_debug:   np.ndarray  # shape (12,) engine-applied action echo

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

    def to_vector(self, session_memory: Optional[np.ndarray] = None) -> np.ndarray:
        """Flat observation vector for policy network input."""
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
        ]                                           # total: 185 dims
        if EXT_OBS:
            # Run B only: rune awareness + inbound-damage vector (10 dims).
            # Distance normalised by the same 1500u reference the engine uses.
            parts.append(self.rune_flags)                              # 5
            parts.append(self.inbound_dmg_dir)                         # 3
            dist_n = (self.inbound_dmg_dist / 1500.0
                      if self.inbound_dmg_dist >= 0.0 else -1.0)
            parts.append([dist_n, self.inbound_dmg_recency])           # 2
        base = np.concatenate(parts).astype(np.float32)

        if session_memory is None:
            memory = np.zeros(OBS_SESSION_MEMORY_DIM, dtype=np.float32)
        else:
            memory = np.asarray(session_memory, dtype=np.float32).reshape(-1)
            if memory.size < OBS_SESSION_MEMORY_DIM:
                memory = np.pad(
                    memory,
                    (0, OBS_SESSION_MEMORY_DIM - memory.size),
                    mode="constant",
                )
            elif memory.size > OBS_SESSION_MEMORY_DIM:
                memory = memory[:OBS_SESSION_MEMORY_DIM]

        return np.concatenate([base, memory]).astype(np.float32)

    BASE_OBS_DIM = OBS_BASE_DIM
    SESSION_MEMORY_DIM = OBS_SESSION_MEMORY_DIM
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
            visible = ent[8] > 0.5
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
    jump:         bool  = False
    fire:         bool  = False
    hook:         int   = 0     # 0=idle 1=fire 2=hold 3=release
    weapon:       int   = 0     # 0=no-change, 1-9=select


def parse_obs(data: bytes) -> Optional[Observation]:
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
    entities = np.array(ents_flat, dtype=np.float32).reshape(ML_MAX_ENTITIES, 9)

    rays_flat = v[i:i + ML_RAY_COUNT * 4]; i += ML_RAY_COUNT * 4
    rays = np.array(rays_flat, dtype=np.float32).reshape(ML_RAY_COUNT, 4)

    hooks_flat = v[i:i + ML_HOOK_ZONES * 8]; i += ML_HOOK_ZONES * 8
    hook_zone_count = v[i]; i += 1
    hook_zones = np.array(hooks_flat, dtype=np.float32).reshape(ML_HOOK_ZONES, 8)

    audio = np.array(v[i:i+5], dtype=np.float32); i += 5

    r_dd, r_dt, r_k, r_d, r_ip, r_ht = v[i:i+6]; i += 6
    r_prox, r_off, r_surv = v[i:i+3]; i += 3
    rune_flags = np.array(v[i:i+5], dtype=np.float32); i += 5
    inbound_dmg_dir = np.array(v[i:i+3], dtype=np.float32); i += 3
    inbound_dmg_dist, inbound_dmg_recency = v[i], v[i+1]; i += 2
    is_terminal = bool(v[i])
    terminal_reason = int(v[i+1])
    i += 4
    self_debug = np.array(v[i:i+4], dtype=np.uint32); i += 4
    entity_debug_flat = v[i:i + ML_MAX_ENTITIES * 4]; i += ML_MAX_ENTITIES * 4
    entity_debug = np.array(entity_debug_flat, dtype=np.uint32).reshape(ML_MAX_ENTITIES, 4)
    action_debug = np.array(v[i:i+12], dtype=np.float32); i += 12

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
        is_terminal=is_terminal,
        terminal_reason=terminal_reason,
        self_debug=self_debug,
        entity_debug=entity_debug,
        action_debug=action_debug,
    )


def pack_action(act: Action, tick: int) -> bytes:
    return struct.pack(
        ACT_FMT,
        ML_ACT_MAGIC, tick,
        float(act.move_forward), float(act.move_right),
        float(act.look_yaw),     float(act.look_pitch),
        int(act.jump), int(act.fire), int(act.hook), int(act.weapon),
    )
