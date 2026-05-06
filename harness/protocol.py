"""
protocol.py — Python-side serialization matching ml_bridge.h structs.

Both sides use plain C struct layout (little-endian, no padding beyond
natural alignment).  struct.calcsize must match sizeof() in C.
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

ML_BASE_PORT   = 27950
ML_OBS_MAGIC   = 0x514D4C4F   # "QMLO"
ML_ACT_MAGIC   = 0x514D4C41   # "QMLA"
ML_MAX_ENTITIES = 8
ML_RAY_COUNT    = 16
ML_HOOK_ZONES   = 4

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
    + "B3x"                                 # is_terminal + 3 pad bytes
)

OBS_SIZE = struct.calcsize(OBS_FMT)

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

    is_terminal:    bool

    @property
    def reward(self) -> float:
        """Composite reward signal."""
        return (
              self.reward_damage_dealt  * 0.003
            + self.reward_kill          * 1.0
            - self.reward_damage_taken  * 0.001
            - self.reward_death         * 0.5
            + self.reward_item_pickup   * 0.1
            + self.reward_hook_traversal * 0.2
        )

    def to_vector(self) -> np.ndarray:
        """Flat observation vector for policy network input."""
        return np.concatenate([
            self.self_state,                        # 10
            self.entities.flatten(),                # 72
            self.rays.flatten(),                    # 64
            self.hook_zones.flatten(),              # 32
            self.audio,                             # 5
            [self.yaw / 180.0, self.pitch / 90.0], # 2 — normalised facing
        ]).astype(np.float32)                       # total: 185 dims

    OBS_DIM = 185


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
    is_terminal = bool(v[i]); i += 1

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
        is_terminal=is_terminal,
    )


def pack_action(act: Action, tick: int) -> bytes:
    return struct.pack(
        ACT_FMT,
        ML_ACT_MAGIC, tick,
        float(act.move_forward), float(act.move_right),
        float(act.look_yaw),     float(act.look_pitch),
        int(act.jump), int(act.fire), int(act.hook), int(act.weapon),
    )
