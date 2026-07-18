"""Frozen trainer-side contract for the first multires Atlas policy.

This module deliberately does not import :mod:`harness.protocol`.  The B4
protocol batch owns the public wire/POD definitions; B5 owns the policy-vector
meaning after a B4 packet has been admitted.  Keeping the boundary explicit
lets the B5 model and tests exist without silently blessing the legacy
209/219-float wire generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


POLICY_GENERATION = "multires-atlas-policy-v1"
FEATURE_SCHEMA = "multires-atlas-features-298-v1"

FACTUAL_DIM = 198
DYN_DIM = 24
RECOVERY_DIM = 16
GUIDE_CANDIDATES = 4
GUIDE_CANDIDATE_DIM = 15
GUIDE_DIM = GUIDE_CANDIDATES * GUIDE_CANDIDATE_DIM
OBS_DIM = FACTUAL_DIM + DYN_DIM + RECOVERY_DIM + GUIDE_DIM

ACTION_DIM = 8
POSTURE_CLASSES = 3
POSTURE_CROUCH_OR_DOWN = 0
POSTURE_NEUTRAL = 1
POSTURE_JUMP_OR_UP = 2
HOOK_CLASSES = 4
WEAPON_CLASSES = 10


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    start: int
    width: int

    @property
    def stop(self) -> int:
        return self.start + self.width

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


# The first 195 factual values preserve the existing always-on factual order;
# the B4 packet adds the three explicit posture/water facts at the end.
SELF = FeatureBlock("self", 0, 10)
ENTITIES = FeatureBlock("entities", SELF.stop, 8 * 9)
RAYS = FeatureBlock("rays", ENTITIES.stop, 16 * 4)
HOOK_ZONES = FeatureBlock("hook_zones", RAYS.stop, 4 * 8)
AUDIO = FeatureBlock("audio", HOOK_ZONES.stop, 5)
VIEW = FeatureBlock("view", AUDIO.stop, 2)
RUNES = FeatureBlock("runes", VIEW.stop, 5)
INBOUND_DAMAGE = FeatureBlock("inbound_damage", RUNES.stop, 5)
POSTURE_FACTS = FeatureBlock("posture_facts", INBOUND_DAMAGE.stop, 3)
DYN = FeatureBlock("dyn", FACTUAL_DIM, DYN_DIM)
RECOVERY = FeatureBlock("recovery", DYN.stop, RECOVERY_DIM)
GUIDES = FeatureBlock("guides", RECOVERY.stop, GUIDE_DIM)

FACTUAL_BLOCKS = (
    SELF,
    ENTITIES,
    RAYS,
    HOOK_ZONES,
    AUDIO,
    VIEW,
    RUNES,
    INBOUND_DAMAGE,
    POSTURE_FACTS,
)
ALL_BLOCKS = FACTUAL_BLOCKS + (DYN, RECOVERY, GUIDES)

POSTURE_FACT_NAMES = (
    "actual_ducked",
    "standing_blocked",
    "water_vertical_mode",
)
DYN_NAMES = (
    "dyn_l2_current_engagement",
    "dyn_l2_current_threat",
    "dyn_l2_current_opportunity",
    "dyn_l2_current_self_fire",
    "dyn_l2_current_confidence",
    "dyn_immediate_thermal_forward",
    "dyn_immediate_thermal_quake_right",
    "dyn_immediate_thermal_up",
    "dyn_immediate_thermal_heat",
    "dyn_combat_threat_forward",
    "dyn_combat_threat_quake_right",
    "dyn_combat_threat_up",
    "dyn_combat_threat_score",
    "dyn_opportunity_forward",
    "dyn_opportunity_quake_right",
    "dyn_opportunity_up",
    "dyn_opportunity_score",
    "dyn_self_fire_forward",
    "dyn_self_fire_quake_right",
    "dyn_self_fire_up",
    "dyn_self_fire_score",
    "dyn_win_margin",
    "dyn_effective_health_norm",
    "dyn_own_dps_share",
)
RECOVERY_NAMES = (
    "recovery_hazard_lava",
    "recovery_hazard_slime",
    "recovery_hazard_hurt",
    "recovery_hazard_void_or_lethal_drop",
    "recovery_hazard_crush_or_current",
    "recovery_hazard_strength",
    "recovery_hull_clearance",
    "recovery_cost_to_safety",
    "recovery_confidence",
    "recovery_primary_forward",
    "recovery_primary_quake_right",
    "recovery_primary_up",
    "recovery_alternate_forward",
    "recovery_alternate_quake_right",
    "recovery_alternate_up",
    "recovery_time_to_impact",
)
GUIDE_CLASS_NAMES = (
    "weapon",
    "ammunition",
    "health",
    "armor",
    "powerup",
    "rune",
    "control",
    "spawn_egress",
)
GUIDE_FIELD_NAMES = (
    "forward",
    "quake_right",
    "up",
    "cost",
    "risk",
    "confidence",
    "availability_belief",
) + tuple(f"class_{name}" for name in GUIDE_CLASS_NAMES)


SELF_FEATURE_NAMES = (
    "self_pos_x", "self_pos_y", "self_pos_z",
    "self_vel_x", "self_vel_y", "self_vel_z",
    "self_health", "self_armor", "self_weapon_id", "self_ammo",
)
ENTITY_FIELDS = (
    "aim_forward", "aim_quake_right", "aim_up",
    "relative_velocity_forward", "relative_velocity_quake_right",
    "relative_velocity_up", "health", "is_enemy", "signed_exposure",
)
RAY_FIELDS = ("dir_x", "dir_y", "dir_z", "distance")
HOOK_FIELDS = (
    "anchor_x", "anchor_y", "anchor_z", "landing_x", "landing_y",
    "landing_z", "distance", "flags",
)
AUDIO_FEATURE_NAMES = (
    "audio_direction_x", "audio_direction_y", "audio_direction_z",
    "audio_age", "audio_alert_level",
)

FEATURE_NAMES = (
    SELF_FEATURE_NAMES
    + tuple(
        f"entity_{slot}_{field}"
        for slot in range(8)
        for field in ENTITY_FIELDS
    )
    + tuple(
        f"ray_{ray}_{field}"
        for ray in range(16)
        for field in RAY_FIELDS
    )
    + tuple(
        f"hook_zone_{zone}_{field}"
        for zone in range(4)
        for field in HOOK_FIELDS
    )
    + AUDIO_FEATURE_NAMES
    + ("view_yaw", "view_pitch")
    + ("rune_resist", "rune_strength", "rune_haste", "rune_regen", "rune_vampire")
    + (
        "inbound_damage_direction_x",
        "inbound_damage_direction_y",
        "inbound_damage_direction_z",
        "inbound_damage_distance",
        "inbound_damage_recency",
    )
    + POSTURE_FACT_NAMES
    + DYN_NAMES
    + RECOVERY_NAMES
    + tuple(
        f"guide_{candidate}_{field}"
        for candidate in range(GUIDE_CANDIDATES)
        for field in GUIDE_FIELD_NAMES
    )
)

if len(FEATURE_NAMES) != OBS_DIM or len(set(FEATURE_NAMES)) != OBS_DIM:
    raise RuntimeError("multires feature schema is not an exact unique 298-vector")
if tuple(block.start for block in ALL_BLOCKS) != tuple(
    0 if index == 0 else ALL_BLOCKS[index - 1].stop
    for index in range(len(ALL_BLOCKS))
):
    raise RuntimeError("multires feature blocks are not contiguous")
if ALL_BLOCKS[-1].stop != OBS_DIM:
    raise RuntimeError("multires feature blocks do not end at OBS_DIM")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(
    FEATURE_NAMES,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")).hexdigest()


def _exact_vector(value: Sequence[float], width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != width:
        raise ValueError(f"{name} must contain exactly {width} floats")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def pack_policy_vector(
    factual: Sequence[float],
    dyn: Sequence[float],
    recovery: Sequence[float],
    guides: Sequence[float],
) -> np.ndarray:
    """Join four B4/B3 blocks without accepting legacy-width fallbacks."""
    vector = np.concatenate((
        _exact_vector(factual, FACTUAL_DIM, "factual"),
        _exact_vector(dyn, DYN_DIM, "dyn"),
        _exact_vector(recovery, RECOVERY_DIM, "recovery"),
        _exact_vector(guides, GUIDE_DIM, "guides"),
    )).astype(np.float32, copy=False)
    if vector.size != OBS_DIM:
        raise AssertionError("internal multires feature packing error")
    return vector


@dataclass(frozen=True)
class GuideDropoutIdentity:
    map_name: str
    policy_version: int
    client_id: str
    tick: int

    def validate(self) -> None:
        if not self.map_name or not self.client_id:
            raise ValueError("guide dropout identity requires map_name and client_id")
        if self.policy_version < 0 or self.tick < 0:
            raise ValueError("guide dropout policy_version/tick must be nonnegative")


@dataclass(frozen=True)
class GuideDropoutConfig:
    global_probability: float = 0.10
    class_probabilities: tuple[float, ...] = (0.0,) * len(GUIDE_CLASS_NAMES)
    tick_bucket_size: int = 10

    def validate(self) -> None:
        probabilities = (self.global_probability,) + tuple(self.class_probabilities)
        if len(self.class_probabilities) != len(GUIDE_CLASS_NAMES):
            raise ValueError("guide dropout requires eight class probabilities")
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("guide dropout probabilities must be within [0, 1]")
        if self.tick_bucket_size <= 0:
            raise ValueError("guide dropout tick_bucket_size must be positive")


@dataclass(frozen=True)
class GuideDropoutResult:
    guides: np.ndarray
    dropped_candidates: tuple[bool, ...]
    global_drop: bool
    candidate_classes: tuple[Optional[int], ...]


def _dropout_sample(seed_material: Mapping[str, object], label: str) -> float:
    digest = hashlib.sha256(
        _canonical_json({"identity": seed_material, "label": label})
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def apply_seeded_guide_dropout(
    guides: Sequence[float],
    identity: GuideDropoutIdentity,
    config: GuideDropoutConfig = GuideDropoutConfig(),
) -> GuideDropoutResult:
    """Deterministically drop advisory guides by causal rollout identity.

    The policy cannot infer worker scheduling from dropout because the seed is
    bound only to map, policy, client, and a frozen tick bucket.  Factual, Dyn,
    and recovery blocks never pass through this function.
    """
    identity.validate()
    config.validate()
    source = _exact_vector(guides, GUIDE_DIM, "guides")
    result = source.copy().reshape(GUIDE_CANDIDATES, GUIDE_CANDIDATE_DIM)
    material = {
        "schema": FEATURE_SCHEMA,
        "map": identity.map_name,
        "policy": int(identity.policy_version),
        "client": identity.client_id,
        "tick_bucket": int(identity.tick) // int(config.tick_bucket_size),
    }
    global_drop = _dropout_sample(material, "global") < config.global_probability
    dropped: list[bool] = []
    classes: list[Optional[int]] = []
    for candidate in range(GUIDE_CANDIDATES):
        row = result[candidate]
        class_bits = row[7:15]
        class_index = int(np.argmax(class_bits)) if float(class_bits.max()) > 0.0 else -1
        classes.append(class_index if class_index >= 0 else None)
        class_probability = (
            config.class_probabilities[class_index] if class_index >= 0 else 0.0
        )
        class_drop = _dropout_sample(
            material, f"candidate-{candidate}-class-{class_index}"
        ) < class_probability
        should_drop = bool(global_drop or class_drop)
        if should_drop:
            row.fill(0.0)
        dropped.append(should_drop)
    return GuideDropoutResult(
        guides=result.reshape(-1),
        dropped_candidates=tuple(dropped),
        global_drop=bool(global_drop),
        candidate_classes=tuple(classes),
    )
