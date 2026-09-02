"""
spatial.py - training-side voxel/spatial reward shaping.

This does not change the C wire protocol. It derives lightweight tactical
signals from the existing observation: bot position, visible enemies, and
hook-zone annotations. Engine-side occupancy voxels can replace or extend this
later without invalidating the current bridge.
"""

import gzip
import json
import logging
import os
import random
from dataclasses import dataclass, field, fields
from math import atan2, degrees, floor, hypot, log1p
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .protocol import (
    ENVIRONMENTAL_MODS,
    ML_ENTITY_DUCKED,
    ML_ENTITY_EPOCH_MASK,
    ML_ENTITY_EPOCH_SHIFT,
    ML_HIT_STREAK_MASK,
    ML_HIT_STREAK_SHIFT,
    OBS_SESSION_MEMORY_DIM,
    Observation,
)
from .item_timing import ItemTimingTable
from tools.map_bundle import (
    farm_map_requires_attestation,
    installed_manifest_name,
    sha256_bytes,
    validate_manifest,
    verify_installed_artifact,
)

log = logging.getLogger(__name__)

HOOK_REQUIRED = 4
BLASTER_ITEM_INDEX = 8
# Fixed policy-readout contract: immediate hostile thermal direction/heat.
# Persistent engagement is the fallback producer for the same tactical signal.
IMMEDIATE_ENGAGEMENT_SLICE = slice(5, 9)

# Columns mirrored into the fast nearest-signal numpy cache. Order is fixed;
# _CELL_COL_INDEX maps field name -> column. Keep in sync with
# SessionMemoryCell's scored fields.
_CELL_COL_FIELDS = (
    "engagement_count",
    "enemy_seen",
    "enemy_lost",
    "damage_dealt",
    "damage_taken",
    "kills",
    "deaths",
    "hazard_damage",
    "hazard_deaths",
    "bad_fight_taken",
    "good_disengage",
    "self_fire",
    "hook_engagement",
    "item_contested",
    "successful_escape",
    "prior_opportunity",
    "prior_threat",
    "readiness",
    "route_bias",
)
_CELL_COL_INDEX = {name: index for index, name in enumerate(_CELL_COL_FIELDS)}


@dataclass
class SessionMemoryCell:
    """Non-decaying per-session tactical marks for one voxel cell."""

    engagement_count: float = 0.0
    enemy_seen: float = 0.0
    enemy_lost: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    kills: float = 0.0
    deaths: float = 0.0
    # wire v5: environmental damage/deaths (MOD water/slime/lava/crush/
    # falling) are deposited here instead of the combat channels, so threat
    # scoring stops blaming enemies for what the map did (de-conflation).
    hazard_damage: float = 0.0
    hazard_deaths: float = 0.0
    # Decision-quality memory (fight bias): fights taken at strongly
    # negative bias that ended in death, and rewarded disengages from bad
    # fights. Typed channels — never folded into combat counters.
    bad_fight_taken: float = 0.0
    good_disengage: float = 0.0
    self_fire: float = 0.0
    hook_engagement: float = 0.0
    item_contested: float = 0.0
    successful_escape: float = 0.0
    # Generator-known priors and live route/item timing overlays. Priors are
    # persisted; the dynamic fields are rebuilt from the route clock each tick.
    prior_opportunity: float = 0.0
    prior_threat: float = 0.0
    readiness: float = 0.0
    route_bias: float = 0.0
    last_tick: int = 0

    @property
    def samples(self) -> float:
        return (
            self.engagement_count
            + self.enemy_seen
            + self.enemy_lost
            + self.kills
            + self.deaths
            + self.hazard_deaths
            + self.hazard_damage / 100.0
            + 0.5 * self.bad_fight_taken
            + 0.5 * self.good_disengage
            + self.self_fire
            + self.hook_engagement
            + self.item_contested
            + self.successful_escape
            + abs(self.prior_opportunity)
            + abs(self.prior_threat)
        )


@dataclass
class ThermalTargetTrack:
    """Per-client, non-persistent hostile heat with an exact subvoxel point."""

    target_id: int
    world_point: np.ndarray
    world_velocity: np.ndarray
    exposure: float
    last_tick: int


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class VoxelSpatialReward:
    """Small voxel-based curriculum reward computed per episode."""

    enabled: bool = True
    voxel_size: float = 256.0
    new_cell_reward: float = 0.02
    stagnation_penalty: float = 0.002
    stagnation_steps: int = 30
    engagement_reward: float = 0.01
    engagement_min: float = 192.0
    engagement_max: float = 1400.0
    combat_navigation_suppression: float = 0.85
    aim_alignment_reward: float = 0.02
    aim_tracking_reward: float = 0.006
    aim_yaw_deg: float = 12.0
    aim_pitch_deg: float = 14.0
    # Legacy hook-use rewards are retained as zero-valued constructor/config
    # compatibility fields only.  Grapple is a high-energy movement actuator,
    # not a rate objective; the correction fields below are its only positive
    # shaping.
    hook_required_reward: float = 0.0
    hook_required_distance: float = 512.0
    hook_cost: float = 0.004
    hook_enemy_reward: float = 0.0
    hook_no_ammo_reward: float = 0.0
    hook_blind_penalty: float = 0.008
    hook_noop_penalty: float = 0.012
    hook_release_overspeed_reward: float = 0.0
    hook_release_idle_penalty: float = 0.002
    hook_correction_progress_reward: float = 0.020
    hook_correction_complete_reward: float = 0.040
    hook_correction_min_heat: float = 0.05
    hook_correction_min_advance: float = 32.0
    hook_correction_progress_epsilon: float = 4.0
    hook_correction_arrival_radius: float = 96.0
    hook_correction_max_anchor_distance: float = 700.0
    hook_correction_timeout_ticks: int = 40
    hook_melee_distance: float = 768.0
    hook_yaw_deg: float = 18.0
    hook_pitch_deg: float = 22.0
    hook_ammo_threshold: float = 0.5
    nominal_speed_min: float = 220.0
    nominal_speed_max: float = 360.0
    movement_intent_min: float = 0.55
    nominal_speed_reward: float = 0.008
    backward_movement_penalty: float = 0.020
    slow_movement_penalty: float = 0.012
    overspeed_penalty: float = 0.008
    horizon_pitch_limit: float = 15.0
    horizon_pitch_penalty: float = 0.006
    level_aim_movement_reward: float = 0.012
    level_aim_min_speed: float = 96.0
    jump_cost: float = 0.006
    slow_jump_penalty: float = 0.014
    hook_overspeed_penalty: float = 0.012
    fire_cost: float = 0.002
    fire_unseen_penalty: float = 0.025
    fire_unaligned_penalty: float = 0.018
    fire_no_ammo_penalty: float = 0.02
    fire_aligned_reward: float = 0.006
    splash_fire_reward: float = 0.004
    splash_yaw_deg: float = 30.0
    splash_pitch_deg: float = 28.0
    splash_max_distance: float = 1200.0
    audio_fire_alert_min: float = 0.2
    audio_fire_max_age: float = 30.0
    session_memory_enabled: bool = True
    # Role gates (design: one store, per-role channels, consumption-point
    # gates). Storage (deposits) is governed by session_memory_enabled and
    # stays on during ablations so runs still collect scoring data. The three
    # consumption points gate independently: policy input (obs tail), reward
    # shaping, and the trainer's direction loss (lattice_direction_coef).
    lattice_obs_enabled: bool = True
    lattice_reward_enabled: bool = True
    # When off, the immediate-engagement obs slice (5:9) is zeroed regardless
    # of producer (thermal-hot or persistent fallback) — the clean ablation
    # switch for that channel.
    immediate_engagement_enabled: bool = True
    # External coach writes (apply_directive) bypass gameplay validation, so
    # they are gated and logged like sidecar attestation.
    directives_enabled: bool = True
    session_memory_search_radius: float = 2048.0
    session_memory_score_scale: float = 8.0
    session_memory_limit: int = 4096
    # Fast nearest-signal path: vectorized numpy filter over a dirty-tracked
    # column cache, with exact Python recomputation of finalist cells so the
    # result is bit-identical to the pure-Python traversal. Q2_FAST_NEAREST=0
    # restores the traversal; Q2_NEAREST_VERIFY=1 cross-checks every call.
    fast_nearest_enabled: bool = True
    nearest_verify: bool = False
    session_memory_engagement_reward: float = 0.004
    session_memory_opportunity_reward: float = 0.006
    session_memory_threat_penalty: float = 0.006
    session_memory_death_aversion: float = 0.010
    # Hazard (environmental MOD) aversion, patterned on death aversion but
    # fed by the typed hazard channel — lava/crush/fall memory repels
    # without polluting combat threat.
    session_memory_hazard_aversion: float = 0.010
    session_memory_self_fire_penalty: float = 0.012
    session_memory_camp_penalty: float = 0.004
    thermal_target_enabled: bool = True
    thermal_target_voxel_size: float = 64.0
    thermal_target_decay: float = 0.75
    thermal_target_max_age_ticks: int = 5
    rust_lattice_enabled: bool = False
    lattice_preload_enabled: bool = True
    lattice_routes_enabled: bool = True
    lattice_prior_scale: float = 8.0
    lattice_route_scale: float = 4.0
    lattice_tick_rate: float = 10.0
    lattice_route_refresh_ticks: int = 5
    # Engine-supplied extended channels (both runs). prox = how-hard-hit
    # aversion; offense/survival = rune-conditioned payoffs.
    damage_prox_aversion: float = 0.004
    offense_rune_reward: float = 0.004
    survival_rune_reward: float = 0.004
    # wire v5 self-exposure shaping: pay a penalty while a visible enemy can
    # see me, pay a covered bonus when I was recently damaged, no enemy is
    # visible, and enemies can no longer see me.
    exposure_penalty: float = 0.004
    cover_reward: float = 0.002
    # Fight decision bias (advisory; reward/deposit path only — it never
    # masks or alters actions). Weighted sum of exchange projection,
    # exposure asymmetry, typed-hazard proximity, outnumbered penalty, and
    # escape readiness, clipped to [-1, 1].
    bias_w_margin: float = 0.5
    bias_w_exposure: float = 0.3
    bias_w_hazard: float = 0.4
    bias_w_outnumbered: float = 0.15
    bias_w_escape: float = 0.1
    bad_fight_penalty: float = 0.010
    disengage_reward: float = 0.006
    good_fight_reward: float = 0.004
    # Decision memory feeds the EXISTING threat read channel only (the
    # 24-float policy tail layout is frozen): cells where I repeatedly took
    # bad fights read as slightly more threatening.
    bad_fight_threat_weight: float = 0.5
    survival_tick_reward: float = 0.0002
    survival_threat_reward: float = 0.0015
    survival_low_health_reward: float = 0.002
    threat_engagement_reward: float = 0.030
    threat_aim_reward: float = 0.040
    threat_fire_reward: float = 0.035
    threat_damage_reward: float = 0.010
    threat_kill_reward: float = 2.000
    threat_ignore_penalty: float = 0.030
    threat_unready_penalty: float = 0.010
    damage_advantage_reward: float = 0.004      # legacy (per-step, now unused)
    damage_disadvantage_penalty: float = 0.003  # legacy (per-step, now unused)
    # Damage-as-lens exchange model (replaces raw damage_margin reward).
    exchange_quality_reward: float = 0.30   # episode: fraction-of-exchange-won ∈[-1,1]
    exchange_press_reward: float = 0.020    # win the trade / chicken: they yield, I take it
    exchange_break_reward: float = 0.010    # disengage from a losing/even trade (the swerve)
    exchange_crash_penalty: float = 0.015   # both commit to an even trade (mutual crash)
    exchange_feed_penalty: float = 0.020    # keep feeding a trade I'm already losing
    exchange_jitter: float = 0.30           # per-step roll on the even→press boundary
    exchange_aggression_mag: float = 0.60   # per-life aggression draw amplitude
    exchange_dps_decay: float = 0.90        # rolling DPS EMA
    # Contextual rune switching: reward acquiring the rune that fits the
    # current health need, scaled by the fit GAIN over the rune you dropped.
    rune_switch_reward: float = 0.05
    rune_low_health: float = 50.0           # below → need survival runes
    rune_high_health: float = 100.0         # above → need offense runes
    frag_advantage_reward: float = 0.20
    frag_disadvantage_penalty: float = 0.35
    episode_win_reward: float = 1.50
    episode_survival_reward: float = 0.50
    episode_loss_penalty: float = 1.00
    episode_idle_penalty: float = 0.35
    win_damage_margin: float = 60.0
    meaningful_contact_min: float = 8.0
    low_health_threshold: float = 35.0
    recent_threat_window: int = 90

    map_name: str = ""
    visited: Set[Tuple[int, int, int]] = field(default_factory=set)
    last_cell: Optional[Tuple[int, int, int]] = None
    steps_in_cell: int = 0
    pos_history: list = field(default_factory=list, init=False)
    session_memories: Dict[
        str, Dict[Tuple[int, int, int], SessionMemoryCell]
    ] = field(default_factory=dict, init=False)
    route_graphs: Dict[str, dict] = field(default_factory=dict, init=False)
    item_timings: Dict[str, ItemTimingTable] = field(default_factory=dict, init=False)
    preloaded_maps: Set[str] = field(default_factory=set, init=False)
    dynamic_cells: Dict[str, Set[Tuple[int, int, int]]] = field(
        default_factory=dict, init=False
    )
    sidecar_sources: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False)
    selected_route: str = ""
    _map_last_tick: Dict[str, int] = field(default_factory=dict, init=False)
    _route_heat_last_tick: Dict[str, int] = field(default_factory=dict, init=False)
    _route_heat_last_axis: Dict[str, str] = field(default_factory=dict, init=False)
    _feature_cache_key: Optional[Tuple[str, int]] = field(default=None, init=False)
    _feature_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _feature_nearest_deaths: float = field(default=0.0, init=False)
    _feature_nearest_hazard: float = field(default=0.0, init=False)
    _rust_indices: Dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _rust_dirty_cells: Dict[str, Set[Tuple[int, int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _rust_removed_cells: Dict[str, Set[Tuple[int, int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _rust_score_events: Dict[
        str, Dict[Tuple[int, int, int], np.ndarray]
    ] = field(default_factory=dict, init=False, repr=False)
    _rust_fallback_reason: str = field(default="", init=False)
    _rust_event_rows_applied: int = field(default=0, init=False)
    # Column cache for the fast nearest-signal path (per map).
    _cell_col_cache: Dict[str, dict] = field(
        default_factory=dict, init=False, repr=False
    )
    _cell_col_dirty: Dict[str, Dict[Tuple[int, int, int], None]] = field(
        default_factory=dict, init=False, repr=False
    )
    _thermal_tracks: Dict[int, ThermalTargetTrack] = field(
        default_factory=dict, init=False, repr=False
    )
    _thermal_last_tick: int = field(default=-1, init=False, repr=False)
    _thermal_last_signal: Tuple[float, float, float, float, float] = field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0), init=False, repr=False
    )
    _hook_correction_target: Optional[Tuple[float, float, float]] = field(
        default=None, init=False, repr=False
    )
    _hook_correction_anchor: Optional[Tuple[float, float, float]] = field(
        default=None, init=False, repr=False
    )
    _hook_correction_hot_target: Optional[Tuple[float, float, float]] = field(
        default=None, init=False, repr=False
    )
    _hook_correction_heat: float = field(default=0.0, init=False)
    _hook_correction_initial_distance: float = field(default=0.0, init=False)
    _hook_correction_best_distance: float = field(default=0.0, init=False)
    _hook_correction_started_tick: int = field(default=-1, init=False)
    _hook_correction_escape: bool = field(default=False, init=False)
    last_visible_count: int = 0
    last_damage_tick: int = -1000000
    last_damage_cell: Optional[Tuple[int, int, int]] = None
    # exchange runtime state (rolling DPS EMA + per-life aggression temperament)
    _dps_self: float = 0.0
    _dps_enemy: float = 0.0
    aggression: float = 0.0
    _prev_rune_idx: int = -1   # rune held last step (for switch detection)
    episode_steps: int = 0
    episode_damage_dealt: float = 0.0
    episode_damage_taken: float = 0.0
    episode_kills: float = 0.0
    episode_deaths: float = 0.0
    episode_bad_fights: float = 0.0
    episode_good_disengages: float = 0.0
    episode_contact_events: float = 0.0
    recent_threat_steps: int = 0
    rng: random.Random = field(default_factory=random.Random, repr=False)

    @classmethod
    def from_env(cls, seed: Optional[int] = None) -> "VoxelSpatialReward":
        def _flag(name: str, default: str = "1") -> bool:
            return os.environ.get(name, default).lower() not in {
                "0", "false", "off", "no"
            }

        enabled = _flag("Q2_SPATIAL_REWARD")
        # Storage gate first: the consumption gates default to following it
        # (preserving the historical Q2_SESSION_MEMORY=0 semantics) but can be
        # overridden independently for role ablations.
        session_memory_enabled = _flag("Q2_SESSION_MEMORY")
        sm_default = "1" if session_memory_enabled else "0"
        return cls(
            enabled=enabled,
            voxel_size=_env_float("Q2_VOXEL_SIZE", 256.0),
            new_cell_reward=_env_float("R_VOXEL_NEW_CELL", 0.02),
            stagnation_penalty=_env_float("R_VOXEL_STAGNATION", 0.002),
            stagnation_steps=_env_int("Q2_VOXEL_STAGNATION_STEPS", 30),
            engagement_reward=_env_float("R_TACTICAL_ENGAGEMENT", 0.01),
            engagement_min=_env_float("Q2_ENGAGEMENT_MIN", 192.0),
            engagement_max=_env_float("Q2_ENGAGEMENT_MAX", 1400.0),
            combat_navigation_suppression=_env_float(
                "Q2_COMBAT_NAV_SUPPRESSION", 0.85
            ),
            aim_alignment_reward=_env_float("R_AIM_ALIGNMENT", 0.02),
            aim_tracking_reward=_env_float("R_AIM_TRACKING", 0.006),
            aim_yaw_deg=_env_float("Q2_AIM_YAW_DEG", 12.0),
            aim_pitch_deg=_env_float("Q2_AIM_PITCH_DEG", 14.0),
            # Do not honor the old positive hook-use knobs.  A stale runtime
            # environment must not silently restore hook-rate farming.
            hook_required_reward=0.0,
            hook_required_distance=_env_float("Q2_HOOK_REQUIRED_DISTANCE", 512.0),
            hook_cost=_env_float("R_HOOK_COST", 0.004),
            hook_enemy_reward=0.0,
            hook_no_ammo_reward=0.0,
            hook_blind_penalty=_env_float("R_HOOK_BLIND_PENALTY", 0.008),
            hook_noop_penalty=_env_float("R_HOOK_NOOP", 0.012),
            hook_release_overspeed_reward=0.0,
            hook_release_idle_penalty=_env_float("R_HOOK_RELEASE_IDLE", 0.002),
            hook_correction_progress_reward=_env_float(
                "R_HOOK_CORRECTION_PROGRESS", 0.020
            ),
            hook_correction_complete_reward=_env_float(
                "R_HOOK_CORRECTION_COMPLETE", 0.040
            ),
            hook_correction_min_heat=_env_float(
                "Q2_HOOK_CORRECTION_MIN_HEAT", 0.05
            ),
            hook_correction_min_advance=_env_float(
                "Q2_HOOK_CORRECTION_MIN_ADVANCE", 32.0
            ),
            hook_correction_progress_epsilon=_env_float(
                "Q2_HOOK_CORRECTION_PROGRESS_EPSILON", 4.0
            ),
            hook_correction_arrival_radius=_env_float(
                "Q2_HOOK_CORRECTION_ARRIVAL_RADIUS", 96.0
            ),
            hook_correction_max_anchor_distance=_env_float(
                "Q2_HOOK_CORRECTION_MAX_ANCHOR_DISTANCE", 700.0
            ),
            hook_correction_timeout_ticks=_env_int(
                "Q2_HOOK_CORRECTION_TIMEOUT_TICKS", 40
            ),
            hook_melee_distance=_env_float("Q2_HOOK_MELEE_DISTANCE", 768.0),
            hook_yaw_deg=_env_float("Q2_HOOK_YAW_DEG", 18.0),
            hook_pitch_deg=_env_float("Q2_HOOK_PITCH_DEG", 22.0),
            hook_ammo_threshold=_env_float("Q2_HOOK_AMMO_THRESHOLD", 0.5),
            nominal_speed_min=_env_float("Q2_NOMINAL_SPEED_MIN", 220.0),
            nominal_speed_max=_env_float("Q2_NOMINAL_SPEED_MAX", 360.0),
            movement_intent_min=_env_float("Q2_MOVEMENT_INTENT_MIN", 0.55),
            nominal_speed_reward=_env_float("R_MOVE_NOMINAL", 0.008),
            backward_movement_penalty=_env_float("R_MOVE_BACKWARD", 0.020),
            slow_movement_penalty=_env_float("R_MOVE_SLOW", 0.012),
            overspeed_penalty=_env_float("R_MOVE_OVERSPEED", 0.008),
            horizon_pitch_limit=_env_float("Q2_HORIZON_PITCH_LIMIT", 15.0),
            horizon_pitch_penalty=_env_float("R_HORIZON_PITCH", 0.006),
            level_aim_movement_reward=_env_float("R_MOVE_LEVEL_AIM", 0.012),
            level_aim_min_speed=_env_float("Q2_LEVEL_AIM_MIN_SPEED", 96.0),
            jump_cost=_env_float("R_JUMP_COST", 0.006),
            slow_jump_penalty=_env_float("R_JUMP_SLOW", 0.014),
            hook_overspeed_penalty=_env_float("R_HOOK_OVERSPEED", 0.012),
            fire_cost=_env_float("R_FIRE_COST", 0.002),
            fire_unseen_penalty=_env_float("R_FIRE_UNSEEN_PENALTY", 0.025),
            fire_unaligned_penalty=_env_float("R_FIRE_UNALIGNED_PENALTY", 0.018),
            fire_no_ammo_penalty=_env_float("R_FIRE_NO_AMMO_PENALTY", 0.02),
            fire_aligned_reward=_env_float("R_FIRE_ALIGNED_REWARD", 0.006),
            splash_fire_reward=_env_float("R_SPLASH_FIRE_REWARD", 0.004),
            splash_yaw_deg=_env_float("Q2_SPLASH_YAW_DEG", 30.0),
            splash_pitch_deg=_env_float("Q2_SPLASH_PITCH_DEG", 28.0),
            splash_max_distance=_env_float("Q2_SPLASH_MAX_DISTANCE", 1200.0),
            audio_fire_alert_min=_env_float("Q2_FIRE_AUDIO_ALERT_MIN", 0.2),
            audio_fire_max_age=_env_float("Q2_FIRE_AUDIO_MAX_AGE", 30.0),
            session_memory_enabled=session_memory_enabled,
            lattice_obs_enabled=_flag("Q2_LATTICE_OBS", sm_default),
            lattice_reward_enabled=_flag("Q2_LATTICE_REWARD", sm_default),
            immediate_engagement_enabled=_flag("Q2_LATTICE_IMMEDIATE"),
            directives_enabled=_flag("Q2_LATTICE_DIRECTIVES"),
            session_memory_search_radius=_env_float(
                "Q2_SESSION_MEMORY_RADIUS", 2048.0
            ),
            session_memory_score_scale=_env_float(
                "Q2_SESSION_MEMORY_SCORE_SCALE", 8.0
            ),
            session_memory_limit=_env_int("Q2_SESSION_MEMORY_LIMIT", 4096),
            fast_nearest_enabled=os.environ.get(
                "Q2_FAST_NEAREST", "1"
            ).lower() not in {"0", "false", "off", "no"},
            nearest_verify=os.environ.get(
                "Q2_NEAREST_VERIFY", "0"
            ).lower() not in {"0", "false", "off", "no"},
            session_memory_engagement_reward=_env_float(
                "R_SESSION_MEMORY_ENGAGEMENT", 0.004
            ),
            session_memory_opportunity_reward=_env_float(
                "R_SESSION_MEMORY_OPPORTUNITY", 0.006
            ),
            session_memory_threat_penalty=_env_float(
                "R_SESSION_MEMORY_THREAT", 0.006
            ),
            session_memory_death_aversion=_env_float(
                "R_SESSION_MEMORY_DEATH_AVERSION", 0.010
            ),
            session_memory_hazard_aversion=_env_float(
                "R_SESSION_MEMORY_HAZARD", 0.010
            ),
            session_memory_self_fire_penalty=_env_float(
                "R_SESSION_MEMORY_SELF_FIRE", 0.012
            ),
            session_memory_camp_penalty=_env_float(
                "R_SESSION_MEMORY_CAMP", 0.004
            ),
            thermal_target_enabled=os.environ.get(
                "Q2_THERMAL_TARGETS", "1"
            ).lower() not in {"0", "false", "off", "no"},
            thermal_target_voxel_size=_env_float(
                "Q2_THERMAL_VOXEL_SIZE", 64.0
            ),
            thermal_target_decay=_env_float("Q2_THERMAL_DECAY", 0.75),
            thermal_target_max_age_ticks=_env_int(
                "Q2_THERMAL_MAX_AGE_TICKS", 5
            ),
            rust_lattice_enabled=bool(_env_int("Q2_RUST_LATTICE", 0)),
            lattice_preload_enabled=os.environ.get(
                "Q2_LATTICE_PRELOAD", "1"
            ).lower() not in {"0", "false", "off", "no"},
            lattice_routes_enabled=os.environ.get(
                "Q2_LATTICE_ROUTES", "1"
            ).lower() not in {"0", "false", "off", "no"},
            lattice_prior_scale=_env_float("Q2_LATTICE_PRIOR_SCALE", 8.0),
            lattice_route_scale=_env_float("Q2_LATTICE_ROUTE_SCALE", 4.0),
            lattice_tick_rate=_env_float("Q2_LATTICE_TICK_RATE", 10.0),
            lattice_route_refresh_ticks=_env_int(
                "Q2_LATTICE_ROUTE_REFRESH_TICKS", 5
            ),
            damage_prox_aversion=_env_float("R_DAMAGE_PROX_AVERSION", 0.004),
            offense_rune_reward=_env_float("R_OFFENSE_RUNE", 0.004),
            survival_rune_reward=_env_float("R_SURVIVAL_RUNE", 0.004),
            exposure_penalty=_env_float("R_EXPOSURE_PENALTY", 0.004),
            cover_reward=_env_float("R_COVER_REWARD", 0.002),
            bias_w_margin=_env_float("Q2_BIAS_W_MARGIN", 0.5),
            bias_w_exposure=_env_float("Q2_BIAS_W_EXPOSURE", 0.3),
            bias_w_hazard=_env_float("Q2_BIAS_W_HAZARD", 0.4),
            bias_w_outnumbered=_env_float("Q2_BIAS_W_OUTNUMBERED", 0.15),
            bias_w_escape=_env_float("Q2_BIAS_W_ESCAPE", 0.1),
            bad_fight_penalty=_env_float("R_BAD_FIGHT_PENALTY", 0.010),
            disengage_reward=_env_float("R_DISENGAGE_REWARD", 0.006),
            good_fight_reward=_env_float("R_GOOD_FIGHT_REWARD", 0.004),
            bad_fight_threat_weight=_env_float(
                "R_BAD_FIGHT_THREAT_WEIGHT", 0.5
            ),
            survival_tick_reward=_env_float("R_SURVIVE_TICK", 0.0002),
            survival_threat_reward=_env_float("R_SURVIVE_THREAT", 0.0015),
            survival_low_health_reward=_env_float(
                "R_SURVIVE_LOW_HEALTH", 0.002
            ),
            threat_engagement_reward=_env_float(
                "R_THREAT_ENGAGEMENT", 0.030
            ),
            threat_aim_reward=_env_float("R_THREAT_AIM", 0.040),
            threat_fire_reward=_env_float("R_THREAT_FIRE", 0.035),
            threat_damage_reward=_env_float("R_THREAT_DAMAGE", 0.010),
            threat_kill_reward=_env_float("R_THREAT_KILL", 2.000),
            threat_ignore_penalty=_env_float("R_THREAT_IGNORE", 0.030),
            threat_unready_penalty=_env_float("R_THREAT_UNREADY", 0.010),
            exchange_quality_reward=_env_float("R_EXCHANGE_QUALITY", 0.30),
            exchange_press_reward=_env_float("R_EXCHANGE_PRESS", 0.020),
            exchange_break_reward=_env_float("R_EXCHANGE_BREAK", 0.010),
            exchange_crash_penalty=_env_float("R_EXCHANGE_CRASH", 0.015),
            exchange_feed_penalty=_env_float("R_EXCHANGE_FEED", 0.020),
            exchange_jitter=_env_float("Q2_EXCHANGE_JITTER", 0.30),
            exchange_aggression_mag=_env_float("Q2_EXCHANGE_AGGRESSION", 0.60),
            rune_switch_reward=_env_float("R_RUNE_SWITCH", 0.05),
            rune_low_health=_env_float("Q2_RUNE_LOW_HEALTH", 50.0),
            rune_high_health=_env_float("Q2_RUNE_HIGH_HEALTH", 100.0),
            damage_advantage_reward=_env_float("R_DAMAGE_ADVANTAGE", 0.004),
            damage_disadvantage_penalty=_env_float(
                "R_DAMAGE_DISADVANTAGE", 0.003
            ),
            frag_advantage_reward=_env_float("R_FRAG_ADVANTAGE", 0.20),
            frag_disadvantage_penalty=_env_float(
                "R_FRAG_DISADVANTAGE", 0.35
            ),
            episode_win_reward=_env_float("R_EPISODE_WIN", 1.50),
            episode_survival_reward=_env_float("R_EPISODE_SURVIVAL", 0.50),
            episode_loss_penalty=_env_float("R_EPISODE_LOSS", 1.00),
            episode_idle_penalty=_env_float("R_EPISODE_IDLE", 0.35),
            win_damage_margin=_env_float("Q2_WIN_DAMAGE_MARGIN", 60.0),
            meaningful_contact_min=_env_float("Q2_MEANINGFUL_CONTACT_MIN", 8.0),
            low_health_threshold=_env_float("Q2_LOW_HEALTH", 35.0),
            recent_threat_window=_env_int("Q2_RECENT_THREAT_WINDOW", 90),
            rng=random.Random(seed),
        )

    def reset(self, map_name: str, obs: Optional[Observation] = None) -> None:
        self._invalidate_feature_cache()
        self.map_name = map_name
        self._memory_for_map(map_name)
        tick = int(getattr(obs, "tick", 0)) if obs is not None else 0
        previous_tick = self._map_last_tick.get(map_name)
        map_reloaded = previous_tick is not None and tick < previous_tick
        self._ensure_map_lattice(map_name, tick, reset_clock=map_reloaded)
        self._map_last_tick[map_name] = tick
        self.visited.clear()
        self.last_cell = None
        self.steps_in_cell = 0
        self.pos_history.clear()
        self.last_visible_count = 0
        self.last_damage_tick = -1000000
        self.last_damage_cell = None
        self._thermal_tracks.clear()
        self._thermal_last_tick = -1
        self._thermal_last_signal = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._clear_hook_correction()
        self._reset_episode_state()
        if obs is not None:
            cell = self.cell_for(obs)
            self.visited.add(cell)
            self.last_cell = cell
            self.pos_history.append(tuple(obs.self_state[:3]))
            self._refresh_route_heat(obs, force=True)

    @staticmethod
    def _sidecar_roots() -> List[Path]:
        """Ordered roots for generated/installed lattice sidecars."""
        roots: List[Path] = []
        configured = os.environ.get("Q2_LATTICE_DIR", "")
        for raw in configured.split(os.pathsep):
            if raw.strip():
                roots.append(Path(raw).expanduser())
        maps_dir = os.environ.get("Q2_MAPS_DIR", "").strip()
        if maps_dir:
            roots.append(Path(maps_dir).expanduser())
        q2_root = Path(os.environ.get("Q2_ROOT", "/home/raymond/q2_lithium_merge"))
        roots.extend((q2_root / "baseq2" / "maps", q2_root / "lithium" / "maps"))
        roots.append(Path(__file__).resolve().parent.parent / "maps" / "generated")
        unique: List[Path] = []
        seen = set()
        for root in roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key not in seen:
                unique.append(root)
                seen.add(key)
        return unique

    def _find_sidecar(self, map_name: str, suffix: str) -> Optional[Path]:
        filename = f"{map_name}.{suffix}.json"
        for root in self._sidecar_roots():
            candidate = root / filename
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _read_attested_sidecar(
        map_name: str, path: Path,
    ) -> tuple[bytes, dict | None]:
        """Verify farm sidecars while retaining legacy local-map support."""
        manifest_path = path.parent / installed_manifest_name(map_name)
        if not manifest_path.is_file():
            if farm_map_requires_attestation(map_name):
                raise ValueError(
                    f"farm sidecar has no installed bundle manifest: {manifest_path}"
                )
            return path.read_bytes(), None
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        validate_manifest(manifest, map_name)
        payload = verify_installed_artifact(path, manifest)
        return payload, {
            "path": str(manifest_path),
            "sha256": sha256_bytes(manifest_payload),
        }

    def _ensure_map_lattice(self, map_name: str, tick: int,
                            reset_clock: bool = False) -> None:
        """Load generator priors once and initialize the live item clock."""
        if not map_name:
            return
        sources = self.sidecar_sources.setdefault(map_name, {})
        if self.lattice_preload_enabled and map_name not in self.preloaded_maps:
            lattice_path = self._find_sidecar(map_name, "lattice")
            if lattice_path is not None:
                try:
                    encoded, attestation = self._read_attested_sidecar(
                        map_name, lattice_path,
                    )
                    payload = json.loads(encoded)
                    self._preload_lattice_payload(map_name, payload)
                    sources["lattice"] = str(lattice_path)
                    if attestation is not None:
                        sources["bundle"] = attestation["path"]
                        sources["bundle_sha256"] = attestation["sha256"]
                except (OSError, ValueError, TypeError, KeyError) as error:
                    sources["lattice_error"] = f"{lattice_path}: {error}"
            self.preloaded_maps.add(map_name)

        if not self.lattice_routes_enabled:
            return
        graph = self.route_graphs.get(map_name)
        if graph is None:
            route_path = self._find_sidecar(map_name, "routes")
            if route_path is not None:
                try:
                    encoded, attestation = self._read_attested_sidecar(
                        map_name, route_path,
                    )
                    graph = json.loads(encoded)
                    self.route_graphs[map_name] = graph
                    sources["routes"] = str(route_path)
                    sources["item_timing"] = str(route_path)
                    if attestation is not None:
                        sources["bundle"] = attestation["path"]
                        sources["bundle_sha256"] = attestation["sha256"]
                except (OSError, ValueError, TypeError, KeyError) as error:
                    sources["routes_error"] = f"{route_path}: {error}"
                    graph = None
        if graph is not None and (map_name not in self.item_timings or reset_clock):
            table = ItemTimingTable()
            table.reset_from_routes(graph, self._game_time(tick))
            self.item_timings[map_name] = table

    def _preload_lattice_payload(self, map_name: str, payload: dict) -> None:
        scale = max(0.1, float(self.lattice_prior_scale))
        memory = self._memory_for_map(map_name)
        self._invalidate_cell_col_cache(map_name)
        for objective in payload.get("objectives", []):
            pos = (objective["x"], objective["y"], objective["z"])
            cell = self.cell_for_pos(pos)
            entry = memory.setdefault(cell, SessionMemoryCell())
            entry.prior_opportunity = max(
                entry.prior_opportunity,
                scale * max(0.0, float(objective.get("value", 1.0))),
            )
        for bounds in payload.get("danger", []):
            if len(bounds) < 6:
                continue
            lo = self.cell_for_pos(bounds[:3])
            hi = self.cell_for_pos(bounds[3:6])
            for ix in range(min(lo[0], hi[0]), max(lo[0], hi[0]) + 1):
                for iy in range(min(lo[1], hi[1]), max(lo[1], hi[1]) + 1):
                    for iz in range(min(lo[2], hi[2]), max(lo[2], hi[2]) + 1):
                        entry = memory.setdefault((ix, iy, iz), SessionMemoryCell())
                        entry.prior_threat = max(entry.prior_threat, scale)

    def _game_time(self, tick: int) -> float:
        return float(tick) / max(0.1, float(self.lattice_tick_rate))

    def _invalidate_feature_cache(self) -> None:
        self._feature_cache_key = None
        self._feature_cache = None
        self._feature_nearest_deaths = 0.0
        self._feature_nearest_hazard = 0.0

    def _mark_rust_dirty(
        self, cell: Tuple[int, int, int], map_name: Optional[str] = None
    ) -> None:
        if not self.rust_lattice_enabled:
            return
        key = map_name or self.map_name or "unknown"
        normalized = tuple(int(value) for value in cell)
        self._rust_dirty_cells.setdefault(key, set()).add(normalized)
        self._rust_removed_cells.setdefault(key, set()).discard(normalized)

    def _mark_rust_removed(
        self, cell: Tuple[int, int, int], map_name: Optional[str] = None
    ) -> None:
        if not self.rust_lattice_enabled:
            return
        key = map_name or self.map_name or "unknown"
        normalized = tuple(int(value) for value in cell)
        self._rust_removed_cells.setdefault(key, set()).add(normalized)
        self._rust_dirty_cells.setdefault(key, set()).discard(normalized)

    def _mark_rust_score_event(
        self,
        cell: Tuple[int, int, int],
        engagement: float = 0.0,
        threat: float = 0.0,
        opportunity: float = 0.0,
        self_fire: float = 0.0,
        deaths: float = 0.0,
        samples: float = 0.0,
        force_confident: bool = False,
        confidence_override: Optional[float] = None,
        map_name: Optional[str] = None,
    ) -> None:
        if not self.rust_lattice_enabled:
            return
        key = map_name or self.map_name or "unknown"
        normalized = tuple(int(value) for value in cell)
        events = self._rust_score_events.setdefault(key, {})
        delta = events.get(normalized)
        if delta is None:
            delta = np.zeros(8, dtype=np.float32)
            delta[7] = np.nan
            events[normalized] = delta
        delta[:6] += (
            engagement,
            threat,
            opportunity,
            self_fire,
            deaths,
            samples,
        )
        if force_confident:
            delta[6] = 1.0
        if confidence_override is not None:
            delta[7] = float(confidence_override)

    def _rust_index(self):
        """Return the current map's index, or permanently fall back to Python."""
        if not self.rust_lattice_enabled:
            return None
        key = self.map_name or "unknown"
        existing = self._rust_indices.get(key)
        if existing is not None:
            return existing
        try:
            from .rust_lattice import StatefulLatticeIndex

            index = StatefulLatticeIndex(self, sync=True)
        except (ImportError, RuntimeError, AttributeError, ValueError) as error:
            self._rust_fallback_reason = str(error)
            self.rust_lattice_enabled = False
            return None
        self._rust_indices[key] = index
        self._rust_dirty_cells.pop(key, None)
        self._rust_removed_cells.pop(key, None)
        self._rust_score_events.pop(key, None)
        return index

    def _flush_rust_index(self):
        index = self._rust_index()
        if index is None:
            return None
        key = self.map_name or "unknown"
        removed = self._rust_removed_cells.pop(key, set())
        dirty = self._rust_dirty_cells.pop(key, set())
        events = self._rust_score_events.pop(key, {})
        if removed:
            index.remove_cells(removed)
        for cell in removed | dirty:
            events.pop(cell, None)
        if events:
            index.apply_score_events(events)
            self._rust_event_rows_applied += len(events)
        if dirty:
            index.apply_cells(self, dirty)
        return index

    def _wanted_route_axis(self, obs: Observation) -> str:
        health = float(obs.self_state[6]) if len(obs.self_state) > 6 else 100.0
        if health <= self.rune_low_health:
            return "survival"
        if health >= self.rune_high_health:
            return "offense"
        return "value"

    def _nearest_pickup_id(
        self, table: ItemTimingTable, obs: Observation
    ) -> Optional[int]:
        if max(0.0, float(getattr(obs, "reward_item_pickup", 0.0))) <= 0.0:
            return None
        pos = tuple(float(v) for v in obs.self_state[:3])
        candidates = [row for row in table.rows.values() if row.present]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda row: hypot(
            row.pos[0] - pos[0], row.pos[1] - pos[1], row.pos[2] - pos[2]))
        distance = hypot(nearest.pos[0] - pos[0], nearest.pos[1] - pos[1],
                         nearest.pos[2] - pos[2])
        return nearest.spawn_id if distance <= self.voxel_size * 1.5 else None

    def _clear_dynamic_heat(self, map_name: str) -> None:
        memory = self._memory_for_map(map_name)
        for cell in self.dynamic_cells.get(map_name, set()):
            entry = memory.get(cell)
            if entry is not None:
                old_threat = max(0.0, -entry.readiness)
                old_opportunity = (
                    max(0.0, entry.readiness) + max(0.0, entry.route_bias)
                )
                entry.readiness = 0.0
                entry.route_bias = 0.0
                self._mark_cell_col_dirty(cell, map_name)
                self._mark_rust_score_event(
                    cell,
                    threat=-old_threat,
                    opportunity=-old_opportunity,
                    confidence_override=self._memory_confidence(entry),
                    map_name=map_name,
                )
        self.dynamic_cells[map_name] = set()

    def _deposit_dynamic(self, map_name: str, deposit: dict,
                         route_bias: float = 0.0) -> None:
        amount = float(deposit.get("amount", 0.0))
        radius = max(
            self.voxel_size * 0.5,
            float(deposit.get("radius", self.voxel_size)),
        )
        center = np.asarray(
            (deposit["x"], deposit["y"], deposit["z"]), dtype=np.float32
        )
        base = self.cell_for_pos(center)
        reach = max(0, int(np.ceil(radius / max(1.0, self.voxel_size))))
        memory = self._memory_for_map(map_name)
        touched = self.dynamic_cells.setdefault(map_name, set())
        for ix in range(base[0] - reach, base[0] + reach + 1):
            for iy in range(base[1] - reach, base[1] + reach + 1):
                for iz in range(base[2] - reach, base[2] + reach + 1):
                    cell = (ix, iy, iz)
                    distance = float(np.linalg.norm(self._cell_center(cell) - center))
                    weight = max(0.0, 1.0 - distance / radius)
                    if weight <= 0.0:
                        continue
                    entry = memory.setdefault(cell, SessionMemoryCell())
                    old_threat = max(0.0, -entry.readiness)
                    old_opportunity = (
                        max(0.0, entry.readiness) + max(0.0, entry.route_bias)
                    )
                    entry.readiness += amount * self.lattice_route_scale * weight
                    entry.route_bias += route_bias * weight
                    touched.add(cell)
                    self._mark_cell_col_dirty(cell, map_name)
                    new_threat = max(0.0, -entry.readiness)
                    new_opportunity = (
                        max(0.0, entry.readiness) + max(0.0, entry.route_bias)
                    )
                    self._mark_rust_score_event(
                        cell,
                        threat=new_threat - old_threat,
                        opportunity=new_opportunity - old_opportunity,
                        confidence_override=self._memory_confidence(entry),
                        map_name=map_name,
                    )

    def _refresh_route_heat(self, obs: Observation, force: bool = False) -> bool:
        table = self.item_timings.get(self.map_name)
        if table is None:
            self.selected_route = ""
            return False
        tick = int(obs.tick)
        pickup_id = self._nearest_pickup_id(table, obs)
        wanted = self._wanted_route_axis(obs)
        last_tick = self._route_heat_last_tick.get(self.map_name, -10**9)
        last_axis = self._route_heat_last_axis.get(self.map_name, "")
        refresh_ticks = max(1, int(self.lattice_route_refresh_ticks))
        if (
            not force
            and pickup_id is None
            and wanted == last_axis
            and tick >= last_tick
            and tick - last_tick < refresh_ticks
        ):
            return False

        t = self._game_time(tick)
        table.observe(t, (), (), (() if pickup_id is None else (pickup_id,)))
        self._clear_dynamic_heat(self.map_name)
        graph = self.route_graphs.get(self.map_name, {})
        route_name = "balanced" if wanted == "value" else wanted
        route = next((r for r in graph.get("routes", [])
                      if r.get("archetype") == route_name), None)
        route_nodes = set(route.get("node_ids", ())) if route else set()
        self.selected_route = route_name if route else "readiness"
        for deposit in table.readiness_deposits(t, want_axis=wanted):
            sid = min(
                table.rows,
                key=lambda key: hypot(
                    table.rows[key].pos[0] - float(deposit["x"]),
                    table.rows[key].pos[1] - float(deposit["y"]),
                    table.rows[key].pos[2] - float(deposit["z"]),
                ),
            ) if table.rows else -1
            bias = (
                0.25 * self.lattice_route_scale
                if sid in route_nodes and float(deposit.get("amount", 0.0)) > 0
                else 0.0
            )
            self._deposit_dynamic(self.map_name, deposit, route_bias=bias)
        self._route_heat_last_tick[self.map_name] = tick
        self._route_heat_last_axis[self.map_name] = wanted
        self._invalidate_feature_cache()
        return True

    def cell_for(self, obs: Observation) -> Tuple[int, int, int]:
        return self.cell_for_pos(obs.self_state[:3])

    def cell_for_pos(self, pos) -> Tuple[int, int, int]:
        size = max(self.voxel_size, 1.0)
        return tuple(int(floor(float(v) / size)) for v in pos)

    def _clear_hook_correction(self) -> None:
        self._hook_correction_target = None
        self._hook_correction_anchor = None
        self._hook_correction_hot_target = None
        self._hook_correction_heat = 0.0
        self._hook_correction_initial_distance = 0.0
        self._hook_correction_best_distance = 0.0
        self._hook_correction_started_tick = -1
        self._hook_correction_escape = False

    def _heated_hook_correction(self, obs: Observation) -> Optional[dict]:
        """Select a reachable hook landing that advances toward lattice heat.

        The opportunity lattice is the destination authority.  Hook zones are
        the reachability authority: a landing is eligible only when its anchor
        is in the live observation and that landing makes concrete progress
        toward the selected heated cell.  This keeps a 1700 u/s grapple pull
        from becoming ordinary locomotion or an action-rate reward.
        """
        direction = self._nearest_memory_signal(obs, "opportunity")
        heat = float(direction[3])
        if heat < max(0.0, float(self.hook_correction_min_heat)):
            return None

        origin = np.asarray(obs.self_state[:3], dtype=np.float32)
        hot_offset = np.asarray(direction[:3], dtype=np.float32)
        if float(np.linalg.norm(hot_offset)) <= 1e-6:
            return None
        radius = max(float(self.session_memory_search_radius), self.voxel_size)
        hot_target = origin + hot_offset * radius
        current_distance = float(np.linalg.norm(hot_target - origin))
        if current_distance <= max(1.0, float(self.hook_correction_arrival_radius)):
            return None

        count = max(0, min(
            int(getattr(obs, "hook_zone_count", 0)),
            getattr(obs, "hook_zones", np.empty((0, 8))).shape[0],
        ))
        best = None
        max_anchor = max(1.0, float(self.hook_correction_max_anchor_distance))
        min_advance = max(0.0, float(self.hook_correction_min_advance))
        for index, zone in enumerate(obs.hook_zones[:count]):
            anchor = np.asarray(zone[:3], dtype=np.float32)
            landing = np.asarray(zone[3:6], dtype=np.float32)
            anchor_distance = float(np.linalg.norm(anchor - origin))
            if anchor_distance <= 1.0 or anchor_distance > max_anchor:
                continue
            landing_distance = float(np.linalg.norm(hot_target - landing))
            advance = current_distance - landing_distance
            if advance < min_advance:
                continue
            # Prefer the landing that removes the most remaining distance;
            # heat breaks ties when multiple live zones reach the same area.
            rank = advance + heat * max(1.0, self.voxel_size)
            if best is None or rank > best["rank"]:
                best = {
                    "rank": float(rank),
                    "zone_index": int(index),
                    "anchor": tuple(float(value) for value in anchor),
                    "landing": tuple(float(value) for value in landing),
                    "hot_target": tuple(float(value) for value in hot_target),
                    "heat": heat,
                    "distance": float(np.linalg.norm(landing - origin)),
                    "advance": float(advance),
                    "required": float(int(zone[7]) & HOOK_REQUIRED != 0),
                }
        return best

    def _start_hook_correction(
        self, candidate: dict, obs: Observation, *, escape: bool
    ) -> None:
        self._hook_correction_target = candidate["landing"]
        self._hook_correction_anchor = candidate["anchor"]
        self._hook_correction_hot_target = candidate["hot_target"]
        self._hook_correction_heat = float(candidate["heat"])
        origin = np.asarray(obs.self_state[:3], dtype=np.float32)
        target = np.asarray(self._hook_correction_target, dtype=np.float32)
        distance = float(np.linalg.norm(target - origin))
        self._hook_correction_initial_distance = distance
        self._hook_correction_best_distance = distance
        self._hook_correction_started_tick = int(obs.tick)
        self._hook_correction_escape = bool(escape)

    def update(self, obs: Observation) -> Tuple[float, Dict[str, float]]:
        if not self.enabled:
            return 0.0, {"spatial_bonus": 0.0}

        bonus = 0.0
        info: Dict[str, float] = {}

        cell = self.cell_for(obs)
        self._map_last_tick[self.map_name] = int(obs.tick)
        self._refresh_route_heat(obs)
        is_new = cell not in self.visited
        if is_new:
            self.visited.add(cell)
            bonus += self.new_cell_reward

        # Track history of positions for displacement-based stagnation checks
        pos = tuple(obs.self_state[:3])
        self.pos_history.append(pos)
        if len(self.pos_history) > self.stagnation_steps:
            self.pos_history.pop(0)

        stagnated = False
        if cell == self.last_cell:
            self.steps_in_cell += 1
            if self.steps_in_cell > self.stagnation_steps:
                stagnated = True
        else:
            self.steps_in_cell = 0
            self.last_cell = cell

        # Hysteresis backup: if the bot is crossing voxel boundaries but moving < 48 units total
        if not stagnated and len(self.pos_history) == self.stagnation_steps:
            old_pos = self.pos_history[0]
            displacement = hypot(pos[0] - old_pos[0], pos[1] - old_pos[1], pos[2] - old_pos[2])
            if displacement < 48.0:
                stagnated = True

        if stagnated:
            bonus -= self.stagnation_penalty

        # Exploration/navigation reward is useful for reaching combat, but it
        # must not remain a competing objective after contact begins.
        navigation_bonus = max(0.0, float(bonus))

        fire_context = self.fire_context(obs)
        entity_count = max(
            0, min(int(obs.entity_count), int(obs.entities.shape[0]))
        )
        enemy_count = sum(
            1 for ent in obs.entities[:entity_count] if ent[7] > 0.5
        )
        visible_count = int(fire_context["enemy_visible_count"])
        nearest_visible = float(fire_context["enemy_visible_nearest"])
        aim_aligned = bool(fire_context["aim_aligned"])
        aim_yaw_error = float(fire_context["aim_yaw_error"])
        aim_pitch_error = float(fire_context["aim_pitch_error"])
        splash_viable = bool(fire_context["splash_viable"])
        splash_weapon = bool(fire_context["splash_weapon"])
        audio_contact = bool(fire_context["audio_contact"])
        damage_dealt = max(
            0.0, float(getattr(obs, "reward_damage_dealt", 0.0))
        )
        kills = max(0.0, float(getattr(obs, "reward_kill", 0.0)))
        action_debug = getattr(obs, "action_debug", ())
        hit_streak = (
            (int(action_debug[11]) & ML_HIT_STREAK_MASK)
            >> ML_HIT_STREAK_SHIFT
        ) if len(action_debug) >= 12 else 0

        tactical_engagement = (
            visible_count > 0 and
            self.engagement_min <= nearest_visible <= self.engagement_max
        )
        if tactical_engagement:
            bonus += self.engagement_reward

        if aim_aligned:
            bonus += self.aim_alignment_reward
        # The hard aligned bonus is intentionally decisive near the firing
        # gate, but it was too sparse to pull a downward-looking policy back
        # toward a visible player.  Add bounded dense progress inside the
        # forward hemisphere; a target behind the view reports no candidate
        # errors and receives no tracking credit.
        aim_tracking_quality = 0.0
        if visible_count > 0 and (
                aim_aligned or aim_yaw_error > 0.0 or aim_pitch_error > 0.0):
            yaw_quality = max(0.0, 1.0 - aim_yaw_error / 90.0)
            pitch_quality = max(0.0, 1.0 - aim_pitch_error / 60.0)
            aim_tracking_quality = yaw_quality * pitch_quality
            bonus += self.aim_tracking_reward * aim_tracking_quality

        hook_required_near = self._has_required_hook_nearby(obs)

        hook_context = self.hook_context(obs)
        hook_action = int(hook_context["hook_action"])
        ammo_depleted = bool(hook_context["ammo_depleted"])
        hook_delta = 0.0
        hook_enemy = False
        hook_no_ammo_melee = False
        hook_blind = False
        hook_traversal = False
        hook_overspeed = False
        hook_fired = hook_action == 1
        hook_noop = hook_action == 2  # reserved by C; intentionally does nothing
        hook_released = hook_action == 3
        hook_release_overspeed = False

        movement = self.movement_context(obs)
        movement_delta = float(movement["movement_discipline"])
        horizontal_speed = float(movement["movement_speed"])
        movement_intent = float(movement["movement_intent"])
        forward_intent = float(movement["forward_intent"])
        backward_intent = float(movement["backward_intent"])
        nominal_speed = bool(movement["movement_nominal"])
        slow_movement = bool(movement["movement_slow"])
        overspeed = bool(movement["movement_overspeed"])
        jump_action = bool(movement["jump_action"])
        jump_slow = bool(movement["jump_slow"])
        bonus += movement_delta

        # Grapple is an escape/correction actuator.  Positive reward is based
        # on one fixed landing target and new best displacement toward it, so
        # repeatedly issuing hook cannot manufacture reward.  The destination
        # must be reachable through a live hook zone and advance toward a
        # positive opportunity/readiness lattice cell.
        hook_candidate = (
            self._heated_hook_correction(obs)
            if self.lattice_reward_enabled
            else None
        )
        damage_now = max(0.0, float(getattr(obs, "reward_damage_taken", 0.0)))
        recent_damage = (
            0 <= int(obs.tick) - int(self.last_damage_tick)
            <= max(1, int(self.recent_threat_window))
        )
        health = float(obs.self_state[6]) if len(obs.self_state) > 6 else 100.0
        escape_needed = bool(
            damage_now > 0.0
            or recent_damage
            or self.recent_threat_steps > 0
            or (health <= self.low_health_threshold and visible_count > 0)
        )
        correction_needed = bool(
            stagnated or slow_movement or hook_required_near or escape_needed
        )
        hook_correction_available = hook_candidate is not None
        hook_correction_started = False
        hook_correction_progress = 0.0
        hook_correction_progress_delta = 0.0
        hook_correction_success = False
        hook_correction_timed_out = False
        if self._hook_correction_target is not None:
            hook_correction_report_target = self._hook_correction_target
            hook_correction_report_anchor = self._hook_correction_anchor
            hook_correction_report_hot_target = self._hook_correction_hot_target
            hook_correction_report_heat = self._hook_correction_heat
            hook_correction_report_escape = self._hook_correction_escape
        elif hook_candidate is not None:
            hook_correction_report_target = hook_candidate["landing"]
            hook_correction_report_anchor = hook_candidate["anchor"]
            hook_correction_report_hot_target = hook_candidate["hot_target"]
            hook_correction_report_heat = float(hook_candidate["heat"])
            hook_correction_report_escape = escape_needed
        else:
            hook_correction_report_target = None
            hook_correction_report_anchor = None
            hook_correction_report_hot_target = None
            hook_correction_report_heat = 0.0
            hook_correction_report_escape = False

        if self._hook_correction_target is not None:
            elapsed = int(obs.tick) - self._hook_correction_started_tick
            if elapsed < 0 or elapsed > max(1, self.hook_correction_timeout_ticks):
                hook_correction_timed_out = True
                self._clear_hook_correction()
            else:
                target = np.asarray(self._hook_correction_target, dtype=np.float32)
                origin = np.asarray(obs.self_state[:3], dtype=np.float32)
                distance = float(np.linalg.norm(target - origin))
                improvement = self._hook_correction_best_distance - distance
                if improvement >= max(
                    0.0, float(self.hook_correction_progress_epsilon)
                ):
                    self._hook_correction_best_distance = distance
                    hook_correction_progress = float(improvement)
                    initial = max(1.0, self._hook_correction_initial_distance)
                    hook_correction_progress_delta = (
                        self.hook_correction_progress_reward
                        * min(1.0, improvement / initial)
                    )
                    hook_delta += hook_correction_progress_delta
                    hook_traversal = True
                if distance <= max(
                    1.0, float(self.hook_correction_arrival_radius)
                ):
                    hook_correction_success = True
                    hook_delta += self.hook_correction_complete_reward

        if hook_fired:
            hook_delta -= self.hook_cost
            if (
                self._hook_correction_target is None
                and hook_candidate is not None
                and correction_needed
                and not hook_correction_success
            ):
                self._start_hook_correction(
                    hook_candidate, obs, escape=escape_needed
                )
                hook_correction_started = True
            elif self._hook_correction_target is None:
                # Blind-hook discipline only applies while the lattice-guided
                # correction role is live; with that role ablated the hook is
                # an unguided actuator and pays cost/discipline terms only.
                if self.lattice_reward_enabled:
                    hook_blind = True
                    hook_delta -= self.hook_blind_penalty
            if overspeed:
                hook_overspeed = True
                hook_delta -= self.hook_overspeed_penalty
        elif hook_noop:
            # Protocol class 2 has no C-side effect. Rewarding it as generic
            # hook use caused the first movement-reset policy to spam it.
            hook_delta -= self.hook_noop_penalty
        elif hook_released:
            if self._hook_correction_target is not None:
                # Release ends this one-shot correction.  It is only positive
                # if arrival was already measured above; overspeed release is
                # safe but no longer a farmable reward event.
                if not hook_correction_success and not overspeed:
                    hook_delta -= self.hook_release_idle_penalty
                self._clear_hook_correction()
            elif overspeed:
                hook_release_overspeed = True
            else:
                hook_delta -= self.hook_release_idle_penalty
        if hook_correction_success:
            self._clear_hook_correction()
        bonus += hook_delta

        fired = self._action_fired(obs)
        fire_delta = 0.0
        fire_unseen = False
        fire_unaligned = False
        fire_no_ammo = False
        fire_aligned = False
        fire_splash = False
        fire_audio_contact = False
        if fired:
            fire_delta -= self.fire_cost
            if ammo_depleted:
                fire_no_ammo = True
                fire_delta -= self.fire_no_ammo_penalty
            if visible_count <= 0:
                if splash_weapon and audio_contact:
                    fire_audio_contact = True
                else:
                    fire_unseen = True
                    fire_delta -= self.fire_unseen_penalty
            elif aim_aligned:
                fire_aligned = True
                fire_delta += self.fire_aligned_reward
            elif splash_viable:
                fire_splash = True
                fire_delta += self.splash_fire_reward
            else:
                fire_unaligned = True
                fire_delta -= self.fire_unaligned_penalty
            bonus += fire_delta

        threat_delta, threat_info = self._threat_reward(
            obs=obs,
            visible_count=visible_count,
            nearest_visible=nearest_visible,
            aim_aligned=aim_aligned,
            fired=fired,
            fire_aligned=fire_aligned,
            fire_splash=fire_splash,
            hook_enemy=hook_enemy,
            ammo_depleted=ammo_depleted,
        )
        bonus += threat_delta

        # Captured before _update_session_memory refreshes last_visible_count:
        # "lost sight of them this tick" is the disengage signal.
        disengage_now = self.last_visible_count > 0 and visible_count == 0

        self._update_session_memory(
            obs=obs,
            cell=cell,
            visible_count=visible_count,
            fired=fired,
            hook_enemy=hook_enemy,
            fire_audio_contact=fire_audio_contact,
            audio_contact=audio_contact,
        )
        # Session marks, DPS state, and possibly route heat changed this tick.
        # Compute once below; env._obs_vector() reuses the exact cached vector
        # (gated view). Reward terms read the ungated internal vector.
        self._invalidate_feature_cache()
        memory_features = self._memory_features_internal(obs)
        memory_delta = 0.0
        current_engagement = float(memory_features[0])
        current_threat = float(memory_features[1])
        current_opportunity = float(memory_features[2])
        current_self_fire = float(memory_features[3])
        nearest_engagement = float(memory_features[8])
        nearest_threat = float(memory_features[12])
        nearest_opportunity = float(memory_features[16])
        combat_ready = (
            float(obs.self_state[6]) > 35.0 and
            not ammo_depleted
        )

        if self.lattice_reward_enabled and combat_ready and visible_count <= 0:
            memory_delta += self.session_memory_engagement_reward * nearest_engagement
            memory_delta += self.session_memory_opportunity_reward * nearest_opportunity

        if self.lattice_reward_enabled and float(obs.self_state[6]) <= 35.0:
            memory_delta -= self.session_memory_threat_penalty * max(
                current_threat, nearest_threat
            )

        # Personal death-spot aversion: repeated deaths in a cell reduce THIS
        # bot's gravity toward it, unconditionally (not just at low health).
        # Each repeat deepens the repulsion until the tanh saturates, directly
        # counteracting the engagement/opportunity pull at lethal locations.
        nearest_deaths = self._feature_nearest_deaths
        entry_here = self._memory_for_map(self.map_name).get(cell)
        current_deaths = 0.0
        if entry_here is not None and entry_here.deaths > 0.0:
            current_deaths = (self._norm_memory_score(3.0 * entry_here.deaths)
                              * self._memory_confidence(entry_here))
        death_aversion = max(current_deaths, nearest_deaths)
        if self.lattice_reward_enabled and death_aversion > 0.0:
            memory_delta -= self.session_memory_death_aversion * death_aversion

        # Hazard aversion: typed environmental memory (lava/slime/water/
        # crush/falling) repels like the death channel, but lives in its own
        # channel so combat threat stays de-conflated.
        nearest_hazard = self._feature_nearest_hazard
        current_hazard = 0.0
        if entry_here is not None and (
            entry_here.hazard_damage > 0.0 or entry_here.hazard_deaths > 0.0
        ):
            current_hazard = (
                self._norm_memory_score(
                    0.03 * entry_here.hazard_damage
                    + 3.0 * entry_here.hazard_deaths
                )
                * self._memory_confidence(entry_here)
            )
        hazard_aversion = max(current_hazard, nearest_hazard)
        if self.lattice_reward_enabled and hazard_aversion > 0.0:
            memory_delta -= self.session_memory_hazard_aversion * hazard_aversion

        if (
            self.lattice_reward_enabled and
            fired and
            visible_count <= 0 and
            not audio_contact and
            current_self_fire > 0.25 and
            current_opportunity < 0.20
        ):
            memory_delta -= self.session_memory_self_fire_penalty * current_self_fire

        if (
            self.lattice_reward_enabled and
            stagnated and
            visible_count <= 0 and
            not audio_contact and
            current_engagement > 0.35
        ):
            memory_delta -= self.session_memory_camp_penalty * current_engagement

        bonus += memory_delta

        # ── fight decision bias (advisory) ────────────────────────────
        # Scalar in [-1, +1]: should I take this fight? Reward shaping and
        # typed lattice deposits only — it NEVER masks or alters actions.
        # Active while alive; the death tick itself stays active so the
        # fatal-bias deposit can land (corpse frames read 0.0).
        deaths_now = max(0.0, float(getattr(obs, "reward_death", 0.0)))
        health_now = (
            float(obs.self_state[6]) if len(obs.self_state) > 6 else 100.0
        )
        bias_active = health_now > 0.0 or deaths_now > 0.0
        margin_now, _ehp_now, _share_now = self._win_margin(obs)
        my_exposure_now = float(fire_context["enemy_visible_exposure_max"])
        self_exposure_now = float(
            np.clip(getattr(obs, "self_exposure", 0.0), 0.0, 1.0)
        )
        escape_ready = int(getattr(obs, "hook_zone_count", 0)) > 0
        margin_term = self.bias_w_margin * margin_now
        exposure_term = self.bias_w_exposure * (
            my_exposure_now - self_exposure_now
        )
        hazard_term = self.bias_w_hazard * hazard_aversion
        outnumbered_term = self.bias_w_outnumbered * max(0, enemy_count - 1)
        escape_term = self.bias_w_escape * (1.0 if escape_ready else 0.0)
        fight_bias = (
            float(np.clip(
                margin_term + exposure_term - hazard_term
                - outnumbered_term + escape_term,
                -1.0, 1.0,
            ))
            if bias_active else 0.0
        )

        engaging = bool(
            (fired and visible_count > 0)
            or (aim_aligned and tactical_engagement)
        )
        bad_fight_penalty_now = 0.0
        disengage_reward_now = 0.0
        good_fight_reward_now = 0.0
        if self.lattice_reward_enabled and bias_active:
            if fight_bias < -0.2 and engaging:
                bad_fight_penalty_now = self.bad_fight_penalty * abs(fight_bias)
                bonus -= bad_fight_penalty_now
                self.episode_bad_fights += 1.0
            if fight_bias < -0.2 and disengage_now:
                disengage_reward_now = self.disengage_reward * abs(fight_bias)
                bonus += disengage_reward_now
                self.episode_good_disengages += 1.0
            if fight_bias > 0.3 and engaging:
                good_fight_reward_now = self.good_fight_reward * fight_bias
                bonus += good_fight_reward_now

        # Decision-quality deposits (storage gate: session_memory_enabled;
        # typed channels, never folded into combat counters).
        if self.session_memory_enabled and bias_active:
            if deaths_now > 0.0 and fight_bias < -0.2:
                self._memory_cell(
                    cell, int(obs.tick)
                ).bad_fight_taken += deaths_now
            if disengage_reward_now > 0.0:
                self._memory_cell(
                    cell, int(obs.tick)
                ).good_disengage += 1.0

        # Engine-supplied extended channels (both runs feel these even when
        # Run A can't see them in its obs vector):
        #   prox    — proximity-weighted damage taken: hard close hits sting
        #             more than far chip, sharpening the dodge/avoid gradient.
        #   offense — damage dealt while holding strength/haste: pays off
        #             fighting aggressively once you've earned an offense rune.
        #   survival— health recovered (regen/vampire) + armour absorbed:
        #             pays off the survival investment.
        prox_dmg   = max(0.0, float(getattr(obs, "reward_damage_taken_prox", 0.0)))
        offense    = max(0.0, float(getattr(obs, "reward_offense", 0.0)))
        survival   = max(0.0, float(getattr(obs, "reward_survival", 0.0)))
        ext_delta  = (-self.damage_prox_aversion * prox_dmg
                      + self.offense_rune_reward * offense
                      + self.survival_rune_reward * survival)
        bonus += ext_delta

        navigation_suppressed = 0.0
        if tactical_engagement and navigation_bonus > 0.0:
            navigation_suppressed = (
                navigation_bonus
                * max(0.0, min(1.0, self.combat_navigation_suppression))
            )
            bonus -= navigation_suppressed

        # Contextual rune switching: reward acquiring (or swapping to) the rune
        # that fits the current health need, scaled by the fit GAIN over the
        # rune dropped. Low health → survival runes pay; healthy → offense; a
        # swap from a mismatched rune to a matched one is the big reward — the
        # "drop strength, grab regen when bleeding" dynamic.
        rune_switch_delta = 0.0
        rf = obs.rune_flags
        cur_rune = int(np.argmax(rf)) if float(rf.sum()) > 0.0 else -1
        if cur_rune != self._prev_rune_idx:
            if cur_rune >= 0:   # acquired/swapped to a rune — a decision worth grading
                health_now = float(obs.self_state[6]) if len(obs.self_state) > 6 else 100.0
                gain = (self._rune_fit(cur_rune, health_now)
                        - self._rune_fit(self._prev_rune_idx, health_now))
                rune_switch_delta = self.rune_switch_reward * gain
                bonus += rune_switch_delta
            self._prev_rune_idx = cur_rune

        # wire v5 self-exposure shaping: being seen by an enemy I can also
        # see is risk I should manage; having no enemy able to see me right
        # after I took damage means I found cover — pay both asymmetrically.
        self_exposure = float(
            np.clip(getattr(obs, "self_exposure", 0.0), 0.0, 1.0)
        )
        exposure_delta = 0.0
        exposed_while_visible = False
        covered_after_damage = False
        if self.lattice_reward_enabled:
            if visible_count > 0 and self_exposure > 0.0:
                exposed_while_visible = True
                exposure_delta -= self.exposure_penalty * self_exposure
            elif visible_count <= 0 and recent_damage and self_exposure < 1.0:
                covered_after_damage = True
                exposure_delta += self.cover_reward * (1.0 - self_exposure)
        bonus += exposure_delta

        info.update({
            "spatial_bonus": float(bonus),
            "ext_damage_prox": float(prox_dmg),
            "ext_offense": float(offense),
            "ext_survival": float(survival),
            "rune_switch": float(rune_switch_delta),
            "voxel_new": float(is_new),
            "voxel_visited": float(len(self.visited)),
            "voxel_cell_x": float(cell[0]),
            "voxel_cell_y": float(cell[1]),
            "voxel_cell_z": float(cell[2]),
            "visible_enemy": float(tactical_engagement),
            "combat_navigation_suppressed": float(navigation_suppressed),
            "entity_count": float(entity_count),
            "enemy_count": float(enemy_count),
            "enemy_visible_any": float(visible_count > 0),
            "enemy_visible_count": float(visible_count),
            "enemy_visible_nearest": float(nearest_visible if visible_count > 0 else 0.0),
            "enemy_visible_exposure_sum": float(
                fire_context["enemy_visible_exposure_sum"]
            ),
            "enemy_visible_exposure_max": float(
                fire_context["enemy_visible_exposure_max"]
            ),
            "aim_aligned": float(aim_aligned),
            "aim_yaw_error": float(aim_yaw_error),
            "aim_pitch_error": float(aim_pitch_error),
            "aim_tracking_quality": float(aim_tracking_quality),
            "target_hit_event": float(damage_dealt > 0.0),
            "target_hit_streak": float(
                hit_streak if damage_dealt > 0.0 else 0
            ),
            "target_repeat_hit_event": float(
                damage_dealt > 0.0 and hit_streak >= 2
            ),
            "target_kill_event": float(kills > 0.0),
            "memory_death_aversion": float(death_aversion),
            "memory_hazard_aversion": float(hazard_aversion),
            "fight_bias": float(fight_bias),
            "bias_margin": float(margin_term),
            "bias_exposure": float(exposure_term),
            "bias_hazard": float(-hazard_term),
            "bias_outnumbered": float(-outnumbered_term),
            "bias_escape": float(escape_term),
            "bias_engaging": float(engaging),
            "bad_fight_penalty": float(bad_fight_penalty_now),
            "disengage_reward": float(disengage_reward_now),
            "good_fight_reward": float(good_fight_reward_now),
            "episode_bad_fights": float(self.episode_bad_fights),
            "episode_good_disengages": float(self.episode_good_disengages),
            "self_exposure": float(self_exposure),
            "exposed_while_visible": float(exposed_while_visible),
            "covered_after_damage": float(covered_after_damage),
            "exposure_shaping": float(exposure_delta),
            "hook_required_near": float(hook_required_near),
            "hook_action": float(hook_action > 0),
            "hook_action_code": float(hook_action),
            "hook_discipline": float(hook_delta),
            "hook_enemy": float(hook_enemy),
            "hook_no_ammo_melee": float(hook_no_ammo_melee),
            "hook_blind": float(hook_blind),
            "hook_traversal": float(hook_traversal),
            "hook_overspeed": float(hook_overspeed),
            "hook_fire_action": float(hook_fired),
            "hook_noop_action": float(hook_noop),
            "hook_release_action": float(hook_released),
            "hook_release_overspeed": float(hook_release_overspeed),
            "hook_correction_available": float(hook_correction_available),
            "hook_correction_needed": float(correction_needed),
            "hook_correction_active": float(
                self._hook_correction_target is not None
            ),
            "hook_correction_started": float(hook_correction_started),
            "hook_correction_escape": float(hook_correction_report_escape),
            "hook_correction_progress": float(hook_correction_progress),
            "hook_correction_progress_reward": float(
                hook_correction_progress_delta
            ),
            "hook_correction_success": float(hook_correction_success),
            "hook_correction_timeout": float(hook_correction_timed_out),
            "hook_correction_heat": float(hook_correction_report_heat),
            "hook_correction_target_x": float(
                hook_correction_report_target[0]
                if hook_correction_report_target is not None else 0.0
            ),
            "hook_correction_target_y": float(
                hook_correction_report_target[1]
                if hook_correction_report_target is not None else 0.0
            ),
            "hook_correction_target_z": float(
                hook_correction_report_target[2]
                if hook_correction_report_target is not None else 0.0
            ),
            "hook_correction_anchor_x": float(
                hook_correction_report_anchor[0]
                if hook_correction_report_anchor is not None else 0.0
            ),
            "hook_correction_anchor_y": float(
                hook_correction_report_anchor[1]
                if hook_correction_report_anchor is not None else 0.0
            ),
            "hook_correction_anchor_z": float(
                hook_correction_report_anchor[2]
                if hook_correction_report_anchor is not None else 0.0
            ),
            "hook_correction_hot_x": float(
                hook_correction_report_hot_target[0]
                if hook_correction_report_hot_target is not None else 0.0
            ),
            "hook_correction_hot_y": float(
                hook_correction_report_hot_target[1]
                if hook_correction_report_hot_target is not None else 0.0
            ),
            "hook_correction_hot_z": float(
                hook_correction_report_hot_target[2]
                if hook_correction_report_hot_target is not None else 0.0
            ),
            "movement_speed": float(horizontal_speed),
            "movement_intent": float(movement_intent),
            "forward_intent": float(forward_intent),
            "backward_intent": float(backward_intent),
            "view_pitch_abs": float(movement["view_pitch_abs"]),
            "horizon_pitch_penalty": float(
                movement["horizon_pitch_penalty"]
            ),
            "level_aim_movement_reward": float(
                movement["level_aim_movement_reward"]
            ),
            "movement_nominal": float(nominal_speed),
            "movement_slow": float(slow_movement),
            "movement_overspeed": float(overspeed),
            "movement_discipline": float(movement_delta),
            "jump_action": float(jump_action),
            "jump_slow": float(jump_slow),
            "weapon_id": float(hook_context["weapon_id"]),
            "requested_weapon": float(hook_context["requested_weapon"]),
            "ammo": float(hook_context["ammo"]),
            "ammo_low": float(hook_context["ammo_low"]),
            "ammo_depleted": float(hook_context["ammo_depleted"]),
            "ammo_weapon_depleted": float(hook_context["ammo_weapon_depleted"]),
            "requested_ammo_weapon_unavailable": float(
                hook_context["requested_ammo_weapon_unavailable"]
            ),
            "fire": float(fired),
            "fire_discipline": float(fire_delta),
            "fire_unseen": float(fire_unseen),
            "fire_unaligned": float(fire_unaligned),
            "fire_no_ammo": float(fire_no_ammo),
            "fire_aligned": float(fire_aligned),
            "fire_splash": float(fire_splash),
            "fire_audio_contact": float(fire_audio_contact),
            **threat_info,
            "session_memory_bonus": float(memory_delta),
            "session_memory_cells": float(len(self._memory_for_map(self.map_name))),
            "session_current_engagement": float(current_engagement),
            "session_current_threat": float(current_threat),
            "session_current_opportunity": float(current_opportunity),
            "session_current_self_fire": float(current_self_fire),
            "session_nearest_engagement": float(nearest_engagement),
            "session_nearest_threat": float(nearest_threat),
            "session_nearest_opportunity": float(nearest_opportunity),
            "thermal_target_tracks": float(len(self._thermal_tracks)),
            "thermal_target_heat": float(self._thermal_last_signal[3]),
            "thermal_target_age": float(self._thermal_last_signal[4]),
            "lattice_prior_loaded": float(
                "lattice" in self.sidecar_sources.get(self.map_name, {})
            ),
            "lattice_routes_loaded": float(
                "routes" in self.sidecar_sources.get(self.map_name, {})
            ),
            "lattice_dynamic_cells": float(
                len(self.dynamic_cells.get(self.map_name, ()))
            ),
            "lattice_route_active": float(bool(self.selected_route)),
        })
        return float(bonus), info

    def finalize_episode(self, terminal_reason: int = 0, truncated: bool = False):
        """Return end-of-episode survival/win shaping and diagnostics."""
        damage_margin = self.episode_damage_dealt - self.episode_damage_taken
        frag_margin = self.episode_kills - self.episode_deaths
        meaningful_contact = (
            self.episode_contact_events > 0.0 or
            self.episode_damage_dealt + self.episode_damage_taken >= self.meaningful_contact_min or
            self.episode_kills > 0.0 or
            self.episode_deaths > 0.0
        )

        outcome_delta = 0.0
        outcome_win = False
        outcome_survival = False
        outcome_loss = False
        outcome_idle = False

        if self.episode_deaths > 0.0:
            outcome_loss = True
            outcome_delta -= self.episode_loss_penalty * self.episode_deaths
        elif frag_margin > 0.0 or damage_margin >= self.win_damage_margin:
            outcome_win = True
            outcome_delta += self.episode_win_reward
        elif meaningful_contact and damage_margin >= 0.0:
            outcome_survival = True
            outcome_delta += self.episode_survival_reward
        elif meaningful_contact:
            outcome_loss = True
            loss_scale = min(1.0, abs(damage_margin) / max(1.0, self.win_damage_margin))
            outcome_delta -= self.episode_loss_penalty * loss_scale
        else:
            outcome_idle = True
            outcome_delta -= self.episode_idle_penalty

        if frag_margin > 0.0:
            outcome_delta += self.frag_advantage_reward * frag_margin
        elif frag_margin < 0.0:
            outcome_delta -= self.frag_disadvantage_penalty * abs(frag_margin)

        # Damage is a LENS, not a target. The old code rewarded raw cumulative
        # damage_margin (unbounded HP) × a per-hitpoint weight → a -108/episode
        # term that dwarfed the whole reward system (see tools/reward_graph.py).
        # Replace with the bounded *fraction of the exchange won* ∈ [-1, 1], so
        # out-damaging matters but can never swamp the frag-based outcome.
        total_dmg = self.episode_damage_dealt + self.episode_damage_taken
        if total_dmg > 1.0:
            exchange_q = (self.episode_damage_dealt - self.episode_damage_taken) / total_dmg
            outcome_delta += self.exchange_quality_reward * exchange_q

        info = {
            "outcome_sample": 1.0,
            "outcome_bonus": float(outcome_delta),
            "outcome_win": float(outcome_win),
            "outcome_survival": float(outcome_survival),
            "outcome_loss": float(outcome_loss),
            "outcome_idle": float(outcome_idle),
            "episode_damage_margin": float(damage_margin),
            "episode_frag_margin": float(frag_margin),
            "episode_contact_events": float(self.episode_contact_events),
            "episode_steps_tracked": float(self.episode_steps),
            "episode_truncated": float(truncated),
            "episode_terminal_reason": float(terminal_reason),
        }
        return float(outcome_delta), info

    def memory_features(self, obs: Observation) -> np.ndarray:
        """Public, gated view of the memory tail for the policy input.

        Role gates are applied here, at the consumption point, never at
        storage: Q2_LATTICE_OBS=0 zeroes the whole tail, and
        Q2_LATTICE_IMMEDIATE=0 zeroes only the immediate-engagement slice
        (5:9) regardless of producer. Reward computation uses the ungated
        internal vector so an obs ablation never silently changes reward
        semantics mid-run.
        """
        features = self._memory_features_internal(obs)
        if not self.lattice_obs_enabled:
            return np.zeros(OBS_SESSION_MEMORY_DIM, dtype=np.float32)
        if not self.immediate_engagement_enabled:
            features = features.copy()
            features[IMMEDIATE_ENGAGEMENT_SLICE] = 0.0
        return features

    def _memory_features_internal(self, obs: Observation) -> np.ndarray:
        """Compact non-decaying session memory features for the policy input.
        Last 3 slots are the survivability projection (always set)."""
        self._update_thermal_tracks(obs)
        cache_key = (self.map_name or "unknown", int(obs.tick))
        if self._feature_cache_key == cache_key and self._feature_cache is not None:
            return self._feature_cache

        if self.session_memory_enabled and self.rust_lattice_enabled:
            try:
                rust_index = self._flush_rust_index()
                if rust_index is not None:
                    features, nearest_deaths = rust_index.features_with_deaths(
                        self, obs
                    )
                    self._feature_nearest_deaths = float(nearest_deaths)
                    return self._finish_memory_features(
                        obs, features, cache_key
                    )
            except (RuntimeError, ValueError, AttributeError) as error:
                self._rust_fallback_reason = str(error)
                self.rust_lattice_enabled = False

        features = np.zeros(OBS_SESSION_MEMORY_DIM, dtype=np.float32)
        nearest = {
            kind: (0.0, 0.0, 0.0, 0.0)
            for kind in (
                "engagement", "threat", "opportunity", "self_fire",
                "deaths", "hazard",
            )
        }
        if self.session_memory_enabled:
            nearest = self._nearest_memory_signals(obs, tuple(nearest))
        self._feature_nearest_deaths = float(nearest["deaths"][3])
        self._feature_nearest_hazard = float(nearest["hazard"][3])

        # Survivability projection (slots 21-23) — always computed so the bot
        # perceives "will I win this race" every step, not just when memory
        # exists. map_help = nearby recovery opportunity (extends survival).
        map_help = float(nearest["opportunity"][3])
        wm, ehp_n, dps_sh = self._win_margin(obs, map_help)
        features[21] = wm
        features[22] = ehp_n
        features[23] = dps_sh

        if not self.session_memory_enabled:
            return self._finish_memory_features(obs, features, cache_key)

        memory = self._memory_for_map(self.map_name)
        if not memory:
            return self._finish_memory_features(obs, features, cache_key)

        cell = self.cell_for(obs)
        current = memory.get(cell)
        if current is not None:
            features[0] = self._norm_memory_score(self._engagement_score(current))
            features[1] = self._norm_memory_score(self._threat_score(current))
            features[2] = self._norm_memory_score(self._opportunity_score(current))
            features[3] = self._norm_memory_score(current.self_fire)
            features[4] = self._memory_confidence(current)

        for offset, kind in (
            (5, "engagement"),
            (9, "threat"),
            (13, "opportunity"),
            (17, "self_fire"),
        ):
            dx, dy, dz, score = nearest[kind]
            features[offset:offset + 4] = (dx, dy, dz, score)

        return self._finish_memory_features(obs, features, cache_key)

    def _finish_memory_features(
        self,
        obs: Observation,
        features: np.ndarray,
        cache_key: Tuple[str, int],
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32).copy()
        dx, dy, dz, heat, age = self._thermal_target_signal(obs)
        self._thermal_last_signal = (dx, dy, dz, heat, age)
        if heat > 0.0:
            # Live/cooling hostile occupancy owns the immediate engagement
            # pull. Persistent map knowledge resumes automatically at expiry.
            features[IMMEDIATE_ENGAGEMENT_SLICE] = (dx, dy, dz, heat)
        self._feature_cache_key = cache_key
        self._feature_cache = features
        return features

    def _memory_for_map(
        self, map_name: str
    ) -> Dict[Tuple[int, int, int], SessionMemoryCell]:
        key = map_name or "unknown"
        return self.session_memories.setdefault(key, {})

    def _memory_cell(
        self, cell: Tuple[int, int, int], tick: int
    ) -> SessionMemoryCell:
        memory = self._memory_for_map(self.map_name)
        entry = memory.get(cell)
        if entry is None:
            entry = SessionMemoryCell()
            memory[cell] = entry
        entry.last_tick = int(tick)
        self._mark_cell_col_dirty(cell)
        self._prune_memory(memory)
        return entry

    def _prune_memory(
        self, memory: Dict[Tuple[int, int, int], SessionMemoryCell]
    ) -> None:
        limit = max(64, int(self.session_memory_limit))
        if len(memory) <= limit:
            return
        overflow = len(memory) - limit
        victims = sorted(
            memory.items(),
            key=lambda item: (item[1].samples, item[1].last_tick),
        )[:overflow]
        for cell, _entry in victims:
            memory.pop(cell, None)
            self._mark_rust_removed(cell)
        if victims:
            # Row removal would reorder the column cache; rebuild it.
            self._invalidate_cell_col_cache(self.map_name)

    def _update_session_memory(
        self,
        obs: Observation,
        cell: Tuple[int, int, int],
        visible_count: int,
        fired: bool,
        hook_enemy: bool,
        fire_audio_contact: bool,
        audio_contact: bool,
    ) -> None:
        if not self.session_memory_enabled:
            return

        tick = int(obs.tick)
        damage_dealt = max(0.0, float(obs.reward_damage_dealt))
        damage_taken = max(0.0, float(obs.reward_damage_taken))
        kills = max(0.0, float(obs.reward_kill))
        deaths = max(0.0, float(obs.reward_death))
        items = max(0.0, float(obs.reward_item_pickup))
        # wire v5 threat de-conflation: the reward channels are per-tick
        # deltas, so the MOD read in this same packet attributes them. A
        # stale/zero MOD falls back to the combat channel — never lose a
        # damage or death event.
        env_damage = (
            damage_taken > 0.0
            and int(getattr(obs, "last_damage_mod", 0)) in ENVIRONMENTAL_MODS
        )
        env_death = (
            deaths > 0.0
            and int(getattr(obs, "last_death_mod", 0)) in ENVIRONMENTAL_MODS
        )
        combat_damage_taken = 0.0 if env_damage else damage_taken
        combat_deaths = 0.0 if env_death else deaths
        visible_contact = visible_count > 0
        exposure_sum, _exposure_max = self._visible_enemy_exposure(obs)
        contact = (
            visible_contact or
            audio_contact or
            damage_dealt > 0.0 or
            damage_taken > 0.0 or
            kills > 0.0 or
            deaths > 0.0
        )

        if contact or fired or hook_enemy or items > 0.0:
            here = self._memory_cell(cell, tick)
            if contact:
                here.engagement_count += 1.0
            if visible_contact:
                here.enemy_seen += exposure_sum
            if fired:
                here.self_fire += 1.0
            if hook_enemy:
                here.hook_engagement += 1.0
            if items > 0.0 and contact:
                here.item_contested += items
            here.damage_dealt += damage_dealt
            here.hazard_damage += damage_taken - combat_damage_taken
            here.damage_taken += combat_damage_taken
            here.kills += kills
            here.hazard_deaths += deaths - combat_deaths
            here.deaths += combat_deaths
            self._mark_rust_score_event(
                cell,
                engagement=(
                    (1.0 if contact else 0.0)
                    + 0.35 * exposure_sum
                    + (0.50 if hook_enemy else 0.0)
                    + (0.35 * items if contact else 0.0)
                ),
                threat=(
                    0.15 * exposure_sum
                    + 0.03 * combat_damage_taken
                    + 3.0 * combat_deaths
                ),
                opportunity=(
                    (0.45 if hook_enemy else 0.0)
                    + (0.45 * items if contact else 0.0)
                    + 0.03 * damage_dealt
                    + 3.0 * kills
                ),
                self_fire=(1.0 if fired else 0.0),
                deaths=3.0 * combat_deaths,
                samples=(
                    (1.0 if contact else 0.0)
                    + exposure_sum
                    + (1.0 if fired else 0.0)
                    + (1.0 if hook_enemy else 0.0)
                    + (items if contact else 0.0)
                    + kills
                    + deaths
                ),
            )

        if visible_contact:
            for enemy_cell, exposure in self._visible_enemy_heat_samples(obs):
                seen = self._memory_cell(enemy_cell, tick)
                seen.engagement_count += 1.0
                seen.enemy_seen += exposure
                seen.damage_dealt += damage_dealt / max(1, visible_count)
                seen.kills += kills / max(1, visible_count)
                if fire_audio_contact:
                    seen.self_fire += 0.25
                dealt_share = damage_dealt / max(1, visible_count)
                kill_share = kills / max(1, visible_count)
                self._mark_rust_score_event(
                    enemy_cell,
                    engagement=1.0 + 0.35 * exposure,
                    threat=0.15 * exposure,
                    opportunity=0.03 * dealt_share + 3.0 * kill_share,
                    self_fire=(0.25 if fire_audio_contact else 0.0),
                    samples=(
                        1.0 + exposure
                        + kill_share
                        + (0.25 if fire_audio_contact else 0.0)
                    ),
                )

        if self.last_visible_count > 0 and visible_count == 0:
            self._memory_cell(cell, tick).enemy_lost += 1.0
            self._mark_rust_score_event(
                cell, engagement=0.25, threat=0.20, samples=1.0
            )

        if damage_taken > 0.0:
            self.last_damage_tick = tick
            self.last_damage_cell = cell
        elif (
            self.last_damage_cell is not None and
            visible_count == 0 and
            0 <= tick - self.last_damage_tick <= 120 and
            cell != self.last_damage_cell
        ):
            self._memory_cell(cell, tick).successful_escape += 1.0
            self._mark_rust_score_event(cell, opportunity=0.20, samples=1.0)
            self.last_damage_cell = None

        self.last_visible_count = visible_count

    @staticmethod
    def _local_to_world_vector(
        local: np.ndarray, yaw_deg: float, pitch_deg: float
    ) -> np.ndarray:
        """Invert Quake AngleVectors' forward/Quake-right/up projection."""
        yaw = np.deg2rad(float(yaw_deg))
        pitch = np.deg2rad(float(pitch_deg))
        sy, cy = np.sin(yaw), np.cos(yaw)
        sp, cp = np.sin(pitch), np.cos(pitch)
        forward = np.array((cp * cy, cp * sy, -sp), dtype=np.float32)
        right = np.array((sy, -cy, 0.0), dtype=np.float32)
        up = np.array((sp * cy, sp * sy, cp), dtype=np.float32)
        vector = np.asarray(local, dtype=np.float32)
        return forward * vector[0] + right * vector[1] + up * vector[2]

    def _target_world_point(self, obs: Observation, ent: np.ndarray) -> np.ndarray:
        # The C target contract is camera-eye-relative. Quake II viewheight is
        # 22 standing and -2 ducked; muzzle clearance is checked separately.
        eye = np.asarray(obs.self_state[:3], dtype=np.float32).copy()
        flags = 0
        if hasattr(obs, "self_debug") and len(obs.self_debug) >= 4:
            flags = int(obs.self_debug[3])
        eye[2] += -2.0 if flags & ML_ENTITY_DUCKED else 22.0
        return eye + self._local_to_world_vector(
            ent[:3], getattr(obs, "yaw", 0.0), getattr(obs, "pitch", 0.0)
        )

    def _update_thermal_tracks(self, obs: Observation) -> None:
        if not self.thermal_target_enabled:
            self._thermal_tracks.clear()
            self._thermal_last_tick = int(obs.tick)
            return

        tick = int(obs.tick)
        if tick == self._thermal_last_tick:
            return
        if tick < self._thermal_last_tick or float(obs.self_state[6]) <= 0.0:
            self._thermal_tracks.clear()
        if float(getattr(obs, "reward_kill", 0.0)) > 0.0:
            # wire v5 per-target hit attribution: the kill tick names the
            # victim as (edict_index, 14-bit life epoch) — the same packing
            # entity_debug uses for track keys — so only the dead hostile's
            # track clears and other tracks keep cooling. Attribution of
            # zero means the server could not name the target; clearing all
            # tracks is still safer than retaining a dead hostile as
            # actionable thermal evidence.
            kill_edict = int(getattr(obs, "last_hit_target_edict", 0))
            kill_epoch = int(getattr(obs, "last_hit_target_epoch", 0))
            if kill_edict > 0:
                self._thermal_tracks.pop(
                    (kill_edict << 14) | (kill_epoch & 0x3FFF), None
                )
            else:
                self._thermal_tracks.clear()

        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        for index, ent in enumerate(obs.entities[:count]):
            exposure = abs(float(ent[8]))
            if not (ent[6] > 0.0 and ent[7] > 0.5 and exposure > 0.0):
                continue
            target_id = index + 1
            if (
                hasattr(obs, "entity_debug")
                and index < obs.entity_debug.shape[0]
                and int(obs.entity_debug[index, 0]) > 0
            ):
                debug = obs.entity_debug[index]
                edict_index = int(debug[0])
                life_epoch = (
                    int(debug[3]) & ML_ENTITY_EPOCH_MASK
                ) >> ML_ENTITY_EPOCH_SHIFT
                target_id = (edict_index << 14) | life_epoch
                for previous_id in tuple(self._thermal_tracks):
                    if (
                        previous_id != target_id
                        and previous_id >> 14 == edict_index
                    ):
                        self._thermal_tracks.pop(previous_id, None)
            rel_velocity = self._local_to_world_vector(
                ent[3:6], getattr(obs, "yaw", 0.0),
                getattr(obs, "pitch", 0.0)
            )
            target_velocity = (
                rel_velocity + np.asarray(obs.self_state[3:6], dtype=np.float32)
            )
            self._thermal_tracks[target_id] = ThermalTargetTrack(
                target_id=target_id,
                world_point=self._target_world_point(obs, ent),
                world_velocity=target_velocity,
                exposure=float(np.clip(exposure, 0.0, 1.0)),
                last_tick=tick,
            )

        max_age = max(0, int(self.thermal_target_max_age_ticks))
        self._thermal_tracks = {
            target_id: track
            for target_id, track in self._thermal_tracks.items()
            if 0 <= tick - track.last_tick <= max_age
        }
        self._thermal_last_tick = tick

    def _thermal_target_signal(
        self, obs: Observation
    ) -> Tuple[float, float, float, float, float]:
        if not self.thermal_target_enabled or not self._thermal_tracks:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        tick = int(obs.tick)
        tick_rate = max(1.0, float(self.lattice_tick_rate))
        decay = float(np.clip(self.thermal_target_decay, 0.0, 1.0))
        voxel_size = max(1.0, float(self.thermal_target_voxel_size))
        origin = np.asarray(obs.self_state[:3], dtype=np.float32)
        best = None
        best_rank = -1.0
        for target_id in sorted(self._thermal_tracks):
            track = self._thermal_tracks[target_id]
            age = max(0, tick - track.last_tick)
            predicted = track.world_point + track.world_velocity * (
                float(age) / tick_rate
            )
            cell = np.floor(predicted / voxel_size)
            center = (cell + 0.5) * voxel_size
            delta = center - origin
            distance = float(np.linalg.norm(delta))
            confidence = (0.5 + 0.5 * track.exposure) * (decay ** age)
            rank = confidence / max(1.0, distance / voxel_size)
            if rank > best_rank:
                best_rank = rank
                best = (delta, confidence, float(age))

        if best is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        delta, confidence, age = best
        radius = max(float(self.session_memory_search_radius), voxel_size)
        direction = np.clip(delta / radius, -1.0, 1.0)
        return (
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
            float(confidence),
            age,
        )

    def _visible_enemy_heat_samples(self, obs: Observation):
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        for ent in obs.entities[:count]:
            if not (ent[7] > 0.5 and abs(float(ent[8])) > 0.0):
                continue
            yield (
                self.cell_for_pos(self._target_world_point(obs, ent)),
                float(np.clip(abs(float(ent[8])), 0.0, 1.0)),
            )

    def _visible_enemy_cells(self, obs: Observation):
        for cell, _exposure in self._visible_enemy_heat_samples(obs):
            yield cell

    def apply_directive(self, map_name: str, action: str,
                        x: float, y: float, z: float,
                        strength: float = 3.0) -> bool:
        """External (LLM-coach) steering: write directly into this bot's
        session memory so the existing gradients change its behavior.
          avoid  -> deaths      (death-aversion repulsion: 'DO NOT GO THERE')
          seek   -> kills       (opportunity pull: 'value/target is here')
          engage -> engagement  (combat-anticipation pull)
          danger -> damage_taken (threat shading without full aversion)
        """
        # External writes bypass gameplay validation, so they are gated
        # (Q2_LATTICE_DIRECTIVES) and logged like sidecar attestation.
        if not self.directives_enabled:
            log.warning(
                "lattice directive REJECTED (directives disabled): "
                "map=%s action=%s pos=(%.0f,%.0f,%.0f)",
                map_name or self.map_name, action, x, y, z,
            )
            return False
        target = map_name or self.map_name
        memory = self._memory_for_map(target)
        cell = self.cell_for_pos((float(x), float(y), float(z)))
        entry = memory.get(cell)
        if entry is None:
            entry = SessionMemoryCell()
            memory[cell] = entry
        self._mark_cell_col_dirty(cell, target)
        amt = max(0.0, float(strength))
        if action == "avoid":
            entry.deaths += amt
        elif action == "seek":
            entry.kills += amt
        elif action == "engage":
            entry.engagement_count += amt
        elif action == "danger":
            entry.damage_taken += amt * 33.0
        else:
            return False
        log.info(
            "lattice directive applied: map=%s action=%s cell=%s strength=%.2f",
            target, action, cell, amt,
        )
        if target == self.map_name:
            self._invalidate_feature_cache()
        self._mark_rust_dirty(cell, target)
        return True

    def _nearest_memory_signal(
        self, obs: Observation, kind: str
    ) -> Tuple[float, float, float, float]:
        return self._nearest_memory_signals(obs, (kind,))[kind]

    def _nearest_memory_signals(
        self, obs: Observation, kinds: Tuple[str, ...]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Find every requested channel, fast path first.

        The fast path is required to be bit-identical to the pure-Python
        traversal: numpy only shortlists finalist cells, and the finalists
        are rescored with the exact slow-loop semantics. Q2_NEAREST_VERIFY=1
        cross-checks every call; Q2_FAST_NEAREST=0 forces the traversal.
        """
        if not self.fast_nearest_enabled:
            return self._nearest_memory_signals_slow(obs, kinds)
        if self.nearest_verify:
            fast = self._nearest_memory_signals_fast(obs, kinds)
            slow = self._nearest_memory_signals_slow(obs, kinds)
            if fast != slow:
                raise AssertionError(
                    f"fast nearest-signal path diverged: {fast} != {slow}"
                )
            return fast
        return self._nearest_memory_signals_fast(obs, kinds)

    def _mark_cell_col_dirty(
        self, cell: Tuple[int, int, int], map_name: Optional[str] = None
    ) -> None:
        key = map_name or self.map_name or "unknown"
        normalized = tuple(int(value) for value in cell)
        self._cell_col_dirty.setdefault(key, {})[normalized] = None

    def _invalidate_cell_col_cache(self, map_name: str) -> None:
        key = map_name or "unknown"
        self._cell_col_cache.pop(key, None)
        self._cell_col_dirty.pop(key, None)

    def _cell_columns(self, map_name: str) -> dict:
        """Numpy mirror of the session-memory dict, updated by dirty rows."""
        memory = self._memory_for_map(map_name)
        cache = self._cell_col_cache.get(map_name)
        if cache is None:
            keys = list(memory.keys())
            cells = (
                np.asarray(keys, dtype=np.float64).reshape(-1, 3)
                if keys else np.zeros((0, 3), dtype=np.float64)
            )
            cols = np.zeros((len(keys), len(_CELL_COL_FIELDS)), dtype=np.float64)
            for row, cell in enumerate(keys):
                entry = memory[cell]
                cols[row] = [
                    float(getattr(entry, name)) for name in _CELL_COL_FIELDS
                ]
            cache = {
                "keys": keys,
                "row_of": {cell: row for row, cell in enumerate(keys)},
                "cells": cells,
                "cols": cols,
            }
            self._cell_col_cache[map_name] = cache
            self._cell_col_dirty.pop(map_name, None)
            return cache

        dirty = self._cell_col_dirty.pop(map_name, None)
        if dirty:
            keys = cache["keys"]
            row_of = cache["row_of"]
            new_cells = []
            new_rows = []
            for cell in dirty:
                entry = memory.get(cell)
                if entry is None:
                    # Pruned cells invalidate the whole cache via
                    # _prune_memory; a missing entry here means that
                    # rebuild already covered this row.
                    continue
                values = [float(getattr(entry, name))
                          for name in _CELL_COL_FIELDS]
                row = row_of.get(cell)
                if row is None:
                    row_of[cell] = len(keys)
                    keys.append(cell)
                    new_cells.append(cell)
                    new_rows.append(values)
                else:
                    cache["cols"][row] = values
            if new_cells:
                cache["cells"] = np.vstack([
                    cache["cells"],
                    np.asarray(new_cells, dtype=np.float64).reshape(-1, 3),
                ])
                cache["cols"] = np.vstack([
                    cache["cols"],
                    np.asarray(new_rows, dtype=np.float64),
                ])
        return cache

    def _exact_cell_rank(
        self,
        entry: SessionMemoryCell,
        cell: Tuple[int, int, int],
        px: float, py: float, pz: float,
        radius: float, voxel_size: float,
        kind: str,
    ) -> Optional[Tuple[float, float]]:
        """One slow-loop iteration for one cell; None when it cannot win."""
        cx = (float(cell[0]) + 0.5) * voxel_size
        cy = (float(cell[1]) + 0.5) * voxel_size
        cz = (float(cell[2]) + 0.5) * voxel_size
        dist = float(hypot(cx - px, cy - py, cz - pz))
        if dist > radius:
            return None
        raw_score = self._memory_score(entry, kind)
        if raw_score <= 0.0:
            return None
        confidence = self._memory_confidence(entry)
        distance_scale = max(1.0, dist / voxel_size)
        score = self._norm_memory_score(raw_score) * confidence
        return score / distance_scale, score

    def _col_raw_score(self, cols: np.ndarray, kind: str) -> np.ndarray:
        """Vectorized _memory_score; elementwise op order matches Python."""
        fi = _CELL_COL_INDEX
        if kind == "engagement":
            return (
                cols[:, fi["engagement_count"]]
                + 0.35 * cols[:, fi["enemy_seen"]]
                + 0.25 * cols[:, fi["enemy_lost"]]
                + 0.50 * cols[:, fi["hook_engagement"]]
                + 0.35 * cols[:, fi["item_contested"]]
            )
        if kind == "threat":
            return (
                0.03 * cols[:, fi["damage_taken"]]
                + 3.0 * cols[:, fi["deaths"]]
                + 0.15 * cols[:, fi["enemy_seen"]]
                + 0.20 * cols[:, fi["enemy_lost"]]
                + cols[:, fi["prior_threat"]]
                + np.maximum(0.0, -cols[:, fi["readiness"]])
                + self.bad_fight_threat_weight * cols[:, fi["bad_fight_taken"]]
            )
        if kind == "opportunity":
            return (
                0.03 * cols[:, fi["damage_dealt"]]
                + 3.0 * cols[:, fi["kills"]]
                + 0.45 * cols[:, fi["hook_engagement"]]
                + 0.45 * cols[:, fi["item_contested"]]
                + 0.20 * cols[:, fi["successful_escape"]]
                + cols[:, fi["prior_opportunity"]]
                + np.maximum(0.0, cols[:, fi["readiness"]])
                + np.maximum(0.0, cols[:, fi["route_bias"]])
            )
        if kind == "self_fire":
            return cols[:, fi["self_fire"]].astype(np.float64)
        if kind == "deaths":
            return 3.0 * cols[:, fi["deaths"]]
        if kind == "hazard":
            return (
                0.03 * cols[:, fi["hazard_damage"]]
                + 3.0 * cols[:, fi["hazard_deaths"]]
            )
        return np.zeros(cols.shape[0], dtype=np.float64)

    def _col_confidence(self, cols: np.ndarray) -> np.ndarray:
        """Vectorized _memory_confidence estimate (finalists recomputed)."""
        fi = _CELL_COL_INDEX
        samples = (
            cols[:, fi["engagement_count"]]
            + cols[:, fi["enemy_seen"]]
            + cols[:, fi["enemy_lost"]]
            + cols[:, fi["kills"]]
            + cols[:, fi["deaths"]]
            + cols[:, fi["hazard_deaths"]]
            + cols[:, fi["hazard_damage"]] / 100.0
            + 0.5 * cols[:, fi["bad_fight_taken"]]
            + 0.5 * cols[:, fi["good_disengage"]]
            + cols[:, fi["self_fire"]]
            + cols[:, fi["hook_engagement"]]
            + cols[:, fi["item_contested"]]
            + cols[:, fi["successful_escape"]]
            + np.abs(cols[:, fi["prior_opportunity"]])
            + np.abs(cols[:, fi["prior_threat"]])
        )
        learned = np.minimum(
            1.0, np.log1p(np.maximum(0.0, samples)) / log1p(24.0)
        )
        known_map_signal = (
            (np.abs(cols[:, fi["prior_opportunity"]]) > 0.0)
            | (np.abs(cols[:, fi["prior_threat"]]) > 0.0)
            | (np.abs(cols[:, fi["readiness"]]) > 0.0)
            | (np.abs(cols[:, fi["route_bias"]]) > 0.0)
        )
        return np.maximum(learned, np.where(known_map_signal, 1.0, 0.0))

    def _nearest_memory_signals_fast(
        self, obs: Observation, kinds: Tuple[str, ...]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        zero = (0.0, 0.0, 0.0, 0.0)
        results = {kind: zero for kind in kinds}
        memory = self._memory_for_map(self.map_name)
        if not memory:
            return results

        pos = obs.self_state[:3]
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        radius = max(self.session_memory_search_radius, self.voxel_size)
        voxel_size = max(self.voxel_size, 1.0)
        scale = max(0.1, float(self.session_memory_score_scale))

        bundle = self._cell_columns(self.map_name)
        if not bundle["keys"]:
            return results
        cells = bundle["cells"]
        cols = bundle["cols"]
        centers = (cells + 0.5) * voxel_size
        delta = centers - np.asarray([px, py, pz], dtype=np.float64)
        # dist/confidence here are estimates (numpy libm differs by ~1 ulp);
        # they only shortlist finalists. Generous margins keep every cell
        # that could win under the exact semantics, and the exact rescore
        # below applies the real filters.
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        within = dist <= radius * (1.0 + 1e-9) + 1e-9
        confidence = self._col_confidence(cols)

        for kind in kinds:
            raw = self._col_raw_score(cols, kind)
            eligible = within & (raw > 0.0)
            if not bool(eligible.any()):
                continue
            score = np.tanh(np.maximum(0.0, raw) / scale) * confidence
            rank = np.where(
                eligible, score / np.maximum(1.0, dist / voxel_size), -1.0
            )
            top_rank = float(np.max(rank))
            if top_rank < 0.0:
                continue
            finalists = np.nonzero(
                rank >= top_rank * (1.0 - 1e-9) - 1e-12
            )[0]
            best_rank = -1.0
            best_row = -1
            best_score = 0.0
            # np.nonzero order is row order == dict insertion order, so the
            # strict-greater scan reproduces the slow loop's tie-breaking.
            for row in finalists:
                cell = bundle["keys"][row]
                exact = self._exact_cell_rank(
                    memory[cell], cell, px, py, pz, radius, voxel_size, kind
                )
                if exact is None:
                    continue
                exact_rank, exact_score = exact
                if exact_rank > best_rank:
                    best_rank = exact_rank
                    best_row = int(row)
                    best_score = exact_score
            if best_row < 0:
                continue
            center = centers[best_row]
            results[kind] = (
                float(np.clip((center[0] - px) / radius, -1.0, 1.0)),
                float(np.clip((center[1] - py) / radius, -1.0, 1.0)),
                float(np.clip((center[2] - pz) / radius, -1.0, 1.0)),
                float(best_score),
            )
        return results

    def _nearest_memory_signals_slow(
        self, obs: Observation, kinds: Tuple[str, ...]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Find every requested channel in one traversal of the voxel map."""
        zero = (0.0, 0.0, 0.0, 0.0)
        results = {kind: zero for kind in kinds}
        memory = self._memory_for_map(self.map_name)
        if not memory:
            return results

        pos = obs.self_state[:3]
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        radius = max(self.session_memory_search_radius, self.voxel_size)
        voxel_size = max(self.voxel_size, 1.0)
        best = {
            kind: {"rank": -1.0, "score": 0.0, "center": None}
            for kind in kinds
        }
        for cell, entry in memory.items():
            cx = (float(cell[0]) + 0.5) * voxel_size
            cy = (float(cell[1]) + 0.5) * voxel_size
            cz = (float(cell[2]) + 0.5) * voxel_size
            dist = float(hypot(cx - px, cy - py, cz - pz))
            if dist > radius:
                continue
            confidence = self._memory_confidence(entry)
            distance_scale = max(1.0, dist / voxel_size)
            for kind in kinds:
                raw_score = self._memory_score(entry, kind)
                if raw_score <= 0.0:
                    continue
                score = self._norm_memory_score(raw_score) * confidence
                rank = score / distance_scale
                if rank > best[kind]["rank"]:
                    best[kind] = {
                        "rank": rank,
                        "score": score,
                        "center": (cx, cy, cz),
                    }

        for kind, candidate in best.items():
            center = candidate["center"]
            if center is None:
                continue
            results[kind] = (
                float(np.clip((center[0] - px) / radius, -1.0, 1.0)),
                float(np.clip((center[1] - py) / radius, -1.0, 1.0)),
                float(np.clip((center[2] - pz) / radius, -1.0, 1.0)),
                float(candidate["score"]),
            )
        return results

    def _cell_center(self, cell: Tuple[int, int, int]) -> np.ndarray:
        size = max(self.voxel_size, 1.0)
        return np.array(
            [(float(v) + 0.5) * size for v in cell],
            dtype=np.float32,
        )

    def _memory_score(self, entry: SessionMemoryCell, kind: str) -> float:
        if kind == "engagement":
            return self._engagement_score(entry)
        if kind == "threat":
            return self._threat_score(entry)
        if kind == "opportunity":
            return self._opportunity_score(entry)
        if kind == "self_fire":
            return float(entry.self_fire)
        if kind == "deaths":
            return 3.0 * float(entry.deaths)
        if kind == "hazard":
            return 0.03 * float(entry.hazard_damage) + 3.0 * float(
                entry.hazard_deaths
            )
        return 0.0

    def _engagement_score(self, entry: SessionMemoryCell) -> float:
        return (
            entry.engagement_count
            + 0.35 * entry.enemy_seen
            + 0.25 * entry.enemy_lost
            + 0.50 * entry.hook_engagement
            + 0.35 * entry.item_contested
        )

    def _threat_score(self, entry: SessionMemoryCell) -> float:
        return (
            0.03 * entry.damage_taken
            + 3.0 * entry.deaths
            + 0.15 * entry.enemy_seen
            + 0.20 * entry.enemy_lost
            + entry.prior_threat
            + max(0.0, -entry.readiness)
            # Decision-quality memory: cells where I kept taking (and dying
            # in) fights the bias said to avoid read as slightly more
            # threatening. This is the ONLY read path for bad_fight_taken;
            # the 24-float policy tail layout is frozen.
            + self.bad_fight_threat_weight * entry.bad_fight_taken
        )

    def _opportunity_score(self, entry: SessionMemoryCell) -> float:
        return (
            0.03 * entry.damage_dealt
            + 3.0 * entry.kills
            + 0.45 * entry.hook_engagement
            + 0.45 * entry.item_contested
            + 0.20 * entry.successful_escape
            + entry.prior_opportunity
            + max(0.0, entry.readiness)
            + max(0.0, entry.route_bias)
        )

    def _norm_memory_score(self, value: float) -> float:
        scale = max(0.1, float(self.session_memory_score_scale))
        return float(np.tanh(max(0.0, float(value)) / scale))

    def _memory_confidence(self, entry: SessionMemoryCell) -> float:
        learned = min(1.0, log1p(max(0.0, entry.samples)) / log1p(24.0))
        known_map_signal = (
            abs(entry.prior_opportunity) > 0.0
            or abs(entry.prior_threat) > 0.0
            or abs(entry.readiness) > 0.0
            or abs(entry.route_bias) > 0.0
        )
        return float(max(learned, 1.0 if known_map_signal else 0.0))

    def _reset_episode_state(self) -> None:
        # Per-life aggression temperament — the symmetry-breaker. Two clones in
        # an even (chicken) standoff draw different leans, so one presses and
        # one swerves instead of mirroring into deadlock. Independent per venv,
        # which is exactly the diversity we want.
        self.aggression = self.rng.uniform(-self.exchange_aggression_mag,
                                           self.exchange_aggression_mag)
        self._dps_self = 0.0
        self._dps_enemy = 0.0
        self._prev_rune_idx = -1
        self.episode_steps = 0
        self.episode_damage_dealt = 0.0
        self.episode_damage_taken = 0.0
        self.episode_kills = 0.0
        self.episode_deaths = 0.0
        self.episode_bad_fights = 0.0
        self.episode_good_disengages = 0.0
        self.episode_contact_events = 0.0
        self.recent_threat_steps = 0

    # rune_flags order (engine ml_obs.c): resist, strength, haste, regen, vampire
    _RUNE_AXIS = ("survival", "offense", "offense", "survival", "value")
    NOMINAL_DPS = 10.0   # fallback per-step damage capability when not measured

    def _nearest_enemy_health(self, obs) -> float:
        """HP of the nearest visible enemy (normalized obs is raw here), else 0."""
        best_d, best_h = 1e9, 0.0
        n = min(int(obs.entity_count), obs.entities.shape[0])
        for k in range(n):
            e = obs.entities[k]
            if e[7] < 0.5 or abs(float(e[8])) <= 0.0:
                continue
            d = float(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
            if d < best_d:
                best_d, best_h = d, float(e[6])
        return best_h

    def _win_margin(self, obs, map_help: float = 0.0):
        """Survivability projection: will I win this DPS race given HP, ammo,
        my output, the enemy's output, and the map's ability to restore me?
        Returns (margin∈[-1,1], effective_HP_norm, my_DPS_share). margin>0 =
        I win the straight race; the map term lets a losing race flip via regear.
        """
        health = float(obs.self_state[6]) if len(obs.self_state) > 6 else 100.0
        armor  = float(obs.self_state[7]) if len(obs.self_state) > 7 else 0.0
        ammo   = float(obs.self_state[9]) if len(obs.self_state) > 9 else 0.0
        rune = int(np.argmax(obs.rune_flags)) if float(obs.rune_flags.sum()) > 0 else -1

        ehp = health + armor
        if rune == 0:                       # resist → ~2× effective HP under fire
            ehp *= 2.0
        # my output: measured during combat, else nominal capability; offense
        # runes double the nominal (measured already reflects them); no ammo → 0.
        my_dps = self._dps_self if self._dps_self > 0.5 else self.NOMINAL_DPS
        if rune in (1, 2) and self._dps_self <= 0.5:
            my_dps *= 2.0
        if ammo <= 0.0:
            my_dps = 0.0
        enemy_dps = self._dps_enemy
        eps = 1e-3
        ehp_norm = min(1.0, ehp / 200.0)
        dps_share = my_dps / (my_dps + enemy_dps + eps)

        if enemy_dps <= 0.5:                # no incoming → no race → safe
            return 1.0, ehp_norm, dps_share
        ttd = ehp / max(enemy_dps, eps)
        ttk = self._nearest_enemy_health(obs)
        ttk = (ttk if ttk > 1.0 else 100.0) / max(my_dps, eps)
        ttd *= (1.0 + max(0.0, min(1.0, map_help)))   # map can extend my survival
        margin = (ttd - ttk) / (ttd + ttk + eps)      # [-1, 1]
        return float(margin), float(ehp_norm), float(dps_share)

    def _rune_fit(self, idx: int, health: float) -> float:
        """How well a held rune fits the current health need. value (vampire)
        bridges — decent everywhere; offense/survival flip with health."""
        if idx < 0:
            return 0.0
        axis = self._RUNE_AXIS[idx]
        if health <= self.rune_low_health:          # bleeding → need to survive
            return {"survival": 1.0, "value": 0.7, "offense": -1.0}[axis]
        if health >= self.rune_high_health:         # healthy → press the attack
            return {"offense": 1.0, "value": 0.5, "survival": -0.5}[axis]
        return {"value": 1.0, "offense": 0.3, "survival": 0.3}[axis]  # mid: vampire shines

    def _exchange_logic(self, obs, dmg_dealt, dmg_taken, health, fired,
                        threat_in_range, combat_ready):
        """Damage-as-lens, gated by the survivability PROJECTION.

        win_margin (HP + ammo + my DPS + enemy DPS + the map's ability to
        restore me) classifies the trade — dominating / even / losing — a
        richer signal than the raw DPS ratio (it knows I'll lose a race I'm
        "winning" on damage if I'm out of ammo or bleeding with no health
        nearby). In the EVEN band it's a game of chicken resolved by the
        opponent's commitment (am I under fire?) against my per-life
        aggression lean + per-step jitter, so clones break symmetry instead
        of deadlocking.
        """
        d = self.exchange_dps_decay
        self._dps_self = d * self._dps_self + dmg_dealt
        self._dps_enemy = d * self._dps_enemy + dmg_taken
        in_fight = (threat_in_range or dmg_dealt > 0.0 or dmg_taken > 0.0
                    or self._dps_enemy > 1.0)
        if not in_fight:
            return 0.0, {}

        margin, _, _ = self._win_margin(obs)   # [-1,1]; >0 I win the projected race
        # aggression (per-life) + jitter (per-step) shift the even→press band:
        # a hawk (lean>0) commits on a slimmer projected margin.
        lean = self.aggression + self.rng.uniform(-self.exchange_jitter,
                                                  self.exchange_jitter)
        hi, lo = 0.15 - 0.15 * lean, -0.15 - 0.15 * lean
        # opponent commitment (frame-free): am I under fire right now / recently?
        opp_press = (dmg_taken > 0.0
                     or float(getattr(obs, "inbound_dmg_recency", 0.0)) > 0.3)
        me_press = (dmg_dealt > 0.0) or fired

        delta = 0.0
        state = "even"
        if margin >= hi:                      # WINNING the projection — press / finish
            state = "dominating"
            if me_press:
                delta += self.exchange_press_reward
        elif margin <= lo:                    # LOSING the projection — break / regear
            state = "losing"
            if me_press and opp_press:
                delta -= self.exchange_feed_penalty   # feeding a race I lose
            else:
                delta += self.exchange_break_reward   # disengage = correct
        else:                                 # EVEN projection — chicken matrix
            if me_press and not opp_press:
                delta += self.exchange_press_reward            # they yielded, take the space
            elif me_press and opp_press:
                if combat_ready and lean > 0.0:
                    delta += 0.5 * self.exchange_press_reward  # hawk w/ gear: accept the fight
                else:
                    delta -= self.exchange_crash_penalty       # mutual crash on an even trade
            else:
                delta += self.exchange_break_reward            # I swerve (dove) — reposition
        return float(delta), {
            "exchange_ratio": round(float(self._dps_self / (self._dps_enemy + 1e-3)), 3),
            "win_margin": round(float(margin), 3),
            "exchange_dominating": float(state == "dominating"),
            "exchange_even": float(state == "even"),
            "exchange_losing": float(state == "losing"),
            "exchange_lean": round(float(lean), 3),
        }

    def _threat_reward(
        self,
        obs: Observation,
        visible_count: int,
        nearest_visible: float,
        aim_aligned: bool,
        fired: bool,
        fire_aligned: bool,
        fire_splash: bool,
        hook_enemy: bool,
        ammo_depleted: bool,
    ):
        damage_dealt = max(0.0, float(obs.reward_damage_dealt))
        damage_taken = max(0.0, float(obs.reward_damage_taken))
        kills = max(0.0, float(obs.reward_kill))
        deaths = max(0.0, float(obs.reward_death))
        health = float(obs.self_state[6]) if len(obs.self_state) > 6 else 100.0
        combat_ready = health > self.low_health_threshold and not ammo_depleted
        threat_in_range = (
            visible_count > 0 and
            self.engagement_min <= nearest_visible <= self.engagement_max
        )
        active_threat = threat_in_range or damage_taken > 0.0

        self.episode_steps += 1
        self.episode_damage_dealt += damage_dealt
        self.episode_damage_taken += damage_taken
        self.episode_kills += kills
        self.episode_deaths += deaths
        if active_threat or damage_dealt > 0.0 or kills > 0.0 or deaths > 0.0:
            self.episode_contact_events += 1.0

        if active_threat:
            self.recent_threat_steps = max(1, int(self.recent_threat_window))
        elif self.recent_threat_steps > 0:
            self.recent_threat_steps -= 1

        threat_delta = 0.0
        if self.recent_threat_steps > 0 and deaths <= 0.0:
            threat_delta += self.survival_threat_reward
            if health <= self.low_health_threshold:
                threat_delta += self.survival_low_health_reward
        elif self.episode_contact_events > 0.0 and deaths <= 0.0:
            threat_delta += self.survival_tick_reward

        if threat_in_range:
            threat_delta += self.threat_engagement_reward
            if aim_aligned:
                threat_delta += self.threat_aim_reward
            if fired and (fire_aligned or fire_splash):
                threat_delta += self.threat_fire_reward
            if combat_ready and not (aim_aligned or fired or hook_enemy):
                threat_delta -= self.threat_ignore_penalty
            if not combat_ready and fired and ammo_depleted:
                threat_delta -= self.threat_unready_penalty

        if damage_dealt > 0.0:
            scale = 1.0 if threat_in_range else 0.5
            threat_delta += self.threat_damage_reward * damage_dealt * scale
        if kills > 0.0:
            threat_delta += self.threat_kill_reward * kills
            threat_delta += self.frag_advantage_reward * kills
        if deaths > 0.0:
            threat_delta -= self.frag_disadvantage_penalty * deaths

        # Damage as LENS: rolling exchange ratio → press / break (chicken),
        # not raw damage as reward.
        ex_delta, ex_info = self._exchange_logic(
            obs, damage_dealt, damage_taken, health, fired,
            threat_in_range, combat_ready)
        threat_delta += ex_delta

        return float(threat_delta), {
            "threat_bonus": float(threat_delta),
            **ex_info,
            "threat_in_range": float(threat_in_range),
            "threat_active": float(active_threat),
            "threat_recent": float(self.recent_threat_steps > 0),
            "threat_damage_dealt": float(damage_dealt),
            "threat_damage_taken": float(damage_taken),
            "threat_kills": float(kills),
            "threat_deaths": float(deaths),
            "threat_ignored": float(
                threat_in_range and combat_ready and not (aim_aligned or fired or hook_enemy)
            ),
            "survival_low_health": float(
                self.recent_threat_steps > 0 and health <= self.low_health_threshold
            ),
            "survival_contact": float(self.episode_contact_events > 0.0),
            "damage_margin_step": float(damage_dealt - damage_taken),
        }

    def fire_context(self, obs: Observation, weapon_id: Optional[int] = None) -> Dict[str, float]:
        visible_count, nearest_visible = self._visible_enemy_stats(obs)
        exposure_sum, exposure_max = self._visible_enemy_exposure(obs)
        aim_aligned, aim_yaw_error, aim_pitch_error = self._aim_alignment(obs)
        splash_weapon = self._is_splash_weapon(obs, weapon_id)
        splash_viable, splash_yaw_error, splash_pitch_error = self._splash_viability(
            obs, weapon_id
        )
        audio_contact = self._audio_contact(obs)
        return {
            "enemy_visible_any": float(visible_count > 0),
            "enemy_visible_count": float(visible_count),
            "enemy_visible_nearest": float(nearest_visible if visible_count > 0 else 0.0),
            "enemy_visible_exposure_sum": exposure_sum,
            "enemy_visible_exposure_max": exposure_max,
            "aim_aligned": float(aim_aligned),
            "aim_yaw_error": float(aim_yaw_error),
            "aim_pitch_error": float(aim_pitch_error),
            "splash_weapon": float(splash_weapon),
            "splash_viable": float(splash_viable),
            "splash_yaw_error": float(splash_yaw_error),
            "splash_pitch_error": float(splash_pitch_error),
            "audio_contact": float(audio_contact),
            "audio_age": float(obs.audio[3]) if len(obs.audio) > 3 else 0.0,
            "audio_alert": float(obs.audio[4]) if len(obs.audio) > 4 else 0.0,
        }

    def hook_context(self, obs: Observation) -> Dict[str, float]:
        visible_count, nearest_visible = self._visible_enemy_stats(obs)
        hook_aligned, hook_yaw_error, hook_pitch_error = self._hook_alignment(obs)
        weapon_id = float(obs.self_state[8]) if len(obs.self_state) > 8 else 0.0
        ammo = float(obs.self_state[9]) if len(obs.self_state) > 9 else 0.0
        requested_weapon = float(self._requested_weapon(obs))
        current_weapon = int(round(weapon_id))
        ammo_weapon_depleted = (
            current_weapon != BLASTER_ITEM_INDEX and
            ammo <= self.hook_ammo_threshold
        )
        requested_ammo_weapon_unavailable = (
            requested_weapon > 1.0 and
            current_weapon == BLASTER_ITEM_INDEX and
            ammo <= self.hook_ammo_threshold
        )
        ammo_depleted = ammo_weapon_depleted or requested_ammo_weapon_unavailable
        ammo_low = ammo_depleted
        enemy_close = (
            visible_count > 0 and
            nearest_visible <= self.hook_melee_distance
        )
        hook_enemy_viable = enemy_close and hook_aligned
        return {
            "hook_action": float(self._action_hook(obs)),
            "hook_enemy_viable": float(hook_enemy_viable),
            "hook_enemy_close": float(enemy_close),
            "hook_yaw_error": float(hook_yaw_error),
            "hook_pitch_error": float(hook_pitch_error),
            "weapon_id": float(weapon_id),
            "requested_weapon": float(requested_weapon),
            "ammo": float(ammo),
            "ammo_low": float(ammo_low),
            "ammo_depleted": float(ammo_depleted),
            "ammo_weapon_depleted": float(ammo_weapon_depleted),
            "requested_ammo_weapon_unavailable": float(requested_ammo_weapon_unavailable),
        }

    def _visible_enemy_stats(self, obs: Observation) -> Tuple[int, float]:
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        visible_count = 0
        nearest = float("inf")
        for ent in obs.entities[:count]:
            is_enemy = ent[7] > 0.5
            visible = abs(float(ent[8])) > 0.0
            if not (is_enemy and visible):
                continue
            # Avoid np.linalg.norm overhead for small 3D slices
            dist = float(hypot(ent[0], ent[1], ent[2]))
            visible_count += 1
            nearest = min(nearest, dist)
        return visible_count, nearest

    def _visible_enemy_exposure(self, obs: Observation) -> Tuple[float, float]:
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        values = [
            float(np.clip(abs(float(ent[8])), 0.0, 1.0))
            for ent in obs.entities[:count]
            if ent[6] > 0.0 and ent[7] > 0.5 and abs(float(ent[8])) > 0.0
        ]
        if not values:
            return 0.0, 0.0
        return float(sum(values)), float(max(values))

    def _aim_alignment(self, obs: Observation) -> Tuple[bool, float, float]:
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        best_yaw = 180.0
        best_pitch = 90.0
        aligned = False
        
        for ent in obs.entities[:count]:
            if not (ent[7] > 0.5 and abs(float(ent[8])) > 0.0):
                continue
            x, y, z = (float(ent[0]), float(ent[1]), float(ent[2]))
            if x <= 0.0:
                continue
            
            yaw_error = abs(degrees(atan2(y, x)))
            pitch_error = abs(degrees(atan2(z, max(hypot(x, y), 1e-3))))
            
            is_this_aligned = (yaw_error <= self.aim_yaw_deg and pitch_error <= self.aim_pitch_deg)
            
            if is_this_aligned:
                aligned = True
                best_yaw = yaw_error
                best_pitch = pitch_error
                break
            else:
                # Track the candidate with the smallest combined angular error
                if (yaw_error + pitch_error) < (best_yaw + best_pitch):
                    best_yaw = yaw_error
                    best_pitch = pitch_error
                    
        if not aligned and best_yaw == 180.0 and best_pitch == 90.0:
            return False, 0.0, 0.0
            
        return aligned, best_yaw, best_pitch

    def _hook_alignment(self, obs: Observation) -> Tuple[bool, float, float]:
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        best_yaw = 180.0
        best_pitch = 90.0
        aligned = False

        for ent in obs.entities[:count]:
            if not (ent[7] > 0.5 and abs(float(ent[8])) > 0.0):
                continue
            x, y, z = (float(ent[0]), float(ent[1]), float(ent[2]))
            if x <= 0.0:
                continue

            yaw_error = abs(degrees(atan2(y, x)))
            pitch_error = abs(degrees(atan2(z, max(hypot(x, y), 1e-3))))
            is_this_aligned = (
                yaw_error <= self.hook_yaw_deg and
                pitch_error <= self.hook_pitch_deg
            )
            if is_this_aligned:
                aligned = True
                best_yaw = yaw_error
                best_pitch = pitch_error
                break
            if (yaw_error + pitch_error) < (best_yaw + best_pitch):
                best_yaw = yaw_error
                best_pitch = pitch_error

        if not aligned and best_yaw == 180.0 and best_pitch == 90.0:
            return False, 0.0, 0.0
        return aligned, best_yaw, best_pitch

    def _splash_viability(
        self, obs: Observation, weapon_id: Optional[int] = None
    ) -> Tuple[bool, float, float]:
        if not self._is_splash_weapon(obs, weapon_id):
            return False, 0.0, 0.0

        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        best_yaw = 180.0
        best_pitch = 90.0
        viable = False
        for ent in obs.entities[:count]:
            if not (ent[7] > 0.5 and abs(float(ent[8])) > 0.0):
                continue
            x, y, z = (float(ent[0]), float(ent[1]), float(ent[2]))
            if x <= 0.0:
                continue
            dist = float(hypot(x, y, z))
            if dist > self.splash_max_distance:
                continue

            yaw_error = abs(degrees(atan2(y, x)))
            pitch_error = abs(degrees(atan2(z, max(hypot(x, y), 1e-3))))
            is_viable = (
                yaw_error <= self.splash_yaw_deg and
                pitch_error <= self.splash_pitch_deg
            )
            if is_viable:
                viable = True
                best_yaw = yaw_error
                best_pitch = pitch_error
                break
            if (yaw_error + pitch_error) < (best_yaw + best_pitch):
                best_yaw = yaw_error
                best_pitch = pitch_error

        if not viable and best_yaw == 180.0 and best_pitch == 90.0:
            return False, 0.0, 0.0
        return viable, best_yaw, best_pitch

    def _has_required_hook_nearby(self, obs: Observation) -> bool:
        count = max(0, min(int(obs.hook_zone_count), obs.hook_zones.shape[0]))
        if count == 0:
            return False
        pos = obs.self_state[:3]
        for zone in obs.hook_zones[:count]:
            flags = int(zone[7])
            if not (flags & HOOK_REQUIRED):
                continue
            # Avoid np.linalg.norm overhead for small 3D slices
            dist = float(hypot(zone[0] - pos[0], zone[1] - pos[1], zone[2] - pos[2]))
            if dist <= self.hook_required_distance:
                return True
        return False

    def _action_fired(self, obs: Observation) -> bool:
        action_debug = getattr(obs, "action_debug", None)
        if action_debug is None or len(action_debug) <= 9:
            return False
        return bool(int(action_debug[9]))

    def movement_context(self, obs: Observation) -> Dict[str, float]:
        """Grade grounded map traversal using the engine-applied action echo."""
        action_debug = getattr(obs, "action_debug", ())
        move_forward = float(action_debug[4]) if len(action_debug) > 4 else 0.0
        move_right = float(action_debug[5]) if len(action_debug) > 5 else 0.0
        jump_action = bool(int(action_debug[8])) if len(action_debug) > 8 else False
        movement_intent = min(1.0, hypot(move_forward, move_right))
        forward_intent = min(1.0, max(0.0, move_forward))
        backward_intent = min(1.0, max(0.0, -move_forward))
        horizontal_speed = hypot(float(obs.self_state[3]), float(obs.self_state[4]))
        wants_movement = movement_intent >= self.movement_intent_min
        nominal_speed = wants_movement and (
            self.nominal_speed_min <= horizontal_speed <= self.nominal_speed_max
        )
        slow_movement = wants_movement and horizontal_speed < self.nominal_speed_min
        overspeed = horizontal_speed > self.nominal_speed_max
        delta = 0.0
        if nominal_speed:
            # Scale by forward intent so circling/knockback cannot farm the
            # speed reward without actually traversing the map.
            delta += self.nominal_speed_reward * forward_intent
        elif slow_movement:
            deficit = 1.0 - horizontal_speed / max(1.0, self.nominal_speed_min)
            delta -= self.slow_movement_penalty * deficit
        elif overspeed and wants_movement:
            excess = min(1.0, (horizontal_speed - self.nominal_speed_max) /
                         max(1.0, self.nominal_speed_max))
            delta -= self.overspeed_penalty * excess

        # Nominal traversal must prefer actual forward travel. Previously the
        # reward used abs(move_forward), making a backwards moonwalk exactly
        # as valuable as moving into the direction the body is facing.
        delta -= self.backward_movement_penalty * backward_intent

        view_pitch = float(getattr(obs, "pitch", 0.0))
        view_pitch_abs = abs(view_pitch)
        visible_count = 0
        if hasattr(obs, "entity_count") and hasattr(obs, "entities"):
            visible_count, _nearest = self._visible_enemy_stats(obs)
        horizon_penalty = 0.0
        if visible_count == 0 and view_pitch_abs > self.horizon_pitch_limit:
            excess = min(
                1.0,
                (view_pitch_abs - self.horizon_pitch_limit) /
                max(1.0, 89.0 - self.horizon_pitch_limit),
            )
            horizon_penalty = self.horizon_pitch_penalty * excess
            delta -= horizon_penalty

        # Make level posture an active traversal objective. Only genuine
        # forward travel with no visible target can collect this reward;
        # backward moonwalking cannot, and target-visible vertical aim stays
        # unconstrained for combat across elevations.
        level_aim_reward = 0.0
        if (
            visible_count == 0
            and forward_intent >= self.movement_intent_min
            and horizontal_speed >= self.level_aim_min_speed
            and horizontal_speed <= self.nominal_speed_max
        ):
            level_quality = max(
                0.0,
                1.0 - view_pitch_abs / 89.0,
            )
            level_aim_reward = (
                self.level_aim_movement_reward
                * forward_intent
                * level_quality
            )
            delta += level_aim_reward

        jump_slow = jump_action and (slow_movement or not wants_movement)
        if jump_action:
            delta -= self.jump_cost
            if jump_slow:
                delta -= self.slow_jump_penalty
        return {
            "movement_speed": float(horizontal_speed),
            "movement_intent": float(movement_intent),
            "forward_intent": float(forward_intent),
            "backward_intent": float(backward_intent),
            "view_pitch_abs": float(view_pitch_abs),
            "horizon_pitch_penalty": float(horizon_penalty),
            "level_aim_movement_reward": float(level_aim_reward),
            "movement_nominal": float(nominal_speed),
            "movement_slow": float(slow_movement),
            "movement_overspeed": float(overspeed),
            "movement_discipline": float(delta),
            "jump_action": float(jump_action),
            "jump_slow": float(jump_slow),
        }

    def _action_hook(self, obs: Observation) -> int:
        action_debug = getattr(obs, "action_debug", None)
        if action_debug is None or len(action_debug) <= 10:
            return 0
        return int(action_debug[10])

    def _requested_weapon(self, obs: Observation) -> int:
        action_debug = getattr(obs, "action_debug", None)
        if action_debug is None or len(action_debug) <= 3:
            return 0
        return int(action_debug[3])

    def _is_splash_weapon(self, obs: Observation, weapon_id: Optional[int] = None) -> bool:
        if weapon_id is not None and int(weapon_id) > 0:
            return int(weapon_id) in {6, 7}
        action_debug = getattr(obs, "action_debug", None)
        if action_debug is None or len(action_debug) <= 3:
            return False
        # ML action IDs: 6=grenade launcher, 7=rocket launcher.
        return int(action_debug[3]) in {6, 7}

    def _audio_contact(self, obs: Observation) -> bool:
        if len(obs.audio) <= 4:
            return False
        age = float(obs.audio[3])
        alert = float(obs.audio[4])
        return age <= self.audio_fire_max_age and alert >= self.audio_fire_alert_min


# ── Lattice checkpoint persistence ──────────────────────────────────────────

_DYNAMIC_CELL_FIELDS = {"readiness", "route_bias"}


def save_lattice_state(instances, path, total_env_steps: int = 0) -> Path:
    """Atomically checkpoint each bot's learned per-map lattice as JSON/gzip."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cell_fields = [f.name for f in fields(SessionMemoryCell)
                   if f.name not in _DYNAMIC_CELL_FIELDS]
    payload = {
        "version": 1,
        "env_steps": int(total_env_steps),
        "instances": [],
    }
    for inst in instances:
        maps = {}
        for map_name, memory in inst.session_memories.items():
            cells = []
            for cell, entry in memory.items():
                values = {name: getattr(entry, name) for name in cell_fields}
                if not any(float(value) != 0.0 for value in values.values()):
                    continue
                cells.append({"cell": list(cell), **values})
            if cells:
                maps[map_name] = cells
        payload["instances"].append({"maps": maps})
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    temporary = target.with_name(target.name + ".tmp")
    if target.suffix == ".gz":
        with gzip.open(temporary, "wb") as handle:
            handle.write(encoded)
    else:
        temporary.write_bytes(encoded)
    os.replace(temporary, target)
    return target


def load_lattice_state(instances, path) -> dict:
    """Restore a matching set of per-bot lattices; tolerate added cell fields."""
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(source.read_text())
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"unsupported lattice state version in {source}")
    valid_fields = {f.name for f in fields(SessionMemoryCell)} - _DYNAMIC_CELL_FIELDS
    restored_cells = 0
    saved_instances = payload.get("instances", [])
    for index, inst in enumerate(instances):
        if index >= len(saved_instances):
            break
        for map_name, cells in saved_instances[index].get("maps", {}).items():
            memory = inst._memory_for_map(map_name)
            for raw in cells:
                cell = tuple(int(v) for v in raw.get("cell", ()))
                if len(cell) != 3:
                    continue
                kwargs = {name: raw[name] for name in valid_fields if name in raw}
                memory[cell] = SessionMemoryCell(**kwargs)
                restored_cells += 1
            # Do NOT mark restored maps as preloaded: generator priors must
            # re-merge (max-merge is idempotent) on the next reset, so a
            # resumed run trains on the same prior substrate as a fresh one
            # instead of shadowing it.
            inst._rust_indices.pop(map_name, None)
            inst._rust_dirty_cells.pop(map_name, None)
            inst._rust_removed_cells.pop(map_name, None)
            inst._rust_score_events.pop(map_name, None)
            inst._invalidate_cell_col_cache(map_name)
        inst._invalidate_feature_cache()
    return {
        "env_steps": int(payload.get("env_steps", 0)),
        "instances": min(len(instances), len(saved_instances)),
        "cells": restored_cells,
    }


# ── Observed-heat telemetry export ───────────────────────────────────────────
#
# Closes the generation loop: session-memory combat data recorded during live
# play is merged across all bot instances and written per map in the
# generator's observed-heat deposit format. Regenerating a map with the SAME
# seed and --observed-heat <file> keeps its geometry but re-places the item
# economy around where the fights actually happened.

def export_observed_heat(instances, out_dir, min_amount: float = 0.05,
                         total_env_steps: int = 0) -> int:
    """Merge session memories across VoxelSpatialReward instances and write
    <out_dir>/<map>.heat.json files. Returns number of maps exported."""
    import json
    from pathlib import Path

    merged: Dict[str, Dict[Tuple[int, int, int], Dict[str, float]]] = {}
    voxel = 256.0
    for inst in instances:
        voxel = max(voxel, float(getattr(inst, "voxel_size", 256.0)))
        for map_name, cells in getattr(inst, "session_memories", {}).items():
            tgt = merged.setdefault(map_name, {})
            for cell, m in cells.items():
                acc = tgt.setdefault(cell, {"weapon": 0.0, "danger": 0.0,
                                            "objective": 0.0})
                acc["weapon"]    += m.engagement_count + 0.3 * m.enemy_seen
                acc["danger"]    += 2.0 * m.deaths + m.damage_taken / 100.0
                acc["objective"] += 2.0 * m.kills + m.item_contested

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for map_name, cells in merged.items():
        if not map_name or map_name == "unknown" or not cells:
            continue
        peak = {ch: max((c[ch] for c in cells.values()), default=0.0)
                for ch in ("weapon", "danger", "objective")}
        deposits = []
        for (ix, iy, iz), acc in cells.items():
            for ch, val in acc.items():
                if peak[ch] <= 0.0:
                    continue
                amount = min(1.5, val / peak[ch])
                if amount < min_amount:
                    continue
                deposits.append({
                    "channel": ch,
                    "x": (ix + 0.5) * voxel,
                    "y": (iy + 0.5) * voxel,
                    "z": (iz + 0.5) * voxel,
                    "amount": round(amount, 4),
                    "radius": voxel * 1.5,
                })
        if not deposits:
            continue
        payload = {"map": map_name, "env_steps": int(total_env_steps),
                   "voxel_size": voxel, "deposits": deposits}
        (out / f"{map_name}.heat.json").write_text(
            json.dumps(payload, indent=1) + "\n")
        n_written += 1
    return n_written
