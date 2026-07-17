"""Per-client own-observation query source for the Rust multires provider.

``OwnObservationQuerySource`` is the concrete ``SpatialQueryInputSource``
consumed by ``harness.rust_multires_provider.RustAtlasSpatialProvider``.  It
derives every ``SpatialQueryInputs`` field exclusively from the admitted
canonical objectives artifact plus the single bound client's own telemetry:

- Objective availability beliefs start from the map-start prior (every
  consumable present at map load), advance only on this client's own pickups,
  and ramp back along the static predicted respawn period.  The source never
  reads exact global item timers or any other client's privileged state; an
  unseen opponent pickup therefore leaves the belief at its prior.
- The three survivability values (win margin, effective-health norm, own-DPS
  share) come from factual health/armor/ammo/runes, the nearest visible
  enemy's health, and this client's own decayed damage exchange.
- Thermal evidence is per-client target heat with the frozen at-most-five-tick
  TTL, built only from currently/recently visible target geometry carrying a
  stable debug identity ``(edict << 14) | life_epoch``.

- Named Dyn24 events (engagement=1, threat=2, opportunity=3, self_fire=4,
  death=5) derive solely from this client's own public observation facts
  (reward channels, ``action_debug`` echo, ``entity_debug`` identities); the
  event ID is ``(server_frame << 3) | kind_code`` with at most one event per
  kind per frame, emitted in ascending kind-code order.
- Environment steps are an exact CAS pair: ``expected_environment_steps`` is
  the pre-frame value and ``environment_steps`` its one-step successor
  (``0``/``1`` on the first frame), committed once after all validation.

Public/private separation: everything emitted here is class-2 advisory or
class-1 factual per the observation taxonomy.  No solver answers, no exact
unobserved timers, no through-wall enemy state.  The physically private
causal telemetry channel is never consulted for event derivation or any
belief; its ``client_life_epoch`` alone is read — via the isolated
``_transport_life_epoch`` accessor — strictly as transport identity because
wire v6 carries no public own-life-epoch field (``self_debug`` is only
``[edict_index, client_slot, source, flags]``).

Transactionality: every ``sample`` either commits completely or leaves the
source untouched.  Any validation failure restores the pre-call frame fence,
epoch fences, objective beliefs, thermal tracks, DPS exchange, event edges,
and environment-step counter, so a corrected retry of the same frame behaves
as if the failed call never happened.

Fail-closed rules: missing/mismatched artifacts, mixed client/map/epoch
identity, stale or repeated server frames, decreasing life epochs, ambiguous
pickup resolution, and nonfinite facts all raise
``RustSpatialProviderError`` subclasses instead of degrading.

Integration note for milestone M4 (this module does not edit the provider):
``RustAtlasSpatialProvider.from_admitted_bundle`` must be constructed with
``input_source=OwnObservationQuerySource.from_objectives_artifact(...)`` using
the same bundle directory, ``expected_atlas_sha256``, and the objectives
digest recorded in the bundle manifest's ``analysis_files``.  One source is
required per (client, map epoch); map rotation constructs a fresh source
alongside the fresh provider.  No wire field currently reports mover blockers
or dynamic penalties to a client, so this source authoritatively reports empty
overlays; when that telemetry lands, M4 must extend ``_ingest_frame`` rather
than injecting overlays around this class.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .client_protocol import ClientTelemetry
from .protocol import (
    ActionDebugIndex,
    ML_ENTITY_DUCKED,
    ML_ENTITY_EPOCH_MASK,
    ML_ENTITY_EPOCH_SHIFT,
)
from .rust_multires_provider import RustSpatialProviderError, SpatialQueryInputs


OBJECTIVES_SCHEMA = "q2-atlas-objectives-v1"

# Frozen five-tick thermal TTL (crates/q2-lattice THERMAL_MAX_AGE_TICKS).
THERMAL_TTL_TICKS = 5

# 10 Hz game cadence; bound into the physics/runtime manifest.
TICK_SECONDS = 0.1

# Deterministic pickup-to-objective identity resolution radius (world units).
# Q2 item touch = 32u item bbox against the player hull; 64 units bounds the
# same-frame drift between the touch origin and the telemetry origin.
PICKUP_RESOLVE_RADIUS = 64.0

# Static vanilla Quake 2 deathmatch respawn periods (seconds) per objective
# class.  These are the public predicted-respawn facts; a period of 0 means
# event-driven/unknown phase, and non-consumable classes never vacate.
CONSUMABLE_RESPAWN_SECONDS: Mapping[str, float] = {
    "weapon": 30.0,
    "ammunition": 30.0,
    "health": 30.0,
    "armor": 20.0,
    "powerup": 60.0,
    "rune": 0.0,
}
NON_CONSUMABLE_CLASSES = frozenset({"control", "spawn_egress"})

_EXCHANGE_DPS_DECAY = 0.90
_NOMINAL_DPS = 10.0
_MILLIUNITS = 1000.0


class QuerySourceError(RustSpatialProviderError):
    """Raised when own-observation query inputs cannot be produced safely."""


@dataclass
class _ObjectiveBelief:
    objective_id: int
    objective_class: str
    classname: str
    world_point: tuple[float, float, float]
    respawn_period_s: float
    consumable: bool
    # Frame at which this client itself consumed the objective; None means the
    # map-start prior (present) still holds for this client's belief.
    consumed_frame: int | None = None

    def availability(self, server_frame: int) -> float:
        if self.consumed_frame is None:
            return 1.0
        if self.respawn_period_s <= 0.0:
            # Event-driven respawn (runes): phase is unknown after our pickup.
            return 0.5
        elapsed_s = (server_frame - self.consumed_frame) * TICK_SECONDS
        return max(0.0, min(1.0, elapsed_s / self.respawn_period_s))


@dataclass(frozen=True)
class _ThermalTrack:
    target_id: int
    world_point: tuple[float, float, float]
    heat: float
    observed_tick: int


@dataclass
class StagedSpatialQuery:
    """One reversible source transaction committed after provider success."""

    inputs: SpatialQueryInputs
    _source: "OwnObservationQuerySource"
    _snapshot: tuple
    _active: bool = True

    def commit(self) -> None:
        if not self._active:
            raise QuerySourceError("spatial query transaction is already closed")
        self._active = False

    def rollback(self) -> None:
        if not self._active:
            raise QuerySourceError("spatial query transaction is already closed")
        self._source._restore(self._snapshot)
        self._active = False


def _require_finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise QuerySourceError(f"{label} is not finite")
    return result


def _local_to_world_vector(
    local: np.ndarray, yaw_deg: float, pitch_deg: float
) -> np.ndarray:
    """Invert Quake AngleVectors' forward/Quake-right/up projection."""
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    forward = np.array((cp * cy, cp * sy, -sp), dtype=np.float64)
    right = np.array((sy, -cy, 0.0), dtype=np.float64)
    up = np.array((sp * cy, sp * sy, cp), dtype=np.float64)
    vector = np.asarray(local, dtype=np.float64)
    return forward * vector[0] + right * vector[1] + up * vector[2]


def load_objectives_artifact(
    objectives_path: Path,
    *,
    expected_objectives_sha256: str,
    expected_atlas_sha256: str,
    expected_atlas_origin: tuple[int, int, int],
    map_name: str,
) -> list[_ObjectiveBelief]:
    """Load and identity-check the admitted canonical objectives artifact."""
    path = Path(objectives_path)
    if not path.is_file():
        raise QuerySourceError("canonical objectives artifact is missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_objectives_sha256:
        raise QuerySourceError("objectives artifact digest differs from admission")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySourceError("objectives artifact is invalid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != OBJECTIVES_SCHEMA
        or not isinstance(document.get("objectives"), list)
    ):
        raise QuerySourceError("objectives artifact schema is not admitted")
    if document.get("atlas_sha256") != expected_atlas_sha256:
        raise QuerySourceError("objectives artifact Atlas digest fence differs")
    if document.get("origin") != list(expected_atlas_origin):
        raise QuerySourceError("objectives artifact Atlas origin fence differs")
    if document.get("canonical_map_id") != map_name:
        raise QuerySourceError("objectives artifact map identity differs")
    beliefs: list[_ObjectiveBelief] = []
    seen_ids: set[int] = set()
    for record in document["objectives"]:
        if not isinstance(record, dict):
            raise QuerySourceError("objective record is not a mapping")
        objective_id = record.get("objective_id")
        objective_class = record.get("class")
        classname = record.get("classname")
        milli = record.get("world_milliunits")
        if (
            type(objective_id) is not int or objective_id < 0
            or not isinstance(objective_class, str)
            or not isinstance(classname, str)
            or not isinstance(milli, list) or len(milli) != 3
            or not all(type(axis) is int for axis in milli)
        ):
            raise QuerySourceError(
                f"objective record {objective_id!r} is malformed"
            )
        if objective_id in seen_ids:
            raise QuerySourceError(f"objective ID {objective_id} is duplicated")
        seen_ids.add(objective_id)
        consumable = objective_class in CONSUMABLE_RESPAWN_SECONDS
        if not consumable and objective_class not in NON_CONSUMABLE_CLASSES:
            raise QuerySourceError(
                f"objective class {objective_class!r} is not admitted"
            )
        world_point = tuple(
            _require_finite(axis / _MILLIUNITS, "objective world point")
            for axis in milli
        )
        beliefs.append(_ObjectiveBelief(
            objective_id=objective_id,
            objective_class=objective_class,
            classname=classname,
            world_point=world_point,  # type: ignore[arg-type]
            respawn_period_s=CONSUMABLE_RESPAWN_SECONDS.get(objective_class, 0.0),
            consumable=consumable,
        ))
    beliefs.sort(key=lambda belief: belief.objective_id)
    return beliefs


class OwnObservationQuerySource:
    """Concrete per-client ``SpatialQueryInputSource`` over own observations."""

    def __init__(
        self,
        *,
        objectives_path: Path,
        expected_objectives_sha256: str,
        expected_atlas_sha256: str,
        atlas_origin: tuple[int, int, int],
        map_name: str,
        client_id: str,
        environment_steps_base: int = 0,
    ):
        if not client_id:
            raise QuerySourceError("query source requires a bound client identity")
        if environment_steps_base < 0:
            raise QuerySourceError("environment_steps base must be nonnegative")
        if (
            not isinstance(atlas_origin, tuple)
            or len(atlas_origin) != 3
            or any(type(value) is not int for value in atlas_origin)
        ):
            raise QuerySourceError("query source requires exact integer Atlas origin")
        self.map_name = map_name
        self.client_id = client_id
        self.atlas_origin = atlas_origin
        self._objectives = load_objectives_artifact(
            Path(objectives_path),
            expected_objectives_sha256=expected_objectives_sha256,
            expected_atlas_sha256=expected_atlas_sha256,
            expected_atlas_origin=atlas_origin,
            map_name=map_name,
        )
        self._environment_steps = int(environment_steps_base)
        self._bound_map_epoch: int | None = None
        self._last_server_frame: int | None = None
        self._last_life_epoch: int | None = None
        self._thermal_tracks: dict[int, _ThermalTrack] = {}
        self._dps_self = 0.0
        self._dps_enemy = 0.0
        # target_id -> last L2 64-unit cell an opportunity event was emitted at.
        self._opportunity_cells: dict[int, tuple[int, int, int]] = {}
        self._last_echoed_fire = False
        self._dead = False

    @classmethod
    def from_objectives_artifact(
        cls,
        *,
        bundle_directory: Path,
        map_name: str,
        expected_objectives_sha256: str,
        expected_atlas_sha256: str,
        atlas_origin: tuple[int, int, int],
        client_id: str,
        environment_steps_base: int = 0,
    ) -> "OwnObservationQuerySource":
        return cls(
            objectives_path=Path(bundle_directory) / f"{map_name}.objectives.json",
            expected_objectives_sha256=expected_objectives_sha256,
            expected_atlas_sha256=expected_atlas_sha256,
            atlas_origin=atlas_origin,
            map_name=map_name,
            client_id=client_id,
            environment_steps_base=environment_steps_base,
        )

    @property
    def environment_steps(self) -> int:
        return self._environment_steps

    # ---- identity fences ---------------------------------------------------

    @staticmethod
    def _transport_life_epoch(telemetry: ClientTelemetry) -> int:
        """Isolated transport-identity read of the private causal channel.

        Wire v6 exposes no public own-life-epoch field (``self_debug`` carries
        only ``[edict_index, client_slot, source, flags]``), so the causal
        record's ``client_life_epoch`` is consumed here strictly as transport
        identity.  This is the ONLY causal field read anywhere in this module;
        causal flags, target/source attribution, and damage facts have zero
        influence on any emitted belief, survivability value, thermal track,
        or Dyn event.
        """
        return int(telemetry.causal.client_life_epoch)

    def _snapshot(self) -> tuple:
        return (
            self._environment_steps,
            self._bound_map_epoch,
            self._last_server_frame,
            self._last_life_epoch,
            dict(self._thermal_tracks),
            self._dps_self,
            self._dps_enemy,
            dict(self._opportunity_cells),
            self._last_echoed_fire,
            self._dead,
            tuple(belief.consumed_frame for belief in self._objectives),
        )

    def _restore(self, snapshot: tuple) -> None:
        (
            self._environment_steps,
            self._bound_map_epoch,
            self._last_server_frame,
            self._last_life_epoch,
            thermal_tracks,
            self._dps_self,
            self._dps_enemy,
            opportunity_cells,
            self._last_echoed_fire,
            self._dead,
            consumed_frames,
        ) = snapshot
        self._thermal_tracks = dict(thermal_tracks)
        self._opportunity_cells = dict(opportunity_cells)
        for belief, consumed_frame in zip(self._objectives, consumed_frames):
            belief.consumed_frame = consumed_frame

    def _check_identity(self, telemetry: ClientTelemetry, map_epoch: int) -> None:
        if telemetry.client_id != self.client_id:
            raise QuerySourceError(
                f"telemetry client {telemetry.client_id!r} differs from bound "
                f"client {self.client_id!r}"
            )
        if telemetry.map_name != self.map_name:
            raise QuerySourceError(
                f"telemetry map {telemetry.map_name!r} differs from bound "
                f"map {self.map_name!r}"
            )
        if map_epoch < 0:
            raise QuerySourceError("map epoch must be nonnegative")
        if self._bound_map_epoch is None:
            self._bound_map_epoch = int(map_epoch)
        elif map_epoch < self._bound_map_epoch:
            raise QuerySourceError("map epoch regressed on bound query source")
        elif map_epoch > self._bound_map_epoch:
            self._reset_for_new_epoch(int(map_epoch))
        life_epoch = self._transport_life_epoch(telemetry)
        if self._last_life_epoch is None:
            self._last_life_epoch = life_epoch
        elif life_epoch < self._last_life_epoch:
            raise QuerySourceError("client life epoch regressed")
        elif life_epoch > self._last_life_epoch:
            self._reset_for_new_life(life_epoch)
        frame = int(telemetry.server_frame)
        if self._last_server_frame is not None and frame <= self._last_server_frame:
            raise QuerySourceError(
                f"server frame {frame} is stale (last {self._last_server_frame})"
            )
        self._last_server_frame = frame

    def _reset_for_new_epoch(self, map_epoch: int) -> None:
        self._bound_map_epoch = map_epoch
        self._last_server_frame = None
        self._last_life_epoch = None
        self._thermal_tracks.clear()
        self._dps_self = 0.0
        self._dps_enemy = 0.0
        self._opportunity_cells.clear()
        self._last_echoed_fire = False
        self._dead = False
        for belief in self._objectives:
            belief.consumed_frame = None

    def _reset_for_new_life(self, life_epoch: int) -> None:
        # Item-timing belief is map-epoch state and survives death; combat
        # exchange and thermal evidence do not.
        self._last_life_epoch = life_epoch
        self._thermal_tracks.clear()
        self._dps_self = 0.0
        self._dps_enemy = 0.0
        self._opportunity_cells.clear()
        self._last_echoed_fire = False
        self._dead = False

    # ---- per-frame ingestion -------------------------------------------------

    def _resolve_own_pickup(self, telemetry: ClientTelemetry) -> None:
        pickup = float(getattr(telemetry.observation, "reward_item_pickup", 0.0))
        if not math.isfinite(pickup):
            raise QuerySourceError("item pickup fact is not finite")
        if pickup <= 0.0:
            return
        origin = telemetry.observation.self_state[:3]
        position = tuple(_require_finite(value, "own position") for value in origin)
        frame = int(telemetry.server_frame)
        candidates = []
        for belief in self._objectives:
            if not belief.consumable:
                continue
            if belief.availability(frame) <= 0.0:
                continue
            distance = math.dist(position, belief.world_point)
            if distance <= PICKUP_RESOLVE_RADIUS:
                candidates.append((distance, belief))
        if len(candidates) != 1:
            raise QuerySourceError(
                "own pickup does not resolve to exactly one objective "
                f"({len(candidates)} candidates within {PICKUP_RESOLVE_RADIUS} units)"
            )
        candidates[0][1].consumed_frame = frame

    def _survivability(
        self, telemetry: ClientTelemetry
    ) -> tuple[float, float, float]:
        obs = telemetry.observation
        self_state = np.asarray(obs.self_state, dtype=np.float64)
        health = _require_finite(self_state[6], "health")
        armor = _require_finite(self_state[7], "armor")
        ammo = _require_finite(self_state[9], "ammo")
        rune_flags = np.asarray(obs.rune_flags, dtype=np.float64)
        if not np.isfinite(rune_flags).all():
            raise QuerySourceError("rune facts are not finite")
        dealt = _require_finite(
            getattr(obs, "reward_damage_dealt", 0.0), "damage dealt"
        )
        taken = _require_finite(
            getattr(obs, "reward_damage_taken", 0.0), "damage taken"
        )
        self._dps_self = _EXCHANGE_DPS_DECAY * self._dps_self + max(0.0, dealt)
        self._dps_enemy = _EXCHANGE_DPS_DECAY * self._dps_enemy + max(0.0, taken)

        rune = int(np.argmax(rune_flags)) if float(rune_flags.sum()) > 0 else -1
        ehp = health + armor
        if rune == 0:  # resist rune ≈ 2× effective HP under fire
            ehp *= 2.0
        my_dps = self._dps_self if self._dps_self > 0.5 else _NOMINAL_DPS
        if rune in (1, 2) and self._dps_self <= 0.5:
            my_dps *= 2.0
        if ammo <= 0.0:
            my_dps = 0.0
        enemy_dps = self._dps_enemy
        eps = 1e-3
        ehp_norm = min(1.0, max(0.0, ehp) / 200.0)
        dps_share = my_dps / (my_dps + enemy_dps + eps)
        if enemy_dps <= 0.5:
            return 1.0, float(ehp_norm), float(dps_share)
        time_to_die = max(0.0, ehp) / max(enemy_dps, eps)
        enemy_health = self._nearest_visible_enemy_health(obs)
        time_to_kill = (enemy_health if enemy_health > 1.0 else 100.0) / max(my_dps, eps)
        margin = (time_to_die - time_to_kill) / (time_to_die + time_to_kill + eps)
        for label, value in (
            ("win margin", margin), ("effective health", ehp_norm),
            ("DPS share", dps_share),
        ):
            _require_finite(value, label)
        return float(margin), float(ehp_norm), float(dps_share)

    @staticmethod
    def _nearest_visible_enemy_health(obs: Any) -> float:
        best_distance, best_health = math.inf, 0.0
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        for index in range(count):
            entity = obs.entities[index]
            if entity[7] < 0.5 or abs(float(entity[8])) <= 0.0:
                continue
            values = [float(entity[axis]) for axis in range(3)] + [float(entity[6])]
            if not all(math.isfinite(value) for value in values):
                raise QuerySourceError("visible target facts are not finite")
            distance = values[0] ** 2 + values[1] ** 2 + values[2] ** 2
            if distance < best_distance:
                best_distance, best_health = distance, values[3]
        return best_health

    def _observe_thermal(self, telemetry: ClientTelemetry) -> frozenset[int]:
        """Ingest this frame's visible-target evidence.

        Returns the stable identities of own-visible targets that are now
        publicly identifiable as killed (debug identity with nonpositive
        health).  Their existing tracks are retained through this frame so a
        same-frame engagement can still attribute, and the caller retires them
        deterministically after event derivation.
        """
        obs = telemetry.observation
        tick = int(telemetry.server_frame)
        killed: set[int] = set()
        if float(obs.self_state[6]) <= 0.0:
            # While dead this client accrues no new visible-target evidence;
            # the caller clears all tracks after this frame's events.
            self._expire_thermal(tick)
            return frozenset()
        yaw = _require_finite(obs.yaw, "yaw")
        pitch = _require_finite(getattr(obs, "pitch", 0.0), "pitch")
        eye = np.asarray(obs.self_state[:3], dtype=np.float64).copy()
        if not np.isfinite(eye).all():
            raise QuerySourceError("own position is not finite")
        flags = 0
        self_debug = getattr(obs, "self_debug", None)
        if self_debug is not None and len(self_debug) >= 4:
            flags = int(self_debug[3])
        eye[2] += -2.0 if flags & ML_ENTITY_DUCKED else 22.0
        count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
        entity_debug = getattr(obs, "entity_debug", None)
        for index in range(count):
            entity = obs.entities[index]
            if (
                entity_debug is None
                or index >= entity_debug.shape[0]
                or int(entity_debug[index, 0]) <= 0
            ):
                # No stable debug identity: not admissible thermal evidence.
                continue
            edict_index = int(entity_debug[index, 0])
            life_epoch = (
                int(entity_debug[index, 3]) & ML_ENTITY_EPOCH_MASK
            ) >> ML_ENTITY_EPOCH_SHIFT
            target_id = (edict_index << 14) | life_epoch
            if float(entity[7]) > 0.5 and float(entity[6]) <= 0.0:
                # Publicly identifiable kill: retain the existing track for
                # same-frame engagement attribution, then retire it.
                killed.add(target_id)
                continue
            exposure = abs(float(entity[8]))
            if not (float(entity[6]) > 0.0 and float(entity[7]) > 0.5 and exposure > 0.0):
                continue
            if not math.isfinite(exposure):
                raise QuerySourceError("target exposure is not finite")
            for previous_id in tuple(self._thermal_tracks):
                if previous_id != target_id and previous_id >> 14 == edict_index:
                    self._thermal_tracks.pop(previous_id, None)
            local = np.asarray(entity[:3], dtype=np.float64)
            if not np.isfinite(local).all():
                raise QuerySourceError("target geometry is not finite")
            world = eye + _local_to_world_vector(local, yaw, pitch)
            self._thermal_tracks[target_id] = _ThermalTrack(
                target_id=target_id,
                world_point=(float(world[0]), float(world[1]), float(world[2])),
                heat=min(1.0, exposure),
                observed_tick=tick,
            )
        self._expire_thermal(tick)
        return frozenset(killed)

    def _expire_thermal(self, tick: int) -> None:
        for target_id, track in tuple(self._thermal_tracks.items()):
            if track.observed_tick > tick:
                raise QuerySourceError("thermal evidence is future-dated")
            if tick - track.observed_tick > THERMAL_TTL_TICKS:
                del self._thermal_tracks[target_id]

    def _strongest_thermal(
        self, tick: int
    ) -> tuple[int, tuple[float, float, float], float, int] | None:
        self._expire_thermal(tick)
        if not self._thermal_tracks:
            return None
        track = max(
            self._thermal_tracks.values(),
            key=lambda item: (item.heat, item.observed_tick, -item.target_id),
        )
        return (track.target_id, track.world_point, track.heat, track.observed_tick)

    @staticmethod
    def _own_eye(telemetry: ClientTelemetry) -> tuple[float, float, float]:
        obs = telemetry.observation
        point = np.asarray(obs.self_state[:3], dtype=np.float64).copy()
        if not np.isfinite(point).all():
            raise QuerySourceError("own position is not finite")
        flags = 0
        self_debug = getattr(obs, "self_debug", None)
        if self_debug is not None and len(self_debug) >= 4:
            flags = int(self_debug[3])
        point[2] += -2.0 if flags & ML_ENTITY_DUCKED else 22.0
        return (float(point[0]), float(point[1]), float(point[2]))

    def _dyn_events(
        self, telemetry: ClientTelemetry, *, emit: bool
    ) -> tuple[tuple[int, str, tuple[float, float, float]], ...]:
        frame = int(telemetry.server_frame)
        if frame <= 0:
            raise QuerySourceError("Dyn events require a positive server frame")
        obs = telemetry.observation
        own = tuple(
            _require_finite(value, "own position")
            for value in obs.self_state[:3]
        )
        eye = self._own_eye(telemetry)
        events: list[tuple[int, str, tuple[float, float, float]]] = []

        def l2(point: tuple[float, float, float]) -> tuple[int, int, int]:
            return tuple(
                int(math.floor((axis - origin) / 64.0))
                for axis, origin in zip(point, self.atlas_origin)
            )

        def add(code: int, kind: str, point: tuple[float, float, float]) -> None:
            events.append(((frame << 3) | code, kind, point))

        # engagement: a one-shot positive own damage-dealt fact attributed to
        # the single current-or-recent (<= five-tick TTL) own-visible target
        # identity; zero or several candidates is ambiguous and omitted, and
        # the deposit point is always the target's world point, never the
        # shooter's.
        damage_dealt = _require_finite(
            getattr(obs, "reward_damage_dealt", 0.0), "damage dealt"
        )
        eligible_tracks = [
            track for track in self._thermal_tracks.values()
            if 0 <= frame - track.observed_tick <= THERMAL_TTL_TICKS
        ]
        if emit and damage_dealt > 0.0 and len(eligible_tracks) == 1:
            add(1, "engagement", eligible_tracks[0].world_point)
        # threat: a one-shot positive own damage-taken fact at own position.
        damage_taken = _require_finite(
            getattr(obs, "reward_damage_taken", 0.0), "damage taken"
        )
        if emit and damage_taken > 0.0:
            add(2, "threat", own)
        # opportunity: per target identity/epoch, only when that identity is
        # newly actionable (no remembered emission) or has moved to a new L2
        # 64-unit cell — never merely for staying visible.  Retired/expired
        # identities forget their cell and become newly actionable again.
        for target_id in tuple(self._opportunity_cells):
            if target_id not in self._thermal_tracks:
                del self._opportunity_cells[target_id]
        candidates = [
            track for track in self._thermal_tracks.values()
            if track.observed_tick == frame
            and self._opportunity_cells.get(track.target_id) != l2(track.world_point)
        ]
        if candidates:
            best = max(candidates, key=lambda item: (item.heat, -item.target_id))
            if emit:
                add(3, "opportunity", best.world_point)
            self._opportunity_cells[best.target_id] = l2(best.world_point)
        # self_fire: only an accepted authoritative FIRE echo, on its rising
        # edge — a continuously held trigger emits exactly once per press.
        debug = np.asarray(getattr(obs, "action_debug", ()), dtype=np.float64)
        if debug.size > int(ActionDebugIndex.FIRE):
            if not np.isfinite(debug).all():
                raise QuerySourceError("action echo facts are not finite")
            accepted_fire = bool(
                int(debug[ActionDebugIndex.FIRE])
                and int(debug[ActionDebugIndex.ACCEPTED])
            )
            if emit and accepted_fire and not self._last_echoed_fire:
                add(4, "self_fire", eye)
            self._last_echoed_fire = accepted_fire
        else:
            self._last_echoed_fire = False
        # death: one-shot public reward_death or public own-health terminal
        # edge at own position; repeated death-screen frames never re-emit.
        death_fact = _require_finite(getattr(obs, "reward_death", 0.0), "death")
        terminal = bool(death_fact > 0.0 or float(obs.self_state[6]) <= 0.0)
        if emit and terminal and not self._dead:
            add(5, "death", own)
        self._dead = terminal
        return tuple(events)

    # ---- SpatialQueryInputSource ------------------------------------------

    def sample(
        self, telemetry: ClientTelemetry, *, map_epoch: int
    ) -> SpatialQueryInputs:
        transaction = self.stage(
            telemetry, map_epoch=map_epoch, emit_dyn_events=True
        )
        transaction.commit()
        return transaction.inputs

    def stage(
        self,
        telemetry: ClientTelemetry,
        *,
        map_epoch: int,
        emit_dyn_events: bool,
    ) -> StagedSpatialQuery:
        snapshot = self._snapshot()
        try:
            inputs = self._sample_inner(
                telemetry,
                map_epoch=map_epoch,
                emit_dyn_events=bool(emit_dyn_events),
            )
        except Exception:
            # Fully transactional: a rejected frame leaves no trace in the
            # frame fence, beliefs, thermal, DPS, event edges, or step CAS.
            self._restore(snapshot)
            raise
        return StagedSpatialQuery(inputs=inputs, _source=self, _snapshot=snapshot)

    def _sample_inner(
        self,
        telemetry: ClientTelemetry,
        *,
        map_epoch: int,
        emit_dyn_events: bool,
    ) -> SpatialQueryInputs:
        self._check_identity(telemetry, map_epoch)
        self._resolve_own_pickup(telemetry)
        killed = self._observe_thermal(telemetry)
        survivability = self._survivability(telemetry)
        frame = int(telemetry.server_frame)
        beliefs = tuple(
            (belief.objective_id, belief.availability(frame))
            for belief in self._objectives
        )
        dyn_events = self._dyn_events(telemetry, emit=emit_dyn_events)
        # Killed identities were retained through event derivation for
        # same-frame engagement attribution; retire them deterministically
        # now.  Own death invalidates every remembered target track.
        for target_id in killed:
            self._thermal_tracks.pop(target_id, None)
            self._opportunity_cells.pop(target_id, None)
        if float(telemetry.observation.self_state[6]) <= 0.0:
            self._thermal_tracks.clear()
            self._opportunity_cells.clear()
        # Single environment-step CAS commit after all validation succeeded.
        expected_environment_steps = self._environment_steps
        self._environment_steps = expected_environment_steps + 1
        return SpatialQueryInputs(
            client_id=telemetry.client_id,
            client_epoch=self._transport_life_epoch(telemetry),
            map_name=telemetry.map_name,
            map_epoch=int(map_epoch),
            server_frame=frame,
            expected_environment_steps=expected_environment_steps,
            environment_steps=self._environment_steps,
            survivability=survivability,
            objective_beliefs=beliefs,
            dyn_events=dyn_events,
            thermal=self._strongest_thermal(frame),
            # Wire v6 carries no mover-blocker or dynamic-penalty channel for a
            # client, so "none observed" is this source's authoritative fact.
            blocked_nodes=(),
            dynamic_penalties=(),
            enabled_mover_blockers=(),
            time_to_impact_seconds=None,
        )
