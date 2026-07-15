"""Pure Quake II entity laws used by the static Atlas analyzer.

This module deliberately separates exact entity-string semantics from collision
and traversal admission.  It has no BSP parser, oracle, filesystem, process, or
runtime dependencies.  Callers may use the exact declared transforms and
candidate sets here, but mover collision remains unknown until the pinned
transformed collision oracle supplies evidence.

Entity properties stay as ordered ``(key, value)`` pairs.  Converting them to a
mapping is incorrect because Quake parses fields in source order and the last
``angle`` or ``angles`` key wins even though the two keys share one destination.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Generic, Protocol, Sequence, TypeVar


Vec3 = tuple[float, float, float]
ChunkKey = tuple[int, int, int]
Properties = Sequence[tuple[str, str]]

STANDING_MINS: Vec3 = (-16.0, -16.0, -24.0)
STANDING_MAXS: Vec3 = (16.0, 16.0, 32.0)
CROUCHED_MINS: Vec3 = (-16.0, -16.0, -24.0)
CROUCHED_MAXS: Vec3 = (16.0, 16.0, 4.0)

L0_CHUNK_HEADER_BYTES = 12 + 8 + 1
L0_BITPLANE_BYTES = 4096 // 8
L0_SCALAR_PLANE_BYTES = 4096
DEFAULT_MAX_L0_CHUNKS = 1_200
DEFAULT_MAX_L0_BYTES = 16 * 1024 * 1024
MAX_COUNTER = (1 << 64) - 1

_FLOAT_PREFIX = re.compile(
    r"^[\t\n\v\f\r ]*([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
)
_INT_PREFIX = re.compile(r"^[\t\n\v\f\r ]*([+-]?\d+)")


class Authority(str, Enum):
    """Whether a result is proven by the modeled engine law."""

    EXACT = "exact"
    UNKNOWN = "unknown"


T = TypeVar("T")


@dataclass(frozen=True)
class AuthorityResult(Generic[T]):
    """A value with an explicit Exact/Unknown admission boundary.

    Unknown results may retain a conservative value, such as the complete set
    of random train branches or a potential mover envelope.  Such a value is
    diagnostic/conservative and must not be promoted to deterministic collision
    or traversal authority.
    """

    authority: Authority
    value: T | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.authority is Authority.EXACT and self.reason:
            raise ValueError("exact results cannot carry an uncertainty reason")
        if self.authority is Authority.UNKNOWN and not self.reason:
            raise ValueError("unknown results require a reason")

    @classmethod
    def exact(cls, value: T) -> "AuthorityResult[T]":
        return cls(Authority.EXACT, value)

    @classmethod
    def unknown(cls, value: T | None, reason: str) -> "AuthorityResult[T]":
        return cls(Authority.UNKNOWN, value, reason)

    @property
    def is_exact(self) -> bool:
        return self.authority is Authority.EXACT


class EntityLike(Protocol):
    index: int
    classname: str
    properties: Sequence[tuple[str, str]]


@dataclass(frozen=True)
class Aabb:
    mins: Vec3
    maxs: Vec3


class GeometryClassification(str, Enum):
    REFERENCE_POSE = "reference_pose"
    ENDPOINT_POSE = "endpoint_pose"
    POTENTIAL_SWEPT_ENVELOPE = "potential_swept_envelope"


@dataclass(frozen=True)
class GeometryClaim:
    """A declared model-bounds claim, never collision authority by itself."""

    classification: GeometryClassification
    bounds: Aabb
    transform_authority: Authority = Authority.EXACT
    collision_authority: Authority = Authority.UNKNOWN
    collision_reason: str = "requires transformed collision-oracle evidence"


@dataclass(frozen=True)
class MovedirSemantics:
    parsed_angles: Vec3
    movedir: Vec3
    runtime_angles: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SlidingMoverSemantics:
    classname: str
    movedir: MovedirSemantics
    lip: int
    distance: float
    pos1: Vec3
    pos2: Vec3
    current_origin: Vec3
    start_open: bool
    reference_pose: GeometryClaim
    endpoint_pose: GeometryClaim
    potential_envelope: GeometryClaim


@dataclass(frozen=True)
class PlatformMoverSemantics:
    """Exact ``func_plat`` endpoint transforms from ``SP_func_plat``.

    Collision at either pose remains oracle-owned.  The union is a
    conservative potential-occupancy envelope for deciding whether a
    model-0-only movement replay depends on platform state.
    """

    lip: int
    height: int | None
    pos1: Vec3
    pos2: Vec3
    current_origin: Vec3
    target_disabled: bool
    reference_pose: GeometryClaim
    endpoint_pose: GeometryClaim
    potential_envelope: GeometryClaim


@dataclass(frozen=True)
class TrainCandidate:
    entity_index: int
    classname: str
    targetname: str
    next_target: str | None
    corner_origin: AuthorityResult[Vec3]
    train_origin: AuthorityResult[Vec3]
    teleport: bool


@dataclass(frozen=True)
class TrainTargetGroup:
    lookup: str
    eligible: tuple[TrainCandidate, ...]
    ignored_matching_entity_indices: tuple[int, ...]

    @property
    def selection_authority(self) -> Authority:
        return Authority.EXACT if len(self.eligible) == 1 else Authority.UNKNOWN


@dataclass(frozen=True)
class TrainTopology:
    initial_target: str | None
    groups: tuple[TrainTargetGroup, ...]
    open_chain_entity_indices: tuple[int, ...]
    unresolved_lookups: tuple[str, ...]
    consecutive_teleport_pairs: tuple[tuple[int, int], ...]
    unexpected_target_entity_indices: tuple[int, ...]

    def group(self, lookup: str) -> TrainTargetGroup | None:
        folded = lookup.casefold()
        return next((group for group in self.groups if group.lookup.casefold() == folded), None)


class RotationAxis(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"


@dataclass(frozen=True)
class RotatingMoverSemantics:
    classname: str
    axis: RotationAxis
    movedir: Vec3
    reverse: bool
    start_open: bool
    distance_degrees: int | None
    current_angles: Vec3
    start_angles: Vec3
    end_angles: Vec3 | None
    origin: Vec3
    reference_pose: GeometryClaim
    endpoint_pose: GeometryClaim | None
    potential_envelope: GeometryClaim


@dataclass(frozen=True)
class TriggerHurtSemantics:
    origin: Vec3
    linked_touch_bounds: Aabb
    standing_forbidden_origins: Aabb
    crouched_forbidden_origins: Aabb
    initially_active: bool
    toggleable: bool
    initial_standing_forbidden_origins: Aabb | None
    initial_crouched_forbidden_origins: Aabb | None
    runtime_standing_forbidden_origins: AuthorityResult[Aabb | None]
    runtime_crouched_forbidden_origins: AuthorityResult[Aabb | None]
    active_state_confidence_u16: int


class L0PlaneKind(str, Enum):
    BIT = "bit"
    SCALAR = "scalar"


@dataclass(frozen=True)
class L0ChunkAccount:
    key: ChunkKey
    bitplanes: tuple[str, ...] = ()
    scalar_planes: tuple[str, ...] = ()

    @property
    def encoded_bytes(self) -> int:
        return (
            L0_CHUNK_HEADER_BYTES
            + len(self.bitplanes) * L0_BITPLANE_BYTES
            + len(self.scalar_planes) * L0_SCALAR_PLANE_BYTES
        )


@dataclass(frozen=True)
class L0BudgetState:
    """Immutable prospective accounting for canonical decompressed L0 bytes."""

    chunks: tuple[L0ChunkAccount, ...] = ()
    max_chunks: int = DEFAULT_MAX_L0_CHUNKS
    max_bytes: int = DEFAULT_MAX_L0_BYTES

    def __post_init__(self) -> None:
        if self.max_chunks < 0 or self.max_bytes < 0:
            raise ValueError("L0 limits must be nonnegative")
        keys = [chunk.key for chunk in self.chunks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate L0 chunk account")
        if keys != sorted(keys, key=lambda key: (key[2], key[1], key[0])):
            raise ValueError("L0 chunk accounts are not in canonical (z,y,x) order")
        for chunk in self.chunks:
            if not chunk.bitplanes and not chunk.scalar_planes:
                raise ValueError("empty L0 chunks are noncanonical")
            if len(chunk.bitplanes) != len(set(chunk.bitplanes)):
                raise ValueError("duplicate L0 bitplane account")
            if len(chunk.scalar_planes) != len(set(chunk.scalar_planes)):
                raise ValueError("duplicate L0 scalar-plane account")
        if len(self.chunks) > self.max_chunks or self.encoded_bytes > self.max_bytes:
            raise ValueError("initial L0 accounting exceeds its limits")

    @property
    def encoded_bytes(self) -> int:
        return sum(chunk.encoded_bytes for chunk in self.chunks)

    def reserve(
        self, key: ChunkKey, kind: L0PlaneKind, plane: str,
    ) -> AuthorityResult["L0Reservation"]:
        """Return the exact prospective state before allocating a plane."""

        if not plane:
            return AuthorityResult.unknown(None, "an L0 plane name cannot be empty")
        if not isinstance(kind, L0PlaneKind):
            return AuthorityResult.unknown(None, "an L0 plane kind must be bit or scalar")
        if len(key) != 3 or any(isinstance(value, bool) or not isinstance(value, int) for value in key):
            return AuthorityResult.unknown(None, "an L0 chunk key must contain three integers")

        existing = next((chunk for chunk in self.chunks if chunk.key == key), None)
        added_chunks = int(existing is None)
        added_bytes = L0_CHUNK_HEADER_BYTES if existing is None else 0
        bitplanes = set(existing.bitplanes if existing else ())
        scalar_planes = set(existing.scalar_planes if existing else ())
        selected = bitplanes if kind is L0PlaneKind.BIT else scalar_planes
        if plane not in selected:
            selected.add(plane)
            added_bytes += L0_BITPLANE_BYTES if kind is L0PlaneKind.BIT else L0_SCALAR_PLANE_BYTES

        prospective_chunks = len(self.chunks) + added_chunks
        prospective_bytes = self.encoded_bytes + added_bytes
        if prospective_bytes > MAX_COUNTER:
            reservation = L0Reservation(
                accepted=False, state=self, added_chunks=added_chunks,
                added_bytes=added_bytes, prospective_chunks=prospective_chunks,
                prospective_bytes=prospective_bytes, rejection="L0 byte count overflow",
            )
            return AuthorityResult.exact(reservation)
        if prospective_chunks > self.max_chunks:
            reservation = L0Reservation(
                accepted=False, state=self, added_chunks=added_chunks,
                added_bytes=added_bytes, prospective_chunks=prospective_chunks,
                prospective_bytes=prospective_bytes,
                rejection=f"L0 chunk count {prospective_chunks} > {self.max_chunks}",
            )
            return AuthorityResult.exact(reservation)
        if prospective_bytes > self.max_bytes:
            reservation = L0Reservation(
                accepted=False, state=self, added_chunks=added_chunks,
                added_bytes=added_bytes, prospective_chunks=prospective_chunks,
                prospective_bytes=prospective_bytes,
                rejection=f"L0 bytes {prospective_bytes} > {self.max_bytes}",
            )
            return AuthorityResult.exact(reservation)

        replacement = L0ChunkAccount(
            key=key,
            bitplanes=tuple(sorted(bitplanes)),
            scalar_planes=tuple(sorted(scalar_planes)),
        )
        chunks = [chunk for chunk in self.chunks if chunk.key != key]
        chunks.append(replacement)
        chunks.sort(key=lambda chunk: (chunk.key[2], chunk.key[1], chunk.key[0]))
        state = L0BudgetState(tuple(chunks), self.max_chunks, self.max_bytes)
        reservation = L0Reservation(
            accepted=True, state=state, added_chunks=added_chunks,
            added_bytes=added_bytes, prospective_chunks=prospective_chunks,
            prospective_bytes=prospective_bytes,
        )
        return AuthorityResult.exact(reservation)


@dataclass(frozen=True)
class L0Reservation:
    accepted: bool
    state: L0BudgetState
    added_chunks: int
    added_bytes: int
    prospective_chunks: int
    prospective_bytes: int
    rejection: str = ""


def ordered_property(properties: Properties, key: str) -> str | None:
    """Return the last case-insensitive key without discarding source order."""

    folded = key.casefold()
    for candidate, value in reversed(properties):
        if candidate.casefold() == folded:
            return value
    return None


def _c_atof(value: str) -> float:
    match = _FLOAT_PREFIX.match(value)
    return float(match.group(1)) if match else 0.0


def _c_atoi(value: str) -> int:
    match = _INT_PREFIX.match(value)
    return int(match.group(1), 10) if match else 0


def _finite_vec(values: Sequence[float]) -> bool:
    return len(values) == 3 and all(math.isfinite(value) for value in values)


def _vector_property(properties: Properties, key: str, default: Vec3) -> AuthorityResult[Vec3]:
    raw = ordered_property(properties, key)
    if raw is None:
        return AuthorityResult.exact(default)
    pieces = raw.split()
    if len(pieces) != 3:
        return AuthorityResult.unknown(None, f"{key} does not contain three vector components")
    try:
        values = tuple(float(piece) for piece in pieces)
    except ValueError:
        return AuthorityResult.unknown(None, f"{key} contains a nonnumeric vector component")
    if not _finite_vec(values):
        return AuthorityResult.unknown(None, f"{key} contains a nonfinite vector component")
    return AuthorityResult.exact(values)  # type: ignore[arg-type]


def entity_angles(properties: Properties) -> AuthorityResult[Vec3]:
    """Reproduce ordered ``angle``/``angles`` parsing into ``s.angles``."""

    selected: tuple[str, str] | None = None
    for key, value in properties:
        if key.casefold() in {"angle", "angles"}:
            selected = (key.casefold(), value)
    if selected is None:
        return AuthorityResult.exact((0.0, 0.0, 0.0))
    key, value = selected
    if key == "angle":
        yaw = _c_atof(value)
        if not math.isfinite(yaw):
            return AuthorityResult.unknown(None, "angle is nonfinite")
        return AuthorityResult.exact((0.0, yaw, 0.0))
    pieces = value.split()
    if len(pieces) != 3:
        return AuthorityResult.unknown(None, "angles does not contain three vector components")
    try:
        values = tuple(float(piece) for piece in pieces)
    except ValueError:
        return AuthorityResult.unknown(None, "angles contains a nonnumeric vector component")
    if not _finite_vec(values):
        return AuthorityResult.unknown(None, "angles contains a nonfinite vector component")
    return AuthorityResult.exact(values)  # type: ignore[arg-type]


def _angle_forward(angles: Vec3) -> Vec3:
    pitch, yaw, _roll = (math.radians(value) for value in angles)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return cp * cy, cp * sy, -sp


def set_movedir(properties: Properties) -> AuthorityResult[MovedirSemantics]:
    """Reproduce Lithium ``G_SetMovedir``, including special -1/-2 angles."""

    parsed = entity_angles(properties)
    if not parsed.is_exact or parsed.value is None:
        return AuthorityResult.unknown(None, parsed.reason)
    angles = parsed.value
    if angles == (0.0, -1.0, 0.0):
        movedir = (0.0, 0.0, 1.0)
    elif angles == (0.0, -2.0, 0.0):
        movedir = (0.0, 0.0, -1.0)
    else:
        movedir = _angle_forward(angles)
    return AuthorityResult.exact(MovedirSemantics(angles, movedir))


def _entity_classname(entity: EntityLike) -> str:
    return ordered_property(entity.properties, "classname") or entity.classname


def _entity_origin(entity: EntityLike) -> AuthorityResult[Vec3]:
    return _vector_property(entity.properties, "origin", (0.0, 0.0, 0.0))


def _spawnflags(entity: EntityLike) -> int:
    value = ordered_property(entity.properties, "spawnflags")
    return _c_atoi(value) if value is not None else 0


def _translated_aabb(mins: Vec3, maxs: Vec3, origin: Vec3) -> Aabb:
    return Aabb(
        tuple(mins[axis] + origin[axis] for axis in range(3)),  # type: ignore[arg-type]
        tuple(maxs[axis] + origin[axis] for axis in range(3)),  # type: ignore[arg-type]
    )


def _union_aabb(first: Aabb, second: Aabb) -> Aabb:
    return Aabb(
        tuple(min(first.mins[axis], second.mins[axis]) for axis in range(3)),  # type: ignore[arg-type]
        tuple(max(first.maxs[axis], second.maxs[axis]) for axis in range(3)),  # type: ignore[arg-type]
    )


def _model_bounds(mins: Sequence[float], maxs: Sequence[float]) -> AuthorityResult[Aabb]:
    lower = tuple(float(value) for value in mins)
    upper = tuple(float(value) for value in maxs)
    if not _finite_vec(lower) or not _finite_vec(upper):
        return AuthorityResult.unknown(None, "model bounds must be finite vec3 values")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        return AuthorityResult.unknown(None, "model mins exceed model maxs")
    return AuthorityResult.exact(Aabb(lower, upper))  # type: ignore[arg-type]


def sliding_mover_semantics(
    entity: EntityLike, model_mins: Sequence[float], model_maxs: Sequence[float],
) -> AuthorityResult[SlidingMoverSemantics]:
    """Resolve ``func_door``/``func_button``/``func_water`` endpoint laws.

    ``model_mins`` and ``model_maxs`` must be the inline cmodel bounds used by
    ``gi.setmodel``.  Their one-unit collision-module expansion is therefore
    already reflected in the size used by the engine.
    """

    classname = _entity_classname(entity).casefold()
    default_lips = {"func_door": 8, "func_button": 4, "func_water": 0}
    if classname not in default_lips:
        return AuthorityResult.unknown(None, f"unsupported sliding mover {classname!r}")
    model = _model_bounds(model_mins, model_maxs)
    origin = _entity_origin(entity)
    movedir = set_movedir(entity.properties)
    for result in (model, origin, movedir):
        if not result.is_exact or result.value is None:
            return AuthorityResult.unknown(None, result.reason)
    model_value = model.value
    origin_value = origin.value
    movedir_value = movedir.value
    size = tuple(model_value.maxs[axis] - model_value.mins[axis] for axis in range(3))

    raw_lip = ordered_property(entity.properties, "lip")
    lip = _c_atoi(raw_lip) if raw_lip is not None else 0
    if lip == 0 and default_lips[classname]:
        lip = default_lips[classname]
    distance = sum(abs(movedir_value.movedir[axis]) * size[axis] for axis in range(3)) - lip
    displacement = tuple(distance * movedir_value.movedir[axis] for axis in range(3))
    closed = origin_value
    opened = tuple(closed[axis] + displacement[axis] for axis in range(3))
    start_open = bool(_spawnflags(entity) & 1) and classname in {"func_door", "func_water"}
    if start_open:
        pos1, pos2, current = opened, closed, opened
    else:
        pos1, pos2, current = closed, opened, closed

    reference_bounds = _translated_aabb(model_value.mins, model_value.maxs, pos1)
    endpoint_bounds = _translated_aabb(model_value.mins, model_value.maxs, pos2)
    return AuthorityResult.exact(SlidingMoverSemantics(
        classname=classname,
        movedir=movedir_value,
        lip=lip,
        distance=distance,
        pos1=pos1,  # type: ignore[arg-type]
        pos2=pos2,  # type: ignore[arg-type]
        current_origin=current,  # type: ignore[arg-type]
        start_open=start_open,
        reference_pose=GeometryClaim(GeometryClassification.REFERENCE_POSE, reference_bounds),
        endpoint_pose=GeometryClaim(GeometryClassification.ENDPOINT_POSE, endpoint_bounds),
        potential_envelope=GeometryClaim(
            GeometryClassification.POTENTIAL_SWEPT_ENVELOPE,
            _union_aabb(reference_bounds, endpoint_bounds),
        ),
    ))


def platform_mover_semantics(
    entity: EntityLike, model_mins: Sequence[float], model_maxs: Sequence[float],
) -> AuthorityResult[PlatformMoverSemantics]:
    """Resolve the complete vertical pose envelope of a Quake II ``func_plat``.

    This reproduces the position law in ``SP_func_plat``: the authored model
    is the top pose, the bottom pose is lower by explicit integer ``height``
    or by model height minus integer ``lip`` (default eight).  A platform with
    a ``targetname`` starts disabled at the top pose; every other platform is
    linked at the bottom pose.  Trigger timing and current state do not change
    the conservative union of those two translated poses.
    """

    classname = _entity_classname(entity).casefold()
    if classname != "func_plat":
        return AuthorityResult.unknown(None, f"unsupported platform mover {classname!r}")
    model = _model_bounds(model_mins, model_maxs)
    origin = _entity_origin(entity)
    for result in (model, origin):
        if not result.is_exact or result.value is None:
            return AuthorityResult.unknown(None, result.reason)
    model_value = model.value
    pos1 = origin.value

    raw_lip = ordered_property(entity.properties, "lip")
    lip = _c_atoi(raw_lip) if raw_lip is not None else 0
    if lip == 0:
        lip = 8
    raw_height = ordered_property(entity.properties, "height")
    parsed_height = _c_atoi(raw_height) if raw_height is not None else 0
    height = parsed_height if parsed_height != 0 else None
    travel = float(
        parsed_height
        if parsed_height != 0
        else (model_value.maxs[2] - model_value.mins[2]) - lip
    )
    if not math.isfinite(travel):
        return AuthorityResult.unknown(None, "platform travel is nonfinite")
    pos2 = (pos1[0], pos1[1], pos1[2] - travel)
    target_disabled = ordered_property(entity.properties, "targetname") is not None
    current = pos1 if target_disabled else pos2

    top_bounds = _translated_aabb(model_value.mins, model_value.maxs, pos1)
    bottom_bounds = _translated_aabb(model_value.mins, model_value.maxs, pos2)
    return AuthorityResult.exact(PlatformMoverSemantics(
        lip=lip,
        height=height,
        pos1=pos1,
        pos2=pos2,
        current_origin=current,
        target_disabled=target_disabled,
        reference_pose=GeometryClaim(GeometryClassification.REFERENCE_POSE, top_bounds),
        endpoint_pose=GeometryClaim(GeometryClassification.ENDPOINT_POSE, bottom_bounds),
        potential_envelope=GeometryClaim(
            GeometryClassification.POTENTIAL_SWEPT_ENVELOPE,
            _union_aabb(top_bounds, bottom_bounds),
        ),
    ))


def train_topology(
    train: EntityLike, entities: Sequence[EntityLike], train_mins: Sequence[float],
) -> AuthorityResult[TrainTopology]:
    """Build the exact candidate topology without choosing random duplicates."""

    mins = tuple(float(value) for value in train_mins)
    if not _finite_vec(mins):
        return AuthorityResult.unknown(None, "train mins must be a finite vec3")
    indices = [entity.index for entity in entities]
    if any(right <= left for left, right in zip(indices, indices[1:])):
        return AuthorityResult.unknown(None, "entities are not in strict edict order")

    initial_target = ordered_property(train.properties, "target")
    if initial_target is None:
        return AuthorityResult.exact(TrainTopology(None, (), (), (), (), ()))

    pending = [initial_target]
    seen: set[str] = set()
    groups: list[TrainTargetGroup] = []
    open_entities: set[int] = set()
    unresolved: dict[str, None] = {}
    unexpected: set[int] = set()
    malformed: list[str] = []

    while pending:
        lookup = pending.pop(0)
        folded = lookup.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        matches = [
            entity for entity in entities
            if (targetname := ordered_property(entity.properties, "targetname")) is not None
            and targetname.casefold() == folded
        ]
        eligible_entities = matches[:8]
        ignored = tuple(entity.index for entity in matches[8:])
        candidates: list[TrainCandidate] = []
        for entity in eligible_entities:
            classname = _entity_classname(entity).casefold()
            if classname != "path_corner":
                unexpected.add(entity.index)
            corner_origin = _entity_origin(entity)
            if corner_origin.is_exact and corner_origin.value is not None:
                destination = tuple(corner_origin.value[axis] - mins[axis] for axis in range(3))
                train_origin = AuthorityResult.exact(destination)  # type: ignore[arg-type]
            else:
                malformed.append(f"entity {entity.index}: {corner_origin.reason}")
                train_origin = AuthorityResult.unknown(None, corner_origin.reason)
            next_target = ordered_property(entity.properties, "target")
            candidate = TrainCandidate(
                entity_index=entity.index,
                classname=classname,
                targetname=ordered_property(entity.properties, "targetname") or "",
                next_target=next_target,
                corner_origin=corner_origin,
                train_origin=train_origin,
                teleport=bool(_spawnflags(entity) & 1),
            )
            candidates.append(candidate)
            if next_target is None:
                open_entities.add(entity.index)
            elif next_target.casefold() not in seen and next_target.casefold() not in {
                item.casefold() for item in pending
            }:
                pending.append(next_target)
        if not candidates:
            unresolved[lookup] = None
        groups.append(TrainTargetGroup(lookup, tuple(candidates), ignored))

    group_by_name = {group.lookup.casefold(): group for group in groups}
    teleport_pairs: set[tuple[int, int]] = set()
    for group in groups:
        for candidate in group.eligible:
            if not candidate.teleport or candidate.next_target is None:
                continue
            next_group = group_by_name.get(candidate.next_target.casefold())
            if next_group is None:
                continue
            for following in next_group.eligible:
                if following.teleport:
                    teleport_pairs.add((candidate.entity_index, following.entity_index))

    topology = TrainTopology(
        initial_target=initial_target,
        groups=tuple(groups),
        open_chain_entity_indices=tuple(sorted(open_entities)),
        unresolved_lookups=tuple(unresolved),
        consecutive_teleport_pairs=tuple(sorted(teleport_pairs)),
        unexpected_target_entity_indices=tuple(sorted(unexpected)),
    )
    duplicate_groups = [group.lookup for group in groups if len(group.eligible) > 1]
    reasons = []
    if duplicate_groups:
        reasons.append("random G_PickTarget selection for " + ", ".join(repr(name) for name in duplicate_groups))
    if unexpected:
        reasons.append("target lookup includes non-path_corner entities")
    if malformed:
        reasons.extend(malformed)
    if reasons:
        return AuthorityResult.unknown(topology, "; ".join(reasons))
    return AuthorityResult.exact(topology)


def _rotation_axis(classname: str, spawnflags: int) -> tuple[RotationAxis, Vec3]:
    if classname == "func_rotating":
        if spawnflags & 4:
            return RotationAxis.X, (0.0, 0.0, 1.0)
        if spawnflags & 8:
            return RotationAxis.Y, (1.0, 0.0, 0.0)
    else:
        if spawnflags & 64:
            return RotationAxis.X, (0.0, 0.0, 1.0)
        if spawnflags & 128:
            return RotationAxis.Y, (1.0, 0.0, 0.0)
    return RotationAxis.Z, (0.0, 1.0, 0.0)


def _angle_basis(angles: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    pitch, yaw, roll = (math.radians(value) for value in angles)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    forward = (cp * cy, cp * sy, -sp)
    right = (
        -sr * sp * cy + cr * sy,
        -sr * sp * sy - cr * cy,
        -sr * cp,
    )
    up = (
        cr * sp * cy + sr * sy,
        cr * sp * sy - sr * cy,
        cr * cp,
    )
    return forward, right, up


def _rotated_model_aabb(model: Aabb, origin: Vec3, angles: Vec3) -> Aabb:
    forward, right, up = _angle_basis(angles)
    points = []
    for x in (model.mins[0], model.maxs[0]):
        for y in (model.mins[1], model.maxs[1]):
            for z in (model.mins[2], model.maxs[2]):
                points.append(tuple(
                    origin[axis] + forward[axis] * x - right[axis] * y + up[axis] * z
                    for axis in range(3)
                ))
    return Aabb(
        tuple(min(point[axis] for point in points) for axis in range(3)),  # type: ignore[arg-type]
        tuple(max(point[axis] for point in points) for axis in range(3)),  # type: ignore[arg-type]
    )


def rotating_mover_semantics(
    entity: EntityLike, model_mins: Sequence[float], model_maxs: Sequence[float],
) -> AuthorityResult[RotatingMoverSemantics]:
    """Resolve continuous and door-rotating entity laws and classify claims."""

    classname = _entity_classname(entity).casefold()
    if classname not in {"func_rotating", "func_door_rotating"}:
        return AuthorityResult.unknown(None, f"unsupported rotating mover {classname!r}")
    model = _model_bounds(model_mins, model_maxs)
    origin = _entity_origin(entity)
    if not model.is_exact or model.value is None:
        return AuthorityResult.unknown(None, model.reason)
    if not origin.is_exact or origin.value is None:
        return AuthorityResult.unknown(None, origin.reason)

    flags = _spawnflags(entity)
    axis, movedir = _rotation_axis(classname, flags)
    reverse = bool(flags & 2)
    if reverse:
        movedir = tuple(-value for value in movedir)  # type: ignore[assignment]
    start_open = False
    distance: int | None = None
    endpoint_angles: Vec3 | None = None

    if classname == "func_rotating":
        parsed_angles = entity_angles(entity.properties)
        if not parsed_angles.is_exact or parsed_angles.value is None:
            return AuthorityResult.unknown(None, parsed_angles.reason)
        current_angles = start_angles = parsed_angles.value
    else:
        raw_distance = ordered_property(entity.properties, "distance")
        distance = _c_atoi(raw_distance) if raw_distance is not None else 0
        if distance == 0:
            distance = 90
        pos1: Vec3 = (0.0, 0.0, 0.0)
        pos2 = tuple(distance * value for value in movedir)
        start_open = bool(flags & 1)
        if start_open:
            current_angles = pos2  # type: ignore[assignment]
            pos1, pos2 = pos2, pos1
            movedir = tuple(-value for value in movedir)  # type: ignore[assignment]
        else:
            current_angles = pos1
        start_angles = pos1  # type: ignore[assignment]
        endpoint_angles = pos2  # type: ignore[assignment]

    reference_bounds = _rotated_model_aabb(model.value, origin.value, current_angles)
    endpoint_claim = None
    if endpoint_angles is not None:
        endpoint_claim = GeometryClaim(
            GeometryClassification.ENDPOINT_POSE,
            _rotated_model_aabb(model.value, origin.value, endpoint_angles),
        )

    radius = max(
        math.sqrt(x * x + y * y + z * z)
        for x in (model.value.mins[0], model.value.maxs[0])
        for y in (model.value.mins[1], model.value.maxs[1])
        for z in (model.value.mins[2], model.value.maxs[2])
    )
    envelope = Aabb(
        tuple(value - radius for value in origin.value),  # type: ignore[arg-type]
        tuple(value + radius for value in origin.value),  # type: ignore[arg-type]
    )
    return AuthorityResult.exact(RotatingMoverSemantics(
        classname=classname,
        axis=axis,
        movedir=movedir,
        reverse=reverse,
        start_open=start_open,
        distance_degrees=distance,
        current_angles=current_angles,
        start_angles=start_angles,
        end_angles=endpoint_angles,
        origin=origin.value,
        reference_pose=GeometryClaim(GeometryClassification.REFERENCE_POSE, reference_bounds),
        endpoint_pose=endpoint_claim,
        potential_envelope=GeometryClaim(
            GeometryClassification.POTENTIAL_SWEPT_ENVELOPE, envelope,
        ),
    ))


def _forbidden_origin_bounds(
    raw_model: Aabb, origin: Vec3, hull_mins: Vec3, hull_maxs: Vec3,
) -> Aabb:
    # CMod_LoadSubmodels expands the trigger by one, SV_LinkEdict expands it by
    # another one, and the querying player's linked AABB contributes one more.
    return Aabb(
        tuple(
            origin[axis] + raw_model.mins[axis] - hull_maxs[axis] - 3.0
            for axis in range(3)
        ),  # type: ignore[arg-type]
        tuple(
            origin[axis] + raw_model.maxs[axis] - hull_mins[axis] + 3.0
            for axis in range(3)
        ),  # type: ignore[arg-type]
    )


def trigger_hurt_semantics(
    entity: EntityLike, raw_model_mins: Sequence[float], raw_model_maxs: Sequence[float],
) -> AuthorityResult[TriggerHurtSemantics]:
    """Reproduce linked-AABB trigger contact and active-state confidence."""

    if _entity_classname(entity).casefold() != "trigger_hurt":
        return AuthorityResult.unknown(None, "entity is not trigger_hurt")
    model = _model_bounds(raw_model_mins, raw_model_maxs)
    origin = _entity_origin(entity)
    if not model.is_exact or model.value is None:
        return AuthorityResult.unknown(None, model.reason)
    if not origin.is_exact or origin.value is None:
        return AuthorityResult.unknown(None, origin.reason)
    flags = _spawnflags(entity)
    initially_active = not bool(flags & 1)
    toggleable = bool(flags & 2)
    linked = Aabb(
        tuple(origin.value[axis] + model.value.mins[axis] - 2.0 for axis in range(3)),  # type: ignore[arg-type]
        tuple(origin.value[axis] + model.value.maxs[axis] + 2.0 for axis in range(3)),  # type: ignore[arg-type]
    )
    standing = _forbidden_origin_bounds(model.value, origin.value, STANDING_MINS, STANDING_MAXS)
    crouched = _forbidden_origin_bounds(model.value, origin.value, CROUCHED_MINS, CROUCHED_MAXS)
    initial_standing = standing if initially_active else None
    initial_crouched = crouched if initially_active else None
    if toggleable:
        runtime_standing = AuthorityResult.unknown(
            standing,
            "trigger_hurt TOGGLE state requires current runtime activation evidence",
        )
        runtime_crouched = AuthorityResult.unknown(
            crouched,
            "trigger_hurt TOGGLE state requires current runtime activation evidence",
        )
        confidence = 0
    else:
        runtime_standing = AuthorityResult.exact(initial_standing)
        runtime_crouched = AuthorityResult.exact(initial_crouched)
        confidence = 0xFFFF
    return AuthorityResult.exact(TriggerHurtSemantics(
        origin=origin.value,
        linked_touch_bounds=linked,
        standing_forbidden_origins=standing,
        crouched_forbidden_origins=crouched,
        initially_active=initially_active,
        toggleable=toggleable,
        initial_standing_forbidden_origins=initial_standing,
        initial_crouched_forbidden_origins=initial_crouched,
        runtime_standing_forbidden_origins=runtime_standing,
        runtime_crouched_forbidden_origins=runtime_crouched,
        active_state_confidence_u16=confidence,
    ))
