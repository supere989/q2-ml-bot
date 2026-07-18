"""Fail-closed Quake II ``trigger_teleport`` Atlas edge authority.

The entity resolver in this module is pure: it derives only the deterministic
portion of the Lithium/Yamagi teleporter law from an admitted entity string and
inline-model catalog.  It never turns entity metadata into collision evidence.

The collision orchestrator accepts only resolved, unique
``trigger_teleport -> info_teleport_destination`` links.  A traversal edge is
materialized only after the pinned CM oracle proves all of the following:

* a stance-clear Atlas L1 player hull intersects the exact inline trigger model
* the runtime destination origin is clear for the same stance
* a supported, allowed-normal landing exists below that origin
* the landing maps to a materialized, stance-clear Atlas L1 node

The engine's ``G_Find`` returns the first case-insensitive ``targetname``
match.  Atlas deliberately requires exactly one match: maps whose result would
depend on edict order are retained as Unknown diagnostics and get no edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Protocol, Sequence

from .atlas_entity_semantics import Authority, entity_angles


Vec3 = tuple[float, float, float]
L1Index = tuple[int, int, int]

MASK_PLAYERSOLID = 33_619_971
EVIDENCE_CM_TRACE_V1 = 1
TELEPORTER_VALIDATION_VERSION = 1
TELEPORTER_COST_Q8 = 4_096
TELEPORTER_RISK_U16 = 0
TELEPORT_DESTINATION_Z_OFFSET = 16.0
SUPPORT_PROBE_DEPTH = 96.0
MIN_SUPPORT_NORMAL_Z = 0.7
MAX_CM_ONLY_SUPPORT_DROP = 18.0

STANDING_MINS: Vec3 = (-16.0, -16.0, -24.0)
STANDING_MAXS: Vec3 = (16.0, 16.0, 32.0)
CROUCHED_MINS: Vec3 = (-16.0, -16.0, -24.0)
CROUCHED_MAXS: Vec3 = (16.0, 16.0, 4.0)


class EntityLike(Protocol):
    index: int
    classname: str
    properties: Sequence[tuple[str, str]]


class ModelLike(Protocol):
    index: int
    mins: Vec3
    maxs: Vec3
    headnode: int


class NavNodeLike(Protocol):
    position: Vec3
    standing_clear: bool
    crouched_clear: bool
    supported: bool


CollisionBatch = Callable[[Sequence[dict[str, object]]], Sequence[Mapping[str, object]]]


@dataclass(frozen=True)
class TeleporterLink:
    """The exact entity-law portion of one trigger teleporter."""

    source_entity_index: int
    source_model_index: int
    source_headnode: int
    source_origin: Vec3
    source_angles: Vec3
    source_bounds_mins: Vec3
    source_bounds_maxs: Vec3
    target: str
    destination_entity_index: int
    destination_origin: Vec3
    arrival_origin: Vec3


@dataclass(frozen=True)
class TeleporterOmission:
    source_entity_index: int
    reason: str
    destination_entity_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "source_entity_index": self.source_entity_index,
            "authority": "unknown",
            "reason": self.reason,
        }
        if self.destination_entity_index is not None:
            value["destination_entity_index"] = self.destination_entity_index
        return value


@dataclass(frozen=True)
class TeleporterResolution:
    links: tuple[TeleporterLink, ...]
    omissions: tuple[TeleporterOmission, ...]


@dataclass(frozen=True)
class TeleporterEdgeAnalysis:
    edges: tuple[dict[str, object], ...]
    omissions: tuple[TeleporterOmission, ...]
    links: tuple[TeleporterLink, ...]

    def report(self) -> dict[str, object]:
        return {
            "schema": "q2-atlas-teleporter-analysis-v1",
            "validation_version": TELEPORTER_VALIDATION_VERSION,
            "resolved_link_count": len(self.links),
            "admitted_edge_count": len(self.edges),
            "authority": "exact-cm-entity-law" if self.edges else "omitted",
            "links": [
                {
                    "source_entity_index": link.source_entity_index,
                    "source_model_index": link.source_model_index,
                    "destination_entity_index": link.destination_entity_index,
                    "target": link.target,
                    "arrival_origin_milliunits": [
                        round(axis * 1000.0) for axis in link.arrival_origin
                    ],
                }
                for link in self.links
            ],
            "omissions": [item.as_dict() for item in self.omissions],
        }


def _last_property(entity: EntityLike, key: str) -> str:
    for candidate, value in reversed(entity.properties):
        if candidate == key:
            return value
    return ""


def _parse_origin(
    entity: EntityLike, *, default: Vec3 | None = None,
) -> Vec3 | None:
    raw = _last_property(entity, "origin")
    if not raw and default is not None:
        return default
    words = raw.split()
    if len(words) != 3:
        return None
    try:
        result = tuple(float(word) for word in words)
    except ValueError:
        return None
    if not all(math.isfinite(axis) for axis in result):
        return None
    return result  # type: ignore[return-value]


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _target_key(value: str) -> str | None:
    """Match the byte-oriented engine comparison only for admitted ASCII."""

    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return None
    return value.lower()


def resolve_trigger_teleporters(
    entities: Sequence[EntityLike],
    brush_submodels: Sequence[Mapping[str, object]],
    models: Sequence[ModelLike],
) -> TeleporterResolution:
    """Resolve deterministic trigger/destination pairs without collision claims.

    Only the old-map ``trigger_teleport`` path is admitted in v1.  The distinct
    ``misc_teleporter`` path owns a spawned trigger, a different destination
    class, and a different Z law; conflating the two would guess engine state.
    """

    ordered_entities = tuple(sorted(entities, key=lambda item: item.index))
    by_index = {entity.index: entity for entity in ordered_entities}
    if len(by_index) != len(ordered_entities):
        return TeleporterResolution(
            (), (TeleporterOmission(0, "duplicate entity indices"),),
        )
    model_by_index = {model.index: model for model in models}
    submodels: dict[int, list[int]] = {}
    for record in brush_submodels:
        entity_index = _strict_int(record.get("entity_index"))
        model_index = _strict_int(record.get("model_index"))
        if entity_index is None or model_index is None:
            continue
        submodels.setdefault(entity_index, []).append(model_index)

    destinations: dict[str, list[EntityLike]] = {}
    for entity in ordered_entities:
        targetname = _last_property(entity, "targetname")
        key = _target_key(targetname) if targetname else None
        if key is not None:
            destinations.setdefault(key, []).append(entity)

    links: list[TeleporterLink] = []
    omissions: list[TeleporterOmission] = []
    for source in ordered_entities:
        if source.classname.casefold() != "trigger_teleport":
            continue
        target = _last_property(source, "target")
        if not target:
            omissions.append(TeleporterOmission(source.index, "missing target"))
            continue
        target_key = _target_key(target)
        if target_key is None:
            omissions.append(TeleporterOmission(
                source.index, "non-ASCII target comparison is not admitted",
            ))
            continue
        matches = destinations.get(target_key, [])
        if not matches:
            omissions.append(TeleporterOmission(source.index, "target destination is missing"))
            continue
        if len(matches) != 1:
            omissions.append(TeleporterOmission(
                source.index, "targetname is ambiguous under case-insensitive G_Find",
            ))
            continue
        destination = matches[0]
        if destination.classname.casefold() != "info_teleport_destination":
            omissions.append(TeleporterOmission(
                source.index,
                "unique target is not info_teleport_destination",
                destination.index,
            ))
            continue
        destination_origin = _parse_origin(destination)
        if destination_origin is None:
            omissions.append(TeleporterOmission(
                source.index, "destination origin is malformed", destination.index,
            ))
            continue
        destination_angles = entity_angles(destination.properties)
        if destination_angles.authority is not Authority.EXACT:
            omissions.append(TeleporterOmission(
                source.index, "destination angles are malformed", destination.index,
            ))
            continue
        source_origin = _parse_origin(source, default=(0.0, 0.0, 0.0))
        if source_origin is None:
            omissions.append(TeleporterOmission(
                source.index, "trigger origin is malformed", destination.index,
            ))
            continue
        source_angles = entity_angles(source.properties)
        if (
            source_angles.authority is not Authority.EXACT
            or source_angles.value is None
        ):
            omissions.append(TeleporterOmission(
                source.index, "trigger angles are malformed", destination.index,
            ))
            continue
        # The CM oracle supports transformed inline models, but the candidate
        # AABB below intentionally admits only the exact axis-aligned linked
        # bounds law.  Rotated trigger volumes remain Unknown rather than using
        # a loose sphere that could create a false source contact.
        if source_angles.value != (0.0, 0.0, 0.0):
            omissions.append(TeleporterOmission(
                source.index, "rotated trigger bounds are not admitted in teleporter v1",
                destination.index,
            ))
            continue
        source_models = submodels.get(source.index, [])
        if len(source_models) != 1:
            omissions.append(TeleporterOmission(
                source.index, "trigger requires exactly one inline brush model",
                destination.index,
            ))
            continue
        model_index = source_models[0]
        if _last_property(source, "model") != f"*{model_index}":
            omissions.append(TeleporterOmission(
                source.index, "trigger model property differs from inline catalog",
                destination.index,
            ))
            continue
        model = model_by_index.get(model_index)
        if model is None or model.index <= 0 or model.headnode < 0:
            omissions.append(TeleporterOmission(
                source.index, "trigger inline model is unavailable", destination.index,
            ))
            continue
        bounds_values = (*model.mins, *model.maxs)
        if not all(math.isfinite(float(axis)) for axis in bounds_values) or any(
            model.mins[axis] > model.maxs[axis] for axis in range(3)
        ):
            omissions.append(TeleporterOmission(
                source.index, "trigger inline model bounds are malformed", destination.index,
            ))
            continue
        arrival = (
            destination_origin[0],
            destination_origin[1],
            destination_origin[2] + TELEPORT_DESTINATION_Z_OFFSET,
        )
        links.append(TeleporterLink(
            source_entity_index=source.index,
            source_model_index=model.index,
            source_headnode=model.headnode,
            source_origin=source_origin,
            source_angles=source_angles.value,
            # CM inline cmodels expand raw dmodel bounds by one unit.  Matching
            # that exact admitted bound prevents the candidate filter from
            # hiding a real transformed-CM contact at a model boundary.
            source_bounds_mins=tuple(
                model.mins[axis] - 1.0 + source_origin[axis] for axis in range(3)
            ),  # type: ignore[arg-type]
            source_bounds_maxs=tuple(
                model.maxs[axis] + 1.0 + source_origin[axis] for axis in range(3)
            ),  # type: ignore[arg-type]
            target=target,
            destination_entity_index=destination.index,
            destination_origin=destination_origin,
            arrival_origin=arrival,
        ))
    return TeleporterResolution(
        tuple(sorted(links, key=lambda item: item.source_entity_index)),
        tuple(sorted(
            omissions,
            key=lambda item: (
                item.source_entity_index,
                -1 if item.destination_entity_index is None else item.destination_entity_index,
                item.reason,
            ),
        )),
    )


def teleporter_seed_points(
    resolution: TeleporterResolution,
    collision_call: CollisionBatch,
) -> tuple[Vec3, ...]:
    """Return only CM-clear, step-supported destinations as L1 flood seeds.

    Entity resolution alone is not seed authority.  In particular, a clear
    arrival with a deep floor below it cannot seed a disconnected component
    until an exact Pmove/fall replay exists.  The edge pass repeats these
    requests so its final evidence remains independently fail closed.
    """

    points: list[Vec3] = []
    for link in resolution.links:
        points.extend(_destination_landings(link, collision_call).values())
    return tuple(dict.fromkeys(points))


def _grid_index(point: Sequence[float], origin: Sequence[int]) -> L1Index:
    return tuple(
        math.floor((point[axis] - origin[axis]) / 16.0) for axis in range(3)
    )  # type: ignore[return-value]


def _box_request(
    identifier: str,
    start: Vec3,
    end: Vec3,
    mins: Vec3,
    maxs: Vec3,
) -> dict[str, object]:
    return {
        "id": identifier,
        "op": "box_trace",
        "start": list(start),
        "end": list(end),
        "mins": list(mins),
        "maxs": list(maxs),
        "mask": MASK_PLAYERSOLID,
    }


def _trace_bool(response: Mapping[str, object], key: str) -> bool | None:
    value = response.get(key)
    return value if isinstance(value, bool) else None


def _trace_fraction(response: Mapping[str, object]) -> float | None:
    value = response.get("fraction")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and 0.0 <= result <= 1.0 else None


def _trace_vec3(response: Mapping[str, object], key: str) -> Vec3 | None:
    value = response.get(key)
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(
        isinstance(axis, bool) or not isinstance(axis, (int, float))
        or not math.isfinite(float(axis))
        for axis in value
    ):
        return None
    return tuple(float(axis) for axis in value)  # type: ignore[return-value]


def _support_normal(response: Mapping[str, object]) -> Vec3 | None:
    plane = response.get("plane")
    if not isinstance(plane, Mapping):
        return None
    return _trace_vec3(plane, "normal")


def _strict_clear(response: Mapping[str, object]) -> bool:
    startsolid = _trace_bool(response, "startsolid")
    allsolid = _trace_bool(response, "allsolid")
    fraction = _trace_fraction(response)
    return startsolid is False and allsolid is False and fraction == 1.0


def _strict_supported(
    response: Mapping[str, object], arrival_origin: Vec3,
) -> Vec3 | None:
    startsolid = _trace_bool(response, "startsolid")
    allsolid = _trace_bool(response, "allsolid")
    fraction = _trace_fraction(response)
    endpos = _trace_vec3(response, "endpos")
    normal = _support_normal(response)
    if (
        startsolid is not False or allsolid is not False
        or fraction is None or fraction >= 1.0
        or endpos is None or normal is None or normal[2] < MIN_SUPPORT_NORMAL_Z
    ):
        return None
    # The teleport touch places the player at ``arrival_origin``; it does not
    # place them at the support trace endpoint.  CM-only authority may admit a
    # stable step-height support, but it must never turn a 96-unit probe into
    # an invented post-teleport fall trajectory.  Deeper landings require an
    # exact Pmove/fall replay and are omitted by this v1 helper.
    if (
        abs(endpos[0] - arrival_origin[0]) > 1e-4
        or abs(endpos[1] - arrival_origin[1]) > 1e-4
        or endpos[2] > arrival_origin[2] + 1e-4
        or arrival_origin[2] - endpos[2] > MAX_CM_ONLY_SUPPORT_DROP + 1e-4
    ):
        return None
    return endpos


def _destination_landings(
    link: TeleporterLink,
    collision_call: CollisionBatch,
) -> dict[str, Vec3]:
    requests: list[dict[str, object]] = []
    for stance, mins, maxs in (
        ("standing", STANDING_MINS, STANDING_MAXS),
        ("crouched", CROUCHED_MINS, CROUCHED_MAXS),
    ):
        requests.extend([
            _box_request(
                f"teleport:{link.source_entity_index}:destination:{stance}:clear",
                link.arrival_origin, link.arrival_origin, mins, maxs,
            ),
            _box_request(
                f"teleport:{link.source_entity_index}:destination:{stance}:support",
                link.arrival_origin,
                (
                    link.arrival_origin[0], link.arrival_origin[1],
                    link.arrival_origin[2] - SUPPORT_PROBE_DEPTH,
                ),
                mins, maxs,
            ),
        ])
    results = tuple(collision_call(requests))
    if len(results) != len(requests):
        return {}
    landings: dict[str, Vec3] = {}
    for ordinal, stance in enumerate(("standing", "crouched")):
        clear = results[ordinal * 2]
        support = results[ordinal * 2 + 1]
        landing = (
            _strict_supported(support, link.arrival_origin)
            if _strict_clear(clear) else None
        )
        if landing is not None:
            landings[stance] = landing
    return landings


def _aabb_intersects(
    left_mins: Vec3, left_maxs: Vec3, right_mins: Vec3, right_maxs: Vec3,
) -> bool:
    # Positive-volume overlap is sufficient evidence planning.  Mere tangent
    # bounds are not promoted to a trigger touch.
    return all(
        left_maxs[axis] > right_mins[axis]
        and right_maxs[axis] > left_mins[axis]
        for axis in range(3)
    )


def _player_bounds(position: Vec3, mins: Vec3, maxs: Vec3) -> tuple[Vec3, Vec3]:
    return (
        tuple(position[axis] + mins[axis] for axis in range(3)),
        tuple(position[axis] + maxs[axis] for axis in range(3)),
    )  # type: ignore[return-value]


def build_teleporter_edge(
    link: TeleporterLink,
    source: L1Index,
    target: L1Index,
    stance: str,
    *,
    evidence: int,
    validation_version: int,
) -> dict[str, object] | None:
    """Purely construct the canonical typed edge after exact proofs exist."""

    if source == target or stance not in {"standing", "crouched"}:
        return None
    if evidence != EVIDENCE_CM_TRACE_V1:
        return None
    if validation_version != TELEPORTER_VALIDATION_VERSION:
        return None
    if not (0 <= link.destination_entity_index <= 0xFFFFFFFF):
        return None
    return {
        "source": list(source),
        "target": list(target),
        "edge_type": "teleporter",
        "stance": stance,
        "flags": 0,
        "blocker": 0,
        "cost": TELEPORTER_COST_Q8,
        "risk": TELEPORTER_RISK_U16,
        "confidence": 65_535,
        "evidence": evidence,
        "validation_version": validation_version,
        "auxiliary": link.destination_entity_index,
    }


def prove_trigger_teleporter_edges(
    resolution: TeleporterResolution,
    nodes: Mapping[L1Index, NavNodeLike],
    atlas_origin: tuple[int, int, int],
    collision_call: CollisionBatch,
) -> TeleporterEdgeAnalysis:
    """Challenge resolved links through exact transformed/model-0 CM traces."""

    edges: list[dict[str, object]] = []
    omissions = list(resolution.omissions)
    seen_edges: set[tuple[L1Index, L1Index, str, int]] = set()
    ordered_nodes = sorted(nodes.items(), key=lambda item: (item[0][2], item[0][1], item[0][0]))

    for link in resolution.links:
        landing_by_stance: dict[str, L1Index] = {}
        for stance, landing in _destination_landings(link, collision_call).items():
            target_key = _grid_index(landing, atlas_origin)
            target_node = nodes.get(target_key)
            if target_node is None or not target_node.supported:
                continue
            if stance == "standing" and not target_node.standing_clear:
                continue
            if stance == "crouched" and not target_node.crouched_clear:
                continue
            landing_by_stance[stance] = target_key
        if not landing_by_stance:
            omissions.append(TeleporterOmission(
                link.source_entity_index,
                "destination origin is blocked, unsupported, or absent from Atlas L1",
                link.destination_entity_index,
            ))
            continue

        contact_requests: list[dict[str, object]] = []
        contact_meta: list[tuple[L1Index, str, L1Index]] = []
        for source_key, source_node in ordered_nodes:
            if not source_node.supported:
                continue
            for stance, mins, maxs in (
                ("standing", STANDING_MINS, STANDING_MAXS),
                ("crouched", CROUCHED_MINS, CROUCHED_MAXS),
            ):
                target_key = landing_by_stance.get(stance)
                if target_key is None:
                    continue
                if stance == "standing" and not source_node.standing_clear:
                    continue
                if stance == "crouched" and not source_node.crouched_clear:
                    continue
                player_mins, player_maxs = _player_bounds(source_node.position, mins, maxs)
                if not _aabb_intersects(
                    player_mins, player_maxs,
                    link.source_bounds_mins, link.source_bounds_maxs,
                ):
                    continue
                request = _box_request(
                    f"teleport:{link.source_entity_index}:contact:{stance}:"
                    f"{source_key[0]}:{source_key[1]}:{source_key[2]}",
                    source_node.position, source_node.position, mins, maxs,
                )
                request.update({
                    "op": "transformed_box_trace",
                    "headnode": link.source_headnode,
                    "origin": list(link.source_origin),
                    "angles": list(link.source_angles),
                })
                contact_requests.append(request)
                contact_meta.append((source_key, stance, target_key))
        if not contact_requests:
            omissions.append(TeleporterOmission(
                link.source_entity_index,
                "no stance-clear Atlas L1 origin intersects trigger bounds",
                link.destination_entity_index,
            ))
            continue
        contact_results = tuple(collision_call(contact_requests))
        if len(contact_results) != len(contact_requests):
            omissions.append(TeleporterOmission(
                link.source_entity_index, "trigger collision response count differs",
                link.destination_entity_index,
            ))
            continue
        admitted = 0
        standing_sources: set[L1Index] = set()
        for (source_key, stance, target_key), response in zip(contact_meta, contact_results):
            startsolid = _trace_bool(response, "startsolid")
            allsolid = _trace_bool(response, "allsolid")
            if startsolid is not True and allsolid is not True:
                continue
            # Standing contact subsumes crouched contact from the same origin;
            # keep the less restrictive deterministic edge only once.
            if stance == "crouched" and source_key in standing_sources:
                continue
            edge = build_teleporter_edge(
                link, source_key, target_key, stance,
                evidence=EVIDENCE_CM_TRACE_V1,
                validation_version=TELEPORTER_VALIDATION_VERSION,
            )
            if edge is None:
                continue
            identity = (source_key, target_key, stance, link.destination_entity_index)
            if identity in seen_edges:
                continue
            seen_edges.add(identity)
            if stance == "standing":
                standing_sources.add(source_key)
            edges.append(edge)
            admitted += 1
        if admitted == 0:
            omissions.append(TeleporterOmission(
                link.source_entity_index,
                "transformed CM proved no trigger contact from Atlas L1",
                link.destination_entity_index,
            ))

    edges.sort(key=lambda edge: (
        edge["source"][2], edge["source"][1], edge["source"][0],
        edge["target"][2], edge["target"][1], edge["target"][0],
        edge["stance"], edge["auxiliary"],
    ))
    return TeleporterEdgeAnalysis(
        tuple(edges),
        tuple(sorted(
            omissions,
            key=lambda item: (
                item.source_entity_index,
                -1 if item.destination_entity_index is None else item.destination_entity_index,
                item.reason,
            ),
        )),
        resolution.links,
    )


__all__ = [
    "EVIDENCE_CM_TRACE_V1",
    "TELEPORTER_COST_Q8",
    "TELEPORTER_VALIDATION_VERSION",
    "TeleporterEdgeAnalysis",
    "TeleporterLink",
    "TeleporterOmission",
    "TeleporterResolution",
    "build_teleporter_edge",
    "prove_trigger_teleporter_edges",
    "resolve_trigger_teleporters",
    "teleporter_seed_points",
]
