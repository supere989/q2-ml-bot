from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from harness.atlas_teleporter_edges import (
    EVIDENCE_CM_TRACE_V1,
    TELEPORTER_COST_Q8,
    TELEPORTER_VALIDATION_VERSION,
    build_teleporter_edge,
    prove_trigger_teleporter_edges,
    resolve_trigger_teleporters,
    teleporter_seed_points,
)


@dataclass(frozen=True)
class Entity:
    index: int
    classname: str
    properties: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Model:
    index: int
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]
    headnode: int


@dataclass(frozen=True)
class Node:
    position: tuple[float, float, float]
    standing_clear: bool = True
    crouched_clear: bool = True
    supported: bool = True


def entity(index: int, classname: str, *properties: tuple[str, str]) -> Entity:
    return Entity(index, classname, (("classname", classname), *properties))


def fixture_entities(*destinations: Entity) -> tuple[Entity, ...]:
    return (
        entity(
            1, "trigger_teleport", ("model", "*1"),
            ("target", "EXIT"),
        ),
        *destinations,
    )


def fixture_model() -> Model:
    return Model(1, (-16.0, -16.0, 0.0), (16.0, 16.0, 32.0), 17)


def fixture_submodels() -> tuple[Mapping[str, object], ...]:
    return ({"entity_index": 1, "model_index": 1},)


def clear_response(request: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": request["id"], "startsolid": False, "allsolid": False,
        "fraction": 1.0, "endpos": request["end"],
        "plane": {"normal": [0.0, 0.0, 0.0]},
    }


class TeleporterOracle:
    def __init__(
        self, *, blocked_destination: bool = False, contact: bool = True,
        support_z: float = 0.0,
    ):
        self.blocked_destination = blocked_destination
        self.contact = contact
        self.support_z = support_z
        self.requests: list[dict[str, object]] = []

    def __call__(
        self, requests: Sequence[dict[str, object]],
    ) -> Sequence[Mapping[str, object]]:
        self.requests.extend(requests)
        responses: list[dict[str, object]] = []
        for request in requests:
            identifier = str(request["id"])
            if ":destination:" in identifier and identifier.endswith(":clear"):
                if self.blocked_destination:
                    responses.append({
                        "id": request["id"], "startsolid": True,
                        "allsolid": False, "fraction": 0.0,
                        "endpos": request["start"],
                        "plane": {"normal": [0.0, 0.0, 0.0]},
                    })
                else:
                    responses.append(clear_response(request))
            elif ":destination:" in identifier and identifier.endswith(":support"):
                responses.append({
                    "id": request["id"], "startsolid": False,
                    "allsolid": False, "fraction": 0.25,
                    "endpos": [128.0, 0.0, self.support_z],
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                })
            elif ":contact:" in identifier:
                responses.append({
                    "id": request["id"], "startsolid": self.contact,
                    "allsolid": False, "fraction": 0.0 if self.contact else 1.0,
                    "endpos": request["start"],
                    "plane": {"normal": [0.0, 0.0, 0.0]},
                })
            else:
                raise AssertionError(f"unexpected request {identifier}")
        return responses


def exact_resolution():
    destination = entity(
        7, "info_teleport_destination", ("targetname", "exit"),
        ("origin", "128 0 0"), ("angle", "90"),
    )
    return resolve_trigger_teleporters(
        fixture_entities(destination), fixture_submodels(), (fixture_model(),),
    )


def test_unique_target_is_case_insensitive_and_uses_runtime_destination_origin() -> None:
    resolution = exact_resolution()
    assert resolution.omissions == ()
    assert len(resolution.links) == 1
    link = resolution.links[0]
    assert link.source_entity_index == 1
    assert link.source_model_index == 1
    assert link.source_headnode == 17
    assert link.destination_entity_index == 7
    assert link.destination_origin == (128.0, 0.0, 0.0)
    assert link.arrival_origin == (128.0, 0.0, 16.0)


def test_ambiguous_missing_and_wrong_class_targets_are_unknown_not_edges() -> None:
    duplicate_a = entity(
        7, "info_teleport_destination", ("targetname", "EXIT"),
        ("origin", "128 0 0"),
    )
    duplicate_b = entity(
        8, "info_teleport_destination", ("targetname", "exit"),
        ("origin", "256 0 0"),
    )
    ambiguous = resolve_trigger_teleporters(
        fixture_entities(duplicate_a, duplicate_b), fixture_submodels(),
        (fixture_model(),),
    )
    assert ambiguous.links == ()
    assert ambiguous.omissions[0].reason == (
        "targetname is ambiguous under case-insensitive G_Find"
    )

    missing = resolve_trigger_teleporters(
        fixture_entities(), fixture_submodels(), (fixture_model(),),
    )
    assert missing.links == ()
    assert missing.omissions[0].reason == "target destination is missing"

    wrong = resolve_trigger_teleporters(
        fixture_entities(entity(
            9, "info_notnull", ("targetname", "EXIT"), ("origin", "128 0 0"),
        )),
        fixture_submodels(), (fixture_model(),),
    )
    assert wrong.links == ()
    assert wrong.omissions[0].reason == (
        "unique target is not info_teleport_destination"
    )

    malformed = resolve_trigger_teleporters(
        fixture_entities(entity(
            10, "info_teleport_destination", ("targetname", "EXIT"),
            ("origin", "128 invalid 0"),
        )),
        fixture_submodels(), (fixture_model(),),
    )
    assert malformed.links == ()
    assert malformed.omissions[0].reason == "destination origin is malformed"

    no_target_source = entity(1, "trigger_teleport", ("model", "*1"))
    no_target = resolve_trigger_teleporters(
        (no_target_source,), fixture_submodels(), (fixture_model(),),
    )
    assert no_target.links == ()
    assert no_target.omissions[0].reason == "missing target"


def test_valid_trigger_contact_and_destination_clearance_emit_typed_edge() -> None:
    resolution = exact_resolution()
    nodes = {
        (0, 0, 1): Node((0.0, 0.0, 16.0)),
        (8, 0, 0): Node((128.0, 0.0, 0.0)),
    }
    oracle = TeleporterOracle()
    assert teleporter_seed_points(resolution, oracle) == ((128.0, 0.0, 0.0),)
    analysis = prove_trigger_teleporter_edges(
        resolution, nodes, (0, 0, 0), oracle,
    )
    assert analysis.omissions == ()
    assert analysis.report()["authority"] == "exact-cm-entity-law"
    assert len(analysis.edges) == 1
    edge = analysis.edges[0]
    assert edge == {
        "source": [0, 0, 1],
        "target": [8, 0, 0],
        "edge_type": "teleporter",
        "stance": "standing",
        "flags": 0,
        "blocker": 0,
        "cost": TELEPORTER_COST_Q8,
        "risk": 0,
        "confidence": 65_535,
        "evidence": EVIDENCE_CM_TRACE_V1,
        "validation_version": TELEPORTER_VALIDATION_VERSION,
        "auxiliary": 7,
    }
    contact = [request for request in oracle.requests if ":contact:" in str(request["id"])]
    assert contact
    assert all(request["op"] == "transformed_box_trace" for request in contact)
    assert all(request["headnode"] == 17 for request in contact)


def test_blocked_destination_and_unproven_contact_emit_no_edge() -> None:
    resolution = exact_resolution()
    nodes = {
        (0, 0, 1): Node((0.0, 0.0, 16.0)),
        (8, 0, 0): Node((128.0, 0.0, 0.0)),
    }
    blocked = prove_trigger_teleporter_edges(
        resolution, nodes, (0, 0, 0), TeleporterOracle(blocked_destination=True),
    )
    assert blocked.edges == ()
    assert blocked.omissions[-1].reason == (
        "destination origin is blocked, unsupported, or absent from Atlas L1"
    )
    assert teleporter_seed_points(
        resolution, TeleporterOracle(blocked_destination=True),
    ) == ()

    no_contact = prove_trigger_teleporter_edges(
        resolution, nodes, (0, 0, 0), TeleporterOracle(contact=False),
    )
    assert no_contact.edges == ()
    assert no_contact.omissions[-1].reason == (
        "transformed CM proved no trigger contact from Atlas L1"
    )


def test_cm_only_destination_cannot_invent_a_deep_post_teleport_fall() -> None:
    resolution = exact_resolution()
    # Exact arrival Z is 16. A support at -64 is an 80-unit fall and may only
    # become a topology edge after exact Pmove/fall replay, not this CM probe.
    nodes = {
        (0, 0, 1): Node((0.0, 0.0, 16.0)),
        (8, 0, -4): Node((128.0, 0.0, -64.0)),
    }
    analysis = prove_trigger_teleporter_edges(
        resolution, nodes, (0, 0, 0), TeleporterOracle(support_z=-64.0),
    )
    assert analysis.edges == ()
    assert analysis.omissions[-1].reason == (
        "destination origin is blocked, unsupported, or absent from Atlas L1"
    )
    assert teleporter_seed_points(
        resolution, TeleporterOracle(support_z=-64.0),
    ) == ()


def test_edge_constructor_rejects_unsealed_evidence_and_self_edges() -> None:
    link = exact_resolution().links[0]
    assert build_teleporter_edge(
        link, (0, 0, 0), (1, 0, 0), "standing",
        evidence=0, validation_version=1,
    ) is None
    assert build_teleporter_edge(
        link, (0, 0, 0), (0, 0, 0), "standing",
        evidence=1, validation_version=1,
    ) is None
