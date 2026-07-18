from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import pytest

from tools.source_route_contract import (
    SourceRouteContractError,
    validate_source_route_contract,
)


MAP_ID = "source_route_contract_test"
ARCHETYPES = ("offense", "survival", "control", "balanced")


def _graph() -> dict[str, Any]:
    nodes = [
        {
            "id": node_id,
            "type": "item",
            "class": f"item_{node_id}",
            "x": 32 + node_id * 32,
            "y": 32 + (node_id % 2) * 64,
            "z": 24,
            "room": 0,
            "source_component": 0,
        }
        for node_id in range(4)
    ]
    nodes.extend(
        {
            "id": node_id,
            "type": "spawn",
            "x": 192 + (node_id - 4) * 64,
            "y": 64 + (node_id % 2) * 64,
            "z": 24,
            "room": 0,
            "source_component": 0,
        }
        for node_id in range(4, 12)
    )
    routes = [
        {
            "archetype": archetype,
            "start_room": 0,
            "start_node_id": 4,
            "source_component": 0,
            "node_ids": [index, (index + 1) % 4],
        }
        for index, archetype in enumerate(ARCHETYPES)
    ]
    graph: dict[str, Any] = {
        "version": 2,
        "nodes": nodes,
        "edges": [],
        "routes": routes,
    }
    _refresh_distances(graph)
    return graph


def _refresh_distances(graph: dict[str, Any]) -> None:
    nodes = {node["id"]: node for node in graph["nodes"]}
    for route in graph["routes"]:
        loop = [
            nodes[route["start_node_id"]],
            *(nodes[node_id] for node_id in route["node_ids"]),
            nodes[route["start_node_id"]],
        ]
        route["dist"] = sum(
            math.dist(
                (source["x"], source["y"], source["z"]),
                (target["x"], target["y"], target["z"]),
            )
            for source, target in zip(loop, loop[1:])
        )


def test_cross_room_route_needs_no_metadata_edge_when_component_matches() -> None:
    graph = _graph()
    nodes = graph["nodes"]
    nodes[0]["room"] = 1

    report = validate_source_route_contract(graph, MAP_ID)

    assert report["all_selected_endpoints_share_source_standing_component"] is True
    assert report["exact_start_nodes_declared"] is True
    assert report["room_edges_used_as_reachability"] is False
    assert report["edge_count"] == 0


@pytest.mark.parametrize(
    ("endpoint_room", "edges"),
    [
        (0, []),
        (1, [{"a": 0, "b": 1}]),
    ],
    ids=("same-room", "metadata-connected-rooms"),
)
def test_component_mismatch_rejects_even_when_room_metadata_would_connect(
    endpoint_room: int, edges: list[dict[str, int]],
) -> None:
    graph = _graph()
    nodes = graph["nodes"]
    nodes[0]["room"] = endpoint_room
    nodes[0]["source_component"] = 1
    graph["edges"] = edges

    with pytest.raises(
        SourceRouteContractError,
        match="endpoint node 0 does not share source standing component 0",
    ):
        validate_source_route_contract(graph, MAP_ID)


def test_exact_start_node_selects_component_not_first_spawn_in_room() -> None:
    graph = _graph()
    for route in graph["routes"]:
        route["start_node_id"] = 5
    _refresh_distances(graph)

    report = validate_source_route_contract(graph, MAP_ID)

    assert {route["start_node_id"] for route in report["routes"]} == {5}
    assert {route["source_component"] for route in report["routes"]} == {0}
    assert report["spawn_count"] == 8


def test_split_spawn_components_are_rejected_before_route_publication() -> None:
    graph = _graph()
    graph["nodes"][-1]["source_component"] = 1

    with pytest.raises(
        SourceRouteContractError,
        match="spawn nodes must share one non-null source standing component",
    ):
        validate_source_route_contract(graph, MAP_ID)


def test_duplicate_route_spawn_origins_are_rejected() -> None:
    graph = _graph()
    first = graph["nodes"][4]
    duplicate = graph["nodes"][-1]
    for axis in "xyz":
        duplicate[axis] = first[axis]

    with pytest.raises(
        SourceRouteContractError,
        match="deathmatch spawn origins are not unique",
    ):
        validate_source_route_contract(graph, MAP_ID)


@pytest.mark.parametrize("delta", (-1, 1), ids=("seven", "nine"))
def test_exactly_eight_spawn_nodes_are_required(delta: int) -> None:
    graph = _graph()
    if delta < 0:
        graph["nodes"].pop()
    else:
        graph["nodes"].append({
            "id": 12,
            "type": "spawn",
            "x": 704,
            "y": 64,
            "z": 24,
            "room": 0,
            "source_component": 0,
        })

    with pytest.raises(
        SourceRouteContractError,
        match="must declare exactly eight deathmatch spawn nodes",
    ):
        validate_source_route_contract(graph, MAP_ID)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("start_node_id", 99, "is not a declared spawn"),
        ("start_room", 1, "start room does not match spawn node"),
        ("source_component", 1, "source component does not match spawn node"),
    ],
)
def test_exact_start_contract_fails_closed(
    field: str, value: int, message: str,
) -> None:
    graph = _graph()
    graph["routes"][0][field] = value

    with pytest.raises(SourceRouteContractError, match=message):
        validate_source_route_contract(graph, MAP_ID)


def test_v1_route_graph_is_rejected_without_fallback() -> None:
    graph = _graph()
    graph["version"] = 1

    with pytest.raises(
        SourceRouteContractError, match="route graph version must be exactly 2"
    ):
        validate_source_route_contract(graph, MAP_ID)


@pytest.mark.parametrize(
    ("mode", "value"),
    [
        ("missing", None),
        ("explicit-none", None),
        ("boolean", True),
        ("negative", -1),
    ],
)
def test_selected_endpoint_component_is_mandatory_and_valid(
    mode: str, value: object,
) -> None:
    graph = _graph()
    endpoint = graph["nodes"][0]
    if mode == "missing":
        del endpoint["source_component"]
    else:
        endpoint["source_component"] = value

    with pytest.raises(
        SourceRouteContractError, match=r"source (?:standing )?component"
    ):
        validate_source_route_contract(graph, MAP_ID)


@pytest.mark.parametrize("value", [None, True, -1])
def test_route_component_is_mandatory_nonnegative_integer(value: object) -> None:
    graph = _graph()
    graph["routes"][0]["source_component"] = value

    with pytest.raises(SourceRouteContractError, match="source component"):
        validate_source_route_contract(graph, MAP_ID)


def test_route_report_is_deterministic_and_binds_component_labels() -> None:
    graph = _graph()
    first = validate_source_route_contract(graph, MAP_ID)
    second = validate_source_route_contract(deepcopy(graph), MAP_ID)
    changed = deepcopy(graph)
    changed["nodes"][3]["source_component"] = 7

    assert first == second
    with pytest.raises(SourceRouteContractError, match="source standing component"):
        validate_source_route_contract(changed, MAP_ID)


def test_source_geodesic_may_exceed_endpoint_loop_geometry() -> None:
    graph = _graph()
    graph["routes"][0]["dist"] += 2048.0

    report = validate_source_route_contract(graph, MAP_ID)
    route = report["routes"][0]

    assert route["source_geodesic_overhead"] == pytest.approx(2048.0)
    assert report["published_dist_covers_endpoint_loop_geometry"] is True
    assert "published_dist_matches_endpoint_loop" not in report


def test_published_distance_cannot_undercut_endpoint_loop_geometry() -> None:
    graph = _graph()
    graph["routes"][0]["dist"] -= 0.001

    with pytest.raises(
        SourceRouteContractError, match="falls below endpoint-loop geometry"
    ):
        validate_source_route_contract(graph, MAP_ID)


def test_unproven_floor_assignment_booleans_are_not_published() -> None:
    report = validate_source_route_contract(_graph(), MAP_ID)

    assert "all_item_nodes_floor_assigned" not in report
    assert "all_spawns_and_route_endpoints_floor_assigned" not in report
