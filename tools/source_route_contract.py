#!/usr/bin/env python3
"""Fail-closed validation for generator source route sidecars.

This module validates only the generator's source-level route claim.  It does
not infer compiled collision or Atlas connectivity; the compiled analyzer
remains authoritative after the source-freeze boundary.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROUTE_CONTRACT_SCHEMA = "q2-generator-source-route-contract-v2"
ROUTE_ARCHETYPES = ("offense", "survival", "control", "balanced")
MAX_ENDPOINT_DISTANCE_ERROR = 0.5


class SourceRouteContractError(ValueError):
    """Raised when a source route sidecar violates the freeze contract."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceRouteContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceRouteContractError(f"{label} must be an object")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceRouteContractError(f"{label} must be a nonnegative integer")
    return value


def _room(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceRouteContractError(f"{label} is not floor-assigned")
    return value


def _component(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceRouteContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SourceRouteContractError(f"{label} must be finite")
    return result


def _origin(node: Mapping[str, Any], label: str) -> tuple[float, float, float]:
    return tuple(_number(node.get(axis), f"{label} {axis}") for axis in "xyz")


def load_source_route_contract(path: Path, map_id: str) -> dict[str, Any]:
    """Read and validate one generator ``.routes.json`` sidecar."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SourceRouteContractError(
                    f"{map_id} route graph contains non-finite JSON token {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceRouteContractError(
            f"cannot read {map_id} route graph: {exc}"
        ) from exc
    return validate_source_route_contract(value, map_id)


def validate_source_route_contract(
    value: object, map_id: str,
) -> dict[str, Any]:
    """Validate and normalize one source route graph for freeze evidence."""
    graph = _mapping(value, f"{map_id} route graph")
    if graph.get("version") != 2 or isinstance(graph.get("version"), bool):
        raise SourceRouteContractError(
            f"{map_id} route graph version must be exactly 2"
        )
    nodes_value = graph.get("nodes")
    edges_value = graph.get("edges")
    routes_value = graph.get("routes")
    if not isinstance(nodes_value, list) or not isinstance(edges_value, list):
        raise SourceRouteContractError(
            f"{map_id} route graph lacks nodes/edges arrays"
        )
    if not isinstance(routes_value, list):
        raise SourceRouteContractError(f"{map_id} route graph lacks routes array")

    nodes: dict[int, Mapping[str, Any]] = {}
    item_origins: dict[tuple[float, float, float], int] = {}
    spawn_origins: dict[tuple[float, float, float], int] = {}
    spawn_by_room: dict[int, list[Mapping[str, Any]]] = {}
    components_by_node: dict[int, int | None] = {}
    normalized_nodes = []
    for position, raw_node in enumerate(nodes_value):
        node = _mapping(raw_node, f"{map_id} node {position}")
        node_id = _integer(node.get("id"), f"{map_id} node {position} ID")
        if node_id in nodes:
            raise SourceRouteContractError(f"{map_id} duplicates node ID {node_id}")
        nodes[node_id] = node
        node_type = node.get("type")
        if node_type not in {"item", "spawn"}:
            raise SourceRouteContractError(
                f"{map_id} node {node_id} has unknown type {node_type!r}"
            )
        room = _room(node.get("room"), f"{map_id} {node_type} node {node_id}")
        origin = _origin(node, f"{map_id} {node_type} node {node_id}")
        source_component = _component(
            node.get("source_component"),
            f"{map_id} {node_type} node {node_id} source component",
        )
        components_by_node[node_id] = source_component
        normalized_nodes.append({
            "id": node_id,
            "type": node_type,
            "room": room,
            "origin": list(origin),
            "source_component": source_component,
        })
        if node_type == "item":
            if origin in item_origins:
                raise SourceRouteContractError(
                    f"{map_id} item origins are not globally unique: nodes "
                    f"{item_origins[origin]} and {node_id} share {origin}"
                )
            item_origins[origin] = node_id
        else:
            spawn_origins.setdefault(origin, node_id)
            spawn_by_room.setdefault(room, []).append(node)
    for room_spawns in spawn_by_room.values():
        room_spawns.sort(key=lambda node: int(node["id"]))

    normalized_edges = []
    edge_pairs: set[tuple[int, int]] = set()
    for edge_index, raw_edge in enumerate(edges_value):
        edge = _mapping(raw_edge, f"{map_id} edge {edge_index}")
        left = _integer(edge.get("a"), f"{map_id} edge {edge_index} a")
        right = _integer(edge.get("b"), f"{map_id} edge {edge_index} b")
        if left == right:
            raise SourceRouteContractError(
                f"{map_id} edge {edge_index} is a self-loop"
            )
        pair = (min(left, right), max(left, right))
        if pair in edge_pairs:
            raise SourceRouteContractError(
                f"{map_id} edge {edge_index} duplicates undirected edge {pair}"
            )
        edge_pairs.add(pair)
        normalized_edges.append(list(pair))

    archetypes = [
        route.get("archetype") if isinstance(route, Mapping) else None
        for route in routes_value
    ]
    if (
        len(archetypes) != len(ROUTE_ARCHETYPES)
        or any(not isinstance(archetype, str) for archetype in archetypes)
        or Counter(archetypes) != Counter(ROUTE_ARCHETYPES)
    ):
        raise SourceRouteContractError(
            f"{map_id} must contain each route archetype exactly once"
        )

    routes_by_archetype = {
        str(route["archetype"]): _mapping(route, f"{map_id} route")
        for route in routes_value
    }
    zero_length_legs = 0
    route_endpoint_count = 0
    normalized_routes = []
    for route_index, archetype in enumerate(ROUTE_ARCHETYPES):
        route = routes_by_archetype[archetype]
        start_room = _room(
            route.get("start_room"),
            f"{map_id} route {route_index} start room",
        )
        start_node_id = _integer(
            route.get("start_node_id"),
            f"{map_id} route {route_index} start node ID",
        )
        start = nodes.get(start_node_id)
        if start is None or start.get("type") != "spawn":
            raise SourceRouteContractError(
                f"{map_id} route {route_index} start node {start_node_id} "
                "is not a declared spawn"
            )
        if start.get("room") != start_room:
            raise SourceRouteContractError(
                f"{map_id} route {route_index} start room does not match "
                f"spawn node {start_node_id}"
            )
        source_component = _integer(
            route.get("source_component"),
            f"{map_id} route {route_index} source component",
        )
        if components_by_node[start_node_id] != source_component:
            raise SourceRouteContractError(
                f"{map_id} route {route_index} source component does not match "
                f"spawn node {start_node_id}"
            )
        node_ids = route.get("node_ids")
        if not isinstance(node_ids, list) or len(node_ids) < 2:
            raise SourceRouteContractError(
                f"{map_id} route {route_index} requires at least two item endpoints"
            )
        if (
            any(isinstance(node_id, bool) or not isinstance(node_id, int)
                for node_id in node_ids)
            or len(set(node_ids)) != len(node_ids)
        ):
            raise SourceRouteContractError(
                f"{map_id} route {route_index} has duplicate or invalid endpoints"
            )

        endpoints = [start]
        for endpoint_index, node_id in enumerate(node_ids):
            if node_id not in nodes:
                raise SourceRouteContractError(
                    f"{map_id} route {route_index} endpoint {endpoint_index} is invalid"
                )
            node = nodes[node_id]
            if node.get("type") != "item":
                raise SourceRouteContractError(
                    f"{map_id} route {route_index} endpoint node {node_id} "
                    "is not an item"
                )
            if components_by_node[node_id] != source_component:
                raise SourceRouteContractError(
                    f"{map_id} route {route_index} endpoint node {node_id} "
                    f"does not share source standing component {source_component}"
                )
            endpoints.append(node)
        endpoints.append(start)
        route_endpoint_count += len(node_ids)

        origins = [
            _origin(node, f"{map_id} route {route_index} endpoint")
            for node in endpoints
        ]
        for source, target in zip(origins, origins[1:]):
            if source == target:
                zero_length_legs += 1
        geometric_distance = sum(
            math.dist(source, target)
            for source, target in zip(origins, origins[1:])
        )
        published_distance = _number(
            route.get("dist"), f"{map_id} route {route_index} dist"
        )
        distance_error = abs(published_distance - geometric_distance)
        if distance_error > MAX_ENDPOINT_DISTANCE_ERROR:
            raise SourceRouteContractError(
                f"{map_id} route {route_index} dist {published_distance:g} "
                f"differs from endpoint-loop geometry {geometric_distance:.6f}"
            )
        normalized_routes.append({
            "archetype": archetype,
            "start_room": start_room,
            "start_node_id": start_node_id,
            "source_component": source_component,
            "node_ids": list(node_ids),
            "endpoint_loop_sha256": _sha256([list(origin) for origin in origins]),
            "published_dist": published_distance,
            "endpoint_loop_geometry": geometric_distance,
            "distance_error": distance_error,
        })

    if zero_length_legs:
        raise SourceRouteContractError(
            f"{map_id} has {zero_length_legs} zero-length route legs"
        )
    collisions = set(item_origins).intersection(spawn_origins)
    if collisions:
        origin = min(collisions)
        raise SourceRouteContractError(
            f"{map_id} item node {item_origins[origin]} overlaps spawn node "
            f"{spawn_origins[origin]} at {origin}"
        )

    normalized_contract = {
        "map": map_id,
        "nodes": sorted(normalized_nodes, key=lambda node: int(node["id"])),
        "edges": sorted(normalized_edges),
        "routes": normalized_routes,
    }
    return {
        "schema": ROUTE_CONTRACT_SCHEMA,
        "map": map_id,
        "route_contract_sha256": _sha256(normalized_contract),
        "archetypes": list(ROUTE_ARCHETYPES),
        "node_count": len(nodes),
        "edge_count": len(edges_value),
        "route_count": len(routes_value),
        "item_origin_count": len(item_origins),
        "spawn_count": sum(len(spawns) for spawns in spawn_by_room.values()),
        "route_endpoint_count": route_endpoint_count,
        "routes": normalized_routes,
        "zero_length_route_legs": 0,
        "minimum_distinct_item_endpoints_per_route": 2,
        "all_route_endpoints_are_items": True,
        "published_dist_matches_endpoint_loop": True,
        "all_item_nodes_floor_assigned": True,
        "item_spawn_origin_collisions": 0,
        "all_spawns_and_route_endpoints_floor_assigned": True,
        "globally_unique_item_origins": True,
        "all_selected_endpoints_share_source_standing_component": True,
        "exact_start_nodes_declared": True,
        "room_edges_used_as_reachability": False,
    }
