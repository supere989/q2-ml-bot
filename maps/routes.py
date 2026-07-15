#!/usr/bin/env python3
"""routes.py — route-graph over a generated map's lattice.

A map's item economy + room connectivity define how a competent player moves:
a *route* is a loop that collects high-value items as they respawn. This module
turns the generator's rooms/connections/items into:

  - nodes:  item spawn locations (typed, with respawn period + tactical value)
            and player spawns,
  - edges:  risk-weighted room-to-room connections,
  - routes: K archetype loops (offense / survival / control / balanced), each
            annotated with the items it threads, distance, risk, and value.

Emitted as <map>.routes.json. The runtime item-timing belief state consumes
node respawn periods to predict availability; the policy consumes route value
+ risk to choose buffed-intercept paths. "Build the route-graph first" — this
is that substrate.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

# Item respawn periods (game-seconds). Quake-2 DM defaults; runes use Lithium
# spawn timing and are marked separately (period 0 = event-driven, not fixed).
RESPAWN_S = {
    "weapon_": 30.0, "ammo_": 30.0,
    "item_armor_body": 20.0, "item_armor_combat": 20.0, "item_armor_shard": 20.0,
    "item_health": 30.0, "item_health_large": 30.0, "item_health_mega": 20.0,
    "item_adrenaline": 60.0, "item_quad": 60.0, "item_invulnerability": 60.0,
    "rune_": 25.0,   # placed training runes respawn on a fixed timer (engine PLACED_RUNE_RESPAWN)
}

# Tactical value + axis tag. axis: 'offense' | 'survival' | 'value' | 'mobility'.
ITEM_AXIS = {
    "weapon_railgun": ("offense", 1.0), "weapon_rocketlauncher": ("offense", 0.9),
    "weapon_hyperblaster": ("offense", 0.6), "weapon_chaingun": ("offense", 0.55),
    "weapon_supershotgun": ("offense", 0.5), "weapon_grenadelauncher": ("offense", 0.5),
    "item_quad": ("offense", 1.5), "rune_strength": ("offense", 1.2), "rune_haste": ("offense", 1.0),
    "item_armor_body": ("survival", 1.0), "item_armor_combat": ("survival", 0.8),
    "item_health_mega": ("survival", 1.1), "item_health_large": ("survival", 0.6),
    "item_health": ("survival", 0.3), "rune_regen": ("survival", 1.1),
    "rune_vampire": ("value", 1.2),     # bridges offense/survival
    "item_adrenaline": ("survival", 0.5),
}
DEFAULT_AXIS = ("value", 0.3)

ROUTE_ARCHETYPES = ["offense", "survival", "control", "balanced"]
PLAYER_ORIGIN_FLOOR_OFFSET = 24.0
# Generated endpoints are normally exact integers at floor + 24.  Keep the
# admissible band below Quake's 18-unit stair step so a distinct ledge or
# platform cannot inherit the room below through an XY-only match.
ROOM_FLOOR_BAND_TOLERANCE = 8.0
SOURCE_NAV_CELL = 16
PLAYER_XY_HALF = 16
PLAYER_STANDING_HEIGHT = 56
MAX_SOURCE_NAV_CELLS = 250_000


class RouteGraphError(ValueError):
    """Raised when the generator cannot publish a complete route contract."""


def _period(cls: str) -> float:
    # Most-specific (longest) matching key wins, so item_health_mega (20s)
    # isn't shadowed by the item_health (30s) prefix.
    best = None
    for k, v in RESPAWN_S.items():
        if (cls == k or cls.startswith(k)) and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else 30.0


def _axis(cls: str) -> Tuple[str, float]:
    if cls in ITEM_AXIS:
        return ITEM_AXIS[cls]
    for k, v in ITEM_AXIS.items():
        if cls.startswith(k.split("_")[0] + "_"):
            return v
    return DEFAULT_AXIS


def _dist(a, b) -> float:
    return math.dist(a, b)


def source_room_index(rooms: Sequence[object], x: float, y: float,
                      z: float) -> int:
    """Resolve one endpoint to the same floor-banded room used by routes."""
    compatible = []
    for index, current in enumerate(rooms):
        if not (
            current.wx <= x <= current.wx + current.w
            and current.wy <= y <= current.wy + current.d
        ):
            continue
        floor_error = abs(
            z - (current.floor_z + PLAYER_ORIGIN_FLOOR_OFFSET)
        )
        if floor_error <= ROOM_FLOOR_BAND_TOLERANCE:
            compatible.append((floor_error, index))
    # Overlapping generated rooms are legal. Resolve vertical bands by closest
    # floor, then stable room ordinal for exact ties.
    return min(compatible)[1] if compatible else -1


def _solid_bounds(value, label: str) -> Tuple[float, float, float, float, float, float]:
    """Return one ordered finite source-solid AABB."""
    try:
        bounds = tuple(float(getattr(value, axis)) for axis in (
            "x0", "y0", "z0", "x1", "y1", "z1",
        ))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RouteGraphError(f"{label} has invalid bounds") from exc
    if (
        not all(math.isfinite(number) for number in bounds)
        or any(bounds[axis] >= bounds[axis + 3] for axis in range(3))
    ):
        raise RouteGraphError(f"{label} has unordered/non-finite bounds")
    return bounds  # type: ignore[return-value]


def _standing_components(
    rooms: Sequence[object],
    nodes: Sequence[Mapping[str, object]],
    standing_blockers: Sequence[object],
    lava_pools: Sequence[object],
) -> Dict[int, int]:
    """Build conservative final-source standing components for route proposals.

    This is deliberately not collision authority.  It rejects proposals that
    are already contradicted by the generator's final room floors, blockers, or
    lava, while the compiled Atlas independently challenges every surviving
    endpoint and segment.  Room IDs and declared connection records never join
    components.
    """
    room_records = []
    rooms_by_floor: Dict[float, List[Tuple[float, float, float, float, float]]] = (
        defaultdict(list)
    )
    for index, room in enumerate(rooms):
        try:
            x0 = float(getattr(room, "wx"))
            y0 = float(getattr(room, "wy"))
            x1 = x0 + float(getattr(room, "w"))
            y1 = y0 + float(getattr(room, "d"))
            floor_z = float(getattr(room, "floor_z"))
            ceiling_z = float(getattr(room, "ceil_z", math.inf))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RouteGraphError(f"room {index} has invalid source geometry") from exc
        if (
            not all(math.isfinite(value) for value in (x0, y0, x1, y1, floor_z))
            or x0 >= x1 or y0 >= y1
            or ceiling_z < floor_z + PLAYER_STANDING_HEIGHT
        ):
            raise RouteGraphError(f"room {index} cannot support a standing component")
        record = (x0, y0, x1, y1, ceiling_z)
        room_records.append((floor_z, *record))
        rooms_by_floor[floor_z].append(record)

    obstacles = [
        *(
            _solid_bounds(value, f"standing blocker {index}")
            for index, value in enumerate(standing_blockers)
        ),
        *(
            _solid_bounds(value, f"lava pool {index}")
            for index, value in enumerate(lava_pools)
        ),
    ]

    def supported(
        floor_z: float, x0: float, y0: float, x1: float, y1: float,
    ) -> bool:
        """Require the complete swept hull footprint on one authored floor."""
        return any(
            room_x0 <= x0 and x1 <= room_x1
            and room_y0 <= y0 and y1 <= room_y1
            and floor_z + PLAYER_STANDING_HEIGHT <= ceiling_z
            for room_x0, room_y0, room_x1, room_y1, ceiling_z
            in rooms_by_floor.get(floor_z, ())
        )

    def clear(
        floor_z: float, x0: float, y0: float, x1: float, y1: float,
    ) -> bool:
        if not supported(floor_z, x0, y0, x1, y1):
            return False
        z0 = floor_z
        z1 = floor_z + PLAYER_STANDING_HEIGHT
        return not any(
            # Quake's swept hull treats side and ceiling contact as blocked;
            # strict XY comparisons incorrectly admit an exact 32-unit gap
            # for a 32-unit-wide player.  Floor contact remains legal, hence
            # the intentionally asymmetric vertical comparisons.
            obstacle_x1 >= x0 and obstacle_x0 <= x1
            and obstacle_y1 >= y0 and obstacle_y0 <= y1
            and obstacle_z1 > z0 and obstacle_z0 <= z1
            for obstacle_x0, obstacle_y0, obstacle_z0,
            obstacle_x1, obstacle_y1, obstacle_z1 in obstacles
        )

    cells = set()
    half = PLAYER_XY_HALF
    center_offset = SOURCE_NAV_CELL / 2.0
    for floor_z, x0, y0, x1, y1, _ceiling_z in room_records:
        first_x = math.ceil((x0 + half - center_offset) / SOURCE_NAV_CELL)
        last_x = math.floor((x1 - half - center_offset) / SOURCE_NAV_CELL)
        first_y = math.ceil((y0 + half - center_offset) / SOURCE_NAV_CELL)
        last_y = math.floor((y1 - half - center_offset) / SOURCE_NAV_CELL)
        for cell_x in range(first_x, last_x + 1):
            center_x = cell_x * SOURCE_NAV_CELL + center_offset
            for cell_y in range(first_y, last_y + 1):
                key = (cell_x, cell_y, floor_z)
                if key in cells:
                    continue
                center_y = cell_y * SOURCE_NAV_CELL + center_offset
                if clear(
                    floor_z,
                    center_x - half, center_y - half,
                    center_x + half, center_y + half,
                ):
                    cells.add(key)
                    if len(cells) > MAX_SOURCE_NAV_CELLS:
                        raise RouteGraphError(
                            "final-source standing component exceeds cell budget"
                        )

    ordered_cells = sorted(cells, key=lambda key: (key[2], key[1], key[0]))
    cell_vertex = {key: index for index, key in enumerate(ordered_cells)}
    parent = list(range(len(ordered_cells) + len(nodes)))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Root choice is deterministic and independent of insertion order.
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for key in ordered_cells:
        source = cell_vertex[key]
        for delta_x, delta_y in ((1, 0), (0, 1)):
            neighbor = (key[0] + delta_x, key[1] + delta_y, key[2])
            target = cell_vertex.get(neighbor)
            if target is None:
                continue
            source_x = key[0] * SOURCE_NAV_CELL + center_offset
            source_y = key[1] * SOURCE_NAV_CELL + center_offset
            target_x = neighbor[0] * SOURCE_NAV_CELL + center_offset
            target_y = neighbor[1] * SOURCE_NAV_CELL + center_offset
            if clear(
                key[2],
                min(source_x, target_x) - half,
                min(source_y, target_y) - half,
                max(source_x, target_x) + half,
                max(source_y, target_y) + half,
            ):
                union(source, target)

    endpoint_offset = len(ordered_cells)
    admitted_endpoints = set()
    for node_index, node in enumerate(nodes):
        room_index = node.get("room")
        if (
            isinstance(room_index, bool) or not isinstance(room_index, int)
            or room_index < 0 or room_index >= len(room_records)
        ):
            continue
        floor_z = room_records[room_index][0]
        try:
            point_x = float(node["x"])
            point_y = float(node["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RouteGraphError(f"route node {node_index} has invalid origin") from exc
        if not clear(
            floor_z,
            point_x - half, point_y - half,
            point_x + half, point_y + half,
        ):
            continue
        endpoint = endpoint_offset + node_index
        for cell_x in range(
            math.floor(point_x / SOURCE_NAV_CELL) - 1,
            math.floor(point_x / SOURCE_NAV_CELL) + 2,
        ):
            center_x = cell_x * SOURCE_NAV_CELL + center_offset
            for cell_y in range(
                math.floor(point_y / SOURCE_NAV_CELL) - 1,
                math.floor(point_y / SOURCE_NAV_CELL) + 2,
            ):
                key = (cell_x, cell_y, floor_z)
                target = cell_vertex.get(key)
                if target is None:
                    continue
                center_y = cell_y * SOURCE_NAV_CELL + center_offset
                if clear(
                    floor_z,
                    min(point_x, center_x) - half,
                    min(point_y, center_y) - half,
                    max(point_x, center_x) + half,
                    max(point_y, center_y) + half,
                ):
                    union(endpoint, target)
                    admitted_endpoints.add(node_index)

    component_roots = sorted({
        find(endpoint_offset + index) for index in admitted_endpoints
    })
    component_ids = {root: index for index, root in enumerate(component_roots)}
    return {
        index: component_ids[find(endpoint_offset + index)]
        for index in sorted(admitted_endpoints)
    }


def source_endpoint_components(
    rooms: Sequence[object],
    endpoints: Sequence[Tuple[float, float, float]],
    standing_blockers: Sequence[object],
    lava_pools: Sequence[object],
) -> Dict[int, int]:
    """Return exact conservative source components for proposed endpoints."""
    nodes = [
        {
            "id": index,
            "x": x,
            "y": y,
            "z": z,
            "room": source_room_index(rooms, x, y, z),
        }
        for index, (x, y, z) in enumerate(endpoints)
    ]
    return _standing_components(rooms, nodes, standing_blockers, lava_pools)


def build_route_graph(
    rooms, connections, items, spawns, lava_pools, *, standing_blockers,
    hook_required_edges=None,
) -> dict:
    """items: list of (cls, x, y, z). spawns: list of (x, y, z)."""
    item_origins = [(int(x), int(y), int(z)) for _cls, x, y, z in items]
    if len(item_origins) != len(set(item_origins)):
        raise RouteGraphError("route graph contains stacked item origins")
    spawn_origins = {(int(x), int(y), int(z)) for x, y, z in spawns}
    if spawn_origins.intersection(item_origins):
        raise RouteGraphError("route graph item overlaps a spawn origin")
    # --- room centres + danger ---
    rc = [
        (r.wx + r.w / 2, r.wy + r.d / 2,
         r.floor_z + PLAYER_ORIGIN_FLOOR_OFFSET)
        for r in rooms
    ]
    lava_c = [((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, (b.z0 + b.z1) / 2)
              for b in lava_pools]

    def edge_risk(ca, cb) -> float:
        if not lava_c:
            return 0.0
        mid = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2, (ca[2] + cb[2]) / 2)
        d = min(_dist(mid, lc) for lc in lava_c)
        return round(max(0.0, 1.0 - d / 768.0), 3)   # within ~768u of lava = risky

    # Connection rows remain source metadata for diagnostics and risk priors.
    # They are not reachability evidence and never join standing components.
    edges = []
    for c in connections:
        if not (0 <= c.a < len(rc) and 0 <= c.b < len(rc)):
            continue
        dist = _dist(rc[c.a], rc[c.b])
        risk = edge_risk(rc[c.a], rc[c.b])
        edges.append({"a": c.a, "b": c.b, "kind": c.kind,
                      "dist": round(dist), "risk": risk})
    edges.sort(key=lambda value: (
        value["a"], value["b"], value["kind"], value["dist"], value["risk"]
    ))

    # --- nodes: items + spawns ---
    nodes = []
    item_nodes = []
    for cls, x, y, z in items:
        axis, val = _axis(cls)
        nid = len(nodes)
        node = {"id": nid, "type": "item", "class": cls,
                "x": int(x), "y": int(y), "z": int(z),
                "room": source_room_index(rooms, x, y, z),
                "respawn_s": _period(cls),
                "axis": axis, "value": val}
        nodes.append(node)
        item_nodes.append(node)
    for x, y, z in spawns:
        nodes.append({"id": len(nodes), "type": "spawn",
                      "x": int(x), "y": int(y), "z": int(z),
                      "room": source_room_index(rooms, x, y, z)})

    components = _standing_components(
        rooms, nodes, standing_blockers, lava_pools,
    )
    # Source component IDs are proposal evidence, not compiled authority.  They
    # make the exact standing-hull relation used for route selection explicit
    # in the sidecar so the source-freeze validator never has to reinterpret
    # diagnostic room edges as reachability.  A floor-assigned node can still
    # be absent when its standing hull is blocked; such a node is ineligible
    # for every published route.
    for node in nodes:
        node["source_component"] = components.get(node["id"])

    def route_metric(source, target):
        """Return a lower-bound metric only within one source component."""
        source_component = components.get(source["id"])
        target_component = components.get(target["id"])
        if (
            source_component is None or target_component is None
            or source_component != target_component
        ):
            return None
        source_point = (source["x"], source["y"], source["z"])
        target_point = (target["x"], target["y"], target["z"])
        physical = _dist(source_point, target_point)
        exposure = physical * edge_risk(source_point, target_point)
        return physical, exposure, physical + exposure

    def archetype_pool(archetype):
        if archetype == "offense":
            axes = ("offense", "value")
            pool = [node for node in item_nodes if node["axis"] in axes]
        elif archetype == "survival":
            axes = ("survival", "value")
            pool = [node for node in item_nodes if node["axis"] in axes]
        else:
            pool = list(item_nodes)
        return [node for node in pool if node["room"] >= 0]

    # --- generate archetype routes: value-weighted greedy loops ---
    def gen_route(archetype, start):
        pool = archetype_pool(archetype)
        if not pool:
            return None
        visited, seq, current = [], [], start
        budget = 6 if archetype != "control" else max(6, len(pool))
        remaining = list(pool)
        while remaining and len(seq) < budget:
            # Pick the best reachable value using endpoint-aware 3D cost.
            # Distinct same-room points therefore never receive a fake zero.
            def score(n):
                metric = route_metric(current, n)
                if metric is None:
                    return -math.inf
                return n["value"] / (1.0 + metric[2] / 1024.0)
            reachable = [
                node for node in remaining
                if route_metric(current, node) is not None
            ]
            if not reachable:
                break
            nxt = max(reachable, key=lambda node: (score(node), -node["id"]))
            seq.append(nxt)
            visited.append(nxt["id"])
            current = nxt
            remaining.remove(nxt)
        if len(seq) < 2:
            return None
        # close the loop back to start
        loop = [start, *seq, start]
        metrics = [
            route_metric(source, target)
            for source, target in zip(loop, loop[1:])
        ]
        if any(metric is None for metric in metrics):
            return None
        path_total = sum(metric[0] for metric in metrics)
        exposure = sum(metric[1] for metric in metrics)
        if path_total <= 0:
            return None
        # Room centres and predecessor edges are an internal selection/risk
        # model, not published route waypoints.  The compiled Atlas challenges
        # only this exact spawn/item endpoint loop, so claim its deterministic
        # geometric lower bound and let the oracle publish authoritative cost.
        claimed_distance = sum(
            _dist(
                (source["x"], source["y"], source["z"]),
                (target["x"], target["y"], target["z"]),
            )
            for source, target in zip(loop, loop[1:])
        )
        if claimed_distance <= 0:
            return None
        risk = round(exposure / path_total, 3)
        return {
            "archetype": archetype,
            "node_ids": visited,
            "items": [n["class"] for n in seq],
            "respawn_s": [n["respawn_s"] for n in seq],
            "dist": round(claimed_distance),
            "risk": risk,
            "value": round(sum(n["value"] for n in seq), 2),
        }

    spawn_nodes = [
        node for node in nodes
        if node["type"] == "spawn" and node["room"] >= 0
    ]
    first_spawn_by_start = {}
    for node in spawn_nodes:
        component = components.get(node["id"])
        if component is None:
            continue
        first_spawn_by_start.setdefault((node["room"], component), node)
    # Retain the original spawn-room rotation, but use its canonical first
    # node because the claim normalizer resolves a route's start the same way.
    route_starts = list(first_spawn_by_start.values())
    routes = []
    if route_starts:
        for i, arch in enumerate(ROUTE_ARCHETYPES):
            offset = i % len(route_starts)
            ordered_starts = route_starts[offset:] + route_starts[:offset]
            pool = archetype_pool(arch)
            start = next((
                candidate for candidate in ordered_starts
                if sum(
                    route_metric(candidate, node) is not None for node in pool
                ) >= 2
            ), None)
            if start is None:
                raise RouteGraphError(
                    f"no spawn component has two reachable {arch} items"
                )
            r = gen_route(arch, start)
            if r is None:
                raise RouteGraphError(f"could not build mandatory {arch} route")
            r["id"] = len(routes)
            r["start_room"] = start["room"]
            r["start_node_id"] = start["id"]
            r["source_component"] = components[start["id"]]
            routes.append(r)

    return {
        "version": 2,
        "respawn_table": RESPAWN_S,
        "nodes": nodes,
        "edges": edges,
        "routes": routes,
    }
