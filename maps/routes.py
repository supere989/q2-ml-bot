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
import heapq
import math
from typing import Dict, List, Tuple

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


def build_route_graph(rooms, connections, items, spawns, lava_pools,
                      hook_required_edges=None) -> dict:
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

    def room_of(x, y, z):
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
        # Overlapping generated rooms are legal.  Resolve their vertical
        # bands by closest floor, then stable room ordinal for exact ties.
        return min(compatible)[1] if compatible else -1

    def edge_risk(ca, cb) -> float:
        if not lava_c:
            return 0.0
        mid = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2, (ca[2] + cb[2]) / 2)
        d = min(_dist(mid, lc) for lc in lava_c)
        return round(max(0.0, 1.0 - d / 768.0), 3)   # within ~768u of lava = risky

    # --- room adjacency graph (undirected) ---
    adj: Dict[int, List[Tuple[int, float, float]]] = {i: [] for i in range(len(rooms))}
    edges = []
    for c in connections:
        if not (0 <= c.a < len(rc) and 0 <= c.b < len(rc)):
            continue
        dist = _dist(rc[c.a], rc[c.b])
        risk = edge_risk(rc[c.a], rc[c.b])
        adj[c.a].append((c.b, dist, risk))
        adj[c.b].append((c.a, dist, risk))
        edges.append({"a": c.a, "b": c.b, "kind": c.kind,
                      "dist": round(dist), "risk": risk})
    for neighbors in adj.values():
        neighbors.sort(key=lambda value: (value[0], value[1], value[2]))
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
                "room": room_of(x, y, z), "respawn_s": _period(cls),
                "axis": axis, "value": val}
        nodes.append(node)
        item_nodes.append(node)
    for x, y, z in spawns:
        nodes.append({"id": len(nodes), "type": "spawn",
                      "x": int(x), "y": int(y), "z": int(z),
                      "room": room_of(x, y, z)})

    # --- all-pairs deterministic paths over rooms (Dijkstra; rooms are few) ---
    def dijkstra(src):
        # The search minimizes risk-weighted distance and retains the exact
        # room-centre predecessor path for selection and risk exposure.  Those
        # hidden centres are not part of the published endpoint cost claim.
        best = {
            src: (0.0, 0.0, (src,), 0.0, ()),
        }
        pq = [(0.0, 0.0, (src,), src, 0.0, ())]
        while pq:
            weighted, physical, path, u, exposure, path_edges = heapq.heappop(pq)
            current = best.get(u)
            if current is None or (weighted, physical, path) != current[:3]:
                continue
            for v, w, risk in adj[u]:
                candidate = (
                    weighted + w * (1.0 + risk),
                    physical + w,
                    path + (v,),
                    exposure + w * risk,
                    path_edges + ((u, v, w, risk),),
                )
                previous = best.get(v)
                if previous is None or candidate[:3] < previous[:3]:
                    best[v] = candidate
                    heapq.heappush(
                        pq,
                        (
                            candidate[0], candidate[1], candidate[2], v,
                            candidate[3], candidate[4],
                        ),
                    )
        return best

    room_paths = {i: dijkstra(i) for i in range(len(rooms))}

    def route_metric(source, target):
        """Return physical distance, risk exposure, and selection cost."""
        ra = source["room"]
        rb = target["room"]
        if ra < 0 or rb < 0 or ra >= len(rooms) or rb >= len(rooms):
            return None
        source_point = (source["x"], source["y"], source["z"])
        target_point = (target["x"], target["y"], target["z"])
        if ra == rb:
            physical = _dist(source_point, target_point)
            exposure = physical * edge_risk(source_point, target_point)
            return physical, exposure, physical + exposure
        room_path = room_paths[ra].get(rb)
        if room_path is None:
            return None
        legs = [
            (
                _dist(source_point, rc[ra]),
                edge_risk(source_point, rc[ra]),
            ),
            *((edge[2], edge[3]) for edge in room_path[4]),
            (
                _dist(rc[rb], target_point),
                edge_risk(rc[rb], target_point),
            ),
        ]
        physical = sum(distance for distance, _risk in legs)
        exposure = sum(distance * risk for distance, risk in legs)
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
    first_spawn_by_room = {}
    for node in spawn_nodes:
        first_spawn_by_room.setdefault(node["room"], node)
    # Retain the original spawn-room rotation, but use its canonical first
    # node because the claim normalizer resolves a route's start the same way.
    route_starts = list(first_spawn_by_room.values())
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
            routes.append(r)

    return {
        "version": 1,
        "respawn_table": RESPAWN_S,
        "nodes": nodes,
        "edges": edges,
        "routes": routes,
    }
