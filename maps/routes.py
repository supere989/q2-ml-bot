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

    # --- all-pairs shortest path over rooms (Dijkstra; rooms are few) ---
    def dijkstra(src):
        dist = {i: math.inf for i in range(len(rooms))}
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w, risk in adj[u]:
                nd = d + w * (1.0 + risk)   # risk inflates effective travel cost
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    room_dist = {i: dijkstra(i) for i in range(len(rooms))}

    def route_cost(source, target):
        ra = source["room"]
        rb = target["room"]
        if ra < 0 or rb < 0 or ra >= len(rooms) or rb >= len(rooms):
            return math.inf
        source_point = (source["x"], source["y"], source["z"])
        target_point = (target["x"], target["y"], target["z"])
        if ra == rb:
            return _dist(source_point, target_point)
        inter_room = room_dist[ra].get(rb, math.inf)
        if not math.isfinite(inter_room):
            return math.inf
        return (
            _dist(source_point, rc[ra])
            + inter_room
            + _dist(rc[rb], target_point)
        )

    # --- generate archetype routes: value-weighted greedy loops ---
    def gen_route(archetype, start):
        if archetype == "offense":
            pool = [n for n in item_nodes if n["axis"] in ("offense", "value")]
        elif archetype == "survival":
            pool = [n for n in item_nodes if n["axis"] in ("survival", "value")]
        else:  # control / balanced: everything, value-weighted
            pool = list(item_nodes)
        pool = [n for n in pool if n["room"] >= 0]
        if not pool:
            return None
        visited, seq, current = [], [], start
        budget = 6 if archetype != "control" else max(6, len(pool))
        remaining = list(pool)
        while remaining and len(seq) < budget:
            # Pick the best reachable value using endpoint-aware 3D cost.
            # Distinct same-room points therefore never receive a fake zero.
            def score(n):
                c = route_cost(current, n)
                if not math.isfinite(c):
                    return -math.inf
                return n["value"] / (1.0 + c / 1024.0)
            reachable = [
                node for node in remaining
                if math.isfinite(route_cost(current, node))
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
        total = sum(route_cost(source, target)
                    for source, target in zip(loop, loop[1:]))
        if not math.isfinite(total):
            return None
        risk = round(sum(edge_risk(rc[seq[i]["room"]], rc[seq[i + 1]["room"]])
                         for i in range(len(seq) - 1)) / max(1, len(seq) - 1), 3)
        return {
            "archetype": archetype,
            "node_ids": visited,
            "items": [n["class"] for n in seq],
            "respawn_s": [n["respawn_s"] for n in seq],
            "dist": round(total),
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
    route_starts = [first_spawn_by_room[node["room"]] for node in spawn_nodes]
    routes = []
    if route_starts:
        for i, arch in enumerate(ROUTE_ARCHETYPES):
            start = route_starts[i % len(route_starts)]
            r = gen_route(arch, start)
            if r:
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
