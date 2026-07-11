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
    rc = [(r.wx + r.w / 2, r.wy + r.d / 2, r.floor_z) for r in rooms]
    lava_c = [((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, (b.z0 + b.z1) / 2)
              for b in lava_pools]

    def room_of(x, y):
        for i, r in enumerate(rooms):
            if r.wx <= x <= r.wx + r.w and r.wy <= y <= r.wy + r.d:
                return i
        # nearest room centre fallback
        return min(range(len(rc)), key=lambda i: _dist((x, y, rc[i][2]),
                                                        (x, y, rc[i][2]))) if rc else -1

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
        if c.a >= len(rc) or c.b >= len(rc):
            continue
        dist = _dist(rc[c.a], rc[c.b])
        risk = edge_risk(rc[c.a], rc[c.b])
        adj[c.a].append((c.b, dist, risk))
        adj[c.b].append((c.a, dist, risk))
        edges.append({"a": c.a, "b": c.b, "kind": c.kind,
                      "dist": round(dist), "risk": risk})

    # --- nodes: items + spawns ---
    nodes = []
    item_nodes = []
    for cls, x, y, z in items:
        axis, val = _axis(cls)
        nid = len(nodes)
        node = {"id": nid, "type": "item", "class": cls,
                "x": int(x), "y": int(y), "z": int(z),
                "room": room_of(x, y), "respawn_s": _period(cls),
                "axis": axis, "value": val}
        nodes.append(node)
        item_nodes.append(node)
    for x, y, z in spawns:
        nodes.append({"id": len(nodes), "type": "spawn",
                      "x": int(x), "y": int(y), "z": int(z), "room": room_of(x, y)})

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

    def route_cost(ra, rb):
        if ra < 0 or rb < 0 or ra >= len(rooms) or rb >= len(rooms):
            return math.inf
        return room_dist[ra].get(rb, math.inf)

    # --- generate archetype routes: value-weighted greedy loops ---
    def gen_route(archetype, start_room):
        if archetype == "offense":
            pool = [n for n in item_nodes if n["axis"] in ("offense", "value")]
        elif archetype == "survival":
            pool = [n for n in item_nodes if n["axis"] in ("survival", "value")]
        else:  # control / balanced: everything, value-weighted
            pool = list(item_nodes)
        pool = [n for n in pool if n["room"] >= 0]
        if not pool:
            return None
        visited, seq, cur = [], [], start_room
        budget = 6 if archetype != "control" else max(6, len(pool))
        remaining = list(pool)
        while remaining and len(seq) < budget:
            # pick best value/cost item reachable from current room
            def score(n):
                c = route_cost(cur, n["room"])
                if not math.isfinite(c) or c <= 0:
                    c = 1.0
                return n["value"] / (1.0 + c / 1024.0)
            nxt = max(remaining, key=score)
            if not math.isfinite(route_cost(cur, nxt["room"])):
                break
            seq.append(nxt)
            visited.append(nxt["id"])
            cur = nxt["room"]
            remaining.remove(nxt)
        if len(seq) < 2:
            return None
        # close the loop back to start
        total = sum(route_cost(seq[i]["room"], seq[i + 1]["room"])
                    for i in range(len(seq) - 1))
        total += route_cost(start_room, seq[0]["room"])
        total += route_cost(seq[-1]["room"], start_room)
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

    spawn_rooms = [room_of(x, y) for x, y, z in spawns] or [0]
    routes = []
    for i, arch in enumerate(ROUTE_ARCHETYPES):
        sr = spawn_rooms[i % len(spawn_rooms)]
        r = gen_route(arch, sr)
        if r:
            r["id"] = len(routes)
            r["start_room"] = sr
            routes.append(r)

    return {
        "version": 1,
        "respawn_table": RESPAWN_S,
        "nodes": nodes,
        "edges": edges,
        "routes": routes,
    }
