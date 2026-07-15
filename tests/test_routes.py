from types import SimpleNamespace
import json

from maps.generator import generate_map
from maps.routes import build_route_graph


def room(x: int, *, y: int = 0, floor_z: int = 0,
         width: int = 100, depth: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        wx=x, wy=y, w=width, d=depth, floor_z=floor_z,
    )


def connection(left: int, right: int) -> SimpleNamespace:
    return SimpleNamespace(a=left, b=right, kind="hall")


def test_unreachable_high_value_item_does_not_suppress_reachable_routes() -> None:
    graph = build_route_graph(
        rooms=[room(0), room(200), room(1000)],
        connections=[connection(0, 1)],
        items=[
            ("item_health", 20, 20, 24),
            ("weapon_supershotgun", 220, 20, 24),
            ("item_quad", 1020, 20, 24),
        ],
        spawns=[(30, 30, 24)],
        lava_pools=[],
    )

    assert graph["routes"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    for route in graph["routes"]:
        assert len(route["node_ids"]) >= 2
        assert {nodes[node_id]["room"] for node_id in route["node_ids"]} <= {0, 1}


def test_outside_point_remains_unassigned() -> None:
    graph = build_route_graph(
        rooms=[room(0), room(1000)],
        connections=[],
        items=[("item_health", 900, 300, 24)],
        spawns=[],
        lava_pools=[],
    )

    assert graph["nodes"][0]["room"] == -1


def test_unassigned_spawn_does_not_fallback_to_room_zero() -> None:
    graph = build_route_graph(
        rooms=[room(0)],
        connections=[],
        items=[
            ("weapon_supershotgun", 20, 20, 24),
            ("weapon_rocketlauncher", 80, 20, 24),
        ],
        spawns=[(50, 50, 285)],
        lava_pools=[],
    )

    assert graph["nodes"][-1]["room"] == -1
    assert graph["routes"] == []


def test_elevated_same_xy_item_never_enters_routes() -> None:
    items = [
        ("weapon_supershotgun", 20, 20, 24),
        ("weapon_rocketlauncher", 80, 20, 24),
        ("item_health", 20, 80, 24),
        ("item_armor_combat", 80, 80, 24),
        # Observed failure pattern: a supported high platform shares the base
        # room's XY footprint but has no room-floor band or inbound route.
        ("item_quad", 50, 50, 285),
    ]
    graph = build_route_graph(
        rooms=[room(0)],
        connections=[],
        items=items,
        spawns=[(10, 10, 24)],
        lava_pools=[],
    )

    assert graph["nodes"][4]["room"] == -1
    assert [route["archetype"] for route in graph["routes"]] == [
        "offense", "survival", "control", "balanced",
    ]
    assert all(4 not in route["node_ids"] for route in graph["routes"])


def test_overlapping_rooms_choose_closest_compatible_floor_band() -> None:
    graph = build_route_graph(
        rooms=[
            room(0, floor_z=0),
            room(0, floor_z=256),
            room(0, floor_z=8),
        ],
        connections=[],
        items=[
            ("item_health", 20, 20, 24),
            ("item_health", 20, 20, 280),
            ("item_health", 20, 20, 28),
            ("item_health", 20, 20, 100),
        ],
        spawns=[],
        lava_pools=[],
    )

    assert [node["room"] for node in graph["nodes"]] == [0, 1, 0, -1]


def test_same_room_loop_uses_local_3d_distance_deterministically() -> None:
    kwargs = {
        "rooms": [room(0)],
        "connections": [],
        "items": [
            ("weapon_supershotgun", 20, 10, 24),
            ("weapon_rocketlauncher", 80, 10, 24),
        ],
        "spawns": [(10, 10, 24)],
        "lava_pools": [],
    }

    first = build_route_graph(**kwargs)
    second = build_route_graph(**kwargs)
    offense = next(
        route for route in first["routes"] if route["archetype"] == "offense"
    )

    assert offense["dist"] == 140
    assert first == second


def test_generator_places_every_published_item_on_a_route_floor_band(
    tmp_path,
) -> None:
    map_path, _ = generate_map(
        "route_floor", 91470600, tmp_path, style="arena_lanes"
    )
    routes = json.loads(map_path.with_suffix(".routes.json").read_text())
    item_nodes = [node for node in routes["nodes"] if node["type"] == "item"]
    spawn_nodes = [node for node in routes["nodes"] if node["type"] == "spawn"]

    assert item_nodes
    assert spawn_nodes
    assert all(node["room"] >= 0 for node in [*item_nodes, *spawn_nodes])
    assert [route["archetype"] for route in routes["routes"]] == [
        "offense", "survival", "control", "balanced",
    ]
    quad = next(node for node in item_nodes if node["class"] == "item_quad")
    assert quad["room"] >= 0
