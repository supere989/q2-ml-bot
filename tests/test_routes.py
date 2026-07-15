from types import SimpleNamespace
import json
import math

import pytest

from maps.generator import generate_map
from maps.routes import ROUTE_ARCHETYPES, RouteGraphError, build_route_graph


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
            ("item_armor_combat", 40, 20, 24),
            ("weapon_supershotgun", 220, 20, 24),
            ("weapon_rocketlauncher", 240, 20, 24),
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
        assert 4 not in route["node_ids"]


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
            ("item_health", 20, 80, 24),
            ("item_armor_combat", 80, 80, 24),
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


def test_route_graph_rejects_stacked_items_and_incomplete_archetypes() -> None:
    with pytest.raises(RouteGraphError, match="stacked item origins"):
        build_route_graph(
            rooms=[room(0)], connections=[],
            items=[
                ("weapon_supershotgun", 20, 20, 24),
                ("weapon_rocketlauncher", 20, 20, 24),
            ],
            spawns=[(10, 10, 24)], lava_pools=[],
        )

    with pytest.raises(RouteGraphError, match="two reachable offense items"):
        build_route_graph(
            rooms=[room(0)], connections=[],
            items=[
                ("weapon_rocketlauncher", 20, 20, 24),
                ("item_health", 40, 20, 24),
                ("item_armor_combat", 60, 20, 24),
            ],
            spawns=[(10, 10, 24)], lava_pools=[],
        )


def test_mandatory_route_repairs_isolated_rotated_start() -> None:
    graph = build_route_graph(
        rooms=[room(0), room(200), room(1000)],
        connections=[connection(0, 1)],
        items=[
            ("weapon_supershotgun", 20, 20, 24),
            ("weapon_rocketlauncher", 220, 20, 24),
            ("item_health", 40, 20, 24),
            ("item_armor_combat", 240, 20, 24),
            ("item_adrenaline", 1020, 20, 24),
        ],
        # Control initially rotates to isolated room 2.  It must choose a
        # connected source component rather than publishing no route.
        spawns=[(10, 10, 24), (210, 10, 24), (1010, 10, 24)],
        lava_pools=[],
    )

    assert [route["archetype"] for route in graph["routes"]] == ROUTE_ARCHETYPES
    control = next(route for route in graph["routes"]
                   if route["archetype"] == "control")
    assert control["start_room"] == 0
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(control["node_ids"]) >= 2
    assert len(set(control["node_ids"])) == len(control["node_ids"])
    assert {nodes[node_id]["room"] for node_id in control["node_ids"]} <= {0, 1}


def test_route_distance_matches_endpoints_and_risk_follows_closed_dijkstra() -> None:
    lava = SimpleNamespace(x0=140, x1=160, y0=40, y1=60, z0=14, z1=34)
    graph = build_route_graph(
        rooms=[room(0), room(200), room(2000)],
        connections=[connection(0, 1), connection(1, 2)],
        items=[
            ("weapon_rocketlauncher", 2020, 50, 24),
            ("weapon_supershotgun", 2080, 50, 24),
            ("item_health", 2020, 80, 24),
            ("item_armor_combat", 2080, 80, 24),
        ],
        spawns=[(10, 50, 24)],
        lava_pools=[lava],
    )

    offense = next(route for route in graph["routes"]
                   if route["archetype"] == "offense")
    # The published cost is the exact endpoint-loop geometry (the room-centre
    # path stays internal), while both opening and closing legs traverse the
    # predecessor path and contribute nonzero lava exposure.
    assert offense["dist"] == 4140
    assert 0.0 < offense["risk"] < 1.0


def _assert_complete_generated_routes(routes: dict) -> None:
    item_nodes = [node for node in routes["nodes"] if node["type"] == "item"]
    item_origins = [(node["x"], node["y"], node["z"]) for node in item_nodes]
    spawn_nodes = [node for node in routes["nodes"] if node["type"] == "spawn"]
    spawn_origins = {(node["x"], node["y"], node["z"]) for node in spawn_nodes}
    assert len(item_origins) == len(set(item_origins))
    assert not spawn_origins.intersection(item_origins)
    assert all(node["room"] >= 0 for node in [*item_nodes, *spawn_nodes])
    assert [route["archetype"] for route in routes["routes"]] == ROUTE_ARCHETYPES

    nodes = {node["id"]: node for node in routes["nodes"]}
    first_spawn_by_room = {}
    for node in spawn_nodes:
        first_spawn_by_room.setdefault(node["room"], node)
    for route in routes["routes"]:
        assert len(route["node_ids"]) >= 2
        assert len(set(route["node_ids"])) == len(route["node_ids"])
        start = first_spawn_by_room[route["start_room"]]
        sequence = [start, *(nodes[node_id] for node_id in route["node_ids"]), start]
        assert all(
            (left["x"], left["y"], left["z"])
            != (right["x"], right["y"], right["z"])
            for left, right in zip(sequence, sequence[1:])
        )
        geometric = sum(
            math.dist(
                (left["x"], left["y"], left["z"]),
                (right["x"], right["y"], right["z"]),
            )
            for left, right in zip(sequence, sequence[1:])
        )
        assert route["dist"] == round(geometric)
        ratio = max(route["dist"], geometric) / min(route["dist"], geometric)
        difference = abs(route["dist"] - geometric)
        assert not (ratio > 2.0 and difference > 1024.0)
        assert 0.0 <= route["risk"] <= 1.0


def test_open_seed_91470002_repairs_mandatory_control_route(tmp_path) -> None:
    map_path, _ = generate_map(
        "control_route_regression", 91470002, tmp_path, style="open"
    )
    routes = json.loads(map_path.with_suffix(".routes.json").read_text())

    _assert_complete_generated_routes(routes)
    control = next(route for route in routes["routes"]
                   if route["archetype"] == "control")
    assert len(control["node_ids"]) >= 2


@pytest.mark.parametrize(
    ("style", "seed"),
    [
        ("arena_open", 91470400),
        ("arena_lanes", 91470602),
    ],
)
def test_historical_duplicate_item_seeds_have_unique_route_legs(
    tmp_path, style, seed,
) -> None:
    map_path, _ = generate_map(
        f"unique_{style}_{seed}", seed, tmp_path, style=style
    )
    routes = json.loads(map_path.with_suffix(".routes.json").read_text())
    _assert_complete_generated_routes(routes)


def test_generator_route_population_audit(tmp_path) -> None:
    cases = [
        ("open", 91470002), ("open", 71500401),
        ("towers", 91470103), ("towers", 71500602),
        ("canyon", 91470201), ("canyon", 71500703),
        ("pits", 91470301), ("pits", 71500404),
        ("arena_open", 91470402), ("arena_open", 71501001),
        ("arena_vertical", 91470502), ("arena_vertical", 71501403),
        ("arena_lanes", 91470600), ("arena_lanes", 71501104),
        ("mixed", 71500501), ("mixed", 71500504),
    ]
    for style, seed in cases:
        output = tmp_path / f"{style}_{seed}"
        output.mkdir()
        map_path, _ = generate_map("audit", seed, output, style=style)
        routes = json.loads(map_path.with_suffix(".routes.json").read_text())
        _assert_complete_generated_routes(routes)


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
