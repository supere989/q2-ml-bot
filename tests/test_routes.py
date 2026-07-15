from types import SimpleNamespace
import json
import math

import pytest

from maps.generator import generate_map
from maps.routes import (
    ROUTE_ARCHETYPES,
    RouteGraphError,
    _standing_components,
    build_route_graph,
)


def room(x: int, *, y: int = 0, floor_z: int = 0,
         width: int = 100, depth: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        wx=x, wy=y, w=width, d=depth, floor_z=floor_z,
    )


def connection(left: int, right: int) -> SimpleNamespace:
    return SimpleNamespace(a=left, b=right, kind="hall")


def blocker(x0: int, y0: int, z0: int,
            x1: int, y1: int, z1: int) -> SimpleNamespace:
    return SimpleNamespace(x0=x0, y0=y0, z0=z0, x1=x1, y1=y1, z1=z1)


def test_unreachable_high_value_item_does_not_suppress_reachable_routes() -> None:
    graph = build_route_graph(
        rooms=[room(0, width=128, depth=128),
               room(64, width=160, depth=128), room(1000)],
        connections=[connection(0, 1)],
        items=[
            ("item_health", 20, 20, 24),
            ("item_armor_combat", 40, 20, 24),
            ("weapon_supershotgun", 140, 20, 24),
            ("weapon_rocketlauncher", 160, 20, 24),
            ("item_quad", 1020, 20, 24),
        ],
        spawns=[(50, 50, 24)],
        lava_pools=[],
        standing_blockers=[],
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
        standing_blockers=[],
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
        standing_blockers=[],
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
        spawns=[(50, 50, 24)],
        lava_pools=[],
        standing_blockers=[],
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
        standing_blockers=[],
    )

    assert [node["room"] for node in graph["nodes"]] == [0, 1, 0, -1]


def test_same_room_loop_uses_local_3d_distance_deterministically() -> None:
    kwargs = {
        "rooms": [room(0, width=128, depth=128)],
        "connections": [],
        "items": [
            ("weapon_supershotgun", 36, 26, 24),
            ("weapon_rocketlauncher", 96, 26, 24),
            ("item_health", 36, 96, 24),
            ("item_armor_combat", 96, 96, 24),
        ],
        "spawns": [(26, 26, 24)],
        "lava_pools": [],
        "standing_blockers": [],
    }

    first = build_route_graph(**kwargs)
    second = build_route_graph(**kwargs)
    offense = next(
        route for route in first["routes"] if route["archetype"] == "offense"
    )

    assert offense["dist"] == 140
    assert first == second


def test_final_wall_splits_one_room_into_distinct_route_components() -> None:
    graph = build_route_graph(
        rooms=[room(0, width=256, depth=160)],
        connections=[],
        items=[
            ("weapon_supershotgun", 48, 40, 24),
            ("weapon_rocketlauncher", 80, 40, 24),
            ("item_health", 48, 120, 24),
            ("item_armor_combat", 80, 120, 24),
            # Same source room, but physically isolated behind the final wall.
            ("item_quad", 208, 80, 24),
        ],
        spawns=[(48, 80, 24)],
        lava_pools=[],
        standing_blockers=[blocker(120, 0, 0, 136, 160, 96)],
    )

    nodes = {node["id"]: node for node in graph["nodes"]}
    assert [route["archetype"] for route in graph["routes"]] == ROUTE_ARCHETYPES
    assert nodes[0]["source_component"] is not None
    assert nodes[0]["source_component"] == nodes[5]["source_component"]
    assert nodes[4]["source_component"] is not None
    assert nodes[4]["source_component"] != nodes[5]["source_component"]
    assert all(
        all(nodes[node_id]["x"] < 120 for node_id in route["node_ids"])
        for route in graph["routes"]
    )


def test_split_room_retains_one_route_start_per_source_component() -> None:
    graph = build_route_graph(
        rooms=[room(0, width=256, depth=160)],
        connections=[],
        items=[
            ("weapon_supershotgun", 176, 40, 24),
            ("weapon_rocketlauncher", 208, 40, 24),
            ("item_health", 176, 120, 24),
            ("item_armor_combat", 208, 120, 24),
        ],
        # The canonical first spawn has no items in its half. The second spawn
        # shares room metadata but owns the only route-capable component.
        spawns=[(48, 80, 24), (208, 80, 24)],
        lava_pools=[],
        standing_blockers=[blocker(120, 0, 0, 136, 160, 96)],
    )

    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes[4]["room"] == nodes[5]["room"] == 0
    assert nodes[4]["source_component"] != nodes[5]["source_component"]
    assert all(route["start_node_id"] == 5 for route in graph["routes"])


def test_exact_player_width_gap_is_blocked_by_hull_contact() -> None:
    rooms = [room(0, width=256, depth=128)]
    nodes = [
        {"id": 0, "room": 0, "x": 48, "y": 48, "z": 24},
        {"id": 1, "room": 0, "x": 208, "y": 48, "z": 24},
    ]
    # The two solids leave y=32..64: exactly the player's 32-unit width.
    # A hull centered at y=48 touches both faces and CM rejects the sweep.
    wall = [
        blocker(96, 0, 0, 160, 32, 96),
        blocker(96, 64, 0, 160, 128, 96),
    ]

    components = _standing_components(rooms, nodes, wall, [])

    assert components[0] != components[1]


def test_metadata_connection_never_bridges_disjoint_floor_components() -> None:
    with pytest.raises(RouteGraphError, match="two reachable offense items"):
        build_route_graph(
            rooms=[
                room(0, width=128, depth=128),
                room(256, width=128, depth=128),
            ],
            # This row remains useful diagnostic metadata, but emits no floor
            # or standing-hull witness and therefore grants no reachability.
            connections=[connection(0, 1)],
            items=[
                ("weapon_supershotgun", 48, 48, 24),
                ("weapon_rocketlauncher", 304, 48, 24),
                ("item_health", 48, 80, 24),
                ("item_armor_combat", 304, 80, 24),
            ],
            spawns=[(80, 80, 24)],
            lava_pools=[],
            standing_blockers=[],
        )

    graph = build_route_graph(
        rooms=[
            room(0, width=128, depth=160),
            room(256, width=128, depth=160),
        ],
        connections=[connection(0, 1)],
        items=[
            ("weapon_supershotgun", 48, 40, 24),
            ("weapon_rocketlauncher", 80, 40, 24),
            ("item_health", 48, 120, 24),
            ("item_armor_combat", 80, 120, 24),
            ("item_quad", 304, 80, 24),
        ],
        spawns=[(48, 80, 24)],
        lava_pools=[],
        standing_blockers=[],
    )
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes[4]["source_component"] is not None
    assert nodes[4]["source_component"] != nodes[5]["source_component"]
    assert all(
        4 not in route["node_ids"]
        and route["source_component"] == nodes[5]["source_component"]
        for route in graph["routes"]
    )


def test_lava_strip_splits_components_deterministically() -> None:
    kwargs = {
        "rooms": [room(0, width=256, depth=160)],
        "connections": [],
        "items": [
            ("weapon_supershotgun", 48, 40, 24),
            ("weapon_rocketlauncher", 80, 40, 24),
            ("item_health", 48, 120, 24),
            ("item_armor_combat", 80, 120, 24),
            ("item_quad", 208, 80, 24),
        ],
        "spawns": [(48, 80, 24)],
        "lava_pools": [blocker(120, 0, 0, 136, 160, 96)],
        "standing_blockers": [],
    }

    first = build_route_graph(**kwargs)
    second = build_route_graph(**kwargs)
    nodes = {node["id"]: node for node in first["nodes"]}

    assert first == second
    assert nodes[4]["source_component"] is not None
    assert nodes[4]["source_component"] != nodes[5]["source_component"]
    assert all(
        4 not in route["node_ids"]
        and route["source_component"] == nodes[5]["source_component"]
        for route in first["routes"]
    )


def test_route_graph_rejects_stacked_items_and_incomplete_archetypes() -> None:
    with pytest.raises(RouteGraphError, match="stacked item origins"):
        build_route_graph(
            rooms=[room(0)], connections=[],
            items=[
                ("weapon_supershotgun", 20, 20, 24),
                ("weapon_rocketlauncher", 20, 20, 24),
            ],
            spawns=[(50, 50, 24)], lava_pools=[], standing_blockers=[],
        )

    with pytest.raises(RouteGraphError, match="two reachable offense items"):
        build_route_graph(
            rooms=[room(0)], connections=[],
            items=[
                ("weapon_rocketlauncher", 20, 20, 24),
                ("item_health", 40, 20, 24),
                ("item_armor_combat", 60, 20, 24),
            ],
            spawns=[(50, 50, 24)], lava_pools=[], standing_blockers=[],
        )


def test_mandatory_route_repairs_isolated_rotated_start() -> None:
    graph = build_route_graph(
        rooms=[room(0, width=128, depth=128),
               room(64, width=192, depth=128), room(1000, width=128, depth=128)],
        connections=[connection(0, 1)],
        items=[
            ("weapon_supershotgun", 32, 32, 24),
            ("weapon_rocketlauncher", 160, 32, 24),
            ("item_health", 64, 64, 24),
            ("item_armor_combat", 192, 64, 24),
            ("item_adrenaline", 1020, 20, 24),
        ],
        # Control initially rotates to isolated room 2.  It must choose a
        # connected source component rather than publishing no route.
        spawns=[(48, 96, 24), (144, 96, 24), (1050, 50, 24)],
        lava_pools=[],
        standing_blockers=[],
    )

    assert [route["archetype"] for route in graph["routes"]] == ROUTE_ARCHETYPES
    control = next(route for route in graph["routes"]
                   if route["archetype"] == "control")
    assert control["start_room"] == 0
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(control["node_ids"]) >= 2
    assert len(set(control["node_ids"])) == len(control["node_ids"])
    assert {nodes[node_id]["room"] for node_id in control["node_ids"]} <= {0, 1}


def test_route_distance_and_risk_use_connected_endpoint_lower_bound() -> None:
    lava = blocker(1020, 40, 14, 1040, 60, 34)
    graph = build_route_graph(
        rooms=[room(0, width=800, depth=128),
               room(700, width=800, depth=128),
               room(1400, width=800, depth=128)],
        connections=[connection(0, 1), connection(1, 2)],
        items=[
            ("weapon_rocketlauncher", 2020, 50, 24),
            ("weapon_supershotgun", 2080, 50, 24),
            ("item_health", 2020, 80, 24),
            ("item_armor_combat", 2080, 80, 24),
        ],
        spawns=[(50, 50, 24)],
        lava_pools=[lava],
        standing_blockers=[],
    )

    offense = next(route for route in graph["routes"]
                   if route["archetype"] == "offense")
    # Published cost remains exact endpoint-loop geometry. The source component
    # only filters impossible legs; it is not a compiled route/cost authority.
    assert offense["dist"] == 4060
    assert 0.0 < offense["risk"] < 1.0


def _assert_complete_generated_routes(routes: dict) -> None:
    assert routes["version"] == 2
    item_nodes = [node for node in routes["nodes"] if node["type"] == "item"]
    item_origins = [(node["x"], node["y"], node["z"]) for node in item_nodes]
    spawn_nodes = [node for node in routes["nodes"] if node["type"] == "spawn"]
    spawn_origins = {(node["x"], node["y"], node["z"]) for node in spawn_nodes}
    assert len(item_origins) == len(set(item_origins))
    assert not spawn_origins.intersection(item_origins)
    assert all(node["room"] >= 0 for node in [*item_nodes, *spawn_nodes])
    assert [route["archetype"] for route in routes["routes"]] == ROUTE_ARCHETYPES

    nodes = {node["id"]: node for node in routes["nodes"]}
    for route in routes["routes"]:
        assert len(route["node_ids"]) >= 2
        assert len(set(route["node_ids"])) == len(route["node_ids"])
        start = nodes[route["start_node_id"]]
        assert start["type"] == "spawn"
        assert start["room"] == route["start_room"]
        assert start["source_component"] == route["source_component"]
        assert route["source_component"] is not None
        assert all(
            nodes[node_id]["source_component"] == route["source_component"]
            for node_id in route["node_ids"]
        )
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


def test_retired_arena_open_contact_route_repairs_within_component(
    tmp_path,
) -> None:
    map_path, _ = generate_map(
        "contact_route_regression", 71430402, tmp_path, style="arena_open"
    )
    routes = json.loads(map_path.with_suffix(".routes.json").read_text())

    _assert_complete_generated_routes(routes)
    nodes = {node["id"]: node for node in routes["nodes"]}
    components = {
        node["source_component"] for node in nodes.values()
    }
    assert None not in components
    assert len(components) == 1
    assert all(
        nodes[node_id]["source_component"] == route["source_component"]
        for route in routes["routes"]
        for node_id in route["node_ids"]
    )


def test_arena_vertical_seed_71433501_seeds_spawn_component_route_axes(
    tmp_path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    map_path, _ = generate_map(
        "spawn_route_anchor_regression", 71433501, first,
        style="arena_vertical",
    )
    cold_map, _ = generate_map(
        "spawn_route_anchor_regression", 71433501, second,
        style="arena_vertical",
    )
    assert map_path.read_bytes() == cold_map.read_bytes()
    assert (
        map_path.with_suffix(".routes.json").read_bytes()
        == cold_map.with_suffix(".routes.json").read_bytes()
    )
    routes = json.loads(map_path.with_suffix(".routes.json").read_text())

    _assert_complete_generated_routes(routes)
    spawn_components = {
        node["source_component"] for node in routes["nodes"]
        if node["type"] == "spawn"
    }
    assert None not in spawn_components
    for axes in ({"offense", "value"}, {"survival", "value"}):
        assert any(
            sum(
                node["type"] == "item"
                and node["source_component"] == component
                and node["axis"] in axes
                for node in routes["nodes"]
            ) >= 2
            for component in spawn_components
        )


def test_route_refusal_publishes_no_partial_source_member(
    tmp_path, monkeypatch,
) -> None:
    def refuse_route_graph(**_kwargs):
        raise RouteGraphError("injected route refusal")

    monkeypatch.setattr("maps.routes.build_route_graph", refuse_route_graph)
    with pytest.raises(RouteGraphError, match="injected route refusal"):
        generate_map("atomic_route_refusal", 7, tmp_path, style="open")

    assert list(tmp_path.iterdir()) == []


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
