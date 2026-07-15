from types import SimpleNamespace

from maps.routes import build_route_graph


def room(x: int) -> SimpleNamespace:
    return SimpleNamespace(wx=x, wy=0, w=100, d=100, floor_z=0)


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


def test_outside_point_falls_back_to_nearest_room_center() -> None:
    graph = build_route_graph(
        rooms=[room(0), room(1000)],
        connections=[],
        items=[("item_health", 900, 300, 24)],
        spawns=[],
        lava_pools=[],
    )

    assert graph["nodes"][0]["room"] == 1
