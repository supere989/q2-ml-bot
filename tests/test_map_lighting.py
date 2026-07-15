from argparse import Namespace
import json
import struct

from maps.generator import (
    FloorLightRegion,
    HorizontalSurface,
    MIN_FLOOR_LIGHT_COVERAGE,
    MIN_SAFE_HEADROOM,
    MapWriter,
    MapGenerator,
    PALETTES,
    Room,
    SolidBox,
    T_CEIL,
    TOWER_H_MIN,
    floor_light_coverage,
    generate_map,
    unsafe_horizontal_sandwiches,
)
from tools.validate_maps import (
    Q2_BSP_LIGHTING_LUMP,
    Q2_BSP_LUMPS,
    _compiled_lightdata_metrics,
    _parse_entities,
    static_validate,
)


def _static_args():
    return Namespace(
        min_spawns=0,
        min_spawn_distance=0.0,
        min_span=0.0,
        min_spawn_area=0.0,
        min_weapons=0,
        min_pickups=0,
        min_hook_zones=0,
        min_light_coverage=MIN_FLOOR_LIGHT_COVERAGE,
    )


def _drop_region_lights(text: str, region_id: str) -> str:
    """Remove point-entity blocks for one floor region from map source."""
    output = []
    block = []
    depth = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if depth == 0 and stripped == "{":
            block = [line]
            depth = 1
            continue
        if depth > 0:
            block.append(line)
            if stripped == "{":
                depth += 1
            elif stripped == "}":
                depth -= 1
                if depth == 0:
                    entity = "".join(block)
                    if not (
                        '"classname" "light"' in entity
                        and f'"_ml_region" "{region_id}"' in entity
                    ):
                        output.append(entity)
                    block = []
            continue
        output.append(line)
    return "".join(output)


def _drop_interior_light(text: str, zone_id: str) -> str:
    output = []
    block = []
    depth = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if depth == 0 and stripped == "{":
            block = [line]
            depth = 1
            continue
        if depth > 0:
            block.append(line)
            if stripped == "{":
                depth += 1
            elif stripped == "}":
                depth -= 1
                if depth == 0:
                    entity = "".join(block)
                    if not (
                        '"_ml_interior_light" "1"' in entity
                        and f'"_ml_zone" "{zone_id}"' in entity
                    ):
                        output.append(entity)
                    block = []
            continue
        output.append(line)
    return "".join(output)


def _insert_world_brushes(text: str, brushes) -> str:
    marker = "\n}\n\n{"
    assert marker in text
    return text.replace(marker, "\n" + "\n".join(brushes) + marker, 1)


def test_platform_shadow_gets_under_platform_light():
    generator = MapGenerator(seed=7, style="open")
    generator.rooms = [
        Room(
            gx=0, gy=0, wx=0, wy=0, w=512, d=512,
            floor_z=0, ceil_z=256, kind="room",
        )
    ]
    platform = SolidBox(128, 128, 120, 384, 384, 136)
    generator.spawn_blockers.append(platform)
    generator.light_occluders.append(platform)

    generator._emit_floor_lighting()

    assert len(generator.light_regions) == 1
    assert any(source.kind == "under_platform"
               for source in generator.light_sources)
    region = generator.light_regions[0]
    region_sources = [
        source for source in generator.light_sources
        if source.region_id == region.region_id
    ]
    assert floor_light_coverage(
        region, region_sources, generator.light_occluders
    ) >= MIN_FLOOR_LIGHT_COVERAGE


def test_interior_anchor_rejects_sample_without_safe_headroom():
    generator = MapGenerator(seed=1, style="towers")
    room = Room(
        gx=0, gy=0, wx=0, wy=0, w=256, d=256,
        floor_z=0, ceil_z=320, kind="room",
        platforms=[{
            "z": 240, "thick": 16,
            "x0": 32, "y0": 32, "x1": 224, "y1": 224,
        }],
    )
    generator.rooms = [room]
    lower_platform = SolidBox(32, 32, 88, 144, 224, 112)
    upper_platform = SolidBox(32, 32, 232, 224, 224, 256)
    generator.spawn_blockers.extend([lower_platform, upper_platform])
    generator.light_occluders.extend([lower_platform, upper_platform])
    generator.light_regions = [FloorLightRegion(
        region_id="floor_0_0_0",
        bounds=(0, 0, 256, 256),
        floor_z=0,
        # This coarse point fits a standing hull but has only 88u headroom.
        samples=((128.0, 128.0, 1.0),),
    )]

    generator._emit_interior_lighting()

    zone = next(
        item for item in generator.interior_light_zones
        if item.zone_id == "under_platform_0_0"
    )
    source = next(
        item for item in generator.interior_light_sources
        if item.region_id == zone.zone_id
    )
    assert zone.anchor == (160.0, 128.0, 1.0)
    assert generator._player_column_is_clear(
        zone.anchor[0], zone.anchor[1], zone.floor_z
    )
    assert source.origin == (160.0, 128.0, 96.0)


def test_horizontal_sandwich_threshold_uses_player_hull_and_safe_headroom():
    lower = HorizontalSurface(
        "lower", "platform", SolidBox(0, 0, 100, 128, 128, 116)
    )
    admitting = HorizontalSurface(
        "admitting", "roof", SolidBox(0, 0, 172, 128, 128, 188)
    )
    safe = HorizontalSurface(
        "safe", "roof", SolidBox(0, 0, 116 + MIN_SAFE_HEADROOM,
                                  128, 128, 228)
    )
    narrow = HorizontalSurface(
        "narrow", "roof", SolidBox(0, 0, 172, 47, 128, 188)
    )

    assert unsafe_horizontal_sandwiches([lower, admitting])
    assert not unsafe_horizontal_sandwiches([lower, safe])
    assert not unsafe_horizontal_sandwiches([lower, narrow])


def test_seeded_lighting_output_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    generate_map("same", 42, first, style="arena_vertical")
    generate_map("same", 42, second, style="arena_vertical")

    assert (first / "same.map").read_bytes() == (second / "same.map").read_bytes()
    assert ((first / "same.meta.json").read_bytes() ==
            (second / "same.meta.json").read_bytes())


def test_towers_seed_71425107_emits_direct_interior_lights(tmp_path):
    generate_map("towers_light_regression", 71425107, tmp_path, style="towers")

    result = static_validate(
        tmp_path / "towers_light_regression.map", _static_args()
    )

    assert result["interior_lighting_ok"] is True
    assert result["lighting_ok"] is True
    assert result["static_ok"] is True


def test_arena_vertical_seed_71426502_keeps_lava_out_of_building_zone(tmp_path):
    map_path, _ = generate_map(
        "lava_building_regression", 71426502, tmp_path,
        style="arena_vertical",
    )
    meta = json.loads(map_path.with_suffix(".meta.json").read_text())
    lattice = json.loads(map_path.with_suffix(".lattice.json").read_text())

    assert meta["large_buildings"] == 2
    building = next(
        zone for zone in meta["lighting"]["interior_zones"]
        if zone["id"] == "building_1"
    )
    x0, y0, x1, y1 = building["bounds"]
    protected = (
        x0, y0, building["floor_z"],
        x1, y1, max(building["ceiling_z"],
                    building["floor_z"] + MIN_SAFE_HEADROOM),
    )
    for danger in lattice["danger"]:
        assert (
            danger[3] <= protected[0] or danger[0] >= protected[3]
            or danger[4] <= protected[1] or danger[1] >= protected[4]
            or danger[5] <= protected[2] or danger[2] >= protected[5]
        )

    entities = _parse_entities(map_path.read_text())
    assert any(
        entity.get("_ml_interior_light") == "1"
        and entity.get("_ml_zone") == "building_1"
        for entity in entities
    )
    result = static_validate(map_path, _static_args())
    assert result["interior_lighting_ok"] is True
    assert result["static_ok"] is True


def test_arena_vertical_seed_71431503_rejects_low_ceiling_tower_sandwich(
    tmp_path,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    first_map, _ = generate_map(
        "objective_tower_regression", 71431503, first,
        style="arena_vertical",
    )
    second_map, _ = generate_map(
        "objective_tower_regression", 71431503, second,
        style="arena_vertical",
    )

    assert first_map.read_bytes() == second_map.read_bytes()
    assert (
        first_map.with_suffix(".meta.json").read_bytes()
        == second_map.with_suffix(".meta.json").read_bytes()
    )
    lattice = json.loads(first_map.with_suffix(".lattice.json").read_text())
    assert lattice["objectives"]
    assert all(
        objective["height"] >= TOWER_H_MIN + 32
        for objective in lattice["objectives"]
    )
    generator = MapGenerator(seed=71431503, style="arena_vertical")
    generator.generate()
    tower_surfaces = [
        surface for surface in generator.horizontal_surfaces
        if surface.kind == "tower"
    ]
    assert len(tower_surfaces) == generator.tower_count
    assert all(
        surface.box.z1 - surface.box.z0 >= TOWER_H_MIN
        for surface in tower_surfaces
    )
    assert unsafe_horizontal_sandwiches(generator.horizontal_surfaces) == []
    result = static_validate(first_map, _static_args())
    assert result["unsafe_horizontal_sandwiches"] == 0
    assert result["static_ok"] is True


def test_towers_seed_71432101_preserves_promised_corner_interiors(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    map_path, _ = generate_map(
        "corner_objective_regression", 71432101, first,
        style="towers",
    )
    cold_map, _ = generate_map(
        "corner_objective_regression", 71432101, second,
        style="towers",
    )
    assert map_path.read_bytes() == cold_map.read_bytes()
    assert (
        map_path.with_suffix(".meta.json").read_bytes()
        == cold_map.with_suffix(".meta.json").read_bytes()
    )
    metadata = json.loads(map_path.with_suffix(".meta.json").read_text())
    corner_zones = [
        zone for zone in metadata["lighting"]["interior_zones"]
        if zone["kind"] == "corner_pocket"
    ]

    assert metadata["corner_pockets"] == len(corner_zones)
    assert metadata["corner_pockets"] > 0
    result = static_validate(map_path, _static_args())
    assert result["interior_lighting_ok"] is True
    assert result["static_ok"] is True


def test_structure_clearance_protects_reserved_interior_volume():
    generator = MapGenerator(seed=1, style="open")
    generator._interior_zone_specs = [
        ("corner_0", "corner_pocket", (100, 100, 200, 200), 0, 256),
    ]

    assert not generator._structure_is_clear(
        SolidBox(120, 120, 0, 180, 180, 128)
    )
    assert generator._structure_is_clear(
        SolidBox(220, 120, 0, 280, 180, 128)
    )


def test_validator_rejects_map_missing_a_regions_lights(tmp_path):
    generate_map("shadowed", 42, tmp_path, style="arena_vertical")
    map_path = tmp_path / "shadowed.map"

    baseline = static_validate(map_path, _static_args())
    assert baseline["lighting_ok"] is True
    assert baseline["under_platform_lights"] > 0

    entities = _parse_entities(map_path.read_text())
    shadowed_region = next(
        ent["_ml_region"] for ent in entities
        if ent.get("classname") == "light"
        and ent.get("_ml_kind") == "under_platform"
    )
    map_path.write_text(
        _drop_region_lights(map_path.read_text(), shadowed_region)
    )

    invalid = static_validate(map_path, _static_args())
    assert invalid["lighting_ok"] is False
    assert invalid["static_ok"] is False


def test_validator_rejects_missing_dedicated_interior_light(tmp_path):
    generate_map("interior", 42, tmp_path, style="arena_vertical")
    map_path = tmp_path / "interior.map"
    baseline = static_validate(map_path, _static_args())
    assert baseline["interior_lighting_ok"] is True

    entities = _parse_entities(map_path.read_text())
    zone_id = next(
        ent["_ml_zone"] for ent in entities
        if ent.get("_ml_interior_light") == "1"
    )
    map_path.write_text(_drop_interior_light(map_path.read_text(), zone_id))

    invalid = static_validate(map_path, _static_args())
    assert invalid["interior_lighting_ok"] is False
    assert invalid["lighting_ok"] is False
    assert invalid["static_ok"] is False


def test_validator_rejects_player_admitting_stacked_overheads(tmp_path):
    generate_map("sandwich", 7, tmp_path, style="open")
    map_path = tmp_path / "sandwich.map"
    writer = MapWriter()
    lower = writer.make_box_brush(
        10000, 10000, 100, 10128, 10128, 116,
        tf=T_CEIL, tc=T_CEIL, tw=T_CEIL,
    )
    upper = writer.make_box_brush(
        10000, 10000, 172, 10128, 10128, 188,
        tf=T_CEIL, tc=T_CEIL, tw=T_CEIL,
    )
    map_path.write_text(_insert_world_brushes(
        map_path.read_text(), [lower, upper]
    ))

    invalid = static_validate(map_path, _static_args())
    assert invalid["unsafe_horizontal_sandwiches"] >= 1
    assert invalid["static_ok"] is False


def test_generated_spawns_have_safe_columns_and_escape_paths(tmp_path):
    generate_map("spawn_escape", 1, tmp_path, style="arena_lanes")
    result = static_validate(tmp_path / "spawn_escape.map", _static_args())

    assert result["blocked_spawn_columns"] == 0
    assert result["trapped_spawns"] == 0
    assert result["static_ok"] is True


def test_lethal_guards_reject_arena_vertical_low_headroom_overlap():
    generator = MapGenerator(seed=1, style="arena_vertical")
    generator.rooms = [
        # The high arena supplies the selected floor-union surface.
        Room(
            gx=1, gy=0, wx=512, wy=0, w=1536, d=1536,
            floor_z=96, ceil_z=538, kind="arena",
        ),
        # This lower room reproduces the overlap from arena_vertical seed
        # 91470501: its 129u ceiling crosses the high floor's standing column.
        Room(
            gx=1, gy=1, wx=512, wy=512, w=1024, d=1024,
            floor_z=0, ceil_z=129, kind="room",
        ),
    ]
    for room in generator.rooms:
        generator.spawn_blockers.extend([
            SolidBox(
                room.wx, room.wy, -16,
                room.wx + room.w, room.wy + room.d, room.floor_z,
            ),
            SolidBox(
                room.wx, room.wy, room.ceil_z - 8,
                room.wx + room.w, room.wy + room.d, room.ceil_z + 16,
            ),
        ])

    generator._plan_lethal_drop_guards()

    segments = {
        tuple(edge["segment"]) for edge in generator.lethal_edges
    }
    assert (512, 512, 512, 1536, 96) not in segments
    # The adjacent high-floor edge has full arena headroom and remains guarded.
    assert (512, 0, 512, 512, 96) in segments
    assert len(generator.lethal_edges) == len(generator.lethal_guard_walls)


def test_lethal_guard_keeps_edge_with_alternate_clear_witness():
    generator = MapGenerator(seed=1, style="arena_vertical")
    generator.rooms = [
        Room(
            gx=0, gy=0, wx=0, wy=0, w=512, d=512,
            floor_z=0, ceil_z=256, kind="arena",
        )
    ]
    generator.spawn_blockers.extend([
        SolidBox(0, 0, -16, 512, 512, 0),
        SolidBox(0, 0, 248, 512, 512, 272),
        # Obstruct only the west edge's first (midpoint) standing witness.
        SolidBox(16, 224, 32, 64, 288, 88),
    ])

    generator._plan_lethal_drop_guards()

    assert {
        edge["side"] for edge in generator.lethal_edges
    } == {"west", "east", "south", "north"}
    assert len(generator.lethal_edges) == 4


def test_validator_rejects_missing_lethal_drop_guard(tmp_path):
    generate_map("guarded", 42, tmp_path, style="arena_vertical")
    map_path = tmp_path / "guarded.map"
    baseline = static_validate(map_path, _static_args())
    assert baseline["lethal_drop_ok"] is True
    assert baseline["lethal_edges"] > 0
    assert baseline["missing_lethal_guards"] == 0

    metadata = json.loads((tmp_path / "guarded.meta.json").read_text())
    bounds = metadata["safety"]["guard_walls"][0]
    palette = PALETTES[metadata["palette"]]
    guard_brush = MapWriter().make_box_brush(
        *bounds,
        tf=palette["trim"], tc=palette["trim"], tw=palette["wall"],
    )
    source = map_path.read_text()
    assert guard_brush in source
    map_path.write_text(source.replace(guard_brush, "", 1))

    invalid = static_validate(map_path, _static_args())
    assert invalid["missing_lethal_guards"] == 1
    assert invalid["lethal_drop_ok"] is False
    assert invalid["static_ok"] is False


def test_compiled_bsp_requires_nonempty_qrad_lightdata(tmp_path):
    map_path = tmp_path / "compiled.map"
    map_path.write_text("")
    header = bytearray(8 + Q2_BSP_LUMPS * 8)
    struct.pack_into("<4sI", header, 0, b"IBSP", 38)
    struct.pack_into(
        "<II", header, 8 + Q2_BSP_LIGHTING_LUMP * 8, len(header), 4096
    )
    map_path.with_suffix(".bsp").write_bytes(header + bytes(4096))

    result = _compiled_lightdata_metrics(map_path)
    assert result["qrad_lightdata_ok"] is True
    assert result["lightdata_bytes"] == 4096
