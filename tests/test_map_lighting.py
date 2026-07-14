from argparse import Namespace
import json
import struct

from maps.generator import (
    HorizontalSurface,
    MIN_FLOOR_LIGHT_COVERAGE,
    MIN_SAFE_HEADROOM,
    MapWriter,
    MapGenerator,
    PALETTES,
    Room,
    SolidBox,
    T_CEIL,
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
