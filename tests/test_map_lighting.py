from argparse import Namespace

from maps.generator import (
    MIN_FLOOR_LIGHT_COVERAGE,
    MapGenerator,
    Room,
    SolidBox,
    floor_light_coverage,
    generate_map,
)
from tools.validate_maps import _parse_entities, static_validate


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
