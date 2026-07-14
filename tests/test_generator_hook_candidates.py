from __future__ import annotations

import json
import math

from maps.generator import (
    FloorLightRegion,
    HOOK_CEILING,
    HOOK_EYE_Z,
    MAX_HOOK_CANDIDATES_V2,
    PLAYER_MINS_Z,
    HorizontalSurface,
    MapGenerator,
    Room,
    SolidBox,
    generate_map,
)


def _minimal_generator() -> MapGenerator:
    generator = MapGenerator(seed=1, style="open")
    room = Room(
        gx=0, gy=0, wx=0, wy=0, w=512, d=512,
        floor_z=0, ceil_z=256, kind="room",
    )
    generator.rooms = [room]
    generator.horizontal_surfaces = [
        HorizontalSurface(
            "room_ceiling_0", "ceiling", SolidBox(0, 0, 256, 512, 512, 272)
        )
    ]
    generator.light_regions = [
        FloorLightRegion(
            "floor_0_0_0", (0, 0, 512, 512), 0,
            ((64.0, 64.0, 1.0), (192.0, 64.0, 1.0)),
        )
    ]
    generator.spawn_points = [(64, 64, 24), (192, 64, 24)]
    return generator


def _l1(point: list[int]) -> tuple[int, int, int]:
    return tuple(math.floor(axis / 16_000) for axis in point)


def test_hook_candidates_use_real_ceiling_and_standing_origin() -> None:
    generator = _minimal_generator()

    generator._annotate_hook_zones()

    assert generator.hook_zones
    assert all(zone.anchor[2] == 256.0 for zone in generator.hook_zones)
    assert all(zone.anchor[2] != 256.0 - 16.0 for zone in generator.hook_zones)
    assert all(zone.landing[2] == -PLAYER_MINS_Z for zone in generator.hook_zones)
    assert all(zone.flags == HOOK_CEILING for zone in generator.hook_zones)


def test_platform_underface_never_becomes_a_top_landing_claim() -> None:
    generator = _minimal_generator()
    generator.horizontal_surfaces.append(HorizontalSurface(
        "platform_0_0", "platform", SolidBox(32, 32, 128, 96, 96, 152)
    ))

    generator._annotate_hook_zones()

    assert all(zone.landing[:2] != (64.0, 64.0) for zone in generator.hook_zones)
    assert all(zone.anchor[2] != 128.0 for zone in generator.hook_zones)
    assert all(zone.landing[2] < zone.anchor[2] for zone in generator.hook_zones)


def test_missing_distinct_source_emits_no_fabricated_candidate() -> None:
    generator = _minimal_generator()
    generator.light_regions[0] = FloorLightRegion(
        "floor_0_0_0", (0, 0, 512, 512), 0, ((64.0, 64.0, 1.0),)
    )
    generator.spawn_points = [(64, 64, 24)]

    generator._annotate_hook_zones()

    assert generator.hook_zones == []
    assert generator.hook_claim_candidates_v2 == []
    assert generator.hook_claim_candidates_v2_manifest()["records"] == []


def test_generated_hook_candidate_schema_order_and_distance_are_stable(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    generate_map("same", 7142026, first, style="arena_vertical")
    generate_map("same", 7142026, second, style="arena_vertical")

    first_meta = json.loads((first / "same.meta.json").read_text())
    second_meta = json.loads((second / "same.meta.json").read_text())
    first_pool = first_meta["hook_claim_candidates_v2"]
    second_pool = second_meta["hook_claim_candidates_v2"]

    assert first_pool == second_pool
    assert set(first_pool) == {
        "schema", "tick_msec", "status", "bundle_admissible", "records",
    }
    assert first_pool["schema"] == "q2-hook-claim-candidates-v2"
    assert first_pool["tick_msec"] == 100
    assert first_pool["status"] == "unproven"
    assert first_pool["bundle_admissible"] is False
    records = first_pool["records"]
    assert 6 <= len(records) <= MAX_HOOK_CANDIDATES_V2
    assert [record["claim_id"] for record in records] == sorted(
        record["claim_id"] for record in records
    )
    assert len({record["claim_id"] for record in records}) == len(records)

    expected_record_keys = {
        "claim_id", "source_milliunits", "anchor_milliunits",
        "landing_milliunits", "release_after_ticks", "distance_milliunits",
        "flags",
    }
    for record in records:
        assert set(record) == expected_record_keys
        assert _l1(record["source_milliunits"]) != _l1(record["landing_milliunits"])
        eye = list(record["source_milliunits"])
        eye[2] += HOOK_EYE_Z * 1000
        expected_distance = round(math.dist(eye, record["anchor_milliunits"]))
        assert record["distance_milliunits"] == expected_distance
        assert record["landing_milliunits"][2] in {
            (region["floor_z"] - PLAYER_MINS_Z) * 1000
            for region in first_meta["lighting"]["regions"]
        }
        assert record["flags"] == HOOK_CEILING


def test_runtime_sidecar_is_unique_eight_field_unproven_projection(tmp_path) -> None:
    generate_map("projection", 42, tmp_path, style="arena_lanes")
    metadata = json.loads((tmp_path / "projection.meta.json").read_text())
    lines = (tmp_path / "projection.json").read_text().splitlines()
    records = [line.split() for line in lines if line and not line.startswith("#")]

    assert "bundle_admissible: false" in "\n".join(lines[:4])
    assert records
    assert all(len(record) == 8 for record in records)
    geometries = [tuple(record[:6] + record[7:8]) for record in records]
    assert len(geometries) == len(set(geometries))
    assert len(records) == metadata["hook_zones"]
    assert len(metadata["hook_claim_candidates_v2"]["records"]) > len(records)


def test_seeded_styles_publish_only_real_ceiling_geometry() -> None:
    for style in ("open", "towers", "arena_vertical"):
        generator = MapGenerator(seed=7, style=style)
        generator.generate()
        allowed = {
            (float(surface.box.z0), surface.kind)
            for surface in generator.horizontal_surfaces
            if surface.kind in {"ceiling", "light_panel"}
        }

        assert len(generator.hook_zones) >= 6
        assert all((zone.anchor[2], "ceiling") in allowed
                   or (zone.anchor[2], "light_panel") in allowed
                   for zone in generator.hook_zones)
        assert all(zone.flags == HOOK_CEILING for zone in generator.hook_zones)
