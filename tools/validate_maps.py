#!/usr/bin/env python3
"""
Validate generated Q2 ML maps for basic 4-player training playability.

Checks static .map/.json structure, recomputes per-region direct floor-light
coverage, and can optionally smoke-load every map in q2ded through the training
environment.
"""

import argparse
import json
import math
import os
import random
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOCAL_Q2_ROOT = ROOT.parent / "q2_lithium_merge"
if LOCAL_Q2_ROOT.exists():
    os.environ.setdefault("Q2_ROOT", str(LOCAL_Q2_ROOT))
sys.path.insert(0, str(ROOT))

from maps.generator import (  # noqa: E402
    FloorLightRegion,
    HorizontalSurface,
    InteriorLightZone,
    MIN_INTERIOR_LIGHT_RADIUS,
    MIN_INTERIOR_LIGHT_VALUE,
    LightSource,
    MIN_FLOOR_LIGHT_COVERAGE,
    MIN_FLOOR_LIGHT_VALUE,
    MIN_SAFE_HEADROOM,
    MIN_SANDWICH_OVERLAP,
    PLAYER_DIAMETER,
    PLAYER_H,
    SPAWN_ESCAPE_DISTANCE,
    SPAWN_ESCAPE_STEP,
    SolidBox,
    WORLD_AMBIENT_LIGHT,
    floor_light_coverage,
    light_reaches_sample,
    unsafe_horizontal_sandwiches,
)


ENTITY_RE = re.compile(r'"([^"\\]+)"\s+"([^"\\]*)"')
ORIGIN_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)
POINT_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)"
)

PLAYER_XY_HALF = 16.0
PLAYER_MINS_Z = -24.0
PLAYER_MAXS_Z = 32.0
SPAWN_SOLID_MARGIN = 48.0
FLOOR_EPSILON = 1.0
MAX_MAP_LIGHTING = 0x200000
Q2_BSP_VERSION = 38
Q2_BSP_LUMPS = 19
Q2_BSP_LIGHTING_LUMP = 7


@dataclass(frozen=True)
class BrushGeometry:
    bounds: Tuple[float, float, float, float, float, float]
    textures: Tuple[str, ...]


def _parse_entities(text: str) -> List[Dict[str, str]]:
    ents: List[Dict[str, str]] = []
    depth = 0
    block: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "{":
            if depth == 0:
                block = []
            depth += 1
            continue
        if stripped == "}":
            depth -= 1
            if depth == 0 and block:
                ent: Dict[str, str] = {}
                for item in block:
                    match = ENTITY_RE.search(item)
                    if match:
                        ent[match.group(1)] = match.group(2)
                if ent.get("classname"):
                    ents.append(ent)
            continue
        if depth == 1 and '"' in line:
            block.append(line)
    return ents


def _origin(ent: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    match = ORIGIN_RE.match(ent.get("origin", ""))
    if not match:
        return None
    return tuple(float(match.group(i)) for i in range(1, 4))


def _parse_brush_geometry(text: str) -> List[BrushGeometry]:
    brushes: List[BrushGeometry] = []
    depth = 0
    brush_lines: Optional[List[str]] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "{":
            depth += 1
            if depth == 2:
                brush_lines = []
            continue
        if stripped == "}":
            if depth == 2 and brush_lines is not None:
                points = [
                    tuple(float(match.group(i)) for i in range(1, 4))
                    for item in brush_lines
                    for match in POINT_RE.finditer(item)
                ]
                if points:
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    zs = [point[2] for point in points]
                    textures = []
                    for item in brush_lines:
                        matches = list(POINT_RE.finditer(item))
                        if len(matches) < 3:
                            continue
                        tail = item[matches[2].end():].split()
                        if tail:
                            textures.append(tail[0].lower())
                    brushes.append(BrushGeometry(
                        bounds=(min(xs), min(ys), min(zs),
                                max(xs), max(ys), max(zs)),
                        textures=tuple(textures),
                    ))
                brush_lines = None
            depth -= 1
            continue
        if depth == 2 and brush_lines is not None:
            brush_lines.append(line)
    return brushes


def _parse_brush_aabbs(
    text: str,
) -> List[Tuple[float, float, float, float, float, float]]:
    return [brush.bounds for brush in _parse_brush_geometry(text)]


def _is_overhead_surface(brush: BrushGeometry) -> bool:
    """Classify emitted ceiling/light/platform/roof brushes by texture."""
    x0, y0, z0, x1, y1, z1 = brush.bounds
    if not (0.0 < z1 - z0 <= 32.0):
        return False
    if (x1 - x0 < MIN_SANDWICH_OVERLAP or
            y1 - y0 < MIN_SANDWICH_OVERLAP):
        return False
    textures = brush.textures
    has_ceiling = any("/ceil" in texture for texture in textures)
    has_light = any("/baselt" in texture for texture in textures)
    has_metal = any("/metal" in texture for texture in textures)
    has_trim = any("/flat" in texture for texture in textures)
    return has_ceiling or has_light or (has_metal and (has_trim or has_ceiling))


def _horizontal_sandwich_metrics(text: str) -> Dict[str, object]:
    brushes = _parse_brush_geometry(text)
    surfaces = []
    for index, brush in enumerate(brushes):
        if not _is_overhead_surface(brush):
            continue
        x0, y0, z0, x1, y1, z1 = brush.bounds
        # Generated platforms have a solid, same-footprint 8u trim assembly
        # touching their underside. Measure free space from that true bottom;
        # treating the metal slab alone would invent a traversable 8u void.
        if (any("/metal" in texture for texture in brush.textures) and
                any("/flat" in texture for texture in brush.textures)):
            for attached in brushes:
                ax0, ay0, az0, ax1, ay1, az1 = attached.bounds
                if (attached.textures and
                        all("/flat" in texture for texture in attached.textures) and
                        (ax0, ay0, ax1, ay1) == (x0, y0, x1, y1) and
                        abs(az1 - z0) <= FLOOR_EPSILON):
                    z0 = min(z0, az0)
        if any("/baselt" in texture for texture in brush.textures):
            for attached in brushes:
                ax0, ay0, az0, ax1, ay1, az1 = attached.bounds
                if (any("/ceil" in texture for texture in attached.textures) and
                        ax0 <= x0 and ay0 <= y0 and ax1 >= x1 and ay1 >= y1 and
                        abs(az0 - z1) <= FLOOR_EPSILON):
                    z1 = max(z1, az1)
        surfaces.append(HorizontalSurface(
            surface_id=f"brush_{index}",
            kind="parsed_overhead",
            box=SolidBox(x0, y0, z0, x1, y1, z1),
        ))
    unsafe = unsafe_horizontal_sandwiches(surfaces)
    first_unsafe = None
    if unsafe:
        first, second, gap = unsafe[0]
        first_unsafe = {
            "lower": first.surface_id,
            "upper": second.surface_id,
            "gap": gap,
        }
    return {
        "horizontal_surfaces": len(surfaces),
        "unsafe_horizontal_sandwiches": len(unsafe),
        "minimum_safe_headroom": MIN_SAFE_HEADROOM,
        "minimum_sandwich_overlap": MIN_SANDWICH_OVERLAP,
        "first_unsafe_sandwich": first_unsafe,
    }


def _spawn_intersects_brush(
    spawn: Tuple[float, float, float],
    brush: Tuple[float, float, float, float, float, float],
) -> bool:
    x, y, z = spawn
    x0, y0, z0, x1, y1, z1 = brush
    half = PLAYER_XY_HALF + SPAWN_SOLID_MARGIN
    px0, px1 = x - half, x + half
    py0, py1 = y - half, y + half
    pz0, pz1 = z + PLAYER_MINS_Z, z + PLAYER_MAXS_Z
    return not (
        x1 <= px0 or x0 >= px1 or
        y1 <= py0 or y0 >= py1 or
        z1 <= pz0 or z0 >= pz1
    )


def _spawn_has_floor(
    spawn: Tuple[float, float, float],
    brushes: List[Tuple[float, float, float, float, float, float]],
) -> bool:
    x, y, z = spawn
    bottom = z + PLAYER_MINS_Z
    for x0, y0, _z0, x1, y1, z1 in brushes:
        if abs(z1 - bottom) > FLOOR_EPSILON:
            continue
        if x - PLAYER_XY_HALF < x0 or x + PLAYER_XY_HALF > x1:
            continue
        if y - PLAYER_XY_HALF < y0 or y + PLAYER_XY_HALF > y1:
            continue
        return True
    return False


def _player_column_is_clear(
    x: float,
    y: float,
    floor_z: float,
    brushes: List[Tuple[float, float, float, float, float, float]],
) -> bool:
    px0, px1 = x - PLAYER_XY_HALF, x + PLAYER_XY_HALF
    py0, py1 = y - PLAYER_XY_HALF, y + PLAYER_XY_HALF
    bottom = floor_z + FLOOR_EPSILON
    top = floor_z + MIN_SAFE_HEADROOM
    return not any(
        x1 > px0 and x0 < px1 and
        y1 > py0 and y0 < py1 and
        z1 > bottom and z0 < top
        for x0, y0, z0, x1, y1, z1 in brushes
    )


def _position_has_floor(
    x: float,
    y: float,
    floor_z: float,
    brushes: List[Tuple[float, float, float, float, float, float]],
) -> bool:
    return any(
        abs(z1 - floor_z) <= FLOOR_EPSILON and
        x - PLAYER_XY_HALF >= x0 and x + PLAYER_XY_HALF <= x1 and
        y - PLAYER_XY_HALF >= y0 and y + PLAYER_XY_HALF <= y1
        for x0, y0, _z0, x1, y1, z1 in brushes
    )


def _spawn_has_horizontal_escape(
    spawn: Tuple[float, float, float],
    brushes: List[Tuple[float, float, float, float, float, float]],
) -> bool:
    x, y, z = spawn
    floor_z = z + PLAYER_MINS_Z
    if not _player_column_is_clear(x, y, floor_z, brushes):
        return False
    diagonal = math.sqrt(0.5)
    directions = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (diagonal, diagonal), (diagonal, -diagonal),
        (-diagonal, diagonal), (-diagonal, -diagonal),
    )
    for dx, dy in directions:
        clear = True
        for distance in range(
                SPAWN_ESCAPE_STEP,
                SPAWN_ESCAPE_DISTANCE + SPAWN_ESCAPE_STEP,
                SPAWN_ESCAPE_STEP):
            px = x + dx * distance
            py = y + dy * distance
            if (not _position_has_floor(px, py, floor_z, brushes) or
                    not _player_column_is_clear(px, py, floor_z, brushes)):
                clear = False
                break
        if clear:
            return True
    return False


def _hook_counts(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    zones = 0
    required = 0
    for line in path.read_text(errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        zones += 1
        try:
            if int(float(parts[7])) & 4:
                required += 1
        except ValueError:
            pass
    return zones, required


def _compiled_lightdata_metrics(map_path: Path) -> Dict[str, object]:
    """Read the Quake II BSP lighting lump as direct qrad evidence."""
    bsp_path = map_path.with_suffix(".bsp")
    if not bsp_path.exists():
        return {
            "compiled_bsp_present": False,
            "qrad_lightdata_ok": None,
            "lightdata_bytes": 0,
        }
    try:
        header_size = 8 + Q2_BSP_LUMPS * 8
        with bsp_path.open("rb") as handle:
            header = handle.read(header_size)
        if len(header) != header_size:
            raise ValueError("truncated BSP header")
        ident, version = struct.unpack_from("<4sI", header, 0)
        if ident != b"IBSP" or version != Q2_BSP_VERSION:
            raise ValueError(f"unsupported BSP {ident!r} v{version}")
        offset, lightdata_bytes = struct.unpack_from(
            "<II", header, 8 + Q2_BSP_LIGHTING_LUMP * 8
        )
        file_size = bsp_path.stat().st_size
        if lightdata_bytes and (
                offset < header_size or offset + lightdata_bytes > file_size):
            raise ValueError("lighting lump lies outside BSP file")
        qrad_ok = 0 < lightdata_bytes <= MAX_MAP_LIGHTING
        return {
            "compiled_bsp_present": True,
            "qrad_lightdata_ok": qrad_ok,
            "lightdata_bytes": lightdata_bytes,
        }
    except (OSError, struct.error, ValueError) as exc:
        return {
            "compiled_bsp_present": True,
            "qrad_lightdata_ok": False,
            "lightdata_bytes": 0,
            "qrad_lightdata_error": str(exc),
        }


def _floor_lighting_metrics(
    map_path: Path,
    ents: List[Dict[str, str]],
    worldspawn: Dict[str, str],
    required_coverage: float,
) -> Dict[str, object]:
    """Recompute floor coverage from map lights and generator metadata.

    The metadata supplies fixed spawn-clear samples and horizontal occluders;
    the source list always comes from the map itself.  Removing a light,
    lowering its intensity, or moving it behind a platform therefore fails
    validation instead of trusting the generator's recorded coverage value.
    """
    result: Dict[str, object] = {
        "light_regions": 0,
        "floor_lights": 0,
        "under_platform_lights": 0,
        "interior_light_zones": 0,
        "interior_lights": 0,
        "interior_lighting_ok": False,
        "min_light_coverage": 0.0,
        "lighting_ok": False,
    }
    meta_path = map_path.with_suffix(".meta.json")
    if not meta_path.exists():
        result["lighting_error"] = "missing .meta.json lighting contract"
        return result

    try:
        metadata = json.loads(meta_path.read_text())
        contract = metadata["lighting"]
        if int(contract.get("version", 0)) != 2:
            raise ValueError("unsupported lighting contract version")
        contract_min = float(contract["minimum_coverage"])
        min_light_value = max(
            MIN_FLOOR_LIGHT_VALUE,
            int(contract.get("minimum_light_value", MIN_FLOOR_LIGHT_VALUE)),
        )
        regions = [
            FloorLightRegion(
                region_id=str(item["id"]),
                bounds=tuple(int(value) for value in item["bounds"]),
                floor_z=int(item["floor_z"]),
                samples=tuple(
                    tuple(float(value) for value in sample)
                    for sample in item["samples"]
                ),
            )
            for item in contract["regions"]
        ]
        occluders = [
            SolidBox(*(int(value) for value in box))
            for box in contract["occluders"]
        ]
        interior_min_value = max(
            MIN_INTERIOR_LIGHT_VALUE,
            int(contract["minimum_interior_light_value"]),
        )
        interior_min_radius = max(
            MIN_INTERIOR_LIGHT_RADIUS,
            float(contract["minimum_interior_light_radius"]),
        )
        hull = contract["player_hull"]
        if (int(hull["width"]) != PLAYER_DIAMETER or
                int(hull["standing_height"]) != PLAYER_H or
                int(hull["minimum_safe_headroom"]) < MIN_SAFE_HEADROOM or
                int(hull["minimum_sandwich_overlap"]) < MIN_SANDWICH_OVERLAP or
                int(hull["spawn_escape_distance"]) < SPAWN_ESCAPE_DISTANCE):
            raise ValueError("weakened player hull/headroom contract")
        interior_zones = [
            InteriorLightZone(
                zone_id=str(item["id"]),
                kind=str(item["kind"]),
                bounds=tuple(int(value) for value in item["bounds"]),
                floor_z=int(item["floor_z"]),
                ceiling_z=int(item["ceiling_z"]),
                anchor=tuple(float(value) for value in item["anchor"]),
            )
            for item in contract["interior_zones"]
        ]
        expected_room_zones = int(metadata["enterable_room_zones"])
        expected_corner_zones = int(metadata["corner_pockets"])
        expected_building_zones = int(metadata["large_buildings"])
        expected_under_platform_zones = int(
            metadata["enterable_under_platforms"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["lighting_error"] = f"invalid lighting contract: {exc}"
        return result

    tagged = [
        ent for ent in ents
        if ent.get("classname") == "light"
        and ent.get("_ml_floor_light") == "1"
    ]
    sources: List[LightSource] = []
    invalid_sources = 0
    for ent in tagged:
        origin = _origin(ent)
        try:
            if origin is None:
                raise ValueError("missing origin")
            source = LightSource(
                region_id=ent["_ml_region"],
                kind=ent["_ml_kind"],
                origin=origin,
                radius=float(ent["_ml_radius"]),
                value=int(float(ent.get("light", "0"))),
            )
        except (KeyError, TypeError, ValueError):
            invalid_sources += 1
            continue
        if source.value >= min_light_value:
            sources.append(source)

    tagged_interior = [
        ent for ent in ents
        if ent.get("classname") == "light"
        and ent.get("_ml_interior_light") == "1"
    ]
    interior_sources: List[LightSource] = []
    invalid_interior_sources = 0
    for ent in tagged_interior:
        origin = _origin(ent)
        try:
            if origin is None:
                raise ValueError("missing origin")
            source = LightSource(
                region_id=ent["_ml_zone"],
                kind=ent["_ml_kind"],
                origin=origin,
                radius=float(ent["_ml_radius"]),
                value=int(float(ent.get("light", "0"))),
            )
        except (KeyError, TypeError, ValueError):
            invalid_interior_sources += 1
            continue
        if (source.value < interior_min_value or
                source.radius < interior_min_radius):
            invalid_interior_sources += 1
            continue
        interior_sources.append(source)

    region_ids = {region.region_id for region in regions}
    coverages = []
    for region in regions:
        region_sources = [
            source for source in sources
            if source.region_id == region.region_id
        ]
        coverages.append(floor_light_coverage(region, region_sources, occluders))

    try:
        expected_regions = int(worldspawn.get("_ml_light_regions", "-1"))
        expected_lights = int(worldspawn.get("_ml_floor_lights", "-1"))
        expected_interior_zones = int(
            worldspawn.get("_ml_interior_zones", "-1")
        )
        expected_interior_lights = int(
            worldspawn.get("_ml_interior_lights", "-1")
        )
        map_contract_min = float(
            worldspawn.get("_ml_min_light_coverage", "0")
        )
        map_min_value = int(float(
            worldspawn.get("_ml_min_floor_light_value", "0")
        ))
        map_interior_min_value = int(float(
            worldspawn.get("_ml_min_interior_light_value", "0")
        ))
        map_interior_min_radius = float(
            worldspawn.get("_ml_min_interior_light_radius", "0")
        )
        world_ambient = int(float(worldspawn.get("light", "0")))
    except ValueError as exc:
        result["lighting_error"] = f"invalid world lighting contract: {exc}"
        return result
    effective_min = max(required_coverage, contract_min)
    min_coverage = min(coverages) if coverages else 0.0
    source_region_ids = {source.region_id for source in sources}
    zone_ids = {zone.zone_id for zone in interior_zones}
    room_zone_kinds = {"arena", "room", "hallway"}
    zone_shapes_valid = all(
        zone.ceiling_z - zone.floor_z >= MIN_SAFE_HEADROOM and
        zone.bounds[2] - zone.bounds[0] >= PLAYER_DIAMETER and
        zone.bounds[3] - zone.bounds[1] >= PLAYER_DIAMETER and
        zone.bounds[0] <= zone.anchor[0] <= zone.bounds[2] and
        zone.bounds[1] <= zone.anchor[1] <= zone.bounds[3] and
        abs(zone.anchor[2] - (zone.floor_z + 1)) <= FLOOR_EPSILON
        for zone in interior_zones
    )
    zone_population_valid = (
        sum(zone.kind in room_zone_kinds for zone in interior_zones) ==
        expected_room_zones and
        sum(zone.kind == "corner_pocket" for zone in interior_zones) ==
        expected_corner_zones and
        sum(zone.kind == "building" for zone in interior_zones) ==
        expected_building_zones and
        sum(zone.kind == "under_platform" for zone in interior_zones) ==
        expected_under_platform_zones
    )
    interior_source_ids = {source.region_id for source in interior_sources}
    zone_source_counts = {
        zone_id: sum(source.region_id == zone_id
                     for source in interior_sources)
        for zone_id in zone_ids
    }
    zones_by_id = {zone.zone_id: zone for zone in interior_zones}
    interior_sources_inside = True
    interior_sources_direct = True
    for source in interior_sources:
        zone = zones_by_id.get(source.region_id)
        if zone is None:
            interior_sources_inside = False
            interior_sources_direct = False
            continue
        x0, y0, x1, y1 = zone.bounds
        sx, sy, sz = source.origin
        if not (x0 <= sx <= x1 and y0 <= sy <= y1 and
                zone.floor_z + PLAYER_H <= sz <= zone.ceiling_z - 8 and
                source.kind == zone.kind):
            interior_sources_inside = False
        if not light_reaches_sample(source, zone.anchor, occluders):
            interior_sources_direct = False
    interior_lighting_ok = (
        bool(interior_zones) and
        len(zone_ids) == len(interior_zones) and
        zone_shapes_valid and
        zone_population_valid and
        expected_interior_zones == len(interior_zones) and
        expected_interior_lights == len(tagged_interior) and
        invalid_interior_sources == 0 and
        len(interior_sources) == len(tagged_interior) and
        interior_source_ids == zone_ids and
        all(count >= 1 for count in zone_source_counts.values()) and
        interior_sources_inside and
        interior_sources_direct and
        map_interior_min_value >= MIN_INTERIOR_LIGHT_VALUE and
        map_interior_min_radius >= MIN_INTERIOR_LIGHT_RADIUS
    )
    lighting_ok = (
        worldspawn.get("_ml_lighting_version") == "2" and
        bool(regions) and
        len(region_ids) == len(regions) and
        expected_regions == len(regions) and
        expected_lights == len(tagged) and
        invalid_sources == 0 and
        len(sources) == len(tagged) and
        source_region_ids <= region_ids and
        map_contract_min >= required_coverage and
        map_min_value >= MIN_FLOOR_LIGHT_VALUE and
        min_coverage + 1e-9 >= effective_min and
        world_ambient >= WORLD_AMBIENT_LIGHT and
        interior_lighting_ok
    )
    result.update({
        "light_regions": len(regions),
        "floor_lights": len(tagged),
        "under_platform_lights": sum(
            ent.get("_ml_kind") == "under_platform" for ent in tagged
        ),
        "interior_light_zones": len(interior_zones),
        "interior_lights": len(tagged_interior),
        "interior_lighting_ok": interior_lighting_ok,
        "world_ambient_light": world_ambient,
        "min_light_coverage": round(min_coverage, 4),
        "lighting_ok": lighting_ok,
    })
    if not lighting_ok:
        result["lighting_error"] = (
            f"minimum direct coverage {min_coverage:.3f}; "
            f"required {effective_min:.3f}; valid lights "
            f"{len(sources)}/{len(tagged)}; interior "
            f"{len(interior_sources)}/{len(interior_zones)}; "
            f"ambient {world_ambient}"
        )
    return result


def static_validate(map_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    text = map_path.read_text(errors="ignore")
    ents = _parse_entities(text)
    brushes = _parse_brush_aabbs(text)
    sandwiches = _horizontal_sandwich_metrics(text)
    compiled_lighting = _compiled_lightdata_metrics(map_path)
    spawns = [
        _origin(ent)
        for ent in ents
        if ent.get("classname") == "info_player_deathmatch"
    ]
    spawns = [spawn for spawn in spawns if spawn is not None]
    weapons = [
        ent for ent in ents
        if ent.get("classname", "").startswith("weapon_")
    ]
    pickups = [
        ent for ent in ents
        if ent.get("classname", "").startswith(("item_", "ammo_"))
    ]
    worldspawn = next((ent for ent in ents if ent.get("classname") == "worldspawn"), {})
    sky_name = worldspawn.get("sky", "")
    sky_ok = sky_name.lower() != "e1u1/skip"
    hook_zones, required_hooks = _hook_counts(map_path.with_suffix(".json"))
    lighting = _floor_lighting_metrics(
        map_path,
        ents,
        worldspawn,
        getattr(args, "min_light_coverage", MIN_FLOOR_LIGHT_COVERAGE),
    )
    if (compiled_lighting["compiled_bsp_present"] and
            not compiled_lighting["qrad_lightdata_ok"]):
        lighting["lighting_ok"] = False
        lighting["lighting_error"] = (
            f"{lighting.get('lighting_error', '')}; compiled BSP has no valid "
            "qrad lighting lump"
        ).strip("; ")

    min_spawn_dist = 0.0
    if len(spawns) > 1:
        min_spawn_dist = min(
            math.dist(a, b)
            for idx, a in enumerate(spawns)
            for b in spawns[idx + 1:]
        )

    xs = [spawn[0] for spawn in spawns] or [0.0]
    ys = [spawn[1] for spawn in spawns] or [0.0]
    xspan = max(xs) - min(xs)
    yspan = max(ys) - min(ys)
    spawn_bbox_area = xspan * yspan
    blocked_spawns = sum(
        1
        for spawn in spawns
        if any(_spawn_intersects_brush(spawn, brush) for brush in brushes)
    )
    unsupported_spawns = sum(
        1 for spawn in spawns if not _spawn_has_floor(spawn, brushes)
    )
    blocked_spawn_columns = sum(
        1 for x, y, z in spawns
        if not _player_column_is_clear(x, y, z + PLAYER_MINS_Z, brushes)
    )
    trapped_spawns = sum(
        1 for spawn in spawns
        if not _spawn_has_horizontal_escape(spawn, brushes)
    )

    static_ok = (
        len(spawns) >= args.min_spawns and
        min_spawn_dist >= args.min_spawn_distance and
        xspan >= args.min_span and
        yspan >= args.min_span and
        spawn_bbox_area >= args.min_spawn_area and
        blocked_spawns == 0 and
        unsupported_spawns == 0 and
        blocked_spawn_columns == 0 and
        trapped_spawns == 0 and
        sandwiches["unsafe_horizontal_sandwiches"] == 0 and
        sky_ok and
        len(weapons) >= args.min_weapons and
        len(pickups) >= args.min_pickups and
        hook_zones >= args.min_hook_zones and
        lighting["lighting_ok"]
    )

    return {
        "map": map_path.stem,
        "spawns": len(spawns),
        "min_spawn_dist": round(min_spawn_dist, 1),
        "xspan": round(xspan, 1),
        "yspan": round(yspan, 1),
        "spawn_bbox_area": round(spawn_bbox_area),
        "blocked_spawns": blocked_spawns,
        "unsupported_spawns": unsupported_spawns,
        "blocked_spawn_columns": blocked_spawn_columns,
        "trapped_spawns": trapped_spawns,
        "sky": sky_name,
        "sky_ok": sky_ok,
        "weapons": len(weapons),
        "pickups": len(pickups),
        "hook_zones": hook_zones,
        "required_hooks": required_hooks,
        **sandwiches,
        **compiled_lighting,
        **lighting,
        "static_ok": static_ok,
    }


def _action_script(step: int, rng: random.Random) -> np.ndarray:
    """Deterministic movement/firing sweep for playability validation."""
    action = np.zeros(8, dtype=np.float32)
    action[5] = 1.0
    action[7] = float(1 + (step // 60) % 5)

    if step < 48:
        action[2] = 30.0
        return action

    action[0] = rng.uniform(-1.0, 1.0)
    action[1] = rng.uniform(-0.85, 0.85)
    action[2] = rng.uniform(-35.0, 35.0)
    action[3] = rng.uniform(-8.0, 8.0)
    action[4] = 1.0 if rng.random() < 0.12 else 0.0
    action[6] = 0.0
    action[7] = float(rng.randint(1, 6))
    return action


def runtime_validate(map_name: str, steps: int, n_bots: int, args: argparse.Namespace) -> Dict[str, object]:
    from harness.env import Q2MultiEnv

    env = Q2MultiEnv(
        server_id=0,
        map_name=map_name,
        map_pool=[map_name],
        n_bots=n_bots,
        port_offset=args.runtime_port_offset,
        maxclients=args.runtime_maxclients,
        ml_slot=args.runtime_ml_slot,
        num_ml_bots=args.runtime_num_ml_bots,
        max_ep_steps=max(steps + 5, 10),
    )
    try:
        obs = env.reset_all()
        rewards = []
        infos = []
        for _ in range(steps):
            results = env.step_all([
                np.zeros(8, dtype=np.float32)
                for _slot in range(env.n_ml)
            ])
            for result in results:
                rewards.append(float(result[1]))
                infos.append(result[4])
        return {
            "runtime_ok": True,
            "obs_shape": list(obs[0].shape),
            "reward_sum": round(sum(rewards), 4),
            "last_info": infos[-1] if infos else {},
        }
    except Exception as exc:
        return {
            "runtime_ok": False,
            "runtime_error": str(exc),
        }
    finally:
        env.close()


def combat_validate(map_name: str, steps: int, n_bots: int, args: argparse.Namespace) -> Dict[str, object]:
    from harness.env import Q2MultiEnv

    stats = {
        "combat_steps": 0,
        "combat_attempts": 0,
        "visible_enemy_steps": 0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "kills": 0.0,
        "deaths": 0.0,
        "items": 0.0,
        "spatial_bonus": 0.0,
        "max_voxel_visited": 0.0,
        "timeouts": 0,
    }
    last_info: Dict[str, object] = {}
    try:
        for attempt in range(args.combat_attempts):
            rng = random.Random(args.combat_seed + attempt)
            env = Q2MultiEnv(
                server_id=0,
                map_name=map_name,
                map_pool=[map_name],
                n_bots=n_bots,
                port_offset=args.runtime_port_offset,
                maxclients=args.runtime_maxclients,
                ml_slot=args.runtime_ml_slot,
                num_ml_bots=args.runtime_num_ml_bots,
                max_ep_steps=max(steps + 5, 10),
            )
            try:
                env.reset_all()
                stats["combat_attempts"] += 1
                for step in range(steps):
                    actions = [
                        _action_script(step + slot * 19, rng)
                        for slot in range(env.n_ml)
                    ]
                    results = env.step_all(actions)
                    stop_attempt = False
                    for result in results:
                        info = result[4]
                        last_info = info
                        visible = info.get("enemy_visible_any", info.get("visible_enemy", 0.0))
                        stats["combat_steps"] += 1
                        stats["visible_enemy_steps"] += int(float(visible) > 0.0)
                        stats["damage_dealt"] += float(info.get("damage_dealt", 0.0))
                        stats["damage_taken"] += float(info.get("damage_taken", 0.0))
                        stats["kills"] += float(info.get("kills", 0.0))
                        stats["deaths"] += float(info.get("deaths", 0.0))
                        stats["items"] += float(info.get("items", 0.0))
                        stats["spatial_bonus"] += float(info.get("spatial_bonus", 0.0))
                        stats["max_voxel_visited"] = max(
                            stats["max_voxel_visited"],
                            float(info.get("voxel_visited", 0.0)),
                        )
                        stats["timeouts"] += int(bool(info.get("timeout")))
                        stop_attempt = stop_attempt or bool(result[2] or result[3] or info.get("timeout"))
                    if stop_attempt:
                        break
            finally:
                env.close()
        damage_ok = (
            stats["damage_taken"] >= args.min_damage_taken or
            stats["damage_dealt"] >= args.min_damage_dealt or
            stats["kills"] > 0.0 or
            stats["deaths"] > 0.0
        )
        combat_ok = (
            stats["timeouts"] == 0 and
            stats["visible_enemy_steps"] >= args.min_visible_steps and
            stats["max_voxel_visited"] >= args.min_visited_cells and
            (damage_ok or not args.require_damage)
        )
        return {
            "combat_ok": bool(combat_ok),
            **{key: round(value, 4) if isinstance(value, float) else value
               for key, value in stats.items()},
            "combat_last_info": last_info,
        }
    except Exception as exc:
        return {
            "combat_ok": False,
            "combat_error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        default=str(ROOT / "maps" / "generated"),
        help="directory containing generated .map/.json files",
    )
    parser.add_argument("--glob", default="mlmap_*.map")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--runtime-steps", type=int, default=3)
    parser.add_argument("--runtime-port-offset", type=int, default=30)
    parser.add_argument("--runtime-maxclients", type=int, default=12)
    parser.add_argument("--runtime-ml-slot", type=int, default=11)
    parser.add_argument("--runtime-num-ml-bots", type=int, default=1)
    parser.add_argument("--combat", action="store_true")
    parser.add_argument("--combat-steps", type=int, default=360)
    parser.add_argument("--combat-attempts", type=int, default=1)
    parser.add_argument("--combat-seed", type=int, default=1337)
    parser.add_argument("--min-visible-steps", type=int, default=12)
    parser.add_argument("--min-visited-cells", type=float, default=3.0)
    parser.add_argument("--min-damage-taken", type=float, default=1.0)
    parser.add_argument("--min-damage-dealt", type=float, default=1.0)
    parser.add_argument("--require-damage", action="store_true")
    parser.add_argument("--n-bots", type=int, default=4)
    parser.add_argument("--min-spawns", type=int, default=4)
    parser.add_argument("--min-spawn-distance", type=float, default=384.0)
    parser.add_argument("--min-span", type=float, default=1024.0)
    parser.add_argument("--min-spawn-area", type=float, default=1_000_000.0)
    parser.add_argument("--min-weapons", type=int, default=4)
    parser.add_argument("--min-pickups", type=int, default=8)
    parser.add_argument("--min-hook-zones", type=int, default=6)
    parser.add_argument(
        "--min-light-coverage",
        type=float,
        default=MIN_FLOOR_LIGHT_COVERAGE,
        help="minimum direct-light sample fraction in every spawn-clear floor region",
    )
    args = parser.parse_args()

    map_paths = sorted(Path(args.generated_dir).glob(args.glob))
    if not map_paths:
        print(f"No maps matched {args.glob!r} in {args.generated_dir}", file=sys.stderr)
        return 2

    rows = []
    for path in map_paths:
        row = static_validate(path, args)
        if args.runtime and row["static_ok"]:
            row.update(runtime_validate(row["map"], args.runtime_steps, args.n_bots, args))
        if args.combat and row["static_ok"]:
            row.update(combat_validate(row["map"], args.combat_steps, args.n_bots, args))
        rows.append(row)

    print(json.dumps(rows, indent=2))
    static_ok = sum(1 for row in rows if row.get("static_ok"))
    runtime_rows = [row for row in rows if "runtime_ok" in row]
    runtime_ok = sum(1 for row in runtime_rows if row.get("runtime_ok"))
    combat_rows = [row for row in rows if "combat_ok" in row]
    combat_ok = sum(1 for row in combat_rows if row.get("combat_ok"))
    print(f"SUMMARY static_ok={static_ok}/{len(rows)}")
    if runtime_rows:
        print(f"SUMMARY runtime_ok={runtime_ok}/{len(runtime_rows)}")
    if combat_rows:
        print(f"SUMMARY combat_ok={combat_ok}/{len(combat_rows)}")

    if static_ok != len(rows):
        return 1
    if runtime_rows and runtime_ok != len(runtime_rows):
        return 1
    if combat_rows and combat_ok != len(combat_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
