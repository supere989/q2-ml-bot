"""Fail-closed compiled-world challenges for generated-v6 claims.

Generator records are proposals.  This module validates their frozen wire
contract and source identities, then admits only facts reproduced from the
compiled BSP through B1 metadata/collision authority and evidenced Atlas
navigation. Hook-v2 claims are independently replayed by the Atlas analyzer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any

from .hook_claims_v2 import (
    HookClaimsV2Error,
    load_materialization,
    validate_record as validate_hook_record_v2,
)


CLAIMS_SCHEMA = "q2-generator-claims-v2"
ORACLE_STATUS = "oracle"
VALIDATION_VERSION = 1
EVIDENCE_CM_CONTENTS_V1 = 3
EVIDENCE_ATLAS_ROUTE_V1 = 5
EVIDENCE_LINKED_AABB_V1 = 6
MASK_PLAYERSOLID = 33_619_971
MASK_SHOT = 100_663_299
CONTENTS_LAVA = 8
STANDING_MINS = [-16, -16, -24]
STANDING_MAXS = [16, 16, 32]
INFINITE_COST_Q8 = 0xFFFFFFFF
MIN_FLOOR_LIGHT_VALUE = 650.0
MIN_INTERIOR_LIGHT_VALUE = 800.0
MIN_INTERIOR_LIGHT_RADIUS = 320.0

_MAP_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_CLAIM_RE = re.compile(r"^[a-z0-9:_-]{1,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class GeneratedClaimProbeError(RuntimeError):
    """A proposal is malformed, identity-mismatched, or not oracle-proven."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise GeneratedClaimProbeError(
            f"{label} keys differ: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _list(value: Any, label: str, *, minimum: int = 0) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise GeneratedClaimProbeError(f"{label} must have at least {minimum} records")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeneratedClaimProbeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise GeneratedClaimProbeError(f"{label} must be at least {minimum}")
    return value


def _vec3(value: Any, label: str) -> tuple[int, int, int]:
    items = _list(value, label)
    if len(items) != 3:
        raise GeneratedClaimProbeError(f"{label} must contain three integers")
    return tuple(_integer(item, f"{label}[{index}]") for index, item in enumerate(items))  # type: ignore[return-value]


def _claim_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _CLAIM_RE.fullmatch(value):
        raise GeneratedClaimProbeError(f"{label} is not a canonical claim ID")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise GeneratedClaimProbeError(f"{label} is not a lowercase SHA-256")
    return value


def validate_generator_claims(value: Any) -> dict[str, Any]:
    """Validate the exact frozen q2-generator-claims-v2 structure."""
    claims = _exact_mapping(value, {
        "schema", "map", "generator", "source_files", "spawns",
        "hazard_claims", "hook_claims", "route_claims", "routes",
    }, "generator claims")
    if claims["schema"] != CLAIMS_SCHEMA or claims["generator"] != "v6":
        raise GeneratedClaimProbeError("claims schema/generator is not frozen v6")
    if not isinstance(claims["map"], str) or not _MAP_RE.fullmatch(claims["map"]):
        raise GeneratedClaimProbeError("claims map is not canonical")

    sources = _exact_mapping(claims["source_files"], {
        "map_sha256", "meta_sha256", "lattice_sha256",
        "hook_zones_sha256", "hook_materialization_sha256", "routes_sha256",
    }, "claims source_files")
    for name, digest in sources.items():
        _sha(digest, f"source_files.{name}")

    all_ids: set[str] = set()

    def unique(identifier: str, label: str) -> str:
        claim = _claim_id(identifier, label)
        if claim in all_ids:
            raise GeneratedClaimProbeError(f"duplicate claim ID {claim}")
        all_ids.add(claim)
        return claim

    spawn_origins: set[tuple[int, int, int]] = set()
    spawn_items = _list(claims["spawns"], "spawns", minimum=8)
    if len(spawn_items) != 8:
        raise GeneratedClaimProbeError("generator claims require exactly eight spawns")
    for index, item in enumerate(spawn_items):
        record = _exact_mapping(item, {"claim_id", "origin_milliunits"}, f"spawn {index}")
        unique(record["claim_id"], f"spawn {index} claim_id")
        origin = _vec3(record["origin_milliunits"], f"spawn {index} origin")
        if origin in spawn_origins:
            raise GeneratedClaimProbeError("duplicate spawn origin")
        spawn_origins.add(origin)

    for index, item in enumerate(_list(claims["hazard_claims"], "hazards", minimum=1)):
        record = _exact_mapping(
            item, {"claim_id", "type", "bounds_milliunits"}, f"hazard {index}"
        )
        unique(record["claim_id"], f"hazard {index} claim_id")
        if record["type"] not in {"lava", "hurt"}:
            raise GeneratedClaimProbeError(f"hazard {index} has unknown type")
        bounds = _list(record["bounds_milliunits"], f"hazard {index} bounds")
        if len(bounds) != 6:
            raise GeneratedClaimProbeError(f"hazard {index} bounds must contain six integers")
        numbers = [_integer(number, f"hazard {index} bounds") for number in bounds]
        if any(numbers[axis] >= numbers[axis + 3] for axis in range(3)):
            raise GeneratedClaimProbeError(f"hazard {index} bounds are inverted")

    for index, item in enumerate(_list(claims["hook_claims"], "hooks", minimum=6)):
        try:
            record = validate_hook_record_v2(item, f"hook {index}")
        except HookClaimsV2Error as error:
            raise GeneratedClaimProbeError(str(error)) from error
        unique(record["claim_id"], f"hook {index} claim_id")
    if len(claims["hook_claims"]) != 6:
        raise GeneratedClaimProbeError("generator claims require exactly six hooks")

    route_claims: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(_list(claims["route_claims"], "route claims", minimum=1)):
        record = _exact_mapping(item, {
            "claim_id", "route_id", "source_milliunits", "target_milliunits",
        }, f"route claim {index}")
        claim = unique(record["claim_id"], f"route claim {index} claim_id")
        _claim_id(record["route_id"], f"route claim {index} route_id")
        _vec3(record["source_milliunits"], f"route claim {index} source")
        _vec3(record["target_milliunits"], f"route claim {index} target")
        route_claims[claim] = record

    seen_routes: set[str] = set()
    used_segments: set[str] = set()
    for index, item in enumerate(_list(claims["routes"], "routes", minimum=1)):
        record = _exact_mapping(item, {
            "route_id", "archetype", "claimed_cost_q8", "segment_claim_ids",
        }, f"route {index}")
        route_id = _claim_id(record["route_id"], f"route {index} route_id")
        if route_id in seen_routes:
            raise GeneratedClaimProbeError(f"duplicate route {route_id}")
        seen_routes.add(route_id)
        if record["archetype"] not in {"offense", "survival", "control", "balanced"}:
            raise GeneratedClaimProbeError(f"route {route_id} has unknown archetype")
        cost = _integer(record["claimed_cost_q8"], f"route {route_id} cost", minimum=1)
        if cost >= INFINITE_COST_Q8:
            raise GeneratedClaimProbeError(f"route {route_id} cost is infinite")
        segments = _list(record["segment_claim_ids"], f"route {route_id} segments", minimum=2)
        for segment in segments:
            segment_id = _claim_id(segment, f"route {route_id} segment")
            if segment_id not in route_claims:
                raise GeneratedClaimProbeError(f"route {route_id} references unknown {segment_id}")
            if route_claims[segment_id]["route_id"] != route_id:
                raise GeneratedClaimProbeError(f"{segment_id} belongs to a different route")
            if segment_id in used_segments:
                raise GeneratedClaimProbeError(f"route segment {segment_id} is reused")
            used_segments.add(segment_id)
    if used_segments != set(route_claims):
        raise GeneratedClaimProbeError("not every route claim belongs to one route")
    return claims


def generator_claims_sha256(claims: Mapping[str, Any]) -> str:
    validate_generator_claims(claims)
    return _sha256_bytes(_canonical_json(claims) + b"\n")


def load_generator_claims(path: Path, expected_map: str | None = None) -> tuple[dict[str, Any], str]:
    """Load canonical claims and verify all five sibling source identities."""
    path = path.resolve()
    suffix = ".generator-claims.json"
    if not path.name.endswith(suffix):
        raise GeneratedClaimProbeError("claims path must end in .generator-claims.json")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeneratedClaimProbeError("claims file is not valid JSON") from error
    claims = validate_generator_claims(value)
    canonical = _canonical_json(claims) + b"\n"
    if raw != canonical:
        raise GeneratedClaimProbeError("claims file is not canonical sorted JSON plus LF")
    if expected_map is not None and claims["map"] != expected_map:
        raise GeneratedClaimProbeError("claims map differs from requested map")
    stem_name = path.name[:-len(suffix)]
    if claims["map"] != stem_name:
        raise GeneratedClaimProbeError("claims map differs from claims filename")
    stem = path.parent / stem_name
    source_paths = {
        "map_sha256": stem.with_suffix(".map"),
        "meta_sha256": stem.with_suffix(".meta.json"),
        "lattice_sha256": Path(f"{stem}.lattice.json"),
        "hook_zones_sha256": stem.with_suffix(".json"),
        "hook_materialization_sha256": Path(f"{stem}.hook-materialization.json"),
        "routes_sha256": Path(f"{stem}.routes.json"),
    }
    for name, source in source_paths.items():
        if not source.is_file():
            raise GeneratedClaimProbeError(f"claims source is missing: {source.name}")
        if _sha256_file(source) != claims["source_files"][name]:
            raise GeneratedClaimProbeError(f"claims source digest differs: {source.name}")
    return claims, _sha256_bytes(canonical)


def load_generator_safety(path: Path, claims: Mapping[str, Any]) -> dict[str, Any]:
    """Load strict non-authoritative lethal-edge/guard proposals from metadata."""
    suffix = ".generator-claims.json"
    if not path.name.endswith(suffix):
        raise GeneratedClaimProbeError("claims path must end in .generator-claims.json")
    meta_path = path.parent / f"{path.name[:-len(suffix)]}.meta.json"
    if not meta_path.is_file() or _sha256_file(meta_path) != claims["source_files"]["meta_sha256"]:
        raise GeneratedClaimProbeError("generator safety metadata identity differs")
    try:
        metadata = json.loads(meta_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeneratedClaimProbeError("generator metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise GeneratedClaimProbeError("generator metadata is not an object")
    safety = _exact_mapping(metadata.get("safety"), {
        "version", "guard_height", "guard_thickness", "lethal_edges", "guard_walls",
    }, "generator safety")
    if (
        _integer(safety["version"], "safety version") != 1
        or _integer(safety["guard_height"], "guard height") != 96
        or _integer(safety["guard_thickness"], "guard thickness") != 16
    ):
        raise GeneratedClaimProbeError("generator safety dimensions differ from v1")
    edges = _list(safety["lethal_edges"], "lethal edges", minimum=1)
    walls = _list(safety["guard_walls"], "guard walls", minimum=1)
    normalized_edges = []
    for index, value in enumerate(edges):
        edge = _exact_mapping(value, {"side", "segment"}, f"lethal edge {index}")
        if edge["side"] not in {"west", "east", "south", "north"}:
            raise GeneratedClaimProbeError(f"lethal edge {index} side is invalid")
        segment = _list(edge["segment"], f"lethal edge {index} segment")
        if len(segment) != 5:
            raise GeneratedClaimProbeError(f"lethal edge {index} segment is invalid")
        normalized_edges.append({
            "side": edge["side"],
            "segment": [_integer(item, f"lethal edge {index} segment") for item in segment],
        })
    normalized_walls = []
    for index, value in enumerate(walls):
        wall = _list(value, f"guard wall {index}")
        if len(wall) != 6:
            raise GeneratedClaimProbeError(f"guard wall {index} bounds are invalid")
        bounds = [_integer(item, f"guard wall {index} bound") for item in wall]
        if any(bounds[axis] >= bounds[axis + 3] for axis in range(3)):
            raise GeneratedClaimProbeError(f"guard wall {index} bounds are inverted")
        normalized_walls.append(bounds)
    if len(set(tuple(item["segment"]) for item in normalized_edges)) != len(normalized_edges):
        raise GeneratedClaimProbeError("duplicate lethal edge proposal")
    if len(set(map(tuple, normalized_walls))) != len(normalized_walls):
        raise GeneratedClaimProbeError("duplicate guard wall proposal")
    return {
        "version": 1, "guard_height": 96, "guard_thickness": 16,
        "lethal_edges": normalized_edges, "guard_walls": normalized_walls,
    }


def load_generator_hook_materialization(
    path: Path, claims: Mapping[str, Any]
) -> dict[str, Any]:
    """Load the canonical selected-hook identity record bound by claims."""
    suffix = ".generator-claims.json"
    if not path.name.endswith(suffix):
        raise GeneratedClaimProbeError("claims path must end in .generator-claims.json")
    materialization_path = (
        path.parent / f"{path.name[:-len(suffix)]}.hook-materialization.json"
    )
    try:
        document, digest = load_materialization(materialization_path)
    except (HookClaimsV2Error, OSError) as error:
        raise GeneratedClaimProbeError(str(error)) from error
    if digest != claims["source_files"]["hook_materialization_sha256"]:
        raise GeneratedClaimProbeError("hook materialization identity differs")
    if document["selected_records"] != claims["hook_claims"]:
        raise GeneratedClaimProbeError("hook materialization records differ from claims")
    return document


def generated_bsp_provenance(
    bsp: Path, claims: Mapping[str, Any], claims_digest: str
) -> dict[str, Any]:
    """Create the deterministic local provenance record for a compiled proposal."""
    validate_generator_claims(claims)
    _sha(claims_digest, "generator claims digest")
    if generator_claims_sha256(claims) != claims_digest:
        raise GeneratedClaimProbeError("generator claims digest does not match claims")
    bsp = bsp.resolve()
    if not bsp.is_file() or bsp.stem != claims["map"]:
        raise GeneratedClaimProbeError("compiled BSP is missing or has a different map ID")
    return {
        "schema": "q2-generated-bsp-provenance-v1",
        "canonical_id": claims["map"],
        "bsp_sha256": _sha256_file(bsp),
        "bsp_bytes": bsp.stat().st_size,
        "generator_claims_sha256": claims_digest,
        "source_files": dict(claims["source_files"]),
    }


def _origin(entity: Any) -> tuple[float, float, float] | None:
    value = entity.value("origin")
    if not value:
        return 0.0, 0.0, 0.0
    words = value.split()
    if len(words) != 3:
        return None
    try:
        result = tuple(float(word) for word in words)
    except ValueError:
        return None
    return result if all(math.isfinite(number) for number in result) else None  # type: ignore[return-value]


def _model_bounds_milliunits(model: Any, offset: Sequence[float]) -> list[int]:
    return [
        round((model.mins[axis] + offset[axis]) * 1000) for axis in range(3)
    ] + [
        round((model.maxs[axis] + offset[axis]) * 1000) for axis in range(3)
    ]


def _l0_intersecting_cell_count(bounds: Sequence[int]) -> int:
    """Count 4u cells touched by an inclusive runtime linked AABB."""
    return math.prod(
        math.floor(bounds[axis + 3] / 4000)
        - math.floor(bounds[axis] / 4000) + 1
        for axis in range(3)
    )


def _trigger_hurt_runtime_bounds(
    model: Any, offset: Sequence[float]
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce CM setmodel + SV_LinkEdict + player linked-AABB touch law.

    CM_InlineModel spreads raw dmodel bounds by one unit. SV_LinkEdict spreads
    both the trigger and the standing player by one more unit. G_TouchTriggers
    then uses inclusive AABB overlap; it does not perform a convex headnode
    trace for trigger contact.
    """
    raw = _model_bounds_milliunits(model, offset)
    trigger_linked = [
        raw[axis] - 2_000 for axis in range(3)
    ] + [
        raw[axis + 3] + 2_000 for axis in range(3)
    ]
    standing_contact = _trigger_contact_bounds(
        trigger_linked, STANDING_MINS, STANDING_MAXS
    )
    return raw, trigger_linked, standing_contact


def _trigger_contact_bounds(
    trigger_linked: Sequence[int],
    hull_mins: Sequence[float],
    hull_maxs: Sequence[float],
) -> list[int]:
    """Return inclusive player origins whose linked AABB touches a trigger."""
    return [
        trigger_linked[axis] - round(hull_maxs[axis] * 1000) - 1_000
        for axis in range(3)
    ] + [
        trigger_linked[axis + 3] - round(hull_mins[axis] * 1000) + 1_000
        for axis in range(3)
    ]


def _box_request(
    identifier: str,
    start: Sequence[float],
    end: Sequence[float],
    mins: Sequence[float],
    maxs: Sequence[float],
    mask: int,
    *,
    headnode: int | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "id": identifier, "op": "box_trace", "start": list(start), "end": list(end),
        "mins": list(mins), "maxs": list(maxs), "mask": mask,
    }
    if headnode is not None:
        request["headnode"] = headnode
    return request


def _challenge_hazards(
    cm: Any,
    metadata: Any,
    claims: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> tuple[list[dict], dict]:
    hurt_models: dict[tuple[int, ...], tuple[list[int], list[int], bool]] = {}
    for entity in metadata.entities:
        if entity.classname != "trigger_hurt":
            continue
        model_name = entity.value("model")
        offset = _origin(entity)
        if not model_name.startswith("*") or offset is None:
            raise GeneratedClaimProbeError("compiled trigger_hurt lacks a valid brush model")
        try:
            model = metadata.models[int(model_name[1:])]
        except (ValueError, IndexError) as error:
            raise GeneratedClaimProbeError("compiled trigger_hurt model is invalid") from error
        raw, linked, standing_contact = _trigger_hurt_runtime_bounds(model, offset)
        try:
            spawnflags = int(entity.value("spawnflags") or "0", 10)
        except ValueError as error:
            raise GeneratedClaimProbeError("compiled trigger_hurt spawnflags are invalid") from error
        if tuple(raw) in hurt_models:
            raise GeneratedClaimProbeError("compiled trigger_hurt raw bounds are duplicated")
        hurt_models[tuple(raw)] = (
            linked, standing_contact, bool(spawnflags & (1 | 2))
        )

    results: list[dict] = []
    for ordinal, claim in enumerate(claims["hazard_claims"]):
        stateful = False
        bounds = [int(value) for value in claim["bounds_milliunits"]]
        world_bounds = [value / 1000.0 for value in bounds]
        if claim["type"] == "hurt":
            matched = hurt_models.get(tuple(bounds))
            if matched is None:
                raise GeneratedClaimProbeError(f"{claim['claim_id']} has no exact compiled hurt model")
            linked_bounds, standing_contact, stateful = matched
            raw_cells = _l0_intersecting_cell_count(linked_bounds)
            expanded_cells = _l0_intersecting_cell_count(standing_contact)
            evidence = EVIDENCE_LINKED_AABB_V1
        else:
            x = (world_bounds[0] + world_bounds[3]) / 2
            y = (world_bounds[1] + world_bounds[4]) / 2
            start = [x, y, world_bounds[5] + 4]
            end = [x, y, world_bounds[2] - 4]
            trace = cm.call([_box_request(
                f"claim-lava-trace:{ordinal}", start, end, [0, 0, 0], [0, 0, 0],
                CONTENTS_LAVA,
            )])[0]
            endpoint = trace.get("endpos")
            plane = trace.get("plane")
            if (
                trace.get("fraction", 1) >= 1
                or not trace.get("contents", 0) & CONTENTS_LAVA
                or not isinstance(endpoint, list) or len(endpoint) != 3
                or not isinstance(plane, dict)
            ):
                raise GeneratedClaimProbeError(f"{claim['claim_id']} has no compiled lava hit")
            normal = plane.get("normal")
            if not isinstance(normal, list) or len(normal) != 3:
                raise GeneratedClaimProbeError(f"{claim['claim_id']} lava hit has no plane")
            inside = [endpoint[axis] - normal[axis] * 2 for axis in range(3)]
            if not all(
                world_bounds[axis] <= inside[axis] <= world_bounds[axis + 3]
                for axis in range(3)
            ):
                raise GeneratedClaimProbeError(f"{claim['claim_id']} lava hit lies outside claim")
            contents = cm.call([{
                "id": f"claim-lava-contents:{ordinal}", "op": "point_contents",
                "point": inside,
            }])[0]
            if not contents.get("contents", 0) & CONTENTS_LAVA:
                raise GeneratedClaimProbeError(f"{claim['claim_id']} lava interior is not compiled lava")
            # Counts cover the positively evidenced L0 sample and its exact
            # standing-hull Minkowski expansion, not the generator AABB.
            raw_cells = 1
            expanded_cells = 9 * 9 * 15
            evidence = EVIDENCE_CM_CONTENTS_V1
        results.append({
            "claim_id": claim["claim_id"], "type": claim["type"],
            "raw_l0_cells": raw_cells, "expanded_l0_cells": expanded_cells,
            "contained": not (claim["type"] == "hurt" and stateful),
            "status": (
                "unknown" if claim["type"] == "hurt" and stateful else ORACLE_STATUS
            ),
            "evidence": evidence,
            "validation_version": VALIDATION_VERSION,
        })

    hurt_bounds = [bounds for bounds in hurt_models]
    proposed_walls = {tuple(wall) for wall in safety["guard_walls"]}
    expected_walls: list[tuple[int, ...]] = []
    lethal_count = 0
    for edge_index, edge in enumerate(safety["lethal_edges"]):
        x0, y0, x1, y1, floor_z = edge["segment"]
        side = edge["side"]
        if side == "west":
            outward = (-1.0, 0.0)
            wall = (x0, y0, floor_z, x0 + 16, y1, floor_z + 96)
        elif side == "east":
            outward = (1.0, 0.0)
            wall = (x0 - 16, y0, floor_z, x0, y1, floor_z + 96)
        elif side == "south":
            outward = (0.0, -1.0)
            wall = (x0, y0, floor_z, x1, y0 + 16, floor_z + 96)
        else:
            outward = (0.0, 1.0)
            wall = (x0, y0 - 16, floor_z, x1, y1, floor_z + 96)
        expected_walls.append(wall)
        if wall not in proposed_walls:
            raise GeneratedClaimProbeError(f"lethal edge {edge_index} has no exact guard proposal")
        midpoint = ((x0 + x1) / 2000.0, (y0 + y1) / 2000.0)
        floor = floor_z / 1000.0
        inward = [midpoint[0] - outward[0] * 32.0, midpoint[1] - outward[1] * 32.0]
        outside = [midpoint[0] + outward[0] * 32.0, midpoint[1] + outward[1] * 32.0]
        inward_support = cm.call([_box_request(
            f"lethal-inward:{edge_index}",
            [inward[0], inward[1], floor + 48.0],
            [inward[0], inward[1], floor - 64.0],
            STANDING_MINS, STANDING_MAXS, MASK_PLAYERSOLID,
        )])[0]
        plane = inward_support.get("plane")
        if (
            inward_support.get("startsolid")
            or not isinstance(inward_support.get("fraction"), (int, float))
            or inward_support["fraction"] >= 1
            or not isinstance(plane, Mapping)
            or not isinstance(plane.get("normal"), list)
            or len(plane["normal"]) != 3
            or plane["normal"][2] < 0.7
        ):
            raise GeneratedClaimProbeError(f"lethal edge {edge_index} has no compiled interior floor")
        covering_hurt = next((bounds for bounds in hurt_bounds if (
            bounds[0] <= round(outside[0] * 1000) <= bounds[3]
            and bounds[1] <= round(outside[1] * 1000) <= bounds[4]
            and bounds[5] <= floor_z - 64_000
        )), None)
        if covering_hurt is None:
            raise GeneratedClaimProbeError(f"lethal edge {edge_index} has no compiled lethal catch")
        void_trace = cm.call([_box_request(
            f"lethal-void:{edge_index}",
            [outside[0], outside[1], floor + 24.0],
            [outside[0], outside[1], covering_hurt[5] / 1000.0 + 1.0],
            [0, 0, 0], [0, 0, 0], MASK_PLAYERSOLID,
        )])[0]
        if void_trace.get("startsolid") or void_trace.get("fraction") != 1:
            raise GeneratedClaimProbeError(f"lethal edge {edge_index} exterior is not a void drop")
        wall_world = [value / 1000.0 for value in wall]
        guard_samples = []
        for along in (0.1, 0.5, 0.9):
            for height in (8.0, 48.0, 88.0):
                guard_samples.append([
                    wall_world[0] + (wall_world[3] - wall_world[0]) * (
                        0.5 if wall_world[3] - wall_world[0] <= 16.0 else along
                    ),
                    wall_world[1] + (wall_world[4] - wall_world[1]) * (
                        0.5 if wall_world[4] - wall_world[1] <= 16.0 else along
                    ),
                    floor + height,
                ])
        guard_results = cm.call([
            _box_request(
                f"lethal-guard:{edge_index}:{sample_index}",
                sample, sample, [0, 0, 0], [0, 0, 0], MASK_PLAYERSOLID,
            )
            for sample_index, sample in enumerate(guard_samples)
        ])
        if not all(result.get("startsolid") and result.get("allsolid") for result in guard_results):
            raise GeneratedClaimProbeError(f"lethal edge {edge_index} guard is not compiled solid to 96u")
        lethal_count += 1
    if set(expected_walls) != proposed_walls or len(expected_walls) != len(proposed_walls):
        raise GeneratedClaimProbeError("guard proposals do not exactly cover lethal edges")

    hazard_types = sorted({record["type"] for record in results})
    aggregates = {
        "l0_raw_cells": sum(record["raw_l0_cells"] for record in results),
        "l0_expanded_cells": sum(record["expanded_l0_cells"] for record in results),
        "types": hazard_types,
        "lethal_drop_edges": lethal_count,
        "guarded_drop_edges": lethal_count,
        "uncontained_drop_edges": 0,
    }
    return results, aggregates


def _grid_index(point: Sequence[float], origin: Sequence[int]) -> tuple[int, int, int]:
    return tuple(math.floor((point[axis] - origin[axis]) / 16) for axis in range(3))  # type: ignore[return-value]


def _project_route_point(
    cm: Any,
    point_milliunits: Sequence[int],
    nodes: Mapping[tuple[int, int, int], Any],
    origin: Sequence[int],
    identifier: str,
) -> tuple[int, int, int]:
    point = tuple(value / 1000.0 for value in point_milliunits)
    support = cm.call([_box_request(
        f"route-support:{identifier}",
        (point[0], point[1], point[2] + 32.0),
        (point[0], point[1], point[2] - 64.0),
        STANDING_MINS, STANDING_MAXS, MASK_PLAYERSOLID,
    )])[0]
    plane = support.get("plane")
    if (
        support.get("startsolid")
        or not isinstance(support.get("fraction"), (int, float))
        or support["fraction"] >= 1
        or not isinstance(support.get("endpos"), list)
        or not isinstance(plane, dict)
        or not isinstance(plane.get("normal"), list)
        or len(plane["normal"]) != 3
        or plane["normal"][2] < 0.7
    ):
        raise GeneratedClaimProbeError(f"{identifier} has no compiled support")
    grounded = tuple(float(value) for value in support["endpos"])
    nominal = _grid_index(grounded, origin)
    candidates = [
        (key, node) for key, node in nodes.items()
        if key[0] == nominal[0] and key[1] == nominal[1] and abs(key[2] - nominal[2]) <= 2
    ]
    candidates.sort(key=lambda item: (
        math.dist(item[1].position, grounded), item[0][2], item[0][1], item[0][0]
    ))
    if not candidates:
        raise GeneratedClaimProbeError(f"{identifier} has no nearby compiled L1 origin")
    key, node = candidates[0]
    if math.dist(node.position, grounded) > 16.0:
        raise GeneratedClaimProbeError(f"{identifier} support differs from compiled L1 origin")
    return key


def _route_cost(
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    adjacency: Mapping[tuple[int, int, int], Sequence[tuple[tuple[int, int, int], int]]],
) -> int | None:
    queue = [(0, source)]
    best = {source: 0}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        if node == target:
            return cost
        for neighbor, edge_cost in adjacency.get(node, ()):
            candidate = cost + edge_cost
            if candidate >= INFINITE_COST_Q8:
                continue
            if candidate < best.get(neighbor, INFINITE_COST_Q8):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return None


def project_route_claims(
    cm: Any,
    nodes: Mapping[tuple[int, int, int], Any],
    edges: Sequence[Mapping[str, Any]],
    origin: Sequence[int],
    claims: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project route endpoints through CM and challenge graph connectivity/cost."""
    adjacency: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], int]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        source = tuple(_vec3(edge.get("source"), f"edge {index} source"))
        target = tuple(_vec3(edge.get("target"), f"edge {index} target"))
        cost = _integer(edge.get("cost"), f"edge {index} cost", minimum=1)
        if _integer(edge.get("evidence"), f"edge {index} evidence", minimum=1) <= 0:
            raise GeneratedClaimProbeError(f"edge {index} lacks evidence")
        if _integer(edge.get("validation_version"), f"edge {index} version", minimum=1) <= 0:
            raise GeneratedClaimProbeError(f"edge {index} lacks validation version")
        if source not in nodes or target not in nodes:
            raise GeneratedClaimProbeError(f"edge {index} references an absent L1 node")
        adjacency[source].append((target, cost))
    for values in adjacency.values():
        values.sort(key=lambda item: (item[0][2], item[0][1], item[0][0], item[1]))

    projection_cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    results = []
    for claim in claims["route_claims"]:
        source_point = tuple(claim["source_milliunits"])
        target_point = tuple(claim["target_milliunits"])
        if source_point not in projection_cache:
            projection_cache[source_point] = _project_route_point(
                cm, source_point, nodes, origin, f"{claim['claim_id']}:source"
            )
        if target_point not in projection_cache:
            projection_cache[target_point] = _project_route_point(
                cm, target_point, nodes, origin, f"{claim['claim_id']}:target"
            )
        source = projection_cache[source_point]
        target = projection_cache[target_point]
        cost = _route_cost(source, target, adjacency)
        if cost is None:
            raise GeneratedClaimProbeError(f"{claim['claim_id']} has no evidenced Atlas route")
        results.append({
            "claim_id": claim["claim_id"],
            "source_milliunits": list(source_point),
            "target_milliunits": list(target_point),
            "source_l1": list(source), "target_l1": list(target),
            "connected": True, "cost_q8": cost, "status": ORACLE_STATUS,
            "evidence": EVIDENCE_ATLAS_ROUTE_V1,
            "validation_version": VALIDATION_VERSION,
        })
    return results


def compiled_lighting_diagnostics(
    cm: Any,
    metadata: Any,
    spawn_records: Sequence[Mapping[str, Any]],
) -> dict:
    """Diagnose spawn regions from compiled light entities, qrad data, and CM."""
    if metadata.lightmaps.byte_count <= 0 or metadata.faces.lightmapped_count <= 0:
        raise GeneratedClaimProbeError("compiled BSP has no admitted qrad lighting")
    _sha(metadata.lightmaps.sha256, "compiled lightdata")
    regions = {
        _integer(record.get("region_id"), "spawn region", minimum=1)
        for record in spawn_records
    }
    if not regions:
        raise GeneratedClaimProbeError("compiled spawns have no Atlas regions")
    worldspawn = next(
        (entity for entity in metadata.entities if entity.classname == "worldspawn"),
        None,
    )
    if worldspawn is None:
        raise GeneratedClaimProbeError("compiled BSP has no worldspawn lighting contract")
    try:
        floor_threshold = float(worldspawn.value("_ml_min_floor_light_value"))
        interior_threshold = float(worldspawn.value("_ml_min_interior_light_value"))
        interior_radius_threshold = float(worldspawn.value("_ml_min_interior_light_radius"))
    except ValueError as error:
        raise GeneratedClaimProbeError("compiled lighting thresholds are invalid") from error
    if (
        floor_threshold != MIN_FLOOR_LIGHT_VALUE
        or interior_threshold != MIN_INTERIOR_LIGHT_VALUE
        or interior_radius_threshold != MIN_INTERIOR_LIGHT_RADIUS
    ):
        raise GeneratedClaimProbeError("compiled lighting thresholds differ from frozen v6")

    lights = []
    for entity in metadata.entities:
        if entity.classname != "light":
            continue
        point = _origin(entity)
        try:
            intensity = float(entity.value("light"))
            radius = float(entity.value("_ml_radius"))
        except ValueError:
            continue
        floor_light = entity.value("_ml_floor_light") == "1"
        interior_light = entity.value("_ml_interior_light") == "1"
        qualified = (
            floor_light and intensity >= floor_threshold
        ) or (
            interior_light
            and intensity >= interior_threshold
            and radius >= interior_radius_threshold
        )
        if (
            point is not None
            and math.isfinite(intensity)
            and math.isfinite(radius)
            and radius > 0
            and qualified
        ):
            lights.append((entity.index, point, radius))
    if not lights:
        raise GeneratedClaimProbeError("compiled BSP has no qualified v6 light entities")

    dark_regions: set[int] = set()
    for spawn_index, record in enumerate(spawn_records):
        origin_milliunits = _vec3(
            record.get("origin_milliunits"), f"spawn {spawn_index} origin"
        )
        # Authored spawn +9 link offset +22 eye height.
        eye = [
            origin_milliunits[0] / 1000.0,
            origin_milliunits[1] / 1000.0,
            origin_milliunits[2] / 1000.0 + 31.0,
        ]
        eligible_lights = [
            (light_index, light_origin)
            for light_index, light_origin, radius in lights
            if math.dist(eye, light_origin) <= radius
        ]
        requests = [
            _box_request(
                f"lighting:{spawn_index}:{light_index}",
                eye, light_origin, [0, 0, 0], [0, 0, 0], MASK_SHOT,
            )
            for light_index, light_origin in eligible_lights
        ]
        visible = any(
            response.get("fraction") == 1 and not response.get("startsolid")
            for response in (cm.call(requests) if requests else [])
        )
        if not visible:
            dark_regions.add(_integer(record.get("region_id"), "spawn region", minimum=1))
    return {
        "lightdata_bytes": metadata.lightmaps.byte_count,
        "lightdata_sha256": metadata.lightmaps.sha256,
        "lightmapped_faces": metadata.faces.lightmapped_count,
        "spawn_region_count": len(regions),
        "dark_spawn_regions": len(dark_regions),
    }


def analyze_non_hook_claims(
    cm: Any,
    metadata: Any,
    nodes: Mapping[tuple[int, int, int], Any],
    edges: Sequence[Mapping[str, Any]],
    origin: Sequence[int],
    spawn_records: Sequence[Mapping[str, Any]],
    claims: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    """Challenge every generated non-hook claim; unknown evidence raises."""
    validate_generator_claims(claims)
    hazard_claims, hazards = _challenge_hazards(cm, metadata, claims, safety)
    return {
        "hazard_claims": hazard_claims,
        "hazards": hazards,
        "route_claims": project_route_claims(cm, nodes, edges, origin, claims),
        "lighting": compiled_lighting_diagnostics(cm, metadata, spawn_records),
    }
