#!/usr/bin/env python3
"""Challenge generator-v6 claims with a hash-bound compiled Atlas report.

Source and V4 hook sidecar metadata are claims only. Generated-map promotion
requires positive oracle-derived evidence from q2-atlas-analysis-v1 for every
claim.  Missing, failed, or unknown compiled facts always reject promotion.
Stock-map analysis uses a separate criteria path and never inherits v6 tags.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maps.generator import (  # noqa: E402
    DM_SPAWN_COUNT,
    MIN_FLOOR_LIGHT_COVERAGE,
    MIN_SPAWN_SEPARATION,
)
from harness.hook_claims_v4 import (  # noqa: E402
    HookClaimsV4Error,
    load_candidates,
    load_materialization,
    runtime_records_sha256,
    validate_selected_record as validate_hook_record_v4,
    validate_runtime_sidecar,
    validation_trace_sha256,
)
from harness.atlas_source_closure import (  # noqa: E402
    atlas_analyzer_authority_sha256,
)
from harness.ibsp38 import (  # noqa: E402
    BspLimits,
    BspValidationError,
    _lump_bytes as _bsp_lump_bytes,
    _parse_entities as _parse_bsp_entities,
    _parse_lumps as _parse_bsp_lumps,
)
from tools.validate_maps import (  # noqa: E402
    POINT_RE,
    _origin,
    _parse_brush_aabbs,
    _parse_entities,
    static_validate,
)


CLAIMS_SCHEMA = "q2-generator-claims-v3"
ANALYSIS_SCHEMA = "q2-atlas-analysis-v1"
REPORT_SCHEMA = "q2-generator-claim-validation-v1"
ORACLE_STATUS = "oracle"
MILLIUNITS = 1000
Q8 = 256
INFINITE_COST_Q8 = 0xFFFFFFFF
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_ROUTE_COST_ABSOLUTE_ERROR_Q8 = 1024 * Q8
MAX_ROUTE_COST_RATIO = 2.0
MAX_L0_CHUNKS = 1_200
MAX_L0_DECOMPRESSED_BYTES = 16 * 1024 * 1024
MAX_ATLAS_DECOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_ATLAS_RESIDENT_BYTES = 32 * 1024 * 1024
MAX_BUILD_RSS_BYTES = 512 * 1024 * 1024
MAX_FULL_COLD_MILLISECONDS = 300_000
FULL_COLD_EXACT_SUFFIXES = (
    ".atlas.bin", ".atlas.bin.zst", ".navigation.bin.zst",
    ".visibility.bin.zst", ".design-signature.json", ".objectives.json",
)
FULL_COLD_SEMANTIC_SUFFIXES = (
    ".atlas.manifest.json", ".analysis.manifest.json",
)
FULL_COLD_SUFFIXES = FULL_COLD_EXACT_SUFFIXES + FULL_COLD_SEMANTIC_SUFFIXES
OBJECTIVE_SCHEMA = "q2-atlas-objectives-v1"
OBJECTIVE_MEDIA_TYPE = "application/vnd.q2.atlas-objectives-v1"
OBJECTIVE_LIMIT = 8_192
OBJECTIVE_CLASSES = frozenset({
    "weapon", "ammunition", "health", "armor", "powerup", "rune",
    "control", "spawn_egress",
})
OBJECTIVE_GUIDEPOST_ANALYSIS_SCHEMA = "q2-atlas-objective-guidepost-analysis-v1"
# Frozen identical to harness.atlas_analyzer omission reason constants.
OBJECTIVE_OMISSION_BEYOND_FENCE = "beyond_objective_target_max_distance"
OBJECTIVE_OMISSION_NO_TARGET = "no_admitted_supported_passable_l1"
OBJECTIVE_OMISSION_REASONS = frozenset({
    OBJECTIVE_OMISSION_BEYOND_FENCE,
    OBJECTIVE_OMISSION_NO_TARGET,
})
MAX_OBJECTIVE_NEAREST_DISTANCE_MILLIUNITS = 2**63 - 1
STOCK_PROVENANCE = ROOT / "docs/multires/stock-q2dm1-q2dm8.provenance.json"
STOCK_INVENTORY = ROOT / "tests/fixtures/corpus/stock-q2dm1-q2dm8.json"
CLAIM_ID_RE = re.compile(r"^[a-z0-9:_-]{1,127}$")
DROP_UNKNOWN_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PROMOTION_ALLOWED_DROP_UNKNOWN_REASONS = frozenset({
    "no_landing",
    "unsupported_dynamic_mover",
})
HEX = frozenset("0123456789abcdef")


class ClaimValidationError(ValueError):
    """Raised when a claimed or compiled contract is malformed."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_analyzer_sha256() -> str:
    """Use the analyzer's shared source-closure authority identity."""

    return atlas_analyzer_authority_sha256(ROOT)


def _no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClaimValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ClaimValidationError(f"{path.name} exceeds {MAX_INPUT_BYTES} bytes")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ClaimValidationError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClaimValidationError(f"cannot read {path.name}: {exc}") from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaimValidationError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ClaimValidationError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ClaimValidationError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
        or set(value) == {"0"}
    ):
        raise ClaimValidationError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClaimValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ClaimValidationError(f"{label} must be at least {minimum}")
    return value


def _validate_drop_unknown_reasons(
    drop: Mapping[str, Any], label: str, failures: list[str],
) -> None:
    """Require complete canonical Unknown accounting and admissible reasons."""

    reason_counts = _mapping(
        drop.get("unknown_reason_counts"), f"{label} Unknown reason counts",
    )
    if list(reason_counts) != sorted(reason_counts):
        failures.append(f"{label} Unknown reason counts are not canonically sorted")
    accounted = 0
    for reason, value in reason_counts.items():
        count = _integer(
            value, f"{label} Unknown reason {reason!r} count", minimum=1,
        )
        accounted += count
        if not isinstance(reason, str) or not DROP_UNKNOWN_REASON_RE.fullmatch(reason):
            failures.append(f"{label} Unknown reason {reason!r} is malformed")
        elif reason not in PROMOTION_ALLOWED_DROP_UNKNOWN_REASONS:
            failures.append(
                f"{label} Unknown reason {reason} is not promotion-admissible"
            )
    unknown = _integer(
        drop.get("unknown_omitted"), f"{label} unknown omitted", minimum=0,
    )
    if accounted != unknown:
        failures.append(
            f"{label} Unknown reason counts do not cover every omitted candidate"
        )


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClaimValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ClaimValidationError(f"{label} is outside its finite range")
    return result


def _claim_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not CLAIM_ID_RE.fullmatch(value):
        raise ClaimValidationError(f"{label} is not a canonical claim ID")
    return value


def _vec3_int(value: object, label: str) -> tuple[int, int, int]:
    items = _list(value, label)
    if len(items) != 3:
        raise ClaimValidationError(f"{label} must contain three integers")
    return tuple(_integer(item, f"{label}[{index}]") for index, item in enumerate(items))  # type: ignore[return-value]


def _to_milliunits(values: Sequence[object], label: str) -> list[int]:
    result = []
    for index, value in enumerate(values):
        number = _finite(value, f"{label}[{index}]")
        scaled = round(number * MILLIUNITS)
        if not math.isclose(number * MILLIUNITS, scaled, abs_tol=1e-6):
            raise ClaimValidationError(f"{label}[{index}] is not milliunit-exact")
        result.append(int(scaled))
    return result


def _entity_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines: list[str] = []
    depth = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "{":
            if depth == 0:
                lines = []
            lines.append(line)
            depth += 1
        elif stripped == "}":
            if depth <= 0:
                raise ClaimValidationError("unbalanced closing brace in .map source")
            lines.append(line)
            depth -= 1
            if depth == 0:
                blocks.append("".join(lines))
                lines = []
        elif depth:
            lines.append(line)
    if depth or lines:
        raise ClaimValidationError("unterminated entity in .map source")
    return blocks


def _hurt_bounds(source: str) -> list[list[int]]:
    result = []
    for block in _entity_blocks(source):
        entities = _parse_entities(block)
        if not entities or entities[0].get("classname") != "trigger_hurt":
            continue
        bounds = _parse_brush_aabbs(block)
        if not bounds:
            raise ClaimValidationError("trigger_hurt has no axis-aligned brush")
        result.extend(_to_milliunits(box, "trigger_hurt bounds") for box in bounds)
    return sorted(result)


def _parse_hook_claims(path: Path) -> list[dict[str, Any]]:
    claims = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) != 8:
            raise ClaimValidationError(
                f"{path.name}:{line_number}: hook record must have eight fields"
            )
        try:
            parsed = [float(item) for item in fields]
        except ValueError as exc:
            raise ClaimValidationError(
                f"{path.name}:{line_number}: hook record contains non-numeric text"
            ) from exc
        numbers = [
            _finite(item, f"hook field {index}")
            for index, item in enumerate(parsed[:7])
        ]
        flags_number = _finite(parsed[7], "hook flags", minimum=0)
        if not flags_number.is_integer() or int(flags_number) > 7:
            raise ClaimValidationError("hook flags must be an integer in [0,7]")
        claims.append(
            {
                "claim_id": f"hook:{len(claims):04d}",
                "measured_anchor_milliunits": _to_milliunits(
                    numbers[:3], "hook measured anchor"
                ),
                "landing_milliunits": _to_milliunits(numbers[3:6], "hook landing"),
                "distance_milliunits": _to_milliunits(
                    numbers[6:7], "hook distance"
                )[0],
                "flags": int(flags_number),
            }
        )
    if not claims:
        raise ClaimValidationError("generated-v6 hook sidecar contains no claims")
    return claims


def _route_contract(routes: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    if routes.get("version") != 2:
        raise ClaimValidationError("generated route sidecar must have version 2")
    nodes = _list(routes.get("nodes"), "route nodes")
    by_id: dict[int, Mapping[str, Any]] = {}
    components_by_id: dict[int, int | None] = {}
    for index, value in enumerate(nodes):
        node = _mapping(value, f"route node {index}")
        node_id = _integer(node.get("id"), f"route node {index} id", minimum=0)
        if node_id in by_id:
            raise ClaimValidationError(f"duplicate route node id {node_id}")
        _to_milliunits(
            [node.get("x"), node.get("y"), node.get("z")],
            f"route node {node_id} origin",
        )
        raw_component = node.get("source_component")
        components_by_id[node_id] = (
            None if raw_component is None else _integer(
                raw_component,
                f"route node {node_id} source component",
                minimum=0,
            )
        )
        by_id[node_id] = node

    route_claims: list[dict] = []
    route_records: list[dict] = []
    seen_routes: set[str] = set()
    for route_index, value in enumerate(_list(routes.get("routes"), "routes")):
        route = _mapping(value, f"route {route_index}")
        numeric_id = _integer(route.get("id"), f"route {route_index} id", minimum=0)
        route_id = f"route:{numeric_id:04d}"
        if route_id in seen_routes:
            raise ClaimValidationError(f"duplicate route id {route_id}")
        seen_routes.add(route_id)
        archetype = route.get("archetype")
        if archetype not in {"offense", "survival", "control", "balanced"}:
            raise ClaimValidationError(f"{route_id} has unknown archetype")
        claimed_distance = _finite(
            route.get("dist"), f"{route_id} distance", minimum=0
        )
        start_room = _integer(route.get("start_room"), f"{route_id} start room", minimum=0)
        start_node_id = _integer(
            route.get("start_node_id"), f"{route_id} start node id", minimum=0,
        )
        start = by_id.get(start_node_id)
        if (
            start is None or start.get("type") != "spawn"
            or start.get("room") != start_room
        ):
            raise ClaimValidationError(
                f"{route_id} start node is not the declared spawn/room"
            )
        source_component = _integer(
            route.get("source_component"),
            f"{route_id} source component",
            minimum=0,
        )
        if components_by_id[start_node_id] != source_component:
            raise ClaimValidationError(
                f"{route_id} source component differs from its start node"
            )
        item_ids = [
            _integer(item, f"{route_id} node id", minimum=0)
            for item in _list(route.get("node_ids"), f"{route_id} node ids")
        ]
        if len(item_ids) < 2 or any(item not in by_id for item in item_ids):
            raise ClaimValidationError(f"{route_id} has insufficient/unknown nodes")
        if any(
            by_id[item].get("type") != "item"
            or components_by_id[item] != source_component
            for item in item_ids
        ):
            raise ClaimValidationError(
                f"{route_id} endpoints do not share its source standing component"
            )
        path = [start, *(by_id[item] for item in item_ids), start]
        origins = [
            tuple(
                _finite(node.get(axis), f"{route_id} {axis}")
                for axis in ("x", "y", "z")
            )
            for node in path
        ]
        # The source route distance is a deterministic standing-grid geodesic,
        # not compiled collision authority.  Retain its blocker-aware proposal
        # while enforcing the exact endpoint loop's geometric lower bound.
        geometric_distance = sum(
            math.dist(source, target)
            for source, target in zip(origins, origins[1:])
        )
        if claimed_distance < geometric_distance:
            raise ClaimValidationError(
                f"{route_id} distance {claimed_distance:g} falls below "
                f"endpoint-loop geometry {geometric_distance:.6f}"
            )
        if claimed_distance <= 0:
            raise ClaimValidationError(f"{route_id} has no measurable route")
        segment_ids = []
        for segment_index, (source, target) in enumerate(zip(path, path[1:])):
            claim_id = f"{route_id}:segment:{segment_index:04d}"
            segment_ids.append(claim_id)
            route_claims.append(
                {
                    "claim_id": claim_id,
                    "route_id": route_id,
                    "source_milliunits": _to_milliunits(
                        [source.get("x"), source.get("y"), source.get("z")],
                        f"{claim_id} source",
                    ),
                    "target_milliunits": _to_milliunits(
                        [target.get("x"), target.get("y"), target.get("z")],
                        f"{claim_id} target",
                    ),
                }
            )
        route_records.append(
            {
                "route_id": route_id,
                "archetype": archetype,
                "claimed_cost_q8": round(claimed_distance * Q8),
                "segment_claim_ids": segment_ids,
            }
        )
    if not route_records:
        raise ClaimValidationError("generated-v6 sidecar contains no routes")
    return route_claims, route_records


def build_generator_claims(map_path: Path) -> dict[str, Any]:
    map_path = map_path.resolve()
    if map_path.suffix != ".map":
        raise ClaimValidationError("generator source path must end in .map")
    stem = map_path.with_suffix("")
    paths = {
        "map": map_path,
        "meta": stem.with_suffix(".meta.json"),
        "lattice": Path(f"{stem}.lattice.json"),
        "hook_zones": stem.with_suffix(".json"),
        "hook_materialization": Path(f"{stem}.hook-materialization.json"),
        "routes": Path(f"{stem}.routes.json"),
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ClaimValidationError(f"missing generator sidecars: {sorted(missing)}")

    source = map_path.read_text(encoding="utf-8")
    meta = _mapping(load_json(paths["meta"]), "generator metadata")
    lattice = _mapping(load_json(paths["lattice"]), "lattice sidecar")
    routes = _mapping(load_json(paths["routes"]), "route sidecar")
    if meta.get("generator") != "v6" or meta.get("name") != map_path.stem:
        raise ClaimValidationError("metadata is not generator v6 for this map")

    source_spawns = sorted(
        _to_milliunits(origin, "source spawn")
        for entity in _parse_entities(source)
        if entity.get("classname") == "info_player_deathmatch"
        for origin in [_origin(entity)]
        if origin is not None
    )
    lattice_spawns = sorted(
        _to_milliunits(
            [spawn.get("x"), spawn.get("y"), spawn.get("z")],
            f"lattice spawn {index}",
        )
        for index, value in enumerate(_list(lattice.get("spawns"), "lattice spawns"))
        for spawn in [_mapping(value, f"lattice spawn {index}")]
    )
    if source_spawns != lattice_spawns:
        raise ClaimValidationError("source and lattice spawn origins differ")
    if len(source_spawns) != DM_SPAWN_COUNT or meta.get("spawns") != len(source_spawns):
        raise ClaimValidationError(
            f"generated-v6 promotion requires exactly {DM_SPAWN_COUNT} spawns"
        )
    if len({tuple(origin) for origin in source_spawns}) != len(source_spawns):
        raise ClaimValidationError("duplicate generator spawn origin")
    spawns = [
        {"claim_id": f"spawn:{index:04d}", "origin_milliunits": origin}
        for index, origin in enumerate(source_spawns)
    ]

    hazards = []
    danger = sorted(
        _to_milliunits(_list(value, f"danger {index}"), f"danger {index}")
        for index, value in enumerate(_list(lattice.get("danger"), "danger claims"))
    )
    if meta.get("lava_pools") != len(danger):
        raise ClaimValidationError("metadata and lattice lava counts differ")
    for index, bounds in enumerate(danger):
        if len(bounds) != 6 or any(bounds[axis] >= bounds[axis + 3] for axis in range(3)):
            raise ClaimValidationError(f"lava bounds {index} are not ordered")
        hazards.append(
            {
                "claim_id": f"hazard:lava:{index:04d}",
                "type": "lava",
                "bounds_milliunits": bounds,
            }
        )
    hurt = _hurt_bounds(source)
    if meta.get("kill_planes") != len(hurt) or not hurt:
        raise ClaimValidationError("generated-v6 kill-plane hurt contract differs")
    for index, bounds in enumerate(hurt):
        hazards.append(
            {
                "claim_id": f"hazard:hurt:{index:04d}",
                "type": "hurt",
                "bounds_milliunits": bounds,
            }
        )
    hazards.sort(key=lambda claim: claim["claim_id"])

    runtime_hooks = _parse_hook_claims(paths["hook_zones"])
    try:
        materialization, materialization_sha256 = load_materialization(
            paths["hook_materialization"]
        )
        candidates, candidates_sha256, meta_sha256 = load_candidates(paths["meta"])
    except HookClaimsV4Error as error:
        raise ClaimValidationError(str(error)) from error
    bsp_path = stem.with_suffix(".bsp")
    if not bsp_path.is_file():
        raise ClaimValidationError("compiled BSP is required before hook materialization")
    if (
        materialization["map"] != map_path.stem
        or materialization["bsp"] != {
            "sha256": file_sha256(bsp_path), "size_bytes": bsp_path.stat().st_size,
        }
        or materialization["candidates"]["meta_sha256"] != meta_sha256
        or materialization["candidates"]["records_sha256"] != candidates_sha256
        or materialization["candidates"]["record_count"] != len(candidates["records"])
    ):
        raise ClaimValidationError("hook materialization source/BSP identity differs")
    runtime_text = paths["hook_zones"].read_bytes()
    hooks = materialization["selected_records"]
    try:
        validate_runtime_sidecar(
            runtime_text, map_id=map_path.stem,
            bsp_sha256=materialization["bsp"]["sha256"],
            materialization_sha256=materialization_sha256, records=hooks,
        )
    except HookClaimsV4Error as error:
        raise ClaimValidationError(str(error)) from error
    if len(runtime_hooks) != len(hooks) or runtime_records_sha256(hooks) != materialization[
        "runtime_records_sha256"
    ]:
        raise ClaimValidationError("runtime hook rows differ from materialized records")
    for runtime, selected in zip(runtime_hooks, hooks):
        for field in (
            "measured_anchor_milliunits", "landing_milliunits",
            "distance_milliunits", "flags",
        ):
            if runtime[field] != selected[field]:
                raise ClaimValidationError("runtime hook geometry differs from selected v4 record")
    route_claims, route_records = _route_contract(routes)
    claims = {
        "schema": CLAIMS_SCHEMA,
        "map": map_path.stem,
        "generator": "v6",
        "source_files": {
            "map_sha256": file_sha256(paths["map"]),
            "meta_sha256": file_sha256(paths["meta"]),
            "lattice_sha256": file_sha256(paths["lattice"]),
            "hook_zones_sha256": file_sha256(paths["hook_zones"]),
            "hook_materialization_sha256": file_sha256(paths["hook_materialization"]),
            "routes_sha256": file_sha256(paths["routes"]),
        },
        "spawns": spawns,
        "hazard_claims": hazards,
        "hook_claims": hooks,
        "route_claims": route_claims,
        "routes": route_records,
    }
    validate_generator_claims(claims)
    return claims


def _unique_claims(values: object, label: str) -> list[Mapping[str, Any]]:
    records = [_mapping(value, f"{label} item") for value in _list(values, label)]
    ids = [_claim_id(record.get("claim_id"), f"{label} claim_id") for record in records]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ClaimValidationError(f"{label} IDs must be unique and sorted")
    return records


def validate_generator_claims(value: object) -> dict[str, Any]:
    claims = _mapping(value, "generator claims")
    _exact_keys(
        claims,
        {
            "schema", "map", "generator", "source_files", "spawns",
            "hazard_claims", "hook_claims", "route_claims", "routes",
        },
        "generator claims",
    )
    if claims["schema"] != CLAIMS_SCHEMA or claims["generator"] != "v6":
        raise ClaimValidationError("generator claims schema/generator mismatch")
    if not isinstance(claims["map"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,64}", claims["map"]
    ):
        raise ClaimValidationError("invalid generated map name")
    source_files = _mapping(claims["source_files"], "source files")
    _exact_keys(
        source_files,
        {
            "map_sha256", "meta_sha256", "lattice_sha256",
            "hook_zones_sha256", "hook_materialization_sha256", "routes_sha256",
        },
        "source files",
    )
    for name, digest in source_files.items():
        _digest(digest, name)

    spawns = _unique_claims(claims["spawns"], "spawns")
    if len(spawns) != DM_SPAWN_COUNT:
        raise ClaimValidationError("generated claims require exactly eight spawns")
    for spawn in spawns:
        _exact_keys(spawn, {"claim_id", "origin_milliunits"}, "spawn claim")
        _vec3_int(spawn["origin_milliunits"], "spawn origin")

    hazards = _unique_claims(claims["hazard_claims"], "hazard claims")
    if not hazards:
        raise ClaimValidationError("generated claims require hazard claims")
    for hazard in hazards:
        _exact_keys(hazard, {"claim_id", "type", "bounds_milliunits"}, "hazard claim")
        if hazard["type"] not in {"lava", "hurt"}:
            raise ClaimValidationError("unsupported generator hazard type")
        bounds = _list(hazard["bounds_milliunits"], "hazard bounds")
        if len(bounds) != 6:
            raise ClaimValidationError("hazard bounds must contain six integers")
        numbers = [_integer(item, "hazard bound") for item in bounds]
        if any(numbers[axis] >= numbers[axis + 3] for axis in range(3)):
            raise ClaimValidationError("hazard bounds are not ordered")

    hooks = _unique_claims(claims["hook_claims"], "hook claims")
    if len(hooks) != 6:
        raise ClaimValidationError("generated claims require exactly six hook-v4 claims")
    for hook in hooks:
        try:
            validate_hook_record_v4(dict(hook), "hook claim")
        except HookClaimsV4Error as error:
            raise ClaimValidationError(str(error)) from error

    route_claims = _unique_claims(claims["route_claims"], "route claims")
    if not route_claims:
        raise ClaimValidationError("generated claims require route claims")
    segments_by_route: dict[str, list[str]] = {}
    for route_claim in route_claims:
        _exact_keys(
            route_claim,
            {"claim_id", "route_id", "source_milliunits", "target_milliunits"},
            "route claim",
        )
        route_id = _claim_id(route_claim["route_id"], "route id")
        segments_by_route.setdefault(route_id, []).append(route_claim["claim_id"])
        _vec3_int(route_claim["source_milliunits"], "route source")
        _vec3_int(route_claim["target_milliunits"], "route target")

    routes = _list(claims["routes"], "route summaries")
    route_ids = []
    for value in routes:
        route = _mapping(value, "route summary")
        _exact_keys(
            route,
            {"route_id", "archetype", "claimed_cost_q8", "segment_claim_ids"},
            "route summary",
        )
        route_id = _claim_id(route["route_id"], "route summary id")
        route_ids.append(route_id)
        if route["archetype"] not in {"offense", "survival", "control", "balanced"}:
            raise ClaimValidationError("unknown route archetype")
        cost = _integer(route["claimed_cost_q8"], "claimed route cost", minimum=1)
        if cost >= INFINITE_COST_Q8:
            raise ClaimValidationError("claimed route cost uses infinity sentinel")
        segment_ids = [
            _claim_id(item, "route segment id")
            for item in _list(route["segment_claim_ids"], "route segment ids")
        ]
        if len(segment_ids) < 2 or segment_ids != segments_by_route.get(route_id):
            raise ClaimValidationError(f"{route_id} segment summary differs")
    if route_ids != sorted(route_ids) or len(set(route_ids)) != len(route_ids):
        raise ClaimValidationError("route summary IDs must be unique and sorted")
    if set(route_ids) != set(segments_by_route):
        raise ClaimValidationError("route claims and summaries differ")
    return dict(claims)


def generator_claims_sha256(claims: Mapping[str, Any]) -> str:
    validate_generator_claims(claims)
    return sha256_bytes(canonical_bytes(claims))


def _static_args() -> argparse.Namespace:
    return argparse.Namespace(
        min_spawns=DM_SPAWN_COUNT,
        min_spawn_distance=float(MIN_SPAWN_SEPARATION),
        min_span=1024.0,
        min_spawn_area=1_000_000.0,
        min_weapons=4,
        min_pickups=8,
        min_hook_zones=6,
        min_light_coverage=MIN_FLOOR_LIGHT_COVERAGE,
    )


def _criterion(failures: Iterable[str], checked_count: int) -> dict[str, Any]:
    unique = sorted(set(failures))
    return {
        "passed": not unique,
        "checked_count": checked_count,
        "failures": unique,
    }


def _oracle_record(record: object, label: str) -> list[str]:
    value = _mapping(record, label)
    failures = []
    if value.get("status") != ORACLE_STATUS:
        failures.append(f"{label} status is not oracle")
    if value.get("admitted") is not True:
        failures.append(f"{label} is not admitted")
    for name in ("executable_sha256", "physics_identity"):
        try:
            _digest(value.get(name), f"{label} {name}")
        except ClaimValidationError as exc:
            failures.append(str(exc))
    return failures


def _analysis_sections(analysis: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ClaimValidationError("compiled report schema is not q2-atlas-analysis-v1")
    compiled = _mapping(analysis.get("compiled_world"), "compiled_world")
    identity = _mapping(analysis.get("identity"), "analysis identity")
    return compiled, identity


def _canonical_semantic_manifest(path: Path) -> str:
    value = _mapping(load_json(path), "Atlas manifest")
    normalized = dict(value)
    if "build_peak_rss_bytes" not in normalized:
        raise ClaimValidationError("Atlas manifest lacks build_peak_rss_bytes")
    normalized.pop("build_peak_rss_bytes")
    payload = json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
    ).encode("ascii")
    return sha256_bytes(payload)


def _canonical_semantic_analysis_manifest(
    analysis: Mapping[str, Any],
) -> str:
    """Independently reconstruct the candidate manifest compared by cold build."""

    normalized = json.loads(json.dumps(analysis, allow_nan=False))
    if (
        normalized.get("status") != "passed"
        or normalized.get("deterministic_rebuild") is not True
    ):
        raise ClaimValidationError(
            "final analysis manifest lacks a passing independent-cold state"
        )
    normalized["status"] = "candidate"
    normalized["deterministic_rebuild"] = False
    normalized["confidence"] = "pending-independent-cold-rebuild"
    try:
        performance = normalized["performance"]
        performance.pop("full_cold_rebuild", None)
        performance.pop("primary_elapsed_milliseconds")
        normalized["artifacts"]["atlas"].pop("build_peak_rss_bytes")
        normalized["identity"].pop("atlas_manifest_sha256")
        atlas_manifest = normalized["artifacts"]["atlas_manifest"]
        atlas_manifest.pop("sha256")
        atlas_manifest.pop("uncompressed_bytes")
        atlas_manifest["verification"].pop("manifest_sha256")
    except (KeyError, TypeError) as error:
        raise ClaimValidationError(
            "analysis manifest lacks semantic cold-comparison fields"
        ) from error
    return sha256_bytes(json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("ascii"))


def _atlas_header(path: Path) -> tuple[int, tuple[int, int, int, int, int]]:
    with path.open("rb") as handle:
        header = handle.read(136)
    if len(header) != 136 or header[:8] != b"Q2ATL001":
        raise ClaimValidationError("Atlas artifact lacks the Q2ATL001 header")
    schema, byte_order, header_bytes = struct.unpack_from("<HHI", header, 8)
    if schema != 1 or byte_order != 0x454C or header_bytes != 136:
        raise ClaimValidationError("Atlas artifact header contract differs")
    counts = struct.unpack_from("<5Q", header, 56)
    lengths = struct.unpack_from("<5Q", header, 96)
    if header_bytes + sum(lengths) != path.stat().st_size:
        raise ClaimValidationError("Atlas artifact section lengths differ from file size")
    return int(lengths[0]), tuple(int(value) for value in counts)


def _objective_class_for_classname(classname: str) -> str | None:
    """Mirror the frozen analyzer/runtime classname authority exactly."""

    if classname.startswith("weapon_"):
        return "weapon"
    if classname.startswith("ammo_") or classname in {"item_pack", "item_bandolier"}:
        return "ammunition"
    if classname.startswith("item_health") or classname in {
        "item_adrenaline", "item_ancient_head",
    }:
        return "health"
    if classname.startswith("item_armor_") or classname in {
        "item_power_shield", "item_power_screen",
    }:
        return "armor"
    if classname in {
        "item_quad", "item_invulnerability", "item_silencer",
        "item_breather", "item_enviro",
    }:
        return "powerup"
    if classname.startswith(("rune_", "item_rune_")):
        return "rune"
    if classname.startswith(("item_flag_", "item_tech")):
        return "control"
    if classname == "info_player_deathmatch":
        return "spawn_egress"
    return None


def _bounded_integer(
    value: object, label: str, *, minimum: int, maximum: int,
) -> int:
    admitted = _integer(value, label, minimum=minimum)
    if admitted > maximum:
        raise ClaimValidationError(f"{label} must be at most {maximum}")
    return admitted


def _validate_objective_artifact(
    path: Path,
    *,
    map_id: str,
    bsp_sha256: object,
    atlas_sha256: object,
    origin: object,
) -> tuple[Mapping[str, Any], int]:
    document = _mapping(load_json(path), "objective artifact")
    if path.read_bytes() != canonical_bytes(document):
        raise ClaimValidationError(
            "objective artifact is not canonical sorted JSON plus LF"
        )
    _exact_keys(document, {
        "atlas_sha256", "bsp_sha256", "canonical_map_id", "objectives",
        "origin", "schema",
    }, "objective artifact")
    if (
        document.get("schema") != OBJECTIVE_SCHEMA
        or document.get("canonical_map_id") != map_id
        or document.get("bsp_sha256") != bsp_sha256
        or document.get("atlas_sha256") != atlas_sha256
        or document.get("origin") != origin
    ):
        raise ClaimValidationError(
            "objective artifact identity differs from admitted Atlas/map"
        )
    records = _list(document.get("objectives"), "objective records")
    if len(records) > OBJECTIVE_LIMIT:
        raise ClaimValidationError(
            f"objective artifact count exceeds {OBJECTIVE_LIMIT}"
        )
    previous_id = -1
    for ordinal, raw_record in enumerate(records):
        record = _mapping(raw_record, f"objective record {ordinal}")
        _exact_keys(record, {
            "class", "classname", "confidence", "l1_index", "objective_id",
            "risk", "world_milliunits",
        }, f"objective record {ordinal}")
        objective_id = _bounded_integer(
            record.get("objective_id"), f"objective {ordinal} ID",
            minimum=0, maximum=0xFFFF_FFFF,
        )
        if objective_id <= previous_id:
            raise ClaimValidationError(
                "objectives are not strictly ordered by stable ID"
            )
        previous_id = objective_id
        objective_class = record.get("class")
        classname = record.get("classname")
        if objective_class not in OBJECTIVE_CLASSES:
            raise ClaimValidationError(
                f"objective {objective_id} class is not admitted"
            )
        if (
            not isinstance(classname, str)
            or re.fullmatch(r"[a-z0-9_]{1,128}", classname) is None
            or _objective_class_for_classname(classname) != objective_class
        ):
            raise ClaimValidationError(
                f"objective {objective_id} classname/class mapping is invalid"
            )
        for axis, value in enumerate(_list(
            record.get("l1_index"), f"objective {objective_id} L1 index"
        )):
            _bounded_integer(
                value, f"objective {objective_id} L1 axis {axis}",
                minimum=-(2 ** 31), maximum=2 ** 31 - 1,
            )
        if len(record["l1_index"]) != 3:
            raise ClaimValidationError(
                f"objective {objective_id} L1 index must contain three integers"
            )
        for axis, value in enumerate(_list(
            record.get("world_milliunits"),
            f"objective {objective_id} world milliunits",
        )):
            _bounded_integer(
                value, f"objective {objective_id} world axis {axis}",
                minimum=-(2 ** 63), maximum=2 ** 63 - 1,
            )
        if len(record["world_milliunits"]) != 3:
            raise ClaimValidationError(
                f"objective {objective_id} world point must contain three integers"
            )
        _bounded_integer(
            record.get("risk"), f"objective {objective_id} risk",
            minimum=0, maximum=0xFFFF,
        )
        _bounded_integer(
            record.get("confidence"), f"objective {objective_id} confidence",
            minimum=0, maximum=0xFFFF,
        )
    return document, len(records)


def _validate_objective_guideposts(
    value: object,
    *,
    require_zero_omissions: bool = False,
    admitted_count: int | None = None,
) -> Mapping[str, Any]:
    """Validate compiled_world.objective_guideposts fail-closed evidence.

    Omission records are explicit stock/authored accounting evidence. Generated
    promotion requires ``omitted_count == 0`` so no supported objective can be
    silently dropped outside the strict L1 fence.
    """

    guideposts = _mapping(value, "objective guidepost analysis")
    _exact_keys(guideposts, {
        "admitted_count", "omitted_count", "omissions", "schema",
    }, "objective guidepost analysis")
    if guideposts.get("schema") != OBJECTIVE_GUIDEPOST_ANALYSIS_SCHEMA:
        raise ClaimValidationError(
            "objective guidepost analysis schema is not "
            f"{OBJECTIVE_GUIDEPOST_ANALYSIS_SCHEMA}"
        )
    admitted = _integer(
        guideposts.get("admitted_count"),
        "objective guidepost admitted_count", minimum=0,
    )
    omitted = _integer(
        guideposts.get("omitted_count"),
        "objective guidepost omitted_count", minimum=0,
    )
    if admitted_count is not None and admitted != admitted_count:
        raise ClaimValidationError(
            "objective guidepost admitted_count differs from objectives artifact"
        )
    if require_zero_omissions and omitted != 0:
        raise ClaimValidationError(
            "generated objective guidepost analysis must omit zero objectives"
        )
    omissions = _list(guideposts.get("omissions"), "objective guidepost omissions")
    if omitted != len(omissions):
        raise ClaimValidationError(
            "objective guidepost omitted_count differs from omissions length"
        )
    previous_entity_id = -1
    seen_ids: set[int] = set()
    for ordinal, raw_record in enumerate(omissions):
        record = _mapping(raw_record, f"objective omission {ordinal}")
        keys = set(record)
        if keys == {
            "classname", "entity_id", "nearest_distance_milliunits", "reason",
        }:
            _bounded_integer(
                record.get("nearest_distance_milliunits"),
                f"objective omission {ordinal} nearest_distance_milliunits",
                minimum=0,
                maximum=MAX_OBJECTIVE_NEAREST_DISTANCE_MILLIUNITS,
            )
        elif keys != {"classname", "entity_id", "reason"}:
            raise ClaimValidationError(
                f"objective omission {ordinal} keys differ from admitted shape"
            )
        entity_id = _bounded_integer(
            record.get("entity_id"), f"objective omission {ordinal} entity_id",
            minimum=0, maximum=0xFFFF_FFFF,
        )
        if entity_id in seen_ids or entity_id == previous_entity_id:
            raise ClaimValidationError(
                "objective guidepost omissions are not unique by entity_id"
            )
        if entity_id < previous_entity_id:
            raise ClaimValidationError(
                "objective guidepost omissions are not sorted by entity_id"
            )
        seen_ids.add(entity_id)
        previous_entity_id = entity_id
        classname = record.get("classname")
        if (
            not isinstance(classname, str)
            or re.fullmatch(r"[a-z0-9_]{1,128}", classname) is None
            or _objective_class_for_classname(classname) is None
        ):
            raise ClaimValidationError(
                f"objective omission {entity_id} classname is not canonical"
            )
        reason = record.get("reason")
        if reason not in OBJECTIVE_OMISSION_REASONS:
            raise ClaimValidationError(
                f"objective omission {entity_id} reason is not admitted"
            )
    return guideposts


def _bsp_objective_item_entities(bsp_path: Path) -> list[tuple[int, str]]:
    """Parse non-spawn objective-class item entities from a pinned BSP."""

    try:
        data = bsp_path.read_bytes()
        if len(data) > MAX_INPUT_BYTES:
            raise ClaimValidationError("BSP exceeds the admitted input byte limit")
        lumps = _parse_bsp_lumps(data)
        entities = _parse_bsp_entities(
            _bsp_lump_bytes(data, lumps, "entities"), BspLimits(),
        )
    except (BspValidationError, OSError, struct.error, UnicodeError) as error:
        raise ClaimValidationError(
            f"stock BSP entity parse failed for objective accounting: {error}"
        ) from error
    items: list[tuple[int, str]] = []
    for entity in entities:
        classname = entity.classname.casefold()
        objective_class = _objective_class_for_classname(classname)
        if objective_class is None or objective_class == "spawn_egress":
            continue
        items.append((entity.index, classname))
    items.sort(key=lambda item: item[0])
    ids = [entity_id for entity_id, _classname in items]
    if len(ids) != len(set(ids)):
        raise ClaimValidationError(
            "stock BSP objective-class item entity IDs are not unique"
        )
    return items


def _stock_objective_item_accounting(
    bsp_path: Path,
    analysis: Mapping[str, Any],
    analysis_path: Path,
) -> dict[str, Any]:
    """Prove emitted + omitted non-spawn items equal BSP objective-class items.

    Spawn-egress remains under compiled spawn gates and is excluded from this
    union so DM spawn accounting never pollutes item completeness.
    """

    failures: list[str] = []
    checked = 0
    try:
        compiled, _identity = _analysis_sections(analysis)
        guideposts = _validate_objective_guideposts(
            compiled.get("objective_guideposts"),
            require_zero_omissions=False,
        )
        map_id = bsp_path.stem
        objective_path = analysis_path.parent / f"{map_id}.objectives.json"
        grid = _mapping(analysis.get("grid"), "analysis grid")
        identity = _mapping(analysis.get("identity"), "analysis identity")
        document, _objective_count = _validate_objective_artifact(
            objective_path,
            map_id=map_id,
            bsp_sha256=identity.get("bsp_sha256"),
            atlas_sha256=identity.get("atlas_sha256"),
            origin=grid.get("origin"),
        )
        admitted = _integer(
            guideposts.get("admitted_count"),
            "objective guidepost admitted_count", minimum=0,
        )
        if admitted != _objective_count:
            failures.append(
                "objective guidepost admitted_count differs from objectives artifact"
            )

        emitted: dict[int, str] = {}
        for raw_record in _list(document.get("objectives"), "objective records"):
            record = _mapping(raw_record, "stock objective record")
            classname = record.get("classname")
            if not isinstance(classname, str):
                failures.append("stock objective classname is not a string")
                continue
            objective_class = _objective_class_for_classname(classname)
            if objective_class is None:
                failures.append(
                    f"stock objective {record.get('objective_id')} classname "
                    "is not an admitted objective class"
                )
                continue
            if objective_class == "spawn_egress":
                continue
            objective_id = _integer(
                record.get("objective_id"), "stock objective ID", minimum=0,
            )
            if objective_id in emitted:
                failures.append(
                    f"stock emitted objective ID {objective_id} is duplicated"
                )
                continue
            emitted[objective_id] = classname

        omitted: dict[int, str] = {}
        for ordinal, raw_record in enumerate(
            _list(guideposts.get("omissions"), "objective guidepost omissions")
        ):
            record = _mapping(raw_record, f"stock objective omission {ordinal}")
            entity_id = _integer(
                record.get("entity_id"),
                f"stock objective omission {ordinal} entity_id", minimum=0,
            )
            classname = record.get("classname")
            if not isinstance(classname, str):
                failures.append(
                    f"stock objective omission {entity_id} classname is not a string"
                )
                continue
            objective_class = _objective_class_for_classname(classname)
            if objective_class is None:
                failures.append(
                    f"stock objective omission {entity_id} classname is not admitted"
                )
                continue
            if objective_class == "spawn_egress":
                failures.append(
                    f"stock objective omission {entity_id} must not be spawn_egress"
                )
                continue
            if entity_id in omitted:
                failures.append(
                    f"stock objective omission ID {entity_id} is duplicated"
                )
                continue
            omitted[entity_id] = classname

        overlap = sorted(set(emitted) & set(omitted))
        if overlap:
            failures.append(
                "stock objective accounting is not disjoint for entity IDs: "
                + ",".join(str(value) for value in overlap)
            )

        bsp_items = _bsp_objective_item_entities(bsp_path)
        bsp_by_id = {entity_id: classname for entity_id, classname in bsp_items}
        checked = len(bsp_items)
        accounted = dict(emitted)
        for entity_id, classname in omitted.items():
            accounted[entity_id] = classname

        if set(accounted) != set(bsp_by_id):
            missing = sorted(set(bsp_by_id) - set(accounted))
            unknown = sorted(set(accounted) - set(bsp_by_id))
            if missing:
                failures.append(
                    "stock objective accounting dropped BSP item entity IDs: "
                    + ",".join(str(value) for value in missing)
                )
            if unknown:
                failures.append(
                    "stock objective accounting references unknown BSP item "
                    "entity IDs: "
                    + ",".join(str(value) for value in unknown)
                )
        classname_mismatches = sorted(
            entity_id for entity_id, classname in accounted.items()
            if entity_id in bsp_by_id and bsp_by_id[entity_id] != classname
        )
        if classname_mismatches:
            failures.append(
                "stock objective accounting classname differs for entity IDs: "
                + ",".join(str(value) for value in classname_mismatches)
            )
        if len(accounted) != len(bsp_by_id):
            failures.append(
                "stock objective accounting count differs from BSP objective-"
                f"class items ({len(accounted)} vs {len(bsp_by_id)})"
            )
    except ClaimValidationError as error:
        failures.append(str(error))
    return _criterion(sorted(set(failures)), checked)


def _full_cold_authority(
    analysis: Mapping[str, Any], analysis_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Validate evidence, local artifacts, hard budgets, and cold equality.

    The producer must publish both primary and cold digests plus elapsed time.
    Older self-declared ``deterministic_rebuild`` manifests are intentionally
    rejected; those booleans are not evidence.
    """

    failures: list[str] = []
    artifacts: Mapping[str, Any] = {}
    try:
        if analysis_path.read_bytes() != canonical_bytes(analysis):
            failures.append("analysis manifest is not canonical sorted JSON plus LF")
        artifacts = _mapping(analysis.get("artifacts"), "analysis artifacts")
        identity = _mapping(analysis.get("identity"), "analysis identity")
        counts = _mapping(analysis.get("counts"), "analysis counts")
        performance = _mapping(analysis.get("performance"), "analysis performance")
        proof = _mapping(
            performance.get("full_cold_rebuild"), "full-cold proof"
        )
        expected_proof_keys = {
            "schema", "independent_process_launches", "artifact_count",
            "artifact_sha256", "artifact_semantic_sha256",
            "cold_artifact_sha256", "cold_artifact_semantic_sha256",
            "verifier_sha256", "verification", "sample_interval_milliseconds",
            "sampled_process_tree_peak_rss_bytes", "peak_rss_limit_bytes",
            "elapsed_milliseconds", "timeout_limit_milliseconds",
        }
        actual_proof_keys = set(proof)
        if actual_proof_keys != expected_proof_keys:
            missing = sorted(expected_proof_keys - actual_proof_keys)
            extra = sorted(actual_proof_keys - expected_proof_keys)
            failures.append(
                "full-cold producer contract differs; analyzer follow-up must emit "
                f"primary/cold digest and elapsed-time evidence; missing={missing}, "
                f"extra={extra}"
            )
            return _criterion(failures, 0), artifacts
        if proof.get("schema") != "q2-atlas-full-cold-proof-v1":
            failures.append("full-cold proof schema differs")
        if proof.get("independent_process_launches") != 1:
            failures.append("full-cold proof did not use exactly one independent launch")
        if proof.get("artifact_count") != len(FULL_COLD_SUFFIXES):
            failures.append("full-cold proof does not cover eight artifacts")
        if proof.get("sample_interval_milliseconds") != 10:
            failures.append("full-cold RSS sample interval is not the frozen 10 ms")
        _digest(proof.get("verifier_sha256"), "full-cold verifier")

        exact = _mapping(proof.get("artifact_sha256"), "primary artifact digests")
        cold_exact = _mapping(
            proof.get("cold_artifact_sha256"), "cold artifact digests"
        )
        semantic = _mapping(
            proof.get("artifact_semantic_sha256"), "primary semantic digests"
        )
        cold_semantic = _mapping(
            proof.get("cold_artifact_semantic_sha256"), "cold semantic digests"
        )
        if set(exact) != set(FULL_COLD_EXACT_SUFFIXES):
            failures.append("primary exact digest set does not cover six byte-stable artifacts")
        if set(cold_exact) != set(FULL_COLD_EXACT_SUFFIXES):
            failures.append("cold exact digest set does not cover six byte-stable artifacts")
        if set(semantic) != set(FULL_COLD_SEMANTIC_SUFFIXES):
            failures.append(
                "primary semantic digest set does not cover Atlas and analysis manifests"
            )
        if set(cold_semantic) != set(FULL_COLD_SEMANTIC_SUFFIXES):
            failures.append(
                "cold semantic digest set does not cover Atlas and analysis manifests"
            )
        for label, values in (
            ("primary artifact", exact), ("cold artifact", cold_exact),
            ("primary semantic artifact", semantic),
            ("cold semantic artifact", cold_semantic),
        ):
            for suffix, digest in values.items():
                _digest(digest, f"{label} {suffix}")
        if exact != cold_exact or semantic != cold_semantic:
            failures.append("primary and independent-cold artifact hashes differ")

        elapsed = _integer(
            proof.get("elapsed_milliseconds"), "full-cold elapsed milliseconds", minimum=1
        )
        timeout = _integer(
            proof.get("timeout_limit_milliseconds"),
            "full-cold timeout limit milliseconds", minimum=1,
        )
        if timeout != MAX_FULL_COLD_MILLISECONDS:
            failures.append("full-cold timeout limit is not the frozen 300 seconds")
        if elapsed > MAX_FULL_COLD_MILLISECONDS:
            failures.append("full-cold elapsed time exceeds 300 seconds")
        peak_limit = _integer(
            proof.get("peak_rss_limit_bytes"), "full-cold RSS limit", minimum=1
        )
        peak = _integer(
            proof.get("sampled_process_tree_peak_rss_bytes"),
            "full-cold sampled process-tree RSS", minimum=1,
        )
        if peak_limit != MAX_BUILD_RSS_BYTES:
            failures.append("full-cold RSS limit is not the frozen 512 MiB")
        if peak > MAX_BUILD_RSS_BYTES:
            failures.append("full-cold process-tree RSS exceeds 512 MiB")

        map_id = analysis.get("canonical_map_id")
        if not isinstance(map_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", map_id):
            raise ClaimValidationError("analysis canonical_map_id is invalid")
        if map_id != analysis_path.name.removesuffix(".analysis.manifest.json"):
            failures.append("analysis filename differs from canonical map ID")
        base = analysis_path.parent / map_id
        for suffix in FULL_COLD_EXACT_SUFFIXES:
            path = Path(f"{base}{suffix}")
            if not path.is_file():
                failures.append(f"full-cold artifact is missing: {path.name}")
            elif exact.get(suffix) != file_sha256(path):
                failures.append(f"on-disk artifact digest differs: {path.name}")
        atlas_manifest_path = Path(f"{base}.atlas.manifest.json")
        if not atlas_manifest_path.is_file():
            failures.append(f"full-cold artifact is missing: {atlas_manifest_path.name}")
            return _criterion(failures, len(FULL_COLD_SUFFIXES)), artifacts
        elif semantic.get(".atlas.manifest.json") != _canonical_semantic_manifest(
            atlas_manifest_path
        ):
            failures.append("on-disk Atlas manifest semantic digest differs")
        analysis_semantic_digest = _canonical_semantic_analysis_manifest(analysis)
        if semantic.get(".analysis.manifest.json") != analysis_semantic_digest:
            failures.append("on-disk analysis manifest semantic digest differs")
        if cold_semantic.get(".analysis.manifest.json") != analysis_semantic_digest:
            failures.append(
                "independent-cold analysis-manifest semantic digest differs"
            )

        atlas_path = Path(f"{base}.atlas.bin")
        atlas_zst_path = Path(f"{base}.atlas.bin.zst")
        if atlas_path.is_file():
            l0_bytes, header_counts = _atlas_header(atlas_path)
            if l0_bytes > MAX_L0_DECOMPRESSED_BYTES:
                failures.append("Atlas L0 section exceeds 16 MiB")
            if atlas_path.stat().st_size > MAX_ATLAS_DECOMPRESSED_BYTES:
                failures.append("Atlas artifact exceeds 32 MiB decompressed")
            if header_counts[0] > MAX_L0_CHUNKS:
                failures.append("Atlas header exceeds 1200 L0 chunks")
        else:
            l0_bytes, header_counts = 0, (0, 0, 0, 0, 0)
        expected_count_names = (
            "l0_chunks", "l1_nodes", "l1_edges", "l2_cells", "l3_cells"
        )
        analysis_counts = tuple(
            _integer(counts.get(name), f"analysis {name}", minimum=0)
            for name in expected_count_names
        )
        if analysis_counts != header_counts:
            failures.append("analysis counts differ from Atlas header")
        if analysis_counts[0] > MAX_L0_CHUNKS:
            failures.append("analysis exceeds 1200 L0 chunks")

        objective_path = Path(f"{base}.objectives.json")
        grid = _mapping(analysis.get("grid"), "analysis grid")
        _, objective_count = _validate_objective_artifact(
            objective_path,
            map_id=map_id,
            bsp_sha256=identity.get("bsp_sha256"),
            atlas_sha256=identity.get("atlas_sha256"),
            origin=grid.get("origin"),
        )
        compiled_world = _mapping(analysis.get("compiled_world"), "compiled_world")
        _validate_objective_guideposts(
            compiled_world.get("objective_guideposts"),
            admitted_count=objective_count,
        )

        atlas = _mapping(artifacts.get("atlas"), "analysis Atlas artifact")
        if atlas_path.is_file() and atlas.get("uncompressed_sha256") != file_sha256(atlas_path):
            failures.append("analysis Atlas digest differs from artifact")
        if atlas_path.is_file() and atlas.get("uncompressed_bytes") != atlas_path.stat().st_size:
            failures.append("analysis Atlas byte count differs from artifact")
        if _integer(
            atlas.get("uncompressed_bytes"), "analysis Atlas decompressed bytes", minimum=0
        ) > MAX_ATLAS_DECOMPRESSED_BYTES:
            failures.append("analysis Atlas decompressed bytes exceed 32 MiB")
        if atlas_zst_path.is_file() and atlas.get("transport_sha256") != file_sha256(atlas_zst_path):
            failures.append("analysis Atlas transport digest differs")
        if atlas.get("compressed_sha256") != atlas.get("transport_sha256"):
            failures.append("analysis Atlas compressed/transport digests differ")
        _integer(
            atlas.get("compressed_bytes"), "analysis Atlas compressed bytes", minimum=0
        )
        for name, suffix, digest_name in (
            ("navigation", ".navigation.bin.zst", "transport_sha256"),
            ("visibility", ".visibility.bin.zst", "transport_sha256"),
            ("design_signature", ".design-signature.json", "sha256"),
            ("objectives", ".objectives.json", "sha256"),
        ):
            artifact = _mapping(artifacts.get(name), f"analysis {name} artifact")
            if artifact.get(digest_name) != exact.get(suffix):
                failures.append(f"analysis {name} digest differs from full-cold proof")
        objective_metadata = _mapping(
            artifacts.get("objectives"), "analysis objectives artifact"
        )
        _exact_keys(objective_metadata, {
            "count", "schema", "sha256", "uncompressed_bytes",
        }, "analysis objectives artifact")
        if (
            objective_metadata.get("schema") != OBJECTIVE_SCHEMA
            or objective_metadata.get("count") != objective_count
            or objective_metadata.get("uncompressed_bytes") != objective_path.stat().st_size
        ):
            failures.append("analysis objective metadata differs from artifact")
        resident = _integer(
            atlas.get("resident_bytes_estimate"), "Atlas resident estimate", minimum=0
        )
        if resident > MAX_ATLAS_RESIDENT_BYTES:
            failures.append("Atlas resident estimate exceeds 32 MiB")
        build_rss = _integer(
            atlas.get("build_peak_rss_bytes"), "Atlas packer peak RSS", minimum=1
        )
        if (
            atlas.get("build_peak_rss_gate_passed") is not True
            or atlas.get("max_build_rss_bytes") != MAX_BUILD_RSS_BYTES
            or build_rss > MAX_BUILD_RSS_BYTES
        ):
            failures.append("Atlas packer RSS proof does not pass 512 MiB")

        manifest = _mapping(load_json(atlas_manifest_path), "Atlas manifest")
        budgets = _mapping(manifest.get("budgets"), "Atlas manifest budgets")
        expected_budgets = {
            "max_l0_chunks": MAX_L0_CHUNKS,
            "max_l0_decompressed_bytes": MAX_L0_DECOMPRESSED_BYTES,
            "max_atlas_decompressed_bytes": MAX_ATLAS_DECOMPRESSED_BYTES,
            "max_atlas_resident_bytes": MAX_ATLAS_RESIDENT_BYTES,
            "max_build_rss_bytes": MAX_BUILD_RSS_BYTES,
        }
        if budgets != expected_budgets:
            failures.append("Atlas manifest budgets differ from frozen hard limits")
        if _integer(
            manifest.get("build_peak_rss_bytes"), "Atlas manifest peak RSS", minimum=1
        ) > MAX_BUILD_RSS_BYTES:
            failures.append("Atlas manifest packer RSS exceeds 512 MiB")
        if _mapping(manifest.get("counts"), "Atlas manifest counts") != {
            name: value for name, value in zip(expected_count_names, analysis_counts)
        }:
            failures.append("Atlas manifest counts differ from analysis")
        if _mapping(manifest.get("bsp"), "Atlas manifest BSP").get("sha256") != identity.get(
            "bsp_sha256"
        ):
            failures.append("Atlas manifest BSP differs from analysis")
        if _mapping(manifest.get("analyzer"), "Atlas manifest analyzer").get("sha256") != identity.get(
            "analyzer_sha256"
        ):
            failures.append("Atlas manifest analyzer identity differs")
        manifest_artifacts = _mapping(
            manifest.get("artifacts"), "Atlas manifest artifacts"
        )
        objective_identity = _mapping(
            manifest_artifacts.get(objective_path.name),
            "Atlas manifest objective identity",
        )
        _exact_keys(objective_identity, {
            "compressed_size", "counts", "media_type", "sha256_uncompressed",
            "uncompressed_size",
        }, "Atlas manifest objective identity")
        if (
            objective_identity.get("media_type") != OBJECTIVE_MEDIA_TYPE
            or objective_identity.get("sha256_uncompressed") != file_sha256(objective_path)
            or objective_identity.get("uncompressed_size") != objective_path.stat().st_size
            or objective_identity.get("compressed_size") != objective_path.stat().st_size
            or _mapping(
                objective_identity.get("counts"),
                "Atlas manifest objective counts",
            ) != {"objectives": objective_count}
        ):
            failures.append("Atlas manifest objective identity differs from artifact")
        objective_media_names = [
            name for name, raw_identity in manifest_artifacts.items()
            if isinstance(raw_identity, Mapping)
            and raw_identity.get("media_type") == OBJECTIVE_MEDIA_TYPE
        ]
        if objective_media_names != [objective_path.name]:
            failures.append(
                "Atlas manifest must attest exactly one authoritative objective artifact"
            )

        atlas_manifest_artifact = _mapping(
            artifacts.get("atlas_manifest"), "analysis Atlas-manifest artifact"
        )
        expected_manifest_name = f"{map_id}.atlas.manifest.json"
        if atlas_manifest_artifact.get("path") != expected_manifest_name:
            failures.append("analysis Atlas-manifest path differs")
        manifest_digest = file_sha256(atlas_manifest_path)
        if (
            atlas_manifest_artifact.get("sha256") != manifest_digest
            or identity.get("atlas_manifest_sha256") != manifest_digest
        ):
            failures.append("analysis Atlas-manifest identity differs")
        if proof.get("verifier_sha256") != atlas_manifest_artifact.get("verifier_sha256"):
            failures.append("full-cold verifier identity differs from analysis")
        verification = _mapping(proof.get("verification"), "full-cold verification")
        local_verification = dict(_mapping(
            atlas_manifest_artifact.get("verification"), "analysis Atlas verification"
        ))
        local_verification.pop("manifest_sha256", None)
        if verification != local_verification:
            failures.append("full-cold verifier result differs from primary verification")
        _digest(
            verification.get("collision_contract_sha256"),
            "full-cold collision contract",
        )
        expected_verification = {
            "schema": "q2-atlas-verification-v1",
            "passed": True,
            "canonical_map_id": map_id,
            "bsp_sha256": identity.get("bsp_sha256"),
            "artifact_name": f"{map_id}.atlas.bin",
            "atlas_sha256": identity.get("atlas_sha256"),
            "origin": analysis.get("grid", {}).get("origin") if isinstance(analysis.get("grid"), Mapping) else None,
            "counts": {name: value for name, value in zip(expected_count_names, analysis_counts)},
            "collision_contract_sha256": verification.get("collision_contract_sha256"),
        }
        if verification != expected_verification:
            failures.append("full-cold verifier summary differs from analysis identities")
    except (ClaimValidationError, OSError, struct.error, UnicodeError) as exc:
        failures.append(str(exc))
    return _criterion(failures, len(FULL_COLD_SUFFIXES)), artifacts


def _analysis_quality(
    analysis: Mapping[str, Any],
    *,
    bsp_sha256: str,
    claims_sha256: str | None,
    b1_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    compiled, identity = _analysis_sections(analysis)
    failures = []
    if analysis.get("status") != "passed":
        failures.append("analysis status is not passed")
    if identity.get("bsp_sha256") != bsp_sha256:
        failures.append("analysis BSP identity differs from compiled BSP")
    if identity.get("generator_claims_sha256") != claims_sha256:
        failures.append("analysis generator-claims identity differs")
    try:
        _digest(identity.get("atlas_sha256"), "analysis Atlas")
        _digest(identity.get("analyzer_sha256"), "analysis analyzer")
        if identity.get("analyzer_sha256") != _expected_analyzer_sha256():
            failures.append(
                "analysis analyzer identity differs from the local admitted source closure"
            )
    except (ClaimValidationError, OSError) as exc:
        failures.append(str(exc))
    try:
        oracles = _mapping(analysis.get("oracles"), "analysis oracles")
        failures.extend(
            _oracle_record(oracles.get("collision"), "collision oracle")
        )
        collision = _mapping(oracles.get("collision"), "collision oracle")
        b1_collision = _mapping(
            _mapping(b1_gate.get("artifacts"), "B1 artifacts").get(
                "transformed_inline_collision"
            ),
            "B1 transformed collision",
        )
        expected_collision_sha256 = _digest(
            b1_collision.get("cm_oracle_sha256"), "B1 collision executable"
        )
        if collision.get("executable_sha256") != expected_collision_sha256:
            failures.append(
                "collision oracle executable differs from the accepted B1 authority"
            )
    except ClaimValidationError as exc:
        failures.append(str(exc))
    if analysis.get("deterministic_rebuild") is not True:
        failures.append("Atlas deterministic cold rebuild did not pass")
    confidence = analysis.get("confidence")
    if confidence not in {"high", "complete"}:
        failures.append("analysis confidence is not complete/high")
    try:
        _validate_objective_guideposts(
            compiled.get("objective_guideposts"),
            require_zero_omissions=claims_sha256 is not None,
        )
    except ClaimValidationError as exc:
        failures.append(str(exc))
    return _criterion(failures, 9), compiled


def _validate_spawns(
    claims: Mapping[str, Any], compiled: Mapping[str, Any]
) -> dict[str, Any]:
    failures = []
    expected = {
        tuple(spawn["origin_milliunits"])
        for spawn in claims["spawns"]
    }
    records = [_mapping(item, "compiled spawn") for item in _list(compiled.get("spawns"), "compiled spawns")]
    expected_keys = {
        "entity_ordinal", "origin_milliunits", "standing_clear",
        "crouched_clear", "supported", "column_clearance_milliunits",
        "column_clear_96", "l1_index", "region_id", "escape_edge_count",
        "reachable_spawn_ordinals", "cost_to_safety_q8",
    }
    for item in records:
        _exact_keys(item, expected_keys, "compiled spawn")
        _vec3_int(item.get("l1_index"), "compiled spawn L1 index")
        _integer(item.get("region_id"), "compiled spawn region", minimum=0)
    actual = {tuple(_vec3_int(item.get("origin_milliunits"), "compiled spawn origin")) for item in records}
    if actual != expected or len(records) != len(expected):
        failures.append("compiled spawn origins differ from generator claims")
    ordinals = {_integer(item.get("entity_ordinal"), "spawn entity ordinal", minimum=0) for item in records}
    if len(ordinals) != len(records):
        failures.append("compiled spawn entity ordinals are not unique")
    for item in records:
        ordinal = item.get("entity_ordinal")
        if item.get("standing_clear") is not True or item.get("crouched_clear") is not True:
            failures.append(f"spawn {ordinal} lacks compiled stance clearance")
        if item.get("supported") is not True:
            failures.append(f"spawn {ordinal} lacks compiled support")
        if item.get("column_clear_96") is not True or _integer(
            item.get("column_clearance_milliunits"),
            f"spawn {ordinal} column clearance",
            minimum=0,
        ) < 96 * MILLIUNITS:
            failures.append(f"spawn {ordinal} lacks a compiled 96-unit column")
        if _integer(item.get("escape_edge_count"), f"spawn {ordinal} escape edges", minimum=0) <= 0:
            failures.append(f"spawn {ordinal} has no compiled escape edge")
        reachable = set(_list(item.get("reachable_spawn_ordinals"), f"spawn {ordinal} reachable spawns"))
        if reachable != ordinals - {ordinal}:
            failures.append(f"spawn {ordinal} is not mutually connected to all other spawns")
        cost = _integer(item.get("cost_to_safety_q8"), f"spawn {ordinal} safety cost", minimum=0)
        if cost >= INFINITE_COST_Q8:
            failures.append(f"spawn {ordinal} has infinite safety cost")
    origins = sorted(actual)
    if len(origins) > 1:
        minimum_xy = min(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for index, a in enumerate(origins)
            for b in origins[index + 1 :]
        )
        if minimum_xy < MIN_SPAWN_SEPARATION * MILLIUNITS:
            failures.append("compiled spawn separation is below generator-v6 minimum")
    return _criterion(failures, len(records))


def _validated_claim_results(
    results: object,
    expected_claims: Sequence[Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    failures = []
    records = [_mapping(item, f"{label} result") for item in _list(results, f"compiled {label}")]
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        claim_id = _claim_id(record.get("claim_id"), f"{label} result claim_id")
        if claim_id in by_id:
            failures.append(f"duplicate compiled {label} result {claim_id}")
        by_id[claim_id] = record
    expected_ids = [str(item["claim_id"]) for item in expected_claims]
    if sorted(by_id) != expected_ids or len(records) != len(expected_ids):
        failures.append(f"compiled {label} results do not exactly cover claims")
    return by_id, failures


def _validate_hazards(
    claims: Mapping[str, Any], compiled: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = claims["hazard_claims"]
    by_id, failures = _validated_claim_results(
        compiled.get("hazard_claims"), expected, "hazard claims"
    )
    for claim in expected:
        result = by_id.get(claim["claim_id"])
        if result is None:
            continue
        _exact_keys(
            result,
            {
                "claim_id", "type", "raw_l0_cells", "expanded_l0_cells",
                "contained", "status", "evidence", "validation_version",
            },
            "compiled hazard claim",
        )
        if result.get("type") != claim["type"]:
            failures.append(f"{claim['claim_id']} hazard type differs")
        if result.get("status") != ORACLE_STATUS or result.get("contained") is not True:
            failures.append(f"{claim['claim_id']} lacks positive oracle containment")
        if _integer(result.get("raw_l0_cells"), "raw hazard cells", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no raw hazard cells")
        if _integer(result.get("expanded_l0_cells"), "expanded hazard cells", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no hull-expanded hazard cells")
        if _integer(result.get("evidence"), "hazard evidence", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no oracle evidence")
        if _integer(result.get("validation_version"), "hazard validation version", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no validation version")

    hazards = _mapping(compiled.get("hazards"), "compiled hazards")
    _exact_keys(
        hazards,
        {
            "l0_raw_cells", "l0_expanded_cells", "types",
            "lethal_drop_edges", "exact_lethal_candidates_omitted",
            "guarded_drop_edges",
            "uncontained_drop_edges", "classification_status", "evidence",
            "validation_version", "drop_classification",
        },
        "compiled hazards",
    )
    types = set(_list(hazards.get("types"), "compiled hazard types"))
    claimed_types = {claim["type"] for claim in expected}
    if not claimed_types.issubset(types):
        failures.append("aggregate Atlas hazards omit a claimed hazard type")
    if _integer(hazards.get("l0_raw_cells"), "aggregate raw hazard cells", minimum=0) <= 0:
        failures.append("aggregate Atlas has no raw hazard cells")
    if _integer(hazards.get("l0_expanded_cells"), "aggregate expanded hazard cells", minimum=0) <= 0:
        failures.append("aggregate Atlas has no expanded hazard cells")
    if hazards.get("classification_status") != ORACLE_STATUS:
        failures.append("aggregate Atlas hazard classification is not oracle-authoritative")
    if _integer(hazards.get("evidence"), "aggregate hazard evidence", minimum=0) <= 0:
        failures.append("aggregate Atlas hazard classification has no evidence")
    if _integer(
        hazards.get("validation_version"), "aggregate hazard validation version", minimum=0
    ) <= 0:
        failures.append("aggregate Atlas hazard classification has no validation version")
    drop = _mapping(hazards.get("drop_classification"), "exact drop classification")
    _exact_keys(drop, {
        "classification_status", "evidence", "validation_version",
        "candidate_count", "exact_safe", "exact_lethal", "unknown_omitted",
        "unknown_reason_counts", "severity_counts",
    }, "exact drop classification")
    if drop.get("classification_status") != ORACLE_STATUS:
        failures.append("drop classification is not oracle-authoritative")
    drop_evidence = _integer(drop.get("evidence"), "drop evidence", minimum=0)
    drop_version = _integer(
        drop.get("validation_version"), "drop validation version", minimum=0,
    )
    if drop_evidence not in {0, 2, 10}:
        failures.append("drop classification evidence bits differ from admitted stages")
    if (drop_evidence == 0) != (drop_version == 0):
        failures.append("drop validation version does not match admitted evidence")
    if drop_evidence and drop_version != 1:
        failures.append("drop classification validation version differs")
    drop_counts = [
        _integer(drop.get(name), f"drop {name}", minimum=0)
        for name in ("exact_safe", "exact_lethal", "unknown_omitted")
    ]
    if sum(drop_counts) != _integer(
        drop.get("candidate_count"), "drop candidate count", minimum=0
    ):
        failures.append("drop classification counts do not cover every candidate")
    _validate_drop_unknown_reasons(drop, "drop classification", failures)
    exact_count = drop_counts[0] + drop_counts[1]
    if exact_count and drop_evidence != 10:
        failures.append("exact drop classifications lack Pmove+fall evidence")
    if not drop_evidence and exact_count:
        failures.append("zero-evidence drop summary contains exact candidates")
    if _integer(
        hazards.get("exact_lethal_candidates_omitted"),
        "exact lethal candidates omitted", minimum=0,
    ) != drop_counts[1]:
        failures.append("omitted exact-lethal candidate count differs from replay summary")
    severity_counts = _mapping(drop.get("severity_counts"), "drop severity counts")
    if not set(severity_counts).issubset({"none", "footstep", "short", "fall", "far"}):
        failures.append("drop severity classes differ from fall oracle")
    for name, count in severity_counts.items():
        _integer(count, f"drop severity {name}", minimum=0)

    meta = claims.get("_meta")
    safety_failures = []
    expected_edges = 0
    if isinstance(meta, Mapping):
        safety = _mapping(meta.get("safety"), "generator safety")
        expected_edges = len(_list(safety.get("lethal_edges"), "lethal edges"))
    lethal = _integer(hazards.get("lethal_drop_edges"), "lethal drop edges", minimum=0)
    guarded = _integer(hazards.get("guarded_drop_edges"), "guarded drop edges", minimum=0)
    uncontained = _integer(hazards.get("uncontained_drop_edges"), "uncontained drop edges", minimum=0)
    if expected_edges and lethal != expected_edges:
        safety_failures.append("compiled lethal-edge count differs from generator safety contract")
    if lethal <= 0 or guarded != lethal or uncontained != 0:
        safety_failures.append("compiled lethal exterior is not completely guarded")
    return (
        _criterion(failures, len(expected)),
        _criterion(safety_failures, lethal),
    )


def _validate_lighting(
    claims: Mapping[str, Any], compiled: Mapping[str, Any]
) -> dict[str, Any]:
    lighting = _mapping(compiled.get("lighting"), "compiled lighting")
    _exact_keys(
        lighting,
        {
            "lightdata_bytes", "lightdata_sha256", "lightmapped_faces",
            "floor_light_region_count", "floor_light_region_ids",
            "spawn_nav_region_count", "dark_spawns",
        },
        "compiled lighting",
    )
    failures = []
    lightdata = _integer(lighting.get("lightdata_bytes"), "compiled lightdata bytes", minimum=0)
    if lightdata <= 0 or lightdata > 0x200000:
        failures.append("compiled qrad lightdata is empty or exceeds the v6 limit")
    try:
        _digest(lighting.get("lightdata_sha256"), "compiled lightdata")
    except ClaimValidationError as exc:
        failures.append(str(exc))
    if _integer(lighting.get("lightmapped_faces"), "lightmapped faces", minimum=0) <= 0:
        failures.append("compiled BSP has no lightmapped faces")
    meta = claims.get("_meta")
    expected_region_ids: list[str] = []
    if isinstance(meta, Mapping):
        expected_regions = [
            _mapping(item, "generator light region")
            for item in _list(
                _mapping(meta.get("lighting"), "generator lighting").get("regions"),
                "light regions",
            )
        ]
        for region in expected_regions:
            region_id = region.get("id")
            if not isinstance(region_id, str) or not region_id:
                raise ClaimValidationError(
                    "generator light region ID must be a nonempty string"
                )
            expected_region_ids.append(region_id)
        if len(expected_region_ids) != len(set(expected_region_ids)):
            raise ClaimValidationError(
                "generator light region IDs are not unique"
            )
        # Metadata regions are deterministically ordered by spatial bounds,
        # whereas compiled entity tags are canonically ordered by ID.  Compare
        # the exact sets in one shared representation without conflating the
        # two legitimate array-order contracts.
        expected_region_ids.sort()
    actual_region_ids = _list(
        lighting.get("floor_light_region_ids"),
        "compiled floor-light region IDs",
    )
    if not all(isinstance(item, str) and item for item in actual_region_ids):
        raise ClaimValidationError(
            "compiled floor-light region IDs must be nonempty strings"
        )
    if actual_region_ids != sorted(set(actual_region_ids)):
        failures.append("compiled floor-light region IDs are not canonical")
    floor_regions = _integer(
        lighting.get("floor_light_region_count"),
        "compiled floor-light regions", minimum=0,
    )
    if floor_regions != len(actual_region_ids):
        failures.append("compiled floor-light region count is inconsistent")
    if expected_region_ids and floor_regions != len(expected_region_ids):
        failures.append("compiled floor-light region count differs from v6 contract")
    if expected_region_ids and actual_region_ids != expected_region_ids:
        failures.append("compiled floor-light region IDs differ from v6 contract")

    spawn_records = [
        _mapping(item, "compiled spawn")
        for item in _list(compiled.get("spawns"), "compiled spawns")
    ]
    spawn_by_ordinal = {
        _integer(item.get("entity_ordinal"), "spawn entity ordinal", minimum=0): item
        for item in spawn_records
    }
    expected_nav_regions = {
        _integer(item.get("region_id"), "compiled spawn region", minimum=0)
        for item in spawn_records
        if item.get("region_id") != 0
    }
    spawn_nav_regions = _integer(
        lighting.get("spawn_nav_region_count"),
        "compiled spawn navigation regions", minimum=0,
    )
    if spawn_nav_regions != len(expected_nav_regions):
        failures.append("compiled spawn navigation region count is inconsistent")

    dark_spawns = [
        _mapping(item, "dark spawn diagnostic")
        for item in _list(lighting.get("dark_spawns"), "dark spawn diagnostics")
    ]
    dark_ordinals = []
    for record in dark_spawns:
        _exact_keys(
            record,
            {
                "entity_ordinal", "nav_region_id",
                "eligible_light_entity_ordinals",
            },
            "dark spawn diagnostic",
        )
        ordinal = _integer(
            record.get("entity_ordinal"), "dark spawn entity ordinal", minimum=0,
        )
        nav_region = _integer(
            record.get("nav_region_id"), "dark spawn navigation region", minimum=1,
        )
        eligible = [
            _integer(item, "eligible light entity ordinal", minimum=0)
            for item in _list(
                record.get("eligible_light_entity_ordinals"),
                "eligible light entity ordinals",
            )
        ]
        if eligible != sorted(set(eligible)):
            failures.append(
                f"dark spawn {ordinal} eligible light ordinals are not canonical"
            )
        spawn = spawn_by_ordinal.get(ordinal)
        if spawn is None:
            failures.append(f"dark spawn {ordinal} is not a compiled spawn")
        elif nav_region != spawn.get("region_id"):
            failures.append(
                f"dark spawn {ordinal} navigation region is inconsistent"
            )
        dark_ordinals.append(ordinal)
        failures.append(f"compiled Atlas reports dark spawn {ordinal}")
    if dark_ordinals != sorted(set(dark_ordinals)):
        failures.append("dark spawn diagnostics are not canonical")
    return _criterion(failures, floor_regions)


def _validate_hooks(
    claims: Mapping[str, Any], compiled: Mapping[str, Any],
    analysis: Mapping[str, Any], b1_gate: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    hooks = _mapping(compiled.get("hooks"), "compiled hooks")
    _exact_keys(
        hooks,
        {"authority_admitted", "omission_reason", "edges"},
        "compiled hooks",
    )
    expected = claims["hook_claims"]
    failures = []
    if hooks.get("authority_admitted") is not True or hooks.get("omission_reason") not in (None, ""):
        failures.append("compiled hook authority is absent/omitted")
    by_id, coverage_failures = _validated_claim_results(
        hooks.get("edges"), expected, "hook claims"
    )
    failures.extend(coverage_failures)
    b1_hook = _mapping(
        _mapping(b1_gate.get("artifacts"), "B1 artifacts").get(
            "hook_parity_attestation"
        ),
        "B1 hook attestation",
    )
    expected_physics = b1_hook.get("hook_physics_identity")
    expected_attestation = b1_hook.get("sha256")
    _digest(expected_physics, "B1 hook physics identity")
    _digest(expected_attestation, "B1 hook parity attestation")
    try:
        analysis_hook = _mapping(
            _mapping(analysis.get("oracles"), "analysis oracles").get("hook"),
            "analysis hook oracle",
        )
        if analysis_hook.get("authority_admitted") is not True:
            failures.append("analysis hook authority is not admitted")
        if analysis_hook.get("attestation_sha256") != expected_attestation:
            failures.append("analysis hook parity digest differs from accepted B1 attestation")
        materialized_oracles = _mapping(
            materialization.get("oracles"), "hook materialization oracles"
        )
        if materialized_oracles.get("hook_parity_attestation_sha256") != expected_attestation:
            failures.append("hook claims carry a non-B1 parity attestation digest")
    except ClaimValidationError as exc:
        failures.append(str(exc))
    for claim in expected:
        edge = by_id.get(claim["claim_id"])
        if edge is None:
            continue
        _exact_keys(
            edge,
            {
                "claim_id", "source_l1", "target_l1", "source_milliunits",
                "trace_target_milliunits", "measured_anchor_milliunits",
                "landing_milliunits",
                "release_after_ticks", "distance_milliunits", "flags",
                "trajectory_origin_fixed", "trajectory_sha256",
                "first_grounded_frame_index",
                "landing_l1", "physics_identity", "evidence", "validation_version",
            },
            "compiled hook edge",
        )
        for field in (
            "source_milliunits", "trace_target_milliunits",
            "measured_anchor_milliunits", "landing_milliunits",
        ):
            if tuple(_vec3_int(edge.get(field), f"hook edge {field}")) != tuple(claim[field]):
                failures.append(f"{claim['claim_id']} {field} differs")
        for field in ("release_after_ticks", "distance_milliunits", "flags"):
            if _integer(edge.get(field), f"hook edge {field}", minimum=0) != claim[field]:
                failures.append(f"{claim['claim_id']} {field} differs")
        frames = _list(edge.get("trajectory_origin_fixed"), "hook trajectory")
        if not frames:
            failures.append(f"{claim['claim_id']} has no Pmove trajectory")
        for frame in frames:
            _vec3_int(frame, "hook trajectory frame")
        _digest(edge.get("trajectory_sha256"), "hook trajectory digest")
        if _integer(
            edge.get("first_grounded_frame_index"),
            "hook first grounded frame", minimum=0,
        ) != len(frames) - 1:
            failures.append(f"{claim['claim_id']} trajectory does not end grounded")
        if validation_trace_sha256(
            claim["claim_id"], frames, len(frames) - 1
        ) != edge.get("trajectory_sha256"):
            failures.append(f"{claim['claim_id']} trajectory digest differs")
        if edge.get("physics_identity") != expected_physics:
            failures.append(f"{claim['claim_id']} physics identity differs from B1")
        if _integer(edge.get("evidence"), "hook evidence", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no hook evidence")
        if _integer(edge.get("validation_version"), "hook validation version", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no hook validation version")
        landing_l1 = _vec3_int(edge.get("landing_l1"), "hook landing L1")
        _vec3_int(edge.get("source_l1"), "hook source L1")
        target_l1 = _vec3_int(edge.get("target_l1"), "hook target L1")
        if landing_l1 != target_l1:
            failures.append(f"{claim['claim_id']} target is not its legal landing")
    return _criterion(failures, len(expected))


def _validate_routes(claims: Mapping[str, Any], compiled: Mapping[str, Any]) -> dict[str, Any]:
    expected = claims["route_claims"]
    by_id, failures = _validated_claim_results(
        compiled.get("route_claims"), expected, "route claims"
    )
    for claim in expected:
        result = by_id.get(claim["claim_id"])
        if result is None:
            continue
        _exact_keys(
            result,
            {
                "claim_id", "source_milliunits", "target_milliunits",
                "source_l1", "target_l1", "connected", "cost_q8", "status",
                "evidence", "validation_version",
            },
            "compiled route claim",
        )
        _vec3_int(result.get("source_l1"), "route result source L1")
        _vec3_int(result.get("target_l1"), "route result target L1")
        if result.get("status") != ORACLE_STATUS or result.get("connected") is not True:
            failures.append(f"{claim['claim_id']} lacks oracle connectivity")
        if tuple(_vec3_int(result.get("source_milliunits"), "route result source")) != tuple(claim["source_milliunits"]):
            failures.append(f"{claim['claim_id']} source differs")
        if tuple(_vec3_int(result.get("target_milliunits"), "route result target")) != tuple(claim["target_milliunits"]):
            failures.append(f"{claim['claim_id']} target differs")
        cost = _integer(result.get("cost_q8"), "route result cost", minimum=0)
        if cost >= INFINITE_COST_Q8:
            failures.append(f"{claim['claim_id']} has infinite route cost")
        if _integer(result.get("evidence"), "route evidence", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no route evidence")
        if _integer(result.get("validation_version"), "route validation version", minimum=0) <= 0:
            failures.append(f"{claim['claim_id']} has no route validation version")

    for route in claims["routes"]:
        costs = [
            int(by_id[claim_id]["cost_q8"])
            for claim_id in route["segment_claim_ids"]
            if claim_id in by_id and isinstance(by_id[claim_id].get("cost_q8"), int)
        ]
        if len(costs) != len(route["segment_claim_ids"]):
            continue
        compiled_cost = sum(costs)
        claimed_cost = int(route["claimed_cost_q8"])
        difference = abs(compiled_cost - claimed_cost)
        ratio = max(compiled_cost, claimed_cost) / max(1, min(compiled_cost, claimed_cost))
        if difference > MAX_ROUTE_COST_ABSOLUTE_ERROR_Q8 and ratio > MAX_ROUTE_COST_RATIO:
            failures.append(
                f"{route['route_id']} compiled cost differs excessively from generator claim"
            )
    return _criterion(failures, len(expected))


def _b1_gate(path: Path) -> tuple[dict[str, Any], str]:
    gate = _mapping(load_json(path), "B1 gate")
    if gate.get("batch") != "B1" or gate.get("status") != "green":
        raise ClaimValidationError("B1 gate is not green")
    invariants = _mapping(gate.get("admission_invariants"), "B1 admission invariants")
    for name in (
        "collision_failure_rejects_build",
        "hook_or_parity_absence_forbids_hook_edges",
        "all_edges_require_nonzero_evidence_and_validation_version",
    ):
        if invariants.get(name) is not True:
            raise ClaimValidationError(f"B1 invariant {name} is not green")
    return dict(gate), file_sha256(path)


def validate_generated_map(
    map_path: Path,
    analysis_path: Path,
    *,
    b1_gate_path: Path = ROOT / "docs" / "multires" / "B1-GATE.json",
) -> dict[str, Any]:
    claims = build_generator_claims(map_path)
    # Private context is never serialized into the claims contract.
    claims_context = dict(claims)
    claims_context["_meta"] = load_json(map_path.with_suffix(".meta.json"))
    claims_digest = generator_claims_sha256(claims)
    bsp_path = map_path.with_suffix(".bsp")
    if not bsp_path.is_file():
        raise ClaimValidationError("compiled BSP is required for generated promotion")
    bsp_digest = file_sha256(bsp_path)
    analysis = _mapping(load_json(analysis_path), "compiled analysis")
    b1_gate, b1_gate_digest = _b1_gate(b1_gate_path)
    materialization_path = Path(
        f"{map_path.with_suffix('')}.hook-materialization.json"
    )
    try:
        materialization, _materialization_digest = load_materialization(
            materialization_path
        )
    except (HookClaimsV4Error, OSError) as error:
        raise ClaimValidationError(str(error)) from error

    analysis_quality, compiled = _analysis_quality(
        analysis, bsp_sha256=bsp_digest, claims_sha256=claims_digest,
        b1_gate=b1_gate,
    )
    artifact_authority, _artifacts = _full_cold_authority(
        analysis, analysis_path
    )
    static_result = static_validate(map_path, _static_args())
    static_failures = [] if static_result.get("static_ok") is True else [
        "existing generator-v6 source/static validation failed"
    ]
    criteria = {
        "analysis_quality": analysis_quality,
        "artifact_authority": artifact_authority,
        "source_static_v6": _criterion(static_failures, 1),
        "spawns": _validate_spawns(claims_context, compiled),
    }
    hazards, lethal = _validate_hazards(claims_context, compiled)
    criteria.update(
        {
            "hazards": hazards,
            "lethal_containment": lethal,
            "lighting": _validate_lighting(claims_context, compiled),
            "hooks": _validate_hooks(
                claims_context, compiled, analysis, b1_gate, materialization
            ),
            "routes": _validate_routes(claims_context, compiled),
        }
    )
    failures = sorted(
        f"{name}: {failure}"
        for name, criterion in criteria.items()
        for failure in criterion["failures"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "mode": "generated_v6_promotion",
        "map": map_path.stem,
        "passed": not failures,
        "identities": {
            "b1_gate_sha256": b1_gate_digest,
            "bsp_sha256": bsp_digest,
            "analysis_sha256": file_sha256(analysis_path),
            "generator_claims_sha256": claims_digest,
            **claims["source_files"],
        },
        "criteria": criteria,
        "failures": failures,
    }
    validate_report(report)
    return report


def validate_stock_analysis(
    bsp_path: Path,
    analysis_path: Path,
    *,
    b1_gate_path: Path = ROOT / "docs" / "multires" / "B1-GATE.json",
    stock_provenance_path: Path = STOCK_PROVENANCE,
    stock_inventory_path: Path = STOCK_INVENTORY,
) -> dict[str, Any]:
    """Validate authored-map analysis quality without generator-v6 rules."""

    bsp_digest = file_sha256(bsp_path)
    analysis = _mapping(load_json(analysis_path), "compiled analysis")
    b1_gate, b1_gate_digest = _b1_gate(b1_gate_path)
    quality, compiled = _analysis_quality(
        analysis, bsp_sha256=bsp_digest, claims_sha256=None, b1_gate=b1_gate,
    )
    artifact_authority, _artifacts = _full_cold_authority(
        analysis, analysis_path
    )
    provenance_digest = file_sha256(stock_provenance_path)
    inventory_digest = file_sha256(stock_inventory_path)
    provenance_failures: list[str] = []
    structural_failures: list[str] = []
    provenance = _mapping(load_json(stock_provenance_path), "stock provenance")
    inventory = _mapping(load_json(stock_inventory_path), "stock inventory")
    if provenance.get("schema") != "q2-corpus-provenance-v1":
        provenance_failures.append("stock provenance schema differs")
    if inventory.get("schema") != "q2-stock-map-fixtures-v1":
        structural_failures.append("stock structural-inventory schema differs")
    provenance_records = {
        record.get("canonical_id"): record
        for record in (
            _mapping(value, "stock provenance record")
            for value in _list(provenance.get("records"), "stock provenance records")
        )
    }
    inventory_records = {
        record.get("canonical_id"): record
        for record in (
            _mapping(value, "stock inventory record")
            for value in _list(inventory.get("maps"), "stock inventory maps")
        )
    }
    expected_map_ids = {f"q2dm{number}" for number in range(1, 9)}
    if set(provenance_records) != expected_map_ids:
        provenance_failures.append("stock provenance does not pin exactly q2dm1-q2dm8")
    if set(inventory_records) != expected_map_ids:
        structural_failures.append("stock inventory does not pin exactly q2dm1-q2dm8")
    map_id = bsp_path.stem
    expected_provenance = provenance_records.get(map_id)
    expected_inventory = inventory_records.get(map_id)
    if expected_provenance is None or expected_inventory is None:
        provenance_failures.append(f"{map_id} is not a pinned stock-map identity")
        expected_spawn_count = -1
        expected_items: Mapping[str, Any] = {}
    else:
        if expected_provenance.get("bsp_sha256") != bsp_digest:
            provenance_failures.append("stock BSP differs from pinned provenance")
        if expected_inventory.get("bsp_sha256") != bsp_digest:
            structural_failures.append("stock BSP differs from pinned structural inventory")
        expected_spawn_count = _integer(
            expected_inventory.get("deathmatch_spawn_count"),
            "pinned stock deathmatch-spawn count", minimum=2,
        )
        expected_items = _mapping(
            expected_inventory.get("item_classes"), "pinned stock item-class multiset"
        )

    failures = list(quality["failures"])
    spawns = [_mapping(item, "compiled stock spawn") for item in _list(compiled.get("spawns"), "compiled stock spawns")]
    if len(spawns) != expected_spawn_count:
        structural_failures.append(
            "compiled stock deathmatch-spawn count differs from pinned inventory"
        )
    reachable_by_ordinal: dict[int, set[int]] = {}
    for item in spawns:
        if item.get("standing_clear") is not True or item.get("supported") is not True:
            failures.append(f"stock spawn {item.get('entity_ordinal')} is not clear/supported")
        ordinal = _integer(
            item.get("entity_ordinal"), "stock spawn entity ordinal", minimum=0,
        )
        reachable_by_ordinal[ordinal] = {
            _integer(value, "stock reachable spawn ordinal", minimum=0)
            for value in _list(
                item.get("reachable_spawn_ordinals"), "stock reachable spawns",
            )
        }
    if not any(
        right in reachable_by_ordinal[left]
        and left in reachable_by_ordinal[right]
        for left in sorted(reachable_by_ordinal)
        for right in sorted(reachable_by_ordinal)
        if left < right
    ):
        failures.append("stock map has fewer than two mutually reachable spawns")
    design_path = analysis_path.parent / f"{map_id}.design-signature.json"
    if not design_path.is_file():
        structural_failures.append("stock design signature is missing")
    else:
        design = _mapping(load_json(design_path), "stock design signature")
        design_counts = _mapping(design.get("counts"), "stock design counts")
        if design_counts.get("deathmatch_spawns") != expected_spawn_count:
            structural_failures.append(
                "design-signature spawn count differs from pinned inventory"
            )
        if _mapping(
            design.get("item_class_multiset"), "stock design item-class multiset"
        ) != expected_items:
            structural_failures.append(
                "design-signature item-class multiset differs from pinned inventory"
            )

    hazard_failures: list[str] = []
    hazards = _mapping(compiled.get("hazards"), "compiled stock hazards")
    required_hazard_keys = {
        "l0_raw_cells", "l0_expanded_cells", "types", "lethal_drop_edges",
        "exact_lethal_candidates_omitted",
        "guarded_drop_edges", "uncontained_drop_edges", "classification_status",
        "evidence", "validation_version", "drop_classification",
    }
    if not required_hazard_keys.issubset(hazards):
        hazard_failures.append(
            "stock hazard evidence is incomplete; analyzer follow-up must emit "
            "classification_status, evidence, and validation_version"
        )
    else:
        types = _list(hazards.get("types"), "compiled stock hazard types")
        if types != sorted(set(types)) or not set(types).issubset(
            {"lava", "slime", "hurt", "void", "crush", "current"}
        ):
            hazard_failures.append("compiled stock hazard types are not canonical")
        for name in (
            "l0_raw_cells", "l0_expanded_cells", "lethal_drop_edges",
            "exact_lethal_candidates_omitted", "guarded_drop_edges",
            "uncontained_drop_edges",
        ):
            _integer(hazards.get(name), f"compiled stock {name}", minimum=0)
        if hazards.get("classification_status") != ORACLE_STATUS:
            hazard_failures.append("stock hazard classification is not oracle-authoritative")
        if _integer(hazards.get("evidence"), "stock hazard evidence", minimum=0) <= 0:
            hazard_failures.append("stock hazard classification has no evidence")
        if _integer(
            hazards.get("validation_version"), "stock hazard validation version", minimum=0
        ) <= 0:
            hazard_failures.append("stock hazard classification has no validation version")
        drop = _mapping(
            hazards.get("drop_classification"), "compiled stock drop classification"
        )
        required_drop_keys = {
            "classification_status", "evidence", "validation_version",
            "candidate_count", "exact_safe", "exact_lethal", "unknown_omitted",
            "unknown_reason_counts", "severity_counts",
        }
        if set(drop) != required_drop_keys:
            hazard_failures.append("stock drop classification contract differs")
        else:
            stock_drop_evidence = _integer(
                drop.get("evidence"), "stock drop evidence", minimum=0,
            )
            stock_drop_version = _integer(
                drop.get("validation_version"),
                "stock drop validation version", minimum=0,
            )
            if drop.get("classification_status") != ORACLE_STATUS:
                hazard_failures.append("stock drop classification is not oracle-authoritative")
            if stock_drop_evidence not in {0, 2, 10}:
                hazard_failures.append("stock drop evidence bits differ from admitted stages")
            if (stock_drop_evidence == 0) != (stock_drop_version == 0):
                hazard_failures.append("stock drop evidence/version relation differs")
            if stock_drop_evidence and stock_drop_version != 1:
                hazard_failures.append("stock drop validation version differs")
            stock_counts = [
                _integer(drop.get(name), f"stock drop {name}", minimum=0)
                for name in ("exact_safe", "exact_lethal", "unknown_omitted")
            ]
            if sum(stock_counts) != _integer(
                drop.get("candidate_count"), "stock drop candidate count", minimum=0,
            ):
                hazard_failures.append("stock drop counts do not cover every candidate")
            _validate_drop_unknown_reasons(
                drop, "stock drop classification", hazard_failures,
            )
            if sum(stock_counts[:2]) and stock_drop_evidence != 10:
                hazard_failures.append("stock exact drops lack Pmove+fall evidence")
            if _integer(
                hazards.get("exact_lethal_candidates_omitted"),
                "stock exact lethal candidates omitted", minimum=0,
            ) != stock_counts[1]:
                hazard_failures.append(
                    "stock omitted exact-lethal candidate count differs from replay summary"
                )
    objective_accounting = _stock_objective_item_accounting(
        bsp_path, analysis, analysis_path,
    )
    criteria = {
        "analysis_quality": quality,
        "artifact_authority": artifact_authority,
        "stock_provenance": _criterion(provenance_failures, 1),
        "stock_reachable_spawns": _criterion(failures[len(quality["failures"]):], len(spawns)),
        "stock_structural_inventory": _criterion(structural_failures, 2),
        "stock_objective_accounting": objective_accounting,
        "stock_hazard_classification": _criterion(hazard_failures, 1),
    }
    all_failures = sorted(
        f"{name}: {failure}"
        for name, criterion in criteria.items()
        for failure in criterion["failures"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "mode": "stock_analysis_quality",
        "map": bsp_path.stem,
        "passed": not all_failures,
        "identities": {
            "b1_gate_sha256": b1_gate_digest,
            "bsp_sha256": bsp_digest,
            "analysis_sha256": file_sha256(analysis_path),
            "generator_claims_sha256": None,
            "stock_provenance_sha256": provenance_digest,
            "stock_inventory_sha256": inventory_digest,
        },
        "criteria": criteria,
        "failures": all_failures,
    }
    validate_report(report)
    return report


def validate_report(value: object) -> dict[str, Any]:
    report = _mapping(value, "claim-validation report")
    _exact_keys(
        report,
        {"schema", "mode", "map", "passed", "identities", "criteria", "failures"},
        "claim-validation report",
    )
    if report["schema"] != REPORT_SCHEMA or report["mode"] not in {
        "generated_v6_promotion", "stock_analysis_quality"
    }:
        raise ClaimValidationError("claim-validation report schema/mode mismatch")
    if not isinstance(report["map"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,64}", report["map"]
    ):
        raise ClaimValidationError("claim-validation map name is invalid")
    if not isinstance(report["passed"], bool):
        raise ClaimValidationError("claim-validation passed must be boolean")
    identities = _mapping(report["identities"], "claim-validation identities")
    common_identities = {
        "b1_gate_sha256", "bsp_sha256", "analysis_sha256",
        "generator_claims_sha256",
    }
    generated_identities = common_identities | {
        "map_sha256", "meta_sha256", "lattice_sha256",
        "hook_zones_sha256", "hook_materialization_sha256", "routes_sha256",
    }
    stock_identities = common_identities | {
        "stock_provenance_sha256", "stock_inventory_sha256",
    }
    expected_identities = (
        generated_identities
        if report["mode"] == "generated_v6_promotion"
        else stock_identities
    )
    _exact_keys(identities, expected_identities, "claim-validation identities")
    for name, digest in identities.items():
        if name == "generator_claims_sha256" and digest is None:
            continue
        _digest(digest, f"claim-validation {name}")
    if (
        report["mode"] == "generated_v6_promotion"
        and identities["generator_claims_sha256"] is None
    ) or (
        report["mode"] == "stock_analysis_quality"
        and identities["generator_claims_sha256"] is not None
    ):
        raise ClaimValidationError(
            "generator claims identity contradicts validation mode"
        )
    failures = _list(report["failures"], "claim-validation failures")
    if failures != sorted(set(failures)) or not all(isinstance(item, str) and item for item in failures):
        raise ClaimValidationError("claim-validation failures are not canonical")
    criteria = _mapping(report["criteria"], "claim-validation criteria")
    expected_criteria = (
        {
            "analysis_quality", "artifact_authority", "source_static_v6", "spawns", "hazards",
            "lethal_containment", "lighting", "hooks", "routes",
        }
        if report["mode"] == "generated_v6_promotion"
        else {
            "analysis_quality", "artifact_authority", "stock_provenance",
            "stock_reachable_spawns", "stock_structural_inventory",
            "stock_objective_accounting", "stock_hazard_classification",
        }
    )
    _exact_keys(criteria, expected_criteria, "claim-validation criteria")
    derived = []
    for name, value in criteria.items():
        criterion = _mapping(value, f"criterion {name}")
        _exact_keys(criterion, {"passed", "checked_count", "failures"}, f"criterion {name}")
        criterion_failures = _list(
            criterion["failures"], f"criterion {name} failures"
        )
        if criterion_failures != sorted(set(criterion_failures)) or not all(
            isinstance(item, str) and item for item in criterion_failures
        ):
            raise ClaimValidationError(
                f"criterion {name} failures are not canonical"
            )
        if criterion["passed"] is not (not criterion["failures"]):
            raise ClaimValidationError(f"criterion {name} pass flag contradicts failures")
        _integer(criterion["checked_count"], f"criterion {name} count", minimum=0)
        derived.extend(f"{name}: {failure}" for failure in criterion["failures"])
    if failures != sorted(derived) or report["passed"] is not (not failures):
        raise ClaimValidationError("report pass/failure summary contradicts criteria")
    return dict(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generated-v6", "stock"), required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--map-source", type=Path)
    parser.add_argument("--bsp", type=Path)
    parser.add_argument(
        "--b1-gate", type=Path, default=ROOT / "docs" / "multires" / "B1-GATE.json"
    )
    parser.add_argument("--claims-output", type=Path)
    args = parser.parse_args()
    if args.mode == "generated-v6":
        if args.map_source is None:
            parser.error("--map-source is required for generated-v6")
        claims = build_generator_claims(args.map_source)
        if args.claims_output:
            args.claims_output.write_bytes(canonical_bytes(claims))
        report = validate_generated_map(
            args.map_source, args.analysis, b1_gate_path=args.b1_gate
        )
    else:
        if args.bsp is None:
            parser.error("--bsp is required for stock analysis")
        report = validate_stock_analysis(
            args.bsp, args.analysis, b1_gate_path=args.b1_gate
        )
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
