#!/usr/bin/env python3
"""Challenge generator-v6 claims with a hash-bound compiled Atlas report.

Source and v2 sidecar metadata are claims only.  Generated-map promotion
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
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maps.generator import (  # noqa: E402
    DM_SPAWN_COUNT,
    MIN_FLOOR_LIGHT_COVERAGE,
    MIN_SPAWN_SEPARATION,
)
from harness.hook_claims_v2 import (  # noqa: E402
    HookClaimsV2Error,
    load_candidates,
    load_materialization,
    runtime_records_sha256,
    validate_record as validate_hook_record_v2,
    validate_runtime_sidecar,
    validation_trace_sha256,
)
from tools.validate_maps import (  # noqa: E402
    POINT_RE,
    _origin,
    _parse_brush_aabbs,
    _parse_entities,
    static_validate,
)


CLAIMS_SCHEMA = "q2-generator-claims-v2"
ANALYSIS_SCHEMA = "q2-atlas-analysis-v1"
REPORT_SCHEMA = "q2-generator-claim-validation-v1"
ORACLE_STATUS = "oracle"
MILLIUNITS = 1000
Q8 = 256
INFINITE_COST_Q8 = 0xFFFFFFFF
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_ROUTE_COST_ABSOLUTE_ERROR_Q8 = 1024 * Q8
MAX_ROUTE_COST_RATIO = 2.0
CLAIM_ID_RE = re.compile(r"^[a-z0-9:_-]{1,127}$")
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
                "anchor_milliunits": _to_milliunits(numbers[:3], "hook anchor"),
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
    if routes.get("version") != 1:
        raise ClaimValidationError("generated route sidecar must have version 1")
    nodes = _list(routes.get("nodes"), "route nodes")
    by_id: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(nodes):
        node = _mapping(value, f"route node {index}")
        node_id = _integer(node.get("id"), f"route node {index} id", minimum=0)
        if node_id in by_id:
            raise ClaimValidationError(f"duplicate route node id {node_id}")
        _to_milliunits(
            [node.get("x"), node.get("y"), node.get("z")],
            f"route node {node_id} origin",
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
        spawn_nodes = sorted(
            (
                node
                for node in by_id.values()
                if node.get("type") == "spawn" and node.get("room") == start_room
            ),
            key=lambda node: int(node["id"]),
        )
        if not spawn_nodes:
            raise ClaimValidationError(f"{route_id} has no spawn in its start room")
        item_ids = [
            _integer(item, f"{route_id} node id", minimum=0)
            for item in _list(route.get("node_ids"), f"{route_id} node ids")
        ]
        if len(item_ids) < 2 or any(item not in by_id for item in item_ids):
            raise ClaimValidationError(f"{route_id} has insufficient/unknown nodes")
        path = [spawn_nodes[0], *(by_id[item] for item in item_ids), spawn_nodes[0]]
        origins = [
            tuple(
                _finite(node.get(axis), f"{route_id} {axis}")
                for axis in ("x", "y", "z")
            )
            for node in path
        ]
        # The legacy room graph assigns zero cost between distinct points in
        # one room.  Its coordinate sequence is still a claim, so canonicalize
        # a nonzero geometric lower bound rather than treating zero as an
        # oracle fact or discarding the route.
        geometric_distance = sum(
            math.dist(source, target)
            for source, target in zip(origins, origins[1:])
        )
        normalized_claimed_distance = max(claimed_distance, geometric_distance)
        if normalized_claimed_distance <= 0:
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
                "claimed_cost_q8": round(normalized_claimed_distance * Q8),
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
    except HookClaimsV2Error as error:
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
    except HookClaimsV2Error as error:
        raise ClaimValidationError(str(error)) from error
    if len(runtime_hooks) != len(hooks) or runtime_records_sha256(hooks) != materialization[
        "runtime_records_sha256"
    ]:
        raise ClaimValidationError("runtime hook rows differ from materialized records")
    for runtime, selected in zip(runtime_hooks, hooks):
        for field in (
            "anchor_milliunits", "landing_milliunits", "distance_milliunits", "flags",
        ):
            if runtime[field] != selected[field]:
                raise ClaimValidationError("runtime hook geometry differs from selected v2 record")
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
        raise ClaimValidationError("generated claims require exactly six hook-v2 claims")
    for hook in hooks:
        try:
            validate_hook_record_v2(dict(hook), "hook claim")
        except HookClaimsV2Error as error:
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


def _analysis_quality(
    analysis: Mapping[str, Any],
    *,
    bsp_sha256: str,
    claims_sha256: str | None,
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
    except ClaimValidationError as exc:
        failures.append(str(exc))
    try:
        oracles = _mapping(analysis.get("oracles"), "analysis oracles")
        failures.extend(
            _oracle_record(oracles.get("collision"), "collision oracle")
        )
    except ClaimValidationError as exc:
        failures.append(str(exc))
    if analysis.get("deterministic_rebuild") is not True:
        failures.append("Atlas deterministic cold rebuild did not pass")
    confidence = analysis.get("confidence")
    if confidence not in {"high", "complete"}:
        failures.append("analysis confidence is not complete/high")
    return _criterion(failures, 6), compiled


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
            "lethal_drop_edges", "guarded_drop_edges",
            "uncontained_drop_edges",
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
            "spawn_region_count", "dark_spawn_regions",
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
    expected_regions = 0
    if isinstance(meta, Mapping):
        expected_regions = len(
            _list(_mapping(meta.get("lighting"), "generator lighting").get("regions"), "light regions")
        )
    actual_regions = _integer(lighting.get("spawn_region_count"), "spawn light regions", minimum=0)
    if expected_regions and actual_regions != expected_regions:
        failures.append("compiled spawn-light region count differs from v6 contract")
    if _integer(lighting.get("dark_spawn_regions"), "dark spawn regions", minimum=0) != 0:
        failures.append("compiled Atlas reports dark spawn regions")
    return _criterion(failures, actual_regions)


def _validate_hooks(
    claims: Mapping[str, Any], compiled: Mapping[str, Any], b1_gate: Mapping[str, Any]
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
    expected_physics = (
        _mapping(
            _mapping(b1_gate.get("artifacts"), "B1 artifacts").get("hook_parity_attestation"),
            "B1 hook attestation",
        ).get("hook_physics_identity")
    )
    _digest(expected_physics, "B1 hook physics identity")
    for claim in expected:
        edge = by_id.get(claim["claim_id"])
        if edge is None:
            continue
        _exact_keys(
            edge,
            {
                "claim_id", "source_l1", "target_l1", "source_milliunits",
                "anchor_milliunits", "landing_milliunits",
                "release_after_ticks", "distance_milliunits", "flags",
                "trajectory_origin_fixed", "trajectory_sha256",
                "first_grounded_frame_index",
                "landing_l1", "physics_identity", "evidence", "validation_version",
            },
            "compiled hook edge",
        )
        if tuple(_vec3_int(edge.get("anchor_milliunits"), "hook edge anchor")) != tuple(claim["anchor_milliunits"]):
            failures.append(f"{claim['claim_id']} anchor differs")
        for field in ("source_milliunits", "landing_milliunits"):
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

    analysis_quality, compiled = _analysis_quality(
        analysis, bsp_sha256=bsp_digest, claims_sha256=claims_digest
    )
    static_result = static_validate(map_path, _static_args())
    static_failures = [] if static_result.get("static_ok") is True else [
        "existing generator-v6 source/static validation failed"
    ]
    criteria = {
        "analysis_quality": analysis_quality,
        "source_static_v6": _criterion(static_failures, 1),
        "spawns": _validate_spawns(claims_context, compiled),
    }
    hazards, lethal = _validate_hazards(claims_context, compiled)
    criteria.update(
        {
            "hazards": hazards,
            "lethal_containment": lethal,
            "lighting": _validate_lighting(claims_context, compiled),
            "hooks": _validate_hooks(claims_context, compiled, b1_gate),
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
) -> dict[str, Any]:
    """Validate authored-map analysis quality without generator-v6 rules."""

    bsp_digest = file_sha256(bsp_path)
    analysis = _mapping(load_json(analysis_path), "compiled analysis")
    _gate, b1_gate_digest = _b1_gate(b1_gate_path)
    quality, compiled = _analysis_quality(
        analysis, bsp_sha256=bsp_digest, claims_sha256=None
    )
    failures = list(quality["failures"])
    spawns = [_mapping(item, "compiled stock spawn") for item in _list(compiled.get("spawns"), "compiled stock spawns")]
    if len(spawns) < 2:
        failures.append("stock analysis has fewer than two clear spawns")
    ordinals = {item.get("entity_ordinal") for item in spawns}
    for item in spawns:
        if item.get("standing_clear") is not True or item.get("supported") is not True:
            failures.append(f"stock spawn {item.get('entity_ordinal')} is not clear/supported")
        reachable = set(_list(item.get("reachable_spawn_ordinals"), "stock reachable spawns"))
        if not (reachable & (ordinals - {item.get("entity_ordinal")})):
            failures.append(f"stock spawn {item.get('entity_ordinal')} reaches no other spawn")
    hazards = _mapping(compiled.get("hazards"), "compiled stock hazards")
    _list(hazards.get("types"), "compiled stock hazard types")
    criteria = {
        "analysis_quality": quality,
        "stock_reachable_spawns": _criterion(failures[len(quality["failures"]):], len(spawns)),
        "stock_hazard_classification": _criterion([], 1),
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
    expected_identities = (
        generated_identities
        if report["mode"] == "generated_v6_promotion"
        else common_identities
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
            "analysis_quality", "source_static_v6", "spawns", "hazards",
            "lethal_containment", "lighting", "hooks", "routes",
        }
        if report["mode"] == "generated_v6_promotion"
        else {
            "analysis_quality", "stock_reachable_spawns",
            "stock_hazard_classification",
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
