#!/usr/bin/env python3
"""Run a cheap, fail-closed CM preflight over one compiled generator cohort.

This stage is deliberately narrower than Atlas construction.  It challenges
the exact compiled BSP population immediately after q2tool, before hook
materialization or full Atlas work, and publishes only non-admission evidence.
The final Atlas and generator-claim validators remain the promotion authority.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_analyzer import (  # noqa: E402
    AnalyzerLimits,
    CONTENTS_LAVA,
    CONTENTS_SOLID,
    CROUCHED_MAXS,
    CROUCHED_MINS,
    MASK_PLAYERSOLID,
    OracleProcess,
    STANDING_MAXS,
    STANDING_MINS,
)
from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    load_b1_authority_gate,
)
from harness.ibsp38 import BspValidationError, parse_ibsp38  # noqa: E402
from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    canonical_bytes,
    file_sha256,
    load_declaration,
    sha256_bytes,
    verify_stage_membership,
)
from tools.validate_maps import (  # noqa: E402
    _parse_brush_geometry,
    deathmatch_spawn_origins,
)


SCHEMA = "q2-b2-compiled-cm-preflight-v1"
SUMMARY_SCHEMA = "q2-b2-compiled-cm-preflight-result-v1"
STAGE = "post-q2tool-compiled-cm-preflight"
ADMISSION_STATUS = "non-admissible-preflight-only"
SPAWN_COUNT = 8
SPAWN_LINK_LIFT = 9.0
MIN_COLUMN_UNITS = 96
COLUMN_STEP_UNITS = 4
COLUMN_MAX_SWEEP_UNITS = 128
MIN_SPAWN_XY_SEPARATION_UNITS = 384
ESCAPE_DISTANCE_UNITS = 96
ESCAPE_STEP_UNITS = 16
SUPPORT_DEPTH_UNITS = 96
MILLIUNITS = 1000
MAX_JOBS = 32
MAX_ORACLE_BATCH_TIMEOUT_SECONDS = 60.0
HAZARD_SAMPLE_FRACTIONS = (
    (0.5, 0.5, 0.5),
    (0.25, 0.5, 0.5),
    (0.75, 0.5, 0.5),
    (0.5, 0.25, 0.5),
    (0.5, 0.75, 0.5),
    (0.5, 0.5, 0.25),
    (0.5, 0.5, 0.75),
)
IMPLEMENTATION_PATHS = (
    "tools/run_compiled_cm_preflight.py",
    "harness/atlas_analyzer.py",
    "harness/atlas_b1_authority.py",
    "harness/ibsp38.py",
    "tools/run_generator_cohort.py",
    "tools/validate_maps.py",
)


class CompiledCmPreflightError(RuntimeError):
    """The preflight could not produce trustworthy passing evidence."""


def _box_request(
    identifier: str,
    start: Sequence[float],
    end: Sequence[float],
    mins: Sequence[float],
    maxs: Sequence[float],
) -> dict[str, Any]:
    return {
        "id": identifier,
        "op": "box_trace",
        "start": list(start),
        "end": list(end),
        "mins": list(mins),
        "maxs": list(maxs),
        "mask": MASK_PLAYERSOLID,
    }


def _regular_file(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise CompiledCmPreflightError(f"cannot stat {label}: {error}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise CompiledCmPreflightError(f"{label} is not a regular non-symlink file")
    if executable and mode & 0o111 == 0:
        raise CompiledCmPreflightError(f"{label} is not executable")


def _absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _exclusive_write(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise CompiledCmPreflightError("output parent must be an existing directory")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _implementation_identity(repo_root: Path) -> dict[str, Any]:
    files = []
    for relative in IMPLEMENTATION_PATHS:
        path = repo_root / relative
        _regular_file(path, f"implementation file {relative}")
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return {
        "schema": "q2-b2-compiled-cm-preflight-implementation-v1",
        "files": files,
        "source_closure_sha256": sha256_bytes(canonical_bytes(files)),
    }


def _membership_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report": dict(value),
        "report_sha256": sha256_bytes(canonical_bytes(value)),
    }


def _origin_milliunits(raw: str, label: str) -> tuple[int, int, int]:
    words = raw.split()
    if len(words) != 3:
        raise CompiledCmPreflightError(f"{label} has no three-axis origin")
    output = []
    for word in words:
        try:
            value = float(word)
        except ValueError as error:
            raise CompiledCmPreflightError(f"{label} origin is not numeric") from error
        if not math.isfinite(value):
            raise CompiledCmPreflightError(f"{label} origin is not finite")
        scaled = value * MILLIUNITS
        rounded = round(scaled)
        if abs(scaled - rounded) > 1e-6:
            raise CompiledCmPreflightError(
                f"{label} origin is not representable in milliunits"
            )
        output.append(rounded)
    return tuple(output)  # type: ignore[return-value]


def _source_spawn_milliunits(map_path: Path) -> list[list[int]]:
    origins = deathmatch_spawn_origins(map_path)
    values = []
    for ordinal, origin in enumerate(origins):
        words = " ".join(str(axis) for axis in origin)
        values.append(list(_origin_milliunits(words, f"source spawn {ordinal}")))
    return sorted(values)


def _compiled_spawns(metadata: Any) -> list[tuple[int, tuple[int, int, int]]]:
    values = []
    for entity in metadata.entities:
        if entity.classname != "info_player_deathmatch":
            continue
        values.append((
            entity.index,
            _origin_milliunits(
                entity.value("origin"), f"compiled spawn entity {entity.index}"
            ),
        ))
    if len(values) != SPAWN_COUNT:
        raise CompiledCmPreflightError(
            f"compiled BSP contains {len(values)} deathmatch spawns, expected {SPAWN_COUNT}"
        )
    if len({origin for _, origin in values}) != len(values):
        raise CompiledCmPreflightError("compiled BSP spawn origins are not unique")
    return values


def _strict_json(path: Path, label: str) -> Any:
    def no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CompiledCmPreflightError(
                    f"{label} contains duplicate key {key!r}"
                )
            value[key] = item
        return value

    def invalid_constant(token: str) -> None:
        raise CompiledCmPreflightError(
            f"{label} contains non-finite number {token}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=invalid_constant,
        )
    except CompiledCmPreflightError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompiledCmPreflightError(f"cannot read {label}: {error}") from error


def _bounds_milliunits(value: Any, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise CompiledCmPreflightError(f"{label} must contain six bounds")
    output = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CompiledCmPreflightError(f"{label}[{index}] is not numeric")
        number = float(item)
        if not math.isfinite(number):
            raise CompiledCmPreflightError(f"{label}[{index}] is not finite")
        scaled = number * MILLIUNITS
        rounded = round(scaled)
        if abs(scaled - rounded) > 1e-6:
            raise CompiledCmPreflightError(
                f"{label}[{index}] is not representable in milliunits"
            )
        output.append(rounded)
    if any(output[axis] >= output[axis + 3] for axis in range(3)):
        raise CompiledCmPreflightError(f"{label} is not strictly ordered")
    return output


def _source_hazard_claims(
    lattice_path: Path, map_path: Path,
) -> list[dict[str, Any]]:
    """Reconstruct the canonical early hazard population from exact inputs."""

    lattice = _strict_json(lattice_path, f"{lattice_path.name} lattice")
    if not isinstance(lattice, Mapping):
        raise CompiledCmPreflightError("lattice sidecar must be an object")
    danger = lattice.get("danger")
    if not isinstance(danger, list):
        raise CompiledCmPreflightError("lattice danger claims must be a list")
    lava_bounds = sorted(
        _bounds_milliunits(item, f"danger {index}")
        for index, item in enumerate(danger)
    )
    if len({tuple(bounds) for bounds in lava_bounds}) != len(lava_bounds):
        raise CompiledCmPreflightError("lattice danger bounds are duplicated")

    try:
        source = map_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CompiledCmPreflightError(
            f"cannot read {map_path.name} hazard brushes: {error}"
        ) from error
    hurt_bounds = sorted(
        _bounds_milliunits(brush.bounds, f"trigger_hurt brush {index}")
        for index, brush in enumerate(_parse_brush_geometry(source))
        if brush.owner_classname == "trigger_hurt"
    )
    if len({tuple(bounds) for bounds in hurt_bounds}) != len(hurt_bounds):
        raise CompiledCmPreflightError("trigger_hurt bounds are duplicated")

    claims = [
        {
            "claim_id": f"hazard:lava:{index:04d}",
            "type": "lava",
            "bounds_milliunits": bounds,
        }
        for index, bounds in enumerate(lava_bounds)
    ]
    claims.extend(
        {
            "claim_id": f"hazard:hurt:{index:04d}",
            "type": "hurt",
            "bounds_milliunits": bounds,
        }
        for index, bounds in enumerate(hurt_bounds)
    )
    claims.sort(key=lambda claim: claim["claim_id"])
    return claims


def _entity_origin(entity: Any) -> tuple[float, float, float]:
    raw = entity.value("origin")
    if not raw:
        return (0.0, 0.0, 0.0)
    words = raw.split()
    if len(words) != 3:
        raise CompiledCmPreflightError(
            f"compiled trigger_hurt entity {entity.index} has invalid origin"
        )
    try:
        values = tuple(float(word) for word in words)
    except ValueError as error:
        raise CompiledCmPreflightError(
            f"compiled trigger_hurt entity {entity.index} has invalid origin"
        ) from error
    if not all(math.isfinite(value) for value in values):
        raise CompiledCmPreflightError(
            f"compiled trigger_hurt entity {entity.index} has non-finite origin"
        )
    return values  # type: ignore[return-value]


def _compiled_hurt_models(metadata: Any) -> dict[tuple[int, ...], dict[str, Any]]:
    models: dict[tuple[int, ...], dict[str, Any]] = {}
    for entity in metadata.entities:
        if entity.classname != "trigger_hurt":
            continue
        model_name = entity.value("model")
        if not model_name.startswith("*"):
            raise CompiledCmPreflightError(
                f"compiled trigger_hurt entity {entity.index} lacks an inline model"
            )
        try:
            model = metadata.models[int(model_name[1:])]
        except (ValueError, IndexError) as error:
            raise CompiledCmPreflightError(
                f"compiled trigger_hurt entity {entity.index} has invalid model"
            ) from error
        origin = _entity_origin(entity)
        bounds = _bounds_milliunits(
            [
                *(model.mins[axis] + origin[axis] for axis in range(3)),
                *(model.maxs[axis] + origin[axis] for axis in range(3)),
            ],
            f"compiled trigger_hurt entity {entity.index} bounds",
        )
        key = tuple(bounds)
        if key in models:
            raise CompiledCmPreflightError(
                "compiled trigger_hurt model bounds are duplicated"
            )
        try:
            damage = int(entity.value("dmg"), 10)
            spawnflags = int(entity.value("spawnflags") or "0", 10)
        except ValueError as error:
            raise CompiledCmPreflightError(
                f"compiled trigger_hurt entity {entity.index} has invalid properties"
            ) from error
        models[key] = {
            "entity_index": entity.index,
            "model": model,
            "origin": origin,
            "damage": damage,
            "spawnflags": spawnflags,
        }
    return models


def _hazard_sample_points(bounds: Sequence[int]) -> list[list[float]]:
    return [
        [
            (
                bounds[axis]
                + (bounds[axis + 3] - bounds[axis]) * fractions[axis]
            ) / MILLIUNITS
            for axis in range(3)
        ]
        for fractions in HAZARD_SAMPLE_FRACTIONS
    ]


def _hazard_record(
    oracle: OracleProcess,
    claim: Mapping[str, Any],
    compiled_hurt: Mapping[tuple[int, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    claim_id = str(claim["claim_id"])
    hazard_type = str(claim["type"])
    bounds = [int(value) for value in claim["bounds_milliunits"]]
    points = _hazard_sample_points(bounds)
    failures = []
    requests = []
    expected_contents = CONTENTS_LAVA
    if hazard_type == "lava":
        requests = [
            {
                "id": f"{claim_id}:contents:{sample_index}",
                "op": "point_contents",
                "point": point,
            }
            for sample_index, point in enumerate(points)
        ]
    elif hazard_type == "hurt":
        matched = compiled_hurt.get(tuple(bounds))
        if matched is None:
            failures.append("no exact compiled trigger_hurt inline-model bounds")
        else:
            if int(matched["damage"]) <= 0:
                failures.append("compiled trigger_hurt damage is not positive")
            if int(matched["spawnflags"]) & (1 | 2):
                failures.append("compiled trigger_hurt is stateful or initially disabled")
            model = matched["model"]
            requests = [
                {
                    "id": f"{claim_id}:inline-geometry:{sample_index}",
                    "op": "transformed_point_contents",
                    "point": point,
                    "headnode": model.headnode,
                    "origin": list(matched["origin"]),
                    "angles": [0.0, 0.0, 0.0],
                }
                for sample_index, point in enumerate(points)
            ]
        # The inline brush is compiled as solid geometry.  Runtime
        # trigger_hurt behavior comes from the BSP entity classname/properties;
        # this check deliberately does not call CONTENTS_SOLID "hurt contents".
        expected_contents = CONTENTS_SOLID
    else:
        failures.append("unsupported hazard type")

    responses = oracle.call(requests) if requests else []
    for sample_index, response in enumerate(responses):
        contents = response.get("contents")
        if (
            isinstance(contents, bool)
            or not isinstance(contents, int)
            or (contents & expected_contents) == 0
        ):
            if hazard_type == "lava":
                failures.append(
                    f"CM sample {sample_index} has no compiled lava contents"
                )
            else:
                failures.append(
                    f"CM sample {sample_index} has no compiled inline brush geometry"
                )
    return {
        "claim_id": claim_id,
        "type": hazard_type,
        "bounds_milliunits": bounds,
        "probe_count": len(responses),
        "failures": failures,
        "passed": not failures and len(responses) == len(HAZARD_SAMPLE_FRACTIONS),
    }


def _basic_hazard_containment(
    oracle: OracleProcess,
    claims: Sequence[Mapping[str, Any]],
    metadata: Any,
) -> dict[str, Any]:
    compiled_hurt = _compiled_hurt_models(metadata)
    expected_hurt = {
        tuple(int(value) for value in claim["bounds_milliunits"])
        for claim in claims
        if claim["type"] == "hurt"
    }
    unexpected_hurt = sorted(set(compiled_hurt) - expected_hurt)
    failures = []
    if unexpected_hurt:
        failures.append("compiled BSP has undeclared trigger_hurt model bounds")
    hazards = [
        _hazard_record(oracle, claim, compiled_hurt)
        for claim in claims
    ]
    for hazard in hazards:
        failures.extend(
            f"{hazard['claim_id']}: {failure}"
            for failure in hazard["failures"]
        )
    return {
        "declared_hazard_count": len(claims),
        "checked_hazard_count": len(hazards),
        "hazards": hazards,
        "failures": failures,
        "passed": (
            len(hazards) == len(claims)
            and not failures
            and all(hazard["passed"] for hazard in hazards)
        ),
    }


def _trace_clear(value: Mapping[str, Any]) -> bool:
    return (
        value.get("startsolid") is False
        and value.get("allsolid") is False
        and type(value.get("fraction")) in (int, float)
        and math.isfinite(float(value["fraction"]))
        and float(value["fraction"]) == 1.0
    )


def _support(value: Mapping[str, Any]) -> tuple[bool, int | None]:
    fraction = value.get("fraction")
    plane = value.get("plane")
    endpos = value.get("endpos")
    valid = (
        value.get("startsolid") is False
        and value.get("allsolid") is False
        and type(fraction) in (int, float)
        and math.isfinite(float(fraction))
        and 0.0 <= float(fraction) < 1.0
        and isinstance(plane, Mapping)
        and isinstance(plane.get("normal"), list)
        and len(plane["normal"]) == 3
        and all(type(axis) in (int, float) and math.isfinite(float(axis))
                for axis in plane["normal"])
        and float(plane["normal"][2]) >= 0.7
        and isinstance(endpos, list)
        and len(endpos) == 3
        and all(type(axis) in (int, float) and math.isfinite(float(axis))
                for axis in endpos)
    )
    return bool(valid), None


def _column_clearance(
    oracle: OracleProcess, spawn: Sequence[float], prefix: str
) -> int:
    heights = range(
        COLUMN_STEP_UNITS,
        COLUMN_MAX_SWEEP_UNITS + COLUMN_STEP_UNITS,
        COLUMN_STEP_UNITS,
    )
    requests = [
        _box_request(
            f"{prefix}:column:{height}",
            spawn,
            (spawn[0], spawn[1], spawn[2] + height),
            STANDING_MINS,
            STANDING_MAXS,
        )
        for height in heights
    ]
    clearance = 56
    for height, result in zip(heights, oracle.call(requests)):
        if not _trace_clear(result):
            break
        clearance = 56 + height
    return clearance * MILLIUNITS


def _escape_directions() -> tuple[tuple[float, float], ...]:
    diagonal = math.sqrt(0.5)
    return (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (diagonal, diagonal),
        (diagonal, -diagonal),
        (-diagonal, diagonal),
        (-diagonal, -diagonal),
    )


def _basic_escape(
    oracle: OracleProcess, spawn: Sequence[float], prefix: str
) -> dict[str, Any]:
    directions = _escape_directions()
    sweep_requests = []
    support_requests = []
    for direction_index, (dx, dy) in enumerate(directions):
        target = (
            spawn[0] + dx * ESCAPE_DISTANCE_UNITS,
            spawn[1] + dy * ESCAPE_DISTANCE_UNITS,
            spawn[2],
        )
        sweep_requests.append(_box_request(
            f"{prefix}:escape:{direction_index}:sweep",
            spawn,
            target,
            STANDING_MINS,
            STANDING_MAXS,
        ))
        for distance in range(
            ESCAPE_STEP_UNITS,
            ESCAPE_DISTANCE_UNITS + ESCAPE_STEP_UNITS,
            ESCAPE_STEP_UNITS,
        ):
            point = (
                spawn[0] + dx * distance,
                spawn[1] + dy * distance,
                spawn[2],
            )
            support_requests.append(_box_request(
                f"{prefix}:escape:{direction_index}:support:{distance}",
                point,
                (point[0], point[1], point[2] - SUPPORT_DEPTH_UNITS),
                STANDING_MINS,
                STANDING_MAXS,
            ))
    sweeps = oracle.call(sweep_requests)
    supports = oracle.call(support_requests)
    supports_per_direction = ESCAPE_DISTANCE_UNITS // ESCAPE_STEP_UNITS
    passing = []
    for direction_index, sweep in enumerate(sweeps):
        first = direction_index * supports_per_direction
        direction_supports = supports[first:first + supports_per_direction]
        if _trace_clear(sweep) and all(_support(item)[0] for item in direction_supports):
            passing.append(direction_index)
    return {
        "distance_milliunits": ESCAPE_DISTANCE_UNITS * MILLIUNITS,
        "support_step_milliunits": ESCAPE_STEP_UNITS * MILLIUNITS,
        "passing_direction_indices": passing,
        "passed": bool(passing),
    }


def _spawn_record(
    oracle: OracleProcess,
    entity_ordinal: int,
    authored_origin: tuple[int, int, int],
) -> dict[str, Any]:
    player = (
        authored_origin[0] / MILLIUNITS,
        authored_origin[1] / MILLIUNITS,
        authored_origin[2] / MILLIUNITS + SPAWN_LINK_LIFT,
    )
    prefix = f"spawn:{entity_ordinal}"
    stance_support = oracle.call([
        _box_request(
            f"{prefix}:standing", player, player, STANDING_MINS, STANDING_MAXS
        ),
        _box_request(
            f"{prefix}:crouched", player, player, CROUCHED_MINS, CROUCHED_MAXS
        ),
        _box_request(
            f"{prefix}:support",
            player,
            (player[0], player[1], player[2] - SUPPORT_DEPTH_UNITS),
            STANDING_MINS,
            STANDING_MAXS,
        ),
    ])
    supported, _ = _support(stance_support[2])
    support_end = stance_support[2].get("endpos")
    support_drop = None
    if supported and isinstance(support_end, list):
        support_drop = round((player[2] - float(support_end[2])) * MILLIUNITS)
    column = _column_clearance(oracle, player, prefix)
    escape = _basic_escape(oracle, player, prefix)
    standing_clear = _trace_clear(stance_support[0])
    crouched_clear = _trace_clear(stance_support[1])
    failures = []
    if not standing_clear:
        failures.append("standing hull is not clear at engine-linked spawn")
    if not crouched_clear:
        failures.append("crouched hull is not clear at engine-linked spawn")
    if not supported:
        failures.append("spawn has no bounded CM support")
    if column < MIN_COLUMN_UNITS * MILLIUNITS:
        failures.append("spawn has less than 96 units of oracle-swept column clearance")
    if not escape["passed"]:
        failures.append("spawn has no bounded straight supported escape corridor")
    return {
        "entity_ordinal": entity_ordinal,
        "authored_origin_milliunits": list(authored_origin),
        "engine_link_lift_milliunits": round(SPAWN_LINK_LIFT * MILLIUNITS),
        "standing_clear": standing_clear,
        "crouched_clear": crouched_clear,
        "supported": supported,
        "support_drop_milliunits": support_drop,
        "column_clearance_milliunits": column,
        "column_clear_96": column >= MIN_COLUMN_UNITS * MILLIUNITS,
        "basic_escape": escape,
        "failures": failures,
        "passed": not failures,
    }


def _minimum_spawn_xy(origins: Sequence[tuple[int, int, int]]) -> int:
    if len(origins) < 2:
        return 0
    return round(min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(origins)
        for right in origins[index + 1:]
    ))


def _validate_map(
    row: Mapping[str, Any],
    compiled_dir: Path,
    cm_oracle: Path,
    limits: AnalyzerLimits,
    expected_oracle_sha256: str,
    expected_tool_identity: str,
    expected_source_closure_sha256: str,
    oracle_factory: Callable[..., OracleProcess] = OracleProcess,
) -> dict[str, Any]:
    name = str(row["map"])
    bsp_path = compiled_dir / f"{name}.bsp"
    map_path = compiled_dir / f"{name}.map"
    lattice_path = compiled_dir / f"{name}.lattice.json"
    try:
        bsp_stat = bsp_path.stat()
        bsp_digest = file_sha256(bsp_path)
        metadata = parse_ibsp38(bsp_path)
        compiled_lightdata = {
            "bytes": metadata.lightmaps.byte_count,
            "sha256": metadata.lightmaps.sha256,
            "present": metadata.lightmaps.byte_count > 0,
        }
        source_origins = _source_spawn_milliunits(map_path)
        compiled = _compiled_spawns(metadata)
        hazard_claims = _source_hazard_claims(lattice_path, map_path)
        compiled_origins = sorted([list(origin) for _, origin in compiled])
        failures = []
        if not compiled_lightdata["present"]:
            failures.append("compiled BSP has no lightdata")
        if compiled_origins != source_origins:
            failures.append("compiled BSP spawn origins differ from source map")
        minimum_xy = _minimum_spawn_xy([origin for _, origin in compiled])
        if minimum_xy < MIN_SPAWN_XY_SEPARATION_UNITS * MILLIUNITS:
            failures.append("compiled spawn XY separation is below 384 units")
        with oracle_factory(cm_oracle, bsp_path, "cm", limits) as oracle:
            identity = oracle.identity
            if identity.get("map_sha256") != bsp_digest:
                raise CompiledCmPreflightError("CM oracle map digest differs from BSP")
            if identity.get("tool_identity") != expected_tool_identity:
                raise CompiledCmPreflightError("CM oracle tool identity differs from B1")
            provenance = identity.get("provenance")
            if not isinstance(provenance, Mapping) or (
                provenance.get("source_closure_sha256")
                != expected_source_closure_sha256
            ):
                raise CompiledCmPreflightError(
                    "CM oracle source closure differs from B1"
                )
            spawn_records = [
                _spawn_record(oracle, entity_ordinal, origin)
                for entity_ordinal, origin in compiled
            ]
            basic_hazards = _basic_hazard_containment(
                oracle, hazard_claims, metadata
            )
        for record in spawn_records:
            failures.extend(
                f"spawn {record['entity_ordinal']}: {message}"
                for message in record["failures"]
            )
        failures.extend(
            f"hazard containment: {message}"
            for message in basic_hazards["failures"]
        )
        if not basic_hazards["passed"]:
            failures.append("basic hazard containment did not pass")
        return {
            "ordinal": int(row["ordinal"]),
            "map": name,
            "bsp": {"bytes": bsp_stat.st_size, "sha256": bsp_digest},
            "compiled_lightdata": compiled_lightdata,
            "source_spawn_origins_milliunits": source_origins,
            "compiled_spawn_origins_milliunits": compiled_origins,
            "spawn_origin_sets_match": compiled_origins == source_origins,
            "minimum_spawn_xy_separation_milliunits": minimum_xy,
            "oracle": {
                "executable_sha256": expected_oracle_sha256,
                "tool_identity": identity["tool_identity"],
                "physics_identity": identity["physics_identity"],
                "map_sha256": identity["map_sha256"],
            },
            "spawn_count": len(spawn_records),
            "spawns": spawn_records,
            "basic_escape_scope": (
                "bounded straight standing-hull CM sweeps with 16-unit support "
                "samples; not an all-to-all Atlas reachability claim"
            ),
            "all_to_all_reachability": "not-evaluated-by-preflight",
            "basic_hazard_containment": basic_hazards,
            "failures": failures,
            "passed": not failures,
        }
    except (
        BspValidationError,
        CompiledCmPreflightError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        return {
            "ordinal": int(row["ordinal"]),
            "map": name,
            "failures": [f"{type(error).__name__}: {error}"],
            "passed": False,
        }


def build_report(
    *,
    declaration_path: Path,
    compiled_dir: Path,
    cm_oracle: Path,
    jobs: int,
    oracle_batch_timeout_seconds: float,
    repo_root: Path = ROOT,
    oracle_factory: Callable[..., OracleProcess] = OracleProcess,
) -> dict[str, Any]:
    """Validate one exact compiled stage and return canonical report content."""
    declaration_path = _absolute(declaration_path)
    compiled_dir = _absolute(compiled_dir)
    cm_oracle = _absolute(cm_oracle)
    if not 1 <= jobs <= MAX_JOBS:
        raise CompiledCmPreflightError(f"jobs must be in [1, {MAX_JOBS}]")
    if not math.isfinite(oracle_batch_timeout_seconds) or not (
        0 < oracle_batch_timeout_seconds <= MAX_ORACLE_BATCH_TIMEOUT_SECONDS
    ):
        raise CompiledCmPreflightError(
            "oracle batch timeout must be finite and in (0, 60]"
        )
    _regular_file(declaration_path, "declaration")
    _regular_file(cm_oracle, "CM oracle", executable=True)
    declaration, declaration_sha256 = load_declaration(declaration_path)
    initial_membership = verify_stage_membership(
        declaration, compiled_dir, "compiled"
    )
    implementation = _implementation_identity(repo_root)
    gate = load_b1_authority_gate(repo_root)
    oracle_sha256 = file_sha256(cm_oracle)
    global_failures = []
    if oracle_sha256 != gate.cm_executable_sha256:
        global_failures.append("CM oracle executable bytes differ from B1")
    if initial_membership["passed"] is not True:
        global_failures.extend(
            f"compiled membership: {failure}"
            for failure in initial_membership["failures"]
        )

    limits = AnalyzerLimits(
        oracle_batch_timeout_seconds=oracle_batch_timeout_seconds,
    )
    rows: list[dict[str, Any]] = []
    if not global_failures:
        by_ordinal: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _validate_map,
                    row,
                    compiled_dir,
                    cm_oracle,
                    limits,
                    oracle_sha256,
                    gate.oracle_tool_identity,
                    gate.oracle_source_closure_sha256,
                    oracle_factory,
                ): int(row["ordinal"])
                for row in declaration["maps"]
            }
            for future in as_completed(futures):
                ordinal = futures[future]
                try:
                    by_ordinal[ordinal] = future.result()
                except Exception as error:  # defensive executor boundary
                    by_ordinal[ordinal] = {
                        "ordinal": ordinal,
                        "map": declaration["maps"][ordinal]["map"],
                        "failures": [
                            f"executor {type(error).__name__}: {error}"
                        ],
                        "passed": False,
                    }
        rows = [by_ordinal[index] for index in range(len(declaration["maps"]))]

    try:
        final_declaration, final_declaration_sha256 = load_declaration(
            declaration_path
        )
        final_membership = verify_stage_membership(
            final_declaration, compiled_dir, "compiled"
        )
        final_implementation = _implementation_identity(repo_root)
        final_oracle_sha256 = file_sha256(cm_oracle)
    except (GeneratorCohortError, OSError, CompiledCmPreflightError) as error:
        final_declaration = None
        final_declaration_sha256 = None
        final_membership = None
        final_implementation = None
        final_oracle_sha256 = None
        global_failures.append(f"final input snapshot failed: {error}")

    membership_stable = (
        final_membership == initial_membership
        if final_membership is not None else False
    )
    declaration_stable = (
        final_declaration == declaration
        and final_declaration_sha256 == declaration_sha256
    )
    implementation_stable = final_implementation == implementation
    oracle_stable = final_oracle_sha256 == oracle_sha256
    if not declaration_stable:
        global_failures.append("declaration changed during preflight")
    if not membership_stable:
        global_failures.append("compiled membership or file bytes changed during preflight")
    if not implementation_stable:
        global_failures.append("preflight implementation closure changed during execution")
    if not oracle_stable:
        global_failures.append("CM oracle bytes changed during preflight")
    map_failures = sum(not row.get("passed", False) for row in rows)
    passed = (
        not global_failures
        and len(rows) == len(declaration["maps"])
        and map_failures == 0
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": STAGE,
        "admission_status": ADMISSION_STATUS,
        "promotion_authority": False,
        "cohort_id": declaration["cohort_id"],
        "declaration": {
            "path": str(declaration_path),
            "sha256": declaration_sha256,
            "map_count": len(declaration["maps"]),
        },
        "compiled_root": str(compiled_dir),
        "compiled_membership": _membership_identity(initial_membership),
        "b1_authority": {
            "gate_sha256": file_sha256(repo_root / "docs/multires/B1-GATE.json"),
            "cm_executable_sha256": gate.cm_executable_sha256,
            "cm_tool_identity": gate.oracle_tool_identity,
            "cm_source_closure_sha256": gate.oracle_source_closure_sha256,
        },
        "implementation": implementation,
        "execution": {
            "parallel_jobs": jobs,
            "oracle_batch_timeout_milliseconds": round(
                oracle_batch_timeout_seconds * 1000
            ),
            "map_order": "canonical-declaration-order",
        },
        "checks": {
            "compiled_spawn_origins_exact": True,
            "engine_spawn_link_lift_milliunits": round(
                SPAWN_LINK_LIFT * MILLIUNITS
            ),
            "standing_and_crouched_stationary_hulls": True,
            "support_depth_milliunits": SUPPORT_DEPTH_UNITS * MILLIUNITS,
            "oracle_swept_column_minimum_milliunits": (
                MIN_COLUMN_UNITS * MILLIUNITS
            ),
            "minimum_spawn_xy_separation_milliunits": (
                MIN_SPAWN_XY_SEPARATION_UNITS * MILLIUNITS
            ),
            "bounded_basic_escape": True,
            "compiled_lightdata_presence": True,
            "basic_hazard_containment": True,
            "all_to_all_reachability": "deferred-to-full-Atlas-admission",
        },
        "input_stability": {
            "declaration": declaration_stable,
            "compiled_membership": membership_stable,
            "implementation": implementation_stable,
            "cm_oracle": oracle_stable,
        },
        "maps": rows,
        "map_count": len(rows),
        "pass_count": len(rows) - map_failures,
        "failure_count": map_failures,
        "failures": sorted(set(global_failures)),
        "passed": passed,
    }
    report["canonical_record_sha256"] = sha256_bytes(canonical_bytes(report))
    return report


def run(
    *,
    declaration_path: Path,
    compiled_dir: Path,
    cm_oracle: Path,
    output: Path,
    jobs: int,
    oracle_batch_timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    output = _absolute(output)
    if _is_within(output, _absolute(compiled_dir)):
        raise CompiledCmPreflightError(
            "output must be outside the exact compiled stage root"
        )
    if output.exists() or output.is_symlink():
        raise CompiledCmPreflightError("output already exists")
    report = build_report(
        declaration_path=declaration_path,
        compiled_dir=compiled_dir,
        cm_oracle=cm_oracle,
        jobs=jobs,
        oracle_batch_timeout_seconds=oracle_batch_timeout_seconds,
    )
    payload = canonical_bytes(report)
    _exclusive_write(output, payload)
    return report, hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a parallel, non-admission CM spawn preflight over one exact "
            "post-q2tool generated cohort"
        )
    )
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--oracle-batch-timeout-seconds", type=float, default=10.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, report_sha256 = run(
            declaration_path=args.declaration,
            compiled_dir=args.compiled_dir,
            cm_oracle=args.cm_oracle,
            output=args.output,
            jobs=args.jobs,
            oracle_batch_timeout_seconds=args.oracle_batch_timeout_seconds,
        )
    except (
        B1AuthorityError,
        CompiledCmPreflightError,
        GeneratorCohortError,
        OSError,
        ValueError,
    ) as error:
        print(f"compiled CM preflight error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes({
        "schema": SUMMARY_SCHEMA,
        "stage": STAGE,
        "admission_status": ADMISSION_STATUS,
        "report": str(_absolute(args.output)),
        "report_sha256": report_sha256,
        "map_count": report["map_count"],
        "pass_count": report["pass_count"],
        "failure_count": report["failure_count"],
        "passed": report["passed"],
    }))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
