#!/usr/bin/env python3
"""Fail-closed exact controlled-drop replay through pmove and fall oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any, Callable, Mapping, Sequence


MANIFEST_SCHEMA = "q2-atlas-drop-replay-manifest-v1"
RESULT_SCHEMA = "q2-atlas-drop-replay-v1"
PREPARED_SCHEMA = "q2-atlas-drop-fall-request-v1"
PMOVE_REQUEST_SCHEMA = "q2-atlas-drop-pmove-request-v1"
PMOVE_SCHEMA = "q2-pmove-oracle-v1"
FALL_SCHEMA = "q2-fall-oracle-v1"
MAX_HORIZON_FRAMES = 4096
MAX_ORACLE_STDOUT_BYTES = 8 * 1024 * 1024
SHA_CHARS = frozenset("0123456789abcdef")

OracleRunner = Callable[
    [str, Path, Sequence[dict[str, Any]], Path | None],
    list[dict[str, Any]],
]


class DropReplayError(ValueError):
    """An Exact controlled-drop claim cannot be admitted."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _reject(condition: bool, reason: str, detail: str) -> None:
    if condition:
        raise DropReplayError(reason, detail)


def _object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    _reject(not isinstance(value, Mapping), "invalid_contract", f"{label} must be an object")
    actual = set(value)
    _reject(
        actual != keys,
        "invalid_contract",
        f"{label} fields differ; missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}",
    )
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    _reject(type(value) is not int, "invalid_contract", f"{label} must be an integer")
    _reject(not minimum <= value <= maximum, "invalid_contract", f"{label} is out of range")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    _reject(
        isinstance(value, bool) or not isinstance(value, (int, float)),
        "invalid_contract",
        f"{label} must be a finite number",
    )
    result = float(value)
    _reject(not math.isfinite(result), "invalid_contract", f"{label} must be finite")
    _reject(not minimum <= result <= maximum, "invalid_contract", f"{label} is out of range")
    return result


def _boolean(value: Any, label: str) -> bool:
    _reject(type(value) is not bool, "invalid_contract", f"{label} must be boolean")
    return value


def _string(value: Any, label: str, maximum: int = 127) -> str:
    _reject(not isinstance(value, str) or not value or len(value) > maximum,
            "invalid_contract", f"{label} must be a nonempty string of at most {maximum} characters")
    return value


def _sha256(value: Any, label: str) -> str:
    _reject(
        not isinstance(value, str) or len(value) != 64 or any(char not in SHA_CHARS for char in value),
        "invalid_contract",
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _vec3(value: Any, label: str, minimum: float = -1_048_576,
          maximum: float = 1_048_576) -> list[float]:
    _reject(not isinstance(value, list) or len(value) != 3,
            "invalid_contract", f"{label} must contain exactly three numbers")
    return [_number(component, f"{label}[{axis}]", minimum, maximum)
            for axis, component in enumerate(value)]


def _short3(value: Any, label: str) -> list[int]:
    _reject(not isinstance(value, list) or len(value) != 3,
            "invalid_contract", f"{label} must contain exactly three integers")
    return [_integer(component, f"{label}[{axis}]", -32768, 32767)
            for axis, component in enumerate(value)]


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DropReplayError("invalid_contract", f"value is not canonical JSON: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_command(value: Any, cadence_msec: int, index: int) -> dict[str, Any]:
    _reject(not isinstance(value, Mapping), "invalid_cadence", f"commands[{index}] must be an object")
    allowed = {
        "msec", "angles", "angles_short", "forwardmove", "sidemove",
        "upmove", "buttons", "impulse", "lightlevel",
    }
    unknown = set(value) - allowed
    _reject(bool(unknown), "invalid_cadence", f"commands[{index}] has unknown fields {sorted(unknown)}")
    _reject("msec" not in value, "invalid_cadence", f"commands[{index}].msec is required")
    msec = _integer(value["msec"], f"commands[{index}].msec", 1, 255)
    _reject(msec != cadence_msec, "invalid_cadence",
            f"commands[{index}].msec {msec} differs from declared cadence {cadence_msec}")
    _reject("angles" in value and "angles_short" in value, "invalid_cadence",
            f"commands[{index}] cannot contain both angles encodings")
    result: dict[str, Any] = {"msec": msec}
    if "angles" in value:
        result["angles"] = _vec3(value["angles"], f"commands[{index}].angles", -360, 360)
    if "angles_short" in value:
        result["angles_short"] = _short3(value["angles_short"], f"commands[{index}].angles_short")
    for field in ("forwardmove", "sidemove", "upmove"):
        if field in value:
            result[field] = _integer(value[field], f"commands[{index}].{field}", -32768, 32767)
    for field in ("buttons", "impulse", "lightlevel"):
        if field in value:
            result[field] = _integer(value[field], f"commands[{index}].{field}", 0, 255)
    return result


def _validate_expected_authority(value: Any, label: str, *, map_bound: bool) -> dict[str, str]:
    keys = {"executable_sha256", "tool_identity", "physics_identity"}
    if map_bound:
        keys.add("map_sha256")
    authority = _object(value, keys, label)
    return {field: _sha256(authority[field], f"{label}.{field}") for field in keys}


def _validate_manifest(value: Any) -> dict[str, Any]:
    manifest = _object(value, {
        "schema", "id", "map_path", "horizon_frames", "cadence_msec",
        "dynamic_movers", "pmove", "fall", "authorities",
    }, "drop replay manifest")
    _reject(manifest["schema"] != MANIFEST_SCHEMA, "invalid_contract", "manifest schema mismatch")
    replay_id = _string(manifest["id"], "manifest.id", 80)
    map_path = _string(manifest["map_path"], "manifest.map_path", 4096)
    horizon = _integer(manifest["horizon_frames"], "manifest.horizon_frames", 2, MAX_HORIZON_FRAMES)
    cadence = _integer(manifest["cadence_msec"], "manifest.cadence_msec", 1, 255)
    dynamic_movers = _boolean(manifest["dynamic_movers"], "manifest.dynamic_movers")

    pmove = _object(manifest["pmove"], {
        "origin", "velocity", "pm_type", "pm_flags", "pm_time", "gravity",
        "airaccelerate", "delta_angles_short", "snapinitial", "commands",
    }, "manifest.pmove")
    commands = pmove["commands"]
    _reject(not isinstance(commands, list), "invalid_cadence", "manifest.pmove.commands must be an array")
    _reject(len(commands) != horizon, "invalid_cadence",
            "manifest horizon_frames must exactly equal command count")
    normalized_pmove = {
        "origin": _vec3(pmove["origin"], "manifest.pmove.origin", -4096, 4095.875),
        "velocity": _vec3(pmove["velocity"], "manifest.pmove.velocity", -4096, 4095.875),
        "pm_type": _integer(pmove["pm_type"], "manifest.pmove.pm_type", 0, 4),
        "pm_flags": _integer(pmove["pm_flags"], "manifest.pmove.pm_flags", 0, 255),
        "pm_time": _integer(pmove["pm_time"], "manifest.pmove.pm_time", 0, 255),
        "gravity": _integer(pmove["gravity"], "manifest.pmove.gravity", 0, 32767),
        "airaccelerate": _f32(_number(pmove["airaccelerate"], "manifest.pmove.airaccelerate", -1000, 1000)),
        "delta_angles_short": _short3(pmove["delta_angles_short"], "manifest.pmove.delta_angles_short"),
        "snapinitial": _boolean(pmove["snapinitial"], "manifest.pmove.snapinitial"),
        "commands": [_validate_command(command, cadence, index)
                     for index, command in enumerate(commands)],
    }

    fall = _object(manifest["fall"], {
        "fall_damagemod", "deathmatch", "dmflags", "health",
    }, "manifest.fall")
    normalized_fall = {
        "fall_damagemod": _f32(_number(fall["fall_damagemod"], "manifest.fall.fall_damagemod", 0, 1000)),
        "deathmatch": _boolean(fall["deathmatch"], "manifest.fall.deathmatch"),
        "dmflags": _integer(fall["dmflags"], "manifest.fall.dmflags", 0, 2147483647),
        "health": _integer(fall["health"], "manifest.fall.health", 1, 1_000_000),
    }
    authorities = _object(manifest["authorities"], {"pmove", "fall"}, "manifest.authorities")
    return {
        "schema": MANIFEST_SCHEMA,
        "id": replay_id,
        "map_path": map_path,
        "horizon_frames": horizon,
        "cadence_msec": cadence,
        "dynamic_movers": dynamic_movers,
        "pmove": normalized_pmove,
        "fall": normalized_fall,
        "authorities": {
            "pmove": _validate_expected_authority(authorities["pmove"], "manifest.authorities.pmove", map_bound=True),
            "fall": _validate_expected_authority(authorities["fall"], "manifest.authorities.fall", map_bound=False),
        },
    }


def _validate_provenance(value: Any) -> Mapping[str, Any]:
    provenance = _object(value, {
        "schema", "tool_identity", "source_closure_sha256", "source_closure_count",
        "build_identity_sha256", "compiler", "archiver", "build",
    }, "pmove provenance")
    _reject(provenance["schema"] != "q2-oracle-tool-identity-v1",
            "invalid_pmove_evidence", "pmove provenance schema mismatch")
    for field in ("tool_identity", "source_closure_sha256", "build_identity_sha256"):
        _sha256(provenance[field], f"pmove provenance.{field}")
    _integer(provenance["source_closure_count"], "pmove provenance.source_closure_count", 1, 1_000_000)
    compiler = _object(provenance["compiler"], {"command", "version", "target", "executable_sha256"}, "pmove compiler")
    archiver = _object(provenance["archiver"], {"command", "version", "executable_sha256"}, "pmove archiver")
    build = _object(provenance["build"], {"cflags", "ldflags"}, "pmove build")
    for field in ("command", "version", "target"):
        _string(compiler[field], f"pmove compiler.{field}", 16384)
    for field in ("command", "version"):
        _string(archiver[field], f"pmove archiver.{field}", 16384)
    _sha256(compiler["executable_sha256"], "pmove compiler.executable_sha256")
    _sha256(archiver["executable_sha256"], "pmove archiver.executable_sha256")
    for field in ("cflags", "ldflags"):
        _reject(not isinstance(build[field], str), "invalid_pmove_evidence", f"pmove build.{field} must be a string")
    return provenance


def _validate_pmove_identity(
    value: Any, expected_id: str, expected: Mapping[str, str],
    gravity: int, airaccelerate: float,
) -> dict[str, Any]:
    record = _object(value, {
        "ok", "id", "op", "schema", "tool_identity", "physics_identity",
        "map_sha256", "map_checksum", "parameters", "provenance", "source",
    }, "pmove identity")
    _reject(record["ok"] is not True or record["id"] != expected_id or record["op"] != "identity",
            "invalid_pmove_evidence", "pmove identity operation or id mismatch")
    _reject(record["schema"] != PMOVE_SCHEMA, "invalid_pmove_evidence", "pmove schema mismatch")
    for field in ("tool_identity", "physics_identity", "map_sha256"):
        _sha256(record[field], f"pmove identity.{field}")
        _reject(record[field] != expected[field], "invalid_pmove_evidence", f"pmove {field} mismatch")
    _integer(record["map_checksum"], "pmove identity.map_checksum", 0, 4294967295)
    parameters = _object(record["parameters"], {"gravity", "airaccelerate", "constants"}, "pmove parameters")
    effective_gravity = _integer(parameters["gravity"], "pmove parameters.gravity", 0, 32767)
    _reject(effective_gravity != gravity, "invalid_pmove_evidence", "pmove gravity identity mismatch")
    effective_air = _number(parameters["airaccelerate"], "pmove parameters.airaccelerate", -1000, 1000)
    _reject(effective_air != airaccelerate, "invalid_pmove_evidence", "pmove airaccelerate identity mismatch")
    _string(parameters["constants"], "pmove parameters.constants", 16384)
    provenance = _validate_provenance(record["provenance"])
    _reject(provenance["tool_identity"] != record["tool_identity"],
            "invalid_pmove_evidence", "pmove provenance tool identity mismatch")
    source = _object(record["source"], {
        "collision_sha256", "pmove_sha256", "shared_header_sha256", "shared_source_sha256",
    }, "pmove source")
    for field in source:
        _sha256(source[field], f"pmove source.{field}")
    return dict(record)


PMOVE_STATE_KEYS = {
    "command_index", "origin", "velocity", "origin_fixed", "velocity_fixed",
    "pm_type", "pm_flags", "pm_time", "gravity", "viewangles", "viewheight",
    "mins", "maxs", "grounded", "waterlevel", "watertype", "touch_count",
}


def _validate_pmove_state(value: Any, index: int, gravity: int) -> dict[str, Any]:
    state = _object(value, PMOVE_STATE_KEYS, f"pmove frame {index}")
    command_index = _integer(state["command_index"], f"pmove frame {index}.command_index", 0, MAX_HORIZON_FRAMES - 1)
    _reject(command_index != index, "invalid_pmove_evidence",
            f"pmove frame {index} is missing, duplicated, or out of order")
    origin = _vec3(state["origin"], f"pmove frame {index}.origin", -4096, 4095.875)
    velocity = _vec3(state["velocity"], f"pmove frame {index}.velocity", -4096, 4095.875)
    origin_fixed = _short3(state["origin_fixed"], f"pmove frame {index}.origin_fixed")
    velocity_fixed = _short3(state["velocity_fixed"], f"pmove frame {index}.velocity_fixed")
    _reject(origin != [component * 0.125 for component in origin_fixed],
            "invalid_pmove_evidence", f"pmove frame {index} origin projection is tampered")
    _reject(velocity != [component * 0.125 for component in velocity_fixed],
            "invalid_pmove_evidence", f"pmove frame {index} velocity projection is tampered")
    _integer(state["pm_type"], f"pmove frame {index}.pm_type", 0, 4)
    _integer(state["pm_flags"], f"pmove frame {index}.pm_flags", 0, 255)
    _integer(state["pm_time"], f"pmove frame {index}.pm_time", 0, 255)
    frame_gravity = _integer(state["gravity"], f"pmove frame {index}.gravity", 0, 32767)
    _reject(frame_gravity != gravity, "invalid_pmove_evidence", f"pmove frame {index} gravity mismatch")
    _vec3(state["viewangles"], f"pmove frame {index}.viewangles")
    _number(state["viewheight"], f"pmove frame {index}.viewheight", -4096, 4096)
    mins = _vec3(state["mins"], f"pmove frame {index}.mins", -4096, 4096)
    maxs = _vec3(state["maxs"], f"pmove frame {index}.maxs", -4096, 4096)
    _reject(any(mins[axis] > maxs[axis] for axis in range(3)),
            "invalid_pmove_evidence", f"pmove frame {index} hull is invalid")
    _boolean(state["grounded"], f"pmove frame {index}.grounded")
    try:
        _integer(state["waterlevel"], f"pmove frame {index}.waterlevel", 0, 3)
    except DropReplayError as error:
        raise DropReplayError("invalid_water", error.detail) from error
    _integer(state["watertype"], f"pmove frame {index}.watertype", -2147483648, 2147483647)
    _integer(state["touch_count"], f"pmove frame {index}.touch_count", 0, 32)
    return dict(state)


def _validate_pmove_simulation(
    value: Any, expected_id: str, identity: Mapping[str, Any], horizon: int, gravity: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = _object(value, {
        "ok", "id", "op", "schema", "tool_identity", "physics_identity",
        "map_sha256", "map_checksum", "frames", "final", "command_count",
    }, "pmove simulation")
    _reject(record["ok"] is not True or record["id"] != expected_id or record["op"] != "simulate",
            "invalid_pmove_evidence", "pmove simulate operation or id mismatch")
    for field in ("schema", "tool_identity", "physics_identity", "map_sha256", "map_checksum"):
        _reject(record[field] != identity[field], "invalid_pmove_evidence", f"pmove response {field} mismatch")
    command_count = _integer(record["command_count"], "pmove command_count", 0, MAX_HORIZON_FRAMES)
    _reject(command_count != horizon, "invalid_pmove_evidence", "pmove command_count mismatch")
    frames = record["frames"]
    _reject(not isinstance(frames, list) or len(frames) != horizon,
            "invalid_pmove_evidence", "pmove frame count differs from the bounded horizon")
    validated = [_validate_pmove_state(frame, index, gravity) for index, frame in enumerate(frames)]
    _reject(record["final"] != frames[-1], "invalid_pmove_evidence", "pmove final state differs from last frame")
    return dict(record), validated


def _validate_fall_source(value: Any) -> dict[str, str]:
    source = _object(value, {
        "shared_c_sha256", "shared_h_sha256", "integration_sha256",
        "game_header_sha256", "constants_sha256", "build_contract",
        "tool_closure_sha256",
    }, "fall source")
    result: dict[str, str] = {}
    for field in source:
        if field == "build_contract":
            result[field] = _string(source[field], "fall source.build_contract", 1024)
        else:
            result[field] = _sha256(source[field], f"fall source.{field}")
    return result


def _fall_physics_identity(parameters: Mapping[str, Any], source: Mapping[str, str], constants: str) -> str:
    number = lambda value: format(float(value), ".9g")
    canonical = (
        f"schema={FALL_SCHEMA};tool={source['tool_closure_sha256']};"
        f"shared_c={source['shared_c_sha256']};shared_h={source['shared_h_sha256']};"
        f"integration={source['integration_sha256']};game_header={source['game_header_sha256']};"
        f"constants_sha256={source['constants_sha256']};constants={constants};"
        f"build={source['build_contract']};fall_damagemod={number(parameters['fall_damagemod'])};"
        f"deathmatch={1 if parameters['deathmatch'] else 0};dmflags={parameters['dmflags']}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_fall_identity(
    value: Any, expected_id: str, expected: Mapping[str, str], fall: Mapping[str, Any],
) -> dict[str, Any]:
    record = _object(value, {
        "ok", "id", "op", "schema", "physics_identity", "tool_identity",
        "parameters", "constants", "source",
    }, "fall identity")
    _reject(record["ok"] is not True or record["id"] != expected_id or record["op"] != "identity",
            "invalid_fall_evidence", "fall identity operation or id mismatch")
    _reject(record["schema"] != FALL_SCHEMA, "invalid_fall_evidence", "fall schema mismatch")
    for field in ("tool_identity", "physics_identity"):
        _sha256(record[field], f"fall identity.{field}")
        _reject(record[field] != expected[field], "invalid_fall_evidence", f"fall {field} mismatch")
    parameters = _object(record["parameters"], {"fall_damagemod", "deathmatch", "dmflags"}, "fall parameters")
    effective_modifier = _number(parameters["fall_damagemod"], "fall parameters.fall_damagemod", 0, 1000)
    effective_deathmatch = _boolean(parameters["deathmatch"], "fall parameters.deathmatch")
    effective_dmflags = _integer(parameters["dmflags"], "fall parameters.dmflags", 0, 2147483647)
    _reject(effective_modifier != fall["fall_damagemod"] or
            effective_deathmatch is not fall["deathmatch"] or
            effective_dmflags != fall["dmflags"],
            "invalid_fall_evidence", "fall identity parameters differ from manifest")
    constants = _string(record["constants"], "fall constants", 16384)
    source = _validate_fall_source(record["source"])
    _reject(source["tool_closure_sha256"] != record["tool_identity"],
            "invalid_fall_evidence", "fall source/tool identity mismatch")
    _reject(hashlib.sha256(constants.encode("utf-8")).hexdigest() != source["constants_sha256"],
            "invalid_fall_evidence", "fall constants digest mismatch")
    _reject(_fall_physics_identity(parameters, source, constants) != record["physics_identity"],
            "invalid_fall_evidence", "fall physics identity preimage mismatch")
    return dict(record)


FALL_INPUT_KEYS = {
    "old_velocity_z", "velocity_z", "grapple_release_elapsed", "fall_damagemod",
    "modelindex", "movetype", "grounded", "hook_out", "grapple_present",
    "grapple_state", "waterlevel", "deathmatch", "dmflags", "health",
}


def _validate_fall_evaluation(
    value: Any, expected_id: str, identity: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "ok", "id", "op", "schema", "physics_identity", "tool_identity",
        "input", "suppression", "severity", "delta", "fall_value",
        "fall_time_offset", "emit_event", "set_fall_state", "set_pain_debounce",
        "damage", "apply_damage", "unmitigated_health_after", "unmitigated_lethal",
    }
    record = _object(value, fields, "fall evaluation")
    _reject(record["ok"] is not True or record["id"] != expected_id or record["op"] != "evaluate",
            "invalid_fall_evidence", "fall evaluation operation or id mismatch")
    for field in ("schema", "tool_identity", "physics_identity"):
        _reject(record[field] != identity[field], "invalid_fall_evidence", f"fall response {field} mismatch")
    fall_input = _object(record["input"], FALL_INPUT_KEYS, "fall response input")
    for field in ("old_velocity_z", "velocity_z"):
        _number(fall_input[field], f"fall response input.{field}", -32768, 32768)
    _number(fall_input["grapple_release_elapsed"],
            "fall response input.grapple_release_elapsed", -86400, 86400)
    _number(fall_input["fall_damagemod"], "fall response input.fall_damagemod", 0, 1000)
    _integer(fall_input["modelindex"], "fall response input.modelindex", 0, 255)
    _integer(fall_input["movetype"], "fall response input.movetype", 0, 9)
    _integer(fall_input["grapple_state"], "fall response input.grapple_state", 0, 2)
    _integer(fall_input["waterlevel"], "fall response input.waterlevel", 0, 3)
    _integer(fall_input["dmflags"], "fall response input.dmflags", 0, 2147483647)
    _integer(fall_input["health"], "fall response input.health", -1_000_000, 1_000_000)
    for field in ("grounded", "hook_out", "grapple_present", "deathmatch"):
        _boolean(fall_input[field], f"fall response input.{field}")
    expected_input = {field: request[field] for field in FALL_INPUT_KEYS}
    _reject(dict(fall_input) != expected_input, "invalid_fall_evidence",
            "fall response input differs from exact landing request")
    _reject(record["suppression"] not in {
        "none", "not_player_model", "noclip", "hook_out", "airborne",
        "grapple", "underwater", "below_threshold",
    }, "invalid_fall_evidence", "fall suppression is invalid")
    _reject(record["severity"] not in {"none", "footstep", "short", "fall", "far"},
            "invalid_fall_evidence", "fall severity is invalid")
    _number(record["delta"], "fall delta", 0, 10_000_000)
    _number(record["fall_value"], "fall value", 0, 40)
    _number(record["fall_time_offset"], "fall time offset", 0, 86400)
    for field in ("emit_event", "set_fall_state", "set_pain_debounce", "apply_damage", "unmitigated_lethal"):
        _boolean(record[field], f"fall response.{field}")
    damage = _integer(record["damage"], "fall response.damage", 0, 2147483647)
    health_after = _integer(record["unmitigated_health_after"],
                            "fall response.unmitigated_health_after", -2147483648, 2147483647)
    if record["apply_damage"]:
        _reject(health_after != request["health"] - damage,
                "invalid_fall_evidence", "fall unmitigated health projection mismatch")
    else:
        _reject(health_after != request["health"] or record["unmitigated_lethal"],
                "invalid_fall_evidence", "suppressed fall damage changed health or lethality")
    _reject(record["unmitigated_lethal"] and
            (not record["apply_damage"] or damage <= 0 or health_after > 0),
            "invalid_fall_evidence", "fall lethal flag is inconsistent")
    return dict(record)


def _decode_json_line(line: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON token {value}")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = item
        return result
    value = json.loads(
        line, parse_constant=reject_constant, object_pairs_hook=unique_object,
    )
    _reject(not isinstance(value, dict), "oracle_failure", "oracle response must be an object")
    return value


def _subprocess_runner(
    kind: str, executable: Path, requests: Sequence[dict[str, Any]], map_path: Path | None,
) -> list[dict[str, Any]]:
    command = [str(executable)]
    if kind == "pmove":
        _reject(map_path is None, "pmove_oracle_failure", "pmove map path is missing")
        command += ["--map", str(map_path)]
    payload = b"".join(_canonical(request) + b"\n" for request in requests)
    try:
        completed = subprocess.run(
            command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DropReplayError(f"{kind}_oracle_failure", str(error)) from error
    _reject(completed.returncode != 0, f"{kind}_oracle_failure",
            f"oracle exited {completed.returncode}")
    _reject(len(completed.stdout) > MAX_ORACLE_STDOUT_BYTES, f"{kind}_oracle_failure",
            "oracle response exceeds 8 MiB")
    try:
        text = completed.stdout.decode("utf-8", errors="strict")
        lines = text.splitlines()
        _reject(len(lines) != len(requests) or any(not line for line in lines),
                f"{kind}_oracle_failure", "oracle did not return exactly one record per request")
        return [_decode_json_line(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DropReplayError(f"{kind}_oracle_failure", f"invalid oracle JSON: {error}") from error


def _unknown(replay_id: str, reason: str, detail: str) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "id": replay_id,
        "classification": "Unknown",
        "omit_controlled_drop_edge": True,
        "reason": reason,
        "detail": detail,
    }


def _pmove_requests(normalized: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_id = normalized["id"]
    pmove = normalized["pmove"]
    return (
        {
            "id": f"{replay_id}:pmove-id", "op": "identity",
            "gravity": pmove["gravity"], "airaccelerate": pmove["airaccelerate"],
        },
        {
            "id": f"{replay_id}:pmove", "op": "simulate", **pmove,
        },
    )


def pmove_requests_for_drop_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Purely validate a manifest and construct its two exact Pmove requests."""
    replay_id = manifest.get("id", "") if isinstance(manifest, Mapping) else ""
    if not isinstance(replay_id, str):
        replay_id = ""
    try:
        normalized = _validate_manifest(manifest)
        replay_id = normalized["id"]
        _reject(normalized["dynamic_movers"], "unsupported_dynamic_mover",
                "q2-pmove-oracle v1 has no dynamic brush-mover state conduit")
        identity_request, simulate_request = _pmove_requests(normalized)
        return {
            "schema": PMOVE_REQUEST_SCHEMA,
            "id": replay_id,
            "classification": "NeedsPmoveAuthority",
            "identity_request": identity_request,
            "simulate_request": simulate_request,
        }
    except DropReplayError as error:
        return _unknown(replay_id, error.reason, error.detail)
    except (ValueError, TypeError, KeyError) as error:
        return _unknown(replay_id, "invalid_contract", str(error))


def prepare_drop_fall_request(
    manifest: Mapping[str, Any], *, pmove_identity: Mapping[str, Any],
    pmove_response: Mapping[str, Any], pmove_executable_sha256: str,
    map_sha256: str,
) -> dict[str, Any]:
    """Validate ordered Pmove evidence and derive the one exact fall request.

    This pure stage is intended for analyzers that batch or persist oracle
    processes. It performs no file access and launches no subprocess.
    """
    replay_id = manifest.get("id", "") if isinstance(manifest, Mapping) else ""
    if not isinstance(replay_id, str):
        replay_id = ""
    try:
        normalized = _validate_manifest(manifest)
        replay_id = normalized["id"]
        _reject(normalized["dynamic_movers"], "unsupported_dynamic_mover",
                "q2-pmove-oracle v1 has no dynamic brush-mover state conduit")
        expected_pmove = normalized["authorities"]["pmove"]
        supplied_executable = _sha256(pmove_executable_sha256, "pmove executable SHA-256")
        supplied_map = _sha256(map_sha256, "map SHA-256")
        _reject(supplied_executable != expected_pmove["executable_sha256"],
                "authority_digest_mismatch", "pmove executable SHA-256 mismatch")
        _reject(supplied_map != expected_pmove["map_sha256"],
                "authority_digest_mismatch", "map SHA-256 mismatch")
        identity_request, simulate_request = _pmove_requests(normalized)
        admitted_identity = _validate_pmove_identity(
            pmove_identity, identity_request["id"], expected_pmove,
            normalized["pmove"]["gravity"], normalized["pmove"]["airaccelerate"],
        )
        admitted_response, frames = _validate_pmove_simulation(
            pmove_response, simulate_request["id"], admitted_identity,
            normalized["horizon_frames"], normalized["pmove"]["gravity"],
        )
        landing_index = next(
            (index for index in range(1, len(frames))
             if not frames[index - 1]["grounded"] and frames[index]["grounded"]),
            None,
        )
        _reject(landing_index is None, "no_landing",
                "no false-to-true grounded transition occurred inside the declared horizon")
        previous = frames[landing_index - 1]
        landing = frames[landing_index]
        fall = normalized["fall"]
        fall_identity_request = {
            "id": f"{replay_id}:fall-id", "op": "identity",
            "fall_damagemod": fall["fall_damagemod"],
            "deathmatch": fall["deathmatch"], "dmflags": fall["dmflags"],
        }
        fall_request = {
            "id": f"{replay_id}:fall",
            "op": "evaluate",
            "old_velocity_z": previous["velocity"][2],
            "velocity_z": landing["velocity"][2],
            "grapple_release_elapsed": 1.0,
            "fall_damagemod": fall["fall_damagemod"],
            "modelindex": 255,
            "movetype": 4,
            "grounded": True,
            "hook_out": False,
            "grapple_present": False,
            "grapple_state": 0,
            "waterlevel": landing["waterlevel"],
            "deathmatch": fall["deathmatch"],
            "dmflags": fall["dmflags"],
            "health": fall["health"],
        }
        trajectory_payload = {
            "schema": "q2-pmove-drop-trajectory-evidence-v1",
            "map_sha256": supplied_map,
            "physics_identity": admitted_identity["physics_identity"],
            "request": simulate_request,
            "frames": frames,
            "final": admitted_response["final"],
        }
        return {
            "schema": PREPARED_SCHEMA,
            "id": replay_id,
            "classification": "NeedsFallAuthority",
            "landing": {
                "preceding_command_index": previous["command_index"],
                "command_index": landing["command_index"],
                "origin_fixed": list(landing["origin_fixed"]),
                "origin": list(landing["origin"]),
                "old_velocity_z": previous["velocity"][2],
                "velocity_z": landing["velocity"][2],
                "waterlevel": landing["waterlevel"],
            },
            "trajectory_sha256": hashlib.sha256(_canonical(trajectory_payload)).hexdigest(),
            "pmove_request": simulate_request,
            "fall_identity_request": fall_identity_request,
            "fall_request": fall_request,
            "pmove_executable_sha256": supplied_executable,
            "pmove_identity": admitted_identity,
            "pmove_response_binding": {
                "schema": admitted_response["schema"],
                "tool_identity": admitted_response["tool_identity"],
                "physics_identity": admitted_response["physics_identity"],
                "map_sha256": admitted_response["map_sha256"],
                "map_checksum": admitted_response["map_checksum"],
                "command_count": admitted_response["command_count"],
            },
        }
    except DropReplayError as error:
        return _unknown(replay_id, error.reason, error.detail)
    except (ValueError, TypeError, KeyError) as error:
        return _unknown(replay_id, "invalid_contract", str(error))


def evaluate_drop_evidence(
    manifest: Mapping[str, Any], *, pmove_identity: Mapping[str, Any],
    pmove_response: Mapping[str, Any], fall_identity: Mapping[str, Any],
    fall_response: Mapping[str, Any], pmove_executable_sha256: str,
    fall_executable_sha256: str, map_sha256: str,
) -> dict[str, Any]:
    """Purely admit returned oracle mappings and produce Exact or Unknown."""
    replay_id = manifest.get("id", "") if isinstance(manifest, Mapping) else ""
    if not isinstance(replay_id, str):
        replay_id = ""
    prepared = prepare_drop_fall_request(
        manifest,
        pmove_identity=pmove_identity,
        pmove_response=pmove_response,
        pmove_executable_sha256=pmove_executable_sha256,
        map_sha256=map_sha256,
    )
    if prepared["classification"] == "Unknown":
        return prepared
    try:
        normalized = _validate_manifest(manifest)
        replay_id = normalized["id"]
        expected_fall = normalized["authorities"]["fall"]
        supplied_fall = _sha256(fall_executable_sha256, "fall executable SHA-256")
        _reject(supplied_fall != expected_fall["executable_sha256"],
                "authority_digest_mismatch", "fall executable SHA-256 mismatch")
        admitted_fall_identity = _validate_fall_identity(
            fall_identity, prepared["fall_identity_request"]["id"],
            expected_fall, normalized["fall"],
        )
        admitted_fall_response = _validate_fall_evaluation(
            fall_response, prepared["fall_request"]["id"],
            admitted_fall_identity, prepared["fall_request"],
        )
        lethal = admitted_fall_response["unmitigated_lethal"]
        return {
            "schema": RESULT_SCHEMA,
            "id": replay_id,
            "classification": "Exact",
            "safe": not lethal,
            "lethal": lethal,
            "severity": admitted_fall_response["severity"],
            "landing": prepared["landing"],
            "trajectory_sha256": prepared["trajectory_sha256"],
            "pmove_request": prepared["pmove_request"],
            "fall_request": prepared["fall_request"],
            "authorities": {
                "pmove": {
                    "executable_sha256": prepared["pmove_executable_sha256"],
                    "identity": prepared["pmove_identity"],
                    "response_binding": prepared["pmove_response_binding"],
                },
                "fall": {
                    "executable_sha256": supplied_fall,
                    "identity": admitted_fall_identity,
                    "response": admitted_fall_response,
                },
            },
        }
    except DropReplayError as error:
        return _unknown(replay_id, error.reason, error.detail)
    except (ValueError, TypeError, KeyError) as error:
        return _unknown(replay_id, "invalid_contract", str(error))


def replay_drop(
    manifest: Mapping[str, Any], *, pmove_oracle: str | Path | None,
    fall_oracle: str | Path | None, runner: OracleRunner | None = None,
) -> dict[str, Any]:
    """Replay one controlled drop and emit Exact only under both pinned authorities."""
    replay_id = manifest.get("id", "") if isinstance(manifest, Mapping) else ""
    if not isinstance(replay_id, str):
        replay_id = ""
    try:
        normalized = _validate_manifest(manifest)
        replay_id = normalized["id"]
        _reject(normalized["dynamic_movers"], "unsupported_dynamic_mover",
                "q2-pmove-oracle v1 has no dynamic brush-mover state conduit")
        _reject(pmove_oracle is None, "missing_pmove_authority", "q2-pmove-oracle path is required")
        _reject(fall_oracle is None, "missing_fall_authority", "q2-fall-oracle path is required")
        pmove_path = Path(pmove_oracle)
        fall_path = Path(fall_oracle)
        map_path = Path(normalized["map_path"])
        for path, label in ((pmove_path, "pmove"), (fall_path, "fall"), (map_path, "map")):
            _reject(not path.is_file(), f"missing_{label}_authority", f"{label} file is missing")
        expected_pmove = normalized["authorities"]["pmove"]
        expected_fall = normalized["authorities"]["fall"]
        pmove_executable_sha256 = _file_sha256(pmove_path)
        fall_executable_sha256 = _file_sha256(fall_path)
        _reject(pmove_executable_sha256 != expected_pmove["executable_sha256"],
                "authority_digest_mismatch", "pmove executable SHA-256 mismatch")
        _reject(fall_executable_sha256 != expected_fall["executable_sha256"],
                "authority_digest_mismatch", "fall executable SHA-256 mismatch")
        map_sha256 = _file_sha256(map_path)
        _reject(map_sha256 != expected_pmove["map_sha256"],
                "authority_digest_mismatch", "map SHA-256 mismatch")

        invoke = runner or _subprocess_runner
        identity_request, simulate_request = _pmove_requests(normalized)
        pmove_records = invoke("pmove", pmove_path, [identity_request, simulate_request], map_path)
        _reject(len(pmove_records) != 2, "pmove_oracle_failure",
                "pmove oracle returned the wrong record count")
        prepared = prepare_drop_fall_request(
            normalized,
            pmove_identity=pmove_records[0],
            pmove_response=pmove_records[1],
            pmove_executable_sha256=pmove_executable_sha256,
            map_sha256=map_sha256,
        )
        if prepared["classification"] == "Unknown":
            return prepared
        fall_records = invoke(
            "fall", fall_path,
            [prepared["fall_identity_request"], prepared["fall_request"]], None,
        )
        _reject(len(fall_records) != 2, "fall_oracle_failure",
                "fall oracle returned the wrong record count")
        return evaluate_drop_evidence(
            normalized,
            pmove_identity=pmove_records[0],
            pmove_response=pmove_records[1],
            fall_identity=fall_records[0],
            fall_response=fall_records[1],
            pmove_executable_sha256=pmove_executable_sha256,
            fall_executable_sha256=fall_executable_sha256,
            map_sha256=map_sha256,
        )
    except DropReplayError as error:
        return _unknown(replay_id, error.reason, error.detail)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _unknown(replay_id, "invalid_contract", str(error))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result = _unknown("", "invalid_contract", str(error))
    else:
        result = replay_drop(
            manifest, pmove_oracle=args.pmove_oracle, fall_oracle=args.fall_oracle,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if result["classification"] == "Exact" else 65


if __name__ == "__main__":
    raise SystemExit(main())
