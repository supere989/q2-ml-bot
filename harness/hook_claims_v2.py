"""Strict source-bound hook candidate and materialization contracts.

Generator candidates are proposals.  Only a canonical materialization record
whose identities match the compiled BSP and source metadata can be used to
construct generator claims, and its ``passed`` bit is never replay authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CANDIDATE_SCHEMA = "q2-hook-claim-candidates-v2"
MATERIALIZATION_SCHEMA = "q2-hook-claim-materialization-v2"
RUNTIME_RECORD_COUNT = 6
RECORD_KEYS = {
    "claim_id", "source_milliunits", "anchor_milliunits",
    "landing_milliunits", "release_after_ticks", "distance_milliunits",
    "flags",
}
TRACE_KEYS = {
    "claim_id", "origin_fixed_frames", "first_grounded_frame_index", "sha256",
}
_CLAIM_RE = re.compile(r"^hook:[0-9]{4}:candidate:[0-9]{4}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class HookClaimsV2Error(RuntimeError):
    """A hook-v2 artifact is malformed, stale, or not canonically encoded."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise HookClaimsV2Error(
            f"{label} keys differ: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HookClaimsV2Error(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise HookClaimsV2Error(f"{label} must be at least {minimum}")
    return value


def _vec3(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise HookClaimsV2Error(f"{label} must be three integers")
    return [_integer(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise HookClaimsV2Error(f"{label} is not a lowercase SHA-256")
    return value


def validate_record(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, RECORD_KEYS, label)
    claim_id = record["claim_id"]
    if not isinstance(claim_id, str) or not _CLAIM_RE.fullmatch(claim_id):
        raise HookClaimsV2Error(f"{label} has invalid claim_id")
    source = _vec3(record["source_milliunits"], f"{label}.source_milliunits")
    anchor = _vec3(record["anchor_milliunits"], f"{label}.anchor_milliunits")
    landing = _vec3(record["landing_milliunits"], f"{label}.landing_milliunits")
    release = _integer(record["release_after_ticks"], f"{label}.release_after_ticks", 1)
    if release > 49:
        raise HookClaimsV2Error(f"{label}.release_after_ticks exceeds hook lifetime")
    distance = _integer(record["distance_milliunits"], f"{label}.distance_milliunits", 1)
    flags = _integer(record["flags"], f"{label}.flags", 0)
    if flags > 7:
        raise HookClaimsV2Error(f"{label}.flags exceeds 7")
    eye = (source[0], source[1], source[2] + 22_000)
    expected_distance = round(math.sqrt(sum(
        (anchor[axis] - eye[axis]) ** 2 for axis in range(3)
    )))
    if distance != expected_distance:
        raise HookClaimsV2Error(f"{label}.distance_milliunits is not source-bound")
    if any(value % 125 for value in source):
        raise HookClaimsV2Error(f"{label}.source_milliunits is not exact Pmove fixed-point")
    return {
        "claim_id": claim_id,
        "source_milliunits": source,
        "anchor_milliunits": anchor,
        "landing_milliunits": landing,
        "release_after_ticks": release,
        "distance_milliunits": distance,
        "flags": flags,
    }


def validate_candidates(value: Any) -> dict[str, Any]:
    container = _mapping(value, {
        "schema", "tick_msec", "status", "bundle_admissible", "records",
    }, "hook candidate container")
    if (
        container["schema"] != CANDIDATE_SCHEMA
        or container["tick_msec"] != 100
        or container["status"] != "unproven"
        or container["bundle_admissible"] is not False
    ):
        raise HookClaimsV2Error("hook candidate container is not frozen v2")
    values = container["records"]
    if not isinstance(values, list) or not values:
        raise HookClaimsV2Error("hook candidate container has no records")
    records = [validate_record(item, f"candidate {index}") for index, item in enumerate(values)]
    ids = [record["claim_id"] for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise HookClaimsV2Error("hook candidate IDs are not unique and stably ordered")
    return {
        "schema": CANDIDATE_SCHEMA,
        "tick_msec": 100,
        "status": "unproven",
        "bundle_admissible": False,
        "records": records,
    }


def load_candidates(meta_path: Path) -> tuple[dict[str, Any], str, str]:
    raw = meta_path.read_bytes()
    try:
        metadata = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HookClaimsV2Error("generator metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise HookClaimsV2Error("generator metadata is not an object")
    candidates = validate_candidates(metadata.get("hook_claim_candidates_v2"))
    return (
        candidates,
        sha256_bytes(canonical_json(candidates) + b"\n"),
        sha256_bytes(raw),
    )


def runtime_rows(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = []
    for record in records:
        anchor = record["anchor_milliunits"]
        landing = record["landing_milliunits"]
        values = [*anchor, *landing, record["distance_milliunits"]]
        formatted = [f"{value / 1000.0:.3f}" for value in values]
        lines.append(" ".join([*formatted, str(record["flags"])]))
    return ("\n".join(lines) + "\n").encode("ascii")


def runtime_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(runtime_rows(records))


def render_runtime_sidecar(
    map_id: str,
    bsp_sha256: str,
    materialization_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    records_sha256 = runtime_records_sha256(records)
    header = [
        "# q2-ml-bot hook zones - compiled exact materialization v2",
        "# schema: q2-hook-runtime-sidecar-v2",
        "# passed: true",
        "# bundle_admissible: true",
        f"# map: {map_id}",
        f"# bsp_sha256: {bsp_sha256}",
        f"# materialization_sha256: {materialization_sha256}",
        f"# runtime_records_sha256: {records_sha256}",
        f"# records: {len(records)}",
        "# anchor.xyz landing.xyz distance flags",
    ]
    return ("\n".join(header) + "\n").encode("ascii") + runtime_rows(records)


def validate_runtime_sidecar(
    payload: bytes,
    *,
    map_id: str,
    bsp_sha256: str,
    materialization_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected = render_runtime_sidecar(
        map_id, bsp_sha256, materialization_sha256, records
    )
    if payload != expected:
        raise HookClaimsV2Error(
            "runtime hook sidecar header/rows differ from materialization"
        )


def validation_trace_sha256(
    claim_id: str,
    origin_fixed_frames: Sequence[Sequence[int]],
    first_grounded_frame_index: int,
) -> str:
    return sha256_bytes(canonical_json({
        "claim_id": claim_id,
        "origin_fixed_frames": [list(frame) for frame in origin_fixed_frames],
        "first_grounded_frame_index": first_grounded_frame_index,
    }) + b"\n")


def validate_trace(value: Any, label: str) -> dict[str, Any]:
    trace = _mapping(value, TRACE_KEYS, label)
    claim_id = trace["claim_id"]
    if not isinstance(claim_id, str) or not _CLAIM_RE.fullmatch(claim_id):
        raise HookClaimsV2Error(f"{label}.claim_id is invalid")
    frames = trace["origin_fixed_frames"]
    if not isinstance(frames, list) or not frames:
        raise HookClaimsV2Error(f"{label}.origin_fixed_frames is empty")
    normalized_frames = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != 3:
            raise HookClaimsV2Error(f"{label} frame {index} is not fixed3")
        normalized_frames.append([
            _integer(axis, f"{label} frame {index}") for axis in frame
        ])
    grounded = _integer(
        trace["first_grounded_frame_index"],
        f"{label}.first_grounded_frame_index", 0,
    )
    if grounded != len(normalized_frames) - 1:
        raise HookClaimsV2Error(f"{label} does not end at first grounded frame")
    expected = validation_trace_sha256(claim_id, normalized_frames, grounded)
    if _sha(trace["sha256"], f"{label}.sha256") != expected:
        raise HookClaimsV2Error(f"{label}.sha256 differs")
    return {
        "claim_id": claim_id,
        "origin_fixed_frames": normalized_frames,
        "first_grounded_frame_index": grounded,
        "sha256": expected,
    }


def validate_materialization(value: Any) -> dict[str, Any]:
    document = _mapping(value, {
        "schema", "map", "passed", "bsp", "candidates", "source_projection_sha256",
        "runtime_records_sha256", "selected_records", "validation_traces",
        "oracles", "replay",
        "request_count",
    }, "hook materialization")
    if document["schema"] != MATERIALIZATION_SCHEMA or document["passed"] is not True:
        raise HookClaimsV2Error("hook materialization is not a passed v2 record")
    if not isinstance(document["map"], str) or not document["map"]:
        raise HookClaimsV2Error("hook materialization map is invalid")
    bsp = _mapping(document["bsp"], {"sha256", "size_bytes"}, "materialization BSP")
    _sha(bsp["sha256"], "materialization BSP digest")
    _integer(bsp["size_bytes"], "materialization BSP size", 1)
    candidates = _mapping(document["candidates"], {
        "meta_sha256", "records_sha256", "record_count",
    }, "materialization candidates")
    _sha(candidates["meta_sha256"], "candidate metadata digest")
    _sha(candidates["records_sha256"], "candidate records digest")
    _integer(candidates["record_count"], "candidate record count", RUNTIME_RECORD_COUNT)
    _sha(document["source_projection_sha256"], "source projection digest")
    _sha(document["runtime_records_sha256"], "runtime records digest")
    selected = document["selected_records"]
    if not isinstance(selected, list) or len(selected) != RUNTIME_RECORD_COUNT:
        raise HookClaimsV2Error("materialization must select exactly six records")
    records = [validate_record(item, f"selected record {index}") for index, item in enumerate(selected)]
    ids = [record["claim_id"] for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise HookClaimsV2Error("selected records are not uniquely and stably ordered")
    geometries = [
        (tuple(record["anchor_milliunits"]), tuple(record["landing_milliunits"]), record["flags"])
        for record in records
    ]
    if len(set(geometries)) != RUNTIME_RECORD_COUNT:
        raise HookClaimsV2Error("selected runtime geometries are not unique")
    if runtime_records_sha256(records) != document["runtime_records_sha256"]:
        raise HookClaimsV2Error("materialization runtime record digest differs")
    traces_value = document["validation_traces"]
    if not isinstance(traces_value, list) or len(traces_value) != RUNTIME_RECORD_COUNT:
        raise HookClaimsV2Error("materialization must have six validation traces")
    traces = [
        validate_trace(item, f"validation trace {index}")
        for index, item in enumerate(traces_value)
    ]
    if [trace["claim_id"] for trace in traces] != ids:
        raise HookClaimsV2Error("validation traces differ from selected records")
    oracles = _mapping(document["oracles"], {
        "collision", "pmove", "hook", "fall",
        "hook_parity_attestation_sha256", "b1_runtime_authority_seal",
    }, "materialization oracles")
    for name in ("collision", "pmove", "hook", "fall"):
        oracle = _mapping(oracles[name], {
            "executable_sha256", "tool_identity", "physics_identity", "requests",
        }, f"materialization {name} oracle")
        for key in ("executable_sha256", "tool_identity", "physics_identity"):
            _sha(oracle[key], f"materialization {name}.{key}")
        _integer(oracle["requests"], f"materialization {name}.requests", 1)
    _sha(oracles["hook_parity_attestation_sha256"], "hook parity attestation digest")
    seal = _mapping(oracles["b1_runtime_authority_seal"], {
        "schema", "normative_documents", "hook_parity_attestation_sha256",
        "fixture_bsp_sha256", "analysis_bsp_sha256", "executables", "identities",
    }, "materialization B1 runtime authority seal")
    if seal["schema"] != "q2-b1-runtime-authority-seal-v1":
        raise HookClaimsV2Error("materialization B1 seal schema differs")
    normative = _mapping(
        seal["normative_documents"], {"design_sha256", "plan_sha256"},
        "materialization B1 normative documents",
    )
    executables = _mapping(
        seal["executables"],
        {"cm_sha256", "pmove_sha256", "hook_sha256", "fall_sha256"},
        "materialization B1 executables",
    )
    identities = _mapping(
        seal["identities"], {"collision", "pmove", "hook", "fall"},
        "materialization B1 identities",
    )
    for name, digest in (
        *(normative.items()),
        ("hook_parity_attestation_sha256", seal["hook_parity_attestation_sha256"]),
        ("fixture_bsp_sha256", seal["fixture_bsp_sha256"]),
        ("analysis_bsp_sha256", seal["analysis_bsp_sha256"]),
        *(executables.items()),
    ):
        _sha(digest, f"materialization B1 {name}")
    for name in ("collision", "pmove", "hook", "fall"):
        identity = _mapping(
            identities[name], {"tool_identity", "physics_identity"},
            f"materialization B1 {name} identity",
        )
        _sha(identity["tool_identity"], f"materialization B1 {name} tool")
        _sha(identity["physics_identity"], f"materialization B1 {name} physics")
        if any(
            identity[field] != oracles[name][field]
            for field in ("tool_identity", "physics_identity")
        ):
            raise HookClaimsV2Error(
                f"materialization {name} identity differs from retained B1 seal"
            )
    if (
        seal["hook_parity_attestation_sha256"]
        != oracles["hook_parity_attestation_sha256"]
        or seal["analysis_bsp_sha256"] != bsp["sha256"]
        or any(
            oracles[name]["executable_sha256"]
            != executables["cm_sha256" if name == "collision" else f"{name}_sha256"]
            for name in ("collision", "pmove", "hook", "fall")
        )
    ):
        raise HookClaimsV2Error("materialization B1 seal binding differs")
    replay = _mapping(document["replay"], {
        "analyzer", "analyzer_version", "verifier", "verifier_version",
    }, "materialization replay")
    if replay != {
        "analyzer": "q2-hook-claim-materializer",
        "analyzer_version": "b2-c-v2",
        "verifier": "q2-atlas-analyzer-exact-hook-replay",
        "verifier_version": "b2-a-v2",
    }:
        raise HookClaimsV2Error("materialization replay versions differ")
    _integer(document["request_count"], "materialization request count", 1)
    return document


def load_materialization(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HookClaimsV2Error("hook materialization is not valid JSON") from error
    document = validate_materialization(value)
    canonical = canonical_json(document) + b"\n"
    if raw != canonical:
        raise HookClaimsV2Error("hook materialization is not canonical JSON plus LF")
    return document, sha256_bytes(canonical)
