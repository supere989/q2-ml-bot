"""Deterministic offline Atlas builder using only pinned engine physics oracles."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import zstandard

from .ibsp38 import BspMetadata, EntityMetadata, parse_ibsp38


SCHEMA = "q2-atlas-analysis-v1"
BUILD_PLAN_SCHEMA = "q2-atlas-build-plan-v1"
EVIDENCE_CM_TRACE_V1 = 1
EVIDENCE_PMOVE_V1 = 2
VALIDATION_VERSION = 1
MASK_PLAYERSOLID = 33_619_971
MASK_SHOT = 100_663_299
CONTENTS_SOLID = 1
CONTENTS_WINDOW = 2
CONTENTS_LAVA = 8
CONTENTS_SLIME = 16
CONTENTS_WATER = 32
CONTENTS_MIST = 64
CONTENTS_PLAYERCLIP = 0x10000
CONTENTS_MONSTERCLIP = 0x20000
CONTENTS_CURRENT_0 = 0x40000
CONTENTS_CURRENT_90 = 0x80000
CONTENTS_CURRENT_180 = 0x100000
CONTENTS_CURRENT_270 = 0x200000
CONTENTS_CURRENT_UP = 0x400000
CONTENTS_CURRENT_DOWN = 0x800000
CONTENTS_LADDER = 0x20000000
CONTENTS_CURRENT_MASK = (
    CONTENTS_CURRENT_0 | CONTENTS_CURRENT_90 | CONTENTS_CURRENT_180
    | CONTENTS_CURRENT_270 | CONTENTS_CURRENT_UP | CONTENTS_CURRENT_DOWN
)
CONTENTS_FLUID_MASK = CONTENTS_WATER | CONTENTS_SLIME | CONTENTS_LAVA
SURF_SKY = 4
SURF_SLICK = 2
SURF_WARP = 8
SURF_NODRAW = 128
STANDING_MINS = [-16, -16, -24]
STANDING_MAXS = [16, 16, 32]
CROUCHED_MINS = [-16, -16, -24]
CROUCHED_MAXS = [16, 16, 4]
FROZEN_L0_BIT_PLANE_NAMES = frozenset({
    "solid", "window", "playerclip", "monsterclip", "water", "slime", "lava",
    "mist", "ladder", "hurt", "push_or_gravity", "teleport_trigger",
    "mover_reference_solid", "mover_swept_envelope", "areaportal", "sky",
    "slick", "warp", "nodraw", "hookable_surface", "standing_forbidden_origin",
    "crouched_forbidden_origin", "unknown", "spawn_column", "hook_corridor",
})
FROZEN_L0_SCALAR_PLANE_NAMES = frozenset({
    "current_direction", "hazard_severity", "confidence",
})
SHA256_KEYS = {"tool_identity", "physics_identity", "map_sha256"}
COMMON_RESPONSE_KEYS = {
    "ok", "id", "op", "schema", "tool_identity", "physics_identity",
    "map_sha256", "map_checksum",
}


class AtlasAnalysisError(RuntimeError):
    """The map cannot be analyzed without guessing or exceeding a bound."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _rust_struct_json(value: Any) -> bytes:
    """Match serde struct declaration order for B1 canonical contract seals."""
    return json.dumps(value, sort_keys=False, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AtlasAnalysisError(f"{field} is not a lowercase SHA-256")
    return value


def _exact_keys(value: Any, expected: set[str], field: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise AtlasAnalysisError(
            f"{field} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


@dataclass(frozen=True)
class AnalyzerLimits:
    # Oracle stdout records are deliberately verbose and a 512-record write
    # can fill the child pipe before the parent begins reading. Thirty-two is
    # below that observed transport ceiling while remaining efficiently batched.
    oracle_batch: int = 32
    oracle_batch_timeout_seconds: float = 10.0
    process_exit_timeout_seconds: float = 2.0
    max_l1_nodes: int = 50_000
    max_l1_edges: int = 250_000
    max_oracle_requests: int = 2_000_000
    flood_source_batch: int = 128
    max_visibility_pairs: int = 16_384
    max_pmove_sources: int = 2_048
    full_cold_timeout_seconds: float = 300.0


class OracleProcess:
    """One map-bound NDJSON oracle with bounded requests and hard deadlines."""

    def __init__(
        self,
        binary: Path,
        bsp: Path,
        kind: str,
        limits: AnalyzerLimits,
    ) -> None:
        self.binary = binary.resolve()
        self.bsp = bsp.resolve()
        self.kind = kind
        self.limits = limits
        self.requests = 0
        self._buffer = bytearray()
        # The integrated legacy CM loader has a reproducible trace crash when
        # handed sufficiently long BSP pathnames. Keep authority byte-exact,
        # but bind the child to a private short path and verify its digest
        # against the source after staging. This is transport, not conversion.
        self._staging = tempfile.TemporaryDirectory(prefix="qa-", dir="/tmp")
        oracle_bsp = Path(self._staging.name) / "m.bsp"
        try:
            os.link(self.bsp, oracle_bsp)
        except OSError:
            shutil.copyfile(self.bsp, oracle_bsp)
        if sha256_file(oracle_bsp) != sha256_file(self.bsp):
            self._staging.cleanup()
            raise AtlasAnalysisError(f"{kind} short-path staging digest mismatch")
        self.process = subprocess.Popen(
            [str(self.binary), "--map", str(oracle_bsp)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise AtlasAnalysisError(f"failed to open {kind} oracle pipes")
        identity = self.call([{"id": "identity", "op": "identity"}])[0]
        self.identity = self._validate_identity(identity)

    def _read_lines(self, count: int) -> list[dict]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + self.limits.oracle_batch_timeout_seconds
        output: list[dict] = []
        while len(output) < count:
            while b"\n" in self._buffer and len(output) < count:
                raw, _, remainder = self._buffer.partition(b"\n")
                self._buffer[:] = remainder
                try:
                    output.append(json.loads(raw))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise AtlasAnalysisError(f"{self.kind} emitted invalid JSON") from error
            if len(output) == count:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise AtlasAnalysisError(f"{self.kind} oracle batch timed out")
            ready, _, _ = select.select([self.process.stdout.fileno()], [], [], remaining)
            if not ready:
                self._kill()
                raise AtlasAnalysisError(f"{self.kind} oracle batch timed out")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                detail = b""
                if self.process.stderr is not None:
                    detail = self.process.stderr.read(4096)
                raise AtlasAnalysisError(
                    f"{self.kind} oracle exited early: {detail.decode(errors='replace')}"
                )
            self._buffer.extend(block)
        return output

    def call(self, requests: Sequence[dict]) -> list[dict]:
        if not requests:
            return []
        batch_limit = self.limits.oracle_batch
        if self.kind == "pmove" and any(len(request.get("commands", [])) > 4 for request in requests):
            # Multi-frame Pmove responses are large enough that even 32 can
            # fill a pipe before the parent reads. Keep trajectory batches
            # below the measured pipe capacity.
            batch_limit = min(batch_limit, 4)
        if len(requests) > batch_limit:
            output: list[dict] = []
            for start in range(0, len(requests), batch_limit):
                output.extend(self.call(requests[start:start + batch_limit]))
            return output
        self.requests += len(requests)
        if self.requests > self.limits.max_oracle_requests:
            raise AtlasAnalysisError(f"{self.kind} oracle request budget exceeded")
        assert self.process.stdin is not None
        payload = b"".join(canonical_json(request) + b"\n" for request in requests)
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise AtlasAnalysisError(f"{self.kind} oracle pipe closed") from error
        records = self._read_lines(len(requests))
        for request, record in zip(requests, records):
            if record.get("id") != request.get("id"):
                raise AtlasAnalysisError(f"{self.kind} response ID mismatch")
            if record.get("ok") is not True:
                raise AtlasAnalysisError(
                    f"{self.kind} rejected {request.get('op')}: {record.get('error')}"
                )
            if hasattr(self, "identity"):
                self._validate_response(record)
        return records

    def _validate_identity(self, record: dict) -> dict:
        if self.kind == "cm":
            expected = COMMON_RESPONSE_KEYS | {
                "provenance", "source", "map", "model0", "clusters", "inline_models",
            }
            source_keys = {"collision_sha256", "shared_header_sha256", "shared_source_sha256"}
            schema = "q2-cm-oracle-v1"
        else:
            expected = COMMON_RESPONSE_KEYS | {"parameters", "provenance", "source"}
            source_keys = {
                "collision_sha256", "pmove_sha256", "shared_header_sha256",
                "shared_source_sha256",
            }
            schema = "q2-pmove-oracle-v1"
        _exact_keys(record, expected, f"{self.kind} identity")
        if record["op"] != "identity" or record["schema"] != schema:
            raise AtlasAnalysisError(f"{self.kind} identity schema mismatch")
        for key in SHA256_KEYS:
            _require_sha(record[key], f"{self.kind}.{key}")
        source = _exact_keys(record["source"], source_keys, f"{self.kind} source")
        for key, digest in source.items():
            _require_sha(digest, f"{self.kind}.source.{key}")
        provenance = _exact_keys(record["provenance"], {
            "schema", "tool_identity", "source_closure_sha256", "source_closure_count",
            "build_identity_sha256", "compiler", "archiver", "build",
        }, f"{self.kind} provenance")
        if provenance["schema"] != "q2-oracle-tool-identity-v1":
            raise AtlasAnalysisError("oracle provenance schema mismatch")
        for key in ("tool_identity", "source_closure_sha256", "build_identity_sha256"):
            _require_sha(provenance[key], f"provenance.{key}")
        if provenance["tool_identity"] != record["tool_identity"]:
            raise AtlasAnalysisError("response/provenance tool identity mismatch")
        _exact_keys(provenance["compiler"], {
            "command", "version", "target", "executable_sha256",
        }, "compiler provenance")
        _exact_keys(provenance["archiver"], {
            "command", "version", "executable_sha256",
        }, "archiver provenance")
        _exact_keys(provenance["build"], {"cflags", "ldflags"}, "build provenance")
        if self.kind == "cm":
            model0 = _exact_keys(record["model0"], {"mins", "maxs", "headnode"}, "cm model0")
            for name in ("mins", "maxs"):
                values = model0[name]
                if (
                    not isinstance(values, list) or len(values) != 3
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in values)
                ):
                    raise AtlasAnalysisError(f"cm model0 {name} is invalid")
            if isinstance(model0["headnode"], bool) or not isinstance(model0["headnode"], int):
                raise AtlasAnalysisError("cm model0 headnode is invalid")
        return record

    def _validate_response(self, record: dict) -> None:
        for key in ("schema", "tool_identity", "map_sha256", "map_checksum"):
            if record.get(key) != self.identity.get(key):
                raise AtlasAnalysisError(f"{self.kind} {key} changed within process")
        # Physics identity is parameter-bound. All analyzer calls use the pinned
        # identity parameter block, so a mismatch is an admission failure.
        if record.get("physics_identity") != self.identity.get("physics_identity"):
            raise AtlasAnalysisError(f"{self.kind} physics identity mismatch")

    def _kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=self.limits.process_exit_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._kill()
            self.process.wait()
        if self.process.returncode:
            detail = b""
            if self.process.stderr is not None:
                detail = self.process.stderr.read(4096)
            raise AtlasAnalysisError(
                f"{self.kind} oracle exited {self.process.returncode}: "
                f"{detail.decode(errors='replace')}"
            )
        self._staging.cleanup()

    def __enter__(self) -> "OracleProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None:
            self._kill()
            self.process.wait()
            self._staging.cleanup()
        else:
            self.close()


@dataclass
class NavNode:
    index: tuple[int, int, int]
    position: tuple[float, float, float]
    standing_clear: bool
    crouched_clear: bool
    supported: bool
    contents: int
    floor_normal: tuple[float, float, float]
    region_id: int = 0
    floor_surface_flags: int = 0
    floor_surface_name: str = ""

    def plan(self, passable: bool) -> dict:
        safe = self.standing_clear and self.supported and not (
            self.contents & (CONTENTS_LAVA | CONTENTS_SLIME)
        )
        flags = 0
        if self.standing_clear:
            flags |= 1 << 0
        if self.crouched_clear:
            flags |= 1 << 1
        if safe:
            flags |= 1 << 2
        if self.supported:
            flags |= 1 << 3
        if self.standing_clear and passable:
            flags |= 1 << 4
        if self.crouched_clear and passable:
            flags |= 1 << 5
        hazard = 0
        if self.contents & CONTENTS_LAVA:
            hazard |= 1 << 0
        if self.contents & CONTENTS_SLIME:
            hazard |= 1 << 1
        return {
            "index": list(self.index),
            "flags": flags,
            "floor_normal_class": 1 if self.floor_normal[2] >= 0.7 else 0,
            "clearance_height": 56 if self.standing_clear else 28 if self.crouched_clear else 0,
            "hazard_types": hazard,
            "hazard_severity": 255 if hazard else 0,
            "hazard_clearance": -1 if hazard else 0,
            "cost_to_safety": 0 if safe else 0xFFFFFFFF,
            "region_id": self.region_id,
            "confidence": 65535,
            "evidence": EVIDENCE_CM_TRACE_V1,
            "contents_flags": self.contents & 0xFFFFFFFF,
        }


def _origin(entity: EntityMetadata) -> tuple[float, float, float]:
    words = entity.value("origin").split()
    if len(words) != 3:
        raise AtlasAnalysisError(f"entity {entity.index} has invalid origin")
    try:
        value = tuple(float(word) for word in words)
    except ValueError as error:
        raise AtlasAnalysisError(f"entity {entity.index} has invalid origin") from error
    if not all(math.isfinite(axis) for axis in value):
        raise AtlasAnalysisError(f"entity {entity.index} has non-finite origin")
    return value  # type: ignore[return-value]


def _snapped_origin(model_mins: Sequence[float]) -> tuple[int, int, int]:
    integer_mins = [math.floor(value) for value in model_mins]
    return tuple((value // 256) * 256 for value in integer_mins)  # type: ignore[return-value]


def _oracle_atlas_origin(metadata: BspMetadata, identity: Mapping[str, Any]) -> tuple[int, int, int]:
    """Bind Atlas coordinates to CM's expanded inline-model-0 authority."""
    model = identity["model0"]
    expected_mins = [value - 1.0 for value in metadata.model0.mins]
    expected_maxs = [value + 1.0 for value in metadata.model0.maxs]
    if (
        any(abs(float(actual) - expected) > 1e-4
            for actual, expected in zip(model["mins"], expected_mins))
        or any(abs(float(actual) - expected) > 1e-4
               for actual, expected in zip(model["maxs"], expected_maxs))
        or model["headnode"] != metadata.model0.headnode
    ):
        raise AtlasAnalysisError("metadata/parser model0 differs from collision oracle model0")
    return _snapped_origin(model["mins"])


def _grid_index(point: Sequence[float], origin: Sequence[int], size: int) -> tuple[int, int, int]:
    return tuple(math.floor((point[axis] - origin[axis]) / size) for axis in range(3))  # type: ignore[return-value]


def _center(index: Sequence[int], origin: Sequence[int], size: int) -> tuple[float, float, float]:
    return tuple(origin[axis] + (index[axis] + 0.5) * size for axis in range(3))  # type: ignore[return-value]


def _box_request(
    identifier: str,
    start: Sequence[float],
    end: Sequence[float],
    mins: Sequence[float],
    maxs: Sequence[float],
    mask: int = MASK_PLAYERSOLID,
) -> dict:
    return {
        "id": identifier, "op": "box_trace", "start": list(start), "end": list(end),
        "mins": list(mins), "maxs": list(maxs), "mask": mask,
    }


def _candidate_floor_requests(
    candidates: Sequence[tuple[tuple[int, int, int], tuple[float, float, float]]]
) -> list[dict]:
    requests = []
    for ordinal, (_, point) in enumerate(candidates):
        requests.append(_box_request(
            f"floor:{ordinal}", (point[0], point[1], point[2] + 18),
            (point[0], point[1], point[2] - 32), STANDING_MINS, STANDING_MAXS,
        ))
    return requests


def _node_probe_requests(points: Sequence[tuple[float, float, float]]) -> list[dict]:
    requests = []
    for ordinal, point in enumerate(points):
        requests.extend([
            _box_request(f"stand:{ordinal}", point, point, STANDING_MINS, STANDING_MAXS),
            _box_request(f"crouch:{ordinal}", point, point, CROUCHED_MINS, CROUCHED_MAXS),
            {"id": f"contents:{ordinal}", "op": "point_contents", "point": list(point)},
        ])
    return requests


def _directed_adjacency(
    edges: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for edge in edges:
        adjacency[tuple(edge["source"])].append(tuple(edge["target"]))
    for values in adjacency.values():
        values.sort(key=lambda item: (item[2], item[1], item[0]))
    return adjacency


def _assign_directed_regions(
    nodes: Mapping[tuple[int, int, int], NavNode], edges: Sequence[Mapping[str, Any]],
) -> None:
    """Assign deterministic strongly-connected region IDs (Kosaraju)."""
    adjacency = _directed_adjacency(edges)
    reverse: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].append(source)
    order: list[tuple[int, int, int]] = []
    visited: set[tuple[int, int, int]] = set()
    ordered = sorted(nodes, key=lambda item: (item[2], item[1], item[0]))
    for seed in ordered:
        if seed in visited:
            continue
        stack: list[tuple[tuple[int, int, int], bool]] = [(seed, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(current)
                continue
            if current in visited:
                continue
            visited.add(current)
            stack.append((current, True))
            for neighbor in reversed(adjacency.get(current, [])):
                if neighbor not in visited:
                    stack.append((neighbor, False))
    for node in nodes.values():
        node.region_id = 0
    region = 0
    for seed in reversed(order):
        if nodes[seed].region_id:
            continue
        region += 1
        nodes[seed].region_id = region
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbor in reverse.get(current, []):
                if nodes[neighbor].region_id == 0:
                    nodes[neighbor].region_id = region
                    pending.append(neighbor)


def _spawn_reachability(
    edges: Sequence[Mapping[str, Any]],
    spawn_indices: Mapping[int, tuple[int, int, int]],
) -> dict[int, list[int]]:
    adjacency = _directed_adjacency(edges)
    output: dict[int, list[int]] = {}
    for ordinal, seed in sorted(spawn_indices.items()):
        visited = {seed}
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        output[ordinal] = [
            other for other, key in sorted(spawn_indices.items())
            if other != ordinal and key in visited
        ]
    return output


def _mutually_reachable_spawn_pairs(
    edges: Sequence[Mapping[str, Any]],
    spawn_indices: Mapping[int, tuple[int, int, int]],
) -> list[tuple[int, int]]:
    reachable = _spawn_reachability(edges, spawn_indices)
    return [
        (left, right)
        for left in sorted(reachable)
        for right in reachable[left]
        if left < right and left in reachable.get(right, [])
    ]


def _build_navigation(
    cm: OracleProcess,
    pmove: OracleProcess | None,
    spawns: Sequence[tuple[int, tuple[float, float, float]]],
    origin: tuple[int, int, int],
    limits: AnalyzerLimits,
) -> tuple[dict[tuple[int, int, int], NavNode], list[dict], dict[int, tuple[int, int, int]]]:
    seed_points = [point for _, point in spawns]
    seed_support = cm.call([
        _box_request(f"seed-floor:{ordinal}", point, (point[0], point[1], point[2] - 64),
                     STANDING_MINS, STANDING_MAXS)
        for ordinal, point in enumerate(seed_points)
    ])
    grounded: list[tuple[float, float, float] | None] = []
    for point, support in zip(seed_points, seed_support):
        if (
            support["startsolid"] or support["allsolid"] or support["fraction"] >= 1
            or support["plane"]["normal"][2] < 0.7
        ):
            grounded.append(None)
        else:
            grounded.append(tuple(support["endpos"]))
    supported_points = [point for point in grounded if point is not None]
    probes = cm.call(_node_probe_requests(supported_points))
    nodes: dict[tuple[int, int, int], NavNode] = {}
    queue: deque[tuple[int, int, int]] = deque()
    spawn_indices: dict[int, tuple[int, int, int]] = {}
    probe_ordinal = 0
    for ordinal, ((entity_ordinal, _), point) in enumerate(zip(spawns, grounded)):
        if point is None:
            continue
        stand, crouch, contents = probes[probe_ordinal * 3:probe_ordinal * 3 + 3]
        probe_ordinal += 1
        key = _grid_index(point, origin, 16)
        if (
            (stand["startsolid"] or stand["allsolid"])
            and (crouch["startsolid"] or crouch["allsolid"])
        ):
            continue
        node = NavNode(
            key, point,
            not stand["startsolid"] and not stand["allsolid"],
            not crouch["startsolid"] and not crouch["allsolid"], True,
            contents["contents"], tuple(seed_support[ordinal]["plane"]["normal"]),
            floor_surface_flags=(seed_support[ordinal].get("surface") or {}).get("flags", 0),
            floor_surface_name=(seed_support[ordinal].get("surface") or {}).get("name", ""),
        )
        nodes.setdefault(key, node)
        queue.append(key)
        spawn_indices[entity_ordinal] = key
    if len(spawn_indices) < 2:
        raise AtlasAnalysisError("fewer than two oracle-clear deathmatch spawns")

    edges: list[dict] = []
    edge_keys: set[tuple] = set()
    explored: set[tuple[int, int, int]] = set()
    while queue:
        frontier = []
        while queue and len(frontier) < limits.flood_source_batch:
            key = queue.popleft()
            if key not in explored:
                explored.add(key)
                frontier.append(key)
        candidates: list[tuple[tuple[int, int, int], tuple[float, float, float]]] = []
        sources: list[tuple[int, int, int]] = []
        for source_key in frontier:
            source = nodes[source_key]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate_key = (source_key[0] + dx, source_key[1] + dy, source_key[2])
                point = _center(candidate_key, origin, 16)
                candidates.append((candidate_key, point))
                sources.append(source_key)
        floors = cm.call(_candidate_floor_requests(candidates))
        accepted: list[tuple[tuple[int, int, int], tuple[float, float, float], tuple[int, int, int], dict]] = []
        for source_key, (_, _), floor in zip(sources, candidates, floors):
            if floor["startsolid"] or floor["fraction"] >= 1 or floor["plane"]["normal"][2] < 0.7:
                continue
            point = tuple(floor["endpos"])
            key = _grid_index(point, origin, 16)
            if key == source_key or abs(key[2] - source_key[2]) > 1:
                continue
            accepted.append((source_key, point, key, floor))
        unknown = [(source, point, key, floor) for source, point, key, floor in accepted if key not in nodes]
        unknown_points = [point for _, point, _, _ in unknown]
        unknown_probes = cm.call(_node_probe_requests(unknown_points)) if unknown else []
        for ordinal, (source_key, point, key, floor) in enumerate(unknown):
            stand, crouch, contents = unknown_probes[ordinal * 3:ordinal * 3 + 3]
            standing = not stand["startsolid"] and not stand["allsolid"]
            crouched = not crouch["startsolid"] and not crouch["allsolid"]
            if not (standing or crouched):
                continue
            if len(nodes) >= limits.max_l1_nodes:
                raise AtlasAnalysisError("L1 flood exceeded node budget")
            nodes[key] = NavNode(
                key, point, standing, crouched, True, contents["contents"],
                tuple(floor["plane"]["normal"]),
                floor_surface_flags=(floor.get("surface") or {}).get("flags", 0),
                floor_surface_name=(floor.get("surface") or {}).get("name", ""),
            )
            queue.append(key)
        traces = []
        trace_meta = []
        for source_key, point, key, _ in accepted:
            target = nodes.get(key)
            if target is None:
                continue
            source = nodes[source_key]
            # A diagonal endpoint hull trace is not Quake's step law. Only
            # same-height origins are admitted as ordinary walk edges here;
            # step/drop/jump candidates are challenged through Pmove below.
            if abs(target.position[2] - source.position[2]) > 0.5:
                continue
            mins = STANDING_MINS if source.standing_clear and target.standing_clear else CROUCHED_MINS
            maxs = STANDING_MAXS if source.standing_clear and target.standing_clear else CROUCHED_MAXS
            traces.append(_box_request(f"edge:{len(traces)}", source.position, target.position, mins, maxs))
            trace_meta.append((source_key, key, "standing" if maxs is STANDING_MAXS else "crouched"))
        results = cm.call(traces)
        for (source_key, target_key, stance), trace in zip(trace_meta, results):
            if trace["fraction"] < 1 or trace["startsolid"]:
                continue
            key = (source_key, target_key, "walk", stance)
            if key in edge_keys:
                continue
            edge_keys.add(key)
            edges.append({
                "source": list(source_key), "target": list(target_key), "edge_type": "walk",
                "stance": stance, "flags": 0, "blocker": 0,
                "cost": 4096, "risk": 0, "confidence": 65535,
                "evidence": EVIDENCE_CM_TRACE_V1, "validation_version": VALIDATION_VERSION,
                "auxiliary": 0xFFFFFFFF,
            })
            if len(edges) > limits.max_l1_edges:
                raise AtlasAnalysisError("L1 flood exceeded edge budget")

    # Exact Pmove validates step/drop/jump traversal only from a bounded,
    # deterministic set of spawn and CM-boundary nodes. Atlas v1 omits all
    # unprobed movement candidates; it never extrapolates a usercmd result.
    if pmove is not None:
        requests = []
        metadata = []
        outgoing = defaultdict(int)
        for edge in edges:
            outgoing[tuple(edge["source"])] += 1
        ordered_nodes = sorted(nodes, key=lambda item: (item[2], item[1], item[0]))
        candidates = list(dict.fromkeys(spawn_indices.values()))
        candidates.extend(
            key for key in ordered_nodes if outgoing[key] < 4 and key not in candidates
        )
        for key in candidates[:limits.max_pmove_sources]:
            source = nodes[key]
            for yaw in (0, 90, 180, 270):
                walk_id = f"pmove-walk:{key[0]}:{key[1]}:{key[2]}:{yaw}"
                requests.append({
                    "id": walk_id, "op": "simulate",
                    "origin": list(source.position), "pm_flags": 4, "snapinitial": False,
                    "commands": [
                        {"msec": 50, "angles": [0, yaw, 0], "forwardmove": 300}
                        for _ in range(4)
                    ],
                })
                metadata.append((key, "ground"))
                if outgoing[key] < 2 or key in spawn_indices.values():
                    jump_id = f"pmove-jump:{key[0]}:{key[1]}:{key[2]}:{yaw}"
                    requests.append({
                        "id": jump_id, "op": "simulate",
                        "origin": list(source.position), "pm_flags": 4, "snapinitial": False,
                        "commands": [
                            {"msec": 50, "angles": [0, yaw, 0], "forwardmove": 300, "upmove": 300},
                            *[
                                {"msec": 50, "angles": [0, yaw, 0], "forwardmove": 300}
                                for _ in range(15)
                            ],
                        ],
                    })
                    metadata.append((key, "jump"))
        for (source_key, mode), result in zip(metadata, pmove.call(requests)):
            target_key = _grid_index(result["final"]["origin"], origin, 16)
            if target_key == source_key or target_key not in nodes or not result["final"]["grounded"]:
                continue
            vertical = nodes[target_key].position[2] - nodes[source_key].position[2]
            if mode == "jump":
                if not any(not frame["grounded"] for frame in result["frames"]):
                    continue
                kind = "jump"
            elif vertical < -18:
                kind = "controlled_drop"
            elif abs(vertical) <= 18.0:
                kind = "step"
            else:
                continue
            key = (source_key, target_key, kind, "standing")
            if key in edge_keys:
                continue
            edge_keys.add(key)
            edges.append({
                "source": list(source_key), "target": list(target_key), "edge_type": kind,
                "stance": "standing", "flags": 0, "blocker": 0,
                "cost": 4096, "risk": 0, "confidence": 65535,
                "evidence": EVIDENCE_PMOVE_V1, "validation_version": VALIDATION_VERSION,
                "auxiliary": 0xFFFFFFFF,
            })

    _assign_directed_regions(nodes, edges)
    mutual = _mutually_reachable_spawn_pairs(edges, spawn_indices)
    if not mutual:
        raise AtlasAnalysisError("fewer than two mutually reachable deathmatch spawns")
    return nodes, edges, spawn_indices


def _admissions(
    bsp_sha256: str,
    provenance_sha256: str,
    cm: OracleProcess,
    pmove: OracleProcess | None,
) -> dict:
    def tool(binary: Path, identity: dict) -> dict:
        return {
            "name": "q2-cm-oracle" if identity["schema"] == "q2-cm-oracle-v1" else "q2-pmove-oracle",
            "schema": identity["schema"], "version": 1,
            "executable_sha256": sha256_file(binary),
            "physics_identity_sha256": identity["physics_identity"],
        }

    provenance = cm.identity["provenance"]
    build_contract = (
        f"tool={cm.identity['tool_identity']};source={provenance['source_closure_sha256']};"
        f"build={provenance['build_identity_sha256']}"
    )
    bsp = {"sha256": bsp_sha256, "provenance_sha256": provenance_sha256}
    collision_source = {**cm.identity["source"], "build_contract": build_contract}
    collision = {
        "tool": tool(cm.binary, cm.identity), "bsp": bsp,
        "parameters": {"mask_playersolid": MASK_PLAYERSOLID, "mask_shot": MASK_SHOT},
        "source": collision_source,
    }
    collision["contract_sha256"] = sha256_bytes(_rust_struct_json(collision))
    output: dict[str, Any] = {"collision_oracle": collision}
    if pmove is not None:
        parameters = pmove.identity["parameters"]
        pmove_source = {**pmove.identity["source"], "build_contract": build_contract}
        movement = {
            "tool": tool(pmove.binary, pmove.identity), "bsp": bsp,
            "parameters": {
                "gravity": parameters["gravity"],
                "airaccelerate_f32_bits": struct.unpack("<I", struct.pack("<f", parameters["airaccelerate"]))[0],
                "constants": parameters["constants"],
            },
            "source": pmove_source,
        }
        movement["contract_sha256"] = sha256_bytes(_rust_struct_json(movement))
        output["pmove_oracle"] = movement
    return output


def _l0_chunks(
    nodes: Mapping[tuple[int, int, int], NavNode],
    spawns: Sequence[tuple[int, tuple[float, float, float]]],
    origin: tuple[int, int, int],
    *,
    cm: OracleProcess | None = None,
    metadata: BspMetadata | None = None,
    evidenced_l0_points: Mapping[str, Sequence[Sequence[float]]] | None = None,
    semantic_summary: dict[str, int] | None = None,
) -> list[dict]:
    chunks: dict[tuple[int, int, int], dict[str, Any]] = {}
    semantic_cells: dict[str, set[tuple[int, int, int]]] = defaultdict(set)

    def item_for(l0: tuple[int, int, int]) -> tuple[dict[str, Any], int]:
        chunk = tuple(value // 16 for value in l0)
        local = tuple(value % 16 for value in l0)
        linear = local[0] + 16 * local[1] + 256 * local[2]
        item = chunks.setdefault(chunk, {"key": list(chunk), "bits": defaultdict(set), "scalars": {}})
        return item, linear

    def set_index(l0: tuple[int, int, int], plane: str) -> None:
        item, linear = item_for(l0)
        item["bits"][plane].add(linear)

    def set_bit(point: Sequence[float], plane: str) -> None:
        set_index(_grid_index(point, origin, 4), plane)

    def set_scalar(point: Sequence[float], plane: str, value: int) -> None:
        if not 0 < value <= 255:
            return
        item, linear = item_for(_grid_index(point, origin, 4))
        values = item["scalars"].setdefault(plane, {})
        values[linear] = max(value, values.get(linear, 0))

    def mark_semantic(point: Sequence[float], name: str) -> None:
        semantic_cells[name].add(_grid_index(point, origin, 4))

    def fill_bounds(
        mins: Sequence[float], maxs: Sequence[float], plane: str,
        *, scalar: tuple[str, int] | None = None,
        semantic: str | None = None,
    ) -> None:
        lower = _grid_index(mins, origin, 4)
        # Bounds are half-open.  Subtract an epsilon so an exact upper grid
        # plane does not materialize a neighboring cell.
        upper = _grid_index(tuple(value - 1e-6 for value in maxs), origin, 4)
        if any(upper[axis] < lower[axis] for axis in range(3)):
            return
        cell_count = math.prod(upper[axis] - lower[axis] + 1 for axis in range(3))
        if cell_count > 2_000_000:
            raise AtlasAnalysisError(f"L0 {plane} field exceeds bounded sparse fill")
        for z in range(lower[2], upper[2] + 1):
            for y in range(lower[1], upper[1] + 1):
                for x in range(lower[0], upper[0] + 1):
                    index = (x, y, z)
                    set_index(index, plane)
                    if semantic is not None:
                        semantic_cells[semantic].add(index)
                    if scalar is not None:
                        point = _center(index, origin, 4)
                        set_scalar(point, scalar[0], scalar[1])

    def expanded_bounds(
        mins: Sequence[float], maxs: Sequence[float],
        hull_mins: Sequence[float], hull_maxs: Sequence[float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            tuple(mins[axis] - hull_maxs[axis] for axis in range(3)),
            tuple(maxs[axis] - hull_mins[axis] for axis in range(3)),
        )

    # Retain the exact narrow-band facts already established during the L1
    # flood.  Full open floors remain L1-only; L0 records boundary surfaces.
    boundary_keys: set[tuple[int, int, int]] = set()
    for key, node in nodes.items():
        boundary = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbors = [
                candidate for dz in (-1, 0, 1)
                if (candidate := nodes.get((key[0] + dx, key[1] + dy, key[2] + dz)))
                is not None
            ]
            if not neighbors or all(
                neighbor.standing_clear != node.standing_clear
                or neighbor.crouched_clear != node.crouched_clear
                or neighbor.contents != node.contents
                for neighbor in neighbors
            ):
                boundary = True
                break
        if boundary:
            boundary_keys.add(key)
            set_bit((node.position[0], node.position[1], node.position[2] - 25), "solid")
        surface_point = (node.position[0], node.position[1], node.position[2] - 25)
        if node.floor_surface_flags & SURF_SKY:
            set_bit(surface_point, "sky")
        if node.floor_surface_flags & SURF_SLICK:
            set_bit(surface_point, "slick")
        if node.floor_surface_flags & SURF_WARP:
            set_bit(surface_point, "warp")
        if node.floor_surface_flags & SURF_NODRAW:
            set_bit(surface_point, "nodraw")
        if node.contents & CONTENTS_WATER:
            set_bit(node.position, "water")
        if node.contents & CONTENTS_SLIME:
            set_bit(node.position, "slime")
        if node.contents & CONTENTS_LAVA:
            set_bit(node.position, "lava")
        if node.contents & CONTENTS_WINDOW:
            set_bit(node.position, "window")
        if node.contents & CONTENTS_PLAYERCLIP:
            set_bit(node.position, "playerclip")
        if node.contents & CONTENTS_LADDER:
            set_bit(node.position, "ladder")

    if cm is not None:
        # Challenge every 4-unit player-origin sample inside each reachable
        # L1 column.  This records hull-expanded forbidden origins and raw
        # contents without ever allocating dense world-AABB free space.
        retained_keys = set(boundary_keys)
        retained_keys.update(
            key for key, node in nodes.items()
            if node.contents & (
                CONTENTS_WINDOW | CONTENTS_PLAYERCLIP | CONTENTS_MONSTERCLIP
                | CONTENTS_FLUID_MASK | CONTENTS_MIST | CONTENTS_LADDER
                | CONTENTS_CURRENT_MASK
            )
        )
        retained_keys.update(
            _grid_index(point, origin, 16) for _, point in spawns
            if _grid_index(point, origin, 16) in nodes
        )
        ordered = sorted(retained_keys, key=lambda item: (item[2], item[1], item[0]))
        for start in range(0, len(ordered), 32):
            requests: list[dict] = []
            samples: list[tuple[float, float, float]] = []
            for key in ordered[start:start + 32]:
                node = nodes[key]
                base_x = origin[0] + key[0] * 16
                base_y = origin[1] + key[1] * 16
                for dy in (2, 6, 10, 14):
                    for dx in (2, 6, 10, 14):
                        point = (base_x + dx, base_y + dy, node.position[2])
                        ordinal = len(samples)
                        samples.append(point)
                        requests.extend([
                            _box_request(f"l0-stand:{ordinal}", point, point,
                                         STANDING_MINS, STANDING_MAXS),
                            _box_request(f"l0-crouch:{ordinal}", point, point,
                                         CROUCHED_MINS, CROUCHED_MAXS),
                            _box_request(f"l0-stand-hazard:{ordinal}", point, point,
                                         STANDING_MINS, STANDING_MAXS,
                                         CONTENTS_LAVA | CONTENTS_SLIME),
                            _box_request(f"l0-crouch-hazard:{ordinal}", point, point,
                                         CROUCHED_MINS, CROUCHED_MAXS,
                                         CONTENTS_LAVA | CONTENTS_SLIME),
                            {"id": f"l0-contents:{ordinal}", "op": "point_contents",
                             "point": list(point)},
                        ])
            results = cm.call(requests)
            for ordinal, point in enumerate(samples):
                stand, crouch, stand_hazard, crouch_hazard, contents_result = (
                    results[ordinal * 5:ordinal * 5 + 5]
                )
                contents = contents_result["contents"]
                stand_blocked = stand["startsolid"] or stand["allsolid"]
                crouch_blocked = crouch["startsolid"] or crouch["allsolid"]
                stand_hazardous = stand_hazard["startsolid"] or stand_hazard["allsolid"]
                crouch_hazardous = crouch_hazard["startsolid"] or crouch_hazard["allsolid"]
                if stand_blocked or stand_hazardous:
                    set_bit(point, "standing_forbidden_origin")
                if crouch_blocked or crouch_hazardous:
                    set_bit(point, "crouched_forbidden_origin")
                for mask, plane in (
                    (CONTENTS_SOLID, "solid"), (CONTENTS_WINDOW, "window"),
                    (CONTENTS_PLAYERCLIP, "playerclip"),
                    (CONTENTS_MONSTERCLIP, "monsterclip"),
                    (CONTENTS_WATER, "water"), (CONTENTS_SLIME, "slime"),
                    (CONTENTS_LAVA, "lava"), (CONTENTS_MIST, "mist"),
                    (CONTENTS_LADDER, "ladder"),
                ):
                    if contents & mask:
                        set_bit(point, plane)
                if stand_hazardous:
                    # Expanded hazard semantics remain distinct from raw
                    # point contents through forbidden-origin and severity
                    # planes. A stationary trace identifies the brush type.
                    hazard_contents = stand_hazard.get("contents", 0)
                    if hazard_contents & CONTENTS_LAVA:
                        mark_semantic(point, "lava_expanded")
                        set_scalar(point, "hazard_severity", 255)
                    if hazard_contents & CONTENTS_SLIME:
                        mark_semantic(point, "slime_expanded")
                        set_scalar(point, "hazard_severity", 160)
                current = (contents & CONTENTS_CURRENT_MASK) >> 18
                if current:
                    set_scalar(point, "current_direction", current)

        # Classify uncontained or physically lethal fall approaches from the
        # exact reachable boundary.  Short ordinary drops are not hazards.
        void_candidates: dict[tuple[int, int, int], tuple[float, float, float]] = {}
        for key, node in nodes.items():
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if any((key[0] + dx, key[1] + dy, key[2] + dz) in nodes
                       for dz in (-1, 0, 1)):
                    continue
                point = (node.position[0] + dx * 16, node.position[1] + dy * 16,
                         node.position[2])
                void_candidates.setdefault(_grid_index(point, origin, 4), point)
        candidates = list(void_candidates.values())
        for start in range(0, len(candidates), 512):
            batch = candidates[start:start + 512]
            traces = cm.call([
                _box_request(
                    f"l0-void:{ordinal}", point,
                    (point[0], point[1], point[2] - 2048),
                    STANDING_MINS, STANDING_MAXS,
                )
                for ordinal, point in enumerate(batch)
            ])
            for point, trace in zip(batch, traces):
                drop = point[2] - trace["endpos"][2]
                if trace["startsolid"] or trace["allsolid"]:
                    continue
                if trace["fraction"] >= 1 or drop >= 1400:
                    set_bit(point, "unknown")
                    set_bit(point, "standing_forbidden_origin")
                    set_bit(point, "crouched_forbidden_origin")
                    mark_semantic(point, "lethal_void")
                    set_scalar(point, "hazard_severity", 255)

    if metadata is not None:
        entities = {entity.index: entity for entity in metadata.entities}
        path_corners = {
            entity.value("targetname"): entity
            for entity in metadata.entities
            if entity.classname == "path_corner" and entity.value("targetname")
        }
        for record in metadata.entity_catalog.brush_submodels:
            entity = entities[int(record["entity_index"])]
            model = metadata.models[int(record["model_index"])]
            mins, maxs = model.mins, model.maxs
            classname = entity.classname
            if classname == "trigger_hurt":
                fill_bounds(mins, maxs, "hurt", scalar=("hazard_severity", 255))
                standing = expanded_bounds(mins, maxs, STANDING_MINS, STANDING_MAXS)
                crouched = expanded_bounds(mins, maxs, CROUCHED_MINS, CROUCHED_MAXS)
                fill_bounds(
                    *standing, "standing_forbidden_origin",
                    scalar=("hazard_severity", 255), semantic="hurt_expanded",
                )
                fill_bounds(*standing, "standing_forbidden_origin")
                fill_bounds(*crouched, "crouched_forbidden_origin")
            elif classname in {"trigger_push", "trigger_gravity"}:
                fill_bounds(mins, maxs, "push_or_gravity")
            elif "teleport" in classname:
                fill_bounds(mins, maxs, "teleport_trigger")
            elif classname == "func_areaportal":
                fill_bounds(mins, maxs, "areaportal")

            if not classname.startswith("func_") or classname == "func_areaportal":
                continue
            fill_bounds(mins, maxs, "mover_reference_solid")
            swept_mins = list(mins)
            swept_maxs = list(maxs)
            size = [maxs[axis] - mins[axis] for axis in range(3)]
            displacement = [0.0, 0.0, 0.0]
            if classname == "func_train":
                positions: list[tuple[float, float, float]] = []
                seen: set[str] = set()
                target = entity.value("target")
                closed = False
                while target and target not in seen and len(seen) <= len(path_corners):
                    seen.add(target)
                    corner = path_corners.get(target)
                    if corner is None:
                        break
                    positions.append(_origin(corner))
                    target = corner.value("target")
                if target and positions and target == entity.value("target"):
                    positions.append(positions[0])
                    closed = True
                if not positions:
                    set_bit(model.origin, "unknown")
                    mark_semantic(model.origin, "mover_train_unknown")
                    continue
                if len(positions) == 1:
                    positions.append(positions[0])
                for source, target_position in zip(positions, positions[1:]):
                    segment_mins = [min(source[axis], target_position[axis]) for axis in range(3)]
                    segment_maxs = [
                        max(source[axis], target_position[axis]) + size[axis]
                        for axis in range(3)
                    ]
                    fill_bounds(
                        segment_mins, segment_maxs, "mover_swept_envelope",
                        scalar=("hazard_severity", 192), semantic="crush",
                    )
                if not closed:
                    mark_semantic(positions[-1], "mover_train_open_path")
                continue
            if classname in {"func_door", "func_button", "func_water"}:
                angle = float(entity.value("angle", "0") or 0)
                if angle == -1:
                    direction = (0.0, 0.0, 1.0)
                elif angle == -2:
                    direction = (0.0, 0.0, -1.0)
                else:
                    radians = math.radians(angle)
                    direction = (math.cos(radians), math.sin(radians), 0.0)
                default_lip = 4.0 if classname == "func_button" else 8.0
                lip = float(entity.value("lip", str(default_lip)) or default_lip)
                distance = max(0.0, abs(sum(direction[axis] * size[axis] for axis in range(3))) - lip)
                displacement = [direction[axis] * distance for axis in range(3)]
            elif classname == "func_plat":
                lip = float(entity.value("lip", "8") or 8)
                height = float(entity.value("height", "0") or 0)
                displacement[2] = -(height if height else max(0.0, size[2] - lip))
            elif classname in {"func_rotating", "func_door_rotating"}:
                center = model.origin
                radius = max(
                    math.sqrt(sum((corner[axis] - center[axis]) ** 2 for axis in range(3)))
                    for corner in (
                        (mins[0], mins[1], mins[2]), (mins[0], mins[1], maxs[2]),
                        (mins[0], maxs[1], mins[2]), (mins[0], maxs[1], maxs[2]),
                        (maxs[0], mins[1], mins[2]), (maxs[0], mins[1], maxs[2]),
                        (maxs[0], maxs[1], mins[2]), (maxs[0], maxs[1], maxs[2]),
                    )
                )
                swept_mins = [center[axis] - radius for axis in range(3)]
                swept_maxs = [center[axis] + radius for axis in range(3)]
            for axis in range(3):
                swept_mins[axis] = min(swept_mins[axis], mins[axis] + displacement[axis])
                swept_maxs[axis] = max(swept_maxs[axis], maxs[axis] + displacement[axis])
            if classname not in {"func_wall", "func_object", "func_explosive"}:
                fill_bounds(
                    swept_mins, swept_maxs, "mover_swept_envelope",
                    scalar=("hazard_severity", 192), semantic="crush",
                )
            else:
                fill_bounds(swept_mins, swept_maxs, "mover_swept_envelope")

    # Generated-map glue may inject only already-admitted hook evidence here;
    # the stock analyzer never promotes a CM surface guess to hook authority.
    if evidenced_l0_points:
        if set(evidenced_l0_points) - {"hookable_surface", "hook_corridor"}:
            raise AtlasAnalysisError("unsupported externally evidenced L0 plane")
        for plane, points in evidenced_l0_points.items():
            for point in points:
                if len(point) != 3 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) for value in point
                ):
                    raise AtlasAnalysisError(f"invalid evidenced {plane} L0 point")
                set_bit(point, plane)

    for _, spawn in spawns:
        # A 96-unit column is the half-open interval [0, 96); sampling the
        # endpoint would retain a 100-unit column and can create an unrelated
        # upper chunk at an exact 64-unit boundary.
        for z in range(0, 96, 4):
            set_bit((spawn[0], spawn[1], spawn[2] + z), "spawn_column")
    output = []
    for key in sorted(chunks, key=lambda item: (item[2], item[1], item[0])):
        item = chunks[key]
        unknown_bits = set(item["bits"]) - FROZEN_L0_BIT_PLANE_NAMES
        unknown_scalars = set(item["scalars"]) - FROZEN_L0_SCALAR_PLANE_NAMES
        if unknown_bits or unknown_scalars:
            raise AtlasAnalysisError(
                f"L0 planes differ from frozen Rust schema: "
                f"bits={sorted(unknown_bits)}, scalars={sorted(unknown_scalars)}"
            )
        item["bits"] = {name: sorted(values) for name, values in sorted(item["bits"].items())}
        item["scalars"] = {
            name: [[linear, value] for linear, value in sorted(values.items())]
            for name, values in sorted(item["scalars"].items())
        }
        output.append(item)
    if semantic_summary is not None:
        semantic_summary.update({
            name: len(cells) for name, cells in sorted(semantic_cells.items())
        })
    return output


def _write_binary_json(path: Path, magic: bytes, value: Any) -> dict:
    body = canonical_json(value)
    raw = magic + struct.pack("<Q", len(body)) + body
    compressed = zstandard.ZstdCompressor(level=3, threads=0, write_checksum=False).compress(raw)
    path.write_bytes(compressed)
    return {
        "sha256": sha256_bytes(raw), "uncompressed_bytes": len(raw),
        "transport_sha256": sha256_bytes(compressed), "compressed_bytes": len(compressed),
    }


def _process_tree_rss_bytes(root_pid: int) -> int:
    """Sample Linux resident bytes for one process and its live descendants."""
    pending = [root_pid]
    visited: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(
                encoding="ascii"
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                words = line.split()
                if len(words) >= 2:
                    total += int(words[1]) * 1024
                break
        pending.extend(int(value) for value in children.split())
    return total


def _run_measured_process(
    command: Sequence[str], *, timeout: float,
) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run a process while sampling whole-process-tree RSS at 100 Hz."""
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + timeout
    peak_rss = 0
    while True:
        peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise AtlasAnalysisError(
                f"independent cold analyzer timed out: {stderr.strip()}"
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.01, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), peak_rss


def _full_cold_rebuild(
    bsp: Path,
    primary_dir: Path,
    canonical_map_id: str,
    provenance: Mapping[str, Any],
    *,
    cm_oracle: Path,
    pmove_oracle: Path | None,
    hook_oracle: Path | None,
    hook_attestation: Path | None,
    packer: Path,
    limits: AnalyzerLimits,
    generator_claims_sha256: str | None,
    generator_claims: Mapping[str, Any] | None,
    evidenced_l0_points: Mapping[str, Sequence[Sequence[float]]] | None,
) -> dict:
    """Re-run the complete analyzer in a fresh process and compare artifacts."""
    worker = Path(__file__).resolve().parents[1] / "tools/atlas_cold_worker.py"
    if not worker.is_file():
        raise AtlasAnalysisError(f"independent cold analyzer worker missing: {worker}")
    artifact_suffixes = (
        ".atlas.bin", ".atlas.bin.zst", ".navigation.bin.zst",
        ".visibility.bin.zst", ".design-signature.json", ".routes.json",
    )
    with tempfile.TemporaryDirectory(prefix="q2-atlas-full-cold-") as temporary:
        root = Path(temporary)
        cold_dir = root / "output"
        specification = {
            "schema": "q2-atlas-cold-worker-v1",
            "bsp": str(bsp.resolve()),
            "output_dir": str(cold_dir),
            "canonical_map_id": canonical_map_id,
            "provenance": dict(provenance),
            "cm_oracle": str(cm_oracle.resolve()),
            "pmove_oracle": None if pmove_oracle is None else str(pmove_oracle.resolve()),
            "hook_oracle": None if hook_oracle is None else str(hook_oracle.resolve()),
            "hook_attestation": (
                None if hook_attestation is None else str(hook_attestation.resolve())
            ),
            "packer": str(packer.resolve()),
            "limits": asdict(limits),
            "generator_claims_sha256": generator_claims_sha256,
            "generator_claims": None if generator_claims is None else dict(generator_claims),
            "evidenced_l0_points": (
                None if evidenced_l0_points is None
                else {name: [list(point) for point in points]
                      for name, points in evidenced_l0_points.items()}
            ),
        }
        specification_path = root / "worker.json"
        specification_path.write_bytes(canonical_json(specification) + b"\n")
        completed, peak_rss = _run_measured_process(
            [sys.executable, str(worker), str(specification_path)],
            timeout=limits.full_cold_timeout_seconds,
        )
        if completed.returncode:
            raise AtlasAnalysisError(
                "independent cold analyzer failed: " + completed.stderr.strip()
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise AtlasAnalysisError("independent cold analyzer emitted invalid JSON") from error
        if result != {
            "atlas_sha256": sha256_file(cold_dir / f"{canonical_map_id}.atlas.bin"),
            "canonical_map_id": canonical_map_id,
            "schema": "q2-atlas-cold-worker-result-v1",
        }:
            raise AtlasAnalysisError("independent cold analyzer result contract mismatch")
        digests = {}
        for suffix in artifact_suffixes:
            primary = primary_dir / f"{canonical_map_id}{suffix}"
            cold = cold_dir / f"{canonical_map_id}{suffix}"
            if not primary.is_file() or not cold.is_file():
                raise AtlasAnalysisError(f"independent cold artifact missing: {suffix}")
            primary_digest = sha256_file(primary)
            if primary_digest != sha256_file(cold):
                raise AtlasAnalysisError(f"independent cold artifact mismatch: {suffix}")
            digests[suffix] = primary_digest
    if peak_rss > 512 * 1024 * 1024:
        raise AtlasAnalysisError("independent cold analyzer exceeded 512 MiB peak RSS")
    return {
        "schema": "q2-atlas-full-cold-proof-v1",
        "independent_process_launches": 1,
        "artifact_count": len(artifact_suffixes),
        "artifact_sha256": digests,
        "sample_interval_milliseconds": 10,
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "peak_rss_limit_bytes": 512 * 1024 * 1024,
    }


def _column_clearance(cm: OracleProcess, spawn: Sequence[float]) -> int:
    requests = [
        _box_request(f"column:{height}", spawn, (spawn[0], spawn[1], spawn[2] + height),
                     STANDING_MINS, STANDING_MAXS)
        for height in range(4, 129, 4)
    ]
    clearance = 56
    for height, result in zip(range(4, 129, 4), cm.call(requests)):
        if result["fraction"] < 1 or result["startsolid"]:
            break
        clearance = 56 + height
    return clearance * 1000


def _visibility(
    cm: OracleProcess,
    nodes: Mapping[tuple[int, int, int], NavNode],
    limits: AnalyzerLimits,
) -> dict:
    keys = sorted(nodes, key=lambda item: (item[2], item[1], item[0]))
    pairs = []
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:left_index + 9]:
            if len(pairs) >= limits.max_visibility_pairs:
                break
            pairs.append((left, right))
    pvs_requests = []
    shot_requests = []
    for ordinal, (left, right) in enumerate(pairs):
        a = nodes[left].position
        b = nodes[right].position
        eye_a = (a[0], a[1], a[2] + 22)
        eye_b = (b[0], b[1], b[2] + 22)
        pvs_requests.append({"id": f"pvs:{ordinal}", "op": "pvs", "from": list(eye_a), "to": list(eye_b)})
        shot_requests.append(_box_request(f"shot:{ordinal}", eye_a, eye_b, [0, 0, 0], [0, 0, 0], MASK_SHOT))
    pvs = cm.call(pvs_requests)
    shots = cm.call(shot_requests)
    records = []
    for (left, right), coarse, shot in zip(pairs, pvs, shots):
        records.append({
            "source": list(left), "target": list(right),
            "pvs": coarse["potentially_visible"],
            "mask_shot_clear": shot["fraction"] == 1 and not shot["startsolid"],
            "evidence": EVIDENCE_CM_TRACE_V1, "validation_version": VALIDATION_VERSION,
        })
    return {"schema": "q2-atlas-visibility-v1", "pairs": records}


def _design_signature(metadata: BspMetadata, nodes: Mapping, edges: Sequence[dict]) -> dict:
    degree = defaultdict(int)
    edge_types = defaultdict(int)
    for edge in edges:
        degree[tuple(edge["source"])] += 1
        edge_types[edge["edge_type"]] += 1
    degree_histogram = defaultdict(int)
    for value in degree.values():
        degree_histogram[min(value, 15)] += 1
    return {
        "schema": "q2-atlas-design-signature-v1",
        "coordinate_free": True,
        "bsp_sha256": metadata.sha256,
        "counts": {
            "l1_nodes": len(nodes), "l1_edges": len(edges),
            "deathmatch_spawns": metadata.entity_catalog.deathmatch_spawn_count,
            "items": sum(metadata.entity_catalog.item_classes.values()),
            "faces": metadata.faces.count, "visibility_clusters": metadata.visibility.cluster_count,
        },
        "degree_histogram": {str(key): value for key, value in sorted(degree_histogram.items())},
        "edge_type_histogram": dict(sorted(edge_types.items())),
        "item_class_multiset": dict(sorted(metadata.entity_catalog.item_classes.items())),
        "light": {
            "lightdata_bytes": metadata.lightmaps.byte_count,
            "lightmapped_faces": metadata.faces.lightmapped_count,
        },
    }


def validate_design_signature(value: dict) -> None:
    forbidden = {"origin", "position", "coordinates", "source_l1", "target_l1", "graph", "adjacency"}
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if key.lower() in forbidden:
                    raise AtlasAnalysisError(f"design signature contains forbidden field {key}")
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(current)


def analyze_map(
    bsp: Path,
    output_dir: Path,
    canonical_map_id: str,
    provenance: Mapping[str, Any],
    *,
    cm_oracle: Path,
    pmove_oracle: Path | None,
    hook_oracle: Path | None,
    hook_attestation: Path | None,
    packer: Path,
    limits: AnalyzerLimits = AnalyzerLimits(),
    generator_claims_sha256: str | None = None,
    generator_claims: Mapping[str, Any] | None = None,
    evidenced_l0_points: Mapping[str, Sequence[Sequence[float]]] | None = None,
    independent_cold: bool = True,
) -> dict:
    if generator_claims is not None:
        from .generated_claim_probes import (
            generator_claims_sha256 as claims_sha256,
            validate_generator_claims,
        )

        claims = validate_generator_claims(dict(generator_claims))
        if claims["map"] != canonical_map_id:
            raise AtlasAnalysisError("generator claims map differs from requested map")
        actual_claims_sha256 = claims_sha256(claims)
        if generator_claims_sha256 != actual_claims_sha256:
            raise AtlasAnalysisError("generator claims content/digest mismatch")
    elif generator_claims_sha256 is not None and independent_cold:
        raise AtlasAnalysisError(
            "full cold verification requires canonical generator claims content"
        )
    if evidenced_l0_points and generator_claims is None:
        raise AtlasAnalysisError("external L0 evidence requires bound generator claims")
    metadata = parse_ibsp38(bsp)
    if metadata.sha256 != provenance.get("bsp_sha256"):
        raise AtlasAnalysisError("BSP/provenance digest mismatch")
    provenance_sha256 = sha256_bytes(canonical_json(dict(provenance)))
    spawns = [
        (entity.index, _origin(entity))
        for entity in metadata.entities if entity.classname == "info_player_deathmatch"
    ]
    if len(spawns) < 2:
        raise AtlasAnalysisError("map has fewer than two deathmatch spawns")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / canonical_map_id
    # Quake II's spawn lifecycle raises the selected entity origin by nine
    # units before linking the standing player hull. Analyze that real engine
    # origin while retaining the authored entity origin in the report.
    player_spawns = [
        (entity_ordinal, (point[0], point[1], point[2] + 9.0))
        for entity_ordinal, point in spawns
    ]
    with OracleProcess(cm_oracle, bsp, "cm", limits) as cm:
        if cm.identity["map_sha256"] != metadata.sha256:
            raise AtlasAnalysisError("collision oracle loaded different BSP bytes")
        origin = _oracle_atlas_origin(metadata, cm.identity)
        pmove_context = OracleProcess(pmove_oracle, bsp, "pmove", limits) if pmove_oracle else None
        try:
            if pmove_context and pmove_context.identity["map_sha256"] != metadata.sha256:
                raise AtlasAnalysisError("Pmove oracle loaded different BSP bytes")
            nodes, edges, spawn_indices = _build_navigation(
                cm, pmove_context, player_spawns, origin, limits,
            )
            spawn_reachability = _spawn_reachability(edges, spawn_indices)
            visibility = _visibility(cm, nodes, limits)
            spawn_records = []
            for entity_ordinal, point in spawns:
                key = spawn_indices.get(entity_ordinal)
                node = nodes.get(key) if key else None
                player_point = (point[0], point[1], point[2] + 9.0)
                column = _column_clearance(cm, player_point)
                reachable = []
                if node is not None:
                    reachable = spawn_reachability.get(entity_ordinal, [])
                spawn_records.append({
                    "entity_ordinal": entity_ordinal,
                    "origin_milliunits": [round(value * 1000) for value in point],
                    "standing_clear": bool(node and node.standing_clear),
                    "crouched_clear": bool(node and node.crouched_clear),
                    "supported": bool(node and node.supported),
                    "column_clearance_milliunits": column,
                    "column_clear_96": column >= 96_000,
                    "l1_index": list(key) if key else [0, 0, 0],
                    "region_id": node.region_id if node else 0,
                    "escape_edge_count": sum(1 for edge in edges if tuple(edge["source"]) == key),
                    "reachable_spawn_ordinals": reachable,
                    "cost_to_safety_q8": 0 if node else 0xFFFFFFFF,
                })
            l0_semantics: dict[str, int] = {}
            l0_chunks = _l0_chunks(
                nodes, player_spawns, origin, cm=cm, metadata=metadata,
                evidenced_l0_points=evidenced_l0_points,
                semantic_summary=l0_semantics,
            )
            l0_plane_counts: dict[str, int] = defaultdict(int)
            for chunk in l0_chunks:
                for plane, cells in chunk["bits"].items():
                    l0_plane_counts[plane] += len(cells)
            l0_plane_counts = dict(sorted(l0_plane_counts.items()))
            admissions = _admissions(metadata.sha256, provenance_sha256, cm, pmove_context)
            cm_identity = dict(cm.identity)
            cm_identity.pop("map", None)
            pmove_identity = None if pmove_context is None else dict(pmove_context.identity)
            oracle_manifest = {
                "collision": {
                    "binary_sha256": sha256_file(cm.binary),
                    "identity": cm_identity,
                    "requests": cm.requests,
                },
                "pmove": None if pmove_context is None else {
                    "binary_sha256": sha256_file(pmove_context.binary),
                    "identity": pmove_identity,
                    "requests": pmove_context.requests,
                },
            }
        finally:
            if pmove_context is not None:
                pmove_context.close()

    hook = _admit_hook(hook_oracle, hook_attestation)
    plan_nodes = []
    incident = {tuple(edge["source"]) for edge in edges} | {tuple(edge["target"]) for edge in edges}
    for key in sorted(nodes, key=lambda item: (item[2], item[1], item[0])):
        plan_nodes.append(nodes[key].plan(key in incident))
    plan = {
        "schema": BUILD_PLAN_SCHEMA, "origin": list(origin),
        "bsp": {
            "canonical_map_id": canonical_map_id, "sha256": metadata.sha256,
            "provenance_sha256": provenance_sha256, "size_bytes": metadata.byte_count,
            "ibsp_version": metadata.version,
        },
        "oracles": admissions,
        "chunks": l0_chunks,
        "nodes": plan_nodes,
        "edges": edges,
    }
    plan_path = base.with_suffix(".atlas-plan.json")
    plan_path.write_bytes(canonical_json(plan) + b"\n")
    atlas_raw = base.with_suffix(".atlas.bin")
    atlas_zst = base.with_suffix(".atlas.bin.zst")
    packed = subprocess.run(
        [str(packer), str(plan_path), str(atlas_raw), str(atlas_zst)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=120, check=False,
    )
    if packed.returncode:
        raise AtlasAnalysisError(f"Atlas packer failed: {packed.stderr.strip()}")
    pack_result = json.loads(packed.stdout)
    with tempfile.TemporaryDirectory(prefix="q2-atlas-cold-") as cold:
        cold_raw = Path(cold) / "atlas.bin"
        cold_zst = Path(cold) / "atlas.bin.zst"
        rebuilt = subprocess.run(
            [str(packer), str(plan_path), str(cold_raw), str(cold_zst)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=120, check=False,
        )
        if rebuilt.returncode or cold_raw.read_bytes() != atlas_raw.read_bytes():
            raise AtlasAnalysisError("cold Atlas rebuild digest mismatch")
        if cold_zst.read_bytes() != atlas_zst.read_bytes():
            raise AtlasAnalysisError("cold Atlas transport rebuild mismatch")
    plan_path.unlink()
    navigation = {
        "schema": "q2-atlas-navigation-v1", "origin": list(origin),
        "nodes": plan_nodes, "edges": edges,
    }
    navigation_artifact = _write_binary_json(
        base.with_suffix(".navigation.bin.zst"), b"Q2NAV001", navigation,
    )
    visibility_artifact = _write_binary_json(
        base.with_suffix(".visibility.bin.zst"), b"Q2VIS001", visibility,
    )
    design = _design_signature(metadata, nodes, edges)
    validate_design_signature(design)
    design_path = base.with_suffix(".design-signature.json")
    design_bytes = canonical_json(design) + b"\n"
    design_path.write_bytes(design_bytes)
    routes = {
        "schema": "q2-atlas-routes-v1", "bsp_sha256": metadata.sha256,
        "route_claims": [], "objective_routes": [],
    }
    routes_path = base.with_suffix(".routes.json")
    routes_bytes = canonical_json(routes) + b"\n"
    routes_path.write_bytes(routes_bytes)
    artifacts = {
        "atlas": {
            **pack_result,
            "transport_sha256": sha256_file(atlas_zst),
        },
        "navigation": navigation_artifact,
        "visibility": visibility_artifact,
        "design_signature": {
            "sha256": sha256_bytes(design_bytes), "uncompressed_bytes": len(design_bytes),
        },
        "routes": {"sha256": sha256_bytes(routes_bytes), "uncompressed_bytes": len(routes_bytes)},
    }
    analyzer_inputs = [
        Path(__file__),
        Path(__file__).resolve().parents[1] / "tools/atlas_cold_worker.py",
        Path(__file__).resolve().parents[1] / "crates/q2-lattice/src/bin/q2_atlas_pack.rs",
        Path(__file__).resolve().parents[1] / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md",
    ]
    analyzer_sha256 = sha256_bytes(canonical_json([
        {"path": path.name, "sha256": sha256_file(path)} for path in analyzer_inputs
    ]))
    manifest = {
        "schema": SCHEMA,
        "status": "candidate",
        "deterministic_rebuild": False,
        "confidence": "pending-independent-cold-rebuild",
        "analyzer_version": "b2-a-v1",
        "canonical_map_id": canonical_map_id,
        "bsp": {
            "sha256": metadata.sha256, "bytes": metadata.byte_count, "ibsp_version": metadata.version,
            "provenance_sha256": provenance_sha256,
        },
        "generator_claims_sha256": generator_claims_sha256,
        "grid": {
            "origin": list(origin), "cell_sizes": [4, 16, 64, 256],
            "l0_chunk_dimensions": [16, 16, 16],
            "model0_mins_milliunits": [round(value * 1000) for value in metadata.model0.mins],
            "model0_maxs_milliunits": [round(value * 1000) for value in metadata.model0.maxs],
        },
        "player_hulls": [
            {"name": "standing", "mins": STANDING_MINS, "maxs": STANDING_MAXS},
            {"name": "crouched", "mins": CROUCHED_MINS, "maxs": CROUCHED_MAXS},
        ],
        "identity": {
            "bsp_sha256": metadata.sha256,
            "generator_claims_sha256": generator_claims_sha256,
            "atlas_sha256": pack_result["uncompressed_sha256"],
            "analyzer_sha256": analyzer_sha256,
        },
        "oracles": {
            "collision": {
                "status": "oracle", "admitted": True,
                "executable_sha256": oracle_manifest["collision"]["binary_sha256"],
                "physics_identity": cm_identity["physics_identity"],
                "tool_identity": cm_identity["tool_identity"],
                "requests": oracle_manifest["collision"]["requests"],
            },
            "pmove": None if oracle_manifest["pmove"] is None else {
                "status": "oracle", "admitted": True,
                "executable_sha256": oracle_manifest["pmove"]["binary_sha256"],
                "physics_identity": pmove_identity["physics_identity"],
                "tool_identity": pmove_identity["tool_identity"],
                "requests": oracle_manifest["pmove"]["requests"],
            },
            "hook": hook,
        },
        "compiled_world": {
            "spawns": spawn_records,
            "hazards": {
                "l0_raw_cells": sum(
                    l0_plane_counts.get(name, 0) for name in ("lava", "slime", "hurt")
                ),
                "l0_expanded_cells": sum(
                    l0_semantics.get(name, 0)
                    for name in ("lava_expanded", "slime_expanded", "hurt_expanded")
                ),
                "types": [
                    name for name, present in (
                        ("lava", l0_plane_counts.get("lava", 0)),
                        ("slime", l0_plane_counts.get("slime", 0)),
                        ("hurt", l0_plane_counts.get("hurt", 0)),
                        ("void", l0_semantics.get("lethal_void", 0)),
                        ("crush", l0_semantics.get("crush", 0)),
                    ) if present
                ],
                "lethal_drop_edges": l0_semantics.get("lethal_void", 0),
                "guarded_drop_edges": 0,
                "uncontained_drop_edges": l0_semantics.get("lethal_void", 0),
                "semantic_cells": dict(sorted(l0_semantics.items())),
            },
            "hazard_claims": [],
            "lighting": {
                "lightdata_bytes": metadata.lightmaps.byte_count,
                "lightdata_sha256": metadata.lightmaps.sha256,
                "lightmapped_faces": metadata.faces.lightmapped_count,
                "spawn_region_count": len({record["region_id"] for record in spawn_records if record["region_id"]}),
                "dark_spawn_regions": [],
            },
            "hooks": {
                "authority_admitted": hook["authority_admitted"],
                "omission_reason": hook["omission_reason"], "edges": [],
                "hookable_surface_l0_cells": l0_plane_counts.get("hookable_surface", 0),
                "hook_corridor_l0_cells": l0_plane_counts.get("hook_corridor", 0),
            },
            "route_claims": [],
        },
        "artifacts": artifacts,
        "counts": {
            "l0_chunks": pack_result["l0_chunks"], "l1_nodes": len(nodes),
            "l1_edges": len(edges), "l2_cells": pack_result["l2_cells"],
            "l3_cells": pack_result["l3_cells"],
            "l0_bit_cells": sum(l0_plane_counts.values()),
            "l0_scalar_cells": sum(
                len(cells) for chunk in l0_chunks for cells in chunk["scalars"].values()
            ),
        },
        "l0_plane_counts": l0_plane_counts,
        "confidence_summary": {
            "collision": "exact-engine", "movement": "exact-engine" if pmove_oracle else "omitted",
            "hook": "attested-but-no-discovered-edge" if hook["authority_admitted"] else "omitted",
            "metadata": "b1-c-validated",
        },
        "limitations": sorted([
            "areaportal summaries use declared map-static state only",
            "hook authority is admitted but stock surface discovery emits no hook edge in B2-A v1",
            "mover envelopes are L0 occupancy; no mover traversal edge is emitted without state evidence",
            "L0 is a reachable surface/hazard/spawn narrow band, never dense free-space fill",
        ]),
        "performance": {
            "cm_requests": oracle_manifest["collision"]["requests"],
            "pmove_requests": 0 if oracle_manifest["pmove"] is None else oracle_manifest["pmove"]["requests"],
        },
    }
    if independent_cold:
        cold_proof = _full_cold_rebuild(
            bsp, output_dir, canonical_map_id, provenance,
            cm_oracle=cm_oracle, pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle, hook_attestation=hook_attestation,
            packer=packer, limits=limits,
            generator_claims_sha256=generator_claims_sha256,
            generator_claims=generator_claims,
            evidenced_l0_points=evidenced_l0_points,
        )
        manifest["status"] = "passed"
        manifest["deterministic_rebuild"] = True
        manifest["confidence"] = "high"
        manifest["performance"]["full_cold_rebuild"] = cold_proof
    manifest_path = base.with_suffix(".analysis.manifest.json")
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def _admit_hook(hook_oracle: Path | None, attestation_path: Path | None) -> dict:
    omitted = {
        "authority_admitted": False, "omission_reason": "hook_oracle_or_parity_absent",
        "binary_sha256": None, "physics_identity": None, "attestation_sha256": None,
    }
    if hook_oracle is None or attestation_path is None:
        return omitted
    request = {
        "id": "identity", "op": "identity", "hook_speed": 900,
        "hook_pullspeed": 1700, "hook_sky": False, "hook_maxtime": 5,
    }
    result = subprocess.run(
        [str(hook_oracle)], input=canonical_json(request) + b"\n",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False,
    )
    if result.returncode:
        return {**omitted, "omission_reason": "hook_oracle_failed"}
    try:
        identity = json.loads(result.stdout)
        _exact_keys(identity, {
            "ok", "id", "op", "schema", "physics_identity", "tool_identity", "parameters", "source",
        }, "hook identity")
        if identity["schema"] != "q2-hook-oracle-v1" or identity["op"] != "identity":
            raise AtlasAnalysisError("hook identity schema mismatch")
        _require_sha(identity["physics_identity"], "hook physics identity")
        _require_sha(identity["tool_identity"], "hook tool identity")
        attestation = json.loads(attestation_path.read_bytes())
        if attestation.get("schema") != "q2-hook-parity-attestation-v1" or attestation.get("passed") is not True:
            raise AtlasAnalysisError("hook parity attestation is not passing")
        if attestation.get("identities", {}).get("hook", {}).get("physics_identity") != identity["physics_identity"]:
            raise AtlasAnalysisError("hook parity physics identity mismatch")
        if attestation.get("binaries", {}).get("hook_oracle_sha256") != sha256_file(hook_oracle):
            raise AtlasAnalysisError("hook parity executable mismatch")
        if attestation.get("evidence", {}).get("case_count") != 8:
            raise AtlasAnalysisError("hook parity case count mismatch")
    except (json.JSONDecodeError, AtlasAnalysisError):
        return {**omitted, "omission_reason": "hook_parity_rejected"}
    return {
        "authority_admitted": True,
        "omission_reason": "no_engine_evidenced_hook_corridor",
        "binary_sha256": sha256_file(hook_oracle),
        "physics_identity": identity["physics_identity"],
        "attestation_sha256": sha256_file(attestation_path),
    }
