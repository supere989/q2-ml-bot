"""Deterministic offline Atlas builder using only pinned engine physics oracles."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
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
CONTENTS_PLAYERCLIP = 0x10000
CONTENTS_LADDER = 0x20000000
STANDING_MINS = [-16, -16, -24]
STANDING_MAXS = [16, 16, 32]
CROUCHED_MINS = [-16, -16, -24]
CROUCHED_MAXS = [16, 16, 4]
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
    max_pmove_sources: int = 256


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
        if len(requests) > self.limits.oracle_batch:
            output: list[dict] = []
            for start in range(0, len(requests), self.limits.oracle_batch):
                output.extend(self.call(requests[start:start + self.limits.oracle_batch]))
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
    grounded = []
    for point, support in zip(seed_points, seed_support):
        if support["startsolid"] or support["fraction"] >= 1 or support["plane"]["normal"][2] < 0.7:
            grounded.append(point)
        else:
            grounded.append(tuple(support["endpos"]))
    probes = cm.call(_node_probe_requests(grounded))
    nodes: dict[tuple[int, int, int], NavNode] = {}
    queue: deque[tuple[int, int, int]] = deque()
    spawn_indices: dict[int, tuple[int, int, int]] = {}
    for ordinal, ((entity_ordinal, _), point) in enumerate(zip(spawns, grounded)):
        stand, crouch, contents = probes[ordinal * 3:ordinal * 3 + 3]
        key = _grid_index(point, origin, 16)
        if stand["startsolid"] and crouch["startsolid"]:
            continue
        node = NavNode(
            key, point, not stand["startsolid"], not crouch["startsolid"], True,
            contents["contents"], tuple(seed_support[ordinal]["plane"]["normal"]),
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
            )
            queue.append(key)
        traces = []
        trace_meta = []
        for source_key, point, key, _ in accepted:
            target = nodes.get(key)
            if target is None:
                continue
            source = nodes[source_key]
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

    # Exact Pmove validates step/drop traversal only from a bounded,
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
            key for key in ordered_nodes if outgoing[key] < 2 and key not in candidates
        )
        for key in candidates[:limits.max_pmove_sources]:
            source = nodes[key]
            for yaw in (0, 90, 180, 270):
                requests.append({
                    "id": f"pmove:{key[0]}:{key[1]}:{key[2]}:{yaw}", "op": "simulate",
                    "origin": list(source.position), "pm_flags": 4, "snapinitial": False,
                    "commands": [
                        {"msec": 50, "angles": [0, yaw, 0], "forwardmove": 300}
                        for _ in range(4)
                    ],
                })
                metadata.append(key)
        for source_key, result in zip(metadata, pmove.call(requests)):
            target_key = _grid_index(result["final"]["origin"], origin, 16)
            if target_key == source_key or target_key not in nodes or not result["final"]["grounded"]:
                continue
            vertical = nodes[target_key].position[2] - nodes[source_key].position[2]
            kind = "controlled_drop" if vertical < -18 else "step"
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

    adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for edge in edges:
        adjacency[tuple(edge["source"])].append(tuple(edge["target"]))
    region = 0
    for key in sorted(nodes, key=lambda item: (item[2], item[1], item[0])):
        if nodes[key].region_id:
            continue
        region += 1
        nodes[key].region_id = region
        pending = [key]
        while pending:
            current = pending.pop()
            neighbors = adjacency[current] + [source for source, values in adjacency.items() if current in values]
            for neighbor in neighbors:
                if nodes[neighbor].region_id == 0:
                    nodes[neighbor].region_id = region
                    pending.append(neighbor)
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
) -> list[dict]:
    chunks: dict[tuple[int, int, int], dict[str, Any]] = {}

    def set_bit(point: Sequence[float], plane: str) -> None:
        l0 = _grid_index(point, origin, 4)
        chunk = tuple(value // 16 for value in l0)
        local = tuple(value % 16 for value in l0)
        linear = local[0] + 16 * local[1] + 256 * local[2]
        item = chunks.setdefault(chunk, {"key": list(chunk), "bits": defaultdict(set), "scalars": {}})
        item["bits"][plane].add(linear)

    for key, node in nodes.items():
        # L1 supported_floor is authoritative for ordinary traversable floor.
        # L0 retains only surface/obstacle boundary bands; materializing one
        # solid voxel beneath every reachable L1 origin would be the forbidden
        # complete floor fill described by the Atlas contract.
        boundary = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # A legal 18-unit step commonly changes the snapped L1 z index by
            # one. Treat the horizontal column as present when any of those
            # exact oracle-supported origins exists; same-z-only lookup would
            # mislabel ordinary stairs/sloped floors as obstacle boundaries.
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
            set_bit((node.position[0], node.position[1], node.position[2] - 25), "solid")
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
    for _, spawn in spawns:
        # A 96-unit column is the half-open interval [0, 96); sampling the
        # endpoint would retain a 100-unit column and can create an unrelated
        # upper chunk at an exact 64-unit boundary.
        for z in range(0, 96, 4):
            set_bit((spawn[0], spawn[1], spawn[2] + z), "spawn_column")
    output = []
    for key in sorted(chunks, key=lambda item: (item[2], item[1], item[0])):
        item = chunks[key]
        item["bits"] = {name: sorted(values) for name, values in sorted(item["bits"].items())}
        output.append(item)
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
) -> dict:
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
    origin = _snapped_origin(metadata.model0.mins)
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
        pmove_context = OracleProcess(pmove_oracle, bsp, "pmove", limits) if pmove_oracle else None
        try:
            if pmove_context and pmove_context.identity["map_sha256"] != metadata.sha256:
                raise AtlasAnalysisError("Pmove oracle loaded different BSP bytes")
            nodes, edges, spawn_indices = _build_navigation(
                cm, pmove_context, player_spawns, origin, limits,
            )
            visibility = _visibility(cm, nodes, limits)
            spawn_records = []
            for entity_ordinal, point in spawns:
                key = spawn_indices.get(entity_ordinal)
                node = nodes.get(key) if key else None
                player_point = (point[0], point[1], point[2] + 9.0)
                column = _column_clearance(cm, player_point)
                reachable = []
                if node is not None:
                    reachable = [
                        other for other, other_key in sorted(spawn_indices.items())
                        if other != entity_ordinal and nodes.get(other_key) is not None
                        and nodes[other_key].region_id == node.region_id
                    ]
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
        "chunks": _l0_chunks(nodes, player_spawns, origin),
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
        Path(__file__).resolve().parents[1] / "crates/q2-lattice/src/bin/q2_atlas_pack.rs",
        Path(__file__).resolve().parents[1] / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md",
    ]
    analyzer_sha256 = sha256_bytes(canonical_json([
        {"path": path.name, "sha256": sha256_file(path)} for path in analyzer_inputs
    ]))
    manifest = {
        "schema": SCHEMA,
        "status": "passed",
        "deterministic_rebuild": True,
        "confidence": "high",
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
                "l0_raw_cells": sum(1 for node in nodes.values() if node.contents & (CONTENTS_LAVA | CONTENTS_SLIME)),
                "l0_expanded_cells": 0, "types": sorted(
                    ({"lava"} if any(node.contents & CONTENTS_LAVA for node in nodes.values()) else set()) |
                    ({"slime"} if any(node.contents & CONTENTS_SLIME for node in nodes.values()) else set())
                ),
                "lethal_drop_edges": 0, "guarded_drop_edges": 0, "uncontained_drop_edges": 0,
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
            },
            "route_claims": [],
        },
        "artifacts": artifacts,
        "counts": {
            "l0_chunks": pack_result["l0_chunks"], "l1_nodes": len(nodes),
            "l1_edges": len(edges), "l2_cells": pack_result["l2_cells"],
            "l3_cells": pack_result["l3_cells"],
        },
        "confidence_summary": {
            "collision": "exact-engine", "movement": "exact-engine" if pmove_oracle else "omitted",
            "hook": "attested-but-no-discovered-edge" if hook["authority_admitted"] else "omitted",
            "metadata": "b1-c-validated",
        },
        "limitations": sorted([
            "areaportal summaries use declared map-static state only",
            "hook authority is admitted but stock surface discovery emits no hook edge in B2-A v1",
            "inline mover transforms remain confidence-tagged metadata and are not static corridors",
            "L0 is a reachable surface/hazard/spawn narrow band, never dense free-space fill",
        ]),
        "performance": {
            "cm_requests": oracle_manifest["collision"]["requests"],
            "pmove_requests": 0 if oracle_manifest["pmove"] is None else oracle_manifest["pmove"]["requests"],
        },
    }
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
