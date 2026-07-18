"""Deterministic offline Atlas builder using only pinned engine physics oracles."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import AbstractSet, Any, Iterable, Mapping, Sequence

import zstandard

from .atlas_entity_semantics import (
    Aabb,
    Authority,
    L0BudgetState,
    L0PlaneKind,
    entity_angles,
    ordered_property,
    platform_mover_semantics,
    rotating_mover_semantics,
    sliding_mover_semantics,
    train_swept_geometry,
    train_topology,
    trigger_hurt_semantics,
)
from .atlas_surface_bands import (
    FixedInlineModelPose,
    Model0Pose,
    SurfaceBandChunk,
    SurfaceCandidateGroup,
    discover_scoped_surface_bands,
)
from .ibsp38 import BspMetadata, EntityMetadata, parse_ibsp38
from .generated_claim_probes import (
    GeneratedClaimProbeError,
    analyze_non_hook_claims,
)
from .hook_claims_v4 import (
    HookClaimsV4Error,
    validate_candidate_record as validate_hook_candidate_record_v4,
    validate_materialization as validate_hook_materialization_v4,
    validate_selected_record as validate_hook_selected_record_v4,
    validation_trace_sha256,
)
from .atlas_b1_authority import B1AuthorityError, admit_b1_runtime_authorities
from .atlas_source_closure import atlas_analyzer_authority_sha256
from .atlas_exact_drops import (
    DROP_EVIDENCE_EXACT,
    DROP_VALIDATION_VERSION,
    DropTrajectory,
    ExactDropAnalysisError,
    FallOracleProcess,
    classify_drop_trajectories,
    summarize_drop_classifications,
)
from .atlas_teleporter_edges import (
    prove_trigger_teleporter_edges,
    resolve_trigger_teleporters,
    teleporter_seed_points,
)


SCHEMA = "q2-atlas-analysis-v1"
OBJECTIVE_SCHEMA = "q2-atlas-objectives-v1"
OBJECTIVE_MEDIA_TYPE = "application/vnd.q2.atlas-objectives-v1"
OBJECTIVE_TARGET_MAX_DISTANCE = 160.0
OBJECTIVE_GUIDEPOST_ANALYSIS_SCHEMA = "q2-atlas-objective-guidepost-analysis-v1"
OBJECTIVE_OMISSION_BEYOND_FENCE = "beyond_objective_target_max_distance"
OBJECTIVE_OMISSION_NO_TARGET = "no_admitted_supported_passable_l1"
BUILD_PLAN_SCHEMA = "q2-atlas-build-plan-v2"
EVIDENCE_CM_TRACE_V1 = 1
EVIDENCE_PMOVE_V1 = 2
EVIDENCE_HOOK_LAW_V1 = 4
EVIDENCE_FALL_LAW_V1 = 8
VALIDATION_VERSION = 1
PMOVE_GROUND_HORIZON_FRAMES = 4
PMOVE_JUMP_HORIZON_FRAMES = 16
PMOVE_DROP_HORIZON_FRAMES = 40
PMOVE_LADDER_CLIMB_FRAMES = 44
PMOVE_LADDER_SETTLE_HORIZON_FRAMES = 52
PMOVE_TRAJECTORY_BATCH_SIZE = 32
PMOVE_FIXED_QUANTUM = 0.125
MAX_L0_SEMANTIC_SCRATCH_BYTES = 16 * 1024 * 1024
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
    max_surface_candidate_cells: int = 250_000
    max_surface_request_upper_bound: int = 16_000_000
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
        output: list[dict] = []
        batch: list[dict] = []
        for request in requests:
            multi_frame_pmove = (
                self.kind == "pmove"
                and len(request.get("commands", [])) > 1
            )
            if multi_frame_pmove:
                # A trajectory response can exceed the duplex pipe capacity.
                # Drain any preceding small requests, then write and read this
                # request alone so parent and child cannot block while each is
                # writing the opposite pipe.
                if batch:
                    output.extend(self._call_batch(batch))
                    batch = []
                output.extend(self._call_batch([request]))
                continue
            batch.append(request)
            if len(batch) == self.limits.oracle_batch:
                output.extend(self._call_batch(batch))
                batch = []
        if batch:
            output.extend(self._call_batch(batch))
        return output

    def _call_batch(self, requests: Sequence[dict]) -> list[dict]:
        if (
            self.kind == "pmove"
            and len(requests) > 1
            and any(len(request.get("commands", [])) > 1 for request in requests)
        ):
            raise AtlasAnalysisError(
                "multi-frame pmove request entered a shared transport batch"
            )
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


class HookOracleProcess:
    """Persistent, identity-pinned interface to the attested pure hook law."""

    PARAMETERS = {
        "hook_speed": 900,
        "hook_pullspeed": 1700,
        "hook_sky": False,
        "hook_maxtime": 5,
    }

    def __init__(
        self,
        binary: Path,
        expected_physics_identity: str,
        limits: AnalyzerLimits,
    ) -> None:
        self.binary = binary.resolve()
        self.expected_physics_identity = expected_physics_identity
        self.limits = limits
        self.requests = 0
        self.process = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise AtlasAnalysisError("failed to open hook oracle pipes")
        identity = self.call({"id": "identity", "op": "identity"})
        if identity.get("op") != "identity":
            raise AtlasAnalysisError("hook identity operation mismatch")
        self.identity = identity

    def call(self, request: Mapping[str, Any]) -> dict:
        self.requests += 1
        if self.requests > self.limits.max_oracle_requests:
            raise AtlasAnalysisError("hook oracle request budget exceeded")
        payload = {**self.PARAMETERS, **dict(request)}
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(canonical_json(payload) + b"\n")
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise AtlasAnalysisError("hook oracle pipe closed") from error
        deadline = time.monotonic() + self.limits.oracle_batch_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise AtlasAnalysisError("hook oracle request timed out")
            ready, _, _ = select.select([self.process.stdout.fileno()], [], [], remaining)
            if ready:
                raw = self.process.stdout.readline()
                break
        if not raw:
            detail = b"" if self.process.stderr is None else self.process.stderr.read(4096)
            raise AtlasAnalysisError(
                f"hook oracle exited early: {detail.decode(errors='replace')}"
            )
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AtlasAnalysisError("hook oracle emitted invalid JSON") from error
        if record.get("ok") is not True or record.get("id") != request.get("id"):
            raise AtlasAnalysisError(
                f"hook oracle rejected {request.get('op')}: {record.get('error')}"
            )
        if record.get("schema") != "q2-hook-oracle-v1":
            raise AtlasAnalysisError("hook oracle schema mismatch")
        if record.get("physics_identity") != self.expected_physics_identity:
            raise AtlasAnalysisError("hook oracle physics identity mismatch")
        return record

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
            detail = b"" if self.process.stderr is None else self.process.stderr.read(4096)
            raise AtlasAnalysisError(
                f"hook oracle exited {self.process.returncode}: "
                f"{detail.decode(errors='replace')}"
            )

    def __enter__(self) -> "HookOracleProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None:
            self._kill()
            self.process.wait()
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
    atlas_hazard_types: int = 0
    atlas_hazard_severity: int = 0
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
        hazard = self.atlas_hazard_types
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
            "hazard_severity": max(
                self.atlas_hazard_severity, 255 if hazard else 0,
            ),
            "cost_to_safety": 0 if safe else 0xFFFFFFFF,
            "region_id": self.region_id,
            "confidence": 65535,
            "evidence": EVIDENCE_CM_TRACE_V1,
            "contents_flags": self.contents & 0xFFFFFFFF,
        }


MovementRecord = tuple[
    tuple[int, int, int], str, int, str, dict[str, Any]
]


@dataclass(frozen=True)
class _MoverDependencyIndex:
    """Conservative dynamic-brush occupancy relevant to Pmove trajectories."""

    envelopes: tuple[Aabb, ...]
    globally_unknown: bool = False

    def intersects_box_path(
        self,
        points: Sequence[Sequence[float]],
        mins: Sequence[float],
        maxs: Sequence[float],
    ) -> bool:
        """Conservatively test swept hull segments against mover envelopes."""

        if self.globally_unknown:
            return True
        if not points or any(len(point) != 3 for point in points):
            return True
        if len(mins) != 3 or len(maxs) != 3:
            return True
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for vector in (*points, mins, maxs) for value in vector
        ):
            return True
        hulls = [
            Aabb(
                tuple(point[axis] + mins[axis] for axis in range(3)),
                tuple(point[axis] + maxs[axis] for axis in range(3)),
            )  # type: ignore[arg-type]
            for point in points
        ]
        segments = hulls[:1] + [
            Aabb(
                tuple(min(left.mins[axis], right.mins[axis]) for axis in range(3)),
                tuple(max(left.maxs[axis], right.maxs[axis]) for axis in range(3)),
            )  # type: ignore[arg-type]
            for left, right in zip(hulls, hulls[1:])
        ]
        return any(
            all(
                segment.maxs[axis] >= mover.mins[axis]
                and mover.maxs[axis] >= segment.mins[axis]
                for axis in range(3)
            )
            for segment in segments for mover in self.envelopes
        )

    def intersects_trajectory(
        self, response: Mapping[str, Any], request: Mapping[str, Any] | None = None,
    ) -> bool:
        if self.globally_unknown:
            return True
        frames = response.get("frames")
        if not isinstance(frames, list):
            return True
        previous: Aabb | None = None
        if request is not None and frames:
            initial = request.get("origin")
            first = frames[0]
            if (
                isinstance(initial, list) and len(initial) == 3
                and isinstance(first, Mapping)
                and isinstance(first.get("mins"), list)
                and isinstance(first.get("maxs"), list)
                and len(first["mins"]) == 3 and len(first["maxs"]) == 3
                and all(
                    type(axis) in (int, float) and math.isfinite(axis)
                    for values in (initial, first["mins"], first["maxs"])
                    for axis in values
                )
            ):
                previous = Aabb(
                    tuple(initial[axis] + first["mins"][axis] for axis in range(3)),
                    tuple(initial[axis] + first["maxs"][axis] for axis in range(3)),
                )  # type: ignore[arg-type]
            else:
                return True
        for frame in frames:
            if not isinstance(frame, Mapping):
                return True
            origin = frame.get("origin")
            mins = frame.get("mins")
            maxs = frame.get("maxs")
            if not all(
                isinstance(value, list) and len(value) == 3
                and all(type(axis) in (int, float) and math.isfinite(axis) for axis in value)
                for value in (origin, mins, maxs)
            ):
                return True
            player = Aabb(
                tuple(origin[axis] + mins[axis] for axis in range(3)),
                tuple(origin[axis] + maxs[axis] for axis in range(3)),
            )  # type: ignore[arg-type]
            swept = player if previous is None else Aabb(
                tuple(min(previous.mins[axis], player.mins[axis]) for axis in range(3)),
                tuple(max(previous.maxs[axis], player.maxs[axis]) for axis in range(3)),
            )  # type: ignore[arg-type]
            if any(
                all(
                    swept.maxs[axis] >= mover.mins[axis]
                    and mover.maxs[axis] >= swept.mins[axis]
                    for axis in range(3)
                )
                for mover in self.envelopes
            ):
                return True
            previous = player
        return False


def _dynamic_mover_dependency_index(
    metadata: BspMetadata, *, exclude_entity_index: int | None = None,
) -> _MoverDependencyIndex:
    """Derive conservative mover envelopes without treating them as collision.

    An unsupported mover law poisons all Pmove-derived traversal.  This is
    intentionally fail closed: q2-pmove-oracle v1 has no mover-state conduit.
    """

    entities = {entity.index: entity for entity in metadata.entities}
    envelopes: list[Aabb] = []
    globally_unknown = False

    def translated(bounds: Aabb, translation: Sequence[float]) -> Aabb:
        return Aabb(
            tuple(bounds.mins[axis] + translation[axis] for axis in range(3)),
            tuple(bounds.maxs[axis] + translation[axis] for axis in range(3)),
        )  # type: ignore[arg-type]

    for record in metadata.entity_catalog.brush_submodels:
        entity = entities[int(record["entity_index"])]
        if entity.index == exclude_entity_index:
            continue
        classname = entity.classname.casefold()
        if not classname.startswith("func_") or classname == "func_areaportal":
            continue
        model = metadata.models[int(record["model_index"])]
        cmodel = Aabb(
            tuple(value - 1.0 for value in model.mins),
            tuple(value + 1.0 for value in model.maxs),
        )  # type: ignore[arg-type]
        if classname in {"func_wall", "func_explosive", "func_object"}:
            # These inline models are absent from model-0 Pmove collision even
            # though their game edicts can be solid.  A wall can toggle and an
            # explosive can disappear, so their exact present-pose bounds are
            # a potential-occupancy envelope.  A func_object additionally
            # enters toss physics; without an edict-state/gravity conduit its
            # future occupancy law is unbounded and all traversal must remain
            # Unknown.
            try:
                entity_origin = _origin(entity, default=(0.0, 0.0, 0.0))
                angles = entity_angles(entity.properties)
            except AtlasAnalysisError:
                globally_unknown = True
                continue
            if not angles.is_exact or angles.value != (0.0, 0.0, 0.0):
                globally_unknown = True
                continue
            envelopes.append(translated(cmodel, entity_origin))
            if classname == "func_object":
                globally_unknown = True
        elif classname in {"func_door", "func_button", "func_water"}:
            result = sliding_mover_semantics(entity, cmodel.mins, cmodel.maxs)
            if result.is_exact and result.value is not None:
                envelopes.append(result.value.potential_envelope.bounds)
            else:
                globally_unknown = True
        elif classname == "func_plat":
            result = platform_mover_semantics(entity, cmodel.mins, cmodel.maxs)
            if result.is_exact and result.value is not None:
                envelopes.append(result.value.potential_envelope.bounds)
            else:
                globally_unknown = True
        elif classname in {"func_rotating", "func_door_rotating"}:
            result = rotating_mover_semantics(entity, cmodel.mins, cmodel.maxs)
            if result.is_exact and result.value is not None:
                envelopes.append(result.value.potential_envelope.bounds)
            else:
                globally_unknown = True
        elif classname == "func_train":
            result = train_topology(entity, metadata.entities, cmodel.mins)
            geometry = (
                train_swept_geometry(result.value, cmodel.mins, cmodel.maxs)
                if result.is_exact and result.value is not None else None
            )
            if (
                geometry is None or not geometry.is_exact
                or geometry.value is None or not geometry.value.pose_bounds
            ):
                globally_unknown = True
            else:
                envelopes.extend(geometry.value.pose_bounds)
                envelopes.extend(geometry.value.linear_segment_bounds)
        else:
            # Unimplemented mover classes have no admitted path law.
            # Current-pose bounds cannot prove their future occupancy.
            globally_unknown = True
    return _MoverDependencyIndex(tuple(envelopes), globally_unknown)


def _origin(
    entity: EntityMetadata,
    *,
    default: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    raw = entity.value("origin")
    if not raw and default is not None:
        return default
    words = raw.split()
    if len(words) != 3:
        raise AtlasAnalysisError(f"entity {entity.index} has invalid origin")
    try:
        value = tuple(float(word) for word in words)
    except ValueError as error:
        raise AtlasAnalysisError(f"entity {entity.index} has invalid origin") from error
    if not all(math.isfinite(axis) for axis in value):
        raise AtlasAnalysisError(f"entity {entity.index} has non-finite origin")
    return value  # type: ignore[return-value]


def _authored_item_destinations(
    metadata: BspMetadata,
) -> list[tuple[float, float, float]]:
    """Return point-entity destinations that seed the stance-aware L1 flood."""

    destinations = []
    for entity in metadata.entities:
        classname = entity.classname.casefold()
        if classname == "info_player_deathmatch":
            continue
        objective_class = _objective_class(classname)
        if objective_class is None:
            if classname.startswith(("item_", "weapon_", "ammo_", "key_", "rune_")):
                raise AtlasAnalysisError(
                    f"entity {entity.index} has unsupported objective class {classname}"
                )
            continue
        destinations.append(_origin(entity))
    return destinations


def _objective_class(classname: str) -> str | None:
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


def _objective_target_distance(
    node: NavNode, world_point: Sequence[float],
) -> float:
    return math.sqrt(sum(
        (node.position[axis] - world_point[axis]) ** 2 for axis in range(3)
    ))


def _is_supported_passable_objective_target(
    node: NavNode,
    key: tuple[int, int, int],
    incident: AbstractSet[tuple[int, int, int]],
) -> bool:
    """Return whether an L1 node may host a public objective guidepost."""

    return (
        key in incident
        and node.supported
        and (node.standing_clear or node.crouched_clear)
    )


def _objective_omission(
    entity_id: int,
    classname: str,
    reason: str,
    nearest_distance: float | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "classname": classname,
        "entity_id": entity_id,
        "reason": reason,
    }
    if nearest_distance is not None:
        # Milliunits keep omission evidence integer-stable across cold rebuilds.
        record["nearest_distance_milliunits"] = int(round(nearest_distance * 1000.0))
    return record


def _objective_artifact(
    metadata: BspMetadata,
    nodes: Mapping[tuple[int, int, int], NavNode],
    spawn_indices: Mapping[int, tuple[int, int, int]],
    origin: tuple[int, int, int],
    canonical_map_id: str,
    atlas_sha256: str,
    *,
    incident: AbstractSet[tuple[int, int, int]] | None = None,
    strict_binding: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Emit public objective guideposts bound to admitted L1 targets.

    Generated analysis (`strict_binding=True`, generator claims present) fails
    closed when any supported objective class lacks an admitted
    supported/passable L1 target within ``OBJECTIVE_TARGET_MAX_DISTANCE``.

    Authored/stock analysis without generator claims never rebinds beyond the
    fence: unbound objectives are omitted with deterministic evidence, while
    the objectives JSON schema and 160-unit runtime validator stay unchanged.
    Stock item completeness remains a separate inventory/design-signature check.
    """

    if incident is None:
        incident = set(nodes)
    candidates = {
        key: node for key, node in nodes.items()
        if _is_supported_passable_objective_target(node, key, incident)
    }
    records = []
    omissions: list[dict[str, Any]] = []
    for entity in metadata.entities:
        classname = entity.classname.casefold()
        objective_class = _objective_class(classname)
        if objective_class is None:
            if classname.startswith(("item_", "weapon_", "ammo_", "key_", "rune_")):
                raise AtlasAnalysisError(
                    f"entity {entity.index} has unsupported objective class {classname}"
                )
            continue
        authored = _origin(entity)
        world_point = (
            (authored[0], authored[1], authored[2] + 9.0)
            if classname == "info_player_deathmatch" else authored
        )
        target = spawn_indices.get(entity.index)
        if classname == "info_player_deathmatch" and target is None:
            # Only engine-linked, oracle-clear spawns are public egress facts.
            continue
        nearest_distance: float | None = None
        if target is None:
            ranked = sorted(
                candidates.values(),
                key=lambda node: (
                    _objective_target_distance(node, world_point),
                    node.index[2], node.index[1], node.index[0],
                ),
            )
            if not ranked:
                if nodes:
                    nearest_any = min(
                        nodes.values(),
                        key=lambda node: (
                            _objective_target_distance(node, world_point),
                            node.index[2], node.index[1], node.index[0],
                        ),
                    )
                    nearest_distance = _objective_target_distance(
                        nearest_any, world_point,
                    )
                if strict_binding:
                    raise AtlasAnalysisError(
                        f"objective entity {entity.index} has no admitted L1 target"
                    )
                omissions.append(_objective_omission(
                    entity.index, classname, OBJECTIVE_OMISSION_NO_TARGET,
                    nearest_distance,
                ))
                continue
            target = ranked[0].index
            nearest_distance = _objective_target_distance(ranked[0], world_point)
        node = nodes.get(target)
        if node is None or not _is_supported_passable_objective_target(
            node, target, incident,
        ):
            if candidates:
                nearest = min(
                    candidates.values(),
                    key=lambda candidate: (
                        _objective_target_distance(candidate, world_point),
                        candidate.index[2], candidate.index[1], candidate.index[0],
                    ),
                )
                nearest_distance = _objective_target_distance(nearest, world_point)
            elif nodes:
                nearest_any = min(
                    nodes.values(),
                    key=lambda candidate: (
                        _objective_target_distance(candidate, world_point),
                        candidate.index[2], candidate.index[1], candidate.index[0],
                    ),
                )
                nearest_distance = _objective_target_distance(
                    nearest_any, world_point,
                )
            if strict_binding:
                raise AtlasAnalysisError(
                    f"objective entity {entity.index} target is not in Atlas L1"
                    if node is None else
                    f"objective entity {entity.index} target is not supported/passable"
                )
            omissions.append(_objective_omission(
                entity.index, classname, OBJECTIVE_OMISSION_NO_TARGET,
                nearest_distance,
            ))
            continue
        distance = (
            nearest_distance if nearest_distance is not None
            else _objective_target_distance(node, world_point)
        )
        if distance > OBJECTIVE_TARGET_MAX_DISTANCE:
            if strict_binding:
                raise AtlasAnalysisError(
                    f"objective entity {entity.index} is {distance:.3f} units "
                    f"from admitted L1"
                )
            omissions.append(_objective_omission(
                entity.index, classname, OBJECTIVE_OMISSION_BEYOND_FENCE,
                distance,
            ))
            continue
        hazard = bool(node.contents & (CONTENTS_LAVA | CONTENTS_SLIME))
        records.append({
            "class": objective_class,
            "classname": classname,
            "confidence": 65535,
            "l1_index": list(target),
            "objective_id": entity.index,
            "risk": 65535 if hazard else 0,
            "world_milliunits": _milliunits(world_point, f"objective {entity.index}"),
        })
    records.sort(key=lambda record: record["objective_id"])
    omissions.sort(key=lambda record: record["entity_id"])
    document = {
        "atlas_sha256": atlas_sha256,
        "bsp_sha256": metadata.sha256,
        "canonical_map_id": canonical_map_id,
        "objectives": records,
        "origin": list(origin),
        "schema": OBJECTIVE_SCHEMA,
    }
    return document, omissions


def _objective_guidepost_analysis(
    admitted_count: int,
    omissions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": OBJECTIVE_GUIDEPOST_ANALYSIS_SCHEMA,
        "admitted_count": admitted_count,
        "omitted_count": len(omissions),
        "omissions": [dict(item) for item in omissions],
    }


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
        requests.extend((
            _box_request(
                f"floor-raised:{ordinal}",
                (point[0], point[1], point[2] + 18),
                (point[0], point[1], point[2] - 32),
                STANDING_MINS, STANDING_MAXS,
            ),
            _box_request(
                f"floor-nominal:{ordinal}",
                (point[0], point[1], point[2] + PMOVE_FIXED_QUANTUM),
                (point[0], point[1], point[2] - 32),
                STANDING_MINS, STANDING_MAXS,
            ),
        ))
    return requests


def _supported_floor_candidate(
    source_key: tuple[int, int, int],
    floor: Mapping[str, Any],
    origin: tuple[int, int, int],
) -> tuple[tuple[float, float, float], tuple[int, int, int]] | None:
    """Retain every distinct supported cell found by the bounded floor trace."""

    if (
        floor["startsolid"] or floor["fraction"] >= 1
        or floor["plane"]["normal"][2] < 0.7
    ):
        return None
    point = tuple(floor["endpos"])
    key = _grid_index(point, origin, 16)
    if key == source_key:
        return None
    return point, key


def _node_probe_requests(points: Sequence[tuple[float, float, float]]) -> list[dict]:
    requests = []
    for ordinal, point in enumerate(points):
        requests.extend([
            _box_request(f"stand:{ordinal}", point, point, STANDING_MINS, STANDING_MAXS),
            _box_request(f"crouch:{ordinal}", point, point, CROUCHED_MINS, CROUCHED_MAXS),
            {"id": f"contents:{ordinal}", "op": "point_contents", "point": list(point)},
        ])
    return requests


def _ground_navigation_seeds(
    cm: OracleProcess,
    seeds: Sequence[
        tuple[int | None, tuple[float, float, float], bool]
    ],
) -> tuple[list[tuple[float, float, float] | None], list[dict[str, Any]]]:
    """Ground navigation seeds without moving an engine spawn into a ceiling.

    A deathmatch spawn is already an engine-owned player origin (including the
    nine-unit spawn lift applied by the caller).  Prove that exact standing
    origin clear first, then trace downward from it for support.  Raising the
    hull before the support trace is not part of the spawn lifecycle and can
    begin inside a low ceiling even when the authored spawn is legal.

    Non-spawn destinations retain the bounded raised floor search because an
    authored item/teleporter point is not necessarily a player origin.  A
    second exact-Pmove-quantum start proves legal support when the raised hull
    begins inside a low overhang; the raised result remains preferred when
    both are valid.
    """

    spawn_ordinals = [
        ordinal for ordinal, (_, _, is_spawn) in enumerate(seeds) if is_spawn
    ]
    spawn_clear = cm.call([
        _box_request(
            f"seed-spawn-clear:{ordinal}", point, point,
            STANDING_MINS, STANDING_MAXS,
        )
        for ordinal, (_, point, is_spawn) in enumerate(seeds) if is_spawn
    ])
    clear_by_ordinal = dict(zip(spawn_ordinals, spawn_clear))
    support_requests = []
    support_ranges = []
    for ordinal, (_, point, is_spawn) in enumerate(seeds):
        start = len(support_requests)
        if is_spawn:
            support_requests.append(_box_request(
                f"seed-floor:{ordinal}", point,
                (point[0], point[1], point[2] - 96),
                STANDING_MINS, STANDING_MAXS,
            ))
        else:
            support_requests.extend((
                _box_request(
                    f"seed-floor-raised:{ordinal}",
                    (point[0], point[1], point[2] + 32),
                    (point[0], point[1], point[2] - 96),
                    STANDING_MINS, STANDING_MAXS,
                ),
                _box_request(
                    f"seed-floor-nominal:{ordinal}",
                    (point[0], point[1], point[2] + PMOVE_FIXED_QUANTUM),
                    (point[0], point[1], point[2] - 96),
                    STANDING_MINS, STANDING_MAXS,
                ),
            ))
        support_ranges.append((start, len(support_requests)))
    support_results = cm.call(support_requests)
    support = []
    for start, end in support_ranges:
        candidates = support_results[start:end]
        support.append(next((
            floor for floor in candidates
            if (
                not floor["startsolid"] and not floor["allsolid"]
                and floor["fraction"] < 1
                and floor["plane"]["normal"][2] >= 0.7
            )
        ), candidates[0]))
    grounded: list[tuple[float, float, float] | None] = []
    for ordinal, ((_, _, is_spawn), floor) in enumerate(zip(seeds, support)):
        if is_spawn:
            clear = clear_by_ordinal[ordinal]
            if (
                clear["startsolid"] or clear["allsolid"]
                or clear["fraction"] != 1
            ):
                grounded.append(None)
                continue
        if (
            floor["startsolid"] or floor["allsolid"]
            or floor["fraction"] >= 1
            or floor["plane"]["normal"][2] < 0.7
        ):
            grounded.append(None)
        else:
            grounded.append(tuple(floor["endpos"]))
    return grounded, support


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


def _complete_pmove_source_set(
    nodes: Mapping[tuple[int, int, int], NavNode],
    edges: Sequence[Mapping[str, Any]],
    spawn_indices: Mapping[int, tuple[int, int, int]],
    maximum: int,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Select deterministic component/frontier-stratified Pmove sources.

    The walk graph is the topology authority for ordinary boundary sampling.
    Spawn keys lead the order, followed by every exact ladder-content source,
    then one frontier representative for every weak walk-graph component not
    already covered by a frontier spawn or ladder node. Remaining capacity is
    apportioned by the exact selected/available frontier ratio; within a
    component, deterministic farthest-point sampling prevents a coordinate
    prefix from starving remote or high boundary transitions.

    The expensive sampling work is bounded by ``maximum``.  Unselected
    candidates are still ordered deterministically for complete omission
    accounting, but are never challenged through Pmove and therefore cannot
    emit Pmove-derived edges.
    """

    if maximum < 0:
        raise AtlasAnalysisError("Pmove source maximum is negative")

    def zyx(key: tuple[int, int, int]) -> tuple[int, int, int]:
        return key[2], key[1], key[0]

    def squared_distance(
        left: tuple[int, int, int], right: tuple[int, int, int],
    ) -> int:
        return sum((left[axis] - right[axis]) ** 2 for axis in range(3))

    def farthest(
        candidates: Sequence[tuple[int, int, int]],
        anchors: Sequence[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        if not anchors:
            return min(candidates, key=zyx)
        return min(
            candidates,
            key=lambda candidate: (
                -min(squared_distance(candidate, anchor) for anchor in anchors),
                *zyx(candidate),
            ),
        )

    outgoing: dict[
        tuple[int, int, int], set[tuple[int, int, int]]
    ] = defaultdict(set)
    adjacency: dict[
        tuple[int, int, int], set[tuple[int, int, int]]
    ] = defaultdict(set)
    for edge in edges:
        if edge.get("edge_type") != "walk":
            continue
        source = tuple(edge["source"])
        target = tuple(edge["target"])
        if source not in nodes or target not in nodes:
            raise AtlasAnalysisError("walk edge references an unknown L1 node")
        outgoing[source].add(target)
        adjacency[source].add(target)
        adjacency[target].add(source)

    ordered_nodes = sorted(nodes, key=zyx)
    components: list[list[tuple[int, int, int]]] = []
    component_of: dict[tuple[int, int, int], int] = {}
    for seed in ordered_nodes:
        if seed in component_of:
            continue
        component_id = len(components)
        pending = [seed]
        component_of[seed] = component_id
        component: list[tuple[int, int, int]] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ()), key=zyx, reverse=True):
                if neighbor not in component_of:
                    component_of[neighbor] = component_id
                    pending.append(neighbor)
        components.append(sorted(component, key=zyx))

    spawn_order: list[tuple[int, int, int]] = []
    spawn_seen: set[tuple[int, int, int]] = set()
    for _, key in sorted(spawn_indices.items()):
        if key not in nodes:
            raise AtlasAnalysisError("Pmove spawn source is not an L1 node")
        if key not in spawn_seen:
            spawn_order.append(key)
            spawn_seen.add(key)
    if len(spawn_order) > maximum:
        raise AtlasAnalysisError(
            "Pmove source cap cannot retain every unique spawn source"
        )

    frontiers: list[list[tuple[int, int, int]]] = [
        [key for key in component if len(outgoing[key]) < 4]
        for component in components
    ]
    frontier_sets = [set(component) for component in frontiers]

    # Tier 1: every unique spawn in entity-ordinal order.
    priority = list(spawn_order)
    priority_set = set(priority)

    # Tier 2: every node whose exact point contents contain CONTENTS_LADDER.
    # Ladder traversal is contact-local; coordinate sampling may not omit a
    # legal contact source and thereby make a one-way topology artifact.
    ladder_order = sorted(
        (key for key, node in nodes.items() if node.contents & CONTENTS_LADDER),
        key=zyx,
    )
    for key in ladder_order:
        if key not in priority_set:
            priority.append(key)
            priority_set.add(key)
    if len(priority) > maximum:
        raise AtlasAnalysisError(
            "Pmove source cap cannot retain every exact ladder-content source"
        )

    # Tier 3: before any component receives a secondary frontier source, every
    # walk component gets a frontier representative.  A frontier spawn already
    # satisfies this tier for its component, as does a retained ladder frontier.
    # Otherwise choose the frontier farthest from that component's already
    # mandatory sources, or its canonical first frontier when it has none.
    for component_id, component_frontiers in enumerate(frontiers):
        if not component_frontiers or frontier_sets[component_id] & priority_set:
            continue
        component_mandatory = [
            key for key in priority if component_of[key] == component_id
        ]
        representative = farthest(component_frontiers, component_mandatory)
        priority.append(representative)
        priority_set.add(representative)
    if len(priority) > maximum:
        raise AtlasAnalysisError(
            "Pmove source cap cannot retain mandatory component-frontier coverage"
        )

    # Tier 4: proportionally cover all remaining frontiers.  Exact rational
    # ratios avoid float-dependent ordering.  Farthest-point selection is
    # anchored by both spawn and already-selected frontier sources.
    frontier_selected = [
        len(frontier_sets[index] & priority_set)
        for index in range(len(components))
    ]
    anchors: list[list[tuple[int, int, int]]] = [
        [key for key in priority if component_of[key] == component_id]
        for component_id in range(len(components))
    ]
    remaining: list[list[tuple[int, int, int]]] = [
        [key for key in component if key not in priority_set]
        for component in frontiers
    ]
    nearest_anchor_distance: list[dict[tuple[int, int, int], int]] = [
        {
            candidate: min(
                squared_distance(candidate, anchor)
                for anchor in anchors[component_id]
            )
            for candidate in candidates
        }
        for component_id, candidates in enumerate(remaining)
    ]
    target_count = min(
        maximum,
        len(spawn_seen | set(ladder_order) | set().union(*frontier_sets)),
    )
    while len(priority) < target_count:
        active = [
            component_id for component_id, candidates in enumerate(remaining)
            if candidates
        ]
        if not active:
            break
        component_id = min(
            active,
            key=lambda index: (
                Fraction(frontier_selected[index], len(frontiers[index])),
                zyx(components[index][0]),
            ),
        )
        representative = min(
            remaining[component_id],
            key=lambda candidate: (
                -nearest_anchor_distance[component_id][candidate],
                *zyx(candidate),
            ),
        )
        priority.append(representative)
        priority_set.add(representative)
        anchors[component_id].append(representative)
        remaining[component_id].remove(representative)
        del nearest_anchor_distance[component_id][representative]
        for candidate in remaining[component_id]:
            nearest_anchor_distance[component_id][candidate] = min(
                nearest_anchor_distance[component_id][candidate],
                squared_distance(candidate, representative),
            )
        frontier_selected[component_id] += 1

    # The post-cap order has no edge authority, but remains canonical so its
    # digest accounts for every omitted source independent of input ordering.
    result = priority + [
        key
        for component in remaining
        for key in sorted(component, key=zyx)
        if key not in priority_set
    ]
    selected = result[:maximum]
    omitted = result[maximum:]
    return selected, {
        "schema": "q2-atlas-pmove-source-accounting-v1",
        "selection_rule": (
            "spawn-ladder-component-frontier-proportional-farthest-v3"
        ),
        "maximum": maximum,
        "total": len(result),
        "selected": len(selected),
        "omitted": len(omitted),
        "selected_sources_sha256": sha256_bytes(canonical_json([list(key) for key in selected])),
        "omitted_sources_sha256": sha256_bytes(canonical_json([list(key) for key in omitted])),
        "omitted_sources_emit_pmove_edges": False,
    }


def _movement_edge_stance(
    source: NavNode, target: NavNode, replayed_stance: str,
) -> str | None:
    """Admit only the stance that the exact Pmove request replayed."""

    if (
        replayed_stance == "standing"
        and source.standing_clear and target.standing_clear
    ):
        return "standing"
    if (
        replayed_stance == "crouched"
        and source.crouched_clear and target.crouched_clear
    ):
        return "crouched"
    return None


def _movement_edge_kind(
    mode: str, *, any_airborne: bool, vertical: float,
) -> str | None:
    """Type only the movement transition directly evidenced by Pmove."""

    if mode == "jump":
        return "jump" if any_airborne else None
    if mode == "ladder":
        return "ladder" if any_airborne and vertical > 18.0 else None
    if mode != "ground":
        return None
    if any_airborne:
        return "controlled_drop"
    if 0.5 < abs(vertical) <= 18.0:
        return "step"
    return None


def _movement_edge_cost_q8(
    request: Mapping[str, Any], response: Mapping[str, Any],
    *, landing_command_index: int | None,
) -> int:
    """Encode the exact replay path length through its admitted endpoint."""

    start = request.get("origin")
    frames = response.get("frames")
    if not isinstance(start, list) or len(start) != 3 or not isinstance(frames, list):
        raise AtlasAnalysisError("Pmove edge cost lacks exact trajectory points")
    points: list[Sequence[float]] = [start]
    landing_found = landing_command_index is None
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise AtlasAnalysisError("Pmove edge cost frame is invalid")
        point = frame.get("origin")
        if not isinstance(point, list) or len(point) != 3:
            raise AtlasAnalysisError("Pmove edge cost frame lacks origin")
        points.append(point)
        if landing_command_index is not None and (
            frame.get("command_index") == landing_command_index
        ):
            landing_found = True
            break
    if not landing_found:
        raise AtlasAnalysisError("Pmove edge cost lacks exact landing frame")
    distance = sum(
        math.dist(left, right) for left, right in zip(points, points[1:])
    )
    return min(max(round(distance * 256.0), 4096), 0xFFFFFFFF)


def _movement_requests_for_source(
    key: tuple[int, int, int], source: NavNode, *, outgoing: int,
    is_spawn: bool, parameters: Mapping[str, Any],
) -> list[MovementRecord]:
    """Build the canonical complete stance/mode request set for one source."""

    records: list[MovementRecord] = []
    for stance in ("standing", "crouched"):
        if stance == "standing" and not source.standing_clear:
            continue
        if stance == "crouched" and not source.crouched_clear:
            continue
        for yaw in (0, 90, 180, 270):
            suffix = "" if stance == "standing" else ":crouched"
            command = {
                "msec": 50, "angles": [0, yaw, 0], "forwardmove": 300,
            }
            if stance == "crouched":
                command["upmove"] = -300
            walk_request = {
                "id": (
                    f"drop:{key[0]}:{key[1]}:{key[2]}:"
                    f"ground:{yaw}{suffix}:pmove"
                ),
                "op": "simulate", "origin": list(source.position),
                "velocity": [0.0, 0.0, 0.0], "pm_type": 0,
                "pm_flags": 4 if stance == "standing" else 5,
                "pm_time": 0, "gravity": parameters["gravity"],
                "airaccelerate": parameters["airaccelerate"],
                "delta_angles_short": [0, 0, 0], "snapinitial": False,
                "commands": [
                    dict(command) for _ in range(PMOVE_GROUND_HORIZON_FRAMES)
                ],
            }
            records.append((key, "ground", yaw, stance, walk_request))
            if stance == "standing" and (outgoing < 2 or is_spawn):
                jump_request = {
                    **walk_request,
                    "id": (
                        f"drop:{key[0]}:{key[1]}:{key[2]}:"
                        f"jump:{yaw}:pmove"
                    ),
                    "commands": [
                        {
                            "msec": 50, "angles": [0, yaw, 0],
                            "forwardmove": 300, "upmove": 300,
                        },
                        *[
                            {
                                "msec": 50, "angles": [0, yaw, 0],
                                "forwardmove": 300,
                            }
                            for _ in range(PMOVE_JUMP_HORIZON_FRAMES - 1)
                        ],
                    ],
                }
                records.append((key, "jump", yaw, stance, jump_request))
    return records


def _ladder_request_candidates_for_source(
    key: tuple[int, int, int], source: NavNode,
    *, parameters: Mapping[str, Any],
) -> list[MovementRecord]:
    """Build complete bounded Pmove probes from an exact ladder-content source.

    Point contents makes the source mandatory but does not prove contact. The
    engine contact law is replayed at every exact frame-start hull later,
    through the first safe landing; approaching a ladder can establish contact
    even when the source hull itself does not touch it.
    """

    if not source.contents & CONTENTS_LADDER:
        return []
    candidates: list[MovementRecord] = []
    for stance in ("standing", "crouched"):
        if stance == "standing" and not source.standing_clear:
            continue
        if stance == "crouched" and not source.crouched_clear:
            continue
        for yaw in (0, 90, 180, 270):
            suffix = "" if stance == "standing" else ":crouched"
            identifier = (
                f"drop:{key[0]}:{key[1]}:{key[2]}:"
                f"ladder:{yaw}{suffix}"
            )
            command: dict[str, Any] = {
                "msec": 50, "angles": [-30, yaw, 0], "forwardmove": 300,
            }
            request = {
                "id": f"{identifier}:pmove",
                "op": "simulate", "origin": list(source.position),
                "velocity": [0.0, 0.0, 0.0], "pm_type": 0,
                "pm_flags": 4 if stance == "standing" else 5,
                "pm_time": 0, "gravity": parameters["gravity"],
                "airaccelerate": parameters["airaccelerate"],
                "delta_angles_short": [0, 0, 0], "snapinitial": False,
                "commands": [
                    dict(command) for _ in range(PMOVE_LADDER_CLIMB_FRAMES)
                ],
            }
            candidates.append((key, "ladder", yaw, stance, request))
    return candidates


def _pmove_initial_origin(request: Mapping[str, Any]) -> list[float]:
    """Mirror q2-pmove-oracle's lroundf world-to-1/8-unit conversion."""

    origin = request.get("origin")
    if not isinstance(origin, list) or len(origin) != 3:
        raise AtlasAnalysisError("ladder Pmove request lacks an initial origin")
    snapped: list[float] = []
    for value in origin:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise AtlasAnalysisError("ladder Pmove initial origin is not finite")
        scaled = float(value) * 8.0
        fixed = (
            math.floor(scaled + 0.5)
            if scaled >= 0 else math.ceil(scaled - 0.5)
        )
        if not -32768 <= fixed <= 32767:
            raise AtlasAnalysisError("ladder Pmove initial origin exceeds fixed range")
        snapped.append(fixed * 0.125)
    return snapped


def _ladder_contact_requests(
    record: MovementRecord, response: Mapping[str, Any],
    *, through_command_index: int,
) -> list[dict[str, Any]]:
    """Replay PM_CheckSpecialMovement at exact trajectory frame starts.

    Pmove runs PM_CheckDuck before its one-unit flat-forward ladder trace. The
    current output frame therefore supplies the exact hull used for that
    command, while the prior output frame supplies its exact fixed-point start
    origin. Command zero starts at the oracle's exact 1/8-unit input snap.
    """

    _, mode, yaw, _, request = record
    if mode != "ladder":
        raise AtlasAnalysisError("non-ladder trajectory requested ladder contacts")
    commands = request.get("commands")
    frames = response.get("frames")
    if (
        not isinstance(commands, list) or not isinstance(frames, list)
        or len(commands) != len(frames)
    ):
        raise AtlasAnalysisError("ladder trajectory frame accounting differs")
    if not 0 <= through_command_index < len(frames):
        raise AtlasAnalysisError("ladder contact landing index is out of range")
    identifier = request.get("id")
    if not isinstance(identifier, str) or not identifier.endswith(":pmove"):
        raise AtlasAnalysisError("ladder Pmove request ID is not canonical")
    radians = math.radians(yaw)
    direction = (round(math.cos(radians)), round(math.sin(radians)))
    initial = _pmove_initial_origin(request)
    requests: list[dict[str, Any]] = []
    for command_index in range(through_command_index + 1):
        frame = frames[command_index]
        if not isinstance(frame, Mapping):
            raise AtlasAnalysisError("ladder trajectory frame is invalid")
        start = initial if command_index == 0 else frames[command_index - 1].get("origin")
        mins = frame.get("mins")
        maxs = frame.get("maxs")
        if not all(
            isinstance(value, list) and len(value) == 3
            and all(type(axis) in (int, float) and math.isfinite(float(axis)) for axis in value)
            for value in (start, mins, maxs)
        ):
            raise AtlasAnalysisError("ladder frame-start hull is not finite")
        end = [
            float(start[0]) + direction[0],
            float(start[1]) + direction[1],
            float(start[2]),
        ]
        requests.append(_box_request(
            f"{identifier.removesuffix(':pmove')}:contact:{command_index}",
            start, end, mins, maxs,
        ))
    return requests


def _ladder_contact_trace_admits(response: Mapping[str, Any]) -> bool:
    """Apply the exact PM_CheckSpecialMovement fraction/contents predicate."""

    fraction = response.get("fraction")
    contents = response.get("contents")
    return bool(
        type(fraction) in (int, float)
        and math.isfinite(float(fraction))
        and float(fraction) < 1.0
        and isinstance(contents, int)
        and not isinstance(contents, bool)
        and contents & CONTENTS_LADDER
    )


def _drop_settle_request(
    request: Mapping[str, Any], response: Mapping[str, Any],
    *, horizon_frames: int = PMOVE_DROP_HORIZON_FRAMES,
) -> dict[str, Any] | None:
    """Extend an airborne edge exit with neutral Pmove coast frames.

    The first simulation remains the bounded ground or jump probe. Only a
    trajectory that became airborne without
    a false-to-true grounded transition is replayed from its original state
    under the same canonical ID, with neutral commands appended.  The exact
    oracle therefore returns one continuous trajectory; frames are never
    stitched or ballistically extrapolated.
    """

    frames = response.get("frames")
    commands = request.get("commands")
    if not isinstance(frames, list) or not isinstance(commands, list):
        raise AtlasAnalysisError("Pmove trajectory lacks frames or commands")
    if len(frames) != len(commands) or not frames:
        raise AtlasAnalysisError("Pmove trajectory frame count differs from request")
    grounded: list[bool] = []
    for frame in frames:
        if not isinstance(frame, Mapping) or type(frame.get("grounded")) is not bool:
            raise AtlasAnalysisError("Pmove trajectory has invalid grounded state")
        grounded.append(frame["grounded"])
    if not any(not value for value in grounded):
        return None
    if any(not grounded[index - 1] and grounded[index] for index in range(1, len(grounded))):
        return None
    if horizon_frames <= len(commands):
        return None
    last = commands[-1]
    if not isinstance(last, Mapping):
        raise AtlasAnalysisError("Pmove trajectory command is invalid")
    msec = last.get("msec")
    if type(msec) is not int or msec <= 0:
        raise AtlasAnalysisError("Pmove trajectory command cadence is invalid")
    coast: dict[str, Any] = {"msec": msec}
    if "angles_short" in last:
        coast["angles_short"] = list(last["angles_short"])
    elif "angles" in last:
        coast["angles"] = list(last["angles"])
    else:
        raise AtlasAnalysisError("Pmove trajectory command lacks view angles")
    return {
        **dict(request),
        "commands": [
            *[dict(command) for command in commands],
            *[dict(coast) for _ in range(horizon_frames - len(commands))],
        ],
    }


def _exact_landing_key(
    record: Mapping[str, Any], origin: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    classification = record.get("classification")
    if not isinstance(classification, Mapping) or (
        classification.get("classification") != "Exact"
        or classification.get("lethal") is True
    ):
        return None
    landing = classification.get("landing")
    if not isinstance(landing, Mapping):
        raise AtlasAnalysisError("exact trajectory lacks first landing")
    return _grid_index(landing["origin"], origin, 16)


def _apply_stock_drop_hazards(
    hazards: dict[str, Any], summary: Mapping[str, Any],
) -> None:
    """Record exact lethal landings without inventing void/uncontained facts."""

    hazards["exact_lethal_candidates_omitted"] = int(summary["exact_lethal"])


def _apply_static_drop_hazards(
    nodes: Mapping[tuple[int, int, int], NavNode],
    classifications: Sequence[Mapping[str, Any]],
) -> int:
    """Materialize only exact lethal-drop source proximity into Atlas L1."""

    marked: set[tuple[int, int, int]] = set()
    for record in classifications:
        classification = record.get("classification")
        if not (
            isinstance(classification, Mapping)
            and classification.get("classification") == "Exact"
            and classification.get("lethal") is True
        ):
            continue
        source = record.get("source_l1")
        if (
            not isinstance(source, list)
            or len(source) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in source)
        ):
            raise AtlasAnalysisError("exact lethal-drop source L1 identity is invalid")
        key = tuple(source)
        node = nodes.get(key)
        if node is None:
            raise AtlasAnalysisError("exact lethal-drop source is not admitted Atlas L1")
        node.atlas_hazard_types |= 1 << 3
        node.atlas_hazard_severity = max(node.atlas_hazard_severity, 255)
        marked.add(key)
    return len(marked)


def _add_exact_platform_navigation(
    cm: OracleProcess,
    metadata: BspMetadata,
    nodes: dict[tuple[int, int, int], NavNode],
    edges: list[dict[str, Any]],
    edge_keys: set[tuple[Any, ...]],
    origin: tuple[int, int, int],
) -> None:
    """Add stateful ``func_plat`` boarding and endpoint edges.

    Platform motion is not model-0 Pmove collision.  The pure entity law
    supplies both vertical transforms; transformed CM must prove a supported,
    clear standing origin at each pose, and both model-0 and transformed CM
    must prove the short boarding corridor to a pre-existing static node.
    Every admitted connector and ride remains typed ``mover`` with the edict
    identity and reduced state confidence.  Target-disabled platforms are
    omitted until trigger activation semantics are modeled.
    """

    entities = {entity.index: entity for entity in metadata.entities}

    def transformed_request(
        identifier: str,
        start: Sequence[float],
        end: Sequence[float],
        mins: Sequence[float],
        maxs: Sequence[float],
        *, headnode: int,
        pose_origin: Sequence[float],
    ) -> dict[str, Any]:
        return {
            "id": identifier, "op": "transformed_box_trace",
            "start": list(start), "end": list(end),
            "mins": list(mins), "maxs": list(maxs),
            "headnode": headnode, "mask": MASK_PLAYERSOLID,
            "origin": list(pose_origin), "angles": [0.0, 0.0, 0.0],
        }

    def trace_clear(trace: Mapping[str, Any]) -> bool:
        return (
            trace.get("fraction") == 1
            and trace.get("startsolid") is False
            and trace.get("allsolid") is False
        )

    def add_edge(
        source: tuple[int, int, int], target: tuple[int, int, int],
        *, blocker: int, cost: int,
    ) -> None:
        key = (source, target, "mover", "standing")
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({
            "source": list(source), "target": list(target),
            "edge_type": "mover", "stance": "standing",
            "flags": 0, "blocker": blocker,
            "cost": min(max(cost, 4096), 0xFFFFFFFF),
            "risk": 0, "confidence": 32768,
            "evidence": EVIDENCE_CM_TRACE_V1,
            "validation_version": VALIDATION_VERSION,
            "auxiliary": 0xFFFFFFFF,
        })

    def supports_target(
        trace: Mapping[str, Any], target: Sequence[float],
    ) -> bool:
        normal = (trace.get("plane") or {}).get("normal")
        endpos = trace.get("endpos")
        return (
            trace.get("startsolid") is False
            and trace.get("allsolid") is False
            and type(trace.get("fraction")) in (int, float)
            and 0 <= trace["fraction"] <= 1
            and isinstance(normal, list) and len(normal) == 3 and normal[2] >= 0.7
            and isinstance(endpos, list) and len(endpos) == 3
            and all(abs(float(endpos[axis]) - target[axis]) <= 0.25 for axis in range(3))
        )

    def does_not_precede_target(
        trace: Mapping[str, Any], target: Sequence[float],
    ) -> bool:
        """Accept a second authority's support only when it lies below target.

        Model-0 and an inline platform are traced separately, while the game
        clips movement against their union. During a downward stair trace the
        higher collision wins. A model-0 floor below the exact platform top
        therefore cannot block boarding that top, but a collision above it
        must still reject the connector.
        """

        if trace_clear(trace) or supports_target(trace, target):
            return True
        normal = (trace.get("plane") or {}).get("normal")
        endpos = trace.get("endpos")
        return (
            trace.get("startsolid") is False
            and trace.get("allsolid") is False
            and type(trace.get("fraction")) in (int, float)
            and 0 <= trace["fraction"] <= 1
            and isinstance(normal, list) and len(normal) == 3 and normal[2] >= 0.7
            and isinstance(endpos, list) and len(endpos) == 3
            and float(endpos[2]) <= target[2] + 0.25
        )

    def connector_clear(
        source: NavNode, target: NavNode,
        *, entity_index: int, label: str, headnode: int,
        pose_origin: Sequence[float], direction: str,
    ) -> bool:
        """Validate same-level, step-up, or short safe-down boarding."""

        delta = target.position[2] - source.position[2]
        if abs(delta) > 18.0:
            return False

        def pair(
            suffix: str, start: Sequence[float], end: Sequence[float],
        ) -> list[dict[str, Any]]:
            return [
                _box_request(
                    f"platform-{suffix}-world:{entity_index}:{label}:{direction}",
                    start, end, STANDING_MINS, STANDING_MAXS,
                ),
                transformed_request(
                    f"platform-{suffix}-inline:{entity_index}:{label}:{direction}",
                    start, end, STANDING_MINS, STANDING_MAXS,
                    headnode=headnode, pose_origin=pose_origin,
                ),
            ]

        if abs(delta) <= 0.5:
            return all(trace_clear(trace) for trace in cm.call(pair(
                "board-level", source.position, target.position,
            )))
        if delta > 0:
            # Pmove's canonical stair branch tests the elevated origin, moves
            # forward at +18, then traces down exactly 18 units.  One of the
            # two collision authorities must supply the target support; the
            # other must remain clear to that same endpoint.
            up = (
                source.position[0], source.position[1], source.position[2] + 18.0,
            )
            forward = (target.position[0], target.position[1], up[2])
            down = (target.position[0], target.position[1], source.position[2])
            raised = cm.call(pair("board-step-raised", up, up))
            forward_traces = cm.call(pair("board-step-forward", up, forward))
            if not all(trace_clear(trace) for trace in (*raised, *forward_traces)):
                return False
            down_traces = cm.call(pair("board-step-down", forward, down))
            return (
                any(supports_target(trace, target.position) for trace in down_traces)
                and all(
                    does_not_precede_target(trace, target.position)
                    for trace in down_traces
                )
            )

        # Leaving a platform for a floor no more than STEPSIZE below first
        # moves horizontally at the supported source height, then obtains the
        # exact model-0/inline support with a bounded downward trace. This is a
        # typed mover connector, not an unconditional static walk edge.
        horizontal = (
            target.position[0], target.position[1], source.position[2],
        )
        horizontal_traces = cm.call(pair(
            "board-down-forward", source.position, horizontal,
        ))
        if not all(trace_clear(trace) for trace in horizontal_traces):
            return False
        down = (
            target.position[0], target.position[1], target.position[2] - 2.0,
        )
        down_traces = cm.call(pair("board-down-support", horizontal, down))
        return (
            any(supports_target(trace, target.position) for trace in down_traces)
            and all(
                trace_clear(trace) or supports_target(trace, target.position)
                for trace in down_traces
            )
        )

    # Boarding searches may only terminate at nodes that existed before any
    # platform endpoint was inserted.  One platform can never validate another
    # platform's state-dependent support by graph proximity alone.
    static_nodes = tuple(nodes.items())
    for record in metadata.entity_catalog.brush_submodels:
        entity = entities[int(record["entity_index"])]
        if entity.classname.casefold() != "func_plat":
            continue
        model_index = int(record["model_index"])
        model = metadata.models[model_index]
        cmodel = Aabb(
            tuple(value - 1.0 for value in model.mins),
            tuple(value + 1.0 for value in model.maxs),
        )  # type: ignore[arg-type]
        result = platform_mover_semantics(entity, cmodel.mins, cmodel.maxs)
        if not result.is_exact or result.value is None:
            continue
        platform = result.value
        if platform.target_disabled:
            continue
        # A standing hull must fit wholly over the authored top face.  Merely
        # touching a thin decorative brush is not a boardable lift.
        if (
            cmodel.maxs[0] - cmodel.mins[0] < 32.0
            or cmodel.maxs[1] - cmodel.mins[1] < 32.0
        ):
            continue
        # The endpoint is at the platform center, while a legal static
        # connector can sit beyond a corner by one standing-hull radius plus
        # one L1 cell of quantization.  Bound the search from authored brush
        # geometry; a fixed radius rejects valid large-platform boardings.
        boarding_radius_squared = sum(
            (
                (cmodel.maxs[axis] - cmodel.mins[axis]) * 0.5
                + STANDING_MAXS[axis] + 16.0
            ) ** 2
            for axis in (0, 1)
        )
        other_movers = _dynamic_mover_dependency_index(
            metadata, exclude_entity_index=entity.index,
        )
        pose_specs = (
            ("bottom", platform.pos2, platform.endpoint_pose.bounds),
            ("top", platform.pos1, platform.reference_pose.bounds),
        )
        proposed: list[
            tuple[str, tuple[float, float, float], tuple[float, float, float]]
        ] = []
        support_requests: list[dict[str, Any]] = []
        for label, pose_origin, bounds in pose_specs:
            nominal = (
                (bounds.mins[0] + bounds.maxs[0]) * 0.5,
                (bounds.mins[1] + bounds.maxs[1]) * 0.5,
                bounds.maxs[2] + 24.0,
            )
            proposed.append((label, pose_origin, nominal))
            support_requests.append(transformed_request(
                f"platform-support:{entity.index}:{label}",
                (nominal[0], nominal[1], nominal[2] + 8.0),
                (nominal[0], nominal[1], nominal[2] - 16.0),
                STANDING_MINS, STANDING_MAXS,
                headnode=model.headnode, pose_origin=pose_origin,
            ))
        supports = cm.call(support_requests)
        endpoint_records: list[tuple[
            str, tuple[float, float, float], tuple[int, int, int], NavNode,
        ]] = []
        for (label, pose_origin, _), support in zip(proposed, supports):
            normal = (support.get("plane") or {}).get("normal")
            position = support.get("endpos")
            if (
                support.get("startsolid") is not False
                or support.get("allsolid") is not False
                or type(support.get("fraction")) not in (int, float)
                or not 0 <= support["fraction"] < 1
                or not isinstance(normal, list) or len(normal) != 3
                or normal[2] < 0.7
                or not isinstance(position, list) or len(position) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in position)
            ):
                continue
            exact_position = tuple(float(value) for value in position)
            clear_requests = [
                _box_request(
                    f"platform-world-stand:{entity.index}:{label}",
                    exact_position, exact_position, STANDING_MINS, STANDING_MAXS,
                ),
                _box_request(
                    f"platform-world-crouch:{entity.index}:{label}",
                    exact_position, exact_position, CROUCHED_MINS, CROUCHED_MAXS,
                ),
                transformed_request(
                    f"platform-inline-stand:{entity.index}:{label}",
                    exact_position, exact_position, STANDING_MINS, STANDING_MAXS,
                    headnode=model.headnode, pose_origin=pose_origin,
                ),
                transformed_request(
                    f"platform-inline-crouch:{entity.index}:{label}",
                    exact_position, exact_position, CROUCHED_MINS, CROUCHED_MAXS,
                    headnode=model.headnode, pose_origin=pose_origin,
                ),
                {"id": f"platform-contents:{entity.index}:{label}",
                 "op": "point_contents", "point": list(exact_position)},
            ]
            stand, crouch, inline_stand, inline_crouch, contents = cm.call(clear_requests)
            if not trace_clear(stand) or not trace_clear(inline_stand):
                continue
            crouched_clear = trace_clear(crouch) and trace_clear(inline_crouch)
            key = _grid_index(exact_position, origin, 16)
            endpoint = NavNode(
                key, exact_position, True, crouched_clear, True,
                int(contents["contents"]), tuple(float(value) for value in normal),
                floor_surface_flags=(support.get("surface") or {}).get("flags", 0),
                floor_surface_name=(support.get("surface") or {}).get("name", ""),
            )
            existing = nodes.get(key)
            if existing is None:
                nodes[key] = endpoint
            elif not existing.standing_clear:
                continue
            # Keep the exact transformed-CM endpoint even when its L1 index
            # aliases an existing model-0 node. Connector replay must never
            # substitute that static node's different continuous origin.
            endpoint_records.append((label, pose_origin, key, endpoint))

        if len(endpoint_records) != 2:
            # Remove an endpoint inserted for a platform whose other pose did
            # not prove. Existing static nodes are never removed.
            static_keys = {key for key, _ in static_nodes}
            for _, _, key, _ in endpoint_records:
                if key not in static_keys:
                    nodes.pop(key, None)
            continue

        connector_by_label: dict[str, tuple[int, int, int]] = {}
        for label, pose_origin, endpoint_key, endpoint in endpoint_records:
            candidates = [
                (key, node) for key, node in static_nodes
                if key != endpoint_key and node.standing_clear
                and abs(node.position[2] - endpoint.position[2]) <= 18.0
                and (
                    (node.position[0] - endpoint.position[0]) ** 2
                    + (node.position[1] - endpoint.position[1]) ** 2
                ) <= boarding_radius_squared
            ]
            candidates.sort(key=lambda item: (
                (item[1].position[0] - endpoint.position[0]) ** 2
                + (item[1].position[1] - endpoint.position[1]) ** 2,
                item[0][2], item[0][1], item[0][0],
            ))
            selected = None
            for candidate_key, candidate in candidates:
                if other_movers.intersects_box_path(
                    (endpoint.position, candidate.position),
                    STANDING_MINS, STANDING_MAXS,
                ):
                    continue
                if (
                    connector_clear(
                        endpoint, candidate, entity_index=entity.index,
                        label=label, headnode=model.headnode,
                        pose_origin=pose_origin, direction="out",
                    )
                    and connector_clear(
                        candidate, endpoint, entity_index=entity.index,
                        label=label, headnode=model.headnode,
                        pose_origin=pose_origin, direction="in",
                    )
                ):
                    selected = candidate_key
                    break
            if selected is not None:
                connector_by_label[label] = selected

        if set(connector_by_label) != {"bottom", "top"}:
            static_keys = {key for key, _ in static_nodes}
            for _, _, key, _ in endpoint_records:
                if key not in static_keys:
                    nodes.pop(key, None)
            continue

        endpoints = {label: key for label, _, key, _ in endpoint_records}
        travel_cost = round(abs(platform.pos1[2] - platform.pos2[2]) * 256.0)
        for label in ("bottom", "top"):
            endpoint_key = endpoints[label]
            connector = connector_by_label[label]
            add_edge(endpoint_key, connector, blocker=entity.index, cost=4096)
            add_edge(connector, endpoint_key, blocker=entity.index, cost=4096)
        add_edge(
            endpoints["bottom"], endpoints["top"],
            blocker=entity.index, cost=travel_cost,
        )
        add_edge(
            endpoints["top"], endpoints["bottom"],
            blocker=entity.index, cost=travel_cost,
        )


def _build_navigation(
    cm: OracleProcess,
    pmove: OracleProcess | None,
    fall: FallOracleProcess | None,
    bsp: Path,
    spawns: Sequence[tuple[int, tuple[float, float, float]]],
    origin: tuple[int, int, int],
    limits: AnalyzerLimits,
    candidate_points: Sequence[tuple[float, float, float]] = (),
    mover_dependencies: _MoverDependencyIndex | None = None,
    movement_accounting: dict[str, Any] | None = None,
    metadata: BspMetadata | None = None,
) -> tuple[
    dict[tuple[int, int, int], NavNode],
    list[dict],
    dict[int, tuple[int, int, int]],
    list[dict[str, Any]],
]:
    if mover_dependencies is None:
        raise AtlasAnalysisError("Pmove navigation lacks dynamic-mover authority")
    seeds = [
        (entity_ordinal, point, True) for entity_ordinal, point in spawns
    ] + [
        (None, point, False) for point in candidate_points
    ]
    grounded, seed_support = _ground_navigation_seeds(cm, seeds)
    supported_points = [point for point in grounded if point is not None]
    probes = cm.call(_node_probe_requests(supported_points))
    nodes: dict[tuple[int, int, int], NavNode] = {}
    queue: deque[tuple[int, int, int]] = deque()
    spawn_indices: dict[int, tuple[int, int, int]] = {}
    probe_ordinal = 0
    for ordinal, ((entity_ordinal, _, is_spawn), point) in enumerate(zip(seeds, grounded)):
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
        if is_spawn and entity_ordinal is not None:
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
        floor_results = cm.call(_candidate_floor_requests(candidates))
        accepted: list[tuple[tuple[int, int, int], tuple[float, float, float], tuple[int, int, int], dict]] = []
        for ordinal, (source_key, (_, _)) in enumerate(zip(sources, candidates)):
            selected = next((
                (candidate, floor)
                for floor in floor_results[ordinal * 2:ordinal * 2 + 2]
                if (candidate := _supported_floor_candidate(
                    source_key, floor, origin,
                )) is not None
            ), None)
            if selected is None:
                continue
            candidate, floor = selected
            point, key = candidate
            # The bounded floor trace may find a supported player-origin cell
            # more than one L1 band above or below its neighbor. Retain that
            # cell as a movement candidate: node materialization is not edge
            # authority. The vertical step/jump/drop is admitted only after
            # exact Pmove replay below. Discarding it here makes a real landing
            # absent from the graph before Pmove can prove the traversal.
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

    if metadata is not None:
        _add_exact_platform_navigation(
            cm, metadata, nodes, edges, edge_keys, origin,
        )
        if len(nodes) > limits.max_l1_nodes:
            raise AtlasAnalysisError("platform endpoints exceeded L1 node budget")
        if len(edges) > limits.max_l1_edges:
            raise AtlasAnalysisError("platform traversal exceeded L1 edge budget")

    # Exact Pmove validates step/drop/jump traversal from a bounded,
    # deterministic set of spawn and CM-boundary nodes, plus every exact
    # ladder-content source. Atlas v1 omits all unprobed movement candidates;
    # it never extrapolates a usercmd result.
    drop_classifications: list[dict[str, Any]] = []
    if pmove is not None:
        if fall is None:
            raise AtlasAnalysisError(
                "Pmove traversal lacks mandatory exact fall authority"
            )
        outgoing = defaultdict(int)
        for edge in edges:
            outgoing[tuple(edge["source"])] += 1
        candidates, source_accounting = _complete_pmove_source_set(
            nodes, edges, spawn_indices, limits.max_pmove_sources,
        )
        if movement_accounting is not None:
            movement_accounting.update(source_accounting)
        pmove_edge_start = len(edges)
        trajectory_counts: dict[str, int] = defaultdict(int)
        trajectory_outcomes: dict[str, int] = defaultdict(int)
        trajectory_batches = 0
        settle_replays = 0
        spawn_keys = set(spawn_indices.values())
        ladder_candidates = [
            candidate
            for key in candidates
            for candidate in _ladder_request_candidates_for_source(
                key, nodes[key], parameters=pmove.identity["parameters"],
            )
        ]
        # Reserve the full prospective trajectory authority workload before
        # the first candidate simulation write. Both legal source stances are
        # replayed: preferring standing at a dual-clear node would make a real
        # crouch-only exit invisible. Every ladder replay may require one CM
        # contact trace at each of its 52 frame starts, so that complete worst
        # case is reserved before Pmove. No cap may yield a prefix Atlas.
        ordinary_worst_case = sum(
            4 * (
                int(nodes[key].standing_clear)
                + int(nodes[key].crouched_clear)
                + int(
                    nodes[key].standing_clear
                    and (outgoing[key] < 2 or key in spawn_keys)
                )
            )
            for key in candidates
        )
        worst_case = ordinary_worst_case + len(ladder_candidates)
        maximum_ladder_contact_traces = (
            len(ladder_candidates) * PMOVE_LADDER_SETTLE_HORIZON_FRAMES
        )
        if (
            cm.requests + maximum_ladder_contact_traces
            > cm.limits.max_oracle_requests
        ):
            raise AtlasAnalysisError(
                "ladder contact preflight exceeds complete collision oracle budget"
            )
        if pmove.requests + 3 * worst_case > limits.max_oracle_requests:
            raise AtlasAnalysisError(
                "Pmove trajectory preflight exceeds complete oracle budget"
            )
        if fall.requests + 2 * worst_case > fall.max_requests:
            raise AtlasAnalysisError(
                "fall trajectory preflight exceeds complete oracle budget"
            )

        def trajectory_identifier(request: Mapping[str, Any]) -> str:
            identifier = request.get("id")
            if not isinstance(identifier, str) or not identifier.endswith(":pmove"):
                raise AtlasAnalysisError("Pmove traversal ID is not canonical")
            return identifier.removesuffix(":pmove")

        ladder_candidate_ids = [
            trajectory_identifier(record[4]) for record in ladder_candidates
        ]
        ladder_contact_trace_ids: list[str] = []
        ladder_contact_admitted_ids: list[str] = []
        ladder_contact_admitted: set[str] = set()
        ladder_contact_eligible = 0
        ladder_contact_trace_count = 0

        def process_movement_batch(records: list[MovementRecord]) -> None:
            """Consume one bounded response window and release its frames."""

            nonlocal ladder_contact_eligible, ladder_contact_trace_count
            nonlocal settle_replays, trajectory_batches
            trajectory_batches += 1
            pmove_results = pmove.call([record[4] for record in records])
            settle_indices: list[int] = []
            settle_requests: list[dict[str, Any]] = []
            for index, ((_, mode, _, _, request), result) in enumerate(zip(
                records, pmove_results
            )):
                horizon = (
                    PMOVE_LADDER_SETTLE_HORIZON_FRAMES
                    if mode == "ladder" else PMOVE_DROP_HORIZON_FRAMES
                )
                extended = _drop_settle_request(
                    request, result, horizon_frames=horizon,
                )
                if extended is not None:
                    settle_indices.append(index)
                    settle_requests.append(extended)
            settle_replays += len(settle_requests)
            for index, request, result in zip(
                settle_indices, settle_requests, pmove.call(settle_requests)
            ):
                source_key, mode, yaw, stance, _ = records[index]
                records[index] = (source_key, mode, yaw, stance, request)
                pmove_results[index] = result

            exact_candidates: list[DropTrajectory] = []
            for (source_key, mode, yaw, _, request), result in zip(
                records, pmove_results
            ):
                if any(not frame["grounded"] for frame in result["frames"]):
                    radians = math.radians(yaw)
                    exact_candidates.append(DropTrajectory(
                        identifier=trajectory_identifier(request),
                        source_l1=source_key,
                        direction=(
                            round(math.cos(radians)), round(math.sin(radians))
                        ),
                        mode=mode,
                        request=request,
                        response=result,
                        dynamic_movers=mover_dependencies.intersects_trajectory(
                            result, request
                        ),
                    ))
            try:
                classified = classify_drop_trajectories(
                    exact_candidates, bsp=bsp, pmove_process=pmove,
                    fall_process=fall,
                )
            except ExactDropAnalysisError as error:
                raise AtlasAnalysisError(str(error)) from error
            drop_classifications.extend(classified)
            drop_by_id = {item["id"]: item for item in classified}

            # A source point inside a ladder-content volume may need approach
            # motion before the player hull touches the ladder brush. Mirror
            # PM_CheckSpecialMovement at every exact trajectory frame start
            # through the first safe landing, then retain a per-trajectory
            # contact decision for edge admission below.
            contact_requests: list[dict[str, Any]] = []
            contact_ranges: list[tuple[str, int, int]] = []
            for record, result in zip(records, pmove_results):
                _, mode, _, _, request = record
                if mode != "ladder" or mover_dependencies.intersects_trajectory(
                    result, request
                ):
                    continue
                if not any(not frame["grounded"] for frame in result["frames"]):
                    continue
                identifier = trajectory_identifier(request)
                drop_record = drop_by_id.get(identifier)
                if drop_record is None:
                    raise AtlasAnalysisError(
                        "airborne ladder trajectory lacks complete accounting"
                    )
                classification = drop_record["classification"]
                if (
                    classification.get("classification") != "Exact"
                    or classification.get("lethal") is True
                ):
                    continue
                landing_command_index = classification["landing"][
                    "command_index"
                ]
                requests = _ladder_contact_requests(
                    record, result,
                    through_command_index=landing_command_index,
                )
                start = len(contact_requests)
                contact_requests.extend(requests)
                contact_ranges.append((identifier, start, len(contact_requests)))
                ladder_contact_eligible += 1
                ladder_contact_trace_count += len(requests)
                ladder_contact_trace_ids.extend(
                    str(request["id"]) for request in requests
                )
            contact_results = cm.call(contact_requests)
            if len(contact_results) != len(contact_requests):
                raise AtlasAnalysisError(
                    "ladder contact response accounting differs"
                )
            for identifier, start, end in contact_ranges:
                if any(
                    _ladder_contact_trace_admits(response)
                    for response in contact_results[start:end]
                ):
                    ladder_contact_admitted.add(identifier)
                    ladder_contact_admitted_ids.append(identifier)

            for (source_key, mode, _, source_stance, request), result in zip(
                records, pmove_results
            ):
                any_airborne = any(
                    not frame["grounded"] for frame in result["frames"]
                )
                drop_record = None
                risk = 0
                evidence = EVIDENCE_PMOVE_V1
                validation_version = VALIDATION_VERSION
                landing_command_index = None
                if mover_dependencies.intersects_trajectory(result, request):
                    # Airborne cases remain explicit Unknown classifications;
                    # mover-dependent traversal never emits a static edge.
                    trajectory_outcomes["omitted_dynamic_mover"] += 1
                    continue
                if any_airborne:
                    drop_record = drop_by_id.get(trajectory_identifier(request))
                    if drop_record is None:
                        raise AtlasAnalysisError(
                            "airborne trajectory lacks complete accounting"
                        )
                    classification = drop_record["classification"]
                    if (
                        classification.get("classification") != "Exact"
                        or classification.get("lethal") is True
                    ):
                        outcome = (
                            "omitted_exact_lethal"
                            if classification.get("lethal") is True
                            else "omitted_"
                            + str(classification.get("reason", "unknown"))
                        )
                        trajectory_outcomes[outcome] += 1
                        continue
                    target_key = _exact_landing_key(drop_record, origin)
                    if target_key is None:
                        trajectory_outcomes["omitted_missing_landing"] += 1
                        continue
                    landing_command_index = classification["landing"][
                        "command_index"
                    ]
                    risk = {
                        "none": 0, "footstep": 0, "short": 8192,
                        "fall": 32768, "far": 65535,
                    }.get(classification.get("severity"), 65535)
                    evidence = drop_record["evidence"]
                    validation_version = drop_record["validation_version"]
                    if (
                        mode == "ladder"
                        and trajectory_identifier(request)
                        not in ladder_contact_admitted
                    ):
                        trajectory_outcomes["omitted_no_ladder_contact"] += 1
                        continue
                else:
                    if mode != "ground" or not result["final"]["grounded"]:
                        trajectory_outcomes["omitted_untyped_motion"] += 1
                        continue
                    target_key = _grid_index(
                        result["final"]["origin"], origin, 16,
                    )

                target = nodes.get(target_key)
                if target is None:
                    trajectory_outcomes["omitted_missing_target_node"] += 1
                    continue
                if target_key == source_key:
                    trajectory_outcomes["omitted_same_cell"] += 1
                    continue
                vertical = target.position[2] - nodes[source_key].position[2]
                kind = _movement_edge_kind(
                    mode, any_airborne=any_airborne, vertical=vertical,
                )
                if kind is None:
                    # Flat grounded motion is already represented by local
                    # MASK_PLAYERSOLID walk edges. A Pmove endpoint may be
                    # several L1 cells away and must not masquerade as a
                    # fixed-cost step that bypasses the intervening graph.
                    trajectory_outcomes["omitted_flat_or_untyped_ground"] += 1
                    continue
                if kind == "ladder":
                    # Contact was admitted before this replay. The exact safe
                    # first landing contributes Pmove|Fall, so the typed edge
                    # carries the complete CM|Pmove|Fall authority closure.
                    evidence |= EVIDENCE_CM_TRACE_V1
                stance = _movement_edge_stance(
                    nodes[source_key], target, source_stance,
                )
                if stance is None:
                    trajectory_outcomes["omitted_stance_mismatch"] += 1
                    continue
                edge_key = (source_key, target_key, kind, stance)
                if edge_key in edge_keys:
                    trajectory_outcomes["omitted_duplicate_edge"] += 1
                    continue
                edge_keys.add(edge_key)
                edges.append({
                    "source": list(source_key), "target": list(target_key),
                    "edge_type": kind, "stance": stance, "flags": 0,
                    "blocker": 0,
                    "cost": _movement_edge_cost_q8(
                        request, result,
                        landing_command_index=landing_command_index,
                    ),
                    "risk": risk,
                    "confidence": 65535, "evidence": evidence,
                    "validation_version": validation_version,
                    "auxiliary": 0xFFFFFFFF,
                })
                if len(edges) > limits.max_l1_edges:
                    raise AtlasAnalysisError(
                        "Pmove traversal exceeded L1 edge budget"
                    )
                trajectory_outcomes[f"emitted_{stance}_{kind}"] += 1

        movement_batch: list[MovementRecord] = []

        def enqueue(record: MovementRecord) -> None:
            trajectory_counts[f"{record[3]}_{record[1]}"] += 1
            movement_batch.append(record)
            if len(movement_batch) == PMOVE_TRAJECTORY_BATCH_SIZE:
                process_movement_batch(movement_batch)
                movement_batch.clear()

        for key in candidates:
            for record in _movement_requests_for_source(
                key, nodes[key], outgoing=outgoing[key],
                is_spawn=key in spawn_keys,
                parameters=pmove.identity["parameters"],
            ):
                enqueue(record)
        for record in ladder_candidates:
            enqueue(record)
        if movement_batch:
            process_movement_batch(movement_batch)
        if movement_accounting is not None:
            emitted = edges[pmove_edge_start:]
            emitted_counts: dict[str, int] = defaultdict(int)
            for edge in emitted:
                emitted_counts[
                    f"{edge['stance']}_{edge['edge_type']}"
                ] += 1
            requested_total = sum(trajectory_counts.values())
            movement_accounting["ladder_contact_accounting"] = {
                "schema": "q2-atlas-ladder-contact-accounting-v1",
                "source_nodes": len({record[0] for record in ladder_candidates}),
                "candidate_trajectories": len(ladder_candidates),
                "maximum_prospective_traces": maximum_ladder_contact_traces,
                "safe_landing_trajectories_challenged": ladder_contact_eligible,
                "candidate_traces": ladder_contact_trace_count,
                "admitted_contact_trajectories": len(
                    ladder_contact_admitted_ids
                ),
                "rejected_no_ladder_contact": (
                    ladder_contact_eligible - len(ladder_contact_admitted_ids)
                ),
                "unchallenged_without_exact_safe_landing": (
                    len(ladder_candidates) - ladder_contact_eligible
                ),
                "candidate_ids_sha256": sha256_bytes(
                    canonical_json(ladder_candidate_ids)
                ),
                "trace_ids_sha256": sha256_bytes(
                    canonical_json(ladder_contact_trace_ids)
                ),
                "admitted_ids_sha256": sha256_bytes(
                    canonical_json(ladder_contact_admitted_ids)
                ),
                "unadmitted_trajectories_emit_ladder_edges": False,
            }
            movement_accounting["trajectory_accounting"] = {
                "schema": "q2-atlas-pmove-trajectory-accounting-v1",
                "batch_size": PMOVE_TRAJECTORY_BATCH_SIZE,
                "batch_count": trajectory_batches,
                "ground_horizon_frames": PMOVE_GROUND_HORIZON_FRAMES,
                "jump_horizon_frames": PMOVE_JUMP_HORIZON_FRAMES,
                "settle_horizon_frames": PMOVE_DROP_HORIZON_FRAMES,
                "ladder_climb_frames": PMOVE_LADDER_CLIMB_FRAMES,
                "ladder_settle_horizon_frames": (
                    PMOVE_LADDER_SETTLE_HORIZON_FRAMES
                ),
                "requested": dict(sorted(trajectory_counts.items())),
                "requested_total": requested_total,
                "settle_replays": settle_replays,
                "airborne_classified": len(drop_classifications),
                "emitted": dict(sorted(emitted_counts.items())),
                "emitted_total": len(emitted),
                "outcomes": dict(sorted(trajectory_outcomes.items())),
                "requests_without_new_edge": requested_total - len(emitted),
            }
            if sum(trajectory_outcomes.values()) != requested_total:
                raise AtlasAnalysisError(
                    "Pmove trajectory outcome accounting differs from requests"
                )

    # Unknown/no-landing trajectories remain omitted and never invent void.
    _apply_static_drop_hazards(nodes, drop_classifications)

    _assign_directed_regions(nodes, edges)
    mutual = _mutually_reachable_spawn_pairs(edges, spawn_indices)
    if not mutual:
        raise AtlasAnalysisError("fewer than two mutually reachable deathmatch spawns")
    return nodes, edges, spawn_indices, drop_classifications


def _admissions(
    bsp_sha256: str,
    provenance_sha256: str,
    cm: OracleProcess,
    pmove: OracleProcess | None,
    fall: FallOracleProcess,
    hook: HookOracleProcess | None = None,
    hook_attestation: Path | None = None,
    b1_runtime_authority_seal: Mapping[str, Any] | None = None,
) -> dict:
    def tool(binary: Path, identity: dict) -> dict:
        names = {
            "q2-cm-oracle-v1": "q2-cm-oracle",
            "q2-pmove-oracle-v1": "q2-pmove-oracle",
            "q2-fall-oracle-v1": "q2-fall-oracle",
            "q2-hook-oracle-v1": "q2-hook-oracle",
        }
        return {
            "name": names[identity["schema"]],
            "schema": identity["schema"], "version": 1,
            "executable_sha256": sha256_file(binary),
            "tool_identity_sha256": identity["tool_identity"],
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
    if b1_runtime_authority_seal is None:
        raise AtlasAnalysisError("oracle admissions lack B1 runtime authority seal")
    output: dict[str, Any] = {
        "b1_runtime_authority_seal": dict(b1_runtime_authority_seal),
        "collision_oracle": collision,
    }
    fall_parameters = fall.identity["parameters"]
    fall_record = {
        "tool": tool(fall.binary, fall.identity),
        "parameters": {
            "fall_damagemod_f32_bits": struct.unpack(
                "<I", struct.pack("<f", fall_parameters["fall_damagemod"])
            )[0],
            "deathmatch": fall_parameters["deathmatch"],
            "dmflags": fall_parameters["dmflags"],
            "constants": fall.identity["constants"],
        },
        "source": dict(fall.identity["source"]),
    }
    fall_record["contract_sha256"] = sha256_bytes(_rust_struct_json(fall_record))
    output["fall_oracle"] = fall_record
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
    if hook is not None:
        if pmove is None or hook_attestation is None:
            raise AtlasAnalysisError("hook admission lacks Pmove or parity attestation")
        attestation = json.loads(hook_attestation.read_bytes())
        identity = hook.identity
        parameters = identity["parameters"]
        source = identity["source"]
        fixture = attestation["fixture"]
        binaries = attestation["binaries"]
        evidence = attestation["evidence"]
        identities = attestation["identities"]
        parity = {
            "name": "q2-hook-q2ded-parity",
            "schema": "q2-hook-parity-v1",
            "version": 1,
            "passed": attestation["passed"],
            "case_count": evidence["case_count"],
            "fixture_bsp_sha256": fixture["bsp_sha256"],
            "fixture_provenance_sha256": sha256_bytes(canonical_json(fixture)),
            "fixture_collision_physics_identity_sha256": identities["collision"]["physics_identity"],
            "fixture_pmove_physics_identity_sha256": identities["pmove"]["physics_identity"],
            "hook_physics_identity_sha256": identities["hook"]["physics_identity"],
            "collision_tool_sha256": binaries["cm_oracle_sha256"],
            "pmove_tool_sha256": binaries["pmove_oracle_sha256"],
            "hook_tool_sha256": binaries["hook_oracle_sha256"],
            "q2ded_sha256": binaries["q2ded_sha256"],
            "game_module_sha256": binaries["probe_game_module_sha256"],
            "transcript_sha256": evidence["vector_results_sha256"],
        }
        parity["attestation_sha256"] = sha256_bytes(_rust_struct_json(parity))
        hook_tool = {
            "name": "q2-hook-oracle",
            "schema": "q2-hook-oracle-v1",
            "version": 1,
            "executable_sha256": sha256_file(hook.binary),
            "tool_identity_sha256": identity["tool_identity"],
            "physics_identity_sha256": identity["physics_identity"],
        }
        hook_source = {
            "shared_c_sha256": source["shared_c_sha256"],
            "shared_h_sha256": source["shared_h_sha256"],
            "integration_sha256": source["integration_sha256"],
            "math_sha256": source["math_sha256"],
            "build_contract": source["build_contract"],
        }
        hook_parameters = {
            "hook_speed_f32_bits": struct.unpack("<I", struct.pack("<f", parameters["hook_speed"]))[0],
            "hook_pullspeed_f32_bits": struct.unpack("<I", struct.pack("<f", parameters["hook_pullspeed"]))[0],
            "hook_sky": parameters["hook_sky"],
            "hook_maxtime_f32_bits": struct.unpack("<I", struct.pack("<f", parameters["hook_maxtime"]))[0],
            "full_velocity_overwrite": parameters["full_velocity_overwrite"],
        }
        hook_record = {
            "tool": hook_tool,
            "bsp": bsp,
            "parameters": hook_parameters,
            "source": hook_source,
            "parity": parity,
        }
        hook_record["contract_sha256"] = sha256_bytes(_rust_struct_json(hook_record))
        output["hook_oracle"] = hook_record
    return output


@dataclass(frozen=True)
class _SurfaceCandidateScope:
    """Caller-complete sparse candidates around reachable L1 corridors."""

    groups: tuple[SurfaceCandidateGroup, ...]
    authorized_chunks: tuple[tuple[int, int, int], ...]
    boundary_l1: tuple[tuple[int, int, int], ...]
    candidate_cells: int


def _zyx(key: Sequence[int]) -> tuple[int, int, int]:
    return int(key[2]), int(key[1]), int(key[0])


def _l0_chunk_key(cell: Sequence[int]) -> tuple[int, int, int]:
    return tuple(int(value) // 16 for value in cell)  # type: ignore[return-value]


def _surface_candidate_scope(
    nodes: Mapping[tuple[int, int, int], NavNode],
    origin: tuple[int, int, int],
) -> _SurfaceCandidateScope:
    """Derive surfaces only from the reachable L1 player-origin corridor.

    Every admitted node contributes its exact sampled origin/hull-bottom.  L1
    samples are one 16-unit cell (four L0 cells) apart while an evidenced band
    expands through five additional L0 cells, so neighboring floor bands
    overlap instead of requiring a 4x4 rescan of each L1 cell.  At a stance
    boundary, wall/ceiling seeds are spaced by at most 16 units over the exact
    node-position hull faces.  No chunk or map AABB is scanned.
    """

    candidates: set[tuple[int, int, int]] = set()
    boundary_l1: set[tuple[int, int, int]] = set()

    def compatible_neighbor(
        key: tuple[int, int, int], dx: int, dy: int, node: NavNode,
    ) -> bool:
        for dz in (-1, 0, 1):
            neighbor = nodes.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if neighbor is not None and (
                (node.standing_clear and neighbor.standing_clear)
                or (node.crouched_clear and neighbor.crouched_clear)
            ):
                return True
        return False

    for key in sorted(nodes, key=_zyx):
        node = nodes[key]
        floor_point = (
            node.position[0], node.position[1],
            node.position[2] - 24.0 - 1e-6,
        )
        candidates.add(_grid_index(floor_point, origin, 4))

        missing = [
            (dx, dy) for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1))
            if not compatible_neighbor(key, dx, dy, node)
        ]
        if not missing and node.standing_clear:
            continue
        boundary_l1.add(key)

        hull_top = 32.0 if node.standing_clear else 4.0
        vertical_offsets = list(range(-18, round(hull_top) + 1, 16))
        if not vertical_offsets or vertical_offsets[-1] < hull_top - 2.0:
            vertical_offsets.append(round(hull_top - 2.0))
        for dx, dy in missing:
            if dx:
                x = node.position[0] + dx * 18.0
                for z_offset in vertical_offsets:
                    candidates.add(_grid_index(
                        (x, node.position[1], node.position[2] + z_offset),
                        origin, 4,
                    ))
            else:
                y = node.position[1] + dy * 18.0
                for z_offset in vertical_offsets:
                    candidates.add(_grid_index(
                        (node.position[0], y, node.position[2] + z_offset),
                        origin, 4,
                    ))

        ceiling_heights = [hull_top + 2.0]
        if not node.standing_clear:
            ceiling_heights.extend(float(value) for value in range(10, 33, 8))
        for height in ceiling_heights:
            for x_offset in (-8.0, 8.0):
                for y_offset in (-8.0, 8.0):
                    candidates.add(_grid_index(
                        (node.position[0] + x_offset,
                         node.position[1] + y_offset,
                         node.position[2] + height),
                        origin, 4,
                    ))

    by_chunk: dict[tuple[int, int, int], set[tuple[int, int, int]]] = defaultdict(set)
    for cell in candidates:
        by_chunk[_l0_chunk_key(cell)].add(cell)
    groups = tuple(
        SurfaceCandidateGroup(chunk, tuple(sorted(cells, key=_zyx)))
        for chunk, cells in sorted(by_chunk.items(), key=lambda item: _zyx(item[0]))
    )

    # Retention is restricted to chunks that own an explicit reachable-
    # corridor candidate. The inward band may cross a chunk boundary only when
    # the adjacent chunk independently contains a floor/wall/ceiling candidate;
    # exterior witness-only chunks are never allocated prospectively.
    authorized = set(by_chunk)

    return _SurfaceCandidateScope(
        groups=groups,
        authorized_chunks=tuple(sorted(authorized, key=_zyx)),
        boundary_l1=tuple(sorted(boundary_l1, key=_zyx)),
        candidate_cells=len(candidates),
    )


def _hurt_boundary_chunks(
    scope: _SurfaceCandidateScope,
    nodes: Mapping[tuple[int, int, int], NavNode],
    origin: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Return the exact one-chunk hazard band below reachable boundaries.

    Pmove player origins are quantized to 1/8 unit.  A grounded origin can
    therefore sit just above the nominal 24-unit floor offset, putting the
    floor candidate in the chunk above a compiled kill brush.  Surface
    retention intentionally does not allocate witness-only neighbor chunks;
    hurt retention has a separate, narrower authority for the immediately
    lower chunk of an evidenced reachable boundary column.  Interior columns,
    horizontal dilation, and gaps of two or more chunks are never admitted.
    """

    chunks: set[tuple[int, int, int]] = set()
    for key in scope.boundary_l1:
        node = nodes[key]
        floor_cell = _grid_index(
            (
                node.position[0],
                node.position[1],
                node.position[2] - 24.0 - 1e-6,
            ),
            origin,
            4,
        )
        floor_chunk = _l0_chunk_key(floor_cell)
        chunks.add((floor_chunk[0], floor_chunk[1], floor_chunk[2] - 1))
    return tuple(sorted(chunks, key=_zyx))


def _claimed_hurt_boundary_chunks(
    safety: Mapping[str, Any],
    origin: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Scope generated hurt retention to claimed lethal-edge floor strips.

    Generator safety metadata remains a candidate, never collision authority.
    Its exact edge segments bound where the analyzer may look; the compiled
    trigger AABB supplies every retained cell and the later CM safety probes
    independently challenge the edge, guard, floor, void, and lethal catch.
    The inward witness offset is identical to that compiled probe, while the
    segment is covered at L0-chunk granularity without voxelizing unrelated
    walls, ceilings, or interior obstacle boundaries.
    """

    chunks: set[tuple[int, int, int]] = set()
    edges = safety.get("lethal_edges")
    if not isinstance(edges, list) or not edges:
        raise AtlasAnalysisError("generated hurt scope lacks lethal-edge candidates")
    for ordinal, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise AtlasAnalysisError(
                f"generated lethal-edge candidate {ordinal} is malformed"
            )
        side = edge.get("side")
        segment = edge.get("segment")
        if (
            side not in {"west", "east", "south", "north"}
            or not isinstance(segment, list)
            or len(segment) != 5
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in segment)
        ):
            raise AtlasAnalysisError(
                f"generated lethal-edge candidate {ordinal} is malformed"
            )
        x0, y0, x1, y1, floor_z = segment
        if side in {"west", "east"}:
            if x0 != x1 or y0 >= y1:
                raise AtlasAnalysisError(
                    f"generated lethal-edge candidate {ordinal} geometry differs"
                )
            x = x0 + 32.125 if side == "west" else x0 - 32.125
            lower = _grid_index((x, y0, floor_z), origin, 4)
            upper = _grid_index(
                (x, y1 - 1e-6, floor_z), origin, 4,
            )
        else:
            if y0 != y1 or x0 >= x1:
                raise AtlasAnalysisError(
                    f"generated lethal-edge candidate {ordinal} geometry differs"
                )
            y = y0 + 32.125 if side == "south" else y0 - 32.125
            lower = _grid_index((x0, y, floor_z), origin, 4)
            upper = _grid_index(
                (x1 - 1e-6, y, floor_z), origin, 4,
            )
        low_chunk = _l0_chunk_key(lower)
        high_chunk = _l0_chunk_key(upper)
        for chunk_y in range(low_chunk[1], high_chunk[1] + 1):
            for chunk_x in range(low_chunk[0], high_chunk[0] + 1):
                chunks.add((chunk_x, chunk_y, low_chunk[2] - 1))
    if not chunks:
        raise AtlasAnalysisError("generated hurt scope retained no boundary chunks")
    return tuple(sorted(chunks, key=_zyx))


def _surface_request_upper_bound(
    groups: Sequence[SurfaceCandidateGroup],
) -> int:
    """Bound scoped discovery requests before the first oracle write.

    Occupancy can reach only Manhattan distance five from an input candidate;
    each candidate can create at most six clear-neighbor surface traces. The
    prospective bound deliberately sums each owner group's union, matching the
    group-local execution cache. That is conservative and keeps preflight memory
    proportional to one 64-unit chunk instead of materializing a map-wide
    multi-million-tuple set.
    """

    offsets = tuple(
        (dx, dy, dz)
        for dz in range(-5, 6)
        for dy in range(-5, 6)
        for dx in range(-5, 6)
        if abs(dx) + abs(dy) + abs(dz) <= 5
    )
    total = 0
    for group in groups:
        cells = set(group.cells)
        occupancy = {
            (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            for cell in cells
            for dx, dy, dz in offsets
        }
        total += len(occupancy) + 6 * len(cells)
    return total


@dataclass(frozen=True)
class _ScopedSurfaceExecution:
    chunks: tuple[SurfaceBandChunk, ...]
    budget_state: L0BudgetState
    occupancy_requests: int
    surface_requests: int
    physical_requests: int


def _discover_surface_groups_incrementally(
    *,
    groups: Sequence[SurfaceCandidateGroup],
    origin: tuple[int, int, int],
    pose: Model0Pose | FixedInlineModelPose,
    budget_state: L0BudgetState,
    cm: OracleProcess,
) -> _ScopedSurfaceExecution:
    """Admit one owner chunk at a time, retaining only evidenced chunks.

    Candidate chunks may exceed the final 1200-chunk budget because an empty
    or clear candidate does not allocate L0.  Each group prospectively admits
    its one possible owner chunk before issuing requests; cumulative state is
    carried into the next group.  This preserves fail-closed accounting without
    reserving thousands of empty candidate chunks up front.
    """

    state = budget_state
    chunks: list[SurfaceBandChunk] = []
    occupancy_requests = 0
    surface_requests = 0
    for group in groups:
        discovered = discover_scoped_surface_bands(
            candidate_groups=(group,),
            reachable_chunks=(group.chunk,),
            boundary_chunk=None,
            atlas_origin=tuple(float(value) for value in origin),
            collision_mask=MASK_PLAYERSOLID,
            pose=pose,
            budget_state=state,
            oracle=cm.call,
            batch_size=cm.limits.oracle_batch,
        )
        if not discovered.is_exact or discovered.value is None:
            raise AtlasAnalysisError(
                discovered.reason or "scoped surface collision evidence is unknown"
            )
        result = discovered.value
        if not result.accepted:
            raise AtlasAnalysisError(result.rejection)
        state = result.budget_state
        chunks.extend(result.chunks)
        occupancy_requests += result.request_counts.occupancy
        surface_requests += result.request_counts.surface
    return _ScopedSurfaceExecution(
        chunks=tuple(chunks),
        budget_state=state,
        occupancy_requests=occupancy_requests,
        surface_requests=surface_requests,
        physical_requests=occupancy_requests + surface_requests,
    )


def _aabb_candidate_groups(
    bounds: Aabb,
    origin: tuple[int, int, int],
    retained_chunks: set[tuple[int, int, int]],
    *,
    max_cells: int = 250_000,
) -> tuple[SurfaceCandidateGroup, ...]:
    """Enumerate a fixed-pose AABB only where corridor retention permits it."""

    lower = _grid_index(bounds.mins, origin, 4)
    upper = _grid_index(
        tuple(math.nextafter(value, -math.inf) for value in bounds.maxs), origin, 4
    )
    by_chunk: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    count = 0
    for chunk in sorted(retained_chunks, key=_zyx):
        chunk_low = tuple(value * 16 for value in chunk)
        chunk_high = tuple(value + 15 for value in chunk_low)
        clipped_low = tuple(max(lower[axis], chunk_low[axis]) for axis in range(3))
        clipped_high = tuple(min(upper[axis], chunk_high[axis]) for axis in range(3))
        if any(clipped_high[axis] < clipped_low[axis] for axis in range(3)):
            continue
        cells = by_chunk.setdefault(chunk, [])
        for z in range(clipped_low[2], clipped_high[2] + 1):
            for y in range(clipped_low[1], clipped_high[1] + 1):
                for x in range(clipped_low[0], clipped_high[0] + 1):
                    count += 1
                    if count > max_cells:
                        raise AtlasAnalysisError(
                            "fixed inline-model surface scope exceeds bounded sparse fill"
                        )
                    cells.append((x, y, z))
    return tuple(
        SurfaceCandidateGroup(chunk, tuple(cells))
        for chunk, cells in sorted(by_chunk.items(), key=lambda item: _zyx(item[0]))
    )


def _l0_chunks(
    nodes: Mapping[tuple[int, int, int], NavNode],
    spawns: Sequence[tuple[int, tuple[float, float, float]]],
    origin: tuple[int, int, int],
    *,
    cm: OracleProcess | None = None,
    metadata: BspMetadata | None = None,
    admitted_hooks: Sequence[Mapping[str, Any]] = (),
    semantic_summary: dict[str, int] | None = None,
    surface_summary: dict[str, Any] | None = None,
    budget_state: L0BudgetState | None = None,
    compact_output: bool = False,
    semantic_scratch_max_bytes: int = MAX_L0_SEMANTIC_SCRATCH_BYTES,
    generated_safety: Mapping[str, Any] | None = None,
) -> list[dict]:
    chunks: dict[tuple[int, int, int], dict[str, Any]] = {}
    # Semantic evidence needs exact union counts, not a second materialized
    # copy of every global cell tuple.  A generated kill plane can retain
    # millions of cells; Python tuple/set accounting for those cells alone can
    # exceed the cold producer's 512 MiB RSS limit even though the immutable L0
    # encoding is only a few MiB.  Store one 4096-bit bitmap per semantic/chunk
    # instead (512 bytes, matching the frozen 16^3 L0 chunk geometry).
    semantic_cells: dict[
        str, dict[tuple[int, int, int], bytearray]
    ] = defaultdict(dict)
    semantic_scratch_bytes = 0
    budget = budget_state or L0BudgetState()
    accounted_bits = {
        (chunk.key, plane) for chunk in budget.chunks for plane in chunk.bitplanes
    }
    accounted_scalars = {
        (chunk.key, plane) for chunk in budget.chunks for plane in chunk.scalar_planes
    }

    def admit_bitplane(chunk: tuple[int, int, int], plane: str) -> None:
        """Reserve one immutable bitplane before its first material allocation."""

        nonlocal budget
        account = (chunk, plane)
        if account in accounted_bits:
            return
        reservation_result = budget.reserve(chunk, L0PlaneKind.BIT, plane)
        if not reservation_result.is_exact or reservation_result.value is None:
            raise AtlasAnalysisError(
                reservation_result.reason or "L0 bitplane budget authority is unknown"
            )
        reservation = reservation_result.value
        if not reservation.accepted:
            raise AtlasAnalysisError(reservation.rejection)
        budget = reservation.state
        accounted_bits.add(account)

    def admit_scalarplane(chunk: tuple[int, int, int], plane: str) -> None:
        """Reserve one immutable scalar plane before material allocation."""

        nonlocal budget
        account = (chunk, plane)
        if account in accounted_scalars:
            return
        reservation_result = budget.reserve(chunk, L0PlaneKind.SCALAR, plane)
        if not reservation_result.is_exact or reservation_result.value is None:
            raise AtlasAnalysisError(
                reservation_result.reason
                or "L0 scalar-plane budget authority is unknown"
            )
        reservation = reservation_result.value
        if not reservation.accepted:
            raise AtlasAnalysisError(reservation.rejection)
        budget = reservation.state
        accounted_scalars.add(account)

    def item_for(l0: tuple[int, int, int]) -> tuple[dict[str, Any], int]:
        chunk = tuple(value // 16 for value in l0)
        local = tuple(value % 16 for value in l0)
        linear = local[0] + 16 * local[1] + 256 * local[2]
        item = chunks.setdefault(
            chunk, {"key": list(chunk), "bits": {}, "scalars": {}}
        )
        return item, linear

    def set_index(l0: tuple[int, int, int], plane: str) -> None:
        chunk = _l0_chunk_key(l0)
        admit_bitplane(chunk, plane)
        # Allocation is deliberately after the cumulative prospective check.
        item, linear = item_for(l0)
        bitmap = item["bits"].setdefault(plane, bytearray(512))
        bitmap[linear // 8] |= 1 << (linear % 8)

    def set_bit(point: Sequence[float], plane: str) -> None:
        set_index(_grid_index(point, origin, 4), plane)

    def set_scalar(point: Sequence[float], plane: str, value: int) -> None:
        nonlocal budget
        if not 0 < value <= 255:
            return
        l0 = _grid_index(point, origin, 4)
        chunk = _l0_chunk_key(l0)
        admit_scalarplane(chunk, plane)
        item, linear = item_for(l0)
        values = item["scalars"].setdefault(plane, bytearray(4096))
        values[linear] = max(value, values[linear])

    def mark_semantic_index(index: tuple[int, int, int], name: str) -> None:
        nonlocal semantic_scratch_bytes
        chunk = _l0_chunk_key(index)
        local = tuple(value % 16 for value in index)
        linear = local[0] + 16 * local[1] + 256 * local[2]
        bitmap = semantic_cells[name].get(chunk)
        if bitmap is None:
            if semantic_scratch_bytes + 512 > semantic_scratch_max_bytes:
                raise AtlasAnalysisError(
                    "L0 semantic scratch exceeds bounded sparse accounting"
                )
            bitmap = bytearray(512)
            semantic_cells[name][chunk] = bitmap
            semantic_scratch_bytes += 512
        bitmap[linear // 8] |= 1 << (linear % 8)

    def mark_semantic(point: Sequence[float], name: str) -> None:
        mark_semantic_index(_grid_index(point, origin, 4), name)

    def fill_bounds(
        mins: Sequence[float], maxs: Sequence[float], plane: str,
        *, scalar: tuple[str, int] | None = None,
        semantic: str | None = None,
    ) -> None:
        lower = _grid_index(mins, origin, 4)
        # Bounds are half-open.  Subtract an epsilon so an exact upper grid
        # plane does not materialize a neighboring cell.
        upper = _grid_index(tuple(value - 1e-6 for value in maxs), origin, 4)
        fill_indices(lower, upper, plane, scalar=scalar, semantic=semantic)

    def semantic_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def fill_indices(
        lower: Sequence[int], upper: Sequence[int], plane: str,
        *, scalar: tuple[str, int] | None = None,
        semantic: str | Sequence[str] | None = None,
    ) -> None:
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
                    for name in semantic_names(semantic):
                        mark_semantic_index(index, name)
                    if scalar is not None:
                        point = _center(index, origin, 4)
                        set_scalar(point, scalar[0], scalar[1])

    surface_scope = _surface_candidate_scope(nodes, origin)
    if surface_summary is not None:
        surface_summary.update({
            "schema": "q2-atlas-scoped-surfaces-v1",
            "model0_candidate_cells": surface_scope.candidate_cells,
            "model0_request_upper_bound": 0,
            "model0_authorized_chunks": len(surface_scope.authorized_chunks),
            "model0_boundary_l1_cells": len(surface_scope.boundary_l1),
            "model0_materialized_chunks": 0,
            "model0_materialized_cells": 0,
            "model0_occupancy_requests": 0,
            "model0_surface_requests": 0,
            "model0_physical_requests": 0,
            "inline_fixed_pose_count": 0,
            "inline_candidate_cells": 0,
            "inline_request_upper_bound": 0,
            "inline_materialized_cells": 0,
            "inline_occupancy_requests": 0,
            "inline_surface_requests": 0,
            "inline_physical_requests": 0,
            "inline_models": [],
        })
    if cm is not None and surface_scope.groups:
        request_upper_bound = _surface_request_upper_bound(surface_scope.groups)
        if surface_scope.candidate_cells > cm.limits.max_surface_candidate_cells:
            raise AtlasAnalysisError(
                f"model-0 scoped surface candidates {surface_scope.candidate_cells} "
                f"exceed bound {cm.limits.max_surface_candidate_cells}"
            )
        if request_upper_bound > cm.limits.max_surface_request_upper_bound:
            raise AtlasAnalysisError(
                f"model-0 scoped surface request upper bound {request_upper_bound} "
                f"exceeds planning bound {cm.limits.max_surface_request_upper_bound}"
            )
        surface_result = _discover_surface_groups_incrementally(
            groups=surface_scope.groups,
            origin=origin,
            pose=Model0Pose(),
            budget_state=budget,
            cm=cm,
        )
        budget = surface_result.budget_state
        accounted_bits.update(
            (chunk.key, plane)
            for chunk in budget.chunks for plane in chunk.bitplanes
        )
        material_planes = (
            (SURF_SKY, "sky"), (SURF_SLICK, "slick"),
            (SURF_WARP, "warp"), (SURF_NODRAW, "nodraw"),
        )
        for surface_chunk in surface_result.chunks:
            for cell in surface_chunk.cells:
                set_index(cell.index, "solid")
                for flag, plane in material_planes:
                    if cell.surface_flags & flag:
                        set_index(cell.index, plane)
        if surface_summary is not None:
            surface_summary.update({
                "model0_request_upper_bound": request_upper_bound,
                "model0_materialized_chunks": len(surface_result.chunks),
                "model0_materialized_cells": sum(
                    len(chunk.cells) for chunk in surface_result.chunks
                ),
                "model0_occupancy_requests": surface_result.occupancy_requests,
                "model0_surface_requests": surface_result.surface_requests,
                "model0_physical_requests": surface_result.physical_requests,
            })

    # Contents stored on L1 nodes came from current-map CM point-contents
    # calls. They remain exact facts, separate from collision surface bands.
    for node in nodes.values():
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
        retained_keys = set(surface_scope.boundary_l1)
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

    if metadata is not None:
        entities = {entity.index: entity for entity in metadata.entities}
        retained_chunks = set(surface_scope.authorized_chunks)
        hurt_boundary_chunks = set(
            _claimed_hurt_boundary_chunks(generated_safety, origin)
            if generated_safety is not None
            else _hurt_boundary_chunks(surface_scope, nodes, origin)
        )

        def translated(bounds: Aabb, translation: Sequence[float]) -> Aabb:
            return Aabb(
                tuple(bounds.mins[axis] + translation[axis] for axis in range(3)),
                tuple(bounds.maxs[axis] + translation[axis] for axis in range(3)),
            )  # type: ignore[arg-type]

        def exact_origin(entity: EntityMetadata) -> tuple[float, float, float] | None:
            raw = ordered_property(entity.properties, "origin")
            if raw is None:
                return (0.0, 0.0, 0.0)
            pieces = raw.split()
            if len(pieces) != 3:
                return None
            try:
                value = tuple(float(piece) for piece in pieces)
            except ValueError:
                return None
            return value if all(math.isfinite(axis) for axis in value) else None  # type: ignore[return-value]

        def fill_inclusive_aabb(
            bounds: Aabb, plane: str,
            *, scalar: tuple[str, int] | None = None,
            semantic: str | Sequence[str] | None = None,
        ) -> None:
            fill_indices(
                _grid_index(bounds.mins, origin, 4),
                _grid_index(bounds.maxs, origin, 4),
                plane, scalar=scalar, semantic=semantic,
            )

        def fill_retained_inclusive_aabb(
            bounds: Aabb, plane: str,
            *, scalar: tuple[str, int] | None = None,
            semantic: str | Sequence[str] | None = None,
        ) -> None:
            """Materialize entity fields only inside the sparse L0 permits.

            Large generated kill planes and authored trigger volumes are exact
            engine hazards, but their full AABBs are not permission to allocate
            dense map-wide L0.  Prefer the chunks retained by reachable surface
            discovery.  If they have no intersection, permit only the exact
            one-chunk-lower band beneath evidenced reachable boundary columns.
            This covers fixed-point grounded origins without dilating interior
            floor columns or distant exterior volume.  Inclusive linked-AABB
            contact is preserved at every clipped boundary.
            """

            lower = _grid_index(bounds.mins, origin, 4)
            upper = _grid_index(bounds.maxs, origin, 4)
            clips: list[
                tuple[
                    tuple[int, int, int],
                    tuple[int, int, int],
                    tuple[int, int, int],
                ]
            ] = []
            def intersect(
                permit: set[tuple[int, int, int]],
            ) -> list[
                tuple[
                    tuple[int, int, int],
                    tuple[int, int, int],
                    tuple[int, int, int],
                ]
            ]:
                result = []
                for chunk in sorted(permit, key=_zyx):
                    chunk_low = tuple(value * 16 for value in chunk)
                    chunk_high = tuple(value + 15 for value in chunk_low)
                    clipped_low = tuple(
                        max(lower[axis], chunk_low[axis]) for axis in range(3)
                    )
                    clipped_high = tuple(
                        min(upper[axis], chunk_high[axis]) for axis in range(3)
                    )
                    if any(
                        clipped_high[axis] < clipped_low[axis]
                        for axis in range(3)
                    ):
                        continue
                    result.append((chunk, clipped_low, clipped_high))
                return result

            clips.extend(intersect(retained_chunks))
            if not clips:
                clips.extend(intersect(hurt_boundary_chunks))

            # Reserve this field's complete affected plane set before its first
            # cell is allocated. The authoritative 1,200-chunk/16-MiB budget,
            # not an unrelated dense-cell cap, bounds retained sparse storage.
            for chunk, _clipped_low, _clipped_high in clips:
                admit_bitplane(chunk, plane)
                if scalar is not None:
                    admit_scalarplane(chunk, scalar[0])
            for _chunk, clipped_low, clipped_high in clips:
                fill_indices(
                    clipped_low, clipped_high, plane,
                    scalar=scalar, semantic=semantic,
                )

        def fill_potential_envelope(bounds: Aabb, semantic: str) -> None:
            """Stream a conservative dynamic envelope through retained chunks.

            This is not fixed-surface discovery and must not use that path's
            candidate-cell planning cap. First compute only the small per-chunk
            clipped ranges, then cumulatively admit both immutable L0 planes for
            every affected chunk before allocating any cell set. Materialize
            local linear indices directly so a large train envelope never builds
            an intermediate list of global cell tuples.
            """

            lower = _grid_index(bounds.mins, origin, 4)
            upper = _grid_index(
                tuple(math.nextafter(value, -math.inf) for value in bounds.maxs),
                origin, 4,
            )
            clips: list[
                tuple[
                    tuple[int, int, int],
                    tuple[int, int, int],
                    tuple[int, int, int],
                ]
            ] = []
            cell_count = 0
            for chunk in sorted(retained_chunks, key=_zyx):
                chunk_low = tuple(value * 16 for value in chunk)
                chunk_high = tuple(value + 15 for value in chunk_low)
                clipped_low = tuple(
                    max(lower[axis], chunk_low[axis]) for axis in range(3)
                )
                clipped_high = tuple(
                    min(upper[axis], chunk_high[axis]) for axis in range(3)
                )
                if any(
                    clipped_high[axis] < clipped_low[axis] for axis in range(3)
                ):
                    continue
                cell_count += math.prod(
                    clipped_high[axis] - clipped_low[axis] + 1
                    for axis in range(3)
                )
                if cell_count > 2_000_000:
                    raise AtlasAnalysisError(
                        "L0 mover envelope exceeds bounded sparse fill"
                    )
                clips.append((chunk, clipped_low, clipped_high))

            for chunk, _clipped_low, _clipped_high in clips:
                admit_bitplane(chunk, "mover_swept_envelope")
                admit_bitplane(chunk, "unknown")

            for chunk, clipped_low, clipped_high in clips:
                item = chunks.setdefault(
                    chunk,
                    {"key": list(chunk), "bits": {}, "scalars": {}},
                )
                mover_cells = item["bits"].setdefault(
                    "mover_swept_envelope", bytearray(512)
                )
                unknown_cells = item["bits"].setdefault(
                    "unknown", bytearray(512)
                )
                for z in range(clipped_low[2], clipped_high[2] + 1):
                    local_z = z % 16
                    for y in range(clipped_low[1], clipped_high[1] + 1):
                        local_y = y % 16
                        row = 16 * local_y + 256 * local_z
                        for x in range(clipped_low[0], clipped_high[0] + 1):
                            linear = x % 16 + row
                            byte_index = linear // 8
                            mask = 1 << (linear % 8)
                            mover_cells[byte_index] |= mask
                            unknown_cells[byte_index] |= mask

        def record_inline_unknown(
            entity: EntityMetadata, model_index: int, reason: str,
        ) -> None:
            if surface_summary is not None:
                surface_summary["inline_models"].append({
                    "entity_index": entity.index,
                    "model_index": model_index,
                    "classname": entity.classname,
                    "authority": "unknown",
                    "reason": reason,
                })

        def discover_fixed_pose(
            entity: EntityMetadata,
            model_index: int,
            headnode: int,
            bounds: Aabb,
            pose_origin: tuple[float, float, float],
            pose_angles: tuple[float, float, float],
        ) -> None:
            nonlocal budget
            if cm is None:
                record_inline_unknown(entity, model_index, "collision oracle absent")
                return
            groups = _aabb_candidate_groups(bounds, origin, retained_chunks)
            if not groups:
                record_inline_unknown(
                    entity, model_index, "fixed pose is outside retained reachable corridor"
                )
                return
            request_upper_bound = _surface_request_upper_bound(groups)
            candidate_count = sum(len(group.cells) for group in groups)
            if candidate_count > cm.limits.max_surface_candidate_cells:
                raise AtlasAnalysisError(
                    f"entity {entity.index} fixed-pose candidates {candidate_count} "
                    f"exceed bound {cm.limits.max_surface_candidate_cells}"
                )
            if request_upper_bound > cm.limits.max_surface_request_upper_bound:
                raise AtlasAnalysisError(
                    f"entity {entity.index} fixed-pose surface request upper bound "
                    f"{request_upper_bound} exceeds planning bound "
                    f"{cm.limits.max_surface_request_upper_bound}"
                )
            result = _discover_surface_groups_incrementally(
                groups=groups,
                origin=origin,
                pose=FixedInlineModelPose(
                    model_index=model_index, headnode=headnode,
                    origin=pose_origin, angles=pose_angles,
                ),
                budget_state=budget,
                cm=cm,
            )
            budget = result.budget_state
            accounted_bits.update(
                (chunk.key, plane)
                for chunk in budget.chunks for plane in chunk.bitplanes
            )
            material_planes = (
                (SURF_SKY, "sky"), (SURF_SLICK, "slick"),
                (SURF_WARP, "warp"), (SURF_NODRAW, "nodraw"),
            )
            for surface_chunk in result.chunks:
                for cell in surface_chunk.cells:
                    set_index(cell.index, "mover_reference_solid")
                    for flag, plane in material_planes:
                        if cell.surface_flags & flag:
                            set_index(cell.index, plane)
            if surface_summary is not None:
                surface_summary["inline_fixed_pose_count"] += 1
                surface_summary["inline_candidate_cells"] += sum(
                    len(group.cells) for group in groups
                )
                surface_summary["inline_request_upper_bound"] += request_upper_bound
                surface_summary["inline_materialized_cells"] += sum(
                    len(chunk.cells) for chunk in result.chunks
                )
                surface_summary["inline_occupancy_requests"] += result.occupancy_requests
                surface_summary["inline_surface_requests"] += result.surface_requests
                surface_summary["inline_physical_requests"] += result.physical_requests
                surface_summary["inline_models"].append({
                    "entity_index": entity.index,
                    "model_index": model_index,
                    "classname": entity.classname,
                    "authority": "exact-fixed-transformed-cm",
                    "candidate_cells": sum(len(group.cells) for group in groups),
                    "materialized_cells": sum(len(chunk.cells) for chunk in result.chunks),
                    "occupancy_requests": result.occupancy_requests,
                    "surface_requests": result.surface_requests,
                    "physical_requests": result.physical_requests,
                })

        for record in metadata.entity_catalog.brush_submodels:
            entity = entities[int(record["entity_index"])]
            model_index = int(record["model_index"])
            model = metadata.models[model_index]
            raw_bounds = Aabb(model.mins, model.maxs)
            cmodel_bounds = Aabb(
                tuple(value - 1.0 for value in model.mins),
                tuple(value + 1.0 for value in model.maxs),
            )  # type: ignore[arg-type]
            classname = entity.classname.casefold()

            if classname == "trigger_hurt":
                hurt_result = trigger_hurt_semantics(entity, model.mins, model.maxs)
                if not hurt_result.is_exact or hurt_result.value is None:
                    raise AtlasAnalysisError(
                        hurt_result.reason or f"entity {entity.index} hurt semantics are unknown"
                    )
                hurt = hurt_result.value
                runtime_standing = hurt.runtime_standing_forbidden_origins
                runtime_crouched = hurt.runtime_crouched_forbidden_origins
                if (
                    runtime_standing.authority is Authority.EXACT
                    and runtime_crouched.authority is Authority.EXACT
                ):
                    if runtime_standing.value is not None:
                        fill_retained_inclusive_aabb(
                            hurt.linked_touch_bounds, "hurt",
                            scalar=("hazard_severity", 255),
                            semantic=f"hurt:{entity.index}:raw",
                        )
                        fill_retained_inclusive_aabb(
                            runtime_standing.value, "standing_forbidden_origin",
                            scalar=("hazard_severity", 255),
                            semantic=("hurt_expanded", f"hurt:{entity.index}:expanded"),
                        )
                        if runtime_crouched.value is not None:
                            fill_retained_inclusive_aabb(
                                runtime_crouched.value, "crouched_forbidden_origin",
                                scalar=("hazard_severity", 255),
                                semantic=("hurt_expanded", f"hurt:{entity.index}:expanded"),
                            )
                else:
                    fill_retained_inclusive_aabb(
                        hurt.linked_touch_bounds, "unknown",
                        semantic=("hurt_potential", f"hurt:{entity.index}:potential"),
                    )
                    fill_retained_inclusive_aabb(
                        hurt.standing_forbidden_origins, "unknown",
                        semantic=("hurt_potential", f"hurt:{entity.index}:potential"),
                    )
                    fill_retained_inclusive_aabb(
                        hurt.crouched_forbidden_origins, "unknown",
                        semantic=("hurt_potential", f"hurt:{entity.index}:potential"),
                    )
                continue
            if classname in {"trigger_push", "trigger_gravity"}:
                fill_bounds(raw_bounds.mins, raw_bounds.maxs, "push_or_gravity")
                continue
            if "teleport" in classname:
                fill_bounds(raw_bounds.mins, raw_bounds.maxs, "teleport_trigger")
                continue
            if classname == "func_areaportal":
                fill_bounds(raw_bounds.mins, raw_bounds.maxs, "areaportal")
                continue
            if not classname.startswith("func_"):
                continue

            if classname in {"func_door", "func_button", "func_water"}:
                semantics_result = sliding_mover_semantics(
                    entity, cmodel_bounds.mins, cmodel_bounds.maxs
                )
                if not semantics_result.is_exact or semantics_result.value is None:
                    record_inline_unknown(entity, model_index, semantics_result.reason)
                    continue
                semantics = semantics_result.value
                discover_fixed_pose(
                    entity, model_index, model.headnode,
                    semantics.reference_pose.bounds,
                    semantics.current_origin, (0.0, 0.0, 0.0),
                )
                fill_potential_envelope(
                    semantics.potential_envelope.bounds, "mover_dynamic_unknown"
                )
                continue

            if classname in {"func_rotating", "func_door_rotating"}:
                semantics_result = rotating_mover_semantics(
                    entity, cmodel_bounds.mins, cmodel_bounds.maxs
                )
                if not semantics_result.is_exact or semantics_result.value is None:
                    record_inline_unknown(entity, model_index, semantics_result.reason)
                    continue
                semantics = semantics_result.value
                discover_fixed_pose(
                    entity, model_index, model.headnode,
                    semantics.reference_pose.bounds,
                    semantics.origin, semantics.current_angles,
                )
                fill_potential_envelope(
                    semantics.potential_envelope.bounds, "mover_dynamic_unknown"
                )
                continue

            if classname == "func_plat":
                semantics_result = platform_mover_semantics(
                    entity, cmodel_bounds.mins, cmodel_bounds.maxs
                )
                if not semantics_result.is_exact or semantics_result.value is None:
                    record_inline_unknown(entity, model_index, semantics_result.reason)
                    continue
                semantics = semantics_result.value
                current_bounds = (
                    semantics.reference_pose.bounds
                    if semantics.current_origin == semantics.pos1
                    else semantics.endpoint_pose.bounds
                )
                discover_fixed_pose(
                    entity, model_index, model.headnode, current_bounds,
                    semantics.current_origin, (0.0, 0.0, 0.0),
                )
                fill_potential_envelope(
                    semantics.potential_envelope.bounds, "mover_dynamic_unknown"
                )
                continue

            if classname == "func_train":
                topology_result = train_topology(
                    entity, metadata.entities, cmodel_bounds.mins
                )
                topology = topology_result.value
                geometry = None
                if topology is not None:
                    geometry = train_swept_geometry(
                        topology, cmodel_bounds.mins, cmodel_bounds.maxs,
                    )
                    first = topology.group(topology.initial_target) if topology.initial_target else None
                    if (
                        topology_result.is_exact and first is not None
                        and len(first.eligible) == 1
                        and first.eligible[0].train_origin.is_exact
                        and first.eligible[0].train_origin.value is not None
                    ):
                        first_origin = first.eligible[0].train_origin.value
                        discover_fixed_pose(
                            entity, model_index, model.headnode,
                            translated(cmodel_bounds, first_origin),
                            first_origin, (0.0, 0.0, 0.0),
                        )
                    else:
                        record_inline_unknown(
                            entity, model_index,
                            topology_result.reason or "train initial target is unresolved",
                        )
                else:
                    record_inline_unknown(entity, model_index, topology_result.reason)
                # Retain only exact ordinary Move_Calc sweeps. TELEPORT path
                # corners contribute their endpoint poses but never the empty
                # space crossed by the discontinuity.
                if (
                    topology_result.is_exact and geometry is not None
                    and geometry.is_exact and geometry.value is not None
                ):
                    train_envelopes = (
                        *geometry.value.pose_bounds,
                        *geometry.value.linear_segment_bounds,
                    )
                else:
                    train_envelopes = ()
                for bounds in train_envelopes:
                    fill_potential_envelope(
                        bounds,
                        "mover_dynamic_unknown",
                    )
                continue

            if classname in {"func_wall", "func_object", "func_explosive"}:
                pose_origin = exact_origin(entity)
                angles_result = entity_angles(entity.properties)
                if (
                    pose_origin is None or not angles_result.is_exact
                    or angles_result.value is None
                    or angles_result.value != (0.0, 0.0, 0.0)
                ):
                    record_inline_unknown(
                        entity, model_index,
                        "fixed brush pose has malformed or nonzero unsupported angles",
                    )
                    continue
                discover_fixed_pose(
                    entity, model_index, model.headnode,
                    translated(cmodel_bounds, pose_origin),
                    pose_origin, angles_result.value,
                )
                continue

            # No pure law currently proves other mover endpoints.
            # Retain only a conservative potential envelope and Unknown bit;
            # never turn it into a surface or traversal edge.
            pose_origin = exact_origin(entity)
            if pose_origin is not None:
                fill_potential_envelope(
                    translated(cmodel_bounds, pose_origin), "mover_dynamic_unknown"
                )
            record_inline_unknown(
                entity, model_index, f"no exact fixed-pose law for {classname}"
            )

    # Hook-specific L0 cells are derived only from the exact records replayed
    # in this analyzer process. No caller-supplied coordinates are accepted.
    for hook in admitted_hooks:
        anchor = _world_milliunits(
            hook.get("measured_anchor_milliunits", []), "admitted hook anchor"
        )
        set_bit(anchor, "hookable_surface")
        fixed_frames = hook.get("trajectory_origin_fixed")
        trace_sha256 = hook.get("trajectory_sha256")
        if not isinstance(fixed_frames, list) or not fixed_frames:
            raise AtlasAnalysisError("admitted hook lacks ordered Pmove trajectory")
        if validation_trace_sha256(
            str(hook.get("claim_id")), fixed_frames,
            int(hook.get("first_grounded_frame_index", -1)),
        ) != trace_sha256:
            raise AtlasAnalysisError("admitted hook Pmove trajectory digest differs")
        for fixed in fixed_frames:
            if not isinstance(fixed, list) or len(fixed) != 3 or any(
                isinstance(axis, bool) or not isinstance(axis, int) for axis in fixed
            ):
                raise AtlasAnalysisError("admitted hook Pmove trajectory frame is invalid")
            set_bit(tuple(axis / 8.0 for axis in fixed), "hook_corridor")

    for _, spawn in spawns:
        # A 96-unit column is the half-open interval [0, 96); sampling the
        # endpoint would retain a 100-unit column and can create an unrelated
        # upper chunk at an exact 64-unit boundary.
        for z in range(0, 96, 4):
            set_bit((spawn[0], spawn[1], spawn[2] + z), "spawn_column")
    if len(chunks) != len(budget.chunks):
        raise AtlasAnalysisError("L0 materialization differs from prospective chunk accounting")
    for account in budget.chunks:
        item = chunks.get(account.key)
        if item is None:
            raise AtlasAnalysisError("L0 budget retained a chunk with no materialized cells")
        if set(item["bits"]) != set(account.bitplanes):
            raise AtlasAnalysisError(
                f"L0 bitplane accounting differs for chunk {account.key}"
            )
        if set(item["scalars"]) != set(account.scalar_planes):
            raise AtlasAnalysisError(
                f"L0 scalar-plane accounting differs for chunk {account.key}"
            )
    if surface_summary is not None:
        surface_summary.update({
            "l0_accounted_chunks": len(budget.chunks),
            "l0_accounted_bytes": budget.encoded_bytes,
            "l0_max_chunks": budget.max_chunks,
            "l0_max_bytes": budget.max_bytes,
            "l0_semantic_scratch_bytes": semantic_scratch_bytes,
            "l0_semantic_scratch_max_bytes": semantic_scratch_max_bytes,
        })
    output = []
    mover_count = 0
    for key in sorted(chunks, key=lambda item: (item[2], item[1], item[0])):
        item = chunks[key]
        unknown_bits = set(item["bits"]) - FROZEN_L0_BIT_PLANE_NAMES
        unknown_scalars = set(item["scalars"]) - FROZEN_L0_SCALAR_PLANE_NAMES
        if unknown_bits or unknown_scalars:
            raise AtlasAnalysisError(
                f"L0 planes differ from frozen Rust schema: "
                f"bits={sorted(unknown_bits)}, scalars={sorted(unknown_scalars)}"
            )
        mover_count += sum(
            byte.bit_count()
            for byte in item["bits"].get("mover_swept_envelope", ())
        )
        if compact_output:
            item["bitmaps"] = {
                name: bytes(bitmap).hex()
                for name, bitmap in sorted(item["bits"].items())
            }
            item["scalar_values"] = {
                name: bytes(values).hex()
                for name, values in sorted(item["scalars"].items())
            }
            del item["bits"]
            del item["scalars"]
        else:
            item["bits"] = {
                name: [
                    byte_index * 8 + bit
                    for byte_index, byte in enumerate(bitmap)
                    for bit in range(8)
                    if byte & (1 << bit)
                ]
                for name, bitmap in sorted(item["bits"].items())
            }
            item["scalars"] = {
                name: [
                    [linear, value]
                    for linear, value in enumerate(values)
                    if value
                ]
                for name, values in sorted(item["scalars"].items())
            }
        output.append(item)
    if semantic_summary is not None:
        semantic_summary.update({
            name: sum(
                byte.bit_count()
                for bitmap in semantic_chunks.values()
                for byte in bitmap
            )
            for name, semantic_chunks in sorted(semantic_cells.items())
        })
        # Dynamic mover envelopes have a dedicated immutable bitplane, so its
        # exact union count does not require a second map-wide semantic bitmap.
        if mover_count:
            semantic_summary["mover_dynamic_unknown"] = mover_count
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


def _atlas_channels() -> list[dict[str, Any]]:
    channels = []
    for name in sorted(FROZEN_L0_BIT_PLANE_NAMES):
        channels.append({
            "store": "Atlas", "level": 0, "name": name,
            "encoding": "bitplane", "persistence": "map-static",
        })
    for name in sorted(FROZEN_L0_SCALAR_PLANE_NAMES):
        channels.append({
            "store": "Atlas", "level": 0, "name": name,
            "encoding": "u8", "persistence": "map-static",
        })
    for level in (1, 2, 3):
        for name, encoding in (
            ("clearance", "u16"),
            ("confidence", "u16"),
            ("contents_flags", "u32-bitset"),
            ("cost_to_safety", "u32-q8"),
            ("hazard_clearance", "i32-q8"),
            ("hazard_severity", "u8"),
            ("hazard_types", "u16-bitset"),
            ("stance_passability", "u8-bitset"),
        ):
            channels.append({
                "store": "Atlas", "level": level, "name": name,
                "encoding": encoding, "persistence": "map-static",
            })
    return sorted(channels, key=lambda item: (item["store"], item["level"], item["name"]))


def _write_canonical_atlas_manifest(
    path: Path,
    *,
    canonical_map_id: str,
    metadata: BspMetadata,
    provenance_sha256: str,
    origin: tuple[int, int, int],
    cm_identity: Mapping[str, Any],
    admissions: Mapping[str, Any],
    analyzer_sha256: str,
    generator_sha256: str | None,
    atlas_name: str,
    atlas_raw: Path,
    atlas_zst: Path,
    objective_path: Path,
    objective_count: int,
    pack_result: Mapping[str, Any],
    limitations: Sequence[str],
) -> dict[str, Any]:
    counts = {
        "l0_chunks": int(pack_result["l0_chunks"]),
        "l1_nodes": int(pack_result["l1_nodes"]),
        "l1_edges": int(pack_result["l1_edges"]),
        "l2_cells": int(pack_result["l2_cells"]),
        "l3_cells": int(pack_result["l3_cells"]),
    }
    model0 = cm_identity["model0"]
    oracle_records = {
        "b1_runtime_authority_seal": admissions["b1_runtime_authority_seal"],
        "collision_oracle": admissions["collision_oracle"],
        "fall_oracle": admissions["fall_oracle"],
    }
    if "pmove_oracle" in admissions:
        oracle_records["pmove_oracle"] = admissions["pmove_oracle"]
    if "hook_oracle" in admissions:
        oracle_records["hook_oracle"] = admissions["hook_oracle"]
    manifest = {
        "schema_version": 1,
        "byte_order": "little",
        "atlas_magic": "Q2ATL001",
        "specification_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"
        ),
        "bsp": {
            "canonical_map_id": canonical_map_id,
            "sha256": metadata.sha256,
            "provenance_sha256": provenance_sha256,
            "size_bytes": metadata.byte_count,
            "ibsp_version": metadata.version,
        },
        "analyzer": {
            "name": "q2-atlas-analyzer",
            "version": "b2-a-v4",
            "sha256": analyzer_sha256,
        },
        "oracles": oracle_records,
        "recovery_physics": {
            "hook_walk_budget_ticks": 15,
            "game_tick_hz": 10,
            "walk_speed_q8_per_second": 300 * 256,
        },
        "generator": None if generator_sha256 is None else {
            "name": "q2-map-generator",
            "version": "v6",
            "sha256": generator_sha256,
        },
        "grid": {
            "origin": list(origin),
            "model0_mins": [round(float(value)) for value in model0["mins"]],
            "model0_maxs": [round(float(value)) for value in model0["maxs"]],
            "cell_sizes": [4, 16, 64, 256],
            "l0_chunk_dimensions": [16, 16, 16],
        },
        "player_hulls": [
            {"name": "standing", "mins": STANDING_MINS, "maxs": STANDING_MAXS},
            {"name": "crouched", "mins": CROUCHED_MINS, "maxs": CROUCHED_MAXS},
        ],
        "channels": _atlas_channels(),
        "artifacts": {
            atlas_name: {
                "media_type": "application/vnd.q2.atlas-v1",
                "sha256_uncompressed": sha256_file(atlas_raw),
                "uncompressed_size": atlas_raw.stat().st_size,
                "compressed_size": atlas_zst.stat().st_size,
                "counts": dict(sorted(counts.items())),
            },
            objective_path.name: {
                "media_type": OBJECTIVE_MEDIA_TYPE,
                "sha256_uncompressed": sha256_file(objective_path),
                "uncompressed_size": objective_path.stat().st_size,
                "compressed_size": objective_path.stat().st_size,
                "counts": {"objectives": objective_count},
            },
        },
        "counts": counts,
        "budgets": {
            "max_l0_chunks": 1_200,
            "max_l0_decompressed_bytes": 16 * 1024 * 1024,
            "max_atlas_decompressed_bytes": 32 * 1024 * 1024,
            "max_atlas_resident_bytes": 32 * 1024 * 1024,
            "max_build_rss_bytes": 512 * 1024 * 1024,
        },
        "build_peak_rss_bytes": int(pack_result["build_peak_rss_bytes"]),
        "limitations": sorted(set(limitations)),
        "confidence_summary": (
            "Engine-oracle collision and directed movement; map-static Atlas only"
        ),
    }
    path.write_bytes(_rust_struct_json(manifest) + b"\n")
    return manifest


def _process_tree_rss_bytes(root_pid: int) -> int:
    """Sample Linux resident bytes for one process and its live descendants."""
    pending = [root_pid]
    visited: set[int] = set()
    total = 0
    root_sampled = False
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
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            if pid == root_pid:
                raise AtlasAnalysisError(
                    "independent cold analyzer /proc RSS is unreadable"
                ) from error
            continue
        if pid == root_pid:
            root_sampled = True
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                words = line.split()
                if len(words) >= 2:
                    total += int(words[1]) * 1024
                break
        pending.extend(int(value) for value in children.split())
    if not root_sampled:
        raise AtlasAnalysisError(
            "independent cold analyzer /proc RSS sample is unavailable"
        )
    return total


def _run_measured_process(
    command: Sequence[str], *, timeout: float,
) -> tuple[subprocess.CompletedProcess[str], int, int]:
    """Run a process while sampling whole-process-tree RSS at 100 Hz."""
    started = time.monotonic()
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    peak_rss = 0

    def kill_process_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    while True:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            break
        try:
            peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
        except AtlasAnalysisError:
            kill_process_group()
            process.communicate()
            raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kill_process_group()
            stdout, stderr = process.communicate()
            raise AtlasAnalysisError(
                f"independent cold analyzer timed out: {stderr.strip()}"
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.01, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    if peak_rss <= 0:
        raise AtlasAnalysisError(
            "independent cold analyzer did not produce a positive RSS sample"
        )
    elapsed_milliseconds = max(
        1, math.ceil((time.monotonic() - started) * 1000.0)
    )
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        peak_rss,
        elapsed_milliseconds,
    )


def _normalized_analysis_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only measured/derived fields from a candidate analysis manifest."""

    normalized = json.loads(canonical_json(value))
    if (
        normalized.get("schema") != SCHEMA
        or normalized.get("status") != "candidate"
        or normalized.get("deterministic_rebuild") is not False
    ):
        raise AtlasAnalysisError(
            "cold semantic comparison requires a candidate analysis manifest"
        )
    try:
        normalized["performance"].pop("primary_elapsed_milliseconds")
        if "full_cold_rebuild" in normalized["performance"]:
            raise AtlasAnalysisError(
                "candidate analysis manifest already contains a cold proof"
            )
        normalized["artifacts"]["atlas"].pop("build_peak_rss_bytes")
        normalized["identity"].pop("atlas_manifest_sha256")
        atlas_manifest = normalized["artifacts"]["atlas_manifest"]
        atlas_manifest.pop("sha256")
        atlas_manifest.pop("uncompressed_bytes")
        atlas_manifest["verification"].pop("manifest_sha256")
    except (KeyError, TypeError) as error:
        raise AtlasAnalysisError(
            "analysis manifest lacks required semantic-comparison fields"
        ) from error
    return normalized


def _full_cold_rebuild(
    bsp: Path,
    primary_dir: Path,
    canonical_map_id: str,
    provenance: Mapping[str, Any],
    *,
    cm_oracle: Path,
    pmove_oracle: Path | None,
    hook_oracle: Path | None,
    fall_oracle: Path | None,
    hook_attestation: Path | None,
    packer: Path,
    verifier: Path,
    limits: AnalyzerLimits,
    generator_claims_sha256: str | None,
    generator_claims: Mapping[str, Any] | None,
    generator_safety: Mapping[str, Any] | None,
    hook_materialization: Mapping[str, Any] | None,
    primary_analysis_manifest: Mapping[str, Any],
) -> dict:
    """Re-run the complete analyzer in a fresh process and compare artifacts."""
    worker = Path(__file__).resolve().parents[1] / "tools/atlas_cold_worker.py"
    if not worker.is_file():
        raise AtlasAnalysisError(f"independent cold analyzer worker missing: {worker}")
    artifact_suffixes = (
        ".atlas.bin", ".atlas.bin.zst", ".navigation.bin.zst",
        ".visibility.bin.zst", ".design-signature.json", ".objectives.json",
        ".atlas.manifest.json",
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
            "fall_oracle": None if fall_oracle is None else str(fall_oracle.resolve()),
            "hook_attestation": (
                None if hook_attestation is None else str(hook_attestation.resolve())
            ),
            "packer": str(packer.resolve()),
            "verifier": str(verifier.resolve()),
            "limits": asdict(limits),
            "generator_claims_sha256": generator_claims_sha256,
            "generator_claims": None if generator_claims is None else dict(generator_claims),
            "generator_safety": (
                None if generator_safety is None else dict(generator_safety)
            ),
            "hook_materialization": (
                None if hook_materialization is None else dict(hook_materialization)
            ),
        }
        specification_path = root / "worker.json"
        specification_path.write_bytes(canonical_json(specification) + b"\n")
        completed, peak_rss, elapsed_milliseconds = _run_measured_process(
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
        cold_digests = {}
        semantic_digests = {}
        cold_semantic_digests = {}
        for suffix in artifact_suffixes:
            primary = primary_dir / f"{canonical_map_id}{suffix}"
            cold = cold_dir / f"{canonical_map_id}{suffix}"
            if not primary.is_file() or not cold.is_file():
                raise AtlasAnalysisError(f"independent cold artifact missing: {suffix}")
            if suffix == ".atlas.manifest.json":
                primary_manifest = json.loads(primary.read_bytes())
                cold_manifest = json.loads(cold.read_bytes())
                if "build_peak_rss_bytes" not in primary_manifest or (
                    "build_peak_rss_bytes" not in cold_manifest
                ):
                    raise AtlasAnalysisError(
                        "independent cold Atlas manifest lacks operational RSS field"
                    )
                primary_manifest.pop("build_peak_rss_bytes")
                cold_manifest.pop("build_peak_rss_bytes")
                primary_semantic_digest = sha256_bytes(
                    canonical_json(primary_manifest)
                )
                cold_semantic_digest = sha256_bytes(
                    canonical_json(cold_manifest)
                )
                if primary_manifest != cold_manifest:
                    raise AtlasAnalysisError(
                        "independent cold artifact semantic mismatch: " + suffix
                    )
                semantic_digests[suffix] = primary_semantic_digest
                cold_semantic_digests[suffix] = cold_semantic_digest
            else:
                primary_digest = sha256_file(primary)
                cold_digest = sha256_file(cold)
                if primary_digest != cold_digest:
                    raise AtlasAnalysisError(
                        f"independent cold artifact mismatch: {suffix}"
                    )
                digests[suffix] = primary_digest
                cold_digests[suffix] = cold_digest

        analysis_suffix = ".analysis.manifest.json"
        cold_analysis_path = cold_dir / f"{canonical_map_id}{analysis_suffix}"
        if not cold_analysis_path.is_file():
            raise AtlasAnalysisError(
                "independent cold artifact missing: " + analysis_suffix
            )
        try:
            cold_analysis_manifest = json.loads(cold_analysis_path.read_bytes())
        except json.JSONDecodeError as error:
            raise AtlasAnalysisError(
                "independent cold analysis manifest is invalid JSON"
            ) from error
        primary_analysis_semantics = _normalized_analysis_manifest(
            primary_analysis_manifest
        )
        cold_analysis_semantics = _normalized_analysis_manifest(
            cold_analysis_manifest
        )
        if primary_analysis_semantics != cold_analysis_semantics:
            raise AtlasAnalysisError(
                "independent cold artifact semantic mismatch: " + analysis_suffix
            )
        analysis_semantic_sha256 = sha256_bytes(
            canonical_json(primary_analysis_semantics)
        )
        cold_analysis_semantic_sha256 = sha256_bytes(
            canonical_json(cold_analysis_semantics)
        )
        semantic_digests[analysis_suffix] = analysis_semantic_sha256
        cold_semantic_digests[analysis_suffix] = cold_analysis_semantic_sha256

        verifier_summaries = []
        for directory in (primary_dir, cold_dir):
            verified = subprocess.run(
                [
                    str(verifier),
                    str(directory / f"{canonical_map_id}.atlas.manifest.json"),
                    str(directory / f"{canonical_map_id}.atlas.bin"),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30, check=False,
            )
            if verified.returncode:
                raise AtlasAnalysisError(
                    "independent cold Atlas verification failed: "
                    + verified.stderr.strip()
                )
            try:
                summary = json.loads(verified.stdout)
            except json.JSONDecodeError as error:
                raise AtlasAnalysisError(
                    "independent cold Atlas verifier emitted invalid JSON"
                ) from error
            if summary.get("passed") is not True:
                raise AtlasAnalysisError("independent cold Atlas verifier rejected artifact")
            summary.pop("manifest_sha256", None)
            verifier_summaries.append(summary)
        if verifier_summaries[0] != verifier_summaries[1]:
            raise AtlasAnalysisError("independent cold verifier summaries differ")
    if peak_rss > 512 * 1024 * 1024:
        raise AtlasAnalysisError("independent cold analyzer exceeded 512 MiB peak RSS")
    if elapsed_milliseconds > 300_000:
        raise AtlasAnalysisError("independent cold analyzer exceeded 300 seconds")
    return {
        "schema": "q2-atlas-full-cold-proof-v1",
        "independent_process_launches": 1,
        "artifact_count": len(artifact_suffixes) + 1,
        "artifact_sha256": digests,
        "artifact_semantic_sha256": semantic_digests,
        "cold_artifact_sha256": cold_digests,
        "cold_artifact_semantic_sha256": cold_semantic_digests,
        "verifier_sha256": sha256_file(verifier),
        "verification": verifier_summaries[0],
        "sample_interval_milliseconds": 10,
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "peak_rss_limit_bytes": 512 * 1024 * 1024,
        "elapsed_milliseconds": elapsed_milliseconds,
        "timeout_limit_milliseconds": 300_000,
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


def _world_milliunits(value: Sequence[Any], label: str) -> tuple[float, float, float]:
    if len(value) != 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise AtlasAnalysisError(f"{label} must be three integer milliunits")
    return tuple(item / 1000.0 for item in value)  # type: ignore[return-value]


def _hook_trace_request(
    identifier: str,
    source: Sequence[float],
    anchor: Sequence[float],
) -> dict:
    eye = (source[0], source[1], source[2] + 22.0)
    delta = tuple(anchor[axis] - eye[axis] for axis in range(3))
    distance = math.sqrt(sum(axis * axis for axis in delta))
    if distance <= 0:
        raise AtlasAnalysisError("hook source eye equals claimed anchor")
    end = tuple(anchor[axis] + delta[axis] * 8.0 / distance for axis in range(3))
    return _box_request(identifier, eye, end, [0, 0, 0], [0, 0, 0], MASK_SHOT)


def _hook_trace_admits(trace: Mapping[str, Any], anchor: Sequence[float]) -> bool:
    surface = trace.get("surface")
    endpos = trace.get("endpos")
    return bool(
        trace.get("startsolid") is False
        and isinstance(trace.get("fraction"), (int, float))
        and float(trace["fraction"]) < 1.0
        and isinstance(endpos, list)
        and len(endpos) == 3
        and math.dist(tuple(float(axis) for axis in endpos), tuple(anchor)) <= 16.0
        and isinstance(surface, Mapping)
        and isinstance(surface.get("flags"), int)
        and not (int(surface["flags"]) & SURF_SKY)
        and isinstance(trace.get("contents"), int)
        and bool(int(trace["contents"]) & CONTENTS_SOLID)
    )


def _milliunits(point: Sequence[Any], label: str) -> list[int]:
    if len(point) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) for value in point
    ):
        raise AtlasAnalysisError(f"{label} is not a finite vec3")
    return [round(float(value) * 1000.0) for value in point]


def _pmove_state(frame: Mapping[str, Any], label: str) -> dict[str, int]:
    state: dict[str, int] = {}
    for field in ("pm_type", "pm_flags", "pm_time", "gravity"):
        value = frame.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AtlasAnalysisError(f"{label} has invalid {field}")
        state[field] = value
    return state


def _pmove_fixed(frame: Mapping[str, Any], label: str) -> tuple[int, int, int]:
    value = frame.get("origin_fixed")
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(axis, bool) or not isinstance(axis, int) for axis in value
    ):
        raise AtlasAnalysisError(f"{label} has invalid origin_fixed")
    return tuple(value)  # type: ignore[return-value]


def _ceil_pmove_fixed(point: Sequence[Any], label: str) -> tuple[int, int, int]:
    """Return the first eighth-unit origin not below a CM contact point."""

    if len(point) != 3 or any(
        isinstance(axis, bool) or not isinstance(axis, (int, float))
        or not math.isfinite(axis) for axis in point
    ):
        raise AtlasAnalysisError(f"{label} is not a finite vec3")
    return tuple(math.ceil(float(axis) * 8.0) for axis in point)  # type: ignore[return-value]


def _replay_hook_record_exact(
    cm: OracleProcess,
    pmove: OracleProcess,
    hook_process: HookOracleProcess,
    record_value: Mapping[str, Any],
    *,
    request_prefix: str,
    atlas_origin: tuple[int, int, int],
    expected_landing: bool,
    discover_landing: bool = False,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Replay one exact source/release schedule and return measured landing.

    Discovery traces toward the generator-authored ``trace_target_milliunits``
    and seals the distinct compiled ``measured_anchor_milliunits``. Independent
    analysis uses the same preserved trace target and requires exact measured
    anchor, landing, and ordered fixed-origin trajectory reproduction.
    """

    if discover_landing == expected_landing:
        raise AtlasAnalysisError(
            "hook replay must select exactly one of discovery or strict expectation"
        )

    try:
        record = (
            validate_hook_candidate_record_v4(dict(record_value), request_prefix)
            if discover_landing
            else validate_hook_selected_record_v4(dict(record_value), request_prefix)
        )
    except HookClaimsV4Error as error:
        raise AtlasAnalysisError(str(error)) from error
    pmove_identity = getattr(pmove, "identity", None)
    pmove_parameters = (
        pmove_identity.get("parameters")
        if isinstance(pmove_identity, Mapping) else None
    )
    if not isinstance(pmove_parameters, Mapping):
        raise AtlasAnalysisError("hook replay lacks Pmove identity parameters")
    pmove_gravity = pmove_parameters.get("gravity")
    pmove_airaccelerate = pmove_parameters.get("airaccelerate")
    if (
        isinstance(pmove_gravity, bool) or not isinstance(pmove_gravity, int)
        or pmove_gravity < 0
        or isinstance(pmove_airaccelerate, bool)
        or not isinstance(pmove_airaccelerate, (int, float))
        or not math.isfinite(pmove_airaccelerate)
    ):
        raise AtlasAnalysisError("hook replay Pmove parameters are invalid")
    source_mu = record["source_milliunits"]
    trace_target_mu = record["trace_target_milliunits"]
    expected_measured_anchor_mu = record.get("measured_anchor_milliunits")
    desired_landing_mu = record["landing_milliunits"]
    source = tuple(value / 1000.0 for value in source_mu)
    trace_target = tuple(value / 1000.0 for value in trace_target_mu)
    if (
        not discover_landing
        and _grid_index(source, atlas_origin, 16) == _grid_index(
            tuple(value / 1000.0 for value in desired_landing_mu), atlas_origin, 16
        )
    ):
        return None, "source_and_desired_landing_share_l1", None

    source_clear, source_support, anchor_trace = cm.call([
        _box_request(
            f"{request_prefix}:source-clear", source, source,
            STANDING_MINS, STANDING_MAXS,
        ),
        _box_request(
            f"{request_prefix}:source-support", source,
            (source[0], source[1], source[2] - 96.0),
            STANDING_MINS, STANDING_MAXS,
        ),
        _hook_trace_request(f"{request_prefix}:anchor", source, trace_target),
    ])
    support_plane = source_support.get("plane")
    support_end = source_support.get("endpos")
    source_fixed = tuple(value // 125 for value in source_mu)
    if source_clear.get("startsolid") is True or source_clear.get("allsolid") is True:
        return None, "source_hull_blocked", None
    if (
        source_support.get("startsolid") is True
        or not isinstance(source_support.get("fraction"), (int, float))
        or float(source_support["fraction"]) >= 1.0
        or not isinstance(support_plane, Mapping)
        or not isinstance(support_plane.get("normal"), list)
        or len(support_plane["normal"]) != 3
        or float(support_plane["normal"][2]) < 0.7
        or not isinstance(support_end, list)
        or _ceil_pmove_fixed(support_end, "hook source support") != source_fixed
    ):
        return None, "source_not_exactly_supported", None
    if not _hook_trace_admits(anchor_trace, trace_target):
        return None, "anchor_not_exactly_attachable", None
    measured_anchor_mu = _milliunits(
        anchor_trace.get("endpos", []), "hook anchor trace"
    )
    measured_anchor = tuple(value / 1000.0 for value in measured_anchor_mu)
    if (
        expected_landing
        and measured_anchor_mu != expected_measured_anchor_mu
    ):
        return None, "measured_anchor_fixed_mismatch", None

    source_ground = pmove.call([{
        "id": f"{request_prefix}:source-ground",
        "op": "simulate",
        "origin": list(source),
        "velocity": [0.0, 0.0, 0.0],
        "pm_type": 0,
        "pm_flags": 0,
        "pm_time": 0,
        "gravity": pmove_gravity,
        "airaccelerate": float(pmove_airaccelerate),
        "snapinitial": False,
        "commands": [{"msec": 100, "angles": [0, 0, 0]}],
    }])[0]
    source_frames = source_ground.get("frames")
    if (
        not isinstance(source_frames, list)
        or len(source_frames) != 1
        or not isinstance(source_frames[0], Mapping)
        or source_frames[0].get("grounded") is not True
        or _pmove_fixed(source_frames[0], "hook source Pmove frame") != source_fixed
    ):
        return None, "source_not_pmove_grounded", None

    touch = hook_process.call({
        "id": f"{request_prefix}:touch",
        "op": "touch",
        "target_is_owner": False,
        "owner_has_client": True,
        "target_is_nonblocking": False,
        "target_is_flymissile": False,
        "surface_is_sky": False,
    })
    if touch.get("attached") is not True or touch.get("action") != "attach":
        return None, "hook_touch_rejected", None

    current_origin = source
    current_velocity = (0.0, 0.0, 0.0)
    current_state = {
        "pm_type": 0, "pm_flags": 4, "pm_time": 0,
        "gravity": pmove_gravity,
    }
    previous_grounded = True
    trajectory_fixed: list[list[int]] = []
    for tick in range(record["release_after_ticks"]):
        pull = hook_process.call({
            "id": f"{request_prefix}:pull:{tick:02d}",
            "op": "pull",
            "owner_origin": list(current_origin),
            "hook_origin": list(measured_anchor),
            "enemy_is_client": False,
            "prior_velocity": list(current_velocity),
        })
        velocity = pull.get("velocity")
        if pull.get("full_velocity_overwrite") is not True or not isinstance(
            velocity, list
        ) or len(velocity) != 3:
            raise AtlasAnalysisError("hook pull emitted invalid full velocity overwrite")
        movement = pmove.call([{
            "id": f"{request_prefix}:pmove:{tick:02d}",
            "op": "simulate",
            "origin": list(current_origin),
            "velocity": velocity,
            "airaccelerate": float(pmove_airaccelerate),
            **current_state,
            "snapinitial": False,
            "commands": [{"msec": 100, "angles": [0, 0, 0]}],
        }])[0]
        frames = movement.get("frames")
        if not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0], Mapping):
            raise AtlasAnalysisError("hook pull Pmove response lacks its exact frame")
        frame = frames[0]
        fixed = _pmove_fixed(frame, "hook pull Pmove frame")
        trajectory_fixed.append(list(fixed))
        current_origin = tuple(axis / 8.0 for axis in fixed)
        velocity_fixed = frame.get("velocity_fixed")
        if not isinstance(velocity_fixed, list) or len(velocity_fixed) != 3 or any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in velocity_fixed
        ):
            raise AtlasAnalysisError("hook pull Pmove frame has invalid velocity_fixed")
        current_velocity = tuple(axis / 8.0 for axis in velocity_fixed)
        current_state = _pmove_state(frame, "hook pull Pmove frame")
        grounded = frame.get("grounded")
        if not isinstance(grounded, bool):
            raise AtlasAnalysisError("hook pull Pmove frame has invalid grounded state")
        previous_grounded = grounded

    release = pmove.call([{
        "id": f"{request_prefix}:release",
        "op": "simulate",
        "origin": list(current_origin),
        "velocity": list(current_velocity),
        "airaccelerate": float(pmove_airaccelerate),
        **current_state,
        "snapinitial": False,
        "commands": [
            {"msec": 100, "angles": [0, 0, 0]}
            for _ in range(int(HookOracleProcess.PARAMETERS["hook_maxtime"] * 10))
        ],
    }])[0]
    frames = release.get("frames")
    if not isinstance(frames, list):
        raise AtlasAnalysisError("hook release Pmove response has no ordered frames")
    first_grounded_fixed = None
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or not isinstance(frame.get("grounded"), bool):
            raise AtlasAnalysisError("hook release Pmove frame is malformed")
        grounded = bool(frame["grounded"])
        if not previous_grounded and grounded:
            first_grounded_fixed = _pmove_fixed(
                frame, f"hook release Pmove frame {frame_index}"
            )
            trajectory_fixed.append(list(first_grounded_fixed))
            break
        trajectory_fixed.append(list(_pmove_fixed(
            frame, f"hook release Pmove frame {frame_index}"
        )))
        previous_grounded = grounded
    if first_grounded_fixed is None:
        return None, "no_post_release_grounded_transition", None
    measured_landing_mu = [axis * 125 for axis in first_grounded_fixed]
    measured_landing = tuple(axis / 8.0 for axis in first_grounded_fixed)
    desired_landing = tuple(value / 1000.0 for value in desired_landing_mu)
    if (
        not discover_landing
        and _grid_index(measured_landing, atlas_origin, 16) != _grid_index(
            desired_landing, atlas_origin, 16
        )
    ):
        return None, "measured_landing_outside_desired_l1", None
    if expected_landing and measured_landing_mu != desired_landing_mu:
        return None, "measured_landing_fixed_mismatch", None
    eye_mu = [source_mu[0], source_mu[1], source_mu[2] + 22_000]
    measured_distance = round(math.sqrt(sum(
        (measured_anchor_mu[axis] - eye_mu[axis]) ** 2 for axis in range(3)
    )))
    measured = {
        "claim_id": record["claim_id"],
        "source_milliunits": source_mu,
        "trace_target_milliunits": trace_target_mu,
        "measured_anchor_milliunits": measured_anchor_mu,
        "landing_milliunits": measured_landing_mu,
        "release_after_ticks": record["release_after_ticks"],
        "distance_milliunits": measured_distance,
        "flags": record["flags"],
    }
    if expected_landing and measured != record:
        return None, "sealed_hook_record_fixed_mismatch", None
    first_grounded_frame_index = len(trajectory_fixed) - 1
    trace = {
        "claim_id": measured["claim_id"],
        "origin_fixed_frames": trajectory_fixed,
        "first_grounded_frame_index": first_grounded_frame_index,
        "sha256": validation_trace_sha256(
            measured["claim_id"], trajectory_fixed, first_grounded_frame_index
        ),
    }
    return measured, None, trace


def _analyze_hook_claims(
    cm: OracleProcess,
    pmove: OracleProcess | None,
    hook_process: HookOracleProcess | None,
    nodes: Mapping[tuple[int, int, int], NavNode],
    edges: Sequence[Mapping[str, Any]],
    spawn_indices: Mapping[int, tuple[int, int, int]],
    origin: tuple[int, int, int],
    claims: Mapping[str, Any] | None,
    hook_admission: Mapping[str, Any],
    limits: AnalyzerLimits,
) -> dict:
    """Independently replay six sealed source-bound hook claims exactly."""

    del limits
    if claims is None:
        return {
            "authority_admitted": hook_admission["authority_admitted"],
            "omission_reason": hook_admission["omission_reason"],
            "edges": [],
        }
    proposed = claims.get("hook_claims")
    if not isinstance(proposed, list) or len(proposed) != 6:
        raise AtlasAnalysisError("generated claims require six exact hook-v4 records")
    if (
        hook_admission.get("authority_admitted") is not True
        or pmove is None
        or hook_process is None
    ):
        raise AtlasAnalysisError("generated hook replay authority is absent")

    adjacency = _directed_adjacency(edges)
    reachable = set(spawn_indices.values())
    pending = list(reachable)
    while pending:
        current = pending.pop()
        for neighbor in adjacency.get(current, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)
    accepted = []
    for index, proposed_record in enumerate(proposed):
        measured, reason, trace = _replay_hook_record_exact(
            cm, pmove, hook_process, proposed_record,
            request_prefix=f"sealed-hook:{index:04d}",
            atlas_origin=origin,
            expected_landing=True,
        )
        claim_id = proposed_record.get("claim_id", f"hook:{index:04d}")
        if measured is None:
            raise AtlasAnalysisError(f"{claim_id} exact hook replay rejected: {reason}")
        assert trace is not None
        source = tuple(value / 1000.0 for value in measured["source_milliunits"])
        landing = tuple(value / 1000.0 for value in measured["landing_milliunits"])
        source_key = _grid_index(source, origin, 16)
        target_key = _grid_index(landing, origin, 16)
        if source_key == target_key:
            raise AtlasAnalysisError(f"{claim_id} hook edge does not cross L1 cells")
        source_node = nodes.get(source_key)
        target_node = nodes.get(target_key)
        if source_key not in reachable or source_node is None:
            raise AtlasAnalysisError(f"{claim_id} source is not spawn-reachable without hooks")
        if target_node is None or not target_node.supported or not (
            target_node.standing_clear or target_node.crouched_clear
        ):
            raise AtlasAnalysisError(f"{claim_id} measured landing has no supported L1")
        accepted.append({
            "claim_id": measured["claim_id"],
            "source_l1": list(source_key),
            "target_l1": list(target_key),
            "source_milliunits": measured["source_milliunits"],
            "trace_target_milliunits": measured["trace_target_milliunits"],
            "measured_anchor_milliunits": measured["measured_anchor_milliunits"],
            "landing_milliunits": measured["landing_milliunits"],
            "release_after_ticks": measured["release_after_ticks"],
            "distance_milliunits": measured["distance_milliunits"],
            "flags": measured["flags"],
            "trajectory_origin_fixed": trace["origin_fixed_frames"],
            "trajectory_sha256": trace["sha256"],
            "first_grounded_frame_index": trace["first_grounded_frame_index"],
            "landing_l1": list(target_key),
            "physics_identity": str(hook_admission["physics_identity"]),
            "evidence": EVIDENCE_CM_TRACE_V1 | EVIDENCE_PMOVE_V1 | EVIDENCE_HOOK_LAW_V1,
            "validation_version": VALIDATION_VERSION,
        })
    return {
        "authority_admitted": True,
        "omission_reason": None,
        "edges": accepted,
        "_diagnostic_rejections": [],
    }


def _validate_hook_materialization_binding(
    value: Mapping[str, Any],
    *,
    canonical_map_id: str,
    generator_claims: Mapping[str, Any],
    bsp_sha256: str,
    bsp_size: int,
    cm_oracle: Path | None,
    pmove_oracle: Path | None,
    hook_oracle: Path | None,
    fall_oracle: Path | None,
    hook_attestation: Path | None,
    b1_runtime_authority_seal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a valid materialization to claims, BSP, and executable bytes."""
    try:
        document = validate_hook_materialization_v4(dict(value))
    except HookClaimsV4Error as error:
        raise AtlasAnalysisError(str(error)) from error
    if generator_claims.get("schema") != "q2-generator-claims-v3":
        raise AtlasAnalysisError("generated claims schema mismatch")
    if generator_claims.get("map") != canonical_map_id:
        raise AtlasAnalysisError("generated claims map identity mismatch")
    if sha256_bytes(canonical_json(document) + b"\n") != generator_claims[
        "source_files"
    ]["hook_materialization_sha256"]:
        raise AtlasAnalysisError("hook materialization canonical identity differs")
    if document["map"] != canonical_map_id or (
        document["selected_records"] != generator_claims["hook_claims"]
    ):
        raise AtlasAnalysisError("hook materialization differs from generator claims")
    if document["bsp"] != {"sha256": bsp_sha256, "size_bytes": bsp_size}:
        raise AtlasAnalysisError("hook materialization BSP identity differs")
    if document["candidates"]["meta_sha256"] != generator_claims[
        "source_files"
    ]["meta_sha256"]:
        raise AtlasAnalysisError("hook materialization metadata identity differs")
    materialized_oracles = document["oracles"]
    if (
        b1_runtime_authority_seal is None
        or materialized_oracles["b1_runtime_authority_seal"]
        != dict(b1_runtime_authority_seal)
    ):
        raise AtlasAnalysisError("hook materialization retained B1 seal differs")
    for name, binary in (
        ("collision", cm_oracle), ("pmove", pmove_oracle),
        ("hook", hook_oracle), ("fall", fall_oracle),
    ):
        if binary is None or sha256_file(binary) != materialized_oracles[name][
            "executable_sha256"
        ]:
            raise AtlasAnalysisError(
                f"hook materialization {name} executable identity differs"
            )
    if hook_attestation is None or sha256_file(hook_attestation) != materialized_oracles[
        "hook_parity_attestation_sha256"
    ]:
        raise AtlasAnalysisError("hook materialization parity identity differs")
    return document


def _validate_materialized_oracle_identities(
    materialized_oracles: Mapping[str, Any],
    process_identities: Mapping[str, Mapping[str, Any] | None],
) -> None:
    """Reject a live oracle whose reported identity differs from the seal."""
    for name in ("collision", "pmove", "hook", "fall"):
        identity = process_identities.get(name)
        expected = materialized_oracles[name]
        if identity is None or any(
            identity.get(field) != expected[field]
            for field in ("tool_identity", "physics_identity")
        ):
            raise AtlasAnalysisError(
                f"hook materialization {name} oracle identity differs"
            )


def analyze_map(
    bsp: Path,
    output_dir: Path,
    canonical_map_id: str,
    provenance: Mapping[str, Any],
    *,
    cm_oracle: Path,
    pmove_oracle: Path | None,
    hook_oracle: Path | None,
    fall_oracle: Path | None,
    hook_attestation: Path | None,
    packer: Path,
    verifier: Path | None = None,
    limits: AnalyzerLimits = AnalyzerLimits(),
    generator_claims_sha256: str | None = None,
    generator_claims: Mapping[str, Any] | None = None,
    generator_safety: Mapping[str, Any] | None = None,
    hook_materialization: Mapping[str, Any] | None = None,
    independent_cold: bool = True,
) -> dict:
    primary_started = time.monotonic()
    if limits.full_cold_timeout_seconds != 300.0:
        raise AtlasAnalysisError(
            "Atlas full-cold timeout must remain the frozen 300 seconds"
        )
    if (
        pmove_oracle is None or hook_oracle is None or fall_oracle is None
        or hook_attestation is None
    ):
        raise AtlasAnalysisError(
            "Atlas v1 requires the complete CM/Pmove/hook/fall B1 authority set"
        )
    try:
        b1_authority_seal = admit_b1_runtime_authorities(
            cm_oracle=cm_oracle,
            pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle,
            fall_oracle=fall_oracle,
            hook_parity_attestation=hook_attestation,
            analysis_bsp=bsp,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except B1AuthorityError as error:
        raise AtlasAnalysisError(
            f"B1 runtime authority admission failed: {error}"
        ) from error
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
    item_destinations = _authored_item_destinations(metadata)
    teleporter_resolution = resolve_trigger_teleporters(
        metadata.entities,
        metadata.entity_catalog.brush_submodels,
        metadata.models,
    )
    if generator_claims is not None:
        if generator_safety is None:
            raise AtlasAnalysisError("generated claims lack safety proposals")
        if hook_materialization is None:
            raise AtlasAnalysisError("generated claims lack hook materialization identities")
        hook_materialization = _validate_hook_materialization_binding(
            hook_materialization,
            canonical_map_id=canonical_map_id,
            generator_claims=generator_claims,
            bsp_sha256=metadata.sha256,
            bsp_size=metadata.byte_count,
            cm_oracle=cm_oracle,
            pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle,
            fall_oracle=fall_oracle,
            hook_attestation=hook_attestation,
            b1_runtime_authority_seal=b1_authority_seal.as_dict(),
        )
        materialized_oracles = hook_materialization["oracles"]
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / canonical_map_id
    # Quake II's spawn lifecycle raises the selected entity origin by nine
    # units before linking the standing player hull. Analyze that real engine
    # origin while retaining the authored entity origin in the report.
    player_spawns = [
        (entity_ordinal, (point[0], point[1], point[2] + 9.0))
        for entity_ordinal, point in spawns
    ]
    claim_seed_points: list[tuple[float, float, float]] = []
    if generator_claims is not None:
        for claim in generator_claims["route_claims"]:
            claim_seed_points.extend([
                _world_milliunits(
                    claim["source_milliunits"], f"{claim['claim_id']} source"
                ),
                _world_milliunits(
                    claim["target_milliunits"], f"{claim['claim_id']} target"
                ),
            ])
        for claim in generator_claims["hook_claims"]:
            claim_seed_points.extend([
                _world_milliunits(
                    claim["source_milliunits"], f"{claim['claim_id']} source"
                ),
                _world_milliunits(
                    claim["landing_milliunits"], f"{claim['claim_id']} landing"
                ),
            ])
    claim_seed_points = list(dict.fromkeys(claim_seed_points))
    hook = _admit_hook(hook_oracle, hook_attestation)
    with OracleProcess(cm_oracle, bsp, "cm", limits) as cm:
        if cm.identity["map_sha256"] != metadata.sha256:
            raise AtlasAnalysisError("collision oracle loaded different BSP bytes")
        origin = _oracle_atlas_origin(metadata, cm.identity)
        fall_context = None
        pmove_context = None
        hook_context = None
        pmove_source_accounting: dict[str, Any] = {}
        try:
            fall_context = FallOracleProcess(
                fall_oracle,
                max_requests=limits.max_oracle_requests,
                batch_size=limits.oracle_batch,
                batch_timeout_seconds=limits.oracle_batch_timeout_seconds,
                exit_timeout_seconds=limits.process_exit_timeout_seconds,
            )
            pmove_context = OracleProcess(pmove_oracle, bsp, "pmove", limits)
            if (
                generator_claims is not None
                and hook["authority_admitted"] is True
            ):
                hook_context = HookOracleProcess(
                    hook_oracle, str(hook["physics_identity"]), limits
                )
            if pmove_context and pmove_context.identity["map_sha256"] != metadata.sha256:
                raise AtlasAnalysisError("Pmove oracle loaded different BSP bytes")
            if any((
                cm.identity["tool_identity"]
                != b1_authority_seal.collision_tool_identity,
                cm.identity["physics_identity"]
                != b1_authority_seal.collision_physics_identity,
                pmove_context.identity["tool_identity"]
                != b1_authority_seal.pmove_tool_identity,
                pmove_context.identity["physics_identity"]
                != b1_authority_seal.pmove_physics_identity,
                fall_context.identity["tool_identity"]
                != b1_authority_seal.fall_tool_identity,
                fall_context.identity["physics_identity"]
                != b1_authority_seal.fall_physics_identity,
            )):
                raise AtlasAnalysisError(
                    "persistent oracle identities differ from admitted B1 seal"
                )
            if hook_context is not None and any((
                hook_context.identity["tool_identity"]
                != b1_authority_seal.hook_tool_identity,
                hook_context.identity["physics_identity"]
                != b1_authority_seal.hook_physics_identity,
            )):
                raise AtlasAnalysisError(
                    "persistent hook identity differs from admitted B1 seal"
                )
            if hook_materialization is not None:
                process_identities = {
                    "collision": cm.identity,
                    "pmove": None if pmove_context is None else pmove_context.identity,
                    "hook": None if hook_context is None else hook_context.identity,
                    "fall": fall_context.identity,
                }
                _validate_materialized_oracle_identities(
                    materialized_oracles, process_identities
                )
            navigation_seed_points = list(dict.fromkeys((
                *item_destinations, *claim_seed_points,
                *teleporter_seed_points(teleporter_resolution, cm.call),
            )))
            nodes, edges, spawn_indices, drop_classifications = _build_navigation(
                cm, pmove_context, fall_context, bsp, player_spawns, origin, limits,
                navigation_seed_points,
                _dynamic_mover_dependency_index(metadata),
                pmove_source_accounting,
                metadata,
            )
            teleporter_analysis = prove_trigger_teleporter_edges(
                teleporter_resolution, nodes, origin, cm.call,
            )
            edges.extend(teleporter_analysis.edges)
            if len(edges) > limits.max_l1_edges:
                raise AtlasAnalysisError(
                    "teleporter traversal exceeded L1 edge budget"
                )
            # Teleporter arcs can join otherwise separate static components.
            # Recompute strongly-connected IDs and spawn reachability only
            # after their exact CM evidence has been admitted.
            _assign_directed_regions(nodes, edges)
            spawn_reachability = _spawn_reachability(edges, spawn_indices)
            visibility = _visibility(cm, nodes, limits)
            compiled_hooks = _analyze_hook_claims(
                cm, pmove_context, hook_context, nodes, edges, spawn_indices, origin,
                generator_claims, hook, limits,
            )
            hook_rejections = compiled_hooks.pop("_diagnostic_rejections", [])
            if generator_claims is not None:
                expected_hook_traces = {
                    trace["claim_id"]: trace
                    for trace in hook_materialization["validation_traces"]
                }
                for edge in compiled_hooks["edges"]:
                    expected_trace = expected_hook_traces.get(edge["claim_id"])
                    if expected_trace is None or any(
                        edge[field] != expected_trace[expected_field]
                        for field, expected_field in (
                            ("trajectory_origin_fixed", "origin_fixed_frames"),
                            ("trajectory_sha256", "sha256"),
                            ("first_grounded_frame_index", "first_grounded_frame_index"),
                        )
                    ):
                        raise AtlasAnalysisError(
                            f"{edge['claim_id']} validation trajectory differs from materialization"
                        )
                hook_claims_by_id = {
                    claim["claim_id"]: (index, claim)
                    for index, claim in enumerate(generator_claims["hook_claims"])
                }
                for admitted_edge in compiled_hooks["edges"]:
                    claim_index, claim = hook_claims_by_id[admitted_edge["claim_id"]]
                    source_key = tuple(admitted_edge["source_l1"])
                    target_key = tuple(admitted_edge["target_l1"])
                    cost = max(1, round(
                        math.dist(nodes[source_key].position, nodes[target_key].position) * 256
                    ))
                    edges.append({
                        "source": list(source_key),
                        "target": list(target_key),
                        "edge_type": "hook",
                        "stance": "standing",
                        "flags": int(claim["flags"]),
                        "blocker": 0,
                        "cost": cost,
                        "risk": 0,
                        "confidence": 65535,
                        "evidence": admitted_edge["evidence"],
                        "validation_version": admitted_edge["validation_version"],
                        "auxiliary": claim_index,
                    })
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
            surface_evidence: dict[str, Any] = {}
            l0_chunks = _l0_chunks(
                nodes, player_spawns, origin, cm=cm, metadata=metadata,
                admitted_hooks=compiled_hooks["edges"],
                semantic_summary=l0_semantics,
                surface_summary=surface_evidence,
                compact_output=True,
                generated_safety=generator_safety,
            )
            l0_plane_counts: dict[str, int] = defaultdict(int)
            l0_scalar_cell_count = 0
            for chunk in l0_chunks:
                for plane, bitmap_hex in chunk["bitmaps"].items():
                    l0_plane_counts[plane] += sum(
                        byte.bit_count() for byte in bytes.fromhex(bitmap_hex)
                    )
                for values_hex in chunk["scalar_values"].values():
                    l0_scalar_cell_count += sum(
                        bool(value) for value in bytes.fromhex(values_hex)
                    )
            l0_plane_counts = dict(sorted(l0_plane_counts.items()))
            if generator_claims is not None:
                try:
                    compiled_non_hook = analyze_non_hook_claims(
                        cm, metadata, nodes, edges, origin, spawn_records,
                        generator_claims, generator_safety, l0_semantics,
                    )
                except GeneratedClaimProbeError as error:
                    reasons: dict[str, int] = defaultdict(int)
                    for rejection in hook_rejections:
                        reasons[str(rejection["reason"])] += 1
                    accepted_hook_claims = [
                        edge["claim_id"] for edge in compiled_hooks["edges"]
                    ]
                    raise AtlasAnalysisError(
                        f"{error}; accepted_hook_edges={len(compiled_hooks['edges'])}; "
                        f"accepted_hook_claims={accepted_hook_claims}; "
                        f"hook_rejections={dict(sorted(reasons.items()))}"
                    ) from error
            else:
                raw_hazard_cells = sum(
                    1 for node in nodes.values()
                    if node.contents & (CONTENTS_LAVA | CONTENTS_SLIME)
                )
                compiled_non_hook = {
                    "hazards": {
                        "l0_raw_cells": raw_hazard_cells,
                        "l0_expanded_cells": 0,
                        "types": sorted(
                            ({"lava"} if any(node.contents & CONTENTS_LAVA for node in nodes.values()) else set())
                            | ({"slime"} if any(node.contents & CONTENTS_SLIME for node in nodes.values()) else set())
                        ),
                        "lethal_drop_edges": 0,
                        "exact_lethal_candidates_omitted": 0,
                        "guarded_drop_edges": 0,
                        "uncontained_drop_edges": 0,
                    },
                    "hazard_claims": [],
                    "route_claims": [],
                    "lighting": {
                        "lightdata_bytes": metadata.lightmaps.byte_count,
                        "lightdata_sha256": metadata.lightmaps.sha256,
                        "lightmapped_faces": metadata.faces.lightmapped_count,
                        "floor_light_region_count": 0,
                        "floor_light_region_ids": [],
                        "spawn_nav_region_count": len({
                            record["region_id"] for record in spawn_records
                            if record["region_id"]
                        }),
                        "dark_spawns": [],
                    },
                }
            drop_summary = summarize_drop_classifications(drop_classifications)
            compiled_non_hook["hazards"].update({
                "classification_status": "oracle",
                "evidence": EVIDENCE_CM_TRACE_V1 | drop_summary["evidence"],
                "validation_version": DROP_VALIDATION_VERSION,
                "drop_classification": drop_summary,
                "exact_lethal_candidates_omitted": int(
                    drop_summary["exact_lethal"]
                ),
            })
            if generator_claims is None:
                _apply_stock_drop_hazards(
                    compiled_non_hook["hazards"], drop_summary,
                )
            admissions = _admissions(
                metadata.sha256, provenance_sha256, cm, pmove_context, fall_context,
                hook_context if compiled_hooks["edges"] else None,
                hook_attestation,
                b1_authority_seal.as_dict(),
            )
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
                "fall": {
                    "binary_sha256": sha256_file(fall_context.binary),
                    "identity": dict(fall_context.identity),
                    "requests": fall_context.requests,
                },
            }
        finally:
            if hook_context is not None:
                hook_context.close()
            if pmove_context is not None:
                pmove_context.close()
            if fall_context is not None:
                fall_context.close()
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
    objective_document, objective_omissions = _objective_artifact(
        metadata, nodes, spawn_indices, origin, canonical_map_id,
        pack_result["uncompressed_sha256"],
        incident=incident,
        strict_binding=generator_claims is not None,
    )
    objective_path = base.with_suffix(".objectives.json")
    objective_bytes = canonical_json(objective_document) + b"\n"
    objective_path.write_bytes(objective_bytes)
    objective_guideposts = _objective_guidepost_analysis(
        len(objective_document["objectives"]),
        objective_omissions,
    )
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
        "objectives": {
            "sha256": sha256_bytes(objective_bytes),
            "uncompressed_bytes": len(objective_bytes),
            "count": len(objective_document["objectives"]),
            "schema": OBJECTIVE_SCHEMA,
        },
    }
    analyzer_sha256 = atlas_analyzer_authority_sha256(
        Path(__file__).resolve().parents[1]
    )
    limitations = sorted([
        "L0 is a reachable surface/hazard/spawn narrow band, never dense free-space fill",
        "areaportal summaries use declared map-static state only",
        "hook authority is admitted only for explicit source-bound exact replays",
        "inline mover surfaces require exact transformed CM evidence at a fixed declared pose",
        "dynamic mover envelopes remain Unknown and no mover traversal edge is emitted without state evidence",
        "trigger teleporters require a unique target and exact transformed-CM contact/landing evidence",
    ])
    atlas_manifest_path = base.with_suffix(".atlas.manifest.json")
    generator_source = Path(__file__).resolve().parents[1] / "maps/generator.py"
    _write_canonical_atlas_manifest(
        atlas_manifest_path,
        canonical_map_id=canonical_map_id,
        metadata=metadata,
        provenance_sha256=provenance_sha256,
        origin=origin,
        cm_identity=cm_identity,
        admissions=admissions,
        analyzer_sha256=analyzer_sha256,
        generator_sha256=(
            sha256_file(generator_source) if generator_claims is not None else None
        ),
        atlas_name=atlas_raw.name,
        atlas_raw=atlas_raw,
        atlas_zst=atlas_zst,
        objective_path=objective_path,
        objective_count=len(objective_document["objectives"]),
        pack_result=pack_result,
        limitations=limitations,
    )
    verifier = verifier or packer.with_name("q2-atlas-verify")
    if not verifier.is_file():
        raise AtlasAnalysisError(f"Atlas verifier missing: {verifier}")
    verified = subprocess.run(
        [str(verifier), str(atlas_manifest_path), str(atlas_raw)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, check=False,
    )
    if verified.returncode:
        raise AtlasAnalysisError(f"Atlas verifier failed: {verified.stderr.strip()}")
    try:
        verification = json.loads(verified.stdout)
    except json.JSONDecodeError as error:
        raise AtlasAnalysisError("Atlas verifier emitted invalid JSON") from error
    expected_verification = {
        "schema": "q2-atlas-verification-v1",
        "passed": True,
        "canonical_map_id": canonical_map_id,
        "bsp_sha256": metadata.sha256,
        "manifest_sha256": sha256_file(atlas_manifest_path),
        "artifact_name": atlas_raw.name,
        "atlas_sha256": sha256_file(atlas_raw),
        "origin": list(origin),
        "counts": {
            "l0_chunks": int(pack_result["l0_chunks"]),
            "l1_nodes": int(pack_result["l1_nodes"]),
            "l1_edges": int(pack_result["l1_edges"]),
            "l2_cells": int(pack_result["l2_cells"]),
            "l3_cells": int(pack_result["l3_cells"]),
        },
        "collision_contract_sha256": admissions["collision_oracle"]["contract_sha256"],
    }
    if verification != expected_verification:
        raise AtlasAnalysisError("Atlas verifier summary differs from analyzer identities")
    artifacts["atlas_manifest"] = {
        "path": atlas_manifest_path.name,
        "sha256": sha256_file(atlas_manifest_path),
        "uncompressed_bytes": atlas_manifest_path.stat().st_size,
        "verifier_sha256": sha256_file(verifier),
        "verification": verification,
    }
    primary_elapsed_milliseconds = max(
        1, math.ceil((time.monotonic() - primary_started) * 1000.0)
    )
    if primary_elapsed_milliseconds > 300_000:
        raise AtlasAnalysisError("primary Atlas analysis exceeded 300 seconds")
    manifest = {
        "schema": SCHEMA,
        "status": "candidate",
        "deterministic_rebuild": False,
        "confidence": "pending-independent-cold-rebuild",
        "analyzer_version": "b2-a-v4",
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
            "atlas_manifest_sha256": sha256_file(atlas_manifest_path),
        },
        "admissions": {
            "b1_runtime_authority_seal": b1_authority_seal.as_dict(),
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
            "fall": {
                "status": "oracle", "admitted": True,
                "executable_sha256": oracle_manifest["fall"]["binary_sha256"],
                "physics_identity": oracle_manifest["fall"]["identity"]["physics_identity"],
                "tool_identity": oracle_manifest["fall"]["identity"]["tool_identity"],
                "requests": oracle_manifest["fall"]["requests"],
            },
            "hook": hook,
        },
        "compiled_world": {
            "spawns": spawn_records,
            "hazards": compiled_non_hook["hazards"],
            "hazard_claims": compiled_non_hook["hazard_claims"],
            "surfaces": surface_evidence,
            "entity_l0_semantics": l0_semantics,
            "lighting": compiled_non_hook["lighting"],
            "hooks": compiled_hooks,
            "teleporters": teleporter_analysis.report(),
            "objective_guideposts": objective_guideposts,
            "route_claims": compiled_non_hook["route_claims"],
            "pmove_source_accounting": pmove_source_accounting,
        },
        "artifacts": artifacts,
        "counts": {
            "l0_chunks": pack_result["l0_chunks"], "l1_nodes": len(nodes),
            "l1_edges": len(edges), "l2_cells": pack_result["l2_cells"],
            "l3_cells": pack_result["l3_cells"],
            "l0_bit_cells": sum(l0_plane_counts.values()),
            "l0_scalar_cells": l0_scalar_cell_count,
        },
        "l0_plane_counts": l0_plane_counts,
        "confidence_summary": {
            "collision": "exact-engine", "movement": "exact-engine" if pmove_oracle else "omitted",
            "fall": "exact-lithium",
            "hook": (
                "exact-engine-replayed-edges" if compiled_hooks["edges"]
                else "attested-no-replayed-edge" if hook["authority_admitted"]
                else "omitted"
            ),
            "teleporter": (
                "exact-engine-evidenced"
                if teleporter_analysis.edges else "omitted-or-unknown"
            ),
            "metadata": "b1-c-validated",
        },
        "limitations": limitations,
        "performance": {
            "primary_elapsed_milliseconds": primary_elapsed_milliseconds,
            "primary_timeout_limit_milliseconds": 300_000,
            "cm_requests": oracle_manifest["collision"]["requests"],
            "pmove_requests": 0 if oracle_manifest["pmove"] is None else oracle_manifest["pmove"]["requests"],
            "fall_requests": oracle_manifest["fall"]["requests"],
            "surface_occupancy_requests": (
                surface_evidence["model0_occupancy_requests"]
                + surface_evidence["inline_occupancy_requests"]
            ),
            "surface_hit_requests": (
                surface_evidence["model0_surface_requests"]
                + surface_evidence["inline_surface_requests"]
            ),
            "surface_physical_cm_requests": (
                surface_evidence["model0_physical_requests"]
                + surface_evidence["inline_physical_requests"]
            ),
            "surface_request_upper_bound": (
                surface_evidence["model0_request_upper_bound"]
                + surface_evidence["inline_request_upper_bound"]
            ),
        },
    }
    if independent_cold:
        cold_proof = _full_cold_rebuild(
            bsp, output_dir, canonical_map_id, provenance,
            cm_oracle=cm_oracle, pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle, fall_oracle=fall_oracle,
            hook_attestation=hook_attestation,
            packer=packer, verifier=verifier, limits=limits,
            generator_claims_sha256=generator_claims_sha256,
            generator_claims=generator_claims,
            generator_safety=generator_safety,
            hook_materialization=hook_materialization,
            primary_analysis_manifest=manifest,
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
