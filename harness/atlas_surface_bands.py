"""Exact, sparse CM-derived L0 surface-band discovery.

The caller supplies reachable L1/L0 chunk candidates and may authorize one
additional boundary chunk.  This module never derives or scans a map AABB.  It
uses bounded batches of stationary 4-unit cube traces for occupancy, verifies
exposure with a clear face-neighbor, and traces from that neighbor into the
occupied cell to obtain the authoritative collision plane and surface flags.

Only model 0 and a fixed pose of a real inline model are surface authorities.
A mover sweep is deliberately returned as Unknown: a swept envelope is not a
surface and must not be rasterized as one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol, Sequence

from harness.atlas_entity_semantics import (
    AuthorityResult,
    ChunkKey,
    L0BudgetState,
    L0PlaneKind,
    Vec3,
)


L0_CELL_SIZE = 4.0
L0_CHUNK_CELLS = 16
SURFACE_BAND_CELLS = 6
SURFACE_BAND_WORLD_UNITS = 24
DEFAULT_ORACLE_BATCH_SIZE = 256
MAX_ORACLE_BATCH_SIZE = 1_024

SURF_SLICK = 0x2
SURF_SKY = 0x4
SURF_WARP = 0x8
SURF_NODRAW = 0x80

_MATERIAL_PLANES = (
    (SURF_SKY, "sky"),
    (SURF_SLICK, "slick"),
    (SURF_WARP, "warp"),
    (SURF_NODRAW, "nodraw"),
)

# Canonical neighbor order follows z, then y, then x.
_FACE_DIRECTIONS: tuple[tuple[int, int, int], ...] = (
    (0, 0, -1),
    (0, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)

CellKey = tuple[int, int, int]
OracleRequest = dict[str, object]
OracleResponse = Mapping[str, object]


class BatchCollisionOracle(Protocol):
    def __call__(self, requests: Sequence[OracleRequest]) -> Sequence[OracleResponse]: ...


@dataclass(frozen=True)
class Model0Pose:
    """Static world collision, queried with ``box_trace``."""


@dataclass(frozen=True)
class FixedInlineModelPose:
    """One fixed pose of a real BSP inline model.

    ``model_index`` is retained for Atlas provenance.  The CM request itself is
    identified by the oracle-owned ``headnode`` as required by q2-cm-oracle-v1.
    """

    model_index: int
    headnode: int
    origin: Vec3
    angles: Vec3


@dataclass(frozen=True)
class SweptMoverPose:
    """Diagnostic mover sweep metadata; never surface authority."""

    model_index: int
    start_origin: Vec3
    end_origin: Vec3


CollisionPose = Model0Pose | FixedInlineModelPose | SweptMoverPose


class SurfaceClass(str, Enum):
    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"


@dataclass(frozen=True)
class SurfaceWitness:
    """One exact clear-neighbor-to-occupied CM trace."""

    surface_cell: CellKey
    clear_neighbor: CellKey
    normal: Vec3
    classification: SurfaceClass
    surface_flags: int
    surface_name: str
    surface_value: int


@dataclass(frozen=True)
class SurfaceBandCell:
    index: CellKey
    depth_cells: int
    classifications: tuple[SurfaceClass, ...]
    surface_flags: int
    witnesses: tuple[SurfaceWitness, ...]


@dataclass(frozen=True)
class SurfaceBandChunk:
    key: ChunkKey
    cells: tuple[SurfaceBandCell, ...]


@dataclass(frozen=True)
class SurfaceCandidateGroup:
    """Caller-scoped possible surface cells owned by one authorized chunk."""

    chunk: ChunkKey
    cells: tuple[CellKey, ...]


@dataclass(frozen=True)
class SurfaceBandRequestCounts:
    occupancy: int = 0
    surface: int = 0

    @property
    def total(self) -> int:
        return self.occupancy + self.surface


@dataclass(frozen=True)
class SurfaceBandPlan:
    accepted: bool
    reachable_chunks: tuple[ChunkKey, ...]
    boundary_chunk: ChunkKey | None
    authorized_chunks: tuple[ChunkKey, ...]
    atlas_origin: Vec3
    collision_mask: int
    pose: Model0Pose | FixedInlineModelPose
    occupancy_plane: str
    batch_size: int
    initial_budget_state: L0BudgetState
    worst_case_budget_state: L0BudgetState
    rejection: str = ""


@dataclass(frozen=True)
class ScopedSurfaceBandPlan:
    """An admitted candidate-scoped discovery with canonical per-chunk cells."""

    base_plan: SurfaceBandPlan
    candidate_groups: tuple[SurfaceCandidateGroup, ...]
    accepted: bool
    rejection: str = ""


@dataclass(frozen=True)
class SurfaceBandResult:
    accepted: bool
    chunks: tuple[SurfaceBandChunk, ...]
    budget_state: L0BudgetState
    queried_chunks: int
    rejection: str = ""
    request_counts: SurfaceBandRequestCounts = SurfaceBandRequestCounts()


class _OracleEvidenceError(RuntimeError):
    pass


def _zyx(key: tuple[int, int, int]) -> tuple[int, int, int]:
    return key[2], key[1], key[0]


def _finite_vec3(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(
            not isinstance(component, bool)
            and isinstance(component, (int, float))
            and math.isfinite(float(component))
            for component in value
        )
    )


def _valid_chunk(key: object) -> bool:
    return (
        isinstance(key, tuple)
        and len(key) == 3
        and all(not isinstance(value, bool) and isinstance(value, int) for value in key)
    )


def _valid_cell(key: object) -> bool:
    return _valid_chunk(key)


def _cell_chunk(cell: CellKey) -> ChunkKey:
    return tuple(component // L0_CHUNK_CELLS for component in cell)  # type: ignore[return-value]


def _validate_pose(pose: CollisionPose) -> str:
    if isinstance(pose, Model0Pose):
        return ""
    if isinstance(pose, FixedInlineModelPose):
        if isinstance(pose.model_index, bool) or not isinstance(pose.model_index, int) or pose.model_index <= 0:
            return "a fixed inline-model pose requires a positive model index"
        if isinstance(pose.headnode, bool) or not isinstance(pose.headnode, int):
            return "a fixed inline-model pose requires an integer CM headnode"
        if not _finite_vec3(pose.origin):
            return "a fixed inline-model pose requires a finite origin"
        if not _finite_vec3(pose.angles):
            return "a fixed inline-model pose requires finite angles"
        if any(abs(float(value)) > 1_048_576 for value in pose.origin):
            return "a fixed inline-model origin exceeds the oracle coordinate bound"
        if any(abs(float(value)) > 360 for value in pose.angles):
            return "fixed inline-model angles must be normalized to [-360, 360]"
        return ""
    return "a mover sweep is an envelope, not a fixed collision surface"


def plan_surface_band_discovery(
    *,
    reachable_chunks: Sequence[ChunkKey],
    boundary_chunk: ChunkKey | None,
    atlas_origin: Vec3,
    collision_mask: int,
    pose: CollisionPose,
    budget_state: L0BudgetState,
    batch_size: int = DEFAULT_ORACLE_BATCH_SIZE,
) -> AuthorityResult[SurfaceBandPlan]:
    """Prospectively admit an exact sparse discovery before creating requests.

    Admission conservatively reserves the occupancy plane and every material
    bitplane that a trace could require for every caller-authorized chunk.  The
    returned result later accounts only planes that are actually materialized.
    """

    if isinstance(pose, SweptMoverPose):
        return AuthorityResult.unknown(
            None, "mover sweeps are non-surface envelopes; query explicit fixed poses"
        )
    pose_error = _validate_pose(pose)
    if pose_error:
        return AuthorityResult.unknown(None, pose_error)
    if not _finite_vec3(atlas_origin):
        return AuthorityResult.unknown(None, "atlas_origin must be a finite three-vector")
    if isinstance(collision_mask, bool) or not isinstance(collision_mask, int) or collision_mask < 0:
        return AuthorityResult.unknown(None, "collision_mask must be a nonnegative integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or batch_size > MAX_ORACLE_BATCH_SIZE
    ):
        return AuthorityResult.unknown(
            None, f"batch_size must be in [1, {MAX_ORACLE_BATCH_SIZE}]"
        )
    if any(not _valid_chunk(chunk) for chunk in reachable_chunks):
        return AuthorityResult.unknown(None, "reachable chunks must be integer (x,y,z) tuples")
    if boundary_chunk is not None and not _valid_chunk(boundary_chunk):
        return AuthorityResult.unknown(None, "boundary chunk must be one integer (x,y,z) tuple")

    reachable = tuple(sorted(set(reachable_chunks), key=_zyx))
    authorized_set = set(reachable)
    if boundary_chunk is not None:
        authorized_set.add(boundary_chunk)
    authorized = tuple(sorted(authorized_set, key=_zyx))
    occupancy_plane = "solid" if isinstance(pose, Model0Pose) else "mover_reference_solid"

    prospective = budget_state
    rejection = ""
    for chunk in authorized:
        for plane in (occupancy_plane, *(name for _flag, name in _MATERIAL_PLANES)):
            reservation_result = prospective.reserve(chunk, L0PlaneKind.BIT, plane)
            if not reservation_result.is_exact or reservation_result.value is None:
                return AuthorityResult.unknown(None, "L0 budget reservation lost exact authority")
            reservation = reservation_result.value
            if not reservation.accepted:
                rejection = reservation.rejection
                break
            prospective = reservation.state
        if rejection:
            break

    plan = SurfaceBandPlan(
        accepted=not rejection,
        reachable_chunks=reachable,
        boundary_chunk=boundary_chunk,
        authorized_chunks=authorized,
        atlas_origin=tuple(float(value) for value in atlas_origin),  # type: ignore[arg-type]
        collision_mask=collision_mask,
        pose=pose,
        occupancy_plane=occupancy_plane,
        batch_size=batch_size,
        initial_budget_state=budget_state,
        worst_case_budget_state=prospective,
        rejection=rejection,
    )
    return AuthorityResult.exact(plan)


def plan_scoped_surface_band_discovery(
    *,
    candidate_groups: Sequence[SurfaceCandidateGroup],
    reachable_chunks: Sequence[ChunkKey],
    boundary_chunk: ChunkKey | None,
    atlas_origin: Vec3,
    collision_mask: int,
    pose: CollisionPose,
    budget_state: L0BudgetState,
    batch_size: int = DEFAULT_ORACLE_BATCH_SIZE,
) -> AuthorityResult[ScopedSurfaceBandPlan]:
    """Admit and canonicalize an exact caller-scoped candidate discovery.

    Scope errors are deterministic rejected plans.  Malformed values and
    non-surface poses remain Unknown under the base surface-band laws.
    Validation of every group and cell finishes before execution can issue a
    collision request.
    """

    base_result = plan_surface_band_discovery(
        reachable_chunks=reachable_chunks,
        boundary_chunk=boundary_chunk,
        atlas_origin=atlas_origin,
        collision_mask=collision_mask,
        pose=pose,
        budget_state=budget_state,
        batch_size=batch_size,
    )
    if not base_result.is_exact or base_result.value is None:
        return AuthorityResult.unknown(None, base_result.reason)
    base = base_result.value
    if not base.accepted:
        return AuthorityResult.exact(ScopedSurfaceBandPlan(base, (), False, base.rejection))

    authorized = set(base.authorized_chunks)
    by_chunk: dict[ChunkKey, set[CellKey]] = {}
    for group in candidate_groups:
        if not isinstance(group, SurfaceCandidateGroup):
            return AuthorityResult.unknown(None, "candidate groups must be SurfaceCandidateGroup values")
        if not _valid_chunk(group.chunk):
            return AuthorityResult.unknown(None, "candidate group chunks must be integer (x,y,z) tuples")
        if group.chunk not in authorized:
            rejection = f"candidate group chunk {group.chunk} is outside the authorized chunks"
            return AuthorityResult.exact(ScopedSurfaceBandPlan(base, (), False, rejection))
        if not isinstance(group.cells, (tuple, list)):
            return AuthorityResult.unknown(None, "candidate group cells must be a finite sequence")
        selected = by_chunk.setdefault(group.chunk, set())
        for cell in group.cells:
            if not _valid_cell(cell):
                return AuthorityResult.unknown(None, "candidate cells must be integer (x,y,z) tuples")
            if _cell_chunk(cell) != group.chunk:
                rejection = f"candidate cell {cell} is outside its authorized owner chunk {group.chunk}"
                return AuthorityResult.exact(ScopedSurfaceBandPlan(base, (), False, rejection))
            selected.add(cell)

    canonical = tuple(
        SurfaceCandidateGroup(chunk, tuple(sorted(cells, key=_zyx)))
        for chunk, cells in sorted(by_chunk.items(), key=lambda item: _zyx(item[0]))
    )
    return AuthorityResult.exact(ScopedSurfaceBandPlan(base, canonical, True))


def _cell_center(plan: SurfaceBandPlan, cell: CellKey) -> Vec3:
    return tuple(
        plan.atlas_origin[axis] + (cell[axis] + 0.5) * L0_CELL_SIZE
        for axis in range(3)
    )  # type: ignore[return-value]


def _trace_request(
    plan: SurfaceBandPlan,
    identifier: str,
    start: Vec3,
    end: Vec3,
    mins: Vec3,
    maxs: Vec3,
) -> OracleRequest:
    request: OracleRequest = {
        "id": identifier,
        "op": "box_trace" if isinstance(plan.pose, Model0Pose) else "transformed_box_trace",
        "start": list(start),
        "end": list(end),
        "mins": list(mins),
        "maxs": list(maxs),
        "mask": plan.collision_mask,
    }
    if isinstance(plan.pose, FixedInlineModelPose):
        request.update(
            {
                "headnode": plan.pose.headnode,
                "origin": list(plan.pose.origin),
                "angles": list(plan.pose.angles),
            }
        )
    return request


def _occupancy_request(plan: SurfaceBandPlan, chunk: ChunkKey, cell: CellKey) -> OracleRequest:
    center = _cell_center(plan, cell)
    identifier = f"sb:{chunk[0]}:{chunk[1]}:{chunk[2]}:o:{cell[0]}:{cell[1]}:{cell[2]}"
    return _trace_request(
        plan, identifier, center, center, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)
    )


def _surface_request(
    plan: SurfaceBandPlan, chunk: ChunkKey, clear_cell: CellKey, solid_cell: CellKey
) -> OracleRequest:
    identifier = (
        f"sb:{chunk[0]}:{chunk[1]}:{chunk[2]}:s:"
        f"{clear_cell[0]}:{clear_cell[1]}:{clear_cell[2]}:"
        f"{solid_cell[0]}:{solid_cell[1]}:{solid_cell[2]}"
    )
    return _trace_request(
        plan,
        identifier,
        _cell_center(plan, clear_cell),
        _cell_center(plan, solid_cell),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def iter_chunk_occupancy_batches(
    plan: SurfaceBandPlan, chunk: ChunkKey
):
    """Yield bounded occupancy batches for one authorized chunk and 24u halo."""

    if not plan.accepted or chunk not in plan.authorized_chunks:
        return
    base = tuple(component * L0_CHUNK_CELLS for component in chunk)
    low = tuple(component - SURFACE_BAND_CELLS for component in base)
    high = tuple(component + L0_CHUNK_CELLS + SURFACE_BAND_CELLS - 1 for component in base)
    batch: list[OracleRequest] = []
    for z in range(low[2], high[2] + 1):
        for y in range(low[1], high[1] + 1):
            for x in range(low[0], high[0] + 1):
                batch.append(_occupancy_request(plan, chunk, (x, y, z)))
                if len(batch) == plan.batch_size:
                    yield tuple(batch)
                    batch.clear()
    if batch:
        yield tuple(batch)


def _oracle_batch(
    oracle: BatchCollisionOracle, requests: Sequence[OracleRequest]
) -> tuple[OracleResponse, ...]:
    responses = tuple(oracle(requests))
    if len(responses) != len(requests):
        raise _OracleEvidenceError("collision oracle returned a different batch length")
    by_id: dict[str, OracleResponse] = {}
    for response in responses:
        identifier = response.get("id")
        if not isinstance(identifier, str) or identifier in by_id:
            raise _OracleEvidenceError("collision oracle returned a missing or duplicate id")
        if response.get("ok") is not True:
            raise _OracleEvidenceError(f"collision oracle rejected request {identifier}")
        by_id[identifier] = response
    ordered: list[OracleResponse] = []
    for request in requests:
        identifier = request["id"]
        response = by_id.get(str(identifier))
        if response is None or response.get("op") != request.get("op"):
            raise _OracleEvidenceError(f"collision oracle response mismatch for {identifier}")
        ordered.append(response)
    return tuple(ordered)


def _occupied_from_response(response: OracleResponse) -> bool:
    startsolid = response.get("startsolid")
    allsolid = response.get("allsolid")
    if not isinstance(startsolid, bool) or not isinstance(allsolid, bool):
        raise _OracleEvidenceError("occupancy trace lacks exact startsolid/allsolid booleans")
    return startsolid or allsolid


def _surface_witness(
    response: OracleResponse, solid_cell: CellKey, clear_cell: CellKey
) -> SurfaceWitness | None:
    fraction = response.get("fraction")
    startsolid = response.get("startsolid")
    allsolid = response.get("allsolid")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not isinstance(startsolid, bool)
        or not isinstance(allsolid, bool)
    ):
        raise _OracleEvidenceError("surface trace lacks exact hit fields")
    if startsolid or allsolid or float(fraction) >= 1.0:
        return None
    plane = response.get("plane")
    if not isinstance(plane, Mapping) or not _finite_vec3(plane.get("normal")):
        raise _OracleEvidenceError("surface hit lacks a finite CM plane normal")
    normal = tuple(float(value) for value in plane["normal"])  # type: ignore[index,assignment]
    nz = normal[2]
    if nz >= 0.7:
        classification = SurfaceClass.FLOOR
    elif nz <= -0.7:
        classification = SurfaceClass.CEILING
    else:
        classification = SurfaceClass.WALL

    surface = response.get("surface")
    if surface is None:
        flags, name, value = 0, "", 0
    elif isinstance(surface, Mapping):
        flags, name, value = surface.get("flags"), surface.get("name"), surface.get("value")
        if (
            isinstance(flags, bool)
            or not isinstance(flags, int)
            or not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise _OracleEvidenceError("surface hit has malformed CM surface metadata")
    else:
        raise _OracleEvidenceError("surface hit has malformed CM surface metadata")
    return SurfaceWitness(
        surface_cell=solid_cell,
        clear_neighbor=clear_cell,
        normal=normal,
        classification=classification,
        surface_flags=flags,
        surface_name=name,
        surface_value=value,
    )


def _canonical_witnesses(raw_witnesses: Sequence[SurfaceWitness]) -> tuple[SurfaceWitness, ...]:
    unique = {
        (
            witness.surface_cell,
            witness.clear_neighbor,
            witness.normal,
            witness.classification,
            witness.surface_flags,
            witness.surface_name,
            witness.surface_value,
        ): witness
        for witness in raw_witnesses
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                _zyx(item.surface_cell),
                _zyx(item.clear_neighbor),
                item.normal,
                item.surface_flags,
                item.surface_name,
                item.surface_value,
            ),
        )
    )


def _materialize_band_cells(
    best: Mapping[CellKey, tuple[int, list[SurfaceWitness]]]
) -> tuple[SurfaceBandCell, ...]:
    cells: list[SurfaceBandCell] = []
    for cell in sorted(best, key=_zyx):
        depth, raw_witnesses = best[cell]
        witnesses = _canonical_witnesses(raw_witnesses)
        classifications = tuple(
            sorted({witness.classification for witness in witnesses}, key=lambda item: item.value)
        )
        surface_flags = 0
        for witness in witnesses:
            surface_flags |= witness.surface_flags
        cells.append(
            SurfaceBandCell(
                index=cell,
                depth_cells=depth,
                classifications=classifications,
                surface_flags=surface_flags,
                witnesses=witnesses,
            )
        )
    return tuple(cells)


def _probe_occupancy_cells(
    plan: SurfaceBandPlan,
    owner_chunk: ChunkKey,
    cells: Sequence[CellKey],
    oracle: BatchCollisionOracle,
    occupancy: dict[CellKey, bool],
) -> int:
    pending = tuple(sorted((cell for cell in set(cells) if cell not in occupancy), key=_zyx))
    for offset in range(0, len(pending), plan.batch_size):
        batch_cells = pending[offset : offset + plan.batch_size]
        requests = tuple(_occupancy_request(plan, owner_chunk, cell) for cell in batch_cells)
        responses = _oracle_batch(oracle, requests)
        for cell, response in zip(batch_cells, responses):
            occupancy[cell] = _occupied_from_response(response)
    return len(pending)


def _discover_scoped_group(
    plan: SurfaceBandPlan,
    group: SurfaceCandidateGroup,
    authorized_chunks: set[ChunkKey],
    oracle: BatchCollisionOracle,
) -> tuple[dict[CellKey, tuple[int, list[SurfaceWitness]]], SurfaceBandRequestCounts]:
    """Discover one group with a local, input-proportional occupancy cache."""

    candidates = set(group.cells)
    if not candidates:
        return {}, SurfaceBandRequestCounts()

    exposure_cells = set(candidates)
    for cell in candidates:
        for direction in _FACE_DIRECTIONS:
            exposure_cells.add(
                tuple(cell[axis] + direction[axis] for axis in range(3))  # type: ignore[arg-type]
            )
    occupancy: dict[CellKey, bool] = {}
    occupancy_requests = _probe_occupancy_cells(
        plan, group.chunk, tuple(exposure_cells), oracle, occupancy
    )

    trace_candidates: list[tuple[CellKey, CellKey, OracleRequest]] = []
    for cell in sorted(candidates, key=_zyx):
        if not occupancy[cell]:
            continue
        for direction in _FACE_DIRECTIONS:
            neighbor = tuple(cell[axis] + direction[axis] for axis in range(3))
            if not occupancy[neighbor]:  # type: ignore[index]
                trace_candidates.append(
                    (cell, neighbor, _surface_request(plan, group.chunk, neighbor, cell))  # type: ignore[arg-type]
                )

    witnesses_by_seed: dict[CellKey, list[SurfaceWitness]] = {}
    for offset in range(0, len(trace_candidates), plan.batch_size):
        batch_candidates = trace_candidates[offset : offset + plan.batch_size]
        requests = tuple(candidate[2] for candidate in batch_candidates)
        responses = _oracle_batch(oracle, requests)
        for (solid_cell, clear_cell, _request), response in zip(batch_candidates, responses):
            witness = _surface_witness(response, solid_cell, clear_cell)
            if witness is not None:
                witnesses_by_seed.setdefault(solid_cell, []).append(witness)

    best: dict[CellKey, tuple[int, list[SurfaceWitness]]] = {
        seed: (0, list(witnesses)) for seed, witnesses in witnesses_by_seed.items()
    }
    frontier = set(witnesses_by_seed)
    for depth in range(SURFACE_BAND_CELLS - 1):
        proposed_parents: dict[CellKey, list[CellKey]] = {}
        for cell in sorted(frontier, key=_zyx):
            for direction in _FACE_DIRECTIONS:
                neighbor = tuple(cell[axis] + direction[axis] for axis in range(3))
                if neighbor in best or _cell_chunk(neighbor) not in authorized_chunks:  # type: ignore[arg-type]
                    continue
                proposed_parents.setdefault(neighbor, []).append(cell)  # type: ignore[arg-type]
        if not proposed_parents:
            break
        occupancy_requests += _probe_occupancy_cells(
            plan, group.chunk, tuple(proposed_parents), oracle, occupancy
        )
        next_frontier: set[CellKey] = set()
        for cell in sorted(proposed_parents, key=_zyx):
            if not occupancy[cell]:
                continue
            inherited: list[SurfaceWitness] = []
            for parent in sorted(set(proposed_parents[cell]), key=_zyx):
                inherited.extend(best[parent][1])
            best[cell] = (depth + 1, list(_canonical_witnesses(inherited)))
            next_frontier.add(cell)
        frontier = next_frontier
        if not frontier:
            break

    return best, SurfaceBandRequestCounts(occupancy_requests, len(trace_candidates))


def _discover_chunk(
    plan: SurfaceBandPlan, chunk: ChunkKey, oracle: BatchCollisionOracle
) -> tuple[tuple[SurfaceBandCell, ...], SurfaceBandRequestCounts]:
    occupied: set[CellKey] = set()
    occupancy_requests = 0
    for requests in iter_chunk_occupancy_batches(plan, chunk):
        occupancy_requests += len(requests)
        for request, response in zip(requests, _oracle_batch(oracle, requests)):
            if _occupied_from_response(response):
                center = request["start"]
                if not isinstance(center, list):
                    raise _OracleEvidenceError("internal occupancy request lost its center")
                cell = tuple(
                    int(round((float(center[axis]) - plan.atlas_origin[axis]) / L0_CELL_SIZE - 0.5))
                    for axis in range(3)
                )
                occupied.add(cell)  # type: ignore[arg-type]

    base = tuple(component * L0_CHUNK_CELLS for component in chunk)
    seed_low = tuple(component - (SURFACE_BAND_CELLS - 1) for component in base)
    seed_high = tuple(
        component + L0_CHUNK_CELLS + (SURFACE_BAND_CELLS - 2) for component in base
    )

    candidates: list[tuple[CellKey, CellKey, OracleRequest]] = []
    for cell in sorted(occupied, key=_zyx):
        if any(cell[axis] < seed_low[axis] or cell[axis] > seed_high[axis] for axis in range(3)):
            continue
        for direction in _FACE_DIRECTIONS:
            neighbor = tuple(cell[axis] + direction[axis] for axis in range(3))
            if neighbor not in occupied:
                candidates.append(
                    (cell, neighbor, _surface_request(plan, chunk, neighbor, cell))  # type: ignore[arg-type]
                )

    witnesses_by_seed: dict[CellKey, list[SurfaceWitness]] = {}
    for offset in range(0, len(candidates), plan.batch_size):
        batch_candidates = candidates[offset : offset + plan.batch_size]
        requests = tuple(candidate[2] for candidate in batch_candidates)
        responses = _oracle_batch(oracle, requests)
        for (solid_cell, clear_cell, _request), response in zip(batch_candidates, responses):
            witness = _surface_witness(response, solid_cell, clear_cell)
            if witness is not None:
                witnesses_by_seed.setdefault(solid_cell, []).append(witness)

    target_low = base
    target_high = tuple(component + L0_CHUNK_CELLS - 1 for component in base)
    # A target cell retains witnesses from the nearest exposed seed(s).  Each
    # seed BFS is capped at depth five: seed + five inward cells = six 4u cells.
    best: dict[CellKey, tuple[int, list[SurfaceWitness]]] = {}
    for seed in sorted(witnesses_by_seed, key=_zyx):
        queue = deque([(seed, 0)])
        visited = {seed}
        while queue:
            cell, depth = queue.popleft()
            if all(target_low[axis] <= cell[axis] <= target_high[axis] for axis in range(3)):
                prior = best.get(cell)
                if prior is None or depth < prior[0]:
                    best[cell] = (depth, list(witnesses_by_seed[seed]))
                elif depth == prior[0]:
                    prior[1].extend(witnesses_by_seed[seed])
            if depth == SURFACE_BAND_CELLS - 1:
                continue
            for direction in _FACE_DIRECTIONS:
                neighbor = tuple(cell[axis] + direction[axis] for axis in range(3))
                if neighbor in occupied and neighbor not in visited:
                    visited.add(neighbor)  # type: ignore[arg-type]
                    queue.append((neighbor, depth + 1))  # type: ignore[arg-type]

    return _materialize_band_cells(best), SurfaceBandRequestCounts(
        occupancy_requests, len(candidates)
    )


def _reserve_materialized_chunk(
    state: L0BudgetState, chunk: ChunkKey, occupancy_plane: str, surface_flags: int
) -> tuple[L0BudgetState, str]:
    planes = [occupancy_plane]
    planes.extend(name for flag, name in _MATERIAL_PLANES if surface_flags & flag)
    prospective = state
    for plane in planes:
        reservation_result = prospective.reserve(chunk, L0PlaneKind.BIT, plane)
        if not reservation_result.is_exact or reservation_result.value is None:
            return state, "L0 materialization reservation lost exact authority"
        reservation = reservation_result.value
        if not reservation.accepted:
            return state, reservation.rejection
        prospective = reservation.state
    return prospective, ""


def execute_surface_band_plan(
    plan: SurfaceBandPlan, oracle: BatchCollisionOracle
) -> AuthorityResult[SurfaceBandResult]:
    """Execute an admitted plan using exact, bounded CM batches."""

    if not plan.accepted:
        return AuthorityResult.exact(
            SurfaceBandResult(False, (), plan.initial_budget_state, 0, plan.rejection)
        )
    chunks: list[SurfaceBandChunk] = []
    state = plan.initial_budget_state
    queried_chunks = 0
    occupancy_requests = 0
    surface_requests = 0
    try:
        for chunk in plan.authorized_chunks:
            cells, counts = _discover_chunk(plan, chunk, oracle)
            occupancy_requests += counts.occupancy
            surface_requests += counts.surface
            queried_chunks += 1
            if not cells:
                continue
            combined_flags = 0
            for cell in cells:
                combined_flags |= cell.surface_flags
            prospective, rejection = _reserve_materialized_chunk(
                state, chunk, plan.occupancy_plane, combined_flags
            )
            if rejection:
                return AuthorityResult.exact(
                    SurfaceBandResult(False, (), plan.initial_budget_state, queried_chunks, rejection)
                )
            # L0 allocation occurs only after the exact prospective reservation.
            chunks.append(SurfaceBandChunk(chunk, cells))
            state = prospective
    except _OracleEvidenceError as exc:
        return AuthorityResult.unknown(None, str(exc))
    return AuthorityResult.exact(
        SurfaceBandResult(
            True,
            tuple(chunks),
            state,
            queried_chunks,
            request_counts=SurfaceBandRequestCounts(occupancy_requests, surface_requests),
        )
    )


def execute_scoped_surface_band_plan(
    plan: ScopedSurfaceBandPlan, oracle: BatchCollisionOracle
) -> AuthorityResult[SurfaceBandResult]:
    """Execute candidate-scoped discovery without a whole-chunk occupancy scan."""

    base = plan.base_plan
    if not plan.accepted:
        return AuthorityResult.exact(
            SurfaceBandResult(False, (), base.initial_budget_state, 0, plan.rejection)
        )

    best: dict[CellKey, tuple[int, list[SurfaceWitness]]] = {}
    occupancy_requests = 0
    surface_requests = 0
    queried_chunks = 0
    authorized = set(base.authorized_chunks)
    try:
        for group in plan.candidate_groups:
            if not group.cells:
                continue
            local, counts = _discover_scoped_group(base, group, authorized, oracle)
            queried_chunks += 1
            occupancy_requests += counts.occupancy
            surface_requests += counts.surface
            for cell, (depth, witnesses) in local.items():
                prior = best.get(cell)
                if prior is None or depth < prior[0]:
                    best[cell] = (depth, list(witnesses))
                elif depth == prior[0]:
                    prior[1].extend(witnesses)
    except _OracleEvidenceError as exc:
        return AuthorityResult.unknown(None, str(exc))

    by_chunk: dict[ChunkKey, dict[CellKey, tuple[int, list[SurfaceWitness]]]] = {}
    for cell, evidence in best.items():
        by_chunk.setdefault(_cell_chunk(cell), {})[cell] = evidence

    chunks: list[SurfaceBandChunk] = []
    state = base.initial_budget_state
    counts = SurfaceBandRequestCounts(occupancy_requests, surface_requests)
    for chunk in sorted(by_chunk, key=_zyx):
        cells = _materialize_band_cells(by_chunk[chunk])
        combined_flags = 0
        for cell in cells:
            combined_flags |= cell.surface_flags
        prospective, rejection = _reserve_materialized_chunk(
            state, chunk, base.occupancy_plane, combined_flags
        )
        if rejection:
            return AuthorityResult.exact(
                SurfaceBandResult(
                    False,
                    (),
                    base.initial_budget_state,
                    queried_chunks,
                    rejection,
                    counts,
                )
            )
        chunks.append(SurfaceBandChunk(chunk, cells))
        state = prospective
    return AuthorityResult.exact(
        SurfaceBandResult(True, tuple(chunks), state, queried_chunks, request_counts=counts)
    )


def discover_scoped_surface_bands(
    *,
    candidate_groups: Sequence[SurfaceCandidateGroup],
    reachable_chunks: Sequence[ChunkKey],
    boundary_chunk: ChunkKey | None,
    atlas_origin: Vec3,
    collision_mask: int,
    pose: CollisionPose,
    budget_state: L0BudgetState,
    oracle: BatchCollisionOracle,
    batch_size: int = DEFAULT_ORACLE_BATCH_SIZE,
) -> AuthorityResult[SurfaceBandResult]:
    """Discover exact CM bands only around caller-supplied candidate cells.

    Exact describes collision evidence within the admitted candidate scope; it
    does not assert that the caller supplied every surface candidate in a map.
    """

    planned = plan_scoped_surface_band_discovery(
        candidate_groups=candidate_groups,
        reachable_chunks=reachable_chunks,
        boundary_chunk=boundary_chunk,
        atlas_origin=atlas_origin,
        collision_mask=collision_mask,
        pose=pose,
        budget_state=budget_state,
        batch_size=batch_size,
    )
    if not planned.is_exact or planned.value is None:
        return AuthorityResult.unknown(None, planned.reason)
    return execute_scoped_surface_band_plan(planned.value, oracle)


def discover_surface_bands(
    *,
    reachable_chunks: Sequence[ChunkKey],
    boundary_chunk: ChunkKey | None,
    atlas_origin: Vec3,
    collision_mask: int,
    pose: CollisionPose,
    budget_state: L0BudgetState,
    oracle: BatchCollisionOracle,
    batch_size: int = DEFAULT_ORACLE_BATCH_SIZE,
) -> AuthorityResult[SurfaceBandResult]:
    """Plan and execute exact sparse surface discovery in one call."""

    planned = plan_surface_band_discovery(
        reachable_chunks=reachable_chunks,
        boundary_chunk=boundary_chunk,
        atlas_origin=atlas_origin,
        collision_mask=collision_mask,
        pose=pose,
        budget_state=budget_state,
        batch_size=batch_size,
    )
    if not planned.is_exact or planned.value is None:
        return AuthorityResult.unknown(None, planned.reason)
    return execute_surface_band_plan(planned.value, oracle)
