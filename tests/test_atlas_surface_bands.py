from __future__ import annotations

import math
from typing import Callable, Sequence

from harness.atlas_entity_semantics import Authority, L0BudgetState
from harness.atlas_surface_bands import (
    FixedInlineModelPose,
    Model0Pose,
    SURF_NODRAW,
    SURF_SKY,
    SURF_SLICK,
    SurfaceCandidateGroup,
    SurfaceClass,
    SweptMoverPose,
    discover_scoped_surface_bands,
    discover_surface_bands,
    iter_chunk_occupancy_batches,
    plan_surface_band_discovery,
)


Cell = tuple[int, int, int]


class AnalyticOracle:
    """Small deterministic CM double with separate occupancy/hit metadata."""

    def __init__(
        self,
        occupied: Callable[[Cell], bool],
        normal: tuple[float, float, float],
        *,
        hit_flags: int = 0,
        occupancy_flags: int = 0,
    ) -> None:
        self.occupied = occupied
        self.normal = normal
        self.hit_flags = hit_flags
        self.occupancy_flags = occupancy_flags
        self.calls = 0
        self.max_batch = 0
        self.requests: list[dict[str, object]] = []
        self.occupancy_cells: list[Cell] = []

    @staticmethod
    def _cell(point: object) -> Cell:
        assert isinstance(point, list)
        return tuple(int(round(float(point[axis]) / 4.0 - 0.5)) for axis in range(3))  # type: ignore[return-value]

    def __call__(self, requests: Sequence[dict[str, object]]):
        self.calls += 1
        self.max_batch = max(self.max_batch, len(requests))
        self.requests.extend(requests)
        responses = []
        for request in requests:
            start = self._cell(request["start"])
            end = self._cell(request["end"])
            stationary_cube = (
                start == end
                and request["mins"] == [-2.0, -2.0, -2.0]
                and request["maxs"] == [2.0, 2.0, 2.0]
            )
            if stationary_cube:
                self.occupancy_cells.append(start)
                solid = self.occupied(start)
                fraction = 0.0 if solid else 1.0
                flags = self.occupancy_flags
            else:
                solid = self.occupied(end)
                clear = not self.occupied(start)
                fraction = 0.5 if clear and solid else 1.0
                flags = self.hit_flags
            responses.append(
                {
                    "ok": True,
                    "id": request["id"],
                    "op": request["op"],
                    "fraction": fraction,
                    "startsolid": solid if stationary_cube else False,
                    "allsolid": solid if stationary_cube else False,
                    "plane": {
                        "normal": list(self.normal),
                        "dist": 0.0,
                        "type": 0,
                        "signbits": 0,
                    },
                    "surface": {"name": "test", "flags": flags, "value": 7},
                }
            )
        return responses


class NeverCalledOracle:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, requests):
        self.calls += 1
        raise AssertionError("budget rejection must happen before collision requests")


def _discover(chunk, occupied, normal, **oracle_kwargs):
    oracle = AnalyticOracle(occupied, normal, **oracle_kwargs)
    result = discover_surface_bands(
        reachable_chunks=[chunk],
        boundary_chunk=None,
        atlas_origin=(0.0, 0.0, 0.0),
        collision_mask=1,
        pose=Model0Pose(),
        budget_state=L0BudgetState(),
        oracle=oracle,
        batch_size=31,
    )
    assert result.authority is Authority.EXACT
    assert result.value is not None and result.value.accepted
    assert len(result.value.chunks) == 1
    return result.value, oracle


def test_wall_surface_band_is_six_cells_and_never_seven() -> None:
    result, oracle = _discover((-1, 0, 0), lambda cell: cell[0] < 0, (1.0, 0.0, 0.0))
    line = {
        cell.index[0]: cell
        for cell in result.chunks[0].cells
        if cell.index[1:] == (0, 0)
    }
    assert sorted(line) == [-6, -5, -4, -3, -2, -1]
    assert [line[x].depth_cells for x in sorted(line, reverse=True)] == list(range(6))
    assert all(cell.classifications == (SurfaceClass.WALL,) for cell in line.values())
    assert -7 not in line
    assert oracle.max_batch <= 31


def test_floor_and_ceiling_use_exact_normal_thresholds() -> None:
    floor, _ = _discover((0, 0, -1), lambda cell: cell[2] < 0, (0.0, 0.0, 0.7))
    floor_line = [
        cell for cell in floor.chunks[0].cells if cell.index[0] == 0 and cell.index[1] == 0
    ]
    assert [cell.index[2] for cell in floor_line] == [-6, -5, -4, -3, -2, -1]
    assert all(cell.classifications == (SurfaceClass.FLOOR,) for cell in floor_line)

    ceiling, _ = _discover((0, 0, 0), lambda cell: cell[2] >= 0, (0.0, 0.0, -0.7))
    ceiling_line = [
        cell for cell in ceiling.chunks[0].cells if cell.index[0] == 0 and cell.index[1] == 0
    ]
    assert [cell.index[2] for cell in ceiling_line] == [0, 1, 2, 3, 4, 5]
    assert all(cell.classifications == (SurfaceClass.CEILING,) for cell in ceiling_line)


def test_point_six_nine_is_wall_but_point_seven_is_floor() -> None:
    nx69 = math.sqrt(1.0 - 0.69**2)
    wall, _ = _discover((-1, 0, 0), lambda cell: cell[0] < 0, (nx69, 0.0, 0.69))
    assert wall.chunks[0].cells[0].classifications == (SurfaceClass.WALL,)

    nx70 = math.sqrt(1.0 - 0.7**2)
    floor, _ = _discover((-1, 0, 0), lambda cell: cell[0] < 0, (nx70, 0.0, 0.7))
    assert floor.chunks[0].cells[0].classifications == (SurfaceClass.FLOOR,)


def test_unreachable_sealed_geometry_outside_authorized_halo_is_not_queried() -> None:
    def occupied(cell: Cell) -> bool:
        reachable_wall = cell[0] < 0
        sealed_far_box = all(100 <= component <= 110 for component in cell)
        return reachable_wall or sealed_far_box

    result, oracle = _discover((-1, 0, 0), occupied, (1.0, 0.0, 0.0))
    assert result.queried_chunks == 1
    assert max(cell[0] for cell in oracle.occupancy_cells) == 5
    assert all(not all(100 <= component <= 110 for component in cell) for cell in oracle.occupancy_cells)
    assert all(chunk.key == (-1, 0, 0) for chunk in result.chunks)


def test_material_flags_come_only_from_clear_to_occupied_hit_surface() -> None:
    result, _ = _discover(
        (-1, 0, 0),
        lambda cell: cell[0] < 0,
        (1.0, 0.0, 0.0),
        hit_flags=SURF_SKY,
        occupancy_flags=SURF_SLICK | SURF_NODRAW,
    )
    assert all(cell.surface_flags == SURF_SKY for cell in result.chunks[0].cells)
    assert all(
        witness.surface_flags == SURF_SKY
        for cell in result.chunks[0].cells
        for witness in cell.witnesses
    )
    account = result.budget_state.chunks[0]
    assert account.bitplanes == ("sky", "solid")


def test_fixed_pose_emits_exact_transformed_request_shape() -> None:
    pose = FixedInlineModelPose(
        model_index=3,
        headnode=42,
        origin=(100.0, 50.0, 0.0),
        angles=(0.0, 90.0, 0.0),
    )
    planned = plan_surface_band_discovery(
        reachable_chunks=[(0, 0, 0)],
        boundary_chunk=None,
        atlas_origin=(0.0, 0.0, 0.0),
        collision_mask=65539,
        pose=pose,
        budget_state=L0BudgetState(),
        batch_size=7,
    )
    assert planned.is_exact and planned.value is not None and planned.value.accepted
    request = next(iter(iter_chunk_occupancy_batches(planned.value, (0, 0, 0))))[0]
    assert request == {
        "id": "sb:0:0:0:o:-6:-6:-6",
        "op": "transformed_box_trace",
        "start": [-22.0, -22.0, -22.0],
        "end": [-22.0, -22.0, -22.0],
        "mins": [-2.0, -2.0, -2.0],
        "maxs": [2.0, 2.0, 2.0],
        "mask": 65539,
        "headnode": 42,
        "origin": [100.0, 50.0, 0.0],
        "angles": [0.0, 90.0, 0.0],
    }
    assert "model_index" not in request


def test_cumulative_budget_rejection_precedes_request_allocation() -> None:
    oracle = NeverCalledOracle()
    result = discover_surface_bands(
        reachable_chunks=[(0, 0, 0), (1, 0, 0)],
        boundary_chunk=None,
        atlas_origin=(0.0, 0.0, 0.0),
        collision_mask=1,
        pose=Model0Pose(),
        budget_state=L0BudgetState(max_chunks=1),
        oracle=oracle,
        batch_size=8,
    )
    assert result.is_exact and result.value is not None
    assert not result.value.accepted
    assert result.value.queried_chunks == 0
    assert result.value.chunks == ()
    assert "chunk count 2 > 1" in result.value.rejection
    assert oracle.calls == 0


def test_swept_mover_is_unknown_and_cannot_issue_surface_requests() -> None:
    oracle = NeverCalledOracle()
    result = discover_surface_bands(
        reachable_chunks=[(0, 0, 0)],
        boundary_chunk=(1, 0, 0),
        atlas_origin=(0.0, 0.0, 0.0),
        collision_mask=1,
        pose=SweptMoverPose(2, (0.0, 0.0, 0.0), (64.0, 0.0, 0.0)),
        budget_state=L0BudgetState(),
        oracle=oracle,
    )
    assert result.authority is Authority.UNKNOWN
    assert result.value is None
    assert "non-surface" in result.reason
    assert oracle.calls == 0


def _scoped(groups, reachable, oracle, *, budget_state=None):
    return discover_scoped_surface_bands(
        candidate_groups=groups,
        reachable_chunks=reachable,
        boundary_chunk=None,
        atlas_origin=(0.0, 0.0, 0.0),
        collision_mask=1,
        pose=Model0Pose(),
        budget_state=budget_state or L0BudgetState(),
        oracle=oracle,
        batch_size=11,
    )


def test_scoped_sparse_candidate_probes_only_minimal_exposure_halo() -> None:
    oracle = AnalyticOracle(lambda _cell: False, (1.0, 0.0, 0.0))
    result = _scoped(
        [SurfaceCandidateGroup((0, 0, 0), ((0, 0, 0),))],
        [(0, 0, 0)],
        oracle,
    )
    assert result.is_exact and result.value is not None and result.value.accepted
    assert result.value.chunks == ()
    assert result.value.request_counts.occupancy == 7
    assert result.value.request_counts.surface == 0
    assert result.value.request_counts.total == 7
    assert len(set(oracle.occupancy_cells)) == 7
    assert result.value.request_counts.occupancy < 16**3
    assert result.value.request_counts.occupancy < 28**3


def test_scoped_targeted_inward_expansion_keeps_six_cells_not_seventh() -> None:
    oracle = AnalyticOracle(lambda cell: cell[0] < 0, (1.0, 0.0, 0.0))
    result = _scoped(
        [SurfaceCandidateGroup((-1, 0, 0), ((-1, 0, 0),))],
        [(-1, 0, 0)],
        oracle,
    )
    assert result.is_exact and result.value is not None and result.value.accepted
    line = {
        cell.index[0]: cell
        for cell in result.value.chunks[0].cells
        if cell.index[1:] == (0, 0)
    }
    assert sorted(line) == [-6, -5, -4, -3, -2, -1]
    assert [line[x].depth_cells for x in sorted(line, reverse=True)] == list(range(6))
    assert -7 not in line
    assert result.value.request_counts.surface == 1
    assert result.value.request_counts.occupancy == len(set(oracle.occupancy_cells))
    assert result.value.request_counts.occupancy < 16**3


def test_scoped_surface_witness_sweeps_the_authoritative_cube_to_occupied_center() -> None:
    """Boundary/corner occupancy must not fall back to a center-point trace."""

    oracle = AnalyticOracle(lambda cell: cell[0] < 0, (0.6, 0.0, 0.8))
    result = _scoped(
        [SurfaceCandidateGroup((-1, 0, 0), ((-1, 0, 0),))],
        [(-1, 0, 0)],
        oracle,
    )
    assert result.is_exact and result.value is not None and result.value.accepted
    witness_requests = [
        request for request in oracle.requests if request["start"] != request["end"]
    ]
    assert len(witness_requests) == 1
    assert witness_requests[0]["start"] == [2.0, 2.0, 2.0]
    assert witness_requests[0]["end"] == [-2.0, 2.0, 2.0]
    assert witness_requests[0]["mins"] == [-2.0, -2.0, -2.0]
    assert witness_requests[0]["maxs"] == [2.0, 2.0, 2.0]
    # The sloped normal returned by the separating swept-volume hit remains
    # exact classification evidence.
    assert result.value.chunks[0].cells[0].classifications == (SurfaceClass.FLOOR,)


def test_scoped_overlapping_candidates_and_groups_deduplicate_requests() -> None:
    oracle = AnalyticOracle(lambda cell: cell[0] < 0, (1.0, 0.0, 0.0))
    result = _scoped(
        [
            SurfaceCandidateGroup((-1, 0, 0), ((-1, 0, 0), (-1, 0, 0))),
            SurfaceCandidateGroup((-1, 0, 0), ((-1, 1, 0), (-1, 0, 0))),
        ],
        [(-1, 0, 0)],
        oracle,
    )
    assert result.is_exact and result.value is not None and result.value.accepted
    assert result.value.request_counts.surface == 2
    assert result.value.request_counts.occupancy == len(oracle.occupancy_cells)
    assert len(oracle.occupancy_cells) == len(set(oracle.occupancy_cells))
    surface_ids = [
        request["id"]
        for request in oracle.requests
        if request["start"] != request["end"]
    ]
    assert len(surface_ids) == len(set(surface_ids)) == 2


def test_scoped_candidate_outside_authorized_chunks_rejects_before_oracle() -> None:
    oracle = NeverCalledOracle()
    result = _scoped(
        [SurfaceCandidateGroup((1, 0, 0), ((16, 0, 0),))],
        [(0, 0, 0)],
        oracle,
    )
    assert result.is_exact and result.value is not None
    assert not result.value.accepted
    assert "outside the authorized chunks" in result.value.rejection
    assert result.value.request_counts.total == 0
    assert oracle.calls == 0


def test_scoped_cumulative_budget_rejection_precedes_oracle() -> None:
    oracle = NeverCalledOracle()
    result = _scoped(
        [SurfaceCandidateGroup((0, 0, 0), ((0, 0, 0),))],
        [(0, 0, 0), (1, 0, 0)],
        oracle,
        budget_state=L0BudgetState(max_chunks=1),
    )
    assert result.is_exact and result.value is not None
    assert not result.value.accepted
    assert "chunk count 2 > 1" in result.value.rejection
    assert result.value.request_counts.total == 0
    assert oracle.calls == 0
