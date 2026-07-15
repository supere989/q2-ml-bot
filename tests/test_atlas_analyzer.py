from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
from types import SimpleNamespace

import pytest
import zstandard

from harness.atlas_analyzer import (
    AtlasAnalysisError,
    AnalyzerLimits,
    CONTENTS_LAVA,
    CONTENTS_PLAYERCLIP,
    FROZEN_L0_BIT_PLANE_NAMES,
    FROZEN_L0_SCALAR_PLANE_NAMES,
    NavNode,
    _MoverDependencyIndex,
    _apply_stock_drop_hazards,
    _add_exact_platform_navigation,
    _atlas_channels,
    _analyze_hook_claims,
    _authored_item_destinations,
    _build_navigation,
    _complete_pmove_source_set,
    _drop_settle_request,
    _dynamic_mover_dependency_index,
    _exact_landing_key,
    _ground_navigation_seeds,
    _l0_chunks,
    _movement_edge_cost_q8,
    _movement_edge_kind,
    _movement_edge_stance,
    _movement_requests_for_source,
    _normalized_analysis_manifest,
    _process_tree_rss_bytes,
    _run_measured_process,
    _surface_candidate_scope,
    _surface_request_upper_bound,
    _supported_floor_candidate,
    analyze_map,
    sha256_file,
)
from harness.atlas_entity_semantics import Aabb
from harness.ibsp38 import EntityMetadata


ROOT = Path(__file__).resolve().parents[1]
CLIENT = Path("/home/raymondj/multires-worktrees/integration/q2-ml-client")
LITHIUM = Path("/home/raymondj/multires-worktrees/integration/q2-lithium-3zb2")
ATTESTATION = Path("/home/raymondj/multires-artifacts/atlas-v1/B1/hook-parity-pullspeed-1700.json")
FIXTURE_MODULE = CLIENT / "src/tools/oracle/tests/bsp_fixture.py"


def _fixture_writer():
    spec = importlib.util.spec_from_file_location("b1_oracle_fixture", FIXTURE_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_bsp


def _replace_entities(path: Path, entities: bytes) -> None:
    # Keep every collision lump at the fixture writer's original offset. The
    # compact oracle fixture intentionally has overlapping synthetic BSP node
    # bounds, and relocating those lumps exposes an unrelated legacy CM trace
    # defect. BSP lumps need not be ordered, so append the entity lump and
    # update only directory entry zero.
    data = bytearray(path.read_bytes())
    padding = (-len(data)) & 3
    data.extend(b"\0" * padding)
    entity_offset = len(data)
    data.extend(entities)
    struct.pack_into("<2i", data, 8, entity_offset, len(entities))
    path.write_bytes(data)


class _Q2dm5LowCeilingSeedCm:
    """Reproduce the two legal q2dm5 spawns rejected by a raised hull."""

    player_floor = {
        (-96.0, 448.0, 57.0): 48.03125,
        (-592.0, -72.0, 289.0): 280.03125,
    }

    def __init__(self) -> None:
        self.seen = []

    def call(self, requests):
        self.seen.extend(requests)
        output = []
        for request in requests:
            point = tuple(float(value) for value in request["start"])
            if request["id"].startswith("seed-spawn-clear:"):
                output.append({
                    "startsolid": False, "allsolid": False, "fraction": 1,
                    "endpos": list(request["end"]),
                    "plane": {"normal": [0.0, 0.0, 0.0]},
                })
                continue
            floor = self.player_floor[point]
            output.append({
                "startsolid": False, "allsolid": False, "fraction": 0.1,
                "endpos": [point[0], point[1], floor],
                "plane": {"normal": [0.0, 0.0, 1.0]},
                "surface": {"name": "e2u3/floor1_6", "flags": 0},
            })
        return output


def test_q2dm5_low_ceiling_spawns_ground_from_engine_origin() -> None:
    cm = _Q2dm5LowCeilingSeedCm()
    seeds = [
        (143, (-96.0, 448.0, 57.0), True),
        (145, (-592.0, -72.0, 289.0), True),
    ]
    grounded, _ = _ground_navigation_seeds(cm, seeds)
    assert grounded == [
        (-96.0, 448.0, 48.03125),
        (-592.0, -72.0, 280.03125),
    ]
    floor_requests = [
        request for request in cm.seen
        if request["id"].startswith("seed-floor:")
    ]
    assert [request["start"] for request in floor_requests] == [
        [-96.0, 448.0, 57.0], [-592.0, -72.0, 289.0],
    ]


def test_q2dm5_supported_drop_floor_is_retained_without_inferred_edge() -> None:
    source = (106, 78, 35)
    candidate = _supported_floor_candidate(source, {
        "startsolid": False, "fraction": 0.64,
        "endpos": [-88.0, 504.0, 24.03125],
        "plane": {"normal": [0.0, 0.0, 1.0]},
    }, (-1792, -768, -512))
    assert candidate == (
        (-88.0, 504.0, 24.03125), (106, 79, 33),
    )
    assert abs(candidate[1][2] - source[2]) == 2


def test_authored_item_destinations_include_q2_point_item_classes() -> None:
    metadata = SimpleNamespace(entities=(
        EntityMetadata(1, "item_health", (("origin", "1 2 3"),)),
        EntityMetadata(2, "weapon_railgun", (("origin", "4 5 6"),)),
        EntityMetadata(3, "ammo_slugs", (("origin", "7 8 9"),)),
        EntityMetadata(4, "light", (("origin", "10 11 12"),)),
    ))
    assert _authored_item_destinations(metadata) == [
        (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0),
    ]


def _required_artifacts() -> tuple[Path, Path, Path, Path, Path]:
    cm = CLIENT / "release/q2-cm-oracle"
    pmove = CLIENT / "release/q2-pmove-oracle"
    hook = LITHIUM / "tools/q2-hook-oracle"
    fall = LITHIUM / "tools/q2-fall-oracle"
    missing = [path for path in (cm, pmove, hook, fall, ATTESTATION, FIXTURE_MODULE) if not path.is_file()]
    if missing:
        pytest.skip("B1 integration artifacts unavailable: " + ", ".join(map(str, missing)))
    packer = ROOT / "target/debug/q2-atlas-pack"
    subprocess.run(
        [
            "cargo", "build", "-p", "q2-lattice",
            "--bin", "q2-atlas-pack", "--bin", "q2-atlas-verify",
        ],
        cwd=ROOT, check=True,
    )
    return cm, pmove, hook, fall, packer


def test_synthetic_fixture_cold_rebuilds_all_artifacts(tmp_path: Path) -> None:
    cm, pmove, hook, fall, packer = _required_artifacts()
    bsp = tmp_path / "atlas-fixture.bsp"
    _fixture_writer()(bsp, brushes=[((-128, -128, -128), (128, 128, 0), 1)])
    _replace_entities(
        bsp,
        b'{"classname" "worldspawn"}\n'
        b'{"classname" "info_player_deathmatch" "origin" "-48 0 16"}\n'
        b'{"classname" "info_player_deathmatch" "origin" "48 0 16"}\n\n\0',
    )
    provenance = {
        "canonical_id": "atlas-fixture", "bsp_sha256": sha256_file(bsp),
        "archive_sha256": "1" * 64, "bsp_member": "maps/atlas-fixture.bsp",
        "aliases": [], "source_url": None, "manual_origin": "synthetic test",
        "author": "test", "license_name": "test fixture",
        "license_evidence": "generated in test", "redistribution": "redistributable",
    }
    limits = AnalyzerLimits(
        max_l1_nodes=2_000, max_l1_edges=10_000, max_pmove_sources=128,
    )
    candidate = analyze_map(
        bsp, tmp_path / "candidate", "atlas-fixture", provenance,
        cm_oracle=cm, pmove_oracle=pmove, hook_oracle=hook,
        fall_oracle=fall,
        hook_attestation=ATTESTATION, packer=packer, limits=limits,
        independent_cold=False,
    )
    assert candidate["status"] == "candidate"
    assert candidate["deterministic_rebuild"] is False
    assert candidate["confidence"] == "pending-independent-cold-rebuild"
    assert candidate["analyzer_version"] == "b2-a-v2"
    assert candidate["confidence_summary"]["hook"] == "attested-no-replayed-edge"
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifest = analyze_map(
            bsp, output, "atlas-fixture", provenance,
            cm_oracle=cm, pmove_oracle=pmove, hook_oracle=hook,
            fall_oracle=fall,
            hook_attestation=ATTESTATION, packer=packer, limits=limits,
        )
        assert manifest["status"] == "passed"
        assert manifest["deterministic_rebuild"] is True
        assert manifest["counts"]["l1_nodes"] > 1
        assert manifest["counts"]["l1_edges"] > 0
        assert manifest["oracles"]["collision"]["admitted"] is True
        assert manifest["oracles"]["pmove"]["admitted"] is True
        assert manifest["oracles"]["hook"]["authority_admitted"] is True
        outputs.append(output)
    first = sorted(path.name for path in outputs[0].iterdir())
    second = sorted(path.name for path in outputs[1].iterdir())
    assert first == second
    for name in first:
        left = (outputs[0] / name).read_bytes()
        right = (outputs[1] / name).read_bytes()
        if name == "atlas-fixture.analysis.manifest.json":
            left_manifest = json.loads(left)
            right_manifest = json.loads(right)
            for manifest in (left_manifest, right_manifest):
                atlas = manifest["artifacts"]["atlas"]
                measured = atlas.pop("build_peak_rss_bytes")
                assert 0 < measured <= 512 * 1024 * 1024
                assert atlas["build_peak_rss_measurement"] == "linux_proc_self_status_vmhwm"
                assert atlas["build_peak_rss_gate_passed"] is True
                assert atlas["max_build_rss_bytes"] == 512 * 1024 * 1024
                manifest["identity"].pop("atlas_manifest_sha256")
                atlas_manifest = manifest["artifacts"]["atlas_manifest"]
                atlas_manifest.pop("sha256")
                atlas_manifest["verification"].pop("manifest_sha256")
                proof = manifest["performance"]["full_cold_rebuild"]
                assert proof["artifact_count"] == 8
                assert ".analysis.manifest.json" in proof[
                    "artifact_semantic_sha256"
                ]
                assert (
                    0 < proof["sampled_process_tree_peak_rss_bytes"]
                    <= proof["peak_rss_limit_bytes"]
                )
                proof.pop("sampled_process_tree_peak_rss_bytes")
                assert proof["artifact_sha256"] == proof["cold_artifact_sha256"]
                assert (
                    proof["artifact_semantic_sha256"]
                    == proof["cold_artifact_semantic_sha256"]
                )
                assert 0 < proof.pop("elapsed_milliseconds") <= 300_000
                assert proof["timeout_limit_milliseconds"] == 300_000
                performance = manifest["performance"]
                assert 0 < performance.pop("primary_elapsed_milliseconds") <= 300_000
                assert performance["primary_timeout_limit_milliseconds"] == 300_000
            assert left_manifest == right_manifest, name
        elif name == "atlas-fixture.atlas.manifest.json":
            left_manifest = json.loads(left)
            right_manifest = json.loads(right)
            for manifest in (left_manifest, right_manifest):
                measured = manifest.pop("build_peak_rss_bytes")
                assert 0 < measured <= 512 * 1024 * 1024
            assert left_manifest == right_manifest, name
        else:
            assert left == right, name
    manifest = json.loads((outputs[0] / "atlas-fixture.analysis.manifest.json").read_bytes())
    assert manifest["artifacts"]["atlas"]["uncompressed_sha256"] == sha256_file(
        outputs[0] / "atlas-fixture.atlas.bin"
    )
    trajectory = manifest["compiled_world"]["pmove_source_accounting"][
        "trajectory_accounting"
    ]
    assert trajectory["requested"]["standing_ground"] == trajectory[
        "requested"
    ]["crouched_ground"]
    assert trajectory["requested_total"] > trajectory["batch_size"]
    assert trajectory["batch_count"] == (
        trajectory["requested_total"] + trajectory["batch_size"] - 1
    ) // trajectory["batch_size"]
    assert sum(trajectory["outcomes"].values()) == trajectory["requested_total"]
    assert sum(trajectory["emitted"].values()) == trajectory["emitted_total"]
    compressed_navigation = (
        outputs[0] / "atlas-fixture.navigation.bin.zst"
    ).read_bytes()
    raw_navigation = zstandard.ZstdDecompressor().decompress(
        compressed_navigation
    )
    assert raw_navigation[:8] == b"Q2NAV001"
    navigation = json.loads(raw_navigation[16:])
    assert not any(
        edge["edge_type"] == "step" for edge in navigation["edges"]
    )


def test_l0_without_cm_never_invents_floor_or_boundary_surfaces() -> None:
    nodes = {}
    for y in range(48):
        for x in range(48):
            contents = 0
            if (x, y) == (24, 24):
                contents = CONTENTS_LAVA
            elif (x, y) == (25, 24):
                contents = CONTENTS_PLAYERCLIP
            nodes[(x, y, 0)] = NavNode(
                (x, y, 0), (x * 16.0 + 8, y * 16.0 + 8, 24.0),
                True, True, True, contents, (0.0, 0.0, 1.0),
            )

    chunks = _l0_chunks(nodes, [(1, (400.0, 400.0, 24.0))], (0, 0, 0))
    by_key = {tuple(chunk["key"]): chunk["bits"] for chunk in chunks}

    # Exact L1 contents remain available, but support metadata alone cannot
    # invent a solid surface or surface-material plane.
    assert (4, 4, -1) not in by_key
    assert all("solid" not in planes for planes in by_key.values())
    assert any("lava" in planes for planes in by_key.values())
    assert any("playerclip" in planes for planes in by_key.values())
    assert any("spawn_column" in planes for planes in by_key.values())


def test_surface_scope_uses_exact_node_hull_points_and_bounded_overlap() -> None:
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (7.25, 9.5, 24.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
        (1, 0, 0): NavNode(
            (1, 0, 0), (23.25, 9.5, 24.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
    }
    scope = _surface_candidate_scope(nodes, (0, 0, 0))
    candidates = {cell for group in scope.groups for cell in group.cells}
    left_floor = (1, 2, -1)
    right_floor = (5, 2, -1)
    assert left_floor in candidates and right_floor in candidates
    # Adjacent L1 samples are four L0 cells apart; a depth-five occupied band
    # from either exact sample covers the intervening floor cells.
    assert right_floor[0] - left_floor[0] == 4 <= 5
    assert _surface_request_upper_bound(scope.groups) < 50_000
    assert scope.candidate_cells < 256


class _ExactFloorCm:
    def __init__(self) -> None:
        self.requests = 0
        self.seen = []
        self.limits = AnalyzerLimits(max_oracle_requests=100_000)

    def call(self, requests):
        self.requests += len(requests)
        self.seen.extend(requests)
        output = []
        for request in requests:
            stationary_surface_cube = (
                request.get("mins") == [-2.0, -2.0, -2.0]
                and request.get("maxs") == [2.0, 2.0, 2.0]
                and request.get("start") == request.get("end")
            )
            start = request.get("start", [0.0, 0.0, 0.0])
            end = request.get("end", start)
            occupied = stationary_surface_cube and float(start[2]) < 0.0
            surface_hit = (
                request.get("mins") == [-2.0, -2.0, -2.0]
                and request.get("maxs") == [2.0, 2.0, 2.0]
                and start != end
                and float(start[2]) >= 0.0 and float(end[2]) < 0.0
            )
            record = {
                "ok": True, "id": request["id"], "op": request["op"],
                "startsolid": occupied, "allsolid": occupied,
                "fraction": 0.5 if surface_hit else (0.0 if occupied else 1.0),
                "endpos": list(end), "contents": 0,
                "plane": {"normal": [0.0, 0.0, 1.0], "dist": 0.0,
                          "type": 0, "signbits": 0},
                "surface": {"name": "floor", "flags": 0, "value": 0},
            }
            output.append(record)
        return output


def test_l0_model0_surfaces_require_and_preserve_exact_scoped_cm_evidence() -> None:
    node = NavNode(
        (0, 0, 0), (8.0, 8.0, 24.0), True, True, True, 0,
        (0.0, 0.0, 1.0),
    )
    cm = _ExactFloorCm()
    surfaces = {}
    chunks = _l0_chunks(
        {(0, 0, 0): node}, [], (0, 0, 0), cm=cm,
        surface_summary=surfaces,
    )
    assert any("solid" in chunk["bits"] for chunk in chunks)
    assert surfaces["model0_candidate_cells"] > 0
    assert surfaces["model0_occupancy_requests"] > 0
    assert surfaces["model0_surface_requests"] > 0
    assert surfaces["model0_physical_requests"] <= (
        surfaces["model0_occupancy_requests"] + surfaces["model0_surface_requests"]
    )
    assert surfaces["l0_accounted_chunks"] == len(chunks)


def test_inline_mover_uses_fixed_transformed_cm_and_unknown_dynamic_envelope() -> None:
    door = EntityMetadata(
        index=1,
        classname="func_door",
        properties=(("model", "*1"), ("angle", "0"), ("lip", "8")),
    )
    metadata = SimpleNamespace(
        entities=(door,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0), headnode=0),
            SimpleNamespace(
                mins=(16.0, -8.0, 0.0), maxs=(24.0, 8.0, 16.0),
                origin=(0.0, 0.0, 0.0), headnode=17,
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    node = NavNode(
        (0, 0, 0), (8.0, 8.0, 24.0), True, True, True, 0,
        (0.0, 0.0, 1.0),
    )
    cm = _ExactFloorCm()
    surfaces = {}
    semantics = {}
    chunks = _l0_chunks(
        {(0, 0, 0): node}, [], (0, 0, 0), cm=cm, metadata=metadata,
        surface_summary=surfaces, semantic_summary=semantics,
    )
    transformed = [request for request in cm.seen if request["op"] == "transformed_box_trace"]
    assert transformed
    assert all(request["headnode"] == 17 for request in transformed)
    assert all(request["origin"] == [0.0, 0.0, 0.0] for request in transformed)
    assert all(request["angles"] == [0.0, 0.0, 0.0] for request in transformed)
    assert all("model_index" not in request for request in transformed)
    planes = {name for chunk in chunks for name in chunk["bits"]}
    assert "mover_reference_solid" in planes
    assert {"mover_swept_envelope", "unknown"} <= planes
    assert semantics["mover_dynamic_unknown"] > 0
    assert surfaces["inline_fixed_pose_count"] == 1
    assert surfaces["inline_models"][0]["authority"] == "exact-fixed-transformed-cm"


def test_platform_surface_uses_exact_current_pose_and_full_unknown_envelope() -> None:
    platform = EntityMetadata(
        index=1,
        classname="func_plat",
        properties=(
            ("model", "*1"), ("origin", "0 0 0"),
            ("height", "16"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(platform,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0), headnode=0),
            SimpleNamespace(
                mins=(16.0, -8.0, 0.0), maxs=(24.0, 8.0, 16.0),
                origin=(0.0, 0.0, 0.0), headnode=17,
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    nodes = {
        (0, 0, -2): NavNode(
            (0, 0, -2), (8.0, 8.0, -8.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
        (0, 0, 0): NavNode(
            (0, 0, 0), (8.0, 8.0, 24.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
    }
    cm = _ExactFloorCm()
    surfaces = {}
    semantics = {}
    chunks = _l0_chunks(
        nodes, [], (0, 0, 0), cm=cm, metadata=metadata,
        surface_summary=surfaces, semantic_summary=semantics,
    )
    transformed = [
        request for request in cm.seen
        if request["op"] == "transformed_box_trace"
    ]
    assert transformed
    # Untargeted platforms start linked at their exact lower endpoint.
    assert all(request["origin"] == [0.0, 0.0, -16.0] for request in transformed)
    planes = {name for chunk in chunks for name in chunk["bits"]}
    assert "mover_reference_solid" in planes
    assert {"mover_swept_envelope", "unknown"} <= planes
    assert semantics["mover_dynamic_unknown"] > 0
    assert surfaces["inline_fixed_pose_count"] == 1
    assert surfaces["inline_models"][0]["authority"] == "exact-fixed-transformed-cm"


def test_train_marks_exact_ordinary_segment_sweep_unknown() -> None:
    train = EntityMetadata(
        index=1,
        classname="func_train",
        properties=(("model", "*1"), ("target", "A")),
    )
    path_a = EntityMetadata(
        index=2,
        classname="path_corner",
        properties=(("targetname", "A"), ("target", "B"), ("origin", "0 0 0")),
    )
    path_b = EntityMetadata(
        index=3,
        classname="path_corner",
        properties=(("targetname", "B"), ("origin", "192 0 0")),
    )
    metadata = SimpleNamespace(
        entities=(train, path_a, path_b),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0), headnode=0),
            SimpleNamespace(
                mins=(0.0, 0.0, 0.0), maxs=(8.0, 8.0, 8.0),
                origin=(0.0, 0.0, 0.0), headnode=17,
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    nodes = {
        (index, 0, 0): NavNode(
            (index, 0, 0), (x, 8.0, 24.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        )
        for index, x in ((0, 8.0), (6, 104.0), (12, 200.0))
    }
    semantics = {}
    chunks = _l0_chunks(
        nodes, [], (0, 0, 0), cm=_ExactFloorCm(), metadata=metadata,
        surface_summary={}, semantic_summary=semantics,
    )

    def marked(cell: tuple[int, int, int], plane: str) -> bool:
        chunk_key = tuple(value // 16 for value in cell)
        local = tuple(value % 16 for value in cell)
        linear = local[0] + 16 * local[1] + 256 * local[2]
        return any(
            tuple(chunk["key"]) == chunk_key
            and linear in chunk["bits"].get(plane, ())
            for chunk in chunks
        )

    assert marked((0, 0, 0), "mover_swept_envelope")
    assert marked((48, 0, 0), "mover_swept_envelope")
    assert marked((26, 0, 0), "mover_swept_envelope")
    assert semantics["mover_dynamic_unknown"] > 0


def test_python_l0_names_match_frozen_rust_packer_schema() -> None:
    source = (ROOT / "crates/q2-lattice/src/bin/q2_atlas_pack.rs").read_text()
    for name in FROZEN_L0_BIT_PLANE_NAMES:
        assert f'"{name}" => L0BitPlane::' in source
    for name in FROZEN_L0_SCALAR_PLANE_NAMES:
        assert f'"{name}" => L0ScalarPlane::' in source


def test_atlas_channels_enumerate_exact_frozen_l0_schema() -> None:
    l0 = [channel for channel in _atlas_channels() if channel["level"] == 0]
    bits = {channel["name"] for channel in l0 if channel["encoding"] == "bitplane"}
    scalars = {channel["name"] for channel in l0 if channel["encoding"] == "u8"}
    assert bits == FROZEN_L0_BIT_PLANE_NAMES
    assert scalars == FROZEN_L0_SCALAR_PLANE_NAMES
    assert len(l0) == len(bits) + len(scalars)
    assert all(channel["persistence"] == "map-static" for channel in l0)


@pytest.mark.parametrize(
    "spawnflags,state",
    [("0", "active"), ("1", "off"), ("2", "toggle"), ("3", "toggle")],
)
def test_l0_trigger_hurt_uses_runtime_linked_contact_bounds(
    spawnflags: str, state: str,
) -> None:
    trigger = EntityMetadata(
        index=1,
        classname="trigger_hurt",
        properties=(
            ("model", "*1"), ("origin", "0 0 0"),
            ("spawnflags", spawnflags),
        ),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(mins=(100, 0, 0), maxs=(164, 64, 16)),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    semantics = {}
    chunks = _l0_chunks(
        {}, [], (0, 0, 0), metadata=metadata, semantic_summary=semantics
    )
    counts = {}
    scalar_counts = {}
    for chunk in chunks:
        for name, cells in chunk["bits"].items():
            counts[name] = counts.get(name, 0) + len(cells)
        for name, cells in chunk["scalars"].items():
            scalar_counts[name] = scalar_counts.get(name, 0) + len(cells)
    if state == "toggle":
        assert counts == {"unknown": 13_520}
        assert scalar_counts == {}
        assert semantics == {"hurt_potential": 13_520}
    elif state == "active":
        assert counts["hurt"] == 1_944
        assert counts["standing_forbidden_origin"] == 13_520
        assert counts["crouched_forbidden_origin"] == 8_788
        assert scalar_counts["hazard_severity"] == 13_520
        assert semantics == {"hurt_expanded": 13_520}
    else:
        assert counts == {}
        assert scalar_counts == {}
        assert semantics == {}


def test_process_tree_rss_rejects_unreadable_root() -> None:
    with pytest.raises(AtlasAnalysisError, match="RSS is unreadable"):
        _process_tree_rss_bytes(2_147_483_647)


def test_measured_process_rejects_zero_peak_rss(monkeypatch) -> None:
    class _ZeroRssProcess:
        pid = 12345

        def __init__(self) -> None:
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _ZeroRssProcess())
    monkeypatch.setattr(
        "harness.atlas_analyzer._process_tree_rss_bytes", lambda _pid: 0
    )
    with pytest.raises(AtlasAnalysisError, match="positive RSS sample"):
        _run_measured_process(["fake"], timeout=1.0)


def test_measured_process_kills_process_group_on_sampler_failure(monkeypatch) -> None:
    captured = {}

    class _UnreadableProcess:
        pid = 23456

        def poll(self):
            return None

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(*args, **kwargs):
        captured["start_new_session"] = kwargs.get("start_new_session")
        return _UnreadableProcess()

    def fail_sample(_pid):
        raise AtlasAnalysisError("independent cold analyzer /proc RSS is unreadable")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("harness.atlas_analyzer._process_tree_rss_bytes", fail_sample)
    monkeypatch.setattr(
        os, "killpg", lambda pid, sig: captured.update(pid=pid, sig=sig)
    )
    with pytest.raises(AtlasAnalysisError, match="RSS is unreadable"):
        _run_measured_process(["fake"], timeout=1.0)
    assert captured == {
        "start_new_session": True, "pid": 23456, "sig": signal.SIGKILL,
    }


def _test_node(
    key: tuple[int, int, int], *, standing: bool, crouched: bool,
) -> NavNode:
    return NavNode(
        key, tuple(float(axis * 16) for axis in key), standing, crouched,
        True, 0, (0.0, 0.0, 1.0),
    )


def _walk_edges(
    keys: set[tuple[int, int, int]],
) -> list[dict[str, object]]:
    return [
        {
            "source": list(source),
            "target": list(target),
            "edge_type": "walk",
        }
        for source in sorted(keys)
        for target in (
            (source[0] + 1, source[1], source[2]),
            (source[0] - 1, source[1], source[2]),
            (source[0], source[1] + 1, source[2]),
            (source[0], source[1] - 1, source[2]),
        )
        if target in keys
    ]


def test_pmove_source_cap_has_complete_omission_accounting_before_oracle_calls() -> None:
    nodes = {
        (0, 0, 0): _test_node((0, 0, 0), standing=True, crouched=True),
        (1, 0, 0): _test_node((1, 0, 0), standing=True, crouched=True),
    }
    calls = {"pmove": 0, "fall": 0}
    selected, accounting = _complete_pmove_source_set(
        nodes, _walk_edges(set(nodes)), {1: (0, 0, 0)}, 1,
    )
    assert selected == [(0, 0, 0)]
    assert accounting["total"] == 2
    assert accounting["selected"] == 1
    assert accounting["omitted"] == 1
    assert accounting["omitted_sources_emit_pmove_edges"] is False
    assert accounting["selection_rule"] == (
        "spawn-first-component-frontier-proportional-farthest-v2"
    )
    assert calls == {"pmove": 0, "fall": 0}


def test_pmove_source_cap_fails_before_partial_component_coverage() -> None:
    keys = {(0, 0, 0), (100, 0, 0), (200, 0, 0)}
    nodes = {
        key: _test_node(key, standing=True, crouched=True) for key in keys
    }
    with pytest.raises(
        AtlasAnalysisError,
        match="cannot retain mandatory component-frontier coverage",
    ):
        _complete_pmove_source_set(nodes, [], {1: (0, 0, 0)}, 2)


def test_pmove_source_cap_fails_before_omitting_a_unique_spawn() -> None:
    keys = {(0, 0, 0), (1, 0, 0)}
    nodes = {
        key: _test_node(key, standing=True, crouched=True) for key in keys
    }
    with pytest.raises(
        AtlasAnalysisError, match="cannot retain every unique spawn source",
    ):
        _complete_pmove_source_set(
            nodes, _walk_edges(keys), {1: (0, 0, 0), 2: (1, 0, 0)}, 1,
        )


def test_pmove_sources_cover_remote_spawn_component_frontiers_before_secondaries() -> None:
    # Model the q2dm1 failure mode: a large, low-coordinate component would
    # consume a z/y/x prefix before the smaller components containing spawn
    # ordinals 61 and 69 were reached.  All three spawns are interior nodes, so
    # each component still needs a distinct boundary source after the spawn
    # tier.
    large = {(x, y, 0) for x in range(9) for y in range(9)}
    spawn_61 = (104, 104, 8)
    spawn_69 = (204, -96, 16)
    remote_61 = {
        spawn_61,
        (103, 104, 8), (105, 104, 8), (104, 103, 8), (104, 105, 8),
    }
    remote_69 = {
        spawn_69,
        (203, -96, 16), (205, -96, 16),
        (204, -97, 16), (204, -95, 16),
    }
    keys = large | remote_61 | remote_69
    nodes = {
        key: _test_node(key, standing=True, crouched=True) for key in keys
    }
    edges = _walk_edges(keys)
    spawns = {1: (4, 4, 0), 61: spawn_61, 69: spawn_69}

    selected, accounting = _complete_pmove_source_set(nodes, edges, spawns, 6)

    assert selected[:3] == [spawns[1], spawns[61], spawns[69]]
    assert len(selected) == 6
    component_frontiers = selected[3:]
    assert sum(key in large for key in component_frontiers) == 1
    assert sum(key in remote_61 for key in component_frontiers) == 1
    assert sum(key in remote_69 for key in component_frontiers) == 1
    assert accounting["selected"] == 6
    assert accounting["total"] == 43
    assert accounting["omitted"] == 37


def test_pmove_sources_spatially_distribute_large_component_boundary() -> None:
    keys = {(x, y, 0) for x in range(9) for y in range(9)}
    nodes = {
        key: _test_node(key, standing=True, crouched=True) for key in keys
    }
    selected, _ = _complete_pmove_source_set(
        nodes, _walk_edges(keys), {1: (4, 4, 0)}, 5,
    )

    assert selected[0] == (4, 4, 0)
    assert {(key[0], key[1]) for key in selected[1:]} == {
        (0, 0), (8, 0), (0, 8), (8, 8),
    }


def test_pmove_source_selection_is_independent_of_mapping_and_edge_order() -> None:
    low = {(x, y, 0) for x in range(7) for y in range(7)}
    high = {(x, y, 12) for x in range(20, 25) for y in range(-2, 3)}
    keys = low | high
    nodes = {
        key: _test_node(key, standing=True, crouched=True) for key in keys
    }
    edges = _walk_edges(keys)
    spawns = {69: (22, 0, 12), 1: (3, 3, 0)}

    expected = _complete_pmove_source_set(nodes, edges, spawns, 17)
    reversed_nodes = dict(reversed(list(nodes.items())))
    reversed_spawns = dict(reversed(list(spawns.items())))
    actual = _complete_pmove_source_set(
        reversed_nodes, list(reversed(edges)), reversed_spawns, 17,
    )

    assert actual == expected


def test_crouched_only_source_cannot_emit_standing_movement_edge() -> None:
    source = _test_node((0, 0, 0), standing=False, crouched=True)
    target = _test_node((1, 0, 0), standing=True, crouched=True)
    assert _movement_edge_stance(source, target, "crouched") == "crouched"


def test_standing_replay_cannot_invent_crouched_target_stance() -> None:
    source = _test_node((0, 0, 0), standing=True, crouched=True)
    target = _test_node((1, 0, 0), standing=False, crouched=True)
    assert _movement_edge_stance(source, target, "standing") is None


def test_movement_edge_kind_does_not_turn_flat_ground_into_step() -> None:
    assert _movement_edge_kind(
        "ground", any_airborne=False, vertical=0.0,
    ) is None
    assert _movement_edge_kind(
        "ground", any_airborne=False, vertical=16.0,
    ) == "step"
    assert _movement_edge_kind(
        "ground", any_airborne=True, vertical=0.0,
    ) == "controlled_drop"
    assert _movement_edge_kind(
        "jump", any_airborne=True, vertical=0.0,
    ) == "jump"


def test_movement_edge_cost_uses_exact_path_through_first_landing() -> None:
    request = {"origin": [0.0, 0.0, 0.0]}
    response = {"frames": [
        {"command_index": 0, "origin": [16.0, 0.0, 0.0]},
        {"command_index": 1, "origin": [16.0, 16.0, 0.0]},
        {"command_index": 2, "origin": [48.0, 16.0, 0.0]},
    ]}
    assert _movement_edge_cost_q8(
        request, response, landing_command_index=1,
    ) == 32 * 256
    assert _movement_edge_cost_q8(
        request, response, landing_command_index=None,
    ) == 64 * 256


def test_dual_clear_source_builds_canonical_stance_complete_requests() -> None:
    source = _test_node((1, 2, 3), standing=True, crouched=True)
    records = _movement_requests_for_source(
        (1, 2, 3), source, outgoing=0, is_spawn=False,
        parameters={"gravity": 800, "airaccelerate": 0},
    )
    assert len(records) == 12
    assert [record[1:4] for record in records].count(
        ("ground", 0, "standing")
    ) == 1
    ids = [record[4]["id"] for record in records]
    assert "drop:1:2:3:ground:0:pmove" in ids
    assert "drop:1:2:3:ground:0:crouched:pmove" in ids
    crouched = [record for record in records if record[3] == "crouched"]
    assert len(crouched) == 4
    assert all(record[4]["pm_flags"] == 5 for record in crouched)
    assert all(
        command["upmove"] == -300
        for record in crouched for command in record[4]["commands"]
    )
    jumps = [record for record in records if record[1] == "jump"]
    assert len(jumps) == 4
    assert all(len(record[4]["commands"]) == 16 for record in jumps)


def test_exact_edge_target_uses_first_landing_not_later_final_state() -> None:
    record = {
        "classification": {
            "classification": "Exact",
            "lethal": False,
            "landing": {"origin": [16.0, 0.0, 0.0], "command_index": 2},
        },
        "final": {"origin": [160.0, 0.0, -64.0]},
    }
    assert _exact_landing_key(record, (0, 0, 0)) == (1, 0, 0)


def test_airborne_probe_gets_one_continuous_neutral_settle_replay() -> None:
    request = {
        "id": "drop:1:2:3:ground:90:pmove", "op": "simulate",
        "commands": [
            {"msec": 50, "angles": [0, 90, 0], "forwardmove": 300}
            for _ in range(4)
        ],
    }
    response = {"frames": [
        {"grounded": True}, {"grounded": True},
        {"grounded": False}, {"grounded": False},
    ]}
    extended = _drop_settle_request(request, response, horizon_frames=8)
    assert extended is not None
    assert extended["id"] == request["id"]
    assert extended["commands"][:4] == request["commands"]
    assert extended["commands"][4:] == [
        {"msec": 50, "angles": [0, 90, 0]} for _ in range(4)
    ]
    assert request["commands"][-1]["forwardmove"] == 300


def test_landed_or_grounded_probe_does_not_get_settle_replay() -> None:
    request = {
        "id": "drop:1:2:3:ground:0:pmove", "op": "simulate",
        "commands": [{"msec": 50, "angles": [0, 0, 0]} for _ in range(4)],
    }
    assert _drop_settle_request(
        request, {"frames": [{"grounded": True} for _ in range(4)]},
    ) is None
    assert _drop_settle_request(request, {"frames": [
        {"grounded": True}, {"grounded": False},
        {"grounded": False}, {"grounded": True},
    ]}) is None


def test_dynamic_mover_swept_path_dependence_is_detected() -> None:
    dependency = _MoverDependencyIndex((
        Aabb((20.0, -4.0, -24.0), (24.0, 4.0, 32.0)),
    ))
    request = {"origin": [0.0, 0.0, 0.0]}
    response = {"frames": [{
        "origin": [32.0, 0.0, 0.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert dependency.intersects_trajectory(response, request)


@pytest.mark.parametrize("classname", ["func_wall", "func_object", "func_explosive"])
def test_inline_dynamic_classes_poison_intersecting_pmove(
    classname: str,
) -> None:
    entity = EntityMetadata(
        1, classname, (("classname", classname), ("model", "*1")),
    )
    metadata = SimpleNamespace(
        entities=(entity,),
        models=(
            SimpleNamespace(mins=(-1024.0,) * 3, maxs=(1024.0,) * 3),
            SimpleNamespace(mins=(-8.0,) * 3, maxs=(8.0,) * 3),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    dependency = _dynamic_mover_dependency_index(metadata)
    request = {"origin": [0.0, 0.0, 0.0]}
    response = {"frames": [{
        "origin": [0.0, 0.0, 0.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert dependency.intersects_trajectory(response, request)


def test_func_plat_localizes_unknown_to_exact_vertical_swept_envelope() -> None:
    entity = EntityMetadata(
        1, "func_plat", (
            ("classname", "func_plat"), ("model", "*1"),
            ("origin", "100 0 80"), ("height", "40"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(entity,),
        models=(
            SimpleNamespace(mins=(-1024.0,) * 3, maxs=(1024.0,) * 3),
            SimpleNamespace(mins=(-8.0, -8.0, -4.0), maxs=(8.0, 8.0, 4.0)),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    dependency = _dynamic_mover_dependency_index(metadata)
    assert not dependency.globally_unknown

    remote = {"frames": [{
        "origin": [0.0, 0.0, 0.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert not dependency.intersects_trajectory(remote, {"origin": [0.0, 0.0, 0.0]})

    through_platform = {"frames": [{
        "origin": [100.0, 0.0, 50.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert dependency.intersects_trajectory(
        through_platform, {"origin": [100.0, 0.0, 50.0]},
    )


def test_q2dm6_rotating_door_dependency_uses_tight_axis_sweep() -> None:
    door = EntityMetadata(
        57, "func_door_rotating", (
            ("classname", "func_door_rotating"), ("model", "*1"),
            ("origin", "0 -556 -252"), ("targetname", "t9"),
            ("spawnflags", "64"), ("speed", "30"), ("distance", "-32"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(door,),
        models=(
            SimpleNamespace(mins=(-1024.0,) * 3, maxs=(1024.0,) * 3),
            # q2dm6 model *2 raw dmodel bounds; the dependency builder applies
            # the collision loader's one-unit expansion.
            SimpleNamespace(
                mins=(-48.0, -4.0, -11.0), maxs=(48.0, 220.0, 131.0),
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 57, "model_index": 1},
        )),
    )
    dependency = _dynamic_mover_dependency_index(metadata)
    assert not dependency.globally_unknown

    # The standing hull at the isolated route's x=-72 column ends at -56;
    # the exact X-axis sweep starts at x=-49, so mover state is irrelevant.
    left_route = {"frames": [{
        "origin": [-72.0, -584.0, -232.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert not dependency.intersects_trajectory(
        left_route, {"origin": [-72.0, -600.0, -232.0]},
    )

    overlapping = {"frames": [{
        "origin": [-48.0, -560.0, -232.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert dependency.intersects_trajectory(
        overlapping, {"origin": [-48.0, -576.0, -232.0]},
    )


def test_continuous_rotator_full_cycle_localizes_unknown() -> None:
    rotating = EntityMetadata(
        1, "func_rotating", (
            ("classname", "func_rotating"), ("model", "*1"),
            ("origin", "100 0 0"), ("spawnflags", "4"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(rotating,),
        models=(
            SimpleNamespace(mins=(-1024.0,) * 3, maxs=(1024.0,) * 3),
            SimpleNamespace(mins=(-8.0,) * 3, maxs=(8.0,) * 3),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    dependency = _dynamic_mover_dependency_index(metadata)
    assert not dependency.globally_unknown
    remote = {"frames": [{
        "origin": [0.0, 0.0, 0.0],
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
    }]}
    assert not dependency.intersects_trajectory(remote, {"origin": [0.0, 0.0, 0.0]})


class _ExactPlatformNavigationCm:
    def __init__(self) -> None:
        self.requests = 0
        self.limits = AnalyzerLimits(max_oracle_requests=10_000)

    def call(self, requests):
        self.requests += len(requests)
        output = []
        for request in requests:
            if request["op"] == "point_contents":
                output.append({
                    "ok": True, "id": request["id"], "op": request["op"],
                    "contents": 0,
                })
                continue
            support = request["id"].startswith("platform-support:")
            if support:
                pose_z = float(request["origin"][2])
                endpos = [32.0, 32.0, 40.0 + pose_z]
                fraction = 0.5
                normal = [0.0, 0.0, 1.0]
                surface = {"name": "platform-top", "flags": 0, "value": 0}
                contents = 1
            else:
                endpos = list(request["end"])
                fraction = 1.0
                normal = [0.0, 0.0, 0.0]
                surface = {"name": "", "flags": 0, "value": 0}
                contents = 0
            output.append({
                "ok": True, "id": request["id"], "op": request["op"],
                "startsolid": False, "allsolid": False,
                "fraction": fraction, "endpos": endpos, "contents": contents,
                "plane": {"normal": normal, "dist": 0.0, "type": 0, "signbits": 0},
                "surface": surface,
            })
        return output


class _AliasedPlatformEndpointCm(_ExactPlatformNavigationCm):
    def call(self, requests):
        self.requests += len(requests)
        output = []
        for request in requests:
            if request["op"] == "point_contents":
                output.append({
                    "ok": True, "id": request["id"], "op": request["op"],
                    "contents": 0,
                })
                continue
            support = request["id"].startswith("platform-support:")
            if support:
                z = 14.0 if float(request["origin"][2]) < 0 else 46.0
                endpos = [32.0, 32.0, z]
                fraction = 0.5
                normal = [0.0, 0.0, 1.0]
                surface = {"name": "platform-top", "flags": 0, "value": 0}
                contents = 1
            else:
                endpos = list(request["end"])
                fraction = 1.0
                normal = [0.0, 0.0, 0.0]
                surface = {"name": "", "flags": 0, "value": 0}
                contents = 0
            output.append({
                "ok": True, "id": request["id"], "op": request["op"],
                "startsolid": False, "allsolid": False,
                "fraction": fraction, "endpos": endpos, "contents": contents,
                "plane": {"normal": normal, "dist": 0.0, "type": 0, "signbits": 0},
                "surface": surface,
            })
        self.seen.extend(requests)
        return output


def test_exact_platform_adds_stateful_boarding_and_endpoint_edges() -> None:
    platform = EntityMetadata(
        1, "func_plat", (
            ("classname", "func_plat"), ("model", "*1"),
            ("height", "32"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(platform,),
        models=(
            SimpleNamespace(mins=(-128.0,) * 3, maxs=(128.0,) * 3, headnode=0),
            SimpleNamespace(
                mins=(0.0, 0.0, 0.0), maxs=(64.0, 64.0, 16.0), headnode=17,
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    bottom_static = (5, 2, 0)
    top_static = (5, 2, 2)
    nodes = {
        bottom_static: NavNode(
            bottom_static, (80.0, 32.0, 8.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
        top_static: NavNode(
            top_static, (80.0, 32.0, 40.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
    }
    edges = []
    edge_keys = set()
    _add_exact_platform_navigation(
        _ExactPlatformNavigationCm(), metadata, nodes, edges, edge_keys,
        (0, 0, 0),
    )
    assert {(2, 2, 0), (2, 2, 2)} <= set(nodes)
    assert len(edges) == 6
    assert all(edge["edge_type"] == "mover" for edge in edges)
    assert all(edge["blocker"] == 1 for edge in edges)
    assert all(edge["evidence"] == 1 for edge in edges)
    assert all(edge["validation_version"] == 1 for edge in edges)
    assert all(edge["confidence"] == 32768 for edge in edges)
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(tuple(edge["source"]), set()).add(tuple(edge["target"]))
    visited = {bottom_static}
    pending = [bottom_static]
    while pending:
        for target in adjacency.get(pending.pop(), ()):
            if target not in visited:
                visited.add(target)
                pending.append(target)
    assert top_static in visited


def test_platform_connector_preserves_aliased_exact_origin_and_geometry_radius() -> None:
    platform = EntityMetadata(
        1, "func_plat", (
            ("classname", "func_plat"), ("model", "*1"), ("height", "32"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(platform,),
        models=(
            SimpleNamespace(mins=(-128.0,) * 3, maxs=(128.0,) * 3, headnode=0),
            SimpleNamespace(
                mins=(0.0, 0.0, 0.0), maxs=(64.0, 64.0, 16.0), headnode=17,
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    # Each exact mover endpoint aliases an existing L1 cell whose model-0
    # origin differs by six units. The first valid connector is 80 units from
    # the platform center: outside the former arbitrary 64-unit search.
    nodes = {
        (2, 2, 0): NavNode(
            (2, 2, 0), (32.0, 32.0, 8.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
        (2, 2, 2): NavNode(
            (2, 2, 2), (32.0, 32.0, 40.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
        (7, 2, 0): NavNode(
            (7, 2, 0), (112.0, 32.0, 14.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
        (7, 2, 2): NavNode(
            (7, 2, 2), (112.0, 32.0, 46.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
    }
    cm = _AliasedPlatformEndpointCm()
    cm.seen = []
    edges = []
    _add_exact_platform_navigation(cm, metadata, nodes, edges, set(), (0, 0, 0))
    assert len(edges) == 6
    board_requests = [
        request for request in cm.seen if "platform-board" in request["id"]
    ]
    assert board_requests
    endpoints = {
        tuple(request[field])
        for request in board_requests for field in ("start", "end")
    }
    assert (32.0, 32.0, 14.0) in endpoints
    assert (32.0, 32.0, 46.0) in endpoints
    assert (32.0, 32.0, 8.0) not in endpoints
    assert (32.0, 32.0, 40.0) not in endpoints


def test_target_disabled_platform_does_not_claim_activation_topology() -> None:
    platform = EntityMetadata(
        1, "func_plat", (
            ("classname", "func_plat"), ("model", "*1"),
            ("height", "32"), ("targetname", "manual"),
        ),
    )
    metadata = SimpleNamespace(
        entities=(platform,),
        models=(
            SimpleNamespace(mins=(-128.0,) * 3, maxs=(128.0,) * 3, headnode=0),
            SimpleNamespace(
                mins=(0.0, 0.0, 0.0), maxs=(64.0, 64.0, 16.0), headnode=17,
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    nodes = {
        (5, 2, 0): NavNode(
            (5, 2, 0), (80.0, 32.0, 8.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
    }
    edges = []
    _add_exact_platform_navigation(
        _ExactPlatformNavigationCm(), metadata, nodes, edges, set(),
        (0, 0, 0),
    )
    assert set(nodes) == {(5, 2, 0)}
    assert edges == []


def test_navigation_rejects_missing_dynamic_mover_authority() -> None:
    with pytest.raises(AtlasAnalysisError, match="lacks dynamic-mover authority"):
        _build_navigation(
            SimpleNamespace(), None, None, Path("missing.bsp"), [],
            (0, 0, 0), AnalyzerLimits(), mover_dependencies=None,
        )


def test_cold_analysis_semantics_bind_authority_accounting_and_hazards() -> None:
    manifest = {
        "schema": "q2-atlas-analysis-v1",
        "status": "candidate",
        "deterministic_rebuild": False,
        "performance": {"primary_elapsed_milliseconds": 1},
        "identity": {"atlas_manifest_sha256": "1" * 64},
        "artifacts": {
            "atlas": {"build_peak_rss_bytes": 1024},
            "atlas_manifest": {
                "sha256": "2" * 64,
                "uncompressed_bytes": 999,
                "verification": {"manifest_sha256": "3" * 64},
            },
        },
        "compiled_world": {
            "pmove_source_accounting": {"selected": 7, "omitted": 3},
            "hazards": {"unknown_omitted": 5},
        },
    }
    operationally_different = json.loads(json.dumps(manifest))
    operationally_different["performance"]["primary_elapsed_milliseconds"] = 99
    operationally_different["artifacts"]["atlas"]["build_peak_rss_bytes"] = 4096
    operationally_different["identity"]["atlas_manifest_sha256"] = "4" * 64
    operationally_different["artifacts"]["atlas_manifest"]["sha256"] = "5" * 64
    # build_peak_rss_bytes is embedded in the Atlas manifest.  A different
    # RSS digit width changes the serialized byte count but not its normalized
    # semantics.
    operationally_different["artifacts"]["atlas_manifest"][
        "uncompressed_bytes"
    ] = 1000
    operationally_different["artifacts"]["atlas_manifest"]["verification"][
        "manifest_sha256"
    ] = "6" * 64
    assert _normalized_analysis_manifest(manifest) == _normalized_analysis_manifest(
        operationally_different
    )

    semantically_different = json.loads(json.dumps(operationally_different))
    semantically_different["compiled_world"]["pmove_source_accounting"][
        "omitted"
    ] = 4
    assert _normalized_analysis_manifest(manifest) != _normalized_analysis_manifest(
        semantically_different
    )


def test_unknown_and_lethal_solid_landings_do_not_invent_void_or_uncontained() -> None:
    hazards = {
        "types": ["lava"], "lethal_drop_edges": 0,
        "exact_lethal_candidates_omitted": 0,
        "uncontained_drop_edges": 0,
    }
    _apply_stock_drop_hazards(
        hazards, {"exact_lethal": 1, "unknown_omitted": 7},
    )
    assert hazards["lethal_drop_edges"] == 0
    assert hazards["exact_lethal_candidates_omitted"] == 1
    assert hazards["uncontained_drop_edges"] == 0
    assert "void" not in hazards["types"]
