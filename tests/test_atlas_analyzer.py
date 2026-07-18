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
    BUILD_PLAN_SCHEMA,
    CONTENTS_LADDER,
    CONTENTS_LAVA,
    CONTENTS_PLAYERCLIP,
    FROZEN_L0_BIT_PLANE_NAMES,
    FROZEN_L0_SCALAR_PLANE_NAMES,
    OBJECTIVE_OMISSION_BEYOND_FENCE,
    OBJECTIVE_OMISSION_NO_TARGET,
    OBJECTIVE_TARGET_MAX_DISTANCE,
    NavNode,
    OracleProcess,
    _MoverDependencyIndex,
    _apply_static_drop_hazards,
    _apply_stock_drop_hazards,
    _add_exact_platform_navigation,
    _atlas_channels,
    _analyze_hook_claims,
    _authored_item_destinations,
    _build_navigation,
    _candidate_floor_requests,
    _complete_pmove_source_set,
    _drop_settle_request,
    _dynamic_mover_dependency_index,
    _exact_landing_key,
    _ground_navigation_seeds,
    _claimed_hurt_boundary_chunks,
    _hurt_boundary_chunks,
    _l0_chunks,
    _ladder_contact_requests,
    _ladder_contact_trace_admits,
    _ladder_request_candidates_for_source,
    _movement_edge_cost_q8,
    _movement_edge_kind,
    _movement_edge_stance,
    _movement_requests_for_source,
    _normalized_analysis_manifest,
    _objective_artifact,
    _objective_guidepost_analysis,
    _process_tree_rss_bytes,
    _project_generated_hurt_chunks,
    _run_measured_process,
    _surface_candidate_scope,
    _surface_request_upper_bound,
    _supported_floor_candidate,
    analyze_map,
    canonical_json,
    sha256_file,
)
from harness.atlas_entity_semantics import Aabb, L0BudgetState
from harness.ibsp38 import EntityMetadata
from maps.generator import (
    KILL_PLANE_DROP,
    KILL_PLANE_MARGIN,
    WALL_T,
    MapGenerator,
)


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


class _RecordingOracleStdin:
    def __init__(self) -> None:
        self.batches: list[list[dict]] = []

    def write(self, payload: bytes) -> int:
        self.batches.append([
            json.loads(line) for line in payload.splitlines()
        ])
        return len(payload)

    def flush(self) -> None:
        pass


def _transport_only_oracle(kind: str) -> tuple[OracleProcess, _RecordingOracleStdin]:
    oracle = OracleProcess.__new__(OracleProcess)
    oracle.kind = kind
    oracle.limits = AnalyzerLimits(oracle_batch=32, max_oracle_requests=1_000)
    oracle.requests = 0
    stdin = _RecordingOracleStdin()
    oracle.process = SimpleNamespace(stdin=stdin)

    def read_lines(count: int) -> list[dict]:
        batch = stdin.batches[-1]
        assert len(batch) == count
        return [{
            "id": request["id"], "ok": True, "op": request["op"],
        } for request in batch]

    oracle._read_lines = read_lines
    return oracle, stdin


def _pmove_request(index: int, frame_count: int) -> dict:
    return {
        "id": f"pmove-{index}", "op": "simulate",
        "commands": [{"msec": 100} for _ in range(frame_count)],
    }


def test_multiframe_pmove_transport_uses_singleton_leaf_batches() -> None:
    oracle, stdin = _transport_only_oracle("pmove")
    requests = [_pmove_request(index, 52) for index in range(17)]

    responses = oracle.call(requests)

    assert [[request["id"] for request in batch] for batch in stdin.batches] == [
        [request["id"]] for request in requests
    ]
    assert [response["id"] for response in responses] == [
        request["id"] for request in requests
    ]
    assert oracle.requests == len(requests)


def test_mixed_pmove_transport_never_shares_a_multiframe_batch() -> None:
    oracle, stdin = _transport_only_oracle("pmove")
    requests = [
        _pmove_request(0, 1), _pmove_request(1, 1),
        _pmove_request(2, 40),
        _pmove_request(3, 1), _pmove_request(4, 52),
        _pmove_request(5, 1), _pmove_request(6, 1),
    ]

    oracle.call(requests)

    assert [[request["id"] for request in batch] for batch in stdin.batches] == [
        ["pmove-0", "pmove-1"], ["pmove-2"], ["pmove-3"],
        ["pmove-4"], ["pmove-5", "pmove-6"],
    ]

    leaf, leaf_stdin = _transport_only_oracle("pmove")
    with pytest.raises(
        AtlasAnalysisError,
        match="multi-frame pmove request entered a shared transport batch",
    ):
        leaf._call_batch([_pmove_request(7, 1), _pmove_request(8, 40)])
    assert leaf_stdin.batches == []
    assert leaf.requests == 0


@pytest.mark.parametrize("kind", ["cm", "pmove"])
def test_small_oracle_traffic_retains_batch_limit(kind: str) -> None:
    oracle, stdin = _transport_only_oracle(kind)
    requests = [_pmove_request(index, 1) for index in range(40)]

    oracle.call(requests)

    assert [len(batch) for batch in stdin.batches] == [32, 8]


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


class _LowOverhangDestinationCm:
    def __init__(self) -> None:
        self.seen: list[dict] = []

    def call(self, requests):
        self.seen.extend(requests)
        output = []
        for request in requests:
            if request["id"].startswith("seed-floor-raised:"):
                output.append({
                    "startsolid": True, "allsolid": False, "fraction": 0,
                    "endpos": request["start"],
                    "plane": {"normal": [0.0, 0.0, 0.0]},
                })
            elif request["id"].startswith("seed-floor-nominal:"):
                output.append({
                    "startsolid": False, "allsolid": False,
                    "fraction": 0.001462,
                    "endpos": [2368.0, 1856.0, 216.03125],
                    "plane": {"normal": [0.0, 0.0, 1.0]},
                })
            else:
                raise AssertionError(request["id"])
        return output


def test_nonspawn_seed_uses_nominal_support_under_low_overhang() -> None:
    cm = _LowOverhangDestinationCm()

    grounded, support = _ground_navigation_seeds(
        cm, [(32, (2368.0, 1856.0, 216.0), False)],
    )

    assert grounded == [(2368.0, 1856.0, 216.03125)]
    assert support[0]["startsolid"] is False
    assert [request["start"][2] for request in cm.seen] == [248.0, 216.125]


def test_navigation_floor_candidates_preserve_raised_then_nominal_order() -> None:
    requests = _candidate_floor_requests([
        ((78, 112, 39), (1256.0, 1800.0, 120.0)),
    ])

    assert [request["id"] for request in requests] == [
        "floor-raised:0", "floor-nominal:0",
    ]
    assert [request["start"][2] for request in requests] == [138.0, 120.125]
    assert [request["end"][2] for request in requests] == [88.0, 88.0]


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


def test_objective_artifact_is_versioned_atlas_bound_and_uses_admitted_l1() -> None:
    entities = (
        EntityMetadata(1, "info_player_deathmatch", (("origin", "0 0 0"),)),
        EntityMetadata(2, "weapon_railgun", (("origin", "32 0 24"),)),
    )
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (0.0, 0.0, 9.0), True, True, True, 0, (0.0, 0.0, 1.0),
        ),
        (2, 0, 1): NavNode(
            (2, 0, 1), (32.0, 0.0, 24.0), True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }
    artifact, omissions = _objective_artifact(
        SimpleNamespace(entities=entities, sha256="ab" * 32),
        nodes,
        {1: (0, 0, 0)},
        (0, 0, 0),
        "objective-fixture",
        "cd" * 32,
        strict_binding=True,
    )
    assert artifact["schema"] == "q2-atlas-objectives-v1"
    assert artifact["canonical_map_id"] == "objective-fixture"
    assert artifact["atlas_sha256"] == "cd" * 32
    assert omissions == []
    assert artifact["objectives"] == [
        {
            "class": "spawn_egress",
            "classname": "info_player_deathmatch",
            "confidence": 65535,
            "l1_index": [0, 0, 0],
            "objective_id": 1,
            "risk": 0,
            "world_milliunits": [0, 0, 9000],
        },
        {
            "class": "weapon",
            "classname": "weapon_railgun",
            "confidence": 65535,
            "l1_index": [2, 0, 1],
            "objective_id": 2,
            "risk": 0,
            "world_milliunits": [32000, 0, 24000],
        },
    ]


def test_strict_generated_objective_binding_rejects_beyond_fence() -> None:
    entities = (
        EntityMetadata(45, "ammo_grenades", (("origin", "0 0 24"),)),
    )
    nodes = {
        (20, 0, 1): NavNode(
            (20, 0, 1), (320.0, 0.0, 24.0), True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }
    with pytest.raises(AtlasAnalysisError, match="objective entity 45 is 320.000"):
        _objective_artifact(
            SimpleNamespace(entities=entities, sha256="ab" * 32),
            nodes,
            {},
            (0, 0, 0),
            "generated-fixture",
            "cd" * 32,
            strict_binding=True,
        )


def test_authored_objective_omits_unbound_with_deterministic_evidence() -> None:
    entities = (
        # Far from every admitted L1 (nearest is 228 units), like q2dm8 entity 45.
        EntityMetadata(45, "ammo_grenades", (("origin", "0 0 24"),)),
        EntityMetadata(7, "weapon_railgun", (("origin", "400 0 24"),)),
    )
    nodes = {
        (14, 0, 1): NavNode(
            (14, 0, 1), (228.09, 0.0, 24.0), True, True, True, 0, (0.0, 0.0, 1.0),
        ),
        (25, 0, 1): NavNode(
            (25, 0, 1), (400.0, 0.0, 24.0), True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }
    artifact, omissions = _objective_artifact(
        SimpleNamespace(entities=entities, sha256="ab" * 32),
        nodes,
        {},
        (0, 0, 0),
        "stock-fixture",
        "cd" * 32,
        strict_binding=False,
    )
    assert [record["objective_id"] for record in artifact["objectives"]] == [7]
    assert artifact["objectives"][0]["classname"] == "weapon_railgun"
    assert artifact["objectives"][0]["l1_index"] == [25, 0, 1]
    assert omissions == [{
        "classname": "ammo_grenades",
        "entity_id": 45,
        "nearest_distance_milliunits": 228_090,
        "reason": OBJECTIVE_OMISSION_BEYOND_FENCE,
    }]
    guideposts = _objective_guidepost_analysis(len(artifact["objectives"]), omissions)
    assert guideposts["admitted_count"] == 1
    assert guideposts["omitted_count"] == 1
    assert guideposts["omissions"] == omissions


def test_authored_objective_omission_without_supported_passable_target() -> None:
    entities = (
        EntityMetadata(9, "item_quad", (("origin", "0 0 24"),)),
    )
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (0.0, 0.0, 24.0), False, False, False, 0, (0.0, 0.0, 1.0),
        ),
    }
    artifact, omissions = _objective_artifact(
        SimpleNamespace(entities=entities, sha256="ab" * 32),
        nodes,
        {},
        (0, 0, 0),
        "stock-fixture",
        "cd" * 32,
        incident=set(),
        strict_binding=False,
    )
    assert artifact["objectives"] == []
    assert omissions == [{
        "classname": "item_quad",
        "entity_id": 9,
        "nearest_distance_milliunits": 0,
        "reason": OBJECTIVE_OMISSION_NO_TARGET,
    }]


def test_near_objective_within_fence_is_retained() -> None:
    entities = (
        EntityMetadata(3, "ammo_bullets", (("origin", "0 0 24"),)),
    )
    # Nearest admitted L1 is 150 units away, still inside the 160 fence.
    nodes = {
        (9, 0, 1): NavNode(
            (9, 0, 1), (150.0, 0.0, 24.0), True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }
    artifact, omissions = _objective_artifact(
        SimpleNamespace(entities=entities, sha256="ab" * 32),
        nodes,
        {},
        (0, 0, 0),
        "near-fixture",
        "cd" * 32,
        strict_binding=False,
    )
    assert omissions == []
    assert artifact["objectives"] == [{
        "class": "ammunition",
        "classname": "ammo_bullets",
        "confidence": 65535,
        "l1_index": [9, 0, 1],
        "objective_id": 3,
        "risk": 0,
        "world_milliunits": [0, 0, 24000],
    }]
    assert OBJECTIVE_TARGET_MAX_DISTANCE == 160.0


def test_objective_omission_evidence_is_cold_semantically_stable() -> None:
    primary = {
        "schema": "q2-atlas-analysis-v1",
        "status": "candidate",
        "deterministic_rebuild": False,
        "performance": {"primary_elapsed_milliseconds": 12},
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
            "objective_guideposts": _objective_guidepost_analysis(
                1,
                [{
                    "classname": "ammo_grenades",
                    "entity_id": 45,
                    "nearest_distance_milliunits": 228_090,
                    "reason": OBJECTIVE_OMISSION_BEYOND_FENCE,
                }],
            ),
            "pmove_source_accounting": {"selected": 7, "omitted": 3},
        },
    }
    cold = json.loads(json.dumps(primary))
    cold["performance"]["primary_elapsed_milliseconds"] = 44
    cold["artifacts"]["atlas"]["build_peak_rss_bytes"] = 2048
    cold["identity"]["atlas_manifest_sha256"] = "4" * 64
    cold["artifacts"]["atlas_manifest"]["sha256"] = "5" * 64
    cold["artifacts"]["atlas_manifest"]["uncompressed_bytes"] = 1000
    cold["artifacts"]["atlas_manifest"]["verification"]["manifest_sha256"] = "6" * 64
    assert _normalized_analysis_manifest(primary) == _normalized_analysis_manifest(cold)
    assert (
        canonical_json(_normalized_analysis_manifest(primary))
        == canonical_json(_normalized_analysis_manifest(cold))
    )

    drifted = json.loads(json.dumps(cold))
    drifted["compiled_world"]["objective_guideposts"]["omissions"][0][
        "nearest_distance_milliunits"
    ] = 228_091
    assert _normalized_analysis_manifest(primary) != _normalized_analysis_manifest(
        drifted
    )


def test_unknown_objective_class_fails_instead_of_disappearing() -> None:
    metadata = SimpleNamespace(entities=(
        EntityMetadata(1, "item_future_unknown", (("origin", "1 2 3"),)),
    ))
    with pytest.raises(AtlasAnalysisError, match="unsupported objective class"):
        _authored_item_destinations(metadata)


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
    assert candidate["analyzer_version"] == "b2-a-v4"
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
    assert BUILD_PLAN_SCHEMA == "q2-atlas-build-plan-v2"
    assert 'plan.schema != "q2-atlas-build-plan-v2"' in source
    assert "q2-atlas-build-plan-v1" not in source
    assert "bitmaps: BTreeMap<String, String>" in source
    assert "scalar_values: BTreeMap<String, String>" in source
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
    nodes = {
        (x, y, 0): NavNode(
            (x, y, 0), (x * 16.0 + 8.0, y * 16.0 + 8.0, 24.0),
            True, True, True, 0, (0.0, 0.0, 1.0),
        )
        for x in range(5, 12)
        for y in range(-2, 6)
    }
    retained = _surface_candidate_scope(nodes, (0, 0, 0)).authorized_chunks
    semantics = {}
    chunks = _l0_chunks(
        nodes, [], (0, 0, 0), metadata=metadata, semantic_summary=semantics
    )
    assert {tuple(chunk["key"]) for chunk in chunks}.issubset(retained)
    counts = {}
    scalar_counts = {}
    for chunk in chunks:
        for name, cells in chunk["bits"].items():
            counts[name] = counts.get(name, 0) + len(cells)
        for name, cells in chunk["scalars"].items():
            scalar_counts[name] = scalar_counts.get(name, 0) + len(cells)
    if state == "toggle":
        assert counts["unknown"] > 0
        assert scalar_counts == {}
        assert semantics["hurt_potential"] == counts["unknown"]
        assert semantics["hurt:1:potential"] == counts["unknown"]
    elif state == "active":
        assert 0 < counts["hurt"] <= 1_944
        assert 0 < counts["standing_forbidden_origin"] <= 13_520
        assert 0 < counts["crouched_forbidden_origin"] <= 8_788
        assert scalar_counts["hazard_severity"] > 0
        assert semantics["hurt:1:raw"] == counts["hurt"]
        assert semantics["hurt:1:expanded"] == semantics["hurt_expanded"]
    else:
        assert counts == {}
        assert scalar_counts == {}
        assert semantics == {}


def test_map_wide_hurt_is_clipped_to_sparse_reachable_permit() -> None:
    trigger = EntityMetadata(
        index=7,
        classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(
                mins=(-4096, -4096, -240), maxs=(4096, 4096, -64),
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 7, "model_index": 1},
        )),
    )
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (8.0, 8.0, 24.0), True, True, True, 0,
            (0.0, 0.0, 1.0),
        ),
    }
    retained = _surface_candidate_scope(nodes, (0, 0, 0)).authorized_chunks
    semantics: dict[str, int] = {}
    chunks = _l0_chunks(
        nodes, [], (0, 0, 0), metadata=metadata,
        semantic_summary=semantics,
    )
    assert chunks
    assert {tuple(chunk["key"]) for chunk in chunks}.issubset(retained)
    assert 0 < semantics["hurt:7:raw"] < 2_000_000
    assert 0 < semantics["hurt:7:expanded"] < 2_000_000


def test_fixed_point_floor_boundary_admits_only_one_lower_hurt_chunk() -> None:
    """Match the exact chunk alignment that rejected generated cohort 71430."""

    trigger = EntityMetadata(
        index=1,
        classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            # Runtime linking expands this to
            # [-514,-514,-242]..[3586,3586,-62].
            SimpleNamespace(
                mins=(-513, -513, -241), maxs=(3585, 3585, -63),
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    origin = (-512, -512, -512)
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (0.0, 0.0, 24.03125),
            True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }
    scope = _surface_candidate_scope(nodes, origin)
    retained = set(scope.authorized_chunks)
    boundary = set(_hurt_boundary_chunks(scope, nodes, origin))

    assert all(chunk[2] == 8 for chunk in retained)
    assert boundary == {(8, 8, 7)}

    semantics: dict[str, int] = {}
    chunks = _l0_chunks(
        nodes, [], origin, metadata=metadata, semantic_summary=semantics,
    )
    hurt_chunks = {
        tuple(chunk["key"])
        for chunk in chunks
        if chunk["bits"].get("hurt")
    }
    assert hurt_chunks == boundary
    assert semantics["hurt:1:raw"] > 0
    assert semantics["hurt:1:expanded"] > 0

    compact_semantics: dict[str, int] = {}
    compact = _l0_chunks(
        nodes, [], origin, metadata=metadata,
        semantic_summary=compact_semantics, compact_output=True,
    )
    expanded_compact = []
    for chunk in compact:
        expanded_compact.append({
            "key": chunk["key"],
            "bits": {
                name: [
                    byte_index * 8 + bit
                    for byte_index, byte in enumerate(bytes.fromhex(encoded))
                    for bit in range(8)
                    if byte & (1 << bit)
                ]
                for name, encoded in chunk["bitmaps"].items()
            },
            "scalars": {
                name: [
                    [linear, value]
                    for linear, value in enumerate(bytes.fromhex(encoded))
                    if value
                ]
                for name, encoded in chunk["scalar_values"].items()
            },
        })
    assert expanded_compact == chunks
    assert compact_semantics == semantics


def test_generated_hurt_scope_uses_only_lethal_edge_floor_strips() -> None:
    safety = {
        "lethal_edges": [
            {"side": "west", "segment": [0, 0, 0, 512, 0]},
            {"side": "north", "segment": [512, 512, 1024, 512, 0]},
        ],
    }

    chunks = _claimed_hurt_boundary_chunks(safety, (-512, -512, -512))

    assert chunks == tuple(
        [(8, y, 7) for y in range(8, 16)]
        + [(x, 15, 7) for x in range(16, 24)]
    )
    assert len(chunks) == 16


def _generator_kill_plane_linked_bounds(generator: MapGenerator) -> Aabb:
    minimum_floor = min(room.floor_z for room in generator.rooms)
    return Aabb(
        (
            min(room.wx for room in generator.rooms) - KILL_PLANE_MARGIN - 2,
            min(room.wy for room in generator.rooms) - KILL_PLANE_MARGIN - 2,
            minimum_floor - 256 + WALL_T - 2,
        ),
        (
            max(room.wx + room.w for room in generator.rooms)
            + KILL_PLANE_MARGIN + 2,
            max(room.wy + room.d for room in generator.rooms)
            + KILL_PLANE_MARGIN + 2,
            minimum_floor - KILL_PLANE_DROP + 2,
        ),
    )


def test_towers_71812102_retains_raised_edge_sparse_hurt_evidence() -> None:
    """Reproduce the sole failed member of qualification 71812000."""

    generator = MapGenerator(seed=71_812_102, style="towers")
    generator.generate(5)
    safety = {"lethal_edges": generator.lethal_edges}
    origin = (0, 0, -512)
    historical = _claimed_hurt_boundary_chunks(safety, origin)
    linked = _generator_kill_plane_linked_bounds(generator)

    assert min(room.floor_z for room in generator.rooms) == 0
    assert {edge["segment"][4] for edge in generator.lethal_edges} == {96}
    assert {chunk[2] for chunk in historical} == {8}
    projected = _project_generated_hurt_chunks(historical, linked, origin)
    assert {chunk[2] for chunk in projected} == {4, 5, 6, 7}

    trigger = EntityMetadata(
        index=1,
        classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(
                mins=(0.0, 0.0, -240.0),
                maxs=(3584.0, 3584.0, -64.0),
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 1, "model_index": 1},
        )),
    )
    semantics: dict[str, int] = {}
    chunks = _l0_chunks(
        {}, [], origin,
        metadata=metadata,
        semantic_summary=semantics,
        generated_safety=safety,
    )

    assert semantics["hurt:1:raw"] > 0
    assert semantics["hurt:1:expanded"] > 0
    assert {
        tuple(chunk["key"])[2]
        for chunk in chunks
        if chunk["bits"].get("hurt")
    } == {4, 5, 6, 7}


@pytest.mark.parametrize(
    ("style", "seed"),
    tuple(
        (style, 71_812_000 + style_index * 100 + member)
        for style_index, style in enumerate((
            "open", "towers", "canyon", "pits",
            "arena_open", "arena_vertical", "arena_lanes",
        ))
        for member in range(4)
    ),
)
def test_71812000_campaign_has_nonvacuous_projected_hurt_scope(
    style: str, seed: int,
) -> None:
    generator = MapGenerator(seed=seed, style=style)
    generator.generate(5)
    origin = (0, 0, -512)
    historical = _claimed_hurt_boundary_chunks(
        {"lethal_edges": generator.lethal_edges}, origin,
    )
    projected = _project_generated_hurt_chunks(
        historical, _generator_kill_plane_linked_bounds(generator), origin,
    )

    assert projected, f"{style}/{seed} retained no sparse hurt chunks"
    assert {(x, y) for x, y, _z in projected} == {
        (x, y) for x, y, _z in historical
    }


def test_generated_hurt_scope_rejects_malformed_or_empty_edges() -> None:
    with pytest.raises(AtlasAnalysisError, match="lacks lethal-edge"):
        _claimed_hurt_boundary_chunks({"lethal_edges": []}, (0, 0, 0))
    with pytest.raises(AtlasAnalysisError, match="geometry differs"):
        _claimed_hurt_boundary_chunks({
            "lethal_edges": [
                {"side": "west", "segment": [0, 0, 16, 512, 0]},
            ],
        }, (0, 0, 0))


def test_fixed_point_hurt_boundary_does_not_cross_two_chunk_gap() -> None:
    trigger = EntityMetadata(
        index=2,
        classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(
                mins=(-513, -513, -305), maxs=(3585, 3585, -127),
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 2, "model_index": 1},
        )),
    )
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (0.0, 0.0, 24.03125),
            True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }
    semantics: dict[str, int] = {}

    chunks = _l0_chunks(
        nodes, [], (-512, -512, -512), metadata=metadata,
        semantic_summary=semantics,
    )

    assert chunks == []
    assert "hurt:2:raw" not in semantics
    assert "hurt:2:expanded" not in semantics


def test_hurt_semantic_scratch_is_prospectively_bounded() -> None:
    trigger = EntityMetadata(
        index=2,
        classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(mins=(-1, -1, -1), maxs=(1, 1, 1)),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 2, "model_index": 1},
        )),
    )
    nodes = {
        (0, 0, 0): NavNode(
            (0, 0, 0), (0.0, 0.0, 24.0),
            True, True, True, 0, (0.0, 0.0, 1.0),
        ),
    }

    with pytest.raises(
        AtlasAnalysisError,
        match="L0 semantic scratch exceeds bounded sparse accounting",
    ):
        _l0_chunks(
            nodes, [], (0, 0, 0), metadata=metadata,
            semantic_summary={}, semantic_scratch_max_bytes=511,
        )


def test_hurt_boundary_excludes_compatible_interior_columns() -> None:
    origin = (-512, -512, -512)
    nodes = {
        (x, y, 0): NavNode(
            (x, y, 0), (x * 16.0, y * 16.0, 24.03125),
            True, True, True, 0, (0.0, 0.0, 1.0),
        )
        for x in range(-1, 2)
        for y in range(-1, 2)
    }
    scope = _surface_candidate_scope(nodes, origin)
    boundary = set(_hurt_boundary_chunks(scope, nodes, origin))
    without_center = dict(nodes)
    del without_center[(0, 0, 0)]
    without_center_scope = _surface_candidate_scope(without_center, origin)

    assert (0, 0, 0) not in scope.boundary_l1
    assert boundary == set(
        _hurt_boundary_chunks(without_center_scope, without_center, origin)
    )


def test_hurt_inclusive_upper_grid_boundary_is_retained() -> None:
    trigger = EntityMetadata(
        index=8, classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            # Runtime linking expands this to [-1, -1, -1]..[64, 64, 16].
            # The exact upper values lie on 4u grid planes and are inclusive.
            SimpleNamespace(mins=(0, 0, 0), maxs=(63, 63, 15)),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 8, "model_index": 1},
        )),
    )
    nodes = {
        (x, y, 0): NavNode(
            (x, y, 0), (x * 16.0 + 8.0, y * 16.0 + 8.0, 24.0),
            True, True, True, 0, (0.0, 0.0, 1.0),
        )
        for x in range(-1, 6)
        for y in range(-1, 6)
    }
    chunks = _l0_chunks(nodes, [], (0, 0, 0), metadata=metadata)

    def marked(index: tuple[int, int, int]) -> bool:
        chunk_key = tuple(value // 16 for value in index)
        local = tuple(value % 16 for value in index)
        linear = local[0] + 16 * local[1] + 256 * local[2]
        return any(
            tuple(chunk["key"]) == chunk_key
            and linear in chunk["bits"].get("hurt", ())
            for chunk in chunks
        )

    assert marked((-1, -1, -1))
    assert marked((16, 16, 4))


def test_retained_hurt_above_two_million_cells_uses_authoritative_budget() -> None:
    trigger = EntityMetadata(
        index=12, classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(
                mins=(-4096, -4096, -256), maxs=(4096, 4096, 256),
            ),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 12, "model_index": 1},
        )),
    )
    nodes = {
        (x, y, 0): NavNode(
            (x, y, 0), (x * 16.0 + 8.0, y * 16.0 + 8.0, 24.0),
            True, True, True, 0, (0.0, 0.0, 1.0),
        )
        for x in range(100)
        for y in range(80)
    }
    semantics: dict[str, int] = {}

    chunks = _l0_chunks(
        nodes, [], (0, 0, 0), metadata=metadata,
        semantic_summary=semantics, compact_output=True,
    )

    assert len(chunks) == 677
    assert semantics["hurt:12:raw"] == 2_772_992
    assert semantics["hurt:12:expanded"] == 2_772_992


def test_overlapping_hurts_keep_distinct_deterministic_sparse_counts() -> None:
    triggers = (
        EntityMetadata(
            index=3, classname="trigger_hurt",
            properties=(("model", "*1"), ("spawnflags", "12")),
        ),
        EntityMetadata(
            index=9, classname="trigger_hurt",
            properties=(("model", "*2"), ("spawnflags", "12")),
        ),
    )
    metadata = SimpleNamespace(
        entities=triggers,
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(mins=(-64, -64, -16), maxs=(32, 32, 16)),
            SimpleNamespace(mins=(-16, -16, -16), maxs=(80, 80, 16)),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 3, "model_index": 1},
            {"entity_index": 9, "model_index": 2},
        )),
    )
    nodes = {
        (x, y, 0): NavNode(
            (x, y, 0), (x * 16.0 + 8.0, y * 16.0 + 8.0, 24.0),
            True, True, True, 0, (0.0, 0.0, 1.0),
        )
        for x in range(-6, 7)
        for y in range(-6, 7)
    }

    first_semantics: dict[str, int] = {}
    second_semantics: dict[str, int] = {}
    first = _l0_chunks(
        nodes, [], (0, 0, 0), metadata=metadata,
        semantic_summary=first_semantics,
    )
    second = _l0_chunks(
        nodes, [], (0, 0, 0), metadata=metadata,
        semantic_summary=second_semantics,
    )

    assert first == second
    assert first_semantics == second_semantics
    for entity_index in (3, 9):
        assert first_semantics[f"hurt:{entity_index}:raw"] > 0
        assert first_semantics[f"hurt:{entity_index}:expanded"] > 0
    assert first_semantics["hurt_expanded"] < sum(
        first_semantics[f"hurt:{entity_index}:expanded"]
        for entity_index in (3, 9)
    )


def test_hurt_sparse_planes_reserve_all_chunks_before_materialization() -> None:
    trigger = EntityMetadata(
        index=4, classname="trigger_hurt",
        properties=(("model", "*1"), ("spawnflags", "12")),
    )
    metadata = SimpleNamespace(
        entities=(trigger,),
        models=(
            SimpleNamespace(mins=(0, 0, 0), maxs=(0, 0, 0)),
            SimpleNamespace(mins=(-1024, -64, -64), maxs=(1024, 64, 64)),
        ),
        entity_catalog=SimpleNamespace(brush_submodels=(
            {"entity_index": 4, "model_index": 1},
        )),
    )
    nodes = {
        index: NavNode(
            index,
            (index[0] * 16.0 + 8.0, 8.0, 24.0),
            True, True, True, 0, (0.0, 0.0, 1.0),
        )
        for index in ((-32, 0, 0), (32, 0, 0))
    }

    with pytest.raises(AtlasAnalysisError, match="L0 chunk count"):
        _l0_chunks(
            nodes, [], (0, 0, 0), metadata=metadata,
            budget_state=L0BudgetState(max_chunks=1),
        )


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
    contents: int = 0,
) -> NavNode:
    return NavNode(
        key, tuple(float(axis * 16) for axis in key), standing, crouched,
        True, contents, (0.0, 0.0, 1.0),
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
        "spawn-ladder-component-frontier-proportional-farthest-v3"
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


def test_pmove_sources_retain_every_exact_ladder_node_before_sampling() -> None:
    spawn = (0, 0, 0)
    ladder = (100, 0, 0)
    nodes = {
        spawn: _test_node(spawn, standing=True, crouched=True),
        ladder: _test_node(
            ladder, standing=True, crouched=True, contents=CONTENTS_LADDER,
        ),
    }
    selected, accounting = _complete_pmove_source_set(
        nodes, [], {1: spawn}, 2,
    )
    assert selected == [spawn, ladder]
    assert accounting["omitted"] == 0

    with pytest.raises(
        AtlasAnalysisError,
        match="cannot retain every exact ladder-content source",
    ):
        _complete_pmove_source_set(nodes, [], {1: spawn}, 1)


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
    assert _movement_edge_kind(
        "ladder", any_airborne=True, vertical=344.0,
    ) == "ladder"
    assert _movement_edge_kind(
        "ladder", any_airborne=True, vertical=18.0,
    ) is None
    assert _movement_edge_kind(
        "ladder", any_airborne=False, vertical=344.0,
    ) is None


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


def test_ladder_requests_require_source_contents_and_mirror_engine_contact() -> None:
    key = (1, 2, 3)
    ordinary = _test_node(key, standing=True, crouched=True)
    assert _ladder_request_candidates_for_source(
        key, ordinary, parameters={"gravity": 800, "airaccelerate": 0},
    ) == []

    ladder = _test_node(
        key, standing=True, crouched=True, contents=CONTENTS_LADDER,
    )
    candidates = _ladder_request_candidates_for_source(
        key, ladder, parameters={"gravity": 800, "airaccelerate": 0},
    )
    assert len(candidates) == 8
    assert {record[3] for record in candidates} == {"standing", "crouched"}
    assert all(record[1] == "ladder" for record in candidates)
    assert all(len(record[4]["commands"]) == 44 for record in candidates)
    assert all(
        command["angles"][0] == -30
        and command["forwardmove"] == 300
        and "upmove" not in command
        for record in candidates for command in record[4]["commands"]
    )


def test_ladder_contacts_replay_exact_frame_start_hulls_through_landing() -> None:
    key = (1, 2, 3)
    source = _test_node(
        key, standing=True, crouched=False, contents=CONTENTS_LADDER,
    )
    candidates = _ladder_request_candidates_for_source(
        key, source, parameters={"gravity": 800, "airaccelerate": 0},
    )
    record = candidates[0]
    response = {"frames": [
        {
            "origin": [17.0, 32.0, 48.0],
            "mins": [-16.0, -16.0, -24.0],
            "maxs": [16.0, 16.0, 32.0],
        },
        {
            "origin": [18.0, 32.0, 48.0],
            "mins": [-16.0, -16.0, -24.0],
            "maxs": [16.0, 16.0, 32.0],
        },
        *[
            {
                "origin": [18.0, 32.0, 48.0],
                "mins": [-16.0, -16.0, -24.0],
                "maxs": [16.0, 16.0, 32.0],
            }
            for _ in range(42)
        ],
    ]}
    contacts = _ladder_contact_requests(
        record, response, through_command_index=1,
    )
    assert len(contacts) == 2
    # Command zero starts from the Pmove oracle's exact 1/8-unit input snap.
    assert contacts[0]["start"] == [16.0, 32.0, 48.0]
    assert contacts[0]["end"] == [17.0, 32.0, 48.0]
    # Command one starts at the prior exact Pmove output, not the source.
    assert contacts[1]["start"] == [17.0, 32.0, 48.0]
    assert contacts[1]["end"] == [18.0, 32.0, 48.0]
    assert contacts[1]["mins"] == [-16.0, -16.0, -24.0]
    assert contacts[1]["maxs"] == [16.0, 16.0, 32.0]
    assert all(contact["mask"] == 33_619_971 for contact in contacts)


def test_ladder_contact_predicate_matches_pmove_special_movement() -> None:
    responses = [
        {"fraction": 0.5, "contents": CONTENTS_LADDER},
        {"fraction": 1.0, "contents": CONTENTS_LADDER},
        {"fraction": 0.5, "contents": 1},
        {"fraction": 1.0, "contents": 1},
    ]
    assert [_ladder_contact_trace_admits(item) for item in responses] == [
        True, False, False, False,
    ]


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


def test_ladder_probe_gets_exactly_eight_neutral_settle_frames() -> None:
    source = _test_node(
        (1, 2, 3), standing=True, crouched=False, contents=CONTENTS_LADDER,
    )
    request = _ladder_request_candidates_for_source(
        (1, 2, 3), source,
        parameters={"gravity": 800, "airaccelerate": 0},
    )[0][4]
    response = {"frames": [{"grounded": False} for _ in range(44)]}
    extended = _drop_settle_request(request, response, horizon_frames=52)
    assert extended is not None
    assert len(extended["commands"]) == 52
    assert extended["commands"][:44] == request["commands"]
    assert extended["commands"][44:] == [
        {"msec": 50, "angles": [-30, 0, 0]} for _ in range(8)
    ]


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


def test_exact_lethal_drop_marks_predamage_l1_but_unknown_never_invents_it() -> None:
    node = NavNode(
        (0, 0, 0), (8.0, 8.0, 24.0), True, True, True, 0,
        (0.0, 0.0, 1.0),
    )
    nodes = {(0, 0, 0): node}
    assert _apply_static_drop_hazards(nodes, [
        {
            "source_l1": [0, 0, 0],
            "classification": {"classification": "Unknown", "reason": "no_landing"},
        },
        {
            "source_l1": [0, 0, 0],
            "classification": {"classification": "Exact", "lethal": True},
        },
    ]) == 1
    plan = node.plan(True)
    assert plan["hazard_types"] & (1 << 3)
    assert plan["hazard_severity"] == 255
    assert "hazard_clearance" not in plan

    with pytest.raises(AtlasAnalysisError, match="not admitted Atlas L1"):
        _apply_static_drop_hazards(nodes, [{
            "source_l1": [9, 9, 9],
            "classification": {"classification": "Exact", "lethal": True},
        }])
