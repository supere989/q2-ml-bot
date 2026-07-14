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

from harness.atlas_analyzer import (
    AtlasAnalysisError,
    AnalyzerLimits,
    CONTENTS_LAVA,
    CONTENTS_PLAYERCLIP,
    FROZEN_L0_BIT_PLANE_NAMES,
    FROZEN_L0_SCALAR_PLANE_NAMES,
    NavNode,
    _atlas_channels,
    _analyze_hook_claims,
    _l0_chunks,
    _process_tree_rss_bytes,
    _run_measured_process,
    _surface_candidate_scope,
    _surface_request_upper_bound,
    analyze_map,
    sha256_file,
)
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


def _required_artifacts() -> tuple[Path, Path, Path, Path]:
    cm = CLIENT / "release/q2-cm-oracle"
    pmove = CLIENT / "release/q2-pmove-oracle"
    hook = LITHIUM / "tools/q2-hook-oracle"
    missing = [path for path in (cm, pmove, hook, ATTESTATION, FIXTURE_MODULE) if not path.is_file()]
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
    return cm, pmove, hook, packer


def test_synthetic_fixture_cold_rebuilds_all_artifacts(tmp_path: Path) -> None:
    cm, pmove, hook, packer = _required_artifacts()
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
        max_l1_nodes=2_000, max_l1_edges=10_000, max_pmove_sources=32,
    )
    candidate = analyze_map(
        bsp, tmp_path / "candidate", "atlas-fixture", provenance,
        cm_oracle=cm, pmove_oracle=pmove, hook_oracle=hook,
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
                assert proof["artifact_count"] == 7
                assert (
                    0 < proof["sampled_process_tree_peak_rss_bytes"]
                    <= proof["peak_rss_limit_bytes"]
                )
                proof.pop("sampled_process_tree_peak_rss_bytes")
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


def test_train_marks_candidate_poses_unknown_without_guessing_between_them() -> None:
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
    assert not marked((26, 0, 0), "mover_swept_envelope")
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
