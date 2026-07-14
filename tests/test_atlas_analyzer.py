from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import subprocess

import pytest

from harness.atlas_analyzer import (
    AnalyzerLimits,
    CONTENTS_LAVA,
    CONTENTS_PLAYERCLIP,
    NavNode,
    _l0_chunks,
    analyze_map,
    sha256_file,
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


def _required_artifacts() -> tuple[Path, Path, Path, Path]:
    cm = CLIENT / "release/q2-cm-oracle"
    pmove = CLIENT / "release/q2-pmove-oracle"
    hook = LITHIUM / "tools/q2-hook-oracle"
    missing = [path for path in (cm, pmove, hook, ATTESTATION, FIXTURE_MODULE) if not path.is_file()]
    if missing:
        pytest.skip("B1 integration artifacts unavailable: " + ", ".join(map(str, missing)))
    packer = ROOT / "target/debug/q2-atlas-pack"
    subprocess.run(
        ["cargo", "build", "-p", "q2-lattice", "--bin", "q2-atlas-pack"],
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
            assert left_manifest == right_manifest, name
        else:
            assert left == right, name
    manifest = json.loads((outputs[0] / "atlas-fixture.analysis.manifest.json").read_bytes())
    assert manifest["artifacts"]["atlas"]["uncompressed_sha256"] == sha256_file(
        outputs[0] / "atlas-fixture.atlas.bin"
    )


def test_l0_omits_open_floor_interior_but_keeps_required_bands() -> None:
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

    # A fully surrounded ordinary floor chunk has no L0 allocation: L1 owns
    # its support fact. Boundary rows still retain a solid surface band.
    assert (4, 4, -1) not in by_key
    assert any("solid" in planes for planes in by_key.values())
    assert any("lava" in planes for planes in by_key.values())
    assert any("playerclip" in planes for planes in by_key.values())
    assert any("spawn_column" in planes for planes in by_key.values())
