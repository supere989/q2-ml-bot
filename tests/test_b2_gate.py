from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest
import zstandard

from tools.assemble_b2_gate import (
    B2GateError,
    B2GatePaths,
    EXPECTED_COHORT,
    EXPECTED_DESIGN_SHA256,
    _exact_directory_files,
    _decode_dyn_snapshot,
    _dyn_source_authority,
    _expected_71432_rows,
    _validate_dyn_evidence,
    _validate_test_report,
    _parser,
)
from tools.run_generator_cohort import canonical_bytes
from tools.run_b2_test_suite import (
    B2TestSuiteError,
    _commands,
    _parse_counts,
    _rename_noreplace,
)
import tools.run_compiled_static_campaign as compiled_static_campaign


ROOT = Path(__file__).resolve().parents[1]
SHA = "12" * 32
COMMIT = "34" * 20


def _implementation() -> dict:
    return {
        "repository_commit": COMMIT,
        "repository_tree": "56" * 20,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": SHA,
        "atlas_analyzer_authority_file_count": 1,
        "generator_sha256": "78" * 32,
        "routes_sha256": "9a" * 32,
    }


def _paths(tmp_path: Path) -> B2GatePaths:
    dummy = tmp_path / "unused"
    return B2GatePaths(
        design=dummy,
        plan=dummy,
        repo_root=ROOT,
        b1_gate=dummy,
        cm_oracle=dummy,
        pmove_oracle=dummy,
        hook_oracle=dummy,
        fall_oracle=dummy,
        hook_attestation=dummy,
        atlas_verifier=dummy,
        declaration=dummy,
        source_dir=dummy,
        source_cold_dir=dummy,
        source_freeze_report=dummy,
        compiled_dir=dummy,
        compiled_membership_report=dummy,
        compiled_static_report=dummy,
        materialized_dir=dummy,
        materialized_membership_report=dummy,
        claims_dir=tmp_path / "claims",
        claims_prepare_report=dummy,
        analysis_dir=tmp_path / "analysis",
        generated_build_report=dummy,
        generated_validation_report=dummy,
        stock_provenance=dummy,
        stock_inventory=dummy,
        stock_bsp_dir=dummy,
        stock_analysis_dir=dummy,
        stock_validation_dir=dummy,
        dyn_evidence_executable=tmp_path / "q2-dyn-evidence",
        dyn_evidence_report=tmp_path / "dyn/b2-dyn-evidence.json",
        test_report=dummy,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_evidence(path_text: str, payload: bytes) -> dict:
    return {"path": path_text, "sha256": _sha256(payload), "size_bytes": len(payload)}


def _encoded_snapshot(
    client: int, atlas_sha256: str, map_sha256: str
) -> bytes:
    values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.75)
    l2 = struct.pack("<iii7f", 0, 0, 0, *values)
    l3 = struct.pack("<iii7f", 0, 0, 0, *values)
    uncompressed = l2 + l3
    compressed = zstandard.ZstdCompressor(level=3).compress(uncompressed)
    header = bytearray()
    header += b"Q2LAT002"
    header += struct.pack("<HHI", 2, 0x454C, 208)
    header += struct.pack("<BbHI", 1, 3, 0, 0)
    header += struct.pack("<QQ", len(uncompressed), len(compressed))
    header += hashlib.sha256(uncompressed).digest()
    header += bytes.fromhex(atlas_sha256)
    header += bytes.fromhex(map_sha256)
    header += struct.pack("<qqq", 0, 0, 0)
    header += struct.pack("<II", 64, 256)
    header += struct.pack("<QQIIQQ", 1, 17, client, 4, 1, 1)
    assert len(header) == 208
    return bytes(header) + compressed


def _write_dyn_fixture(tmp_path: Path) -> tuple[B2GatePaths, dict, dict]:
    paths = _paths(tmp_path)
    paths.claims_dir.mkdir()
    paths.analysis_dir.mkdir()
    paths.dyn_evidence_report.parent.mkdir()
    map_id = _expected_71432_rows()[0]["map"]

    bsp = b"IBSP fixture"
    (paths.claims_dir / f"{map_id}.bsp").write_bytes(bsp)
    header = bytearray(136)
    header[:8] = b"Q2ATL001"
    struct.pack_into("<HHI", header, 8, 1, 0x454C, 136)
    counts = {"l0_chunks": 1, "l1_nodes": 1, "l1_edges": 0, "l2_cells": 1, "l3_cells": 1}
    struct.pack_into("<5Q", header, 56, *counts.values())
    struct.pack_into("<5Q", header, 96, 8, 0, 0, 0, 0)
    atlas_bytes = bytes(header) + b"l0-bytes"
    atlas_path = paths.analysis_dir / f"{map_id}.atlas.bin"
    atlas_path.write_bytes(atlas_bytes)
    atlas_manifest_bytes = canonical_bytes({
        "specification_sha256": EXPECTED_DESIGN_SHA256,
        "bsp": {"canonical_map_id": map_id, "sha256": _sha256(bsp)},
        "analyzer": {
            "name": "q2-atlas-analyzer",
            "version": "b2-a-v4",
            "sha256": SHA,
        },
    })
    atlas_manifest_path = paths.analysis_dir / f"{map_id}.atlas.manifest.json"
    atlas_manifest_path.write_bytes(atlas_manifest_bytes)
    analysis = {
        "grid": {"origin": [0, 0, 0]},
        "counts": counts,
        "artifacts": {
            "atlas": {
                "resident_bytes_estimate": 100,
                "build_peak_rss_bytes": 200,
            }
        },
    }
    (paths.analysis_dir / f"{map_id}.analysis.manifest.json").write_bytes(
        canonical_bytes(analysis)
    )

    executable = b"pinned q2-dyn-evidence executable fixture"
    paths.dyn_evidence_executable.write_bytes(executable)
    snapshots = []
    snapshot_payloads = []
    for client in range(4):
        payload = _encoded_snapshot(client, _sha256(atlas_bytes), _sha256(bsp))
        snapshot_payloads.append(payload)
        name = f"client{client}.q2lat002"
        (paths.dyn_evidence_report.parent / name).write_bytes(payload)
        snapshots.append({
            "client_id": client,
            "file": _file_evidence(
                f"/mnt/c/b2-dyn-evidence/{name}", payload
            ),
            "magic": "Q2LAT002",
            "schema_version": 2,
            "l2_cells": 1,
            "l3_cells": 1,
            "resident_bytes": 352,
            "byte_identical_roundtrip": True,
        })
    distribution = {"p50_ns": 10, "p95_ns": 20, "p99_ns": 30, "max_ns": 40}
    source_authority = _dyn_source_authority(ROOT, COMMIT)
    report = {
        "schema": "q2-b2-dyn-evidence-v1",
        "passed": True,
        "authority": {
            "specification_sha256": EXPECTED_DESIGN_SHA256,
            "analyzer_name": "q2-atlas-analyzer",
            "analyzer_version": "b2-a-v4",
            "analyzer_authority_sha256": SHA,
            "crate_commit": COMMIT,
            "executable_sha256": _sha256(executable),
            "canonical_map_id": map_id,
            "map_epoch": 1,
            "environment_steps": 17,
        },
        "provenance": {
            "embedded_repo_commit": COMMIT,
            "executable": _file_evidence(
                "/home/raymondj/bin/q2-dyn-evidence", executable
            ),
            **source_authority,
        },
        "host": {
            "hostname": "DESKTOP-RTX2080",
            "kernel_release": "6.18.33.2-microsoft-standard-WSL2",
            "architecture": "x86_64",
        },
        "atlas": {
            "manifest": _file_evidence(str(atlas_manifest_path), atlas_manifest_bytes),
            "artifact": _file_evidence(str(atlas_path), atlas_bytes),
            "bsp": _file_evidence(str(paths.claims_dir / f"{map_id}.bsp"), bsp),
            "origin": [0, 0, 0],
            "counts": counts,
            "resident_bytes": 100,
            "representative_l2_cells": 1,
            "lookup": "origin-indexed exact L2 aggregate binary search in the admitted resident Atlas",
        },
        "dyn_state": {
            "snapshot_magic": "Q2LAT002",
            "schema_version": 2,
            "client_ids": [0, 1, 2, 3],
            "client_count": 4,
            "common_environment_steps": 17,
            "population": "deterministic per-client representative channels over admitted Atlas L2 cells; authority identities are never synthetic",
            "snapshots": snapshots,
            "combined_compressed_bytes": sum(map(len, snapshot_payloads)),
            "combined_resident_bytes": 1408,
            "combined_limit_bytes": 8 * 1024 * 1024,
            "batch_ids_and_step_admitted": True,
        },
        "negative_fences_and_limits": {
            "stale_atlas_sha256_rejected": True,
            "stale_map_sha256_rejected": True,
            "stale_origin_rejected": True,
            "stale_map_epoch_rejected": True,
            "stale_environment_step_rejected": True,
            "wrong_client_count_rejected": True,
            "duplicate_client_rejected": True,
            "retired_schema_rejected": True,
            "mixed_schema_rejected": True,
            "payload_digest_corruption_rejected": True,
            "cell_size_mismatch_rejected": True,
            "soft_compressed_limit_reported": True,
            "hard_compressed_limit_rejected": True,
            "hard_resident_limit_rejected": True,
            "materialized_cell_limit_rejected": True,
        },
        "performance": {
            "scope": "one accepted resident transition: exact admitted Atlas L2 lookup plus 24-float Dyn feature assembly for clients 0..3",
            "resident_samples": 2000,
            "warmup_samples": 256,
            "clients_per_sample": 4,
            "atlas_lookup": distribution,
            "dyn_feature_assembly": distribution,
            "total": distribution,
            "total_p99_limit_ns": 500000,
            "total_p99_passed": True,
            "feature_width": 24,
        },
    }
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))
    declaration = {"cohort_id": EXPECTED_COHORT, "maps": _expected_71432_rows()}
    return paths, declaration, report


def test_71432_identity_is_exact_and_cli_has_no_discovery_flags() -> None:
    rows = _expected_71432_rows()
    assert len(rows) == 28
    assert rows[0] == {
        "ordinal": 0,
        "map": "b2g26_open_71432000",
        "seed": 71432000,
        "style": "open",
        "grid": 5,
        "observed_heat": None,
    }
    assert rows[-1]["map"] == "b2g26_arena_lanes_71432603"
    options = _parser().format_help()
    assert "--declaration" in options
    assert "--source-dir" in options
    assert "--stock-analysis-dir" in options
    assert "--dyn-evidence-report" in options
    assert "--glob" not in options
    assert "--generated-dir" not in options
    assert "--expected-count" not in options


def test_exact_directory_membership_rejects_extra_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "only").write_text("ok")
    _exact_directory_files(root, {"only"}, "fixture")
    (root / "extra").write_text("no")
    with pytest.raises(B2GateError, match="extra"):
        _exact_directory_files(root, {"only"}, "fixture")
    (root / "extra").unlink()
    (root / "link").symlink_to(root / "only")
    with pytest.raises(B2GateError, match="symlinks"):
        _exact_directory_files(root, {"only"}, "fixture")


def test_canonical_dyn_fixture_binds_atlas_and_rejects_stale_authority(
    tmp_path: Path,
) -> None:
    paths, declaration, report = _write_dyn_fixture(tmp_path)
    evidence, budget = _validate_dyn_evidence(
        paths.dyn_evidence_report, _implementation(), declaration, paths
    )
    assert evidence["magic"] == "Q2LAT002"
    assert budget["four_client_feature_assembly_p99_ns"] == 30
    report["authority"]["analyzer_authority_sha256"] = "ff" * 32
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="stale"):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


def test_dyn_fixture_rejects_snapshot_substitution(tmp_path: Path) -> None:
    paths, declaration, _report = _write_dyn_fixture(tmp_path)
    with (paths.dyn_evidence_report.parent / "client2.q2lat002").open("ab") as stream:
        stream.write(b"substitution")
    with pytest.raises(B2GateError, match="bytes differ"):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


def test_dyn_fixture_rejects_fabricated_source_and_executable_provenance(
    tmp_path: Path,
) -> None:
    paths, declaration, report = _write_dyn_fixture(tmp_path)
    report["provenance"]["helper_source_closure"]["inputs"][0]["size_bytes"] += 1
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="current source closure"):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )

    second = tmp_path / "second"
    second.mkdir()
    paths, declaration, report = _write_dyn_fixture(second)
    report["provenance"]["executable"]["sha256"] = "ff" * 32
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="executable"):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


def test_fake_magic_only_dyn_snapshot_is_rejected() -> None:
    with pytest.raises(B2GateError, match="complete Q2LAT002 envelope"):
        _decode_dyn_snapshot(
            b"Q2LAT002" + b"\0" * 8,
            atlas_sha256="11" * 32,
            map_sha256="22" * 32,
            origin=[0, 0, 0],
            map_epoch=1,
            environment_steps=1,
            client_count=4,
        )


def test_test_report_recomputes_log_hash_and_rejects_tamper(tmp_path: Path) -> None:
    report_path = tmp_path / "b2-test-report.json"
    runs = []
    for name, command in _commands("python3"):
        if name == "python":
            payload = b"3 passed\n"
            counts = (3, 0, 0)
        elif name in {"rust-tests", "dyn-tests"}:
            payload = (
                b"test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured\n"
            )
            counts = (2, 0, 0)
        else:
            payload = b""
            counts = (1, 0, 0)
        log = tmp_path / f"{name}.log"
        log.write_bytes(payload)
        runs.append({
            "name": name,
            "command": command,
            "exit_code": 0,
            "passed_count": counts[0],
            "skipped_count": counts[1],
            "ignored_count": counts[2],
            "log": {
                "path": str(log),
                "bytes": log.stat().st_size,
                "sha256": _sha256(payload),
            },
        })
    report = {
        "schema": "q2-b2-test-report-v1",
        "implementation": _implementation(),
        "runs": runs,
        "failures": [],
        "passed": True,
    }
    report_path.write_bytes(canonical_bytes(report))
    assert _validate_test_report(report_path, _implementation())["passed_count"] == 11
    log = tmp_path / "python.log"
    log.write_text("changed\n")
    with pytest.raises(B2GateError, match="identity differs"):
        _validate_test_report(report_path, _implementation())


def test_b2_gate_schemas_are_strict() -> None:
    gate_schema = json.loads(
        (ROOT / "schemas/q2-multires-b2-gate-v1.schema.json").read_text()
    )
    test_schema = json.loads(
        (ROOT / "schemas/q2-b2-test-report-v1.schema.json").read_text()
    )
    assert gate_schema["additionalProperties"] is False
    assert gate_schema["properties"]["generated_cohort"]["properties"][
        "cohort_id"
    ]["const"] == EXPECTED_COHORT
    assert gate_schema["properties"]["representative_budgets"]["properties"][
        "four_client_feature_assembly_p99_ns"
    ]["exclusiveMaximum"] == 500000
    assert test_schema["additionalProperties"] is False
    assert test_schema["properties"]["passed"]["const"] is True


def test_test_evidence_adapter_has_fixed_suites_and_parses_counts() -> None:
    assert [name for name, _command in _commands("python3")] == [
        "python",
        "rust-fmt",
        "rust-clippy",
        "rust-tests",
        "dyn-fmt",
        "dyn-clippy",
        "dyn-tests",
    ]
    assert _parse_counts("python", b"422 passed, 11 skipped in 3.0s\n", 0) == (
        422,
        11,
        0,
    )
    cargo = (
        b"test result: ok. 42 passed; 0 failed; 2 ignored; 0 measured\n"
        b"test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured\n"
    )
    assert _parse_counts("rust-tests", cargo, 0) == (45, 0, 2)


def test_test_evidence_publication_never_replaces_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source")
    (destination / "destination.txt").write_text("destination")
    with pytest.raises(B2TestSuiteError, match="destination appeared"):
        _rename_noreplace(source, destination)
    assert (source / "source.txt").read_text() == "source"
    assert (destination / "destination.txt").read_text() == "destination"


def test_compiled_static_campaign_recomputes_every_declared_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = {
        "maps": [
            {"ordinal": 0, "map": "map_a"},
            {"ordinal": 1, "map": "map_b"},
        ]
    }
    monkeypatch.setattr(
        compiled_static_campaign,
        "load_declaration",
        lambda _path: (declaration, SHA),
    )
    monkeypatch.setattr(
        compiled_static_campaign,
        "verify_stage_membership",
        lambda *_args: {"passed": True, "failures": []},
    )
    visited = []

    def validator(path: Path) -> dict:
        visited.append(path.name)
        return {"map": path.stem, "static_ok": True}

    report = compiled_static_campaign.validate_compiled_static(
        tmp_path / "declaration.json", tmp_path / "compiled", validator=validator
    )
    assert visited == ["map_a.map", "map_b.map"]
    assert report == {
        "schema": "q2-generator-v6-compiled-static-campaign-v1",
        "map_count": 2,
        "pass_count": 2,
        "passed": True,
        "maps": [
            {"map": "map_a", "static_ok": True},
            {"map": "map_b", "static_ok": True},
        ],
    }


def test_compiled_static_campaign_fails_closed_on_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        compiled_static_campaign,
        "load_declaration",
        lambda _path: ({"maps": []}, SHA),
    )
    monkeypatch.setattr(
        compiled_static_campaign,
        "verify_stage_membership",
        lambda *_args: {"passed": False, "failures": ["unexpected file"]},
    )
    with pytest.raises(
        compiled_static_campaign.CompiledStaticCampaignError,
        match="unexpected file",
    ):
        compiled_static_campaign.validate_compiled_static(
            tmp_path / "declaration.json", tmp_path / "compiled"
        )
