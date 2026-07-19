from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import pytest
import zstandard

from harness.hook_claims_v4 import (
    render_runtime_sidecar,
    runtime_records_sha256,
    selected_records_sha256,
    validation_trace_sha256,
    validation_traces_sha256,
)
from harness.atlas_analyzer import _rust_struct_json
import tools.assemble_b2_gate as b2_gate
import tools.run_b2_test_suite as b2_test_suite
import tools.validate_b2_final_cohort_plan as final_plan
from tools.assemble_b2_gate import (
    ACTIVE_71454_QUALIFICATION_SUCCESSOR_PATHS,
    ACTIVE_FINAL_AUTHORITY,
    ActiveFinalAuthority,
    B2GateError,
    B2GatePaths,
    EXPECTED_DESIGN_SHA256,
    RETIRED_71446_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71447_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71448_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71449_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71450_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71451_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71452_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_71453_QUALIFICATION_SUCCESSOR_PATHS,
    RETIRED_COHORT_71446,
    RETIRED_COHORT_71447,
    RETIRED_COHORT_71448,
    RETIRED_COHORT_71449,
    RETIRED_COHORT_71450,
    RETIRED_COHORT_71451,
    RETIRED_COHORT_71452,
    RETIRED_COHORT_71453,
    STOCK_PROVENANCE_COMPACT_SHA256,
    STOCK_PROVENANCE_SHA256,
    assemble_gate,
    _exact_directory_files,
    _decode_dyn_snapshot,
    _dyn_source_authority,
    _historical_71446_rows,
    _preflight_implementation_identity,
    _validate_compiled_cm_preflight,
    _validate_declaration,
    _validate_dyn_evidence,
    _validate_materialized,
    _validate_qualification_successor,
    _validate_source_route_contract,
    _validate_source_spawn_origin_binding,
    _validate_source_spawn_origin_binding_pass_count,
    _validate_test_report,
    validate_preactivation_test_binding,
    _parser,
    _load_stock_provenance,
    _require_active_final_authority,
    validate_gate,
)
from tools.run_generator_cohort import STAGE_SUFFIXES, canonical_bytes
from tools.assemble_b2_qualification import activation_successor_policy
from tools.run_b2_test_suite import (
    B2TestSuiteError,
    _commands,
    _parse_counts,
    _rename_noreplace,
)
import tools.run_compiled_static_campaign as compiled_static_campaign
from tools.source_route_contract import ROUTE_CONTRACT_SCHEMA


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
        declaration=ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json",
        source_dir=dummy,
        source_cold_dir=dummy,
        source_freeze_report=dummy,
        compiled_dir=dummy,
        compiled_membership_report=dummy,
        compiled_static_report=dummy,
        compiled_cm_preflight_report=dummy,
        materialized_dir=dummy,
        materialized_membership_report=dummy,
        claims_dir=tmp_path / "claims",
        claims_prepare_report=dummy,
        analysis_dir=tmp_path / "analysis",
        generated_build_report=dummy,
        generated_validation_report=tmp_path / "generated-promotion.json",
        stock_provenance=dummy,
        stock_inventory=dummy,
        stock_bsp_dir=dummy,
        stock_analysis_dir=dummy,
        stock_validation_dir=dummy,
        dyn_evidence_executable=tmp_path / "q2-dyn-evidence",
        dyn_argv_preflight_report=tmp_path / "dyn-argv-preflight.json",
        dyn_origin_binding_report=tmp_path / "dyn-origin-binding.json",
        dyn_evidence_report=tmp_path / "dyn/b2-dyn-evidence.json",
        preactivation_test_report=dummy,
        test_report=dummy,
        qualification_report=dummy,
        final_lifecycle_evidence=dummy,
        expected_assembly_command_sha256="ab" * 32,
    )


def _selected_hook_record(index: int) -> dict:
    source = [index * 32_000, 0, 24_125]
    anchor = [source[0] + 16_000, 0, 200_000]
    eye = [source[0], source[1], source[2] + 22_000]
    return {
        "claim_id": f"hook:{index:04d}:candidate:0000",
        "source_milliunits": source,
        "trace_target_milliunits": anchor,
        "measured_anchor_milliunits": anchor,
        "landing_milliunits": [source[0] + 16_000, 0, 24_000],
        "release_after_ticks": 2,
        "distance_milliunits": round(math.sqrt(sum(
            (anchor[axis] - eye[axis]) ** 2 for axis in range(3)
        ))),
        "flags": 1,
    }


def _v4_materialization(
    map_id: str, bsp_payload: bytes, source_projection: bytes,
) -> dict:
    records = [_selected_hook_record(index) for index in range(6)]
    traces = []
    for record in records:
        fixed = [[value // 125 for value in record["landing_milliunits"]]]
        traces.append({
            "claim_id": record["claim_id"],
            "origin_fixed_frames": fixed,
            "first_grounded_frame_index": 0,
            "sha256": validation_trace_sha256(record["claim_id"], fixed, 0),
        })
    oracle_records = {
        name: {
            "executable_sha256": format(index + 5, "x") * 64,
            "tool_identity": format(index + 6, "x") * 64,
            "physics_identity": format(index + 7, "x") * 64,
            "requests": index + 1,
        }
        for index, name in enumerate(("collision", "pmove", "hook", "fall"))
    }
    bsp_sha256 = _sha256(bsp_payload)
    seal = {
        "schema": "q2-b1-runtime-authority-seal-v1",
        "normative_documents": {
            "design_sha256": "a" * 64,
            "plan_sha256": "b" * 64,
        },
        "hook_parity_attestation_sha256": "c" * 64,
        "fixture_bsp_sha256": "d" * 64,
        "analysis_bsp_sha256": bsp_sha256,
        "executables": {
            "cm_sha256": oracle_records["collision"]["executable_sha256"],
            "pmove_sha256": oracle_records["pmove"]["executable_sha256"],
            "hook_sha256": oracle_records["hook"]["executable_sha256"],
            "fall_sha256": oracle_records["fall"]["executable_sha256"],
        },
        "identities": {
            name: {
                field: oracle_records[name][field]
                for field in ("tool_identity", "physics_identity")
            }
            for name in ("collision", "pmove", "hook", "fall")
        },
    }
    return {
        "schema": "q2-hook-claim-materialization-v4",
        "map": map_id,
        "passed": True,
        "landing_policy": "compiled-first-grounded-exact-v4",
        "bsp": {"sha256": bsp_sha256, "size_bytes": len(bsp_payload)},
        "candidates": {
            "meta_sha256": "2" * 64,
            "records_sha256": "3" * 64,
            "record_count": 42,
        },
        "source_projection_sha256": _sha256(source_projection),
        "runtime_records_sha256": runtime_records_sha256(records),
        "selected_records": records,
        "validation_traces": traces,
        "oracles": oracle_records | {
            "hook_parity_attestation_sha256": "c" * 64,
            "b1_runtime_authority_seal": seal,
        },
        "fresh_strict_replay": {
            "schema": "q2-hook-fresh-strict-replay-v4",
            "passed": True,
            "record_count": 6,
            "selected_records_sha256": selected_records_sha256(records),
            "validation_traces_sha256": validation_traces_sha256(traces),
            "oracles": {
                name: oracle_records[name] | {"requests": 1}
                for name in ("collision", "pmove", "hook")
            },
        },
        "replay": {
            "analyzer": "q2-hook-claim-materializer",
            "analyzer_version": "b2-c-v4",
            "verifier": "q2-atlas-analyzer-exact-hook-replay",
            "verifier_version": "b2-a-v4",
        },
        "request_count": 13,
    }


def _write_materialized_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[B2GatePaths, dict, dict]:
    compiled = tmp_path / "compiled"
    materialized = tmp_path / "materialized"
    compiled.mkdir()
    materialized.mkdir()
    membership_report = tmp_path / "materialized-membership.json"
    membership_report.write_bytes(canonical_bytes({}))
    paths = replace(
        _paths(tmp_path),
        compiled_dir=compiled,
        materialized_dir=materialized,
        materialized_membership_report=membership_report,
    )
    declaration = {"maps": [{"map": "fixture"}]}
    monkeypatch.setattr(
        b2_gate,
        "_membership_with_declaration",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        b2_gate,
        "_require_report_equals",
        lambda *_args, **_kwargs: None,
    )
    source_projection = b"# non-admissible generator hook projection\n"
    bsp_payload = b"IBSP materialized continuity fixture"
    payloads = {
        ".map": b"map fixture\n",
        ".json": source_projection,
        ".meta.json": canonical_bytes({"fixture": "meta"}),
        ".lattice.json": canonical_bytes({"fixture": "lattice"}),
        ".routes.json": canonical_bytes({"fixture": "routes"}),
        ".bsp": bsp_payload,
    }
    for suffix in STAGE_SUFFIXES["compiled"]:
        (compiled / f"fixture{suffix}").write_bytes(payloads[suffix])
        (materialized / f"fixture{suffix}").write_bytes(payloads[suffix])

    document = _v4_materialization(
        "fixture", bsp_payload, source_projection,
    )
    attestation_payload = canonical_bytes(document)
    attestation = materialized / "fixture.hook-materialization.json"
    attestation.write_bytes(attestation_payload)
    (materialized / "fixture.json").write_bytes(render_runtime_sidecar(
        "fixture",
        _sha256(bsp_payload),
        _sha256(attestation_payload),
        document["selected_records"],
    ))
    return paths, declaration, document


def _materialized_authorities(
    document: dict,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    oracles = document["oracles"]
    seal = oracles["b1_runtime_authority_seal"]
    return (
        {
            "cm": oracles["collision"]["executable_sha256"],
            "pmove": oracles["pmove"]["executable_sha256"],
            "hook": oracles["hook"]["executable_sha256"],
            "fall": oracles["fall"]["executable_sha256"],
            "hook_attestation": oracles["hook_parity_attestation_sha256"],
            "atlas_verifier": "e" * 64,
        },
        {
            "design": {"sha256": seal["normative_documents"]["design_sha256"]},
            "plan": {"sha256": seal["normative_documents"]["plan_sha256"]},
        },
    )


def _validate_materialized_fixture(
    paths: B2GatePaths, declaration: dict, document: dict,
) -> dict:
    binaries, normative = _materialized_authorities(document)
    return _validate_materialized(
        paths, declaration, SHA, binaries, normative,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_evidence(path_text: str, payload: bytes) -> dict:
    return {"path": path_text, "sha256": _sha256(payload), "size_bytes": len(payload)}


def test_materialized_v4_runtime_upgrade_is_exactly_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )

    result = _validate_materialized_fixture(paths, declaration, document)

    attestation = paths.materialized_dir / "fixture.hook-materialization.json"
    assert result["attestation_set_sha256"] == _sha256(canonical_bytes([
        _sha256(attestation.read_bytes())
    ]))
    assert (paths.materialized_dir / "fixture.json").read_bytes() == (
        render_runtime_sidecar(
            "fixture",
            _sha256((paths.materialized_dir / "fixture.bsp").read_bytes()),
            _sha256(attestation.read_bytes()),
            document["selected_records"],
        )
    )


def test_materialized_unchanged_source_projection_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    (paths.materialized_dir / "fixture.json").write_bytes(
        (paths.compiled_dir / "fixture.json").read_bytes()
    )

    with pytest.raises(B2GateError, match="runtime sidecar was not upgraded"):
        _validate_materialized_fixture(paths, declaration, document)


def test_materialized_tampered_runtime_sidecar_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    runtime = paths.materialized_dir / "fixture.json"
    runtime.write_bytes(runtime.read_bytes() + b"tampered\n")

    with pytest.raises(B2GateError, match="invalid V4 runtime sidecar"):
        _validate_materialized_fixture(paths, declaration, document)


@pytest.mark.parametrize("binding", ["bsp", "materialization"])
def test_materialized_wrong_runtime_sidecar_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    attestation = paths.materialized_dir / "fixture.hook-materialization.json"
    bsp_sha256 = _sha256((paths.materialized_dir / "fixture.bsp").read_bytes())
    materialization_sha256 = _sha256(attestation.read_bytes())
    if binding == "bsp":
        bsp_sha256 = "0" * 64
    else:
        materialization_sha256 = "0" * 64
    (paths.materialized_dir / "fixture.json").write_bytes(
        render_runtime_sidecar(
            "fixture", bsp_sha256, materialization_sha256,
            document["selected_records"],
        )
    )

    with pytest.raises(B2GateError, match="invalid V4 runtime sidecar"):
        _validate_materialized_fixture(paths, declaration, document)


def test_materialized_wrong_source_projection_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    document["source_projection_sha256"] = "0" * 64
    (paths.materialized_dir / "fixture.hook-materialization.json").write_bytes(
        canonical_bytes(document)
    )

    with pytest.raises(B2GateError, match="source projection differs"):
        _validate_materialized_fixture(paths, declaration, document)


def test_materialized_wrong_attested_bsp_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    document["bsp"]["sha256"] = "0" * 64
    document["oracles"]["b1_runtime_authority_seal"][
        "analysis_bsp_sha256"
    ] = "0" * 64
    (paths.materialized_dir / "fixture.hook-materialization.json").write_bytes(
        canonical_bytes(document)
    )

    with pytest.raises(B2GateError, match="BSP binding differs"):
        _validate_materialized_fixture(paths, declaration, document)


def test_materialized_wrong_admitted_oracle_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    binaries, normative = _materialized_authorities(document)
    document["oracles"]["hook"]["executable_sha256"] = "0" * 64
    document["fresh_strict_replay"]["oracles"]["hook"][
        "executable_sha256"
    ] = "0" * 64
    document["oracles"]["b1_runtime_authority_seal"]["executables"][
        "hook_sha256"
    ] = "0" * 64
    (paths.materialized_dir / "fixture.hook-materialization.json").write_bytes(
        canonical_bytes(document)
    )

    with pytest.raises(B2GateError, match="oracle differs from admitted B1"):
        _validate_materialized(
            paths, declaration, SHA, binaries, normative,
        )


def test_materialized_wrong_admitted_hook_attestation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    binaries, normative = _materialized_authorities(document)
    document["oracles"]["hook_parity_attestation_sha256"] = "0" * 64
    document["oracles"]["b1_runtime_authority_seal"][
        "hook_parity_attestation_sha256"
    ] = "0" * 64
    (paths.materialized_dir / "fixture.hook-materialization.json").write_bytes(
        canonical_bytes(document)
    )

    with pytest.raises(B2GateError, match="hook parity differs from admitted B1"):
        _validate_materialized(
            paths, declaration, SHA, binaries, normative,
        )


def test_materialized_wrong_normative_seal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    binaries, normative = _materialized_authorities(document)
    document["oracles"]["b1_runtime_authority_seal"][
        "normative_documents"
    ]["design_sha256"] = "0" * 64
    (paths.materialized_dir / "fixture.hook-materialization.json").write_bytes(
        canonical_bytes(document)
    )

    with pytest.raises(
        B2GateError, match="retained B1 normative documents differ",
    ):
        _validate_materialized(
            paths, declaration, SHA, binaries, normative,
        )


@pytest.mark.parametrize(
    "suffix",
    [".map", ".meta.json", ".lattice.json", ".routes.json", ".bsp"],
)
def test_materialized_immutable_compiled_inputs_remain_byte_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    paths, declaration, document = _write_materialized_fixture(
        tmp_path, monkeypatch,
    )
    immutable = paths.materialized_dir / f"fixture{suffix}"
    immutable.write_bytes(immutable.read_bytes() + b"tampered")

    with pytest.raises(
        B2GateError,
        match=r"materialized immutable input changed across stages",
    ):
        _validate_materialized_fixture(paths, declaration, document)


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
    map_id = _historical_71446_rows()[0]["map"]

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
    atlas_manifest = {
        "schema_version": 1,
        "byte_order": "little",
        "atlas_magic": "Q2ATL001",
        "specification_sha256": EXPECTED_DESIGN_SHA256,
        "bsp": {"canonical_map_id": map_id, "sha256": _sha256(bsp)},
        "analyzer": {
            "name": "q2-atlas-analyzer",
            "version": "b2-a-v4",
            "sha256": SHA,
        },
    }
    atlas_manifest_bytes = _rust_struct_json(atlas_manifest) + b"\n"
    assert atlas_manifest_bytes != canonical_bytes(atlas_manifest)
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
            "machine_identity_sha256": _sha256(
                b"0123456789abcdef0123456789abcdef"
            ),
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
    producer_argv_without_origin = [
        str(paths.dyn_evidence_executable),
        "--repo-root",
        str(paths.repo_root),
        "--atlas",
        str(atlas_path),
        "--manifest",
        str(atlas_manifest_path),
        "--bsp",
        str(paths.claims_dir / f"{map_id}.bsp"),
        "--expected-map-id",
        map_id,
        "--expected-analyzer-authority",
        SHA,
        "--expected-crate-commit",
        COMMIT,
        "--map-epoch",
        "1",
        "--environment-steps",
        "17",
        "--samples",
        "2000",
        "--output",
        str(paths.dyn_evidence_report.parent),
    ]
    paths.dyn_argv_preflight_report.write_bytes(
        canonical_bytes(
            {
                "schema": "q2-b2-dyn-argv-shape-preflight-v2",
                "passed": True,
                "repository": _implementation(),
                "executable": _file_evidence(
                    str(paths.dyn_evidence_executable), executable
                ),
                "origin_binding_status": "deferred-until-promoted-artifact",
                "producer_argv_without_origin": producer_argv_without_origin,
                "preflight_argv": [
                    *producer_argv_without_origin,
                    "--preflight-only",
                    "true",
                ],
                "producer_output_absent_before": True,
                "producer_output_absent_after": True,
                "preflight_stdout_sha256": _sha256(
                    canonical_bytes(
                        {
                            "passed": True,
                            "schema": "q2-b2-dyn-argv-shape-preflight-v2",
                        }
                    )
                ),
                "preflight_stderr_sha256": _sha256(b""),
            }
        )
    )
    paths.generated_validation_report.write_bytes(
        canonical_bytes({"fixture": "generated-promotion"})
    )
    origin_ordinal = producer_argv_without_origin.index(
        "--expected-analyzer-authority"
    )
    producer_argv = [
        *producer_argv_without_origin[:origin_ordinal],
        "--expected-origin",
        "0,0,0",
        *producer_argv_without_origin[origin_ordinal:],
    ]
    artifact_paths = {
        "atlas": atlas_path,
        "atlas_manifest": atlas_manifest_path,
        "analysis_manifest": (
            paths.analysis_dir / f"{map_id}.analysis.manifest.json"
        ),
        "bsp": paths.claims_dir / f"{map_id}.bsp",
    }
    paths.dyn_origin_binding_report.write_bytes(
        canonical_bytes(
            {
                "schema": "q2-b2-dyn-origin-binding-v1",
                "passed": True,
                "shape_preflight": _file_evidence(
                    str(paths.dyn_argv_preflight_report),
                    paths.dyn_argv_preflight_report.read_bytes(),
                ),
                "promotion": _file_evidence(
                    str(paths.generated_validation_report),
                    paths.generated_validation_report.read_bytes(),
                ),
                "declaration": _file_evidence(
                    str(paths.declaration), paths.declaration.read_bytes()
                ),
                "repository": _implementation(),
                "executable": _file_evidence(
                    str(paths.dyn_evidence_executable), executable
                ),
                "artifacts": {
                    name: _file_evidence(str(path), path.read_bytes())
                    for name, path in artifact_paths.items()
                },
                "identity": {
                    "canonical_map_id": map_id,
                    "origin": [0, 0, 0],
                    "origin_token": "0,0,0",
                    "analyzer_authority_sha256": SHA,
                    "atlas_sha256": _sha256(atlas_bytes),
                    "atlas_manifest_sha256": _sha256(atlas_manifest_bytes),
                    "analysis_manifest_sha256": _sha256(
                        artifact_paths["analysis_manifest"].read_bytes()
                    ),
                    "bsp_sha256": _sha256(bsp),
                },
                "producer_argv": producer_argv,
                "parser_preflight_argv": [
                    *producer_argv,
                    "--preflight-only",
                    "true",
                ],
                "artifact_preflight_argv": [
                    *producer_argv,
                    "--verify-artifacts-only",
                    "true",
                ],
                "producer_output_absent_before": True,
                "producer_output_absent_after": True,
                "parser_preflight_stdout_sha256": _sha256(
                    canonical_bytes(
                        {
                            "passed": True,
                            "schema": "q2-b2-dyn-argv-shape-preflight-v2",
                        }
                    )
                ),
                "parser_preflight_stderr_sha256": _sha256(b""),
                "artifact_preflight_stdout_sha256": _sha256(
                    canonical_bytes(
                        {
                            "passed": True,
                            "schema": "q2-b2-dyn-artifact-preflight-v1",
                        }
                    )
                ),
                "artifact_preflight_stderr_sha256": _sha256(b""),
            }
        )
    )
    declaration = {
        "cohort_id": RETIRED_COHORT_71446,
        "maps": _historical_71446_rows(),
    }
    return paths, declaration, report


def test_retired_71446_identity_remains_readable_and_cli_is_explicit() -> None:
    rows = _historical_71446_rows()
    assert len(rows) == 28
    assert rows[0] == {
        "ordinal": 0,
        "map": "b2g26_open_71446000",
        "seed": 71446000,
        "style": "open",
        "grid": 5,
        "observed_heat": None,
    }
    assert rows[-1]["map"] == "b2g26_arena_lanes_71446603"
    options = _parser().format_help()
    assert "--declaration" in options
    assert "--source-dir" in options
    assert "--stock-analysis-dir" in options
    assert "--dyn-evidence-report" in options
    assert "--dyn-origin-binding-report" in options
    assert "--compiled-cm-preflight-report" in options
    assert "--preactivation-test-report" in options
    assert "--qualification-report" in options
    assert "--final-lifecycle-evidence" in options
    assert "--glob" not in options
    assert "--generated-dir" not in options
    assert "--expected-count" not in options


@pytest.mark.parametrize(
    ("declaration", "number"),
    [
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71438-DECLARATION.json",
            "71438",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71439-DECLARATION.json",
            "71439",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71440-DECLARATION.json",
            "71440",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71441-DECLARATION.json",
            "71441",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71442-DECLARATION.json",
            "71442",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71443-DECLARATION.json",
            "71443",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71444-DECLARATION.json",
            "71444",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71445-DECLARATION.json",
            "71445",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71446-DECLARATION.json",
            "71446",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71447-DECLARATION.json",
            "71447",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71448-DECLARATION.json",
            "71448",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71449-DECLARATION.json",
            "71449",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71450-DECLARATION.json",
            "71450",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71451-DECLARATION.json",
            "71451",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71452-DECLARATION.json",
            "71452",
        ),
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71453-DECLARATION.json",
            "71453",
        ),
    ],
)
def test_gate_refuses_retired_declaration_before_evidence(
    declaration: Path, number: str,
) -> None:
    with pytest.raises(B2GateError, match=rf"{number}.*permanently retired"):
        _validate_declaration(declaration)


def test_71453_is_retired_and_71454_is_the_active_final_authority() -> None:
    authority = _require_active_final_authority()
    assert ACTIVE_FINAL_AUTHORITY == authority
    assert authority.cohort_id == "b2g26_final_71454"
    assert authority.declaration_sha256 == (
        "8c20d51dd59f1f1cdbdd8171c7d8a75ae98fd68af49fa72992035142134e3986"
    )
    assert authority.immutable_declaration_path == (
        "docs/multires/B2-GENERATED-COHORT-71454-DECLARATION.json"
    )
    assert (
        authority.qualification_successor_paths
        == ACTIVE_71454_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71453 == "b2g26_final_71453"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71453-DECLARATION.json"
        in RETIRED_71453_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71452 == "b2g26_final_71452"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71452-DECLARATION.json"
        in RETIRED_71452_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71451 == "b2g26_final_71451"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71451-DECLARATION.json"
        in RETIRED_71451_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71450 == "b2g26_final_71450"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71450-DECLARATION.json"
        in RETIRED_71450_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71449 == "b2g26_final_71449"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71449-DECLARATION.json"
        in RETIRED_71449_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71448 == "b2g26_final_71448"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71448-DECLARATION.json"
        in RETIRED_71448_QUALIFICATION_SUCCESSOR_PATHS
    )
    assert RETIRED_COHORT_71447 == "b2g26_final_71447"
    assert (
        "docs/multires/B2-GENERATED-COHORT-71447-DECLARATION.json"
        in RETIRED_71447_QUALIFICATION_SUCCESSOR_PATHS
    )


def test_current_alias_is_byte_identical_but_only_immutable_71454_is_active() -> None:
    alias = ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
    immutable = ROOT / "docs/multires/B2-GENERATED-COHORT-71454-DECLARATION.json"
    assert alias.read_bytes() == immutable.read_bytes()
    with pytest.raises(B2GateError, match="explicitly activated immutable path"):
        _validate_declaration(alias)
    declaration, digest = _validate_declaration(immutable)
    assert declaration["cohort_id"] == "b2g26_final_71454"
    assert digest == ACTIVE_FINAL_AUTHORITY.declaration_sha256


def test_gate_requires_the_exact_activated_immutable_declaration_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = json.loads(
        (
            ROOT / "docs/multires/B2-GENERATED-COHORT-71453-DECLARATION.json"
        ).read_bytes()
    )
    declaration["cohort_id"] = "b2g26_final_99001"
    for row in declaration["maps"]:
        row["map"] = f"b2g26_fresh_path_{row['ordinal']:02d}"
        row["seed"] = 99_001_000 + row["ordinal"]
    immutable = tmp_path / "immutable-declaration.json"
    immutable.write_bytes(canonical_bytes(declaration))
    digest = hashlib.sha256(immutable.read_bytes()).hexdigest()
    alias = tmp_path / "byte-identical-alias.json"
    alias.write_bytes(immutable.read_bytes())
    monkeypatch.setattr(
        b2_gate,
        "ACTIVE_FINAL_AUTHORITY",
        ActiveFinalAuthority(
            cohort_id=declaration["cohort_id"],
            declaration_sha256=digest,
            immutable_declaration_path=str(immutable),
            qualification_successor_paths=frozenset({str(immutable)}),
        ),
    )

    with pytest.raises(B2GateError, match="activated immutable path"):
        _validate_declaration(alias)
    loaded, loaded_digest = _validate_declaration(immutable)
    assert loaded == declaration
    assert loaded_digest == digest


def test_real_stock_provenance_uses_exact_committed_writer_bytes() -> None:
    path = ROOT / "docs/multires/stock-q2dm1-q2dm8.provenance.json"
    raw = path.read_bytes()
    provenance = _load_stock_provenance(path)

    assert hashlib.sha256(raw).hexdigest() == STOCK_PROVENANCE_SHA256
    assert len(raw) == 4989
    assert [row["canonical_id"] for row in provenance["records"]] == [
        f"q2dm{number}" for number in range(1, 9)
    ]
    compact = canonical_bytes(provenance)
    assert hashlib.sha256(compact).hexdigest() == STOCK_PROVENANCE_COMPACT_SHA256
    assert len(compact) == 4193
    assert compact != raw


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"schema":"x","schema":"y","records":[]}\n', "duplicate JSON key"),
        (b'{"records":[NaN],"schema":"q2-corpus-provenance-v1"}\n', "non-finite JSON token"),
        (b'{"records":[],"schema":"q2-corpus-provenance-v1"}\n', "writer-canonical"),
    ],
)
def test_stock_provenance_loader_rejects_non_writer_artifacts(
    tmp_path: Path, payload: bytes, message: str,
) -> None:
    path = tmp_path / "provenance.json"
    path.write_bytes(payload)
    with pytest.raises(B2GateError, match=message):
        _load_stock_provenance(path)


def test_stock_provenance_loader_rejects_writer_canonical_semantic_rewrite(
    tmp_path: Path,
) -> None:
    source = ROOT / "docs/multires/stock-q2dm1-q2dm8.provenance.json"
    provenance = json.loads(source.read_bytes())
    provenance["records"][0]["author"] += " rewritten"
    path = tmp_path / "provenance.json"
    path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(B2GateError, match="exact committed digest differs"):
        _load_stock_provenance(path)


def test_qualification_successor_uses_the_frozen_71454_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified = _implementation()
    current = dict(qualified)
    current["repository_commit"] = "ab" * 20
    current["repository_tree"] = "cd" * 20
    diff = "".join(
        f"{'A' if path.endswith('71454-DECLARATION.json') else 'M'}\t{path}\n"
        for path in activation_successor_policy()["allowed_changed_paths"]
    ).encode("utf-8")

    def capture(_root: Path, arguments: list[str]):
        if arguments[0] == "merge-base":
            return b2_gate.subprocess.CompletedProcess(arguments, 0, b"", b"")
        return b2_gate.subprocess.CompletedProcess(arguments, 0, diff, b"")

    monkeypatch.setattr(b2_gate, "_git_capture", capture)
    relation = _validate_qualification_successor(
        ROOT, qualified, current, activation_successor_policy()
    )
    assert relation["stable_authority_equal"] is True
    assert relation["changed_paths"] == sorted(
        activation_successor_policy()["allowed_changed_paths"]
    )


def test_qualification_successor_rejects_unqualified_producer_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified = _implementation()
    current = dict(qualified)
    current["repository_commit"] = "ab" * 20
    current["repository_tree"] = "cd" * 20
    current["generator_sha256"] = "ef" * 32
    monkeypatch.setattr(
        b2_gate,
        "_git_capture",
        lambda *_args: pytest.fail("git must not run after authority drift"),
    )
    with pytest.raises(B2GateError, match="producer/analyzer authority differs"):
        _validate_qualification_successor(
            ROOT,
            qualified,
            current,
            activation_successor_policy(),
        )


def test_qualification_successor_rejects_authority_expanded_extra_tool_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The later authority object cannot self-authorize an extra tool edit."""

    qualified = _implementation()
    current = dict(qualified)
    current["repository_commit"] = "ab" * 20
    current["repository_tree"] = "cd" * 20
    policy = activation_successor_policy()
    extra_path = "tools/unauthorized_activation_tool.py"
    expanded_authority = ActiveFinalAuthority(
        cohort_id=policy["cohort_id"],
        declaration_sha256="00" * 32,
        immutable_declaration_path=policy["immutable_declaration_path"],
        qualification_successor_paths=frozenset(
            [*policy["allowed_changed_paths"], extra_path]
        ),
    )
    diff = "".join(
        f"{'A' if path == policy['immutable_declaration_path'] else 'M'}\t{path}\n"
        for path in [*policy["allowed_changed_paths"], extra_path]
    ).encode("utf-8")

    def capture(_root: Path, arguments: list[str]):
        if arguments[0] == "merge-base":
            return b2_gate.subprocess.CompletedProcess(arguments, 0, b"", b"")
        return b2_gate.subprocess.CompletedProcess(arguments, 0, diff, b"")

    monkeypatch.setattr(b2_gate, "ACTIVE_FINAL_AUTHORITY", expanded_authority)
    monkeypatch.setattr(b2_gate, "_git_capture", capture)
    with pytest.raises(B2GateError, match="changed files differ"):
        _validate_qualification_successor(ROOT, qualified, current, policy)


def _write_compiled_cm_gate_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[B2GatePaths, dict, dict[str, str]]:
    paths = _paths(tmp_path)
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text("{}\n", encoding="utf-8")
    b1_gate = tmp_path / "B1-GATE.json"
    b1_gate.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "compiled-cm.json"
    paths = replace(
        paths,
        compiled_dir=compiled,
        declaration=declaration_path,
        b1_gate=b1_gate,
        compiled_cm_preflight_report=report_path,
    )
    declaration = {
        "cohort_id": RETIRED_COHORT_71446,
        "maps": _historical_71446_rows(),
    }
    membership = {"schema": "fixture-membership", "passed": True, "failures": []}
    monkeypatch.setattr(b2_gate, "verify_stage_membership", lambda *_args: membership)
    binaries = {
        "cm": "01" * 32,
        "cm_tool_identity": "02" * 32,
        "cm_source_closure_sha256": "03" * 32,
    }
    report = {
        "schema": b2_gate.COMPILED_CM_PREFLIGHT_SCHEMA,
        "stage": b2_gate.COMPILED_CM_PREFLIGHT_STAGE,
        "admission_status": b2_gate.COMPILED_CM_PREFLIGHT_STATUS,
        "promotion_authority": False,
        "cohort_id": RETIRED_COHORT_71446,
        "declaration": {
            "path": str(declaration_path.absolute()),
            "sha256": SHA,
            "map_count": 28,
        },
        "compiled_root": str(compiled.absolute()),
        "compiled_membership": {
            "report": membership,
            "report_sha256": _sha256(canonical_bytes(membership)),
        },
        "b1_authority": {
            "gate_sha256": _sha256(b1_gate.read_bytes()),
            "cm_executable_sha256": binaries["cm"],
            "cm_tool_identity": binaries["cm_tool_identity"],
            "cm_source_closure_sha256": binaries["cm_source_closure_sha256"],
        },
        "implementation": _preflight_implementation_identity(ROOT),
        "execution": {
            "parallel_jobs": 4,
            "oracle_batch_timeout_milliseconds": 10_000,
            "map_order": "canonical-declaration-order",
        },
        "checks": {
            "compiled_spawn_origins_exact": True,
            "engine_spawn_link_lift_milliunits": 9000,
            "standing_and_crouched_stationary_hulls": True,
            "support_depth_milliunits": 96_000,
            "oracle_swept_column_minimum_milliunits": 96_000,
            "minimum_spawn_xy_separation_milliunits": 384_000,
            "bounded_basic_escape": True,
            "basic_hazard_containment": True,
            "compiled_lightdata_presence": True,
            "all_to_all_reachability": "deferred-to-full-Atlas-admission",
        },
        "input_stability": {
            "declaration": True,
            "compiled_membership": True,
            "implementation": True,
            "cm_oracle": True,
        },
        "maps": [],
        "map_count": 28,
        "pass_count": 28,
        "failure_count": 0,
        "failures": [],
        "passed": True,
    }
    report["canonical_record_sha256"] = _sha256(canonical_bytes(report))
    return paths, report, binaries


def test_compiled_cm_gate_rejects_old_report_without_hazard_and_lightdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, report, binaries = _write_compiled_cm_gate_fixture(tmp_path, monkeypatch)
    report["checks"].pop("basic_hazard_containment")
    report["checks"].pop("compiled_lightdata_presence")
    paths.compiled_cm_preflight_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="compiled-CM checks keys differ"):
        _validate_compiled_cm_preflight(
            paths,
            {
                "cohort_id": RETIRED_COHORT_71446,
                "maps": _historical_71446_rows(),
            },
            SHA, binaries,
        )


def test_compiled_cm_gate_rejects_unstable_input_before_map_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, report, binaries = _write_compiled_cm_gate_fixture(tmp_path, monkeypatch)
    report["input_stability"]["cm_oracle"] = False
    report["canonical_record_sha256"] = _sha256(canonical_bytes({
        key: value for key, value in report.items()
        if key != "canonical_record_sha256"
    }))
    paths.compiled_cm_preflight_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="input stability failed"):
        _validate_compiled_cm_preflight(
            paths,
            {
                "cohort_id": RETIRED_COHORT_71446,
                "maps": _historical_71446_rows(),
            },
            SHA, binaries,
        )


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


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("schema", "q2-generator-source-route-contract-v1", "schema differs"),
        ("spawn_count", 7, "spawn count differs"),
        (
            "all_spawns_share_source_standing_component",
            False,
            "spawn component failed",
        ),
        (
            "published_dist_covers_endpoint_loop_geometry",
            False,
            "endpoint-loop geometry failed",
        ),
    ],
)
def test_source_route_gate_fails_closed_on_required_predicates(
    field: str, invalid: object, message: str
) -> None:
    route_contract = {
        "schema": ROUTE_CONTRACT_SCHEMA,
        "spawn_count": 8,
        "all_spawns_share_source_standing_component": True,
        "published_dist_covers_endpoint_loop_geometry": True,
        "all_selected_endpoints_share_source_standing_component": True,
        "exact_start_nodes_declared": True,
        "room_edges_used_as_reachability": False,
    }
    _validate_source_route_contract(route_contract, "fixture")
    route_contract[field] = invalid
    with pytest.raises(B2GateError, match=message):
        _validate_source_route_contract(route_contract, "fixture")


def _write_source_spawn_binding_fixture(
    tmp_path: Path,
) -> tuple[Path, dict, dict]:
    origins = [
        [float(128 + index * 64), float(256 + (index % 2) * 64), 24.0]
        for index in range(8)
    ]
    source_map = tmp_path / "fixture.map"
    source_map.write_text(
        "".join(
            "{\n"
            '"classname" "info_player_deathmatch"\n'
            f'"origin" "{origin[0]:g} {origin[1]:g} {origin[2]:g}"\n'
            "}\n"
            for origin in origins
        ),
        encoding="utf-8",
    )
    origins_sha256 = _sha256(canonical_bytes(origins))
    route_contract = {
        "spawn_origins": [list(origin) for origin in origins],
        "spawn_origins_sha256": origins_sha256,
        "spawn_source_component": 3,
        "all_spawn_origins_unique": True,
        "all_spawns_share_source_standing_component": True,
    }
    binding = {
        "schema": "q2-generator-source-spawn-origin-binding-v1",
        "source_artifact": ".map",
        "source_parser": "tools.validate_maps.deathmatch_spawn_origins-v1",
        "deathmatch_spawn_count": 8,
        "spawn_origins": [list(origin) for origin in origins],
        "source_spawn_origins_sha256": origins_sha256,
        "route_spawn_origins_sha256": origins_sha256,
        "route_contract_exact_match": True,
        "all_spawn_origins_unique": True,
        "source_component": 3,
        "all_spawns_share_source_standing_component": True,
    }
    return source_map, route_contract, binding


def test_source_spawn_origin_gate_accepts_exact_artifact_binding(
    tmp_path: Path,
) -> None:
    source_map, route_contract, binding = _write_source_spawn_binding_fixture(
        tmp_path
    )
    _validate_source_spawn_origin_binding_pass_count(28)
    _validate_source_spawn_origin_binding(
        binding, route_contract, source_map, "fixture"
    )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("schema", "retired", "schema differs"),
        ("source_artifact", ".lattice.json", "authority differs"),
        ("source_parser", "unbound-parser", "authority differs"),
        ("deathmatch_spawn_count", 7, "spawn count differs"),
        ("spawn_origins", None, "differ from source map"),
        ("source_spawn_origins_sha256", "00" * 32, "digest binding differs"),
        ("route_spawn_origins_sha256", "11" * 32, "digest binding differs"),
        ("route_contract_exact_match", False, "predicates failed"),
        ("all_spawn_origins_unique", False, "predicates failed"),
        ("source_component", 4, "component binding differs"),
        (
            "all_spawns_share_source_standing_component",
            False,
            "predicates failed",
        ),
    ],
)
def test_source_spawn_origin_gate_rejects_adversarial_binding_fields(
    tmp_path: Path, field: str, invalid: object, message: str
) -> None:
    source_map, route_contract, binding = _write_source_spawn_binding_fixture(
        tmp_path
    )
    if field == "spawn_origins":
        binding[field][0][0] += 1.0
    else:
        binding[field] = invalid
    with pytest.raises(B2GateError, match=message):
        _validate_source_spawn_origin_binding(
            binding, route_contract, source_map, "fixture"
        )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("spawn_origins", None, "differ from source map"),
        ("spawn_origins_sha256", "22" * 32, "digest binding differs"),
        ("spawn_source_component", None, "component binding differs"),
        ("all_spawn_origins_unique", False, "predicates failed"),
        (
            "all_spawns_share_source_standing_component",
            False,
            "predicates failed",
        ),
    ],
)
def test_source_spawn_origin_gate_rejects_adversarial_route_fields(
    tmp_path: Path, field: str, invalid: object, message: str
) -> None:
    source_map, route_contract, binding = _write_source_spawn_binding_fixture(
        tmp_path
    )
    if field == "spawn_origins":
        route_contract[field][0][0] += 1.0
    else:
        route_contract[field] = invalid
    with pytest.raises(B2GateError, match=message):
        _validate_source_spawn_origin_binding(
            binding, route_contract, source_map, "fixture"
        )


def test_source_spawn_origin_gate_rejects_incomplete_aggregate() -> None:
    with pytest.raises(B2GateError, match="bindings are incomplete"):
        _validate_source_spawn_origin_binding_pass_count(27)


def test_source_spawn_origin_gate_rejects_duplicate_source_entities(
    tmp_path: Path,
) -> None:
    source_map, route_contract, binding = _write_source_spawn_binding_fixture(
        tmp_path
    )
    text = source_map.read_text(encoding="utf-8")
    text = text.replace('"origin" "576 320 24"', '"origin" "128 256 24"')
    source_map.write_text(text, encoding="utf-8")
    with pytest.raises(B2GateError, match="origins are not unique"):
        _validate_source_spawn_origin_binding(
            binding, route_contract, source_map, "fixture"
        )


def test_canonical_dyn_fixture_binds_atlas_and_rejects_stale_authority(
    tmp_path: Path,
) -> None:
    paths, declaration, report = _write_dyn_fixture(tmp_path)
    evidence, budget = _validate_dyn_evidence(
        paths.dyn_evidence_report, _implementation(), declaration, paths
    )
    assert evidence["magic"] == "Q2LAT002"
    assert evidence["machine_identity_sha256"] == report["host"][
        "machine_identity_sha256"
    ]
    assert budget["four_client_feature_assembly_p99_ns"] == 30
    report["authority"]["analyzer_authority_sha256"] = "ff" * 32
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="stale"):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


def test_dyn_gate_rejects_noncompact_atlas_manifest_bytes(
    tmp_path: Path,
) -> None:
    paths, declaration, _report = _write_dyn_fixture(tmp_path)
    manifest = json.loads(
        next(paths.analysis_dir.glob("*.atlas.manifest.json")).read_bytes()
    )
    manifest_path = next(paths.analysis_dir.glob("*.atlas.manifest.json"))
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )

    with pytest.raises(
        B2GateError,
        match="representative Atlas manifest format differs",
    ):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


def test_dyn_gate_rejects_rebound_or_guessed_phase_b_origin(tmp_path: Path) -> None:
    paths, declaration, _report = _write_dyn_fixture(tmp_path)
    binding = json.loads(paths.dyn_origin_binding_report.read_bytes())
    binding["identity"]["origin"] = [-512, -512, -512]
    binding["identity"]["origin_token"] = "-512,-512,-512"
    paths.dyn_origin_binding_report.write_bytes(canonical_bytes(binding))

    with pytest.raises(B2GateError, match="artifact-derived origin identity"):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "keys differ"),
        ("malformed", "lowercase SHA-256"),
        ("placeholder", "placeholder"),
    ],
)
def test_dyn_fixture_rejects_missing_or_forged_machine_identity(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    paths, declaration, report = _write_dyn_fixture(tmp_path)
    if mutation == "missing":
        report["host"].pop("machine_identity_sha256")
    elif mutation == "malformed":
        report["host"]["machine_identity_sha256"] = "A" * 64
    else:
        report["host"]["machine_identity_sha256"] = "0" * 64
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match=message):
        _validate_dyn_evidence(
            paths.dyn_evidence_report, _implementation(), declaration, paths
        )


def test_dyn_fixture_records_changed_valid_machine_identity(tmp_path: Path) -> None:
    paths, declaration, report = _write_dyn_fixture(tmp_path)
    changed = _sha256(b"fedcba9876543210fedcba9876543210")
    assert changed != report["host"]["machine_identity_sha256"]
    report["host"]["machine_identity_sha256"] = changed
    paths.dyn_evidence_report.write_bytes(canonical_bytes(report))

    evidence, _budget = _validate_dyn_evidence(
        paths.dyn_evidence_report, _implementation(), declaration, paths
    )

    assert evidence["machine_identity_sha256"] == changed


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


def _write_green_test_report(
    evidence_root: Path,
    implementation: dict,
) -> tuple[Path, dict]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    report_path = evidence_root / "b2-test-report.json"
    runs = []
    cargo_target = (
        evidence_root.parent / f".{evidence_root.name}.cargo-target"
    ).resolve()
    for name, command in _commands("python3", cargo_target):
        if name == "python-syntax-floor":
            payload = canonical_bytes({
                "enumeration": "git-tracked",
                "failures": [],
                "file_count": 5,
                "files_sha256": "ab" * 32,
                "interpreter": {
                    "executable": "/usr/bin/python3",
                    "implementation": "cpython",
                    "sha256": "cd" * 32,
                    "version": [3, 14, 6],
                },
                "passed": True,
                "schema": "q2-python-syntax-floor-v1",
            })
            counts = (5, 0, 0)
        elif name == "python":
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
        log = evidence_root / f"{name}.log"
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
        "schema": b2_test_suite.REPORT_SCHEMA,
        "implementation": implementation,
        "execution_environment": {
            b2_test_suite.CARGO_TARGET_ENV: str(
                cargo_target
            ),
        },
        "runs": runs,
        "failures": [],
        "passed": True,
    }
    report_path.write_bytes(canonical_bytes(report))
    return report_path, report


def test_test_report_recomputes_log_hash_and_rejects_tamper(tmp_path: Path) -> None:
    report_path, report = _write_green_test_report(tmp_path, _implementation())
    assert _validate_test_report(report_path, _implementation())["passed_count"] == 16

    expected_target = Path(
        report["execution_environment"][b2_test_suite.CARGO_TARGET_ENV]
    )
    report["execution_environment"][b2_test_suite.CARGO_TARGET_ENV] = str(
        tmp_path / "wrong-target"
    )
    report_path.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="target path differs"):
        _validate_test_report(report_path, _implementation())
    report["execution_environment"][b2_test_suite.CARGO_TARGET_ENV] = str(
        expected_target
    )

    expected_target.mkdir()
    report_path.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="still exists"):
        _validate_test_report(report_path, _implementation())
    expected_target.rmdir()

    cargo_run = next(run for run in report["runs"] if run["name"] == "rust-fmt")
    original_config = cargo_run["command"][2]
    cargo_run["command"][2] = 'build.target-dir="/tmp/unbound"'
    report_path.write_bytes(canonical_bytes(report))
    with pytest.raises(B2GateError, match="rust-fmt command differs"):
        _validate_test_report(report_path, _implementation())
    cargo_run["command"][2] = original_config
    report_path.write_bytes(canonical_bytes(report))

    log = tmp_path / "python.log"
    log.write_text("changed\n")
    with pytest.raises(B2GateError, match="identity differs"):
        _validate_test_report(report_path, _implementation())


def test_preactivation_tests_bind_the_qualified_not_final_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified = _implementation()
    final = dict(qualified)
    final["repository_commit"] = "ab" * 20
    final["repository_tree"] = "cd" * 20
    report_path, test_report = _write_green_test_report(
        tmp_path / "preactivation-tests", qualified
    )
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_bytes(b"qualification fixture\n")
    replayed = {
        "implementation": qualified,
        "qualification_id": "b2q26_preactivation_fixture",
        "end_to_end": {"pass_count": 28, "required_pass_count": 20},
        "activation_successor_policy": activation_successor_policy(),
    }
    relation = {
        "qualified_repository_commit": qualified["repository_commit"],
        "qualified_repository_tree": qualified["repository_tree"],
        "final_repository_commit": final["repository_commit"],
        "final_repository_tree": final["repository_tree"],
        "stable_authority_equal": True,
        "changed_paths": ["fixture-declaration.json"],
    }
    monkeypatch.setattr(
        b2_gate,
        "_replay_final_qualification",
        lambda **_kwargs: (replayed, relation),
    )
    authority = ActiveFinalAuthority(
        cohort_id="b2g26_final_99000",
        declaration_sha256="ef" * 32,
        immutable_declaration_path="fixture-declaration.json",
        qualification_successor_paths=frozenset({"fixture-declaration.json"}),
    )
    kwargs = {
        "design": ROOT / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md",
        "plan": ROOT / "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md",
        "repo_root": ROOT,
        "b1_gate": ROOT / "docs/multires/B1-GATE.json",
        "qualification_report": qualification_path,
        "preactivation_test_report": report_path,
        "implementation": final,
        "authority": authority,
    }
    binding = validate_preactivation_test_binding(**kwargs)
    assert binding["preactivation_tests"]["run_count"] == 8
    assert binding["preactivation_tests"]["passed_count"] == 16
    assert binding["activation_successor_policy"] == activation_successor_policy()

    replayed["activation_successor_policy"]["allowed_changed_paths"].append(
        "tools/unauthorized_activation_tool.py"
    )
    with pytest.raises(B2GateError, match="activation-successor policy"):
        validate_preactivation_test_binding(**kwargs)
    replayed["activation_successor_policy"] = activation_successor_policy()

    test_report["implementation"] = final
    report_path.write_bytes(canonical_bytes(test_report))
    with pytest.raises(B2GateError, match="implementation is stale"):
        validate_preactivation_test_binding(**kwargs)


def _write_final_lifecycle_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[B2GatePaths, dict, ActiveFinalAuthority, dict]:
    """Create a canonical preassembly record with host I/O stubbed at its edge."""

    declaration_path = tmp_path / "B2-GENERATED-COHORT-99000-DECLARATION.json"
    declaration_path.write_bytes(b"immutable declaration fixture\n")
    declaration_sha256 = _sha256(declaration_path.read_bytes())
    implementation = _implementation()
    authority = ActiveFinalAuthority(
        cohort_id="b2g26_final_99000",
        declaration_sha256=declaration_sha256,
        immutable_declaration_path=str(declaration_path),
        qualification_successor_paths=frozenset({
            str(declaration_path),
        }),
    )
    source = tmp_path / "source"
    source_cold = tmp_path / "source-cold"
    source_report = tmp_path / "source-freeze.json"
    marker_path = tmp_path / "state" / f"{declaration_sha256}.json"
    marker_path.parent.mkdir()
    marker_path.write_bytes(canonical_bytes({"marker": "fixture"}))
    binding = {
        "schema": "q2-b2-final-execution-binding-v2",
        "host": {
            "hostname": "DESKTOP-RTX2080",
            "kernel_release": "5.15.153.1-microsoft-standard-WSL2",
            "machine_identity": {
                "path": "/etc/machine-id",
                "sha256": "ab" * 32,
            },
            "euid": 1000,
        },
        "state_root": {
            "path": str(marker_path.parent),
            "owner_uid": 1000,
            "mode": "0700",
            "device": 1,
            "inode": 2,
        },
    }
    command_sha256 = "cd" * 32
    stage_executions = [
        {
            "stage": stage,
            "command_sha256": "ef" * 32,
            "returncode": 0,
            "stdout": {"bytes": 0, "sha256": "01" * 32},
            "stderr": {"bytes": 0, "sha256": "02" * 32},
        }
        for stage in b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES
    ]
    lifecycle = {
        "schema": b2_gate.FINAL_LIFECYCLE_EVIDENCE_SCHEMA,
        "status": "ready-for-assembly",
        "cohort_id": authority.cohort_id,
        "declaration": {
            "path": str(declaration_path),
            "sha256": declaration_sha256,
        },
        "plan_sha256": "03" * 32,
        "implementation": implementation,
        "execution_binding": binding,
        "source_authorization_marker": {
            "path": str(marker_path),
            **b2_gate._file_record(marker_path),
        },
        "completed_stages": list(b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES),
        "stage_executions": stage_executions,
        "assembly_command_sha256": command_sha256,
    }
    evidence_path = tmp_path / "lifecycle-preassembly.json"
    evidence_path.write_bytes(canonical_bytes(lifecycle))
    paths = replace(
        _paths(tmp_path),
        declaration=declaration_path,
        source_dir=source,
        source_cold_dir=source_cold,
        source_freeze_report=source_report,
        final_lifecycle_evidence=evidence_path,
        expected_assembly_command_sha256=command_sha256,
    )
    seen: dict[str, object] = {}

    def validate(marker, **kwargs):
        seen["marker"] = marker
        seen.update(kwargs)
        return {
            "plan_sha256": lifecycle["plan_sha256"],
            "execution_binding": binding,
            "implementation": implementation,
        }

    monkeypatch.setattr(b2_gate, "validate_final_source_authorization", validate)
    return paths, lifecycle, authority, seen


def test_final_lifecycle_evidence_binds_the_completed_prefix_and_source_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, lifecycle, authority, seen = _write_final_lifecycle_fixture(
        tmp_path, monkeypatch
    )
    declaration = {"cohort_id": authority.cohort_id}
    summary = b2_gate._validate_final_lifecycle_evidence(
        paths, declaration, authority.declaration_sha256, _implementation(), authority
    )

    assert summary["evidence"] == b2_gate._file_record(
        paths.final_lifecycle_evidence
    )
    assert summary["completed_stages"] == list(
        b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES
    )
    assert seen["marker"] == Path(lifecycle["source_authorization_marker"]["path"])
    assert seen["declaration_path"] == paths.declaration
    assert seen["source_output"] == paths.source_dir
    assert seen["source_cold"] == paths.source_cold_dir
    assert seen["source_report"] == paths.source_freeze_report
    assert seen["implementation"] == _implementation()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["completed_stages"].pop(),
            "completed-stage prefix/order differs",
        ),
        (
            lambda record: record["stage_executions"][3].update({"returncode": 1}),
            "did not exit zero",
        ),
        (
            lambda record: record.update({"assembly_command_sha256": "ff" * 32}),
            "assembly command differs",
        ),
    ],
)
def test_final_lifecycle_evidence_rejects_incomplete_or_rebound_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    paths, lifecycle, authority, _seen = _write_final_lifecycle_fixture(
        tmp_path, monkeypatch
    )
    mutate(lifecycle)
    paths.final_lifecycle_evidence.write_bytes(canonical_bytes(lifecycle))
    with pytest.raises(B2GateError, match=message):
        b2_gate._validate_final_lifecycle_evidence(
            paths, {"cohort_id": authority.cohort_id},
            authority.declaration_sha256, _implementation(), authority,
        )


def test_final_lifecycle_evidence_rejects_marker_file_record_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, lifecycle, authority, _seen = _write_final_lifecycle_fixture(
        tmp_path, monkeypatch
    )
    lifecycle["source_authorization_marker"]["sha256"] = "ff" * 32
    paths.final_lifecycle_evidence.write_bytes(canonical_bytes(lifecycle))
    with pytest.raises(B2GateError, match="marker file record differs"):
        b2_gate._validate_final_lifecycle_evidence(
            paths, {"cohort_id": authority.cohort_id},
            authority.declaration_sha256, _implementation(), authority,
        )


def test_final_lifecycle_gate_summary_rejects_missing_prefix_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, lifecycle, authority, _seen = _write_final_lifecycle_fixture(
        tmp_path, monkeypatch
    )
    summary = b2_gate._validate_final_lifecycle_evidence(
        paths, {"cohort_id": authority.cohort_id},
        authority.declaration_sha256, _implementation(), authority,
    )
    summary["completed_stages"].pop()
    with pytest.raises(B2GateError, match="completed-stage prefix/order differs"):
        b2_gate._validate_gate_final_lifecycle_summary(
            summary, authority=authority, implementation=_implementation()
        )


def test_final_lifecycle_gate_summary_requires_v2_wsl_execution_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _lifecycle, authority, _seen = _write_final_lifecycle_fixture(
        tmp_path, monkeypatch
    )
    summary = b2_gate._validate_final_lifecycle_evidence(
        paths, {"cohort_id": authority.cohort_id},
        authority.declaration_sha256, _implementation(), authority,
    )
    summary["execution_binding"]["schema"] = "q2-b2-final-execution-binding-v1"
    with pytest.raises(B2GateError, match="execution binding schema differs"):
        b2_gate._validate_gate_final_lifecycle_summary(
            summary, authority=authority, implementation=_implementation()
        )


def test_actual_assembly_command_digest_binds_ordered_argv() -> None:
    argv = ["--design", "/tmp/design", "--output", "/tmp/gate"]
    expected = _sha256(canonical_bytes([
        str(Path(sys.executable).resolve()),
        str(Path(b2_gate.__file__).resolve()),
        *argv,
    ]))
    assert b2_gate._actual_assembly_command_sha256(argv) == expected
    assert b2_gate._actual_assembly_command_sha256(list(reversed(argv))) != expected


def test_b2_gate_schemas_are_strict() -> None:
    gate_schema = json.loads(
        (ROOT / "schemas/q2-multires-b2-gate-v1.schema.json").read_text()
    )
    test_schema = json.loads(
        (ROOT / "schemas/q2-b2-test-report-v2.schema.json").read_text()
    )
    assert gate_schema["additionalProperties"] is False
    assert "not" not in gate_schema
    cohort_schema = gate_schema["properties"]["generated_cohort"]["properties"][
        "cohort_id"
    ]
    assert cohort_schema["const"] == "b2g26_final_71454"
    assert gate_schema["properties"]["generated_cohort"]["properties"][
        "declaration_sha256"
    ]["const"] == (
        "8c20d51dd59f1f1cdbdd8171c7d8a75ae98fd68af49fa72992035142134e3986"
    )
    assert gate_schema["properties"]["generated_cohort"]["properties"][
        "compiled_cm_preflight"
    ]["$ref"] == "#/$defs/compiled_cm_preflight_stage"
    assert gate_schema["properties"]["toolchain_qualification"][
        "$ref"
    ] == "#/$defs/toolchain_qualification"
    assert "final_lifecycle" in gate_schema["required"]
    assert gate_schema["properties"]["final_lifecycle"][
        "$ref"
    ] == "#/$defs/final_lifecycle"
    assert gate_schema["$defs"]["final_lifecycle"]["required"] == [
        "evidence",
        "schema",
        "status",
        "cohort_id",
        "declaration",
        "plan_sha256",
        "implementation",
        "execution_binding",
        "source_authorization_marker",
        "completed_stages",
        "stage_executions",
        "assembly_command_sha256",
    ]
    assert gate_schema["$defs"]["final_lifecycle"]["properties"][
        "status"
    ]["const"] == "ready-for-assembly"
    assert gate_schema["$defs"]["final_lifecycle"]["properties"][
        "stage_executions"
    ]["minItems"] == len(b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES)
    assert gate_schema["properties"]["dyn_evidence"]["properties"][
        "argv_preflight"
    ]["required"] == ["shape_preflight", "origin_binding"]
    assert gate_schema["$defs"]["toolchain_qualification"]["properties"][
        "non_admissible"
    ]["const"] is True
    assert "implementation_successor" in gate_schema["$defs"][
        "toolchain_qualification"
    ]["required"]
    assert "activation_successor_policy" in gate_schema["$defs"][
        "toolchain_qualification"
    ]["required"]
    assert "preactivation_tests" in gate_schema["$defs"][
        "toolchain_qualification"
    ]["required"]
    assert gate_schema["$defs"]["toolchain_qualification"]["properties"][
        "preactivation_tests"
    ]["$ref"] == "#/properties/tests"
    assert gate_schema["$defs"]["toolchain_qualification"]["properties"][
        "implementation_successor"
    ]["properties"]["changed_paths"]["const"] == activation_successor_policy()[
        "allowed_changed_paths"
    ]
    assert gate_schema["$defs"]["toolchain_qualification"]["properties"][
        "activation_successor_policy"
    ]["$ref"] == "#/$defs/activation_successor_policy"
    assert gate_schema["$defs"]["activation_successor_policy"]["properties"][
        "allowed_changed_paths"
    ]["const"] == activation_successor_policy()["allowed_changed_paths"]
    assert gate_schema["$defs"]["compiled_cm_preflight_stage"]["properties"][
        "pass_count"
    ]["const"] == 28
    assert gate_schema["properties"]["representative_budgets"]["properties"][
        "four_client_feature_assembly_p99_ns"
    ]["exclusiveMaximum"] == 500000
    assert test_schema["additionalProperties"] is False
    assert test_schema["properties"]["passed"]["const"] is True
    assert test_schema["properties"]["runs"]["minItems"] == 8
    assert test_schema["properties"]["runs"]["maxItems"] == 8


def test_final_lifecycle_stage_contract_is_shared_by_driver_gate_and_schema() -> None:
    gate_schema = json.loads(
        (ROOT / "schemas/q2-multires-b2-gate-v1.schema.json").read_text()
    )
    expected = list(final_plan.PREASSEMBLY_LIFECYCLE_STAGES)
    assert expected == list(b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES)
    assert gate_schema["$defs"]["final_lifecycle"]["properties"][
        "completed_stages"
    ]["const"] == expected


def test_test_evidence_adapter_has_fixed_suites_and_parses_counts() -> None:
    cargo_target = Path("/tmp/b2-test-target")
    commands = _commands("python3", cargo_target)
    assert [name for name, _command in commands] == [
        "python-syntax-floor",
        "python",
        "rust-fmt",
        "rust-clippy",
        "rust-tests",
        "dyn-fmt",
        "dyn-clippy",
        "dyn-tests",
    ]
    cargo_config = "build.target-dir=" + json.dumps(str(cargo_target))
    for name, command in commands:
        if name.startswith("rust-") or name.startswith("dyn-"):
            assert command[:3] == ["cargo", "--config", cargo_config]
    syntax = canonical_bytes({
        "enumeration": "git-tracked",
        "failures": [],
        "file_count": 17,
        "files_sha256": "ab" * 32,
        "interpreter": {
            "executable": "/usr/bin/python3",
            "implementation": "cpython",
            "sha256": "cd" * 32,
            "version": [3, 14, 6],
        },
        "passed": True,
        "schema": "q2-python-syntax-floor-v1",
    })
    assert _parse_counts("python-syntax-floor", syntax, 0) == (17, 0, 0)
    assert _parse_counts("python", b"422 passed, 11 skipped in 3.0s\n", 0) == (
        422,
        11,
        0,
    )
    collection_failure = b"no tests collected, 6 errors in 0.66s\n"
    assert _parse_counts("python", collection_failure, 2) == (0, 0, 0)
    with pytest.raises(B2TestSuiteError, match="successful pytest log"):
        _parse_counts("python", collection_failure, 0)
    ambiguous = b"21 passed in 1.0s\n22 passed in 2.0s\n"
    with pytest.raises(B2TestSuiteError, match="ambiguous pass summaries"):
        _parse_counts("python", ambiguous, 0)
    with pytest.raises(B2TestSuiteError, match="ambiguous pass summaries"):
        _parse_counts("python", ambiguous, 1)
    assert _parse_counts("python", b"3 failed, 19 passed in 2.0s\n", 1) == (
        19,
        0,
        0,
    )
    cargo = (
        b"test result: ok. 42 passed; 0 failed; 2 ignored; 0 measured\n"
        b"test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured\n"
    )
    assert _parse_counts("rust-tests", cargo, 0) == (45, 0, 2)


def test_failed_pytest_evidence_is_published_instead_of_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "b2-tests"
    implementation = _implementation()
    collection_failure = b"no tests collected, 6 errors in 0.66s\n"
    suite_environments = []
    monkeypatch.setattr(
        b2_test_suite, "repository_binding", lambda _root: implementation,
    )
    monkeypatch.setattr(
        b2_test_suite, "_commands",
        lambda _python, _cargo_target: (
            ("python", ["python3", "-m", "pytest", "-q"]),
        ),
    )

    def fake_run(
        command: list[str], **_kwargs: object,
    ) -> b2_test_suite.subprocess.CompletedProcess[bytes]:
        if command[1:3] == ["-B", "-c"]:
            return b2_test_suite.subprocess.CompletedProcess(
                args=command, returncode=0, stdout=b"",
            )
        suite_environments.append(_kwargs["env"])
        assert (tmp_path / ".b2-tests.cargo-target").is_dir()
        return b2_test_suite.subprocess.CompletedProcess(
            args=command, returncode=2, stdout=collection_failure,
        )

    monkeypatch.setattr(b2_test_suite.subprocess, "run", fake_run)

    report = b2_test_suite.run_suite(output, python="python3")

    assert report["passed"] is False
    assert report["failures"] == ["python: exit 2"]
    assert report["schema"] == b2_test_suite.REPORT_SCHEMA
    assert report["execution_environment"] == {
        b2_test_suite.CARGO_TARGET_ENV: str(
            (tmp_path / ".b2-tests.cargo-target").resolve()
        ),
    }
    assert len(suite_environments) == 1
    assert suite_environments[0][b2_test_suite.CARGO_TARGET_ENV] == str(
        (tmp_path / ".b2-tests.cargo-target").resolve()
    )
    assert report["runs"][0]["passed_count"] == 0
    assert (output / "python.log").read_bytes() == collection_failure
    assert json.loads((output / b2_test_suite.REPORT_NAME).read_bytes()) == report
    assert not list(tmp_path.glob(".b2-tests.partial-*"))
    assert not (tmp_path / ".b2-tests.cargo-target").exists()


@pytest.mark.parametrize("missing", ["pytest", "zstandard", "torch"])
def test_python_dependency_preflight_fails_before_suite_or_evidence_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    output = tmp_path / "b2-tests"
    implementation = _implementation()
    suite_commands_created = False
    preflight_commands = []

    monkeypatch.setattr(
        b2_test_suite, "repository_binding", lambda _root: implementation,
    )

    def reject_suite_commands(
        _python: str, _cargo_target: Path,
    ) -> tuple[tuple[str, list[str]], ...]:
        nonlocal suite_commands_created
        suite_commands_created = True
        return ()

    def fail_preflight(
        command: list[str], **_kwargs: object,
    ) -> b2_test_suite.subprocess.CompletedProcess[bytes]:
        preflight_commands.append(command)
        return b2_test_suite.subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout=f"missing Python dependencies: {missing}\n".encode(),
        )

    monkeypatch.setattr(b2_test_suite, "_commands", reject_suite_commands)
    monkeypatch.setattr(b2_test_suite.subprocess, "run", fail_preflight)

    with pytest.raises(
        B2TestSuiteError,
        match=rf"dependency preflight failed.*{missing}",
    ):
        b2_test_suite.run_suite(output, python="/chosen/python")

    assert len(preflight_commands) == 1
    assert preflight_commands[0][:3] == ["/chosen/python", "-B", "-c"]
    preflight_source = preflight_commands[0][3]
    assert 'importlib.import_module(dependency)' in preflight_source
    assert '("pytest", "zstandard", "torch")' in preflight_source
    assert suite_commands_created is False
    assert not output.exists()
    assert not list(tmp_path.glob(".b2-tests.partial-*"))
    assert not (tmp_path / ".b2-tests.cargo-target").exists()


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
