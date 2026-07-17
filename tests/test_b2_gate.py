from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest
import zstandard

from harness.hook_claims_v4 import (
    render_runtime_sidecar,
    runtime_records_sha256,
    selected_records_sha256,
    validation_trace_sha256,
    validation_traces_sha256,
)
import tools.assemble_b2_gate as b2_gate
import tools.run_b2_test_suite as b2_test_suite
from tools.assemble_b2_gate import (
    B2GateError,
    B2GatePaths,
    EXPECTED_COHORT,
    EXPECTED_DESIGN_SHA256,
    QUALIFICATION_SUCCESSOR_PATHS,
    _exact_directory_files,
    _decode_dyn_snapshot,
    _dyn_source_authority,
    _expected_71446_rows,
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
    _parser,
)
from tools.run_generator_cohort import STAGE_SUFFIXES, canonical_bytes
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
        declaration=dummy,
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
        generated_validation_report=dummy,
        stock_provenance=dummy,
        stock_inventory=dummy,
        stock_bsp_dir=dummy,
        stock_analysis_dir=dummy,
        stock_validation_dir=dummy,
        dyn_evidence_executable=tmp_path / "q2-dyn-evidence",
        dyn_evidence_report=tmp_path / "dyn/b2-dyn-evidence.json",
        test_report=dummy,
        qualification_report=dummy,
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
    map_id = _expected_71446_rows()[0]["map"]

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
    declaration = {"cohort_id": EXPECTED_COHORT, "maps": _expected_71446_rows()}
    return paths, declaration, report


def test_fresh_71446_identity_and_cli_contract_are_exact() -> None:
    rows = _expected_71446_rows()
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
    assert "--compiled-cm-preflight-report" in options
    assert "--qualification-report" in options
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
    ],
)
def test_gate_refuses_retired_declaration_before_evidence(
    declaration: Path, number: str,
) -> None:
    with pytest.raises(B2GateError, match=rf"{number}.*permanently retired"):
        _validate_declaration(declaration)


def test_gate_accepts_fresh_current_71446_alias() -> None:
    declaration, _sha256 = _validate_declaration(
        ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
    )
    assert declaration["cohort_id"] == EXPECTED_COHORT
    assert declaration["maps"] == _expected_71446_rows()


def test_qualification_successor_accepts_only_the_declared_authorization_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified = _implementation()
    current = dict(qualified)
    current["repository_commit"] = "ab" * 20
    current["repository_tree"] = "cd" * 20
    diff = "".join(
        f"{'A' if path.endswith('71446-DECLARATION.json') else 'M'}\t{path}\n"
        for path in sorted(QUALIFICATION_SUCCESSOR_PATHS)
    ).encode("utf-8")

    def capture(_root: Path, arguments: list[str]):
        if arguments[0] == "merge-base":
            return b2_gate.subprocess.CompletedProcess(arguments, 0, b"", b"")
        return b2_gate.subprocess.CompletedProcess(arguments, 0, diff, b"")

    monkeypatch.setattr(b2_gate, "_git_capture", capture)
    relation = _validate_qualification_successor(ROOT, qualified, current)
    assert relation["stable_authority_equal"] is True
    assert relation["changed_paths"] == sorted(QUALIFICATION_SUCCESSOR_PATHS)


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
        _validate_qualification_successor(ROOT, qualified, current)


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
    declaration = {"cohort_id": EXPECTED_COHORT, "maps": _expected_71446_rows()}
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
        "cohort_id": EXPECTED_COHORT,
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
            paths, {"cohort_id": EXPECTED_COHORT, "maps": _expected_71446_rows()},
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
            paths, {"cohort_id": EXPECTED_COHORT, "maps": _expected_71446_rows()},
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
    assert _validate_test_report(report_path, _implementation())["passed_count"] == 16
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
    assert gate_schema["properties"]["generated_cohort"]["properties"][
        "compiled_cm_preflight"
    ]["$ref"] == "#/$defs/compiled_cm_preflight_stage"
    assert gate_schema["properties"]["toolchain_qualification"][
        "$ref"
    ] == "#/$defs/toolchain_qualification"
    assert gate_schema["$defs"]["toolchain_qualification"]["properties"][
        "non_admissible"
    ]["const"] is True
    assert "implementation_successor" in gate_schema["$defs"][
        "toolchain_qualification"
    ]["required"]
    assert gate_schema["$defs"]["toolchain_qualification"]["properties"][
        "implementation_successor"
    ]["properties"]["changed_paths"]["minItems"] == len(
        QUALIFICATION_SUCCESSOR_PATHS
    )
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


def test_test_evidence_adapter_has_fixed_suites_and_parses_counts() -> None:
    assert [name for name, _command in _commands("python3")] == [
        "python-syntax-floor",
        "python",
        "rust-fmt",
        "rust-clippy",
        "rust-tests",
        "dyn-fmt",
        "dyn-clippy",
        "dyn-tests",
    ]
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
    monkeypatch.setattr(
        b2_test_suite, "repository_binding", lambda _root: implementation,
    )
    monkeypatch.setattr(
        b2_test_suite, "_commands",
        lambda _python: (("python", ["python3", "-m", "pytest", "-q"]),),
    )

    def fake_run(
        command: list[str], **_kwargs: object,
    ) -> b2_test_suite.subprocess.CompletedProcess[bytes]:
        if command[1:3] == ["-B", "-c"]:
            return b2_test_suite.subprocess.CompletedProcess(
                args=command, returncode=0, stdout=b"",
            )
        return b2_test_suite.subprocess.CompletedProcess(
            args=command, returncode=2, stdout=collection_failure,
        )

    monkeypatch.setattr(b2_test_suite.subprocess, "run", fake_run)

    report = b2_test_suite.run_suite(output, python="python3")

    assert report["passed"] is False
    assert report["failures"] == ["python: exit 2"]
    assert report["runs"][0]["passed_count"] == 0
    assert (output / "python.log").read_bytes() == collection_failure
    assert json.loads((output / b2_test_suite.REPORT_NAME).read_bytes()) == report
    assert not list(tmp_path.glob(".b2-tests.partial-*"))


@pytest.mark.parametrize("missing", ["pytest", "zstandard"])
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
        _python: str,
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
    assert '("pytest", "zstandard")' in preflight_source
    assert suite_commands_created is False
    assert not output.exists()
    assert not list(tmp_path.glob(".b2-tests.partial-*"))


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
