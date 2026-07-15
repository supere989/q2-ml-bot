from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from harness.hook_claims_v2 import (
    HookClaimsV2Error,
    canonical_json as hook_canonical_json,
    load_candidates,
    render_runtime_sidecar,
    runtime_records_sha256,
    sha256_bytes,
    validate_materialization,
    validation_trace_sha256,
)
from maps.generator import generate_map
from tools.generator_claim_validator import (
    ClaimValidationError,
    _canonical_semantic_analysis_manifest,
    _expected_analyzer_sha256,
    build_generator_claims,
    canonical_bytes,
    file_sha256,
    generator_claims_sha256,
    validate_generated_map,
    validate_generator_claims,
    validate_stock_analysis,
)
from tools.validate_maps import Q2_BSP_LIGHTING_LUMP, Q2_BSP_LUMPS
from tools.run_generator_claim_campaign import prepare_claims


ROOT = Path(__file__).resolve().parents[1]
NONZERO = "11" * 32


def _write_compiled_bsp(map_path: Path) -> Path:
    header = bytearray(8 + Q2_BSP_LUMPS * 8)
    struct.pack_into("<4sI", header, 0, b"IBSP", 38)
    lightdata = bytes(range(256)) * 16
    struct.pack_into(
        "<II",
        header,
        8 + Q2_BSP_LIGHTING_LUMP * 8,
        len(header),
        len(lightdata),
    )
    bsp_path = map_path.with_suffix(".bsp")
    bsp_path.write_bytes(header + lightdata)
    return bsp_path


def _fake_materialize(map_path: Path) -> None:
    """Create structurally valid proof fixtures; physics is tested elsewhere."""
    stem = map_path.with_suffix("")
    meta_path = stem.with_suffix(".meta.json")
    runtime_path = stem.with_suffix(".json")
    bsp_path = stem.with_suffix(".bsp")
    candidates, candidate_digest, meta_digest = load_candidates(meta_path)
    source_projection_digest = file_sha256(runtime_path)
    selected = []
    geometries = set()
    for record in candidates["records"]:
        geometry = (
            tuple(record["anchor_milliunits"]),
            tuple(record["landing_milliunits"]), record["flags"],
        )
        if geometry in geometries:
            continue
        geometries.add(geometry)
        selected.append(record)
        if len(selected) == 6:
            break
    assert len(selected) == 6
    traces = []
    for record in selected:
        frames = [[value // 125 for value in record["landing_milliunits"]]]
        traces.append({
            "claim_id": record["claim_id"],
            "origin_fixed_frames": frames,
            "first_grounded_frame_index": 0,
            "sha256": validation_trace_sha256(record["claim_id"], frames, 0),
        })
    b1_gate = json.loads((ROOT / "docs/multires/B1-GATE.json").read_text())
    accepted_hook_attestation = b1_gate["artifacts"]["hook_parity_attestation"][
        "sha256"
    ]
    oracle_records = {
        name: {
            "executable_sha256": format(index + 1, "x") * 64,
            "tool_identity": format(index + 2, "x") * 64,
            "physics_identity": format(index + 3, "x") * 64,
            "requests": 1,
        }
        for index, name in enumerate(("collision", "pmove", "hook", "fall"))
    }
    b1_seal = {
        "schema": "q2-b1-runtime-authority-seal-v1",
        "normative_documents": dict(b1_gate["normative_documents"]),
        "hook_parity_attestation_sha256": accepted_hook_attestation,
        "fixture_bsp_sha256": b1_gate["artifacts"]["hook_parity_attestation"][
            "fixture_bsp_sha256"
        ],
        "analysis_bsp_sha256": file_sha256(bsp_path),
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
    document = validate_materialization({
        "schema": "q2-hook-claim-materialization-v2",
        "map": map_path.stem, "passed": True,
        "bsp": {"sha256": file_sha256(bsp_path), "size_bytes": bsp_path.stat().st_size},
        "candidates": {
            "meta_sha256": meta_digest, "records_sha256": candidate_digest,
            "record_count": len(candidates["records"]),
        },
        "source_projection_sha256": source_projection_digest,
        "runtime_records_sha256": runtime_records_sha256(selected),
        "selected_records": selected,
        "validation_traces": traces,
        "oracles": oracle_records | {
            "hook_parity_attestation_sha256": accepted_hook_attestation,
            "b1_runtime_authority_seal": b1_seal,
        },
        "replay": {
            "analyzer": "q2-hook-claim-materializer",
            "analyzer_version": "b2-c-v2",
            "verifier": "q2-atlas-analyzer-exact-hook-replay",
            "verifier_version": "b2-a-v2",
        },
        "request_count": 4,
    })
    payload = hook_canonical_json(document) + b"\n"
    digest = sha256_bytes(payload)
    Path(f"{stem}.hook-materialization.json").write_bytes(payload)
    runtime_path.write_bytes(render_runtime_sidecar(
        map_path.stem, file_sha256(bsp_path), digest, selected,
    ))


def _route_results(claims: dict) -> list[dict]:
    costs = {}
    for route in claims["routes"]:
        count = len(route["segment_claim_ids"])
        base, remainder = divmod(route["claimed_cost_q8"], count)
        for index, claim_id in enumerate(route["segment_claim_ids"]):
            costs[claim_id] = base + (1 if index < remainder else 0)
    return [
        {
            "claim_id": claim["claim_id"],
            "source_milliunits": claim["source_milliunits"],
            "target_milliunits": claim["target_milliunits"],
            "source_l1": [index, 0, 0],
            "target_l1": [index + 1, 0, 0],
            "connected": True,
            "cost_q8": costs[claim["claim_id"]],
            "status": "oracle",
            "evidence": 1,
            "validation_version": 1,
        }
        for index, claim in enumerate(claims["route_claims"])
    ]


def _analysis(map_path: Path, claims: dict) -> dict:
    meta = json.loads(map_path.with_suffix(".meta.json").read_text())
    hook_materialization = json.loads(
        Path(f"{map_path.with_suffix('')}.hook-materialization.json").read_text()
    )
    hook_traces = {
        trace["claim_id"]: trace
        for trace in hook_materialization["validation_traces"]
    }
    gate = json.loads((ROOT / "docs/multires/B1-GATE.json").read_text())
    physics = gate["artifacts"]["hook_parity_attestation"][
        "hook_physics_identity"
    ]
    spawn_count = len(claims["spawns"])
    return {
        "schema": "q2-atlas-analysis-v1",
        "status": "passed",
        "identity": {
            "bsp_sha256": file_sha256(map_path.with_suffix(".bsp")),
            "generator_claims_sha256": generator_claims_sha256(claims),
            "atlas_sha256": NONZERO,
            "analyzer_sha256": _expected_analyzer_sha256(),
        },
        "oracles": {
            "collision": {
                "status": "oracle",
                "admitted": True,
                "executable_sha256": gate["artifacts"][
                    "transformed_inline_collision"
                ]["cm_oracle_sha256"],
                "physics_identity": "44" * 32,
            },
            "hook": {
                "authority_admitted": True,
                "omission_reason": None,
                "binary_sha256": "66" * 32,
                "physics_identity": physics,
                "attestation_sha256": gate["artifacts"]["hook_parity_attestation"][
                    "sha256"
                ],
            },
        },
        "deterministic_rebuild": True,
        "confidence": "complete",
        "compiled_world": {
            "spawns": [
                {
                    "entity_ordinal": index,
                    "origin_milliunits": claim["origin_milliunits"],
                    "standing_clear": True,
                    "crouched_clear": True,
                    "supported": True,
                    "column_clearance_milliunits": 96000,
                    "column_clear_96": True,
                    "l1_index": [index, 0, 0],
                    "region_id": 1,
                    "escape_edge_count": 2,
                    "reachable_spawn_ordinals": [
                        other for other in range(spawn_count) if other != index
                    ],
                    "cost_to_safety_q8": 0,
                }
                for index, claim in enumerate(claims["spawns"])
            ],
            "hazards": {
                "l0_raw_cells": 100,
                "l0_expanded_cells": 200,
                "types": sorted({claim["type"] for claim in claims["hazard_claims"]}),
                "lethal_drop_edges": len(meta["safety"]["lethal_edges"]),
                "exact_lethal_candidates_omitted": 0,
                "guarded_drop_edges": len(meta["safety"]["lethal_edges"]),
                "uncontained_drop_edges": 0,
                "classification_status": "oracle",
                "evidence": 11,
                "validation_version": 1,
                "drop_classification": {
                    "classification_status": "oracle",
                    "evidence": 0,
                    "validation_version": 0,
                    "candidate_count": 0,
                    "exact_safe": 0,
                    "exact_lethal": 0,
                    "unknown_omitted": 0,
                    "unknown_reason_counts": {},
                    "severity_counts": {},
                },
            },
            "hazard_claims": [
                {
                    "claim_id": claim["claim_id"],
                    "type": claim["type"],
                    "raw_l0_cells": 1,
                    "expanded_l0_cells": 2,
                    "contained": True,
                    "status": "oracle",
                    "evidence": 1,
                    "validation_version": 1,
                }
                for claim in claims["hazard_claims"]
            ],
            "lighting": {
                "lightdata_bytes": 4096,
                "lightdata_sha256": "55" * 32,
                "lightmapped_faces": 64,
                "spawn_region_count": len(meta["lighting"]["regions"]),
                "dark_spawn_regions": 0,
            },
            "hooks": {
                "authority_admitted": True,
                "omission_reason": None,
                "edges": [
                    {
                        "claim_id": claim["claim_id"],
                        "source_l1": [index, 0, 0],
                        "target_l1": [index + 1, 0, 0],
                        "source_milliunits": claim["source_milliunits"],
                        "anchor_milliunits": claim["anchor_milliunits"],
                        "landing_milliunits": claim["landing_milliunits"],
                        "release_after_ticks": claim["release_after_ticks"],
                        "distance_milliunits": claim["distance_milliunits"],
                        "flags": claim["flags"],
                        "trajectory_origin_fixed": hook_traces[claim["claim_id"]][
                            "origin_fixed_frames"
                        ],
                        "trajectory_sha256": hook_traces[claim["claim_id"]]["sha256"],
                        "first_grounded_frame_index": hook_traces[claim["claim_id"]][
                            "first_grounded_frame_index"
                        ],
                        "landing_l1": [index + 1, 0, 0],
                        "physics_identity": physics,
                        "evidence": 1,
                        "validation_version": 1,
                    }
                    for index, claim in enumerate(claims["hook_claims"])
                ],
            },
            "route_claims": _route_results(claims),
        },
    }


def _write_authority_artifacts(
    map_path: Path, analysis: dict, analysis_path: Path,
) -> None:
    """Seal a compact producer fixture with the complete future cold contract."""

    map_id = map_path.stem
    directory = analysis_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    counts = {
        "l0_chunks": 1, "l1_nodes": 8, "l1_edges": 8,
        "l2_cells": 1, "l3_cells": 1,
    }
    header = bytearray(136)
    struct.pack_into("<8sHHI", header, 0, b"Q2ATL001", 1, 0x454C, 136)
    struct.pack_into("<3q", header, 16, 0, 0, 0)
    struct.pack_into("<4I", header, 40, 4, 16, 64, 256)
    struct.pack_into("<5Q", header, 56, *counts.values())
    sections = [b"L0v1", b"N", b"G", b"2", b"3"]
    struct.pack_into("<5Q", header, 96, *(len(value) for value in sections))
    atlas_bytes = bytes(header) + b"".join(sections)
    atlas_path = directory / f"{map_id}.atlas.bin"
    atlas_path.write_bytes(atlas_bytes)
    atlas_zst_path = directory / f"{map_id}.atlas.bin.zst"
    atlas_zst_path.write_bytes(b"fixture-zstd-v1\0" + atlas_bytes)
    navigation_path = directory / f"{map_id}.navigation.bin.zst"
    visibility_path = directory / f"{map_id}.visibility.bin.zst"
    navigation_path.write_bytes(b"fixture-navigation")
    visibility_path.write_bytes(b"fixture-visibility")
    design_path = directory / f"{map_id}.design-signature.json"
    design_path.write_bytes(canonical_bytes({
        "schema": "q2-atlas-design-signature-v1", "coordinate_free": True,
        "bsp_sha256": analysis["identity"]["bsp_sha256"],
        "counts": {"deathmatch_spawns": len(analysis["compiled_world"]["spawns"])},
        "item_class_multiset": {},
    }))
    routes_path = directory / f"{map_id}.routes.json"
    routes_path.write_bytes(canonical_bytes({"schema": "q2-atlas-routes-v1"}))

    analysis["canonical_map_id"] = map_id
    analysis["grid"] = {"origin": [0, 0, 0]}
    analysis["counts"] = {
        **counts, "l0_bit_cells": 1, "l0_scalar_cells": 0,
    }
    atlas_sha = file_sha256(atlas_path)
    verification = {
        "schema": "q2-atlas-verification-v1", "passed": True,
        "canonical_map_id": map_id,
        "bsp_sha256": analysis["identity"]["bsp_sha256"],
        "artifact_name": atlas_path.name, "atlas_sha256": atlas_sha,
        "origin": [0, 0, 0], "counts": counts,
        "collision_contract_sha256": "77" * 32,
    }
    analysis["identity"]["atlas_sha256"] = atlas_sha
    atlas_manifest_path = directory / f"{map_id}.atlas.manifest.json"
    atlas_manifest = {
        "schema_version": 1, "byte_order": "little", "atlas_magic": "Q2ATL001",
        "bsp": {"sha256": analysis["identity"]["bsp_sha256"]},
        "analyzer": {"sha256": analysis["identity"]["analyzer_sha256"]},
        "counts": counts,
        "budgets": {
            "max_l0_chunks": 1200,
            "max_l0_decompressed_bytes": 16 * 1024 * 1024,
            "max_atlas_decompressed_bytes": 32 * 1024 * 1024,
            "max_atlas_resident_bytes": 32 * 1024 * 1024,
            "max_build_rss_bytes": 512 * 1024 * 1024,
        },
        "build_peak_rss_bytes": 16 * 1024 * 1024,
    }
    atlas_manifest_path.write_bytes(canonical_bytes(atlas_manifest))
    manifest_sha = file_sha256(atlas_manifest_path)
    analysis["identity"]["atlas_manifest_sha256"] = manifest_sha
    local_verification = {**verification, "manifest_sha256": manifest_sha}
    verifier_sha = "88" * 32
    analysis["artifacts"] = {
        "atlas": {
            "schema": "q2-atlas-pack-result-v1",
            "uncompressed_sha256": atlas_sha,
            "uncompressed_bytes": atlas_path.stat().st_size,
            "compressed_sha256": file_sha256(atlas_zst_path),
            "compressed_bytes": atlas_zst_path.stat().st_size,
            **counts,
            "resident_bytes_estimate": 1024,
            "build_peak_rss_measurement": "linux_proc_self_status_vmhwm",
            "build_peak_rss_bytes": 16 * 1024 * 1024,
            "max_build_rss_bytes": 512 * 1024 * 1024,
            "build_peak_rss_gate_passed": True,
            "transport_sha256": file_sha256(atlas_zst_path),
        },
        "navigation": {"transport_sha256": file_sha256(navigation_path)},
        "visibility": {"transport_sha256": file_sha256(visibility_path)},
        "design_signature": {"sha256": file_sha256(design_path)},
        "routes": {"sha256": file_sha256(routes_path)},
        "atlas_manifest": {
            "path": atlas_manifest_path.name, "sha256": manifest_sha,
            "uncompressed_bytes": atlas_manifest_path.stat().st_size,
            "verifier_sha256": verifier_sha, "verification": local_verification,
        },
    }
    exact_paths = {
        ".atlas.bin": atlas_path, ".atlas.bin.zst": atlas_zst_path,
        ".navigation.bin.zst": navigation_path,
        ".visibility.bin.zst": visibility_path,
        ".design-signature.json": design_path, ".routes.json": routes_path,
    }
    exact = {suffix: file_sha256(path) for suffix, path in exact_paths.items()}
    analysis["performance"] = {
        "cm_requests": 1, "pmove_requests": 1,
        "primary_elapsed_milliseconds": 500,
    }
    semantic_manifest = dict(atlas_manifest)
    semantic_manifest.pop("build_peak_rss_bytes")
    semantic = {
        ".atlas.manifest.json": sha256_bytes(json.dumps(
            semantic_manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        ).encode("ascii")),
        ".analysis.manifest.json": _canonical_semantic_analysis_manifest(
            analysis
        ),
    }
    analysis["performance"]["full_cold_rebuild"] = {
            "schema": "q2-atlas-full-cold-proof-v1",
            "independent_process_launches": 1, "artifact_count": 8,
            "artifact_sha256": exact, "artifact_semantic_sha256": semantic,
            "cold_artifact_sha256": dict(exact),
            "cold_artifact_semantic_sha256": dict(semantic),
            "verifier_sha256": verifier_sha, "verification": verification,
            "sample_interval_milliseconds": 10,
            "sampled_process_tree_peak_rss_bytes": 32 * 1024 * 1024,
            "peak_rss_limit_bytes": 512 * 1024 * 1024,
            "elapsed_milliseconds": 1000,
            "timeout_limit_milliseconds": 300_000,
    }


def _write_stock_pins(
    directory: Path, bsp_sha256: str, *, spawn_count: int = 8,
    item_classes: dict | None = None,
) -> tuple[Path, Path]:
    item_classes = item_classes or {}
    provenance_path = directory / "stock.provenance.json"
    inventory_path = directory / "stock.inventory.json"
    provenance_path.write_bytes(canonical_bytes({
        "schema": "q2-corpus-provenance-v1",
        "records": [
            {
                "canonical_id": f"q2dm{number}",
                "bsp_sha256": bsp_sha256 if number == 1 else f"{number}" * 64,
            }
            for number in range(1, 9)
        ],
    }))
    inventory_path.write_bytes(canonical_bytes({
        "schema": "q2-stock-map-fixtures-v1",
        "maps": [
            {
                "canonical_id": f"q2dm{number}",
                "bsp_sha256": bsp_sha256 if number == 1 else f"{number}" * 64,
                "deathmatch_spawn_count": spawn_count if number == 1 else 2,
                "item_classes": item_classes if number == 1 else {},
            }
            for number in range(1, 9)
        ],
    }))
    return provenance_path, inventory_path


def _stock_fixture(generated, tmp_path: Path):
    generated_map, claims, _generated_analysis = generated
    stock_bsp = tmp_path / "q2dm1.bsp"
    stock_bsp.write_bytes(generated_map.with_suffix(".bsp").read_bytes())
    analysis_path = tmp_path / "stock-analysis/q2dm1.analysis.manifest.json"
    analysis = _analysis(generated_map, claims)
    analysis["identity"]["bsp_sha256"] = file_sha256(stock_bsp)
    analysis["identity"]["generator_claims_sha256"] = None
    analysis["compiled_world"]["hazards"].update({
        "classification_status": "oracle", "evidence": 1,
        "validation_version": 1,
    })
    _write_authority_artifacts(stock_bsp, analysis, analysis_path)
    analysis_path.write_bytes(canonical_bytes(analysis))
    provenance, inventory = _write_stock_pins(
        tmp_path, file_sha256(stock_bsp)
    )
    return stock_bsp, analysis_path, provenance, inventory


@pytest.fixture()
def generated(tmp_path: Path) -> tuple[Path, dict, Path]:
    map_path, _ = generate_map(
        "claim_fixture", 42, tmp_path, style="arena_vertical"
    )
    _write_compiled_bsp(map_path)
    _fake_materialize(map_path)
    claims = build_generator_claims(map_path)
    analysis_path = tmp_path / "analysis" / "claim_fixture.analysis.manifest.json"
    analysis = _analysis(map_path, claims)
    _write_authority_artifacts(map_path, analysis, analysis_path)
    analysis_path.write_bytes(canonical_bytes(analysis))
    return map_path, claims, analysis_path


def _mutate_analysis(
    analysis_path: Path, mutation
) -> Path:
    value = json.loads(analysis_path.read_text())
    mutation(value)
    analysis_path.write_bytes(canonical_bytes(value))
    return analysis_path


def _set_unknown_drop_summary(value: dict, reasons: dict[str, int]) -> None:
    unknown = sum(reasons.values())
    value["compiled_world"]["hazards"]["drop_classification"].update({
        "classification_status": "oracle",
        "evidence": 0,
        "validation_version": 0,
        "candidate_count": unknown,
        "exact_safe": 0,
        "exact_lethal": 0,
        "unknown_omitted": unknown,
        "unknown_reason_counts": reasons,
        "severity_counts": {},
    })
    value["compiled_world"]["hazards"][
        "exact_lethal_candidates_omitted"
    ] = 0


def _reseal_analysis(
    map_path: Path, analysis_path: Path, mutation,
) -> None:
    value = json.loads(analysis_path.read_text())
    mutation(value)
    _write_authority_artifacts(map_path, value, analysis_path)
    analysis_path.write_bytes(canonical_bytes(value))


def _inflate_first_route(value: dict) -> None:
    first = value["compiled_world"]["route_claims"][0]
    first["cost_q8"] += 10000 * 256


def test_generator_claims_are_canonical_and_byte_deterministic(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    map_a, _ = generate_map("same", 7142026, first, style="arena_lanes")
    map_b, _ = generate_map("same", 7142026, second, style="arena_lanes")
    _write_compiled_bsp(map_a)
    _write_compiled_bsp(map_b)
    _fake_materialize(map_a)
    _fake_materialize(map_b)

    claims_a = build_generator_claims(map_a)
    claims_b = build_generator_claims(map_b)
    assert canonical_bytes(claims_a) == canonical_bytes(claims_b)
    assert generator_claims_sha256(claims_a) == generator_claims_sha256(claims_b)
    assert len(claims_a["spawns"]) == 8
    assert any(claim["type"] == "hurt" for claim in claims_a["hazard_claims"])
    assert claims_a["route_claims"]


def test_complete_compiled_world_proof_passes(generated):
    map_path, _claims, analysis_path = generated
    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is True
    assert all(criterion["passed"] for criterion in report["criteria"].values())
    assert report["mode"] == "generated_v6_promotion"


def test_generated_promotion_allows_accounted_fail_closed_drop_reasons(generated):
    map_path, _claims, analysis_path = generated
    for reason in ("no_landing", "unsupported_dynamic_mover"):
        _reseal_analysis(
            map_path,
            analysis_path,
            lambda value, reason=reason: _set_unknown_drop_summary(
                value, {reason: 3},
            ),
        )

        report = validate_generated_map(map_path, analysis_path)

        assert report["passed"] is True
        assert report["criteria"]["hazards"]["passed"] is True


def test_generated_promotion_rejects_drop_authority_defect_reasons(generated):
    map_path, _claims, analysis_path = generated
    defective = (
        "invalid_pmove_evidence",
        "invalid_fall_evidence",
        "invalid_contract",
        "authority_digest_mismatch",
        "oracle_failure",
        "pmove_oracle_failure",
        "fall_oracle_failure",
    )
    for reason in defective:
        _reseal_analysis(
            map_path,
            analysis_path,
            lambda value, reason=reason: _set_unknown_drop_summary(
                value, {reason: 4},
            ),
        )

        report = validate_generated_map(map_path, analysis_path)

        assert report["passed"] is False
        failures = "\n".join(report["criteria"]["hazards"]["failures"])
        assert f"Unknown reason {reason} is not promotion-admissible" in failures


def test_generated_promotion_rejects_incomplete_or_malformed_reason_counts(
    generated,
):
    map_path, _claims, analysis_path = generated
    _reseal_analysis(
        map_path,
        analysis_path,
        lambda value: _set_unknown_drop_summary(value, {"no_landing": 2}),
    )
    _reseal_analysis(
        map_path,
        analysis_path,
        lambda value: value["compiled_world"]["hazards"][
            "drop_classification"
        ].update(unknown_omitted=3, candidate_count=3),
    )
    report = validate_generated_map(map_path, analysis_path)
    assert report["passed"] is False
    assert "reason counts do not cover every omitted candidate" in "\n".join(
        report["criteria"]["hazards"]["failures"]
    )

    _reseal_analysis(
        map_path,
        analysis_path,
        lambda value: _set_unknown_drop_summary(value, {"Invalid-Pmove": 1}),
    )
    report = validate_generated_map(map_path, analysis_path)
    assert report["passed"] is False
    assert "Unknown reason 'Invalid-Pmove' is malformed" in "\n".join(
        report["criteria"]["hazards"]["failures"]
    )


def test_duplicate_unknown_reason_json_key_is_rejected(generated):
    map_path, _claims, analysis_path = generated
    _reseal_analysis(
        map_path,
        analysis_path,
        lambda value: _set_unknown_drop_summary(value, {"no_landing": 1}),
    )
    payload = analysis_path.read_text()
    original = '"unknown_reason_counts":{"no_landing":1}'
    assert original in payload
    analysis_path.write_text(payload.replace(
        original,
        '"unknown_reason_counts":{"no_landing":1,"no_landing":1}',
        1,
    ))

    with pytest.raises(ClaimValidationError, match="duplicate JSON key"):
        validate_generated_map(map_path, analysis_path)


def test_noncanonical_unknown_reason_key_order_is_rejected(generated):
    map_path, _claims, analysis_path = generated
    _reseal_analysis(
        map_path,
        analysis_path,
        lambda value: _set_unknown_drop_summary(value, {
            "no_landing": 1, "unsupported_dynamic_mover": 1,
        }),
    )
    payload = analysis_path.read_text()
    canonical = (
        '"unknown_reason_counts":{"no_landing":1,'
        '"unsupported_dynamic_mover":1}'
    )
    noncanonical = (
        '"unknown_reason_counts":{"unsupported_dynamic_mover":1,'
        '"no_landing":1}'
    )
    assert canonical in payload
    analysis_path.write_text(payload.replace(canonical, noncanonical, 1))

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert "reason counts are not canonically sorted" in "\n".join(
        report["criteria"]["hazards"]["failures"]
    )


def test_self_declared_pass_without_full_cold_artifacts_rejects(generated):
    map_path, claims, analysis_path = generated
    analysis_path.write_bytes(canonical_bytes(_analysis(map_path, claims)))

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    authority = report["criteria"]["artifact_authority"]
    assert authority["passed"] is False
    assert any("analysis artifacts must be an object" in item for item in authority["failures"])


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda value: value["identity"].update(analyzer_sha256="aa" * 32),
            "analyzer identity differs from the local admitted source closure",
        ),
        (
            lambda value: value["oracles"]["collision"].update(
                executable_sha256="aa" * 32
            ),
            "collision oracle executable differs from the accepted B1 authority",
        ),
    ],
)
def test_self_declared_analyzer_or_collision_authority_rejects(
    generated, mutation, expected,
):
    map_path, _claims, analysis_path = generated
    _mutate_analysis(analysis_path, mutation)

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert expected in "\n".join(
        report["criteria"]["analysis_quality"]["failures"]
    )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda value: value["performance"]["full_cold_rebuild"][
                "cold_artifact_sha256"
            ].update({".atlas.bin": "aa" * 32}),
            "independent-cold artifact hashes differ",
        ),
        (
            lambda value: (
                value["performance"]["full_cold_rebuild"][
                    "artifact_semantic_sha256"
                ].update({".analysis.manifest.json": "aa" * 32}),
                value["performance"]["full_cold_rebuild"][
                    "cold_artifact_semantic_sha256"
                ].update({".analysis.manifest.json": "aa" * 32}),
            ),
            "on-disk analysis manifest semantic digest differs",
        ),
        (
            lambda value: value["counts"].update(l0_chunks=1201),
            "exceeds 1200 L0 chunks",
        ),
        (
            lambda value: value["artifacts"]["atlas"].update(
                uncompressed_bytes=32 * 1024 * 1024 + 1
            ),
            "decompressed bytes exceed 32 MiB",
        ),
        (
            lambda value: value["artifacts"]["atlas"].update(
                resident_bytes_estimate=32 * 1024 * 1024 + 1
            ),
            "resident estimate exceeds 32 MiB",
        ),
        (
            lambda value: value["performance"]["full_cold_rebuild"].update(
                sampled_process_tree_peak_rss_bytes=512 * 1024 * 1024 + 1
            ),
            "process-tree RSS exceeds 512 MiB",
        ),
        (
            lambda value: value["performance"]["full_cold_rebuild"].update(
                elapsed_milliseconds=300_001
            ),
            "elapsed time exceeds 300 seconds",
        ),
    ],
)
def test_full_cold_digest_and_hard_budget_tamper_rejects(
    generated, mutation, expected
):
    map_path, _claims, analysis_path = generated
    _mutate_analysis(analysis_path, mutation)

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert expected in "\n".join(report["criteria"]["artifact_authority"]["failures"])


def test_analysis_semantic_field_tamper_rejects_even_when_proof_maps_agree(
    generated,
):
    map_path, _claims, analysis_path = generated
    _mutate_analysis(
        analysis_path,
        lambda value: value["compiled_world"].update(
            pmove_source_accounting={
                "schema": "q2-atlas-pmove-source-accounting-v1",
                "selected": 1,
                "omitted": 1,
            },
        ),
    )

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert "on-disk analysis manifest semantic digest differs" in "\n".join(
        report["criteria"]["artifact_authority"]["failures"]
    )


def test_missing_seventh_artifact_rejects(generated):
    map_path, _claims, analysis_path = generated
    (analysis_path.parent / f"{map_path.stem}.atlas.manifest.json").unlink()

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert "full-cold artifact is missing" in "\n".join(
        report["criteria"]["artifact_authority"]["failures"]
    )


def test_on_disk_artifact_digest_tamper_rejects(generated):
    map_path, _claims, analysis_path = generated
    navigation_path = analysis_path.parent / f"{map_path.stem}.navigation.bin.zst"
    navigation_path.write_bytes(navigation_path.read_bytes() + b"tamper")

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert "on-disk artifact digest differs" in "\n".join(
        report["criteria"]["artifact_authority"]["failures"]
    )


def test_atlas_l0_section_over_16_mib_rejects(generated):
    map_path, _claims, analysis_path = generated
    atlas_path = analysis_path.parent / f"{map_path.stem}.atlas.bin"
    original = atlas_path.read_bytes()
    header = bytearray(original[:136])
    oversized_l0 = 16 * 1024 * 1024 + 1
    struct.pack_into("<Q", header, 96, oversized_l0)
    atlas_path.write_bytes(bytes(header) + b"L" * oversized_l0 + original[140:])

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert "Atlas L0 section exceeds 16 MiB" in report["criteria"][
        "artifact_authority"
    ]["failures"]


def test_analysis_hook_parity_digest_must_equal_exact_b1_attestation(generated):
    map_path, _claims, analysis_path = generated
    _mutate_analysis(
        analysis_path,
        lambda value: value["oracles"]["hook"].update(
            attestation_sha256="aa" * 32
        ),
    )

    report = validate_generated_map(map_path, analysis_path)

    assert report["passed"] is False
    assert "analysis hook parity digest differs" in "\n".join(
        report["criteria"]["hooks"]["failures"]
    )


def test_claim_materialization_cannot_carry_non_b1_hook_parity(generated):
    map_path, _old_claims, _analysis_path = generated
    stem = map_path.with_suffix("")
    materialization_path = Path(f"{stem}.hook-materialization.json")
    materialization = json.loads(materialization_path.read_text())
    materialization["oracles"]["hook_parity_attestation_sha256"] = "aa" * 32
    with pytest.raises(HookClaimsV2Error, match="B1 seal binding differs"):
        validate_materialization(materialization)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda value: value.update(status="unknown"), "analysis status"),
        (
            lambda value: value["oracles"]["collision"].update(status="unknown"),
            "collision oracle status",
        ),
        (
            lambda value: value["compiled_world"]["spawns"][0].update(
                column_clear_96=False
            ),
            "96-unit column",
        ),
        (
            lambda value: value["compiled_world"]["spawns"][0].update(
                escape_edge_count=0
            ),
            "escape edge",
        ),
        (
            lambda value: value["compiled_world"]["hazard_claims"][0].update(
                contained=False
            ),
            "oracle containment",
        ),
        (
            lambda value: value["compiled_world"]["hazards"].update(
                uncontained_drop_edges=1
            ),
            "not completely guarded",
        ),
        (
            lambda value: value["compiled_world"]["lighting"].update(
                lightdata_bytes=0
            ),
            "lightdata",
        ),
        (
            lambda value: value["compiled_world"]["hooks"].update(
                authority_admitted=False, omission_reason="oracle_unknown"
            ),
            "hook authority",
        ),
        (
            lambda value: value["compiled_world"]["route_claims"][0].update(
                connected=False, status="unknown"
            ),
            "oracle connectivity",
        ),
        (_inflate_first_route, "compiled cost differs excessively"),
    ],
)
def test_unknown_failed_or_tampered_compiled_facts_reject(
    generated, mutation, expected
):
    map_path, _claims, analysis_path = generated
    _mutate_analysis(analysis_path, mutation)

    report = validate_generated_map(map_path, analysis_path)
    assert report["passed"] is False
    assert expected in "\n".join(report["failures"])


def test_claim_digest_substitution_and_missing_result_reject(generated):
    map_path, _claims, analysis_path = generated
    _mutate_analysis(
        analysis_path,
        lambda value: value["identity"].update(
            generator_claims_sha256="aa" * 32
        ),
    )
    report = validate_generated_map(map_path, analysis_path)
    assert report["passed"] is False
    assert any("generator-claims identity differs" in item for item in report["failures"])

    restored = _analysis(map_path, build_generator_claims(map_path))
    _write_authority_artifacts(map_path, restored, analysis_path)
    analysis_path.write_bytes(canonical_bytes(restored))
    _mutate_analysis(
        analysis_path,
        lambda value: value["compiled_world"]["hooks"]["edges"].pop(),
    )
    report = validate_generated_map(map_path, analysis_path)
    assert report["passed"] is False
    assert any("exactly cover claims" in item for item in report["failures"])


def test_source_sidecar_tamper_cannot_be_overridden_by_old_analysis(generated):
    map_path, _claims, analysis_path = generated
    lattice_path = Path(f"{map_path.with_suffix('')}.lattice.json")
    lattice = json.loads(lattice_path.read_text())
    lattice["spawns"][0]["x"] += 16
    lattice_path.write_text(json.dumps(lattice))

    with pytest.raises(ClaimValidationError, match="spawn origins differ"):
        validate_generated_map(map_path, analysis_path)


def test_claim_and_compiled_records_reject_unknown_fields(generated):
    map_path, claims, analysis_path = generated
    changed = deepcopy(claims)
    changed["unexpected"] = True
    with pytest.raises(ClaimValidationError, match="extra"):
        validate_generator_claims(changed)

    changed = deepcopy(claims)
    extra = deepcopy(changed["spawns"][-1])
    extra["claim_id"] = "spawn:9999"
    extra["origin_milliunits"][0] += 1_000_000
    changed["spawns"].append(extra)
    with pytest.raises(ClaimValidationError, match="exactly eight spawns"):
        validate_generator_claims(changed)

    _mutate_analysis(
        analysis_path,
        lambda value: value["compiled_world"]["spawns"][0].update(
            source_claim_override=True
        ),
    )
    with pytest.raises(ClaimValidationError, match="compiled spawn keys differ"):
        validate_generated_map(map_path, analysis_path)


def test_unpinned_map_cannot_enter_stock_analysis_lane(generated):
    map_path, claims, analysis_path = generated
    value = _analysis(map_path, claims)
    value["identity"]["generator_claims_sha256"] = None
    # These would fail generated-v6 promotion but are irrelevant to stock
    # analysis quality: no tagged light-region or hook-claim requirement.
    value["compiled_world"]["lighting"]["dark_spawn_regions"] = 99
    value["compiled_world"]["hooks"] = {
        "authority_admitted": False,
        "omission_reason": "not_requested",
        "edges": [],
    }
    value["compiled_world"]["hazard_claims"] = []
    value["compiled_world"]["route_claims"] = []
    _write_authority_artifacts(map_path, value, analysis_path)
    analysis_path.write_bytes(canonical_bytes(value))

    report = validate_stock_analysis(map_path.with_suffix(".bsp"), analysis_path)
    assert report["passed"] is False
    assert any("not a pinned stock-map identity" in item for item in report["failures"])
    assert set(report["criteria"]) == {
        "analysis_quality", "artifact_authority", "stock_provenance",
        "stock_reachable_spawns", "stock_structural_inventory",
        "stock_hazard_classification",
    }


def test_stock_analysis_requires_pinned_provenance_and_inventory(
    generated, tmp_path: Path,
):
    bsp, analysis, provenance, inventory = _stock_fixture(generated, tmp_path)

    report = validate_stock_analysis(
        bsp, analysis, stock_provenance_path=provenance,
        stock_inventory_path=inventory,
    )

    assert report["passed"] is True
    assert report["identities"]["stock_provenance_sha256"] == file_sha256(
        provenance
    )
    assert report["identities"]["stock_inventory_sha256"] == file_sha256(
        inventory
    )


def test_stock_hazard_cannot_pass_without_oracle_evidence(
    generated, tmp_path: Path,
):
    bsp, analysis, provenance, inventory = _stock_fixture(generated, tmp_path)
    _mutate_analysis(
        analysis,
        lambda value: [
            value["compiled_world"]["hazards"].pop(name)
            for name in ("classification_status", "evidence", "validation_version")
        ],
    )

    report = validate_stock_analysis(
        bsp, analysis, stock_provenance_path=provenance,
        stock_inventory_path=inventory,
    )

    assert report["passed"] is False
    assert "analyzer follow-up must emit" in "\n".join(
        report["criteria"]["stock_hazard_classification"]["failures"]
    )


def test_stock_promotion_rejects_all_unknown_invalid_pmove_evidence(
    generated, tmp_path: Path,
):
    bsp, analysis, provenance, inventory = _stock_fixture(generated, tmp_path)
    _reseal_analysis(
        bsp,
        analysis,
        lambda value: _set_unknown_drop_summary(
            value, {"invalid_pmove_evidence": 12},
        ),
    )

    report = validate_stock_analysis(
        bsp, analysis, stock_provenance_path=provenance,
        stock_inventory_path=inventory,
    )

    assert report["passed"] is False
    assert (
        "Unknown reason invalid_pmove_evidence is not promotion-admissible"
        in "\n".join(
            report["criteria"]["stock_hazard_classification"]["failures"]
        )
    )


@pytest.mark.parametrize(
    "spawn_count,item_classes,expected",
    [
        (9, {}, "deathmatch-spawn count differs"),
        (8, {"weapon_railgun": 1}, "item-class multiset differs"),
    ],
)
def test_stock_structural_facts_are_pinned_not_self_declared(
    generated, tmp_path: Path, spawn_count, item_classes, expected,
):
    bsp, analysis, provenance, _inventory = _stock_fixture(generated, tmp_path)
    _provenance, inventory = _write_stock_pins(
        tmp_path, file_sha256(bsp), spawn_count=spawn_count,
        item_classes=item_classes,
    )

    report = validate_stock_analysis(
        bsp, analysis, stock_provenance_path=provenance,
        stock_inventory_path=inventory,
    )

    assert report["passed"] is False
    assert expected in "\n".join(
        report["criteria"]["stock_structural_inventory"]["failures"]
    )


def test_contract_schemas_forbid_unknown_top_level_fields():
    claims_schema = json.loads(
        (ROOT / "schemas/q2-generator-claims-v2.schema.json").read_text()
    )
    report_schema = json.loads(
        (ROOT / "schemas/q2-generator-claim-validation-v1.schema.json").read_text()
    )
    assert claims_schema["additionalProperties"] is False
    assert report_schema["additionalProperties"] is False


def test_prepare_campaign_is_canonical_and_requires_exact_count(tmp_path: Path):
    map_path, _ = generate_map("campaign", 7, tmp_path, style="open")
    unmaterialized = prepare_claims([map_path], expected_count=1)
    assert unmaterialized["passed"] is False
    _write_compiled_bsp(map_path)
    _fake_materialize(map_path)
    first = prepare_claims([map_path], expected_count=1)
    first_bytes = canonical_bytes(first)
    second = prepare_claims([map_path], expected_count=1)
    assert canonical_bytes(second) == first_bytes
    assert first["passed"] is True
    assert first["pass_count"] == 1

    wrong_count = prepare_claims([map_path], expected_count=2)
    assert wrong_count["passed"] is False
    assert wrong_count["failures"] == ["map count 1 != required 2"]


@pytest.mark.parametrize("mutation", ["bsp", "meta", "candidate", "materialization"])
def test_materialization_source_mutation_rejects_before_claim_publish(
    tmp_path: Path, mutation: str,
) -> None:
    map_path, _ = generate_map("bound", 17, tmp_path, style="arena_vertical")
    bsp_path = _write_compiled_bsp(map_path)
    _fake_materialize(map_path)
    stem = map_path.with_suffix("")
    if mutation == "bsp":
        bsp_path.write_bytes(bsp_path.read_bytes() + b"tamper")
    elif mutation in {"meta", "candidate"}:
        meta_path = stem.with_suffix(".meta.json")
        metadata = json.loads(meta_path.read_text())
        if mutation == "meta":
            metadata["tampered_after_materialization"] = True
        else:
            metadata["hook_claim_candidates_v2"]["records"][0][
                "release_after_ticks"
            ] += 1
        meta_path.write_bytes(canonical_bytes(metadata))
    else:
        materialization_path = Path(f"{stem}.hook-materialization.json")
        materialization = json.loads(materialization_path.read_text())
        materialization["request_count"] += 1
        materialization_path.write_bytes(canonical_bytes(materialization))
    with pytest.raises(ClaimValidationError):
        build_generator_claims(map_path)
