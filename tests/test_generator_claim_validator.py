from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from maps.generator import generate_map
from tools.generator_claim_validator import (
    ClaimValidationError,
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
            "analyzer_sha256": "22" * 32,
        },
        "oracles": {
            "collision": {
                "status": "oracle",
                "admitted": True,
                "executable_sha256": "33" * 32,
                "physics_identity": "44" * 32,
            }
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
                "guarded_drop_edges": len(meta["safety"]["lethal_edges"]),
                "uncontained_drop_edges": 0,
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
                        "anchor_milliunits": claim["anchor_milliunits"],
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


@pytest.fixture()
def generated(tmp_path: Path) -> tuple[Path, dict, Path]:
    map_path, _ = generate_map(
        "claim_fixture", 42, tmp_path, style="arena_vertical"
    )
    _write_compiled_bsp(map_path)
    claims = build_generator_claims(map_path)
    analysis_path = tmp_path / "claim_fixture.analysis.manifest.json"
    analysis_path.write_bytes(canonical_bytes(_analysis(map_path, claims)))
    return map_path, claims, analysis_path


def _mutate_analysis(
    analysis_path: Path, mutation
) -> Path:
    value = json.loads(analysis_path.read_text())
    mutation(value)
    analysis_path.write_bytes(canonical_bytes(value))
    return analysis_path


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

    analysis_path.write_bytes(canonical_bytes(_analysis(map_path, build_generator_claims(map_path))))
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

    _mutate_analysis(
        analysis_path,
        lambda value: value["compiled_world"]["spawns"][0].update(
            source_claim_override=True
        ),
    )
    with pytest.raises(ClaimValidationError, match="compiled spawn keys differ"):
        validate_generated_map(map_path, analysis_path)


def test_stock_analysis_does_not_apply_generator_v6_criteria(generated):
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
    analysis_path.write_bytes(canonical_bytes(value))

    report = validate_stock_analysis(map_path.with_suffix(".bsp"), analysis_path)
    assert report["passed"] is True
    assert set(report["criteria"]) == {
        "analysis_quality",
        "stock_reachable_spawns",
        "stock_hazard_classification",
    }


def test_contract_schemas_forbid_unknown_top_level_fields():
    claims_schema = json.loads(
        (ROOT / "schemas/q2-generator-claims-v1.schema.json").read_text()
    )
    report_schema = json.loads(
        (ROOT / "schemas/q2-generator-claim-validation-v1.schema.json").read_text()
    )
    assert claims_schema["additionalProperties"] is False
    assert report_schema["additionalProperties"] is False


def test_prepare_campaign_is_canonical_and_requires_exact_count(tmp_path: Path):
    map_path, _ = generate_map("campaign", 7, tmp_path, style="open")
    first = prepare_claims([map_path], expected_count=1)
    first_bytes = canonical_bytes(first)
    second = prepare_claims([map_path], expected_count=1)
    assert canonical_bytes(second) == first_bytes
    assert first["passed"] is True
    assert first["pass_count"] == 1

    wrong_count = prepare_claims([map_path], expected_count=2)
    assert wrong_count["passed"] is False
    assert wrong_count["failures"] == ["map count 1 != required 2"]
