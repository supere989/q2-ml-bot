from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.run_b3_design_prior_campaign import (
    B3PriorError,
    BASELINE_STYLE_WEIGHTS_PPM,
    BIAS_SCHEMA,
    CAMPAIGN_SCHEMA,
    LANE_SCHEMA,
    SEEDS_SCHEMA,
    STOCK_MAPS,
    canonical_bytes,
    evaluate_campaign,
    prepare_campaign,
    sha256_bytes,
    validate_bias,
    validate_campaign_report,
    validate_design_signature,
)


ROOT = Path(__file__).resolve().parents[1]


def _implementation() -> dict:
    return {
        "repository_commit": "1" * 40,
        "repository_tree": "2" * 40,
        "git_clean": True,
        "generator_sha256": "3" * 64,
        "analyzer_authority_sha256": "4" * 64,
        "analyzer_authority_file_count": 17,
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _signature(kind: str, ordinal: int, bsp_sha256: str) -> dict:
    if kind == "stock":
        nodes, edges, degree = 100 + ordinal, 100 + ordinal, {"2": 100 + ordinal}
        edge_types = {"walk": edges}
        items = {"weapon_rocketlauncher": 10 + ordinal}
        faces, lightmapped = 100 + ordinal, 75 + ordinal
    elif kind == "baseline":
        nodes, edges, degree = 100 + ordinal, 800 + ordinal, {"8": 100 + ordinal}
        edge_types = {"hook": edges}
        items = {"item_quad": 80 + ordinal}
        faces, lightmapped = 100 + ordinal, 10
    else:
        nodes, edges, degree = 100 + ordinal, 100 + ordinal, {"2": 100 + ordinal}
        edge_types = {"walk": edges}
        items = {"weapon_rocketlauncher": 10 + ordinal}
        faces, lightmapped = 100 + ordinal, 75 + ordinal
    return {
        "schema": "q2-atlas-design-signature-v1",
        "coordinate_free": True,
        "bsp_sha256": bsp_sha256,
        "counts": {
            "l1_nodes": nodes,
            "l1_edges": edges,
            "deathmatch_spawns": 8,
            "items": sum(items.values()),
            "faces": faces,
            "visibility_clusters": 4 + ordinal,
        },
        "degree_histogram": degree,
        "edge_type_histogram": edge_types,
        "item_class_multiset": items,
        "light": {"lightdata_bytes": 4096 + ordinal, "lightmapped_faces": lightmapped},
    }


def _prepare(tmp_path: Path) -> tuple[dict, Path, Path]:
    stock = tmp_path / "stock"
    for ordinal, name in enumerate(STOCK_MAPS):
        _write(stock / f"{name}.design-signature.json", _signature("stock", ordinal, _sha(f"stock-bsp-{ordinal}")))
    bias = {
        "schema": BIAS_SCHEMA,
        "style_weights_ppm": BASELINE_STYLE_WEIGHTS_PPM,
        "knob_delta": {"corridor_prob": 210_000},
    }
    seeds = {"schema": SEEDS_SCHEMA, "seeds": list(range(7000, 7028))}
    bias_path = tmp_path / "bias.json"
    seeds_path = tmp_path / "seeds.json"
    plan_path = tmp_path / "plan.json"
    _write(bias_path, bias)
    _write(seeds_path, seeds)
    plan = prepare_campaign(
        "b3proof", stock, bias_path, seeds_path, plan_path,
        repo_root=ROOT, _implementation=_implementation(),
    )
    return plan, plan_path, stock


def _lane(
    tmp_path: Path,
    plan: dict,
    plan_path: Path,
    lane: str,
    signature_kind: str,
) -> tuple[Path, Path]:
    analysis = tmp_path / f"{lane}-analysis"
    rows = []
    for pair in plan["pairs"]:
        expected = pair[lane]
        bsp = _sha(f"{lane}-bsp-{pair['ordinal']}")
        signature = _signature(signature_kind, pair["ordinal"], bsp)
        path = analysis / f"{expected['map']}.design-signature.json"
        _write(path, signature)
        rows.append({
            "ordinal": pair["ordinal"],
            "map": expected["map"],
            "seed": pair["seed"],
            "style": expected["style"],
            "generator_knobs": expected["generator_knobs"],
            "source_static_passed": True,
            "layout_sha256": _sha(f"{lane}-layout-{pair['ordinal']}"),
            "bsp_sha256": bsp,
            "design_signature": {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        })
    document = {
        "schema": LANE_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "lane": lane,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "evidence_kind": "measured-compiled-atlas",
        "synthetic_claims": False,
        "implementation": _implementation(),
        "authorities": {
            name: {"bytes": index + 1, "sha256": _sha(f"{lane}-authority-{name}")}
            for index, name in enumerate((
                "q2tool", "packer", "verifier", "cm_oracle", "pmove_oracle",
                "hook_oracle", "fall_oracle", "hook_attestation", "b1_gate",
            ))
        },
        "pipeline": {
            name: {"bytes": index + 1, "sha256": _sha(f"{lane}-{name}")}
            for index, name in enumerate((
                "declaration", "source_freeze", "compile", "compiled_cm_preflight",
                "materialization", "claims_prepare", "atlas_build", "claims_validation",
            ))
        },
        "maps": rows,
        "failures": [],
        "passed": True,
    }
    lane_path = tmp_path / f"{lane}.json"
    _write(lane_path, document)
    return lane_path, analysis


def build_green_campaign(tmp_path: Path) -> tuple[dict, Path]:
    plan, plan_path, stock = _prepare(tmp_path)
    baseline_path, baseline_analysis = _lane(tmp_path, plan, plan_path, "baseline", "baseline")
    treatment_path, treatment_analysis = _lane(tmp_path, plan, plan_path, "treatment", "treatment")
    output = tmp_path / "campaign.json"
    report = evaluate_campaign(
        plan_path, stock, baseline_path, baseline_analysis,
        treatment_path, treatment_analysis, output,
        repo_root=ROOT, _implementation=_implementation(),
    )
    return report, output


def test_prepare_and_evaluate_are_deterministic_and_green(tmp_path: Path):
    report, output = build_green_campaign(tmp_path)
    assert report["schema"] == CAMPAIGN_SCHEMA
    assert report["passed"] is True
    assert report["decision"] == {
        "all_metrics_improved": True,
        "aggregate_improved": True,
        "static_pass_rate_preserved": True,
        "layout_diversity_passed": True,
        "style_diversity_passed": True,
        "descriptor_diversity_passed": True,
        "green": True,
    }
    assert output.read_bytes() == canonical_bytes(report)
    assert report["lanes"]["baseline"]["map_count"] == 28
    assert report["lanes"]["treatment"]["static_pass_count"] == 28
    validate_campaign_report(report)


def test_same_style_ticket_is_used_when_style_weights_match(tmp_path: Path):
    plan, _plan_path, _stock = _prepare(tmp_path)
    assert all(row["baseline"]["style"] == row["treatment"]["style"] for row in plan["pairs"])
    assert all(
        row["treatment"]["generator_knobs"]["corridor_prob"]
        == row["baseline"]["generator_knobs"]["corridor_prob"] + 10_000
        for row in plan["pairs"]
    )


def test_design_signature_rejects_coordinate_or_graph_fields():
    signature = _signature("stock", 0, _sha("stock"))
    signature["origin"] = [0, 0, 0]
    with pytest.raises(B3PriorError, match="keys differ"):
        validate_design_signature(signature, "forged")
    signature.pop("origin")
    signature["graph"] = {"edges": []}
    with pytest.raises(B3PriorError, match="keys differ"):
        validate_design_signature(signature, "forged")


def test_bias_rejects_non_allowlisted_knob_and_numeric_bool_forgery():
    bias = {
        "schema": BIAS_SCHEMA,
        "style_weights_ppm": BASELINE_STYLE_WEIGHTS_PPM,
        "knob_delta": {"world_coordinates": 200_000},
    }
    with pytest.raises(B3PriorError, match="allowlist"):
        validate_bias(bias)
    bias["knob_delta"] = {"corridor_prob": False}
    with pytest.raises(B3PriorError, match="integer"):
        validate_bias(bias)


def test_lane_rejects_seed_and_implementation_drift(tmp_path: Path):
    plan, plan_path, stock = _prepare(tmp_path)
    baseline_path, baseline_analysis = _lane(tmp_path, plan, plan_path, "baseline", "baseline")
    treatment_path, treatment_analysis = _lane(tmp_path, plan, plan_path, "treatment", "treatment")
    treatment = json.loads(treatment_path.read_text())
    treatment["maps"][0]["ordinal"] = False
    _write(treatment_path, treatment)
    with pytest.raises(B3PriorError, match="must be an integer"):
        evaluate_campaign(
            plan_path, stock, baseline_path, baseline_analysis,
            treatment_path, treatment_analysis, tmp_path / "out.json",
            repo_root=ROOT, _implementation=_implementation(),
        )
    treatment["maps"][0]["ordinal"] = 0
    treatment["maps"][0]["seed"] += 1
    _write(treatment_path, treatment)
    with pytest.raises(B3PriorError, match="seed/ordinal"):
        evaluate_campaign(
            plan_path, stock, baseline_path, baseline_analysis,
            treatment_path, treatment_analysis, tmp_path / "out.json",
            repo_root=ROOT, _implementation=_implementation(),
        )
    treatment["maps"][0]["seed"] -= 1
    treatment["implementation"]["repository_tree"] = "9" * 40
    _write(treatment_path, treatment)
    with pytest.raises(B3PriorError, match="identity drifted"):
        evaluate_campaign(
            plan_path, stock, baseline_path, baseline_analysis,
            treatment_path, treatment_analysis, tmp_path / "out.json",
            repo_root=ROOT, _implementation=_implementation(),
        )


def test_lane_rejects_synthetic_claim_and_artifact_drift(tmp_path: Path):
    plan, plan_path, stock = _prepare(tmp_path)
    baseline_path, baseline_analysis = _lane(tmp_path, plan, plan_path, "baseline", "baseline")
    treatment_path, treatment_analysis = _lane(tmp_path, plan, plan_path, "treatment", "treatment")
    treatment = json.loads(treatment_path.read_text())
    treatment["synthetic_claims"] = True
    _write(treatment_path, treatment)
    with pytest.raises(B3PriorError, match="synthetic"):
        evaluate_campaign(
            plan_path, stock, baseline_path, baseline_analysis,
            treatment_path, treatment_analysis, tmp_path / "out.json",
            repo_root=ROOT, _implementation=_implementation(),
        )
    treatment["synthetic_claims"] = False
    _write(treatment_path, treatment)
    first = plan["pairs"][0]["treatment"]["map"]
    path = treatment_analysis / f"{first}.design-signature.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(B3PriorError, match="artifact identity drifted"):
        evaluate_campaign(
            plan_path, stock, baseline_path, baseline_analysis,
            treatment_path, treatment_analysis, tmp_path / "out.json",
            repo_root=ROOT, _implementation=_implementation(),
        )


def test_non_improving_treatment_emits_red_regression_decision(tmp_path: Path):
    plan, plan_path, stock = _prepare(tmp_path)
    baseline_path, baseline_analysis = _lane(tmp_path, plan, plan_path, "baseline", "baseline")
    treatment_path, treatment_analysis = _lane(tmp_path, plan, plan_path, "treatment", "baseline")
    output = tmp_path / "red.json"
    report = evaluate_campaign(
        plan_path, stock, baseline_path, baseline_analysis,
        treatment_path, treatment_analysis, output,
        repo_root=ROOT, _implementation=_implementation(),
    )
    assert report["status"] == "red"
    assert report["passed"] is False
    assert report["decision"]["all_metrics_improved"] is False
    assert "all_metrics_improved" in report["failures"]


def test_green_report_rejects_metric_and_diversity_forgery(tmp_path: Path):
    report, _output = build_green_campaign(tmp_path)
    forged = copy.deepcopy(report)
    forged["metrics"]["treatment"]["topology_distance_ppm"] += 1
    with pytest.raises(B3PriorError, match="do not recompute"):
        validate_campaign_report(forged)
    forged = copy.deepcopy(report)
    forged["diversity"]["treatment"]["style_simpson_ppm"] = 1_000_000
    with pytest.raises(B3PriorError, match="style-diversity arithmetic"):
        validate_campaign_report(forged)
    forged = copy.deepcopy(report)
    forged["decision"]["all_metrics_improved"] = False
    with pytest.raises(B3PriorError, match="decision does not recompute"):
        validate_campaign_report(forged)


def test_campaign_report_schema_accepts_green_report(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    report, _output = build_green_campaign(tmp_path)
    schema = json.loads((ROOT / "schemas/q2-b3-design-prior-campaign-v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(report)
