from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import maps.generator as generator_module
from tools import run_b3_design_prior_lanes as lane_module
from tools.run_b3_design_prior_campaign import canonical_bytes
from tools.run_b3_design_prior_lanes import (
    B3LaneError,
    ORACLE_BATCH_TIMEOUT_MAX_SECONDS,
    _parser,
    _preflight_configuration,
    _qknobs,
    bound_generator,
    lane_declaration,
)
from tests.test_b3_design_prior_campaign import _prepare, _sha, _signature


def _file(path: Path, *, executable: bool = False) -> Path:
    path.write_bytes(b"authority\n")
    if executable:
        path.chmod(0o755)
    return path


def _preflight_args(tmp_path: Path, plan_path: Path) -> dict:
    return {
        "plan_path": plan_path,
        "work_root": tmp_path / "work",
        "output_lane": tmp_path / "lane.json",
        "q2tool": _file(tmp_path / "q2tool", executable=True),
        "packer": _file(tmp_path / "q2-atlas-pack", executable=True),
        "verifier": _file(tmp_path / "q2-atlas-verify", executable=True),
        "basedir": tmp_path / "basedir",
        "cm_oracle": _file(tmp_path / "cm", executable=True),
        "pmove_oracle": _file(tmp_path / "pmove", executable=True),
        "hook_oracle": _file(tmp_path / "hook", executable=True),
        "fall_oracle": _file(tmp_path / "fall", executable=True),
        "hook_attestation": _file(tmp_path / "hook.json"),
        "b1_gate": _file(tmp_path / "b1.json"),
        "client_root": tmp_path / "client",
        "lithium_root": tmp_path / "lithium",
        "compile_timeout_seconds": 900.0,
        "materialize_timeout_seconds": 900,
        "oracle_batch_timeout_seconds": 60.0,
        "jobs": 4,
    }


def test_lane_declaration_reuses_balanced_b2_lifecycle(tmp_path: Path):
    plan, _plan_path, _stock = _prepare(tmp_path)
    baseline = lane_declaration(plan, "baseline")
    treatment = lane_declaration(plan, "treatment")
    assert len(baseline["maps"]) == len(treatment["maps"]) == 28
    assert [row["seed"] for row in baseline["maps"]] == [row["seed"] for row in treatment["maps"]]
    assert [row["style"] for row in baseline["maps"]] == [row["style"] for row in treatment["maps"]]
    assert set({
        style: sum(row["style"] == style for row in baseline["maps"])
        for style in set(row["style"] for row in baseline["maps"])
    }.values()) == {4}


def test_bound_generator_applies_only_frozen_knobs_and_restores_class(tmp_path: Path, monkeypatch):
    plan, _plan_path, _stock = _prepare(tmp_path)
    pair = plan["pairs"][0]
    expected = pair["treatment"]
    original_class = generator_module.MapGenerator
    observed = {}

    def fake_generate(name, seed, output, grid_n, style, observed_heat, gym):
        instance = generator_module.MapGenerator(seed=seed, style=style, gym=gym)
        observed.update(_qknobs(instance))

    monkeypatch.setattr(generator_module, "generate_map", fake_generate)
    bound_generator(plan, "treatment")(
        expected["map"], pair["seed"], tmp_path, grid=5, style=expected["style"],
    )
    assert observed == expected["generator_knobs"]
    assert generator_module.MapGenerator is original_class


def test_oracle_timeout_domain_refuses_before_any_output_mutation(tmp_path: Path):
    _plan, plan_path, _stock = _prepare(tmp_path / "plan")
    args = _preflight_args(tmp_path, plan_path)
    for directory in (args["basedir"], args["client_root"], args["lithium_root"]):
        directory.mkdir()
    args["oracle_batch_timeout_seconds"] = 3600.0
    with pytest.raises(B3LaneError, match=r"\(0,60\]"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    assert not args["output_lane"].exists()
    args["oracle_batch_timeout_seconds"] = False
    with pytest.raises(B3LaneError, match=r"\(0,60\]"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()


def test_missing_authority_and_overlap_refuse_before_mutation(tmp_path: Path):
    _plan, plan_path, _stock = _prepare(tmp_path / "plan")
    args = _preflight_args(tmp_path, plan_path)
    for directory in (args["basedir"], args["client_root"], args["lithium_root"]):
        directory.mkdir()
    args["cm_oracle"].unlink()
    with pytest.raises(B3LaneError, match="collision oracle"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    args["cm_oracle"] = _file(tmp_path / "cm2", executable=True)
    args["output_lane"] = args["work_root"] / "lane.json"
    with pytest.raises(B3LaneError, match="disjoint"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()


def test_atlas_binaries_are_preflighted_passed_and_bound(tmp_path: Path, monkeypatch):
    plan, plan_path, _stock = _prepare(tmp_path / "plan")
    args = _preflight_args(tmp_path, plan_path)
    for directory in (args["basedir"], args["client_root"], args["lithium_root"]):
        directory.mkdir()

    args["packer"] = Path("relative-q2-atlas-pack")
    with pytest.raises(B3LaneError, match="Atlas packer must be an absolute path"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    args["packer"] = tmp_path / "q2-atlas-pack"
    args["verifier"] = Path("relative-q2-atlas-verify")
    with pytest.raises(B3LaneError, match="Atlas verifier must be an absolute path"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    args["verifier"] = tmp_path / "q2-atlas-verify"

    args["packer"].unlink()
    with pytest.raises(B3LaneError, match="Atlas packer"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    packer = _file(tmp_path / "q2-atlas-pack-2", executable=True)
    args["packer"] = tmp_path / "q2-atlas-pack-link"
    args["packer"].symlink_to(packer)
    with pytest.raises(B3LaneError, match="Atlas packer"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    args["packer"] = packer
    args["verifier"].chmod(0o644)
    with pytest.raises(B3LaneError, match="Atlas verifier"):
        _preflight_configuration(**args)
    assert not args["work_root"].exists()
    args["verifier"].chmod(0o755)

    def write_report(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(value))

    def declaration(path: Path) -> dict:
        return json.loads(path.read_text(encoding="ascii"))

    def source_freeze(declaration_path, _source, _cold, report_path, **_kwargs):
        rows = [{
            "map": row["map"],
            "source_static": {"static_ok": True},
            "source_files": {".map": {"sha256": _sha(f"layout-{row['ordinal']}")}},
        } for row in declaration(declaration_path)["maps"]]
        report = {"passed": True, "failures": [], "maps": rows}
        write_report(report_path, report)
        return report

    def compile_cohort(*positional, **_kwargs):
        report_path = positional[5]
        report = {"passed": True, "failures": []}
        write_report(report_path, report)
        return report

    def compiled_cm(**kwargs):
        report = {"passed": True, "failures": []}
        write_report(kwargs["output"], report)
        return report, _sha("compiled-cm")

    def materialize(**kwargs):
        report = {"passed": True, "failures": []}
        write_report(kwargs["report_path"], report)
        return report

    def claims(declaration_path, _materialized, claims_dir):
        claims_dir.mkdir()
        for row in declaration(declaration_path)["maps"]:
            (claims_dir / f"{row['map']}.bsp").write_bytes(
                f"bsp-{row['ordinal']}".encode("ascii")
            )
        return {"passed": True, "failures": []}

    observed = {}

    def atlas(declaration_path, claims_dir, analysis_dir, _diagnostics, report_path, **kwargs):
        observed.update({"packer": kwargs["packer"], "verifier": kwargs["verifier"]})
        analysis_dir.mkdir()
        for row in declaration(declaration_path)["maps"]:
            bsp_sha256 = hashlib.sha256(
                (claims_dir / f"{row['map']}.bsp").read_bytes()
            ).hexdigest()
            write_report(
                analysis_dir / f"{row['map']}.design-signature.json",
                _signature("baseline", row["ordinal"], bsp_sha256),
            )
        report = {"passed": True, "failures": []}
        write_report(report_path, report)
        return report

    implementation = plan["implementation"]
    monkeypatch.setattr(
        lane_module,
        "cohort_repository_binding",
        lambda _root: {
            "repository_commit": implementation["repository_commit"],
            "repository_tree": implementation["repository_tree"],
            "git_clean": implementation["git_clean"],
            "generator_sha256": implementation["generator_sha256"],
            "atlas_analyzer_authority_sha256": implementation[
                "analyzer_authority_sha256"
            ],
            "atlas_analyzer_authority_file_count": implementation[
                "analyzer_authority_file_count"
            ],
        },
    )
    monkeypatch.setattr(lane_module, "generate_source_freeze", source_freeze)
    monkeypatch.setattr(lane_module, "compile_generated_cohort", compile_cohort)
    monkeypatch.setattr(lane_module, "run_compiled_cm_preflight", compiled_cm)
    monkeypatch.setattr(lane_module, "materialize_cohort", materialize)
    monkeypatch.setattr(lane_module, "prepare_claims", claims)
    monkeypatch.setattr(lane_module, "build_atlas_campaign", atlas)
    monkeypatch.setattr(
        lane_module, "validate_claim_campaign",
        lambda *_args: {"passed": True, "failures": []},
    )

    report = lane_module.run_lane(
        plan_path=plan_path, lane="baseline", work_root=args["work_root"],
        output_lane=args["output_lane"], q2tool=args["q2tool"],
        packer=args["packer"], verifier=args["verifier"], basedir=args["basedir"],
        cm_oracle=args["cm_oracle"], pmove_oracle=args["pmove_oracle"],
        hook_oracle=args["hook_oracle"], fall_oracle=args["fall_oracle"],
        hook_attestation=args["hook_attestation"], b1_gate=args["b1_gate"],
        client_root=args["client_root"], lithium_root=args["lithium_root"],
    )
    assert observed == {"packer": args["packer"], "verifier": args["verifier"]}
    assert report["authorities"]["packer"]["sha256"] == hashlib.sha256(
        args["packer"].read_bytes()
    ).hexdigest()
    assert report["authorities"]["verifier"]["sha256"] == hashlib.sha256(
        args["verifier"].read_bytes()
    ).hexdigest()


def test_cli_default_cannot_reintroduce_71446_timeout():
    parser = _parser()
    args = parser.parse_args([
        "--plan", "p", "--lane", "baseline", "--work-root", "w",
        "--output-lane", "o", "--q2tool", "q", "--basedir", "b",
        "--packer", "k", "--verifier", "v",
        "--cm-oracle", "c", "--pmove-oracle", "m", "--hook-oracle", "h",
        "--fall-oracle", "f", "--hook-attestation", "a", "--b1-gate", "g",
        "--client-root", "r", "--lithium-root", "l",
    ])
    assert args.oracle_batch_timeout_seconds == ORACLE_BATCH_TIMEOUT_MAX_SECONDS == 60.0
