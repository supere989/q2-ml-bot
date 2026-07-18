from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import maps.generator as generator_module
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
from tests.test_b3_design_prior_campaign import _prepare


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


def test_cli_default_cannot_reintroduce_71446_timeout():
    parser = _parser()
    args = parser.parse_args([
        "--plan", "p", "--lane", "baseline", "--work-root", "w",
        "--output-lane", "o", "--q2tool", "q", "--basedir", "b",
        "--cm-oracle", "c", "--pmove-oracle", "m", "--hook-oracle", "h",
        "--fall-oracle", "f", "--hook-attestation", "a", "--b1-gate", "g",
        "--client-root", "r", "--lithium-root", "l",
    ])
    assert args.oracle_batch_timeout_seconds == ORACLE_BATCH_TIMEOUT_MAX_SECONDS == 60.0
