from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_b2_qualification import (
    DRIVER_STAGES,
    INFRASTRUCTURE_SCHEMA,
    QUALIFICATION_SCHEMA,
    STAGES,
    STAGE_SCHEMA,
    QualificationDriverError,
    _canonical,
    build_plan,
    run_plan,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path: Path) -> argparse.Namespace:
    repo = tmp_path / "repo"
    repo.mkdir()
    boundary = tmp_path / "boundary-proof.json"
    boundary.write_bytes(_canonical({"schema": "fixture-boundary-v1"}))
    values = {
        "workspace": tmp_path / "qualification-workspace",
        "qualification_id": "b2q26_driver_fixture",
        "seed_base": 99_500_000,
        "repo_root": repo,
        "design": tmp_path / "design.md",
        "plan": tmp_path / "plan.md",
        "b1_gate": tmp_path / "b1.json",
        "boundary_proof_report": boundary,
        "syntax_report": tmp_path / "syntax.json",
        "q2tool": tmp_path / "q2tool",
        "basedir": tmp_path / "baseq2",
        "cm_oracle": tmp_path / "cm",
        "pmove_oracle": tmp_path / "pmove",
        "hook_oracle": tmp_path / "hook",
        "fall_oracle": tmp_path / "fall",
        "hook_attestation": tmp_path / "hook-attestation.json",
        "pinned_python": tmp_path / "pinned-python",
        "client_root": tmp_path / "client",
        "lithium_root": tmp_path / "lithium",
        "packer": tmp_path / "packer",
        "verifier": tmp_path / "verifier",
        "source_jobs": 4,
        "compile_jobs": 4,
        "cm_jobs": 4,
        "atlas_jobs": 4,
        "promotion_jobs": 4,
        "compile_timeout": 3600.0,
        "dry_run": False,
        "resume": False,
    }
    return argparse.Namespace(**values)


class FakeTools:
    def __init__(self, plan: dict, fail_at: str | None = None) -> None:
        self.plan = plan
        self.fail_at = fail_at
        self.calls: list[str] = []

    def __call__(self, command, *, cwd, check):
        del cwd, check
        step = next(item for item in self.plan["commands"] if item["command"] == command)
        stage = step["stage"]
        self.calls.append(stage)
        if stage == self.fail_at:
            return SimpleNamespace(returncode=19)
        report_path = Path(step["report"])
        if stage in STAGES:
            index = STAGES.index(stage)
            prior = None if index == 0 else _sha(Path(
                self.plan["commands"][index - 1]["report"]
            ))
            report = {
                "schema": STAGE_SCHEMA,
                "qualification_id": self.plan["qualification_id"],
                "mode": "qualification", "stage": stage,
                "non_admissible": True, "retryable": True,
                "final_cohort_authorized": False,
                "input_report_sha256": prior, "map_count": 28,
                "pass_count": 20, "maps": [], "failures": [],
            }
        elif stage == "infrastructure":
            report = {
                "schema": INFRASTRUCTURE_SCHEMA,
                "qualification_id": self.plan["qualification_id"],
                "mode": "qualification", "non_admissible": True,
                "retryable": True, "final_cohort_authorized": False,
                "stage_report_sha256s": {
                    name: _sha(Path(self.plan["commands"][index]["report"]))
                    for index, name in enumerate(STAGES)
                },
                "failures": [],
            }
        else:
            report = {
                "schema": QUALIFICATION_SCHEMA, "status": "green",
                "qualification_id": self.plan["qualification_id"],
                "non_admissible": True, "retryable": True,
                "final_cohort_authorized": False,
                "authorization": {
                    "final_declaration_allowed_by_this_report": False,
                    "qualification_artifact_reuse_as_final_evidence": False,
                    "passing_subset_admissible": False,
                },
                "failures": [],
            }
        report_path.write_bytes(_canonical(report))
        return SimpleNamespace(returncode=0)


def test_command_plan_is_sequential_and_dry_to_build(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plan = build_plan(args)
    assert [step["stage"] for step in plan["commands"]] == list(DRIVER_STAGES)
    assert not args.workspace.exists()
    assert plan["authorization"] == {
        "non_admissible": True, "final_cohort_authorized": False,
        "deploy_allowed": False, "training_allowed": False,
    }


def test_fake_tools_prove_order_and_exact_raw_hash_chaining(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    fake = FakeTools(plan)
    state = run_plan(plan, resume=False, runner=fake)
    assert fake.calls == list(DRIVER_STAGES)
    assert [row["stage"] for row in state["completed"]] == list(DRIVER_STAGES)
    for earlier, later in zip(STAGES, STAGES[1:]):
        earlier_path = Path(next(
            step["report"] for step in plan["commands"] if step["stage"] == earlier
        ))
        later_path = Path(next(
            step["report"] for step in plan["commands"] if step["stage"] == later
        ))
        assert json.loads(later_path.read_text())["input_report_sha256"] == _sha(earlier_path)


def test_failure_stops_before_dependent_tools(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    fake = FakeTools(plan, fail_at="compile")
    with pytest.raises(QualificationDriverError, match="compile"):
        run_plan(plan, resume=False, runner=fake)
    assert fake.calls == ["source", "compile"]
    state = json.loads((Path(plan["workspace"]) / "driver-state.json").read_text())
    assert [row["stage"] for row in state["completed"]] == ["source"]


def test_resume_revalidates_raw_hashes_before_running_any_tool(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    run_plan(plan, resume=False, runner=FakeTools(plan))
    source_path = Path(plan["commands"][0]["report"])
    source = json.loads(source_path.read_text())
    source["fixture_tamper"] = True
    source_path.write_bytes(_canonical(source))
    fake = FakeTools(plan)
    with pytest.raises(QualificationDriverError, match="resume hash"):
        run_plan(plan, resume=True, runner=fake)
    assert fake.calls == []


def test_clean_resume_skips_every_completed_tool(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    first = run_plan(plan, resume=False, runner=FakeTools(plan))
    fake = FakeTools(plan)
    second = run_plan(plan, resume=True, runner=fake)
    assert second == first
    assert fake.calls == []


def test_existing_workspace_requires_resume_and_plan_has_no_admission_action(
    tmp_path: Path,
) -> None:
    plan = build_plan(_args(tmp_path))
    Path(plan["workspace"]).mkdir()
    with pytest.raises(QualificationDriverError, match="fresh"):
        run_plan(plan, resume=False, runner=FakeTools(plan))
    tokens = " ".join(
        token for step in plan["commands"] for token in step["command"]
    ).lower()
    for forbidden in ("deploy", "training", "trainer", "service", "systemctl"):
        assert forbidden not in tokens
