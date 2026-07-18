from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from tools import run_b2_qualification as qualification
from tools.materialize_generated_cohort import MaterializeCohortError
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
    repo = Path(__file__).resolve().parents[1]
    boundary = tmp_path / "boundary-proof.json"
    boundary.write_bytes(_canonical({"schema": "fixture-boundary-v1"}))
    pinned_python = tmp_path / "pinned-python"
    zstandard_init = tmp_path / "zstandard-init.py"
    zstandard_backend = tmp_path / "zstandard-backend.so"
    for path in (pinned_python, zstandard_init, zstandard_backend):
        path.write_bytes(f"fixture:{path.name}\n".encode("ascii"))
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
        "pinned_python": pinned_python,
        "zstandard_init": zstandard_init,
        "zstandard_backend": zstandard_backend,
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
    for name in (
        "design.md", "plan.md", "b1.json", "syntax.json", "q2tool",
        "cm", "pmove", "hook", "fall", "hook-attestation.json",
        "packer", "verifier",
    ):
        path = tmp_path / name
        path.write_bytes(f"fixture:{name}\n".encode("ascii"))
    (tmp_path / "b1.json").write_bytes(
        (repo / "docs/multires/B1-GATE.json").read_bytes()
    )
    (tmp_path / "baseq2").mkdir()
    (tmp_path / "baseq2/pak0.pak").write_bytes(b"fixture:pak0\n")
    (tmp_path / "client/release").mkdir(parents=True)
    (tmp_path / "client/release/q2-cm-oracle").write_bytes(
        (tmp_path / "cm").read_bytes()
    )
    (tmp_path / "client/release/q2-pmove-oracle").write_bytes(
        (tmp_path / "pmove").read_bytes()
    )
    (tmp_path / "lithium/tools").mkdir(parents=True)
    (tmp_path / "lithium/tools/q2-hook-oracle").write_bytes(
        (tmp_path / "hook").read_bytes()
    )
    return argparse.Namespace(**values)


@pytest.fixture(autouse=True)
def _stub_final_materialization_authority_preflight(monkeypatch) -> None:
    gate = Path(__file__).resolve().parents[1] / "docs/multires/B1-GATE.json"

    def fake_preflight(**kwargs):
        return {
            "schema": "q2-b2-materialization-authority-preflight-v1",
            "passed": True,
            "authorities": {
                "b1_gate": {
                    "filename": gate.name,
                    "expected_sha256": _sha(gate),
                    "actual_sha256": _sha(gate),
                    "passed": True,
                },
                **{
                    label: {
                        "filename": path.name,
                        "expected_sha256": _sha(path),
                        "actual_sha256": _sha(path),
                        "passed": True,
                    }
                    for label, path in {
                        "cm": kwargs["cm_oracle"],
                        "pmove": kwargs["pmove_oracle"],
                        "hook": kwargs["hook_oracle"],
                        "fall": kwargs["fall_oracle"],
                        "hook_attestation": kwargs["hook_parity_attestation"],
                    }.items()
                },
            },
        }

    monkeypatch.setattr(
        qualification, "preflight_materialization_authorities", fake_preflight
    )


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
            passed_names = {
                f"b2q26_driver_map_{ordinal:02d}"
                for ordinal in range(28 if stage == "source" else 20)
            }
            rows = []
            for ordinal in range(28):
                name = f"b2q26_driver_map_{ordinal:02d}"
                passed = name in passed_names
                rows.append({
                    "ordinal": ordinal, "map": name,
                    "criteria": {"fixture-stage": passed},
                    "evidence_sha256": hashlib.sha256(
                        f"{stage}:{name}".encode("ascii")
                    ).hexdigest(),
                    "failures": [] if passed else ["fixture semantic rejection"],
                    "passed": passed,
                })
            report = {
                "schema": STAGE_SCHEMA,
                "qualification_id": self.plan["qualification_id"],
                "mode": "qualification", "stage": stage,
                "non_admissible": True, "retryable": True,
                "final_cohort_authorized": False,
                "toolchain_authority_sha256": self.plan[
                    "toolchain_authority"
                ]["sha256"],
                "input_report_sha256": prior, "map_count": 28,
                "pass_count": len(passed_names), "maps": rows, "failures": [],
            }
        elif stage == "infrastructure":
            report = {
                "schema": INFRASTRUCTURE_SCHEMA,
                "qualification_id": self.plan["qualification_id"],
                "mode": "qualification", "non_admissible": True,
                "retryable": True, "final_cohort_authorized": False,
                "toolchain_authority_sha256": self.plan[
                    "toolchain_authority"
                ]["sha256"],
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
                "toolchain_authority": {
                    key: self.plan["toolchain_authority"][key]
                    for key in ("bytes", "sha256")
                },
                "end_to_end": {
                    "pass_count": 20,
                    "passed_maps": [
                        f"b2q26_driver_map_{ordinal:02d}" for ordinal in range(20)
                    ],
                },
                "authorization": {
                    "final_declaration_allowed_by_this_report": False,
                    "qualification_artifact_reuse_as_final_evidence": False,
                    "passing_subset_admissible": False,
                },
                "failures": [],
            }
        report_path.write_bytes(_canonical(report))
        return SimpleNamespace(returncode=0)


class _ParserCaptured(Exception):
    pass


def _actual_parser(path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.ArgumentParser:
    captured: dict[str, argparse.ArgumentParser] = {}

    def intercept(parser, args=None, namespace=None):
        del args, namespace
        captured["parser"] = parser
        raise _ParserCaptured

    old_argv = sys.argv
    try:
        with monkeypatch.context() as patch:
            patch.setattr(argparse.ArgumentParser, "parse_args", intercept)
            sys.argv = [str(path)]
            with pytest.raises(_ParserCaptured):
                runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv
    return captured["parser"]


def test_command_plan_is_sequential_and_dry_to_build(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plan = build_plan(args)
    assert [step["stage"] for step in plan["commands"]] == list(DRIVER_STAGES)
    assert not args.workspace.exists()
    assert plan["authorization"] == {
        "non_admissible": True, "final_cohort_authorized": False,
        "deploy_allowed": False, "training_allowed": False,
    }
    assert plan["schema"] == "q2-b2-qualification-driver-plan-v2"
    assert len(plan["pinned_inputs"]) == 17
    assert plan["materialization_authority_preflight"]["passed"] is True


def test_build_plan_runs_exact_final_authority_preflight_before_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    calls = []
    original = qualification.preflight_materialization_authorities

    def recording_preflight(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        qualification, "preflight_materialization_authorities",
        recording_preflight,
    )
    build_plan(args)

    assert len(calls) == 1
    assert calls[0] == {
        "cm_oracle": args.cm_oracle,
        "pmove_oracle": args.pmove_oracle,
        "hook_oracle": args.hook_oracle,
        "fall_oracle": args.fall_oracle,
        "hook_parity_attestation": args.hook_attestation,
    }
    assert not args.workspace.exists()


def test_stale_final_authority_pin_refuses_before_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)

    def reject(**_kwargs):
        raise MaterializeCohortError(
            "authority-preflight", "current B1 gate exact bytes differ"
        )

    monkeypatch.setattr(
        qualification, "preflight_materialization_authorities", reject
    )
    with pytest.raises(
        QualificationDriverError,
        match="final materialization authority preflight failed",
    ):
        build_plan(args)
    assert not args.workspace.exists()


def test_missing_canonical_atlas_authority_fails_before_generation(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    (args.client_root / "release/q2-cm-oracle").unlink()
    with pytest.raises(QualificationDriverError, match="absent or a symlink"):
        build_plan(args)
    assert not args.workspace.exists()


def test_canonical_atlas_authority_must_match_supplied_bytes(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (args.lithium_root / "tools/q2-hook-oracle").write_bytes(b"wrong hook\n")
    with pytest.raises(QualificationDriverError, match="canonical Atlas hook_oracle"):
        build_plan(args)
    assert not args.workspace.exists()


def test_every_generated_argv_parses_with_its_actual_stage_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_plan(_args(tmp_path))
    for step in plan["commands"]:
        command = step["command"]
        parser = _actual_parser(Path(command[1]), monkeypatch)
        parsed = parser.parse_args(command[2:])
        assert parsed is not None, step["stage"]


def test_assemble_receives_every_retained_replay_input(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    command = next(
        step["command"] for step in plan["commands"]
        if step["stage"] == "assemble"
    )
    arguments = dict(zip(command[2::2], command[3::2]))
    assert {
        "--source-root", "--source-cold-root", "--compiled-root",
        "--compile-evidence-root", "--q2tool", "--basedir",
        "--compiled-cm-evidence-root", "--cm-oracle", "--pmove-oracle",
        "--hook-oracle", "--fall-oracle", "--hook-attestation",
        "--python-runtime", "--materialized-root",
        "--materialization-log-root", "--claims-root", "--analysis-root",
        "--atlas-evidence-root", "--promotion-evidence-root",
        "--infrastructure-evidence-root", "--syntax-report",
    }.issubset(arguments)


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
    promotion = json.loads(Path(plan["commands"][6]["report"]).read_text())
    assert promotion["pass_count"] == 20
    assert [row["ordinal"] for row in promotion["maps"] if row["passed"]] == list(range(20))


def test_resume_rejects_pinned_runtime_byte_drift(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plan = build_plan(args)
    run_plan(plan, resume=False, runner=FakeTools(plan))
    args.zstandard_backend.write_bytes(b"drifted backend\n")
    with pytest.raises(QualificationDriverError, match="runtime input drifted"):
        run_plan(plan, resume=True, runner=FakeTools(plan))


def test_resume_rejects_pinned_qualification_input_drift(tmp_path: Path) -> None:
    args = _args(tmp_path)
    plan = build_plan(args)
    run_plan(plan, resume=False, runner=FakeTools(plan))
    args.packer.write_bytes(b"drifted packer\n")
    with pytest.raises(QualificationDriverError, match="qualification input drifted"):
        run_plan(plan, resume=True, runner=FakeTools(plan))


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
