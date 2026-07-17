#!/usr/bin/env python3
"""Run one resumable, disposable, non-admissible B2 qualification workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


PLAN_SCHEMA = "q2-b2-qualification-driver-plan-v1"
STATE_SCHEMA = "q2-b2-qualification-driver-state-v1"
STAGE_SCHEMA = "q2-b2-qualification-stage-v1"
INFRASTRUCTURE_SCHEMA = "q2-b2-qualification-infrastructure-v1"
QUALIFICATION_SCHEMA = "q2-b2-toolchain-qualification-v1"
QUALIFICATION_ID = re.compile(r"^b2q26_[a-z0-9][a-z0-9_-]*$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
STAGES = (
    "source", "compile", "compiled-cm-preflight", "materialization",
    "claims", "atlas-build", "generated-promotion",
)
DRIVER_STAGES = (*STAGES, "infrastructure", "assemble")


class QualificationDriverError(RuntimeError):
    """The qualification driver refused to weaken or ambiguously resume a run."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationDriverError(f"cannot load canonical JSON {path}: {error}") from error
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise QualificationDriverError(f"JSON is not a canonical object: {path}")
    return dict(value), raw


def _arg(command: list[str], name: str, value: Path | str | int | float) -> None:
    command.extend((name, str(value)))


def _tool(repo: Path, name: str) -> Path:
    return repo / "tools" / name


def _step(stage: str, command: Sequence[str], report: Path) -> dict[str, Any]:
    return {"stage": stage, "command": list(command), "report": str(report)}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    workspace = _absolute(args.workspace)
    repo = _absolute(args.repo_root)
    python = _absolute(Path(sys.executable))
    reports = workspace / "reports"
    declaration = workspace / "qualification-declaration.json"
    source = workspace / "source"
    source_cold = workspace / "source-cold"
    compile_staging = workspace / "compile-staging"
    compiled = workspace / "compiled"
    compile_logs = workspace / "compile-logs"
    cm_evidence = workspace / "compiled-cm-evidence"
    materialize_staging = workspace / "materialize-staging"
    materialized = workspace / "materialized"
    materialize_logs = workspace / "materialize-logs"
    claims_staging = workspace / "claims-staging"
    claims = workspace / "claims"
    analysis = workspace / "analysis"
    atlas_evidence = workspace / "atlas-evidence"
    promotion_evidence = workspace / "promotion-evidence"
    infrastructure_evidence = workspace / "infrastructure-evidence"
    report_paths = {
        "source": reports / "source.json",
        "compile": reports / "compile.json",
        "compiled-cm-preflight": reports / "compiled-cm-preflight.json",
        "materialization": reports / "materialization.json",
        "claims": reports / "claims.json",
        "atlas-build": reports / "atlas-build.json",
        "generated-promotion": reports / "generated-promotion.json",
        "infrastructure": reports / "infrastructure.json",
        "assemble": reports / "qualification.json",
    }

    commands: list[dict[str, Any]] = []
    command = [str(python), str(_tool(repo, "run_b2_qualification_source.py"))]
    for name, value in (
        ("--qualification-id", args.qualification_id), ("--seed-base", args.seed_base),
        ("--source-root", source), ("--cold-root", source_cold),
        ("--declaration", declaration), ("--report", report_paths["source"]),
        ("--workers", args.source_jobs), ("--repo-root", repo),
    ):
        _arg(command, name, value)
    commands.append(_step("source", command, report_paths["source"]))

    command = [str(python), str(_tool(repo, "run_b2_qualification_compile.py"))]
    for name, value in (
        ("--declaration", declaration), ("--source-report", report_paths["source"]),
        ("--source-root", source), ("--staging-root", compile_staging),
        ("--compiled-root", compiled), ("--log-root", compile_logs),
        ("--report", report_paths["compile"]), ("--q2tool", _absolute(args.q2tool)),
        ("--basedir", _absolute(args.basedir)), ("--jobs", args.compile_jobs),
        ("--timeout-seconds", args.compile_timeout),
    ):
        _arg(command, name, value)
    commands.append(_step("compile", command, report_paths["compile"]))

    command = [str(python), str(_tool(repo, "run_b2_qualification_compiled_cm.py"))]
    for name, value in (
        ("--declaration", declaration), ("--compile-report", report_paths["compile"]),
        ("--compiled-root", compiled), ("--cm-oracle", _absolute(args.cm_oracle)),
        ("--evidence-root", cm_evidence),
        ("--report", report_paths["compiled-cm-preflight"]),
        ("--jobs", args.cm_jobs),
    ):
        _arg(command, name, value)
    commands.append(_step(
        "compiled-cm-preflight", command, report_paths["compiled-cm-preflight"]
    ))

    command = [str(python), str(_tool(repo, "run_b2_qualification_postcompile.py")), "materialize"]
    for name, value in (
        ("--declaration", declaration),
        ("--prior-report", report_paths["compiled-cm-preflight"]),
        ("--prior-evidence-root", cm_evidence),
        ("--compile-report", report_paths["compile"]),
        ("--compiled-root", compiled), ("--staging-root", materialize_staging),
        ("--materialized-root", materialized), ("--log-root", materialize_logs),
        ("--report", report_paths["materialization"]),
        ("--cm-oracle", _absolute(args.cm_oracle)),
        ("--pmove-oracle", _absolute(args.pmove_oracle)),
        ("--hook-oracle", _absolute(args.hook_oracle)),
        ("--fall-oracle", _absolute(args.fall_oracle)),
        ("--hook-attestation", _absolute(args.hook_attestation)),
        ("--python", _absolute(args.pinned_python)),
    ):
        _arg(command, name, value)
    commands.append(_step("materialization", command, report_paths["materialization"]))

    command = [str(python), str(_tool(repo, "run_b2_qualification_postcompile.py")), "claims"]
    for name, value in (
        ("--declaration", declaration),
        ("--prior-report", report_paths["materialization"]),
        ("--upstream-report", report_paths["compiled-cm-preflight"]),
        ("--materialized-root", materialized), ("--staging-root", claims_staging),
        ("--claims-root", claims), ("--report", report_paths["claims"]),
    ):
        _arg(command, name, value)
    commands.append(_step("claims", command, report_paths["claims"]))

    command = [str(python), str(_tool(repo, "run_b2_qualification_atlas.py"))]
    for name, value in (
        ("--declaration", declaration), ("--claims-report", report_paths["claims"]),
        ("--claims-root", claims), ("--analysis-root", analysis),
        ("--evidence-root", atlas_evidence), ("--report", report_paths["atlas-build"]),
        ("--repo-root", repo), ("--client-root", _absolute(args.client_root)),
        ("--lithium-root", _absolute(args.lithium_root)),
        ("--hook-attestation", _absolute(args.hook_attestation)),
        ("--fall-oracle", _absolute(args.fall_oracle)),
        ("--packer", _absolute(args.packer)), ("--verifier", _absolute(args.verifier)),
        ("--jobs", args.atlas_jobs), ("--python", _absolute(args.pinned_python)),
    ):
        _arg(command, name, value)
    commands.append(_step("atlas-build", command, report_paths["atlas-build"]))

    command = [str(python), str(_tool(repo, "run_b2_qualification_promotion.py"))]
    for name, value in (
        ("--declaration", declaration), ("--atlas-report", report_paths["atlas-build"]),
        ("--claims-root", claims), ("--analysis-root", analysis),
        ("--atlas-evidence-root", atlas_evidence),
        ("--b1-gate", _absolute(args.b1_gate)),
        ("--evidence-root", promotion_evidence),
        ("--report", report_paths["generated-promotion"]),
        ("--repo-root", repo), ("--jobs", args.promotion_jobs),
    ):
        _arg(command, name, value)
    commands.append(_step(
        "generated-promotion", command, report_paths["generated-promotion"]
    ))

    command = [str(python), str(_tool(repo, "run_b2_qualification_infrastructure.py"))]
    _arg(command, "--declaration", declaration)
    for stage in STAGES:
        _arg(command, f"--{stage}-report", report_paths[stage])
    for name, value in (
        ("--source-root", source), ("--compiled-root", compiled),
        ("--materialized-root", materialized), ("--claims-root", claims),
        ("--analysis-root", analysis),
        ("--promotion-evidence-root", promotion_evidence),
        ("--syntax-report", _absolute(args.syntax_report)),
        ("--evidence-root", infrastructure_evidence),
        ("--report", report_paths["infrastructure"]), ("--repo-root", repo),
    ):
        _arg(command, name, value)
    commands.append(_step("infrastructure", command, report_paths["infrastructure"]))

    command = [str(python), str(_tool(repo, "assemble_b2_qualification.py"))]
    for name, value in (
        ("--design", _absolute(args.design)), ("--plan", _absolute(args.plan)),
        ("--repo-root", repo), ("--b1-gate", _absolute(args.b1_gate)),
        ("--boundary-proof-report", _absolute(args.boundary_proof_report)),
        ("--declaration", declaration), ("--source-report", report_paths["source"]),
        ("--compile-report", report_paths["compile"]),
        ("--compiled-cm-preflight-report", report_paths["compiled-cm-preflight"]),
        ("--materialization-report", report_paths["materialization"]),
        ("--claims-report", report_paths["claims"]),
        ("--atlas-build-report", report_paths["atlas-build"]),
        ("--generated-promotion-report", report_paths["generated-promotion"]),
        ("--infrastructure-report", report_paths["infrastructure"]),
        ("--output", report_paths["assemble"]),
    ):
        _arg(command, name, value)
    commands.append(_step("assemble", command, report_paths["assemble"]))

    return {
        "schema": PLAN_SCHEMA,
        "qualification_id": args.qualification_id,
        "workspace": str(workspace),
        "repo_root": str(repo),
        "declaration": str(declaration),
        "boundary_proof_report": str(_absolute(args.boundary_proof_report)),
        "boundary_mode": "preexisting-wsl-compile-and-proof",
        "outputs": sorted(str(path) for path in {
            declaration, source, source_cold, compile_staging, compiled,
            compile_logs, cm_evidence, materialize_staging, materialized,
            materialize_logs, claims_staging, claims, analysis, atlas_evidence,
            promotion_evidence, infrastructure_evidence, *report_paths.values(),
        }),
        "commands": commands,
        "authorization": {
            "non_admissible": True,
            "final_cohort_authorized": False,
            "deploy_allowed": False,
            "training_allowed": False,
        },
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    workspace = Path(str(plan["workspace"]))
    repo = Path(str(plan["repo_root"]))
    if not QUALIFICATION_ID.fullmatch(str(plan["qualification_id"])):
        raise QualificationDriverError("qualification ID is invalid")
    if any("retired" in part.lower() or "final" in part.lower() for part in workspace.parts):
        raise QualificationDriverError("workspace is under a retired/final path")
    try:
        workspace.relative_to(repo)
    except ValueError:
        pass
    else:
        raise QualificationDriverError("qualification workspace must be outside the repository")
    stages = [step["stage"] for step in plan["commands"]]
    if stages != list(DRIVER_STAGES):
        raise QualificationDriverError("qualification command order differs")
    if plan["authorization"] != {
        "non_admissible": True, "final_cohort_authorized": False,
        "deploy_allowed": False, "training_allowed": False,
    }:
        raise QualificationDriverError("qualification authorization differs")
    reports = [Path(step["report"]) for step in plan["commands"]]
    if len(set(reports)) != len(reports) or any(
        path != workspace / "reports" / path.name for path in reports
    ):
        raise QualificationDriverError("reports are not exclusive workspace outputs")
    outputs = [Path(str(path)) for path in plan.get("outputs", [])]
    if len(outputs) != len(set(outputs)):
        raise QualificationDriverError("qualification outputs overlap")
    for path in outputs:
        try:
            relative = path.relative_to(workspace)
        except ValueError as error:
            raise QualificationDriverError(
                f"qualification output escapes workspace: {path}"
            ) from error
        if not relative.parts:
            raise QualificationDriverError("workspace itself cannot be a stage output")
    directory_outputs = [path for path in outputs if path not in reports and path.suffix == ""]
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(directory_outputs)
        for right in directory_outputs[index + 1:]
    ):
        raise QualificationDriverError("qualification output roots are not disjoint")


def _validate_stage_report(
    path: Path, stage: str, qualification_id: str, expected_input: str | None,
) -> str:
    report, raw = _load_canonical(path)
    required = {
        "schema", "qualification_id", "mode", "stage", "non_admissible",
        "retryable", "final_cohort_authorized", "input_report_sha256",
        "map_count", "pass_count", "maps", "failures",
    }
    if (
        not required.issubset(report)
        or report["schema"] != STAGE_SCHEMA
        or report["qualification_id"] != qualification_id
        or report["mode"] != "qualification"
        or report["stage"] != stage
        or report["non_admissible"] is not True
        or report["retryable"] is not True
        or report["final_cohort_authorized"] is not False
        or report["input_report_sha256"] != expected_input
        or report["failures"] != []
    ):
        raise QualificationDriverError(f"{stage} report identity/hash chain differs")
    return _sha256(raw)


def _validate_infrastructure(path: Path, qualification_id: str, hashes: Mapping[str, str]) -> str:
    report, raw = _load_canonical(path)
    if (
        report.get("schema") != INFRASTRUCTURE_SCHEMA
        or report.get("qualification_id") != qualification_id
        or report.get("mode") != "qualification"
        or report.get("non_admissible") is not True
        or report.get("retryable") is not True
        or report.get("final_cohort_authorized") is not False
        or report.get("stage_report_sha256s") != dict(hashes)
        or report.get("failures") != []
    ):
        raise QualificationDriverError("infrastructure report/hash bindings differ")
    return _sha256(raw)


def _validate_assembly(path: Path, qualification_id: str) -> str:
    report, raw = _load_canonical(path)
    if (
        report.get("schema") != QUALIFICATION_SCHEMA
        or report.get("status") != "green"
        or report.get("qualification_id") != qualification_id
        or report.get("non_admissible") is not True
        or report.get("retryable") is not True
        or report.get("final_cohort_authorized") is not False
        or report.get("authorization") != {
            "final_declaration_allowed_by_this_report": False,
            "qualification_artifact_reuse_as_final_evidence": False,
            "passing_subset_admissible": False,
        }
        or report.get("failures") != []
    ):
        raise QualificationDriverError("assembled qualification is not green/non-admissible")
    return _sha256(raw)


def _write_state(path: Path, state: Mapping[str, Any], *, exclusive: bool = False) -> None:
    payload = _canonical(state)
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise QualificationDriverError("stale state temporary file exists")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_plan(
    plan: Mapping[str, Any], *, resume: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    _validate_plan(plan)
    workspace = Path(str(plan["workspace"]))
    state_path = workspace / "driver-state.json"
    plan_sha256 = _sha256(_canonical(plan))
    boundary = Path(str(plan["boundary_proof_report"]))
    if boundary.is_symlink() or not boundary.is_file():
        raise QualificationDriverError("preexisting WSL boundary proof is absent or a symlink")
    boundary_sha256 = _file_sha256(boundary)
    if resume:
        if not workspace.is_dir() or workspace.is_symlink():
            raise QualificationDriverError("resume workspace is absent or a symlink")
        state, _ = _load_canonical(state_path)
        if (
            state.get("schema") != STATE_SCHEMA
            or state.get("qualification_id") != plan["qualification_id"]
            or state.get("plan_sha256") != plan_sha256
            or state.get("boundary_proof_sha256") != boundary_sha256
            or not isinstance(state.get("completed"), list)
        ):
            raise QualificationDriverError("resume state/plan/boundary binding differs")
    else:
        if workspace.exists() or workspace.is_symlink():
            raise QualificationDriverError("qualification workspace must be fresh")
        if not workspace.parent.is_dir():
            raise QualificationDriverError("qualification workspace parent is absent")
        workspace.mkdir(mode=0o755)
        (workspace / "reports").mkdir(mode=0o755)
        state = {
            "schema": STATE_SCHEMA, "qualification_id": plan["qualification_id"],
            "plan_sha256": plan_sha256, "boundary_proof_sha256": boundary_sha256,
            "completed": [],
        }
        _write_state(state_path, state, exclusive=True)

    completed = state["completed"]
    if [row.get("stage") for row in completed] != list(DRIVER_STAGES[:len(completed)]):
        raise QualificationDriverError("resume completion order differs")
    stage_hashes: dict[str, str] = {}
    previous: str | None = None
    for index, step in enumerate(plan["commands"]):
        stage = str(step["stage"])
        report_path = Path(str(step["report"]))
        if stage in STAGES:
            validator = lambda: _validate_stage_report(
                report_path, stage, str(plan["qualification_id"]), previous
            )
        elif stage == "infrastructure":
            validator = lambda: _validate_infrastructure(
                report_path, str(plan["qualification_id"]), stage_hashes
            )
        else:
            validator = lambda: _validate_assembly(
                report_path, str(plan["qualification_id"])
            )
        if index < len(completed):
            digest = validator()
            if completed[index] != {"stage": stage, "report_sha256": digest}:
                raise QualificationDriverError(f"resume hash validation failed at {stage}")
        else:
            if report_path.exists() or report_path.is_symlink():
                raise QualificationDriverError(f"unrecorded stage output exists: {stage}")
            completed_process = runner(
                list(step["command"]), cwd=Path(str(plan["repo_root"])), check=False,
            )
            if completed_process.returncode != 0:
                raise QualificationDriverError(
                    f"qualification stopped at {stage}: exit {completed_process.returncode}"
                )
            digest = validator()
            completed.append({"stage": stage, "report_sha256": digest})
            _write_state(state_path, state)
        if stage in STAGES:
            stage_hashes[stage] = digest
            previous = digest
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--qualification-id", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--boundary-proof-report", type=Path, required=True)
    parser.add_argument("--syntax-report", type=Path, required=True)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--pinned-python", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--lithium-root", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--source-jobs", type=int, default=4)
    parser.add_argument("--compile-jobs", type=int, default=4)
    parser.add_argument("--cm-jobs", type=int, default=4)
    parser.add_argument("--atlas-jobs", type=int, default=4)
    parser.add_argument("--promotion-jobs", type=int, default=4)
    parser.add_argument("--compile-timeout", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.dry_run and args.resume:
            raise QualificationDriverError("dry-run and resume are mutually exclusive")
        plan = build_plan(args)
        _validate_plan(plan)
        if args.dry_run:
            sys.stdout.buffer.write(_canonical(plan))
            return 0
        state = run_plan(plan, resume=args.resume)
        sys.stdout.buffer.write(_canonical(state))
        return 0
    except (QualificationDriverError, OSError) as error:
        print(f"B2 qualification driver refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
