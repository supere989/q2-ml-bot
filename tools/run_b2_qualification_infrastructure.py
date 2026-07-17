#!/usr/bin/env python3
"""Prove qualification infrastructure with retained, byte-bound evidence."""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assemble_b2_qualification import (  # noqa: E402
    INFRASTRUCTURE_SCHEMA,
    REQUIRED_END_TO_END_PASSES,
    STAGES,
)
from tools.b2_qualification_stage_support import (  # noqa: E402
    QualificationStageError,
    canonical_bytes,
    current_implementation,
    exact_flat_files,
    exclusive_write,
    file_record,
    fresh_outputs,
    load_canonical,
    pinned_runtime_record,
    qualification_path,
    rename_noreplace,
    require,
    sha256_bytes,
    validate_declaration,
    validate_stage_report,
)
from tools.generator_claim_validator import (  # noqa: E402
    MAX_BUILD_RSS_BYTES,
    MAX_FULL_COLD_MILLISECONDS,
)
from tools.run_generator_cohort import STAGE_SUFFIXES  # noqa: E402


PINNED_PYTHON_VERSION = [3, 11, 4]
PINNED_PYTHON_SHA256 = "b25abf001748dc7ebb4b25013b2572d4e6913246b4c3b8e8b726b3da45494ff4"
SYNTAX_PYTHON_VERSION = [3, 10, 12]
SYNTAX_PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
CHECK_IDS = (
    "deterministic-cold-rebuild", "exact-stage-membership", "exclusive-create",
    "python310-syntax-floor", "resource-bounds", "timeout-fail-closed",
)


def validate_infrastructure_evidence(
    report: Mapping[str, Any], evidence_root: Path, *,
    declaration: Mapping[str, Any] | None = None,
    stage_reports: Mapping[str, Mapping[str, Any]] | None = None,
    roots: Mapping[str, Path] | None = None,
    syntax_report: Path | None = None,
    runtime_provider: Callable[[], Mapping[str, Any]] = pinned_runtime_record,
) -> None:
    """Replay retained check bytes and, when supplied, every live predicate."""

    exact_flat_files(evidence_root, {f"{check_id}.json" for check_id in CHECK_IDS},
                     "infrastructure evidence")
    checks = report.get("checks")
    require(isinstance(checks, list) and len(checks) == len(CHECK_IDS),
            "infrastructure check membership differs")
    by_id = {str(check.get("id")): check for check in checks if isinstance(check, Mapping)}
    require(set(by_id) == set(CHECK_IDS), "infrastructure check IDs differ")
    bodies: dict[str, Any] = {}
    for check_id in CHECK_IDS:
        evidence, raw = load_canonical(evidence_root / f"{check_id}.json")
        check = by_id[check_id]
        require(sha256_bytes(raw) == check.get("evidence_sha256"),
                f"infrastructure {check_id} evidence hash differs")
        require(set(evidence) == {
            "schema", "qualification_id", "id", "stage_report_sha256s",
            "evidence", "passed",
        }, f"infrastructure {check_id} evidence keys differ")
        require(evidence["schema"] == "q2-b2-qualification-infrastructure-evidence-v2"
                and evidence["qualification_id"] == report.get("qualification_id")
                and evidence["id"] == check_id
                and evidence["stage_report_sha256s"] == report.get("stage_report_sha256s")
                and evidence["passed"] is True,
                f"infrastructure {check_id} evidence binding differs")
        bodies[check_id] = evidence["evidence"]
    supplied = (declaration, stage_reports, roots, syntax_report)
    require(all(value is None for value in supplied) or all(value is not None for value in supplied),
            "complete infrastructure replay inputs are required")
    if declaration is None or stage_reports is None or roots is None or syntax_report is None:
        return
    membership = _membership_evidence(declaration, stage_reports, roots)
    cold, resources = _cold_and_resource_evidence(
        stage_reports, roots["atlas-build"]
    )
    syntax = _syntax_evidence(syntax_report, dict(runtime_provider()))
    probe_root = Path(tempfile.mkdtemp(prefix=".b2q-exclusive-replay-"))
    try:
        exclusive = _exclusive_probe(probe_root)
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)
    timeout = _timeout_probe()
    recomputed = {
        "deterministic-cold-rebuild": cold,
        "exact-stage-membership": membership,
        "exclusive-create": exclusive,
        "python310-syntax-floor": syntax,
        "resource-bounds": resources,
        "timeout-fail-closed": timeout,
    }
    for check_id in CHECK_IDS:
        require(canonical_bytes(bodies[check_id]) == canonical_bytes(recomputed[check_id]),
                f"infrastructure {check_id} evidence differs from live replay")


def _timeout_probe() -> dict[str, Any]:
    command = [sys.executable, "-I", "-c", "import time; time.sleep(60)"]
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=0.05)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return {
        "command": command, "interpreter": file_record(Path(sys.executable).resolve()),
        "timeout_milliseconds": 50, "timed_out": timed_out,
        "exit_code": process.returncode,
        "stdout": {"bytes": len(stdout), "sha256": sha256_bytes(stdout)},
        "stderr": {"bytes": len(stderr), "sha256": sha256_bytes(stderr)},
        "process_group_killed": timed_out and process.returncode == -signal.SIGKILL,
    }


def _exclusive_probe(root: Path) -> dict[str, Any]:
    path = root / ".exclusive-create-probe"
    exclusive_write(path, b"qualification-exclusive-create\n")
    second_errno = None
    try:
        exclusive_write(path, b"must-not-replace\n")
    except FileExistsError as error:
        second_errno = error.errno
    record = file_record(path)
    path.unlink()
    return {
        "first_create": record, "second_create_errno": second_errno,
        "expected_errno": errno.EEXIST,
        "original_bytes_preserved": record == {
            "bytes": len(b"qualification-exclusive-create\n"),
            "sha256": sha256_bytes(b"qualification-exclusive-create\n"),
        },
    }


def _membership_evidence(
    declaration: Mapping[str, Any], stage_reports: Mapping[str, Mapping[str, Any]],
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    all_maps = {str(row["map"]) for row in declaration["maps"]}
    source_passed = {
        str(row["map"])
        for row in stage_reports["source"]["maps"]
        if row["passed"] is True
    }
    require(
        source_passed == all_maps,
        "source producer semantics require a complete 28-map source root",
    )
    stage_passed = {
        stage: {
            str(row["map"])
            for row in stage_reports[stage]["maps"]
            if row["passed"] is True
        }
        for stage in ("compile", "materialization", "claims")
    }
    expected_by_stage = {
        "source": {
            f"{map_id}{suffix}"
            for map_id in all_maps for suffix in STAGE_SUFFIXES["source"]
        },
        # Sparse compilation retains every immutable source artifact and
        # publishes a BSP only for compile-passing members.
        "compile": {
            f"{map_id}{suffix}"
            for map_id in all_maps for suffix in STAGE_SUFFIXES["source"]
        } | {
            f"{map_id}.bsp" for map_id in stage_passed["compile"]
        },
        "materialization": {
            f"{map_id}{suffix}"
            for map_id in stage_passed["materialization"]
            for suffix in STAGE_SUFFIXES["materialized"]
        },
        "claims": {
            f"{map_id}{suffix}"
            for map_id in stage_passed["claims"]
            for suffix in STAGE_SUFFIXES["claims"]
        },
    }
    for stage, expected in expected_by_stage.items():
        exact_flat_files(roots[stage], expected, f"{stage} root")
        records[stage] = {
            name: file_record(roots[stage] / name) for name in sorted(expected)
        }
    atlas_passes = {
        str(row["map"]) for row in stage_reports["atlas-build"]["maps"] if row["passed"]
    }
    expected_analysis = {
        f"{name}{suffix}" for name in atlas_passes for suffix in STAGE_SUFFIXES["analysis"]
    }
    exact_flat_files(roots["atlas-build"], expected_analysis, "atlas-build root")
    records["atlas-build"] = {
        name: file_record(roots["atlas-build"] / name) for name in sorted(expected_analysis)
    }
    expected_promotion = {
        f"{row['ordinal']:03d}-{row['map']}.json" for row in declaration["maps"]
    }
    exact_flat_files(roots["generated-promotion"], expected_promotion,
                     "generated-promotion evidence root")
    records["generated-promotion"] = {
        name: file_record(roots["generated-promotion"] / name)
        for name in sorted(expected_promotion)
    }
    return {"exact_roots": records, "map_count": len(declaration["maps"])}


def _cold_and_resource_evidence(
    stage_reports: Mapping[str, Mapping[str, Any]], analysis_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_rows = stage_reports["source"]["maps"]
    source = [{
        "map": row["map"], "passed": row["passed"],
        "cold_bytes_identical": row["criteria"].get("cold-bytes-identical"),
        "evidence_sha256": row["evidence_sha256"],
    } for row in source_rows]
    require(all(not row["passed"] or row["cold_bytes_identical"] is True for row in source),
            "a passing source row lacks byte-identical cold rebuild evidence")
    cold_maps: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    for row in stage_reports["atlas-build"]["maps"]:
        if not row["passed"]:
            continue
        name = str(row["map"])
        manifest, raw = load_canonical(analysis_root / f"{name}.analysis.manifest.json")
        try:
            proof = manifest["performance"]["full_cold_rebuild"]
            artifact = proof["artifact_sha256"]
            rebuilt = proof["cold_artifact_sha256"]
            semantic = proof["artifact_semantic_sha256"]
            rebuilt_semantic = proof["cold_artifact_semantic_sha256"]
            elapsed = proof["elapsed_milliseconds"]
            peak = proof["sampled_process_tree_peak_rss_bytes"]
            peak_limit = proof["peak_rss_limit_bytes"]
            timeout_limit = proof["timeout_limit_milliseconds"]
        except (KeyError, TypeError) as error:
            raise QualificationStageError(f"{name} full-cold proof is incomplete") from error
        require(artifact == rebuilt and semantic == rebuilt_semantic,
                f"{name} cold-rebuild digests differ")
        require(row["criteria"].get("deterministic-cold-rebuild") is True,
                f"{name} stage row lacks deterministic cold criterion")
        require(isinstance(elapsed, int) and not isinstance(elapsed, bool)
                and 0 < elapsed <= MAX_FULL_COLD_MILLISECONDS,
                f"{name} full-cold elapsed time exceeds limit")
        require(timeout_limit == MAX_FULL_COLD_MILLISECONDS,
                f"{name} full-cold timeout limit differs")
        require(peak_limit == MAX_BUILD_RSS_BYTES and isinstance(peak, int)
                and not isinstance(peak, bool) and 0 < peak <= MAX_BUILD_RSS_BYTES,
                f"{name} full-cold resource bound failed")
        manifest_record = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
        cold_maps.append({
            "map": name, "manifest": manifest_record,
            "artifact_sha256": artifact, "semantic_sha256": semantic,
        })
        resources.append({
            "map": name, "manifest": manifest_record,
            "elapsed_milliseconds": elapsed,
            "timeout_limit_milliseconds": timeout_limit,
            "sampled_process_tree_peak_rss_bytes": peak,
            "peak_rss_limit_bytes": peak_limit,
        })
    require(len(cold_maps) >= REQUIRED_END_TO_END_PASSES,
            "fewer than 20 Atlas maps provide cold/resource evidence")
    return ({"source": source, "atlas": cold_maps}, {"atlas": resources})


def _syntax_evidence(path: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    report, raw = load_canonical(path)
    require(report.get("schema") == "q2-python-syntax-floor-v1" and report.get("passed") is True,
            "Python 3.10 syntax-floor report is not green")
    require(report.get("failures") == [] and isinstance(report.get("file_count"), int)
            and report["file_count"] > 0, "syntax report is incomplete")
    interpreter = report.get("interpreter")
    require(isinstance(interpreter, Mapping), "syntax interpreter record is malformed")
    require(interpreter.get("implementation") == "cpython"
            and interpreter.get("version") == SYNTAX_PYTHON_VERSION
            and interpreter.get("sha256") == SYNTAX_PYTHON_SHA256,
            "syntax report was not produced by the WSL CPython 3.10.12 authority")
    require(runtime.get("implementation") == "cpython"
            and runtime.get("version") == PINNED_PYTHON_VERSION
            and runtime.get("sha256") == PINNED_PYTHON_SHA256,
            "infrastructure producer is not the pinned CPython 3.11.4 runtime")
    return {"report": {"bytes": len(raw), "sha256": sha256_bytes(raw)},
            "syntax_interpreter": dict(interpreter),
            "execution_runtime": dict(runtime), "file_count": report["file_count"],
            "files_sha256": report.get("files_sha256")}


def produce_infrastructure_report(
    *, declaration_path: Path, stage_report_paths: Mapping[str, Path],
    source_root: Path, compiled_root: Path, materialized_root: Path,
    claims_root: Path, analysis_root: Path, promotion_evidence_root: Path,
    syntax_report_path: Path, evidence_root: Path, report_path: Path,
    repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = current_implementation,
    timeout_probe: Callable[[], dict[str, Any]] = _timeout_probe,
    runtime_provider: Callable[[], Mapping[str, Any]] = pinned_runtime_record,
) -> dict[str, Any]:
    require(set(stage_report_paths) == set(STAGES), "all seven stage report paths are required")
    paths = {
        label: qualification_path(path, label)
        for label, path in {
            "declaration": declaration_path, "source root": source_root,
            "compiled root": compiled_root, "materialized root": materialized_root,
            "claims root": claims_root, "analysis root": analysis_root,
            "promotion evidence root": promotion_evidence_root,
            "syntax report": syntax_report_path, "evidence root": evidence_root,
            "report": report_path,
        }.items()
    }
    stage_paths = {
        stage: qualification_path(path, f"{stage} report")
        for stage, path in stage_report_paths.items()
    }
    fresh_outputs({"infrastructure evidence": paths["evidence root"],
                   "infrastructure report": paths["report"]})
    implementation = implementation_provider(repo_root)
    runtime = dict(runtime_provider())
    declaration, _raw, declaration_sha256 = validate_declaration(
        paths["declaration"], implementation
    )
    reports: dict[str, Mapping[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    pass_sets: dict[str, set[str]] = {}
    previous: str | None = None
    for stage in STAGES:
        report, _stage_raw, digest, passed = validate_stage_report(
            stage_paths[stage], stage, declaration, declaration_sha256,
            implementation, expected_input_sha256=previous,
        )
        reports[stage] = report
        report_hashes[stage] = digest
        pass_sets[stage] = passed
        previous = digest
    for earlier, later in zip(STAGES, STAGES[1:]):
        require(pass_sets[later].issubset(pass_sets[earlier]),
                f"{later} relabels a map which did not pass {earlier}")
    require(len(pass_sets["generated-promotion"]) >= REQUIRED_END_TO_END_PASSES,
            "qualification has fewer than 20 promotion passes")

    work = Path(tempfile.mkdtemp(prefix=".b2q-infrastructure-",
                                dir=paths["evidence root"].parent))
    stage = work / "evidence"
    stage.mkdir()
    published = False
    try:
        membership = _membership_evidence(declaration, reports, {
            "source": paths["source root"], "compile": paths["compiled root"],
            "materialization": paths["materialized root"], "claims": paths["claims root"],
            "atlas-build": paths["analysis root"],
            "generated-promotion": paths["promotion evidence root"],
        })
        cold, resources = _cold_and_resource_evidence(reports, paths["analysis root"])
        syntax = _syntax_evidence(paths["syntax report"], runtime)
        exclusive = _exclusive_probe(stage)
        timeout = timeout_probe()
        require(timeout.get("timed_out") is True and timeout.get("process_group_killed") is True,
                "timeout probe did not fail closed and kill its process group")
        bodies = {
            "deterministic-cold-rebuild": cold,
            "exact-stage-membership": membership,
            "exclusive-create": exclusive,
            "python310-syntax-floor": syntax,
            "resource-bounds": resources,
            "timeout-fail-closed": timeout,
        }
        checks: list[dict[str, Any]] = []
        for check_id in CHECK_IDS:
            evidence = {
                "schema": "q2-b2-qualification-infrastructure-evidence-v2",
                "qualification_id": declaration["qualification_id"],
                "id": check_id, "stage_report_sha256s": report_hashes,
                "evidence": bodies[check_id], "passed": True,
            }
            payload = canonical_bytes(evidence)
            exclusive_write(stage / f"{check_id}.json", payload)
            checks.append({
                "id": check_id, "criteria": {"actual-evidence-retained": True,
                                               "qualification-bound": True},
                "evidence_sha256": sha256_bytes(payload), "failures": [], "passed": True,
            })
        exact_flat_files(stage, {f"{check_id}.json" for check_id in CHECK_IDS},
                         "infrastructure evidence")
        report = {
            "schema": INFRASTRUCTURE_SCHEMA,
            "qualification_id": declaration["qualification_id"],
            "mode": "qualification", "non_admissible": True, "retryable": True,
            "final_cohort_authorized": False,
            "declaration_sha256": declaration_sha256,
            "implementation": implementation,
            "toolchain_authority_sha256": declaration[
                "toolchain_authority_sha256"
            ],
            "stage_report_sha256s": report_hashes,
            "checks": checks, "pass_count": len(checks), "failures": [],
        }
        rename_noreplace(stage, paths["evidence root"])
        published = True
        exclusive_write(paths["report"], canonical_bytes(report))
        return report
    except Exception:
        if published and paths["evidence root"].is_dir():
            shutil.rmtree(paths["evidence root"], ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    for stage in STAGES:
        parser.add_argument(f"--{stage}-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--claims-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--promotion-evidence-root", type=Path, required=True)
    parser.add_argument("--syntax-report", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    stage_paths = {
        stage: getattr(args, stage.replace("-", "_") + "_report") for stage in STAGES
    }
    try:
        report = produce_infrastructure_report(
            declaration_path=args.declaration, stage_report_paths=stage_paths,
            source_root=args.source_root, compiled_root=args.compiled_root,
            materialized_root=args.materialized_root, claims_root=args.claims_root,
            analysis_root=args.analysis_root,
            promotion_evidence_root=args.promotion_evidence_root,
            syntax_report_path=args.syntax_report, evidence_root=args.evidence_root,
            report_path=args.report, repo_root=args.repo_root,
        )
    except (QualificationStageError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
