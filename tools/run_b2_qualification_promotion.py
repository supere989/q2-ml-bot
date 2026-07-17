#!/usr/bin/env python3
"""Run independent generated-map promotion for disposable B2 qualification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import tempfile
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b2_qualification_stage_support import (  # noqa: E402
    QualificationStageError,
    canonical_bytes,
    current_implementation,
    exact_flat_files,
    exclusive_write,
    file_record,
    fresh_outputs,
    load_canonical,
    qualification_path,
    pinned_runtime_record,
    rename_noreplace,
    require,
    sha256_bytes,
    stage_report,
    validate_declaration,
    validate_stage_report,
)
from tools.generator_claim_validator import (  # noqa: E402
    ClaimValidationError,
    validate_generated_map,
    validate_report as validate_promotion_report,
)
from tools.run_generator_cohort import STAGE_SUFFIXES  # noqa: E402


DEFAULT_JOBS = min(4, os.cpu_count() or 1)
MAX_JOBS = 8
ATLAS_CRITERIA = {
    "prior-claims-passed", "real-atlas-build", "full-atlas",
    "deterministic-cold-rebuild", "exact-analysis-membership",
}


Validator = Callable[..., dict[str, Any]]


def _validate_atlas_artifacts(
    declaration: Mapping[str, Any], report: Mapping[str, Any],
    analysis_root: Path, evidence_root: Path,
) -> None:
    passed = {str(row["map"]) for row in report["maps"] if row["passed"]}
    exact_flat_files(analysis_root, {
        f"{name}{suffix}" for name in passed for suffix in STAGE_SUFFIXES["analysis"]
    }, "qualification analysis")
    exact_flat_files(evidence_root, {
        f"{row['ordinal']:03d}-{row['map']}{suffix}" for row in declaration["maps"]
        for suffix in (".json", ".stdout.log", ".stderr.log")
    }, "Atlas evidence")
    for row in report["maps"]:
        require(set(row["criteria"]) == ATLAS_CRITERIA,
                f"Atlas criteria differ for {row['map']}")
        stem = f"{row['ordinal']:03d}-{row['map']}"
        evidence, raw = load_canonical(evidence_root / f"{stem}.json")
        require(sha256_bytes(raw) == row["evidence_sha256"],
                f"Atlas evidence hash differs for {row['map']}")
        require(evidence.get("map") == row["map"] and evidence.get("passed") is row["passed"],
                f"Atlas evidence disposition differs for {row['map']}")
        for stream in ("stdout", "stderr"):
            require(file_record(evidence_root / f"{stem}.{stream}.log") == evidence[stream],
                    f"Atlas {stream} evidence differs for {row['map']}")
        if row["passed"]:
            artifacts = evidence.get("artifacts")
            require(isinstance(artifacts, Mapping) and set(artifacts) == set(STAGE_SUFFIXES["analysis"]),
                    f"Atlas artifact evidence is incomplete for {row['map']}")
            for suffix in STAGE_SUFFIXES["analysis"]:
                require(file_record(analysis_root / f"{row['map']}{suffix}") == artifacts[suffix],
                        f"Atlas artifact bytes differ for {row['map']}{suffix}")


def validate_promotion_evidence(
    declaration: Mapping[str, Any], report: Mapping[str, Any],
    claims_root: Path, analysis_root: Path, evidence_root: Path,
) -> None:
    """Replay every promotion row against its retained validator report."""

    exact_flat_files(claims_root, {
        f"{row['map']}{suffix}" for row in declaration["maps"]
        for suffix in STAGE_SUFFIXES["claims"]
    }, "qualification claims")
    exact_flat_files(evidence_root, {
        f"{row['ordinal']:03d}-{row['map']}.json" for row in declaration["maps"]
    }, "promotion evidence")
    for row in report["maps"]:
        path = evidence_root / f"{row['ordinal']:03d}-{row['map']}.json"
        evidence, raw = load_canonical(path)
        require(sha256_bytes(raw) == row["evidence_sha256"],
                f"promotion evidence hash differs for {row['map']}")
        require(set(evidence) == {
            "schema", "ordinal", "map", "atlas_evidence_sha256", "runtime",
            "eligible", "validation", "criteria", "failures", "passed",
        }, f"promotion evidence keys differ for {row['map']}")
        require(evidence["schema"] == "q2-b2-qualification-promotion-map-evidence-v1"
                and evidence["ordinal"] == row["ordinal"]
                and evidence["map"] == row["map"],
                f"promotion evidence identity differs for {row['map']}")
        require(evidence["criteria"] == row["criteria"]
                and evidence["failures"] == row["failures"]
                and evidence["passed"] is row["passed"],
                f"promotion evidence disposition differs for {row['map']}")
        validation = evidence["validation"]
        if row["passed"]:
            require(isinstance(validation, Mapping),
                    f"passing promotion evidence lacks validator report for {row['map']}")
            try:
                validate_promotion_report(validation)
            except ClaimValidationError as error:
                raise QualificationStageError(
                    f"promotion validator report rejected for {row['map']}: {error}"
                ) from error
            identities = validation["identities"]
            require(validation["passed"] is True
                    and identities["analysis_sha256"] == file_record(
                        analysis_root / f"{row['map']}.analysis.manifest.json"
                    )["sha256"]
                    and identities["bsp_sha256"] == file_record(
                        claims_root / f"{row['map']}.bsp"
                    )["sha256"],
                    f"promotion validator identities differ for {row['map']}")
        else:
            require(validation is None or isinstance(validation, Mapping),
                    f"failed promotion validation has malformed evidence for {row['map']}")


def run_qualification_promotion(
    *, declaration_path: Path, atlas_report_path: Path, claims_root: Path,
    analysis_root: Path, atlas_evidence_root: Path, b1_gate_path: Path,
    evidence_root: Path, report_path: Path, repo_root: Path = ROOT,
    jobs: int = DEFAULT_JOBS,
    implementation_provider: Callable[[Path], dict[str, Any]] = current_implementation,
    validator: Validator = validate_generated_map,
    runtime_provider: Callable[[], Mapping[str, Any]] = pinned_runtime_record,
) -> dict[str, Any]:
    paths = {
        label: qualification_path(path, label)
        for label, path in {
            "declaration": declaration_path, "Atlas report": atlas_report_path,
            "claims root": claims_root, "analysis root": analysis_root,
            "Atlas evidence root": atlas_evidence_root, "B1 gate": b1_gate_path,
            "promotion evidence root": evidence_root, "report": report_path,
        }.items()
    }
    require(isinstance(jobs, int) and not isinstance(jobs, bool) and 1 <= jobs <= MAX_JOBS,
            f"jobs must be in [1, {MAX_JOBS}]")
    fresh_outputs({"promotion evidence": paths["promotion evidence root"],
                   "report": paths["report"]})
    implementation = implementation_provider(repo_root)
    runtime = dict(runtime_provider())
    declaration, _raw, declaration_sha256 = validate_declaration(
        paths["declaration"], implementation
    )
    atlas_report, _atlas_raw, atlas_sha256, atlas_passes = validate_stage_report(
        paths["Atlas report"], "atlas-build", declaration, declaration_sha256, implementation
    )
    exact_flat_files(paths["claims root"], {
        f"{row['map']}{suffix}" for row in declaration["maps"]
        for suffix in STAGE_SUFFIXES["claims"]
    }, "qualification claims")
    _validate_atlas_artifacts(
        declaration, atlas_report, paths["analysis root"], paths["Atlas evidence root"]
    )
    repository_b1 = repo_root / "docs/multires/B1-GATE.json"
    require(file_record(paths["B1 gate"]) == file_record(repository_b1),
            "promotion B1 gate is not the current repository authority")

    work = Path(tempfile.mkdtemp(prefix=".b2q-promotion-",
                                dir=paths["promotion evidence root"].parent))
    stage = work / "evidence"
    stage.mkdir()
    published = False
    try:
        def one(declared: Mapping[str, Any]) -> tuple[int, dict[str, Any], bytes]:
            ordinal = int(declared["ordinal"])
            name = str(declared["map"])
            validation: dict[str, Any] | None = None
            failures: list[str] = []
            if name in atlas_passes:
                try:
                    validation = validator(
                        paths["claims root"] / f"{name}.map",
                        paths["analysis root"] / f"{name}.analysis.manifest.json",
                        b1_gate_path=paths["B1 gate"],
                    )
                    validate_promotion_report(validation)
                    failures.extend(str(value) for value in validation["failures"])
                except (ClaimValidationError, OSError, KeyError, TypeError, ValueError) as error:
                    failures.append(str(error))
            else:
                failures.append("Atlas stage did not pass")
            independent_passed = bool(validation and validation.get("passed") is True)
            identities_bound = False
            if independent_passed:
                identities = validation.get("identities", {})
                identities_bound = (
                    identities.get("analysis_sha256") ==
                    file_record(paths["analysis root"] / f"{name}.analysis.manifest.json")["sha256"]
                    and identities.get("bsp_sha256") ==
                    file_record(paths["claims root"] / f"{name}.bsp")["sha256"]
                )
                if not identities_bound:
                    failures.append("independent validator identities differ from input bytes")
            criteria = {
                "prior-atlas-passed": name in atlas_passes,
                "independent-promotion-validation": independent_passed,
                "input-identities-bound": identities_bound,
            }
            if not all(criteria.values()) and not failures:
                failures.append("promotion qualification criteria failed")
            evidence = {
                "schema": "q2-b2-qualification-promotion-map-evidence-v1",
                "ordinal": ordinal, "map": name,
                "atlas_evidence_sha256": atlas_report["maps"][ordinal]["evidence_sha256"],
                "runtime": runtime,
                "eligible": name in atlas_passes, "validation": validation,
                "criteria": criteria, "failures": failures,
                "passed": all(criteria.values()) and not failures,
            }
            payload = canonical_bytes(evidence)
            row = {
                "ordinal": ordinal, "map": name, "criteria": criteria,
                "evidence_sha256": sha256_bytes(payload), "failures": failures,
                "passed": evidence["passed"],
            }
            return ordinal, row, payload

        results: dict[int, tuple[dict[str, Any], bytes]] = {}
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(one, row): row for row in declaration["maps"]}
            for future in as_completed(futures):
                ordinal, row, payload = future.result()
                results[ordinal] = (row, payload)
        rows: list[dict[str, Any]] = []
        for ordinal in range(len(declaration["maps"])):
            row, payload = results[ordinal]
            rows.append(row)
            exclusive_write(stage / f"{ordinal:03d}-{row['map']}.json", payload)
        exact_flat_files(stage, {
            f"{row['ordinal']:03d}-{row['map']}.json" for row in declaration["maps"]
        }, "promotion evidence")
        report = stage_report(
            declaration=declaration, declaration_sha256=declaration_sha256,
            implementation=implementation, stage="generated-promotion",
            input_sha256=atlas_sha256,
            checks={"independent-promotion-validation": True,
                    "exact-evidence-membership": True,
                    "bounded-parallel-workers": True},
            rows=rows,
        )
        rename_noreplace(stage, paths["promotion evidence root"])
        published = True
        exclusive_write(paths["report"], canonical_bytes(report))
        return report
    except Exception:
        if published and paths["promotion evidence root"].is_dir():
            shutil.rmtree(paths["promotion evidence root"], ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--atlas-report", type=Path, required=True)
    parser.add_argument("--claims-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--atlas-evidence-root", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    args = parser.parse_args(argv)
    try:
        report = run_qualification_promotion(
            declaration_path=args.declaration, atlas_report_path=args.atlas_report,
            claims_root=args.claims_root, analysis_root=args.analysis_root,
            atlas_evidence_root=args.atlas_evidence_root, b1_gate_path=args.b1_gate,
            evidence_root=args.evidence_root, report_path=args.report,
            repo_root=args.repo_root, jobs=args.jobs,
        )
    except (QualificationStageError, ClaimValidationError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
