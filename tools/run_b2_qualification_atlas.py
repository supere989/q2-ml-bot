#!/usr/bin/env python3
"""Build qualification-native full Atlas evidence for a disposable population."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
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

from tools.b2_qualification_stage_support import (  # noqa: E402
    QualificationStageError,
    canonical_bytes,
    current_implementation,
    evidence_sha256,
    exact_flat_files,
    exclusive_write,
    file_record,
    fresh_outputs,
    membership_records,
    qualification_path,
    rename_noreplace,
    require,
    sha256_bytes,
    stage_report,
    validate_declaration,
    validate_stage_report,
)
from tools.run_generated_atlas_campaign import (  # noqa: E402
    BUILD_SUMMARY_NAME,
    BuildProcessResult,
    GeneratedAtlasCampaignError,
    _inspect_map_build,
    _snapshot_repository,
)
from tools.run_generator_cohort import STAGE_SUFFIXES  # noqa: E402


DEFAULT_CLIENT = Path("/home/raymondj/multires-worktrees/integration/q2-ml-client")
DEFAULT_LITHIUM = Path("/home/raymondj/multires-worktrees/integration/q2-lithium-3zb2")
DEFAULT_ATTESTATION = Path(
    "/home/raymondj/multires-artifacts/atlas-v1/B1/hook-parity-pullspeed-1700.json"
)
PINNED_PYTHON = Path("/home/raymond/miniconda3/bin/python3.11")
PINNED_PYTHON_SHA256 = "b25abf001748dc7ebb4b25013b2572d4e6913246b4c3b8e8b726b3da45494ff4"
DEFAULT_JOBS = min(4, os.cpu_count() or 1)
MAX_JOBS = 8
DEFAULT_TIMEOUT_SECONDS = 600.0
CLAIMS_CRITERIA = {
    "prior-stage-passed", "immutable-claims", "claims-membership", "input-stability",
}


Builder = Callable[[Mapping[str, Any], Path, Path, Path], BuildProcessResult]


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in [path, *path.rglob("*")]:
        try:
            item.chmod(0o700 if item.is_dir() else 0o600)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def _copy_exact_flat(source: Path, destination: Path, expected: set[str]) -> None:
    exact_flat_files(source, expected, "claims")
    destination.mkdir(mode=0o700)
    for name in sorted(expected):
        source_path = source / name
        before = file_record(source_path)
        target = destination / name
        with source_path.open("rb") as input_stream, target.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        require(file_record(source_path) == before == file_record(target),
                f"claims input changed while snapshotting {name}")
    exact_flat_files(destination, expected, "claims snapshot")


def _default_builder(
    *, client_root: Path, lithium_root: Path, hook_attestation: Path,
    fall_oracle: Path | None, packer: Path | None, verifier: Path | None,
    timeout_seconds: float, python_executable: Path,
) -> Builder:
    def build(
        declared: Mapping[str, Any], claims_root: Path, output_root: Path,
        execution_root: Path,
    ) -> BuildProcessResult:
        name = str(declared["map"])
        command = [
            str(python_executable), "-B",
            str(execution_root / "tools/build_map_atlas.py"),
            "--bsp", str(claims_root / f"{name}.bsp"), "--map-id", name,
            "--generator-claims", str(claims_root / f"{name}.generator-claims.json"),
            "--output", str(output_root), "--client-root", str(client_root),
            "--lithium-root", str(lithium_root), "--hook-attestation", str(hook_attestation),
        ]
        for option, value in (("--fall-oracle", fall_oracle), ("--packer", packer),
                              ("--verifier", verifier)):
            if value is not None:
                command.extend((option, str(value)))
        try:
            process = subprocess.Popen(
                command, cwd=execution_root, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                return BuildProcessResult(-signal.SIGKILL, stdout, stderr + b"\nqualification timeout\n")
        except OSError as error:
            return BuildProcessResult(127, b"", f"{type(error).__name__}: {error}\n".encode())
        return BuildProcessResult(process.returncode, stdout, stderr)
    return build


def _claims_rows(
    declaration: Mapping[str, Any], report: Mapping[str, Any], claims_root: Path,
) -> dict[str, Mapping[str, Any]]:
    expected = {
        f"{row['map']}{suffix}"
        for row in declaration["maps"] for suffix in STAGE_SUFFIXES["claims"]
    }
    exact_flat_files(claims_root, expected, "claims")
    result: dict[str, Mapping[str, Any]] = {}
    for item in report["maps"]:
        require(set(item["criteria"]) == CLAIMS_CRITERIA,
                f"claims criteria differ for {item['map']}")
        files = membership_records(claims_root, item["map"], STAGE_SUFFIXES["claims"])
        expected_digest = evidence_sha256(
            item["map"], files,
            prior_stage_passed=item["criteria"]["prior-stage-passed"],
        )
        require(item["evidence_sha256"] == expected_digest,
                f"claims evidence does not match bytes for {item['map']}")
        result[str(item["map"])] = item
    return result


def build_qualification_atlas(
    *, declaration_path: Path, claims_report_path: Path, claims_root: Path,
    analysis_root: Path, evidence_root: Path, report_path: Path,
    repo_root: Path = ROOT, client_root: Path = DEFAULT_CLIENT,
    lithium_root: Path = DEFAULT_LITHIUM, hook_attestation: Path = DEFAULT_ATTESTATION,
    fall_oracle: Path | None = None, packer: Path | None = None,
    verifier: Path | None = None, jobs: int = DEFAULT_JOBS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    python_executable: Path = PINNED_PYTHON,
    implementation_provider: Callable[[Path], dict[str, Any]] = current_implementation,
    snapshotter: Callable[[Path, Path, Mapping[str, Any]], None] = _snapshot_repository,
    builder: Builder | None = None,
    inspector: Callable[..., Any] = _inspect_map_build,
) -> dict[str, Any]:
    paths = {
        label: qualification_path(path, label)
        for label, path in {
            "declaration": declaration_path, "claims report": claims_report_path,
            "claims root": claims_root, "analysis root": analysis_root,
            "evidence root": evidence_root, "report": report_path,
        }.items()
    }
    require(isinstance(jobs, int) and not isinstance(jobs, bool) and 1 <= jobs <= MAX_JOBS,
            f"jobs must be in [1, {MAX_JOBS}]")
    require(isinstance(timeout_seconds, (int, float)) and not isinstance(timeout_seconds, bool)
            and math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 3600,
            "timeout must be finite and in (0, 3600]")
    fresh_outputs({
        "analysis": paths["analysis root"], "evidence": paths["evidence root"],
        "report": paths["report"],
    })
    implementation = implementation_provider(repo_root)
    declaration, _raw, declaration_sha256 = validate_declaration(
        paths["declaration"], implementation
    )
    claims_report, _claims_raw, claims_sha256, claims_passes = validate_stage_report(
        paths["claims report"], "claims", declaration, declaration_sha256, implementation
    )
    claim_rows = _claims_rows(declaration, claims_report, paths["claims root"])

    work = Path(tempfile.mkdtemp(prefix=".b2q-atlas-", dir=paths["analysis root"].parent))
    published: list[Path] = []
    try:
        claims_snapshot = work / "claims"
        execution_root = work / "execution"
        analysis_stage = work / "analysis"
        evidence_stage = work / "evidence"
        analysis_stage.mkdir()
        evidence_stage.mkdir()
        expected_claims = {
            f"{row['map']}{suffix}" for row in declaration["maps"]
            for suffix in STAGE_SUFFIXES["claims"]
        }
        _copy_exact_flat(paths["claims root"], claims_snapshot, expected_claims)
        snapshotter(repo_root, execution_root, implementation)
        real_builder = builder or _default_builder(
            client_root=client_root, lithium_root=lithium_root,
            hook_attestation=hook_attestation, fall_oracle=fall_oracle,
            packer=packer, verifier=verifier, timeout_seconds=float(timeout_seconds),
            python_executable=python_executable,
        )
        runtime_record: Mapping[str, Any]
        if builder is None:
            runtime_record = file_record(python_executable)
            require(runtime_record["sha256"] == PINNED_PYTHON_SHA256,
                    "Atlas runtime is not the pinned WSL CPython 3.11.4 executable")
        else:
            runtime_record = {"injected_test_builder": True}

        def one(declared: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, bytes]]:
            ordinal = int(declared["ordinal"])
            name = str(declared["map"])
            prior = claim_rows[name]
            output = work / f"map-{ordinal:03d}"
            output.mkdir()
            result = BuildProcessResult(0, b"", b"")
            artifacts: dict[str, Any] = {}
            analysis_binding: dict[str, Any] | None = None
            summary: dict[str, Any] | None = None
            failures: list[str] = []
            invoked = False
            if name in claims_passes:
                invoked = True
                try:
                    result = real_builder(declared, claims_snapshot, output, execution_root)
                    summary, _summary_payload, artifacts, analysis_binding = inspector(
                        output, declared, claims_snapshot, result, implementation
                    )
                except (GeneratedAtlasCampaignError, QualificationStageError, OSError) as error:
                    failures.append(str(error))
            else:
                failures.append("claims stage did not pass")
            passed = invoked and not failures and result.returncode == 0 and analysis_binding is not None
            criteria = {
                "prior-claims-passed": name in claims_passes,
                "real-atlas-build": invoked and result.returncode == 0,
                "full-atlas": analysis_binding is not None,
                "deterministic-cold-rebuild": bool(
                    analysis_binding and analysis_binding.get("deterministic_rebuild") is True
                ),
                "exact-analysis-membership": passed,
            }
            if not all(criteria.values()) and not failures:
                failures.append("Atlas qualification criteria failed")
            evidence = {
                "schema": "q2-b2-qualification-atlas-map-evidence-v1",
                "ordinal": ordinal, "map": name,
                "claims_evidence_sha256": prior["evidence_sha256"],
                "runtime": dict(runtime_record),
                "invoked": invoked, "exit_code": result.returncode,
                "stdout": {"bytes": len(result.stdout), "sha256": sha256_bytes(result.stdout)},
                "stderr": {"bytes": len(result.stderr), "sha256": sha256_bytes(result.stderr)},
                "artifacts": artifacts, "analysis_binding": analysis_binding,
                "build_summary": summary, "criteria": criteria,
                "failures": failures, "passed": all(criteria.values()) and not failures,
            }
            payloads = {
                "evidence": canonical_bytes(evidence), "stdout": result.stdout, "stderr": result.stderr,
            }
            if evidence["passed"]:
                for suffix in STAGE_SUFFIXES["analysis"]:
                    source = output / f"{name}{suffix}"
                    target = analysis_stage / source.name
                    shutil.copyfile(source, target)
                    require(file_record(source) == file_record(target),
                            f"analysis copy differs for {source.name}")
            return ordinal, {
                "ordinal": ordinal, "map": name, "criteria": criteria,
                "evidence_sha256": sha256_bytes(payloads["evidence"]),
                "failures": failures, "passed": evidence["passed"],
            }, payloads

        results: dict[int, tuple[dict[str, Any], dict[str, bytes]]] = {}
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(one, row): row for row in declaration["maps"]}
            for future in as_completed(futures):
                ordinal, row, payloads = future.result()
                results[ordinal] = (row, payloads)
        rows: list[dict[str, Any]] = []
        for ordinal in range(len(declaration["maps"])):
            row, payloads = results[ordinal]
            rows.append(row)
            stem = f"{ordinal:03d}-{row['map']}"
            exclusive_write(evidence_stage / f"{stem}.json", payloads["evidence"])
            exclusive_write(evidence_stage / f"{stem}.stdout.log", payloads["stdout"])
            exclusive_write(evidence_stage / f"{stem}.stderr.log", payloads["stderr"])
        expected_analysis = {
            f"{row['map']}{suffix}" for row in rows if row["passed"]
            for suffix in STAGE_SUFFIXES["analysis"]
        }
        exact_flat_files(analysis_stage, expected_analysis, "qualification analysis stage")
        exact_flat_files(evidence_stage, {
            f"{row['ordinal']:03d}-{row['map']}{suffix}" for row in rows
            for suffix in (".json", ".stdout.log", ".stderr.log")
        }, "Atlas evidence stage")
        report = stage_report(
            declaration=declaration, declaration_sha256=declaration_sha256,
            implementation=implementation, stage="atlas-build", input_sha256=claims_sha256,
            checks={"full-atlas": True, "deterministic-cold-rebuild": True,
                    "bounded-parallel-workers": True, "timeout-fail-closed": True},
            rows=rows,
        )
        report_payload = canonical_bytes(report)
        rename_noreplace(evidence_stage, paths["evidence root"])
        published.append(paths["evidence root"])
        rename_noreplace(analysis_stage, paths["analysis root"])
        published.append(paths["analysis root"])
        exclusive_write(paths["report"], report_payload)
        return report
    except Exception:
        for output in reversed(published):
            if output.is_dir() and not output.is_symlink():
                _remove_tree(output)
        raise
    finally:
        _remove_tree(work)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--claims-report", type=Path, required=True)
    parser.add_argument("--claims-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--client-root", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--lithium-root", type=Path, default=DEFAULT_LITHIUM)
    parser.add_argument("--hook-attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--fall-oracle", type=Path)
    parser.add_argument("--packer", type=Path)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--python", type=Path, default=PINNED_PYTHON)
    args = parser.parse_args(argv)
    try:
        report = build_qualification_atlas(
            declaration_path=args.declaration, claims_report_path=args.claims_report,
            claims_root=args.claims_root, analysis_root=args.analysis_root,
            evidence_root=args.evidence_root, report_path=args.report,
            repo_root=args.repo_root, client_root=args.client_root,
            lithium_root=args.lithium_root, hook_attestation=args.hook_attestation,
            fall_oracle=args.fall_oracle, packer=args.packer, verifier=args.verifier,
            jobs=args.jobs, timeout_seconds=args.timeout_seconds,
            python_executable=args.python,
        )
    except (QualificationStageError, GeneratedAtlasCampaignError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
