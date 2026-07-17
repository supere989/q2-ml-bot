#!/usr/bin/env python3
"""Produce qualification-native materialization and immutable claims stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    admit_hook_parity_attestation,
    load_b1_authority_gate,
)
from tools.assemble_b2_qualification import (  # noqa: E402
    REQUIRED_STAGE_CHECKS,
    STAGE_SCHEMA,
)
from tools.generator_claim_validator import (  # noqa: E402
    ClaimValidationError,
    build_generator_claims,
)
from tools.materialize_generated_cohort import (  # noqa: E402
    MaterializeCohortError,
    _run_process_group,
    _validate_result,
)
from tools.run_b2_qualification_compile import (  # noqa: E402
    COMPILED_SUFFIXES,
    QualificationCompileError,
    _current_implementation,
    _file_record,
    _flat_records,
    _load_json,
    _overlap,
    _qualification_path,
    _rename_noreplace,
    _validate_declaration,
    _validate_implementation,
)
from tools.run_b2_qualification_compiled_cm import (  # noqa: E402
    QualificationCompiledCmError,
    validate_published_qualification_compiled_cm,
)
from tools.run_generator_cohort import canonical_bytes  # noqa: E402


MATERIALIZED_SUFFIXES = (*COMPILED_SUFFIXES, ".hook-materialization.json")
CLAIMS_SUFFIXES = (*MATERIALIZED_SUFFIXES, ".generator-claims.json")
MATERIALIZED_FILE_COUNT = 196
CLAIMS_FILE_COUNT = 224
DEFAULT_TIMEOUT_SECONDS = 900
PINNED_PYTHON = Path("/home/raymond/miniconda3/bin/python3.11")
PINNED_PYTHON_SHA256 = "b25abf001748dc7ebb4b25013b2572d4e6913246b4c3b8e8b726b3da45494ff4"
PINNED_PYTHON_VERSION = [3, 11, 4]
PINNED_ZSTANDARD_VERSION = "0.19.0"
PINNED_ZSTANDARD_INIT_SHA256 = "8a65cd4ab44112e1433a097daee7ce8600047995f3289f13d758bb001c06a553"
PINNED_ZSTANDARD_BACKEND_SHA256 = "40ece7fa91097e53ee4785cef01baae3f220f8dc891e20d94d4e07a1d77c9120"
RUNTIME_PREFLIGHT_TIMEOUT_SECONDS = 300


class QualificationPostcompileError(RuntimeError):
    """A qualification postcompile stage failed without partial publication."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stage_evidence_sha256(
    map_id: str,
    prior_stage_passed: bool,
    files: Mapping[str, Mapping[str, Any]],
) -> str:
    return _sha256(canonical_bytes({
        "map": map_id,
        "prior_stage_passed": prior_stage_passed,
        "files": dict(files),
    }))


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fresh_outputs(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise QualificationPostcompileError(f"output must be fresh: {path}")
        if not path.parent.is_dir():
            raise QualificationPostcompileError(
                f"output parent must exist: {path.parent}"
            )
    all_paths = [*inputs, *outputs]
    if any(
        _overlap(left, right)
        for index, left in enumerate(all_paths)
        for right in all_paths[index + 1:]
    ):
        raise QualificationPostcompileError("input/output paths must be disjoint")


def _sparse_records(
    declaration: Mapping[str, Any], root: Path, suffixes: Sequence[str],
    eligible: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    declared = {str(row["map"]) for row in declaration["maps"]}
    if not eligible.issubset(declared):
        raise QualificationPostcompileError("sparse membership contains undeclared maps")
    expected = {f"{map_id}{suffix}" for map_id in eligible for suffix in suffixes}
    if root.is_symlink() or not root.is_dir():
        raise QualificationPostcompileError(
            f"sparse stage root is absent or a symlink: {root}"
        )
    actual: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise QualificationPostcompileError(
                f"sparse stage contains non-regular entry: {path.name}"
            )
        actual.add(path.name)
    if actual != expected:
        raise QualificationPostcompileError(
            "sparse stage membership differs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return {
        map_id: {
            suffix: _file_record(root / f"{map_id}{suffix}")
            for suffix in suffixes
        }
        for map_id in sorted(eligible)
    }


def _selected_records(
    root: Path, suffixes: Sequence[str], eligible: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        map_id: {
            suffix: _file_record(root / f"{map_id}{suffix}")
            for suffix in suffixes
        }
        for map_id in sorted(eligible)
    }


def _validate_prior_stage(
    path: Path,
    expected_stage: str,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str, set[str]]:
    report, raw = _load_json(path)
    expected_keys = {
        "schema", "qualification_id", "mode", "stage", "non_admissible",
        "retryable", "final_cohort_authorized", "declaration_sha256",
        "implementation", "toolchain_authority_sha256",
        "input_report_sha256", "infrastructure_checks",
        "map_count", "pass_count", "maps", "failures",
    }
    if set(report) != expected_keys:
        raise QualificationPostcompileError("prior stage report keys differ")
    if (
        report["schema"] != STAGE_SCHEMA
        or report["qualification_id"] != declaration["qualification_id"]
        or report["mode"] != "qualification"
        or report["stage"] != expected_stage
        or report["non_admissible"] is not True
        or report["retryable"] is not True
        or report["final_cohort_authorized"] is not False
        or report["declaration_sha256"] != declaration_sha256
        or report["implementation"] != implementation
        or report["toolchain_authority_sha256"]
        != declaration["toolchain_authority_sha256"]
        or report["failures"] != []
        or not isinstance(report["infrastructure_checks"], Mapping)
        or not report["infrastructure_checks"]
        or not REQUIRED_STAGE_CHECKS[expected_stage].issubset(
            report["infrastructure_checks"]
        )
        or not all(
            value is True for value in report["infrastructure_checks"].values()
        )
    ):
        raise QualificationPostcompileError(
            f"{expected_stage} report identity/infrastructure differs"
        )
    rows = report["maps"]
    if (
        not isinstance(rows, list)
        or report["map_count"] != 28
        or len(rows) != 28
    ):
        raise QualificationPostcompileError("prior stage map count differs")
    passed: set[str] = set()
    for declared, row in zip(declaration["maps"], rows):
        if not isinstance(row, Mapping) or set(row) != {
            "ordinal", "map", "criteria", "evidence_sha256", "failures",
            "passed",
        }:
            raise QualificationPostcompileError("prior stage map row differs")
        recomputed = (
            isinstance(row["criteria"], Mapping)
            and bool(row["criteria"])
            and all(value is True for value in row["criteria"].values())
            and row["failures"] == []
        )
        if (
            row["ordinal"] != declared["ordinal"]
            or row["map"] != declared["map"]
            or row["passed"] is not recomputed
            or not isinstance(row["evidence_sha256"], str)
            or len(row["evidence_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["evidence_sha256"])
        ):
            raise QualificationPostcompileError("prior stage row binding differs")
        if recomputed:
            passed.add(str(row["map"]))
    if report["pass_count"] != len(passed):
        raise QualificationPostcompileError("prior stage pass count differs")
    return report, raw, _sha256(raw), passed


def _validate_materialization_prior(
    *, report_path: Path, upstream_report_path: Path,
    declaration: Mapping[str, Any], declaration_sha256: str,
    implementation: Mapping[str, Any],
    materialized_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], bytes, str, set[str]]:
    upstream, upstream_raw, _upstream_sha, upstream_passed = _validate_prior_stage(
        upstream_report_path, "compiled-cm-preflight", declaration,
        declaration_sha256, implementation,
    )
    report, raw, digest, passed = _validate_prior_stage(
        report_path, "materialization", declaration, declaration_sha256,
        implementation,
    )
    if report["input_report_sha256"] != _sha256(upstream_raw):
        raise QualificationPostcompileError(
            "materialization input report hash does not bind supplied compiled-CM report"
        )
    expected_criteria = {
        "prior-stage-passed", "real-authority-materialization",
        "materialized-membership", "input-stability", "pinned-python-runtime",
    }
    for declared, upstream_row, row in zip(
        declaration["maps"], upstream["maps"], report["maps"]
    ):
        map_id = str(declared["map"])
        prior_passed = map_id in upstream_passed
        has_artifacts = map_id in materialized_records
        expected_row_criteria = {
            "prior-stage-passed": prior_passed,
            "real-authority-materialization": (
                not prior_passed or has_artifacts
            ),
            "materialized-membership": True,
            "input-stability": True,
            "pinned-python-runtime": True,
        }
        if (
            set(row["criteria"]) != expected_criteria
            or row["criteria"] != expected_row_criteria
            or upstream_row["passed"] is not prior_passed
            or (map_id in passed) is not has_artifacts
        ):
            raise QualificationPostcompileError(
                f"materialization eligibility differs for {map_id}"
            )
        if has_artifacts:
            expected_digest = stage_evidence_sha256(
                map_id, prior_passed, materialized_records[map_id]
            )
            if (
                not prior_passed
                or row["evidence_sha256"] != expected_digest
                or not all(row["criteria"].values())
                or row["failures"] != []
                or row["passed"] is not True
            ):
                raise QualificationPostcompileError(
                    f"materialization raw evidence differs for {map_id}"
                )
        else:
            expected_failure_digest = _sha256(canonical_bytes({
                "map": map_id, "stage": "materialization",
                "failures": row["failures"],
            }))
            if (
                row["passed"] is not False or not row["failures"]
                or row["evidence_sha256"] != expected_failure_digest
                or (
                    not prior_passed
                    and row["failures"]
                    != ["prior stage did not pass this map"]
                )
            ):
                raise QualificationPostcompileError(
                    f"materialization failure evidence differs for {map_id}"
                )
    if report["pass_count"] != len(materialized_records):
        raise QualificationPostcompileError(
            "materialization pass count differs from sparse artifact root"
        )
    return report, raw, digest, passed


def _authority_records(
    *, repo_root: Path, cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path,
    fall_oracle: Path, hook_attestation: Path,
) -> dict[str, Any]:
    try:
        gate = load_b1_authority_gate(repo_root)
        parity = admit_hook_parity_attestation(
            hook_attestation, repo_root=repo_root
        )
    except B1AuthorityError as error:
        raise QualificationPostcompileError(
            f"fresh B1 authority rejected: {error}"
        ) from error
    paths = {
        "cm": cm_oracle,
        "pmove": pmove_oracle,
        "hook": hook_oracle,
        "fall": fall_oracle,
        "hook_attestation": hook_attestation,
        "b1_gate": repo_root / "docs/multires/B1-GATE.json",
    }
    records = {name: _file_record(path) for name, path in paths.items()}
    expected = {
        "cm": gate.cm_executable_sha256,
        "pmove": gate.pmove_executable_sha256,
        "hook": parity.hook_executable_sha256,
        "fall": gate.fall_executable_sha256,
        "hook_attestation": parity.attestation_sha256,
    }
    if any(records[name]["sha256"] != digest for name, digest in expected.items()):
        raise QualificationPostcompileError("supplied authority bytes differ from fresh B1")
    return records


def _run_runtime_probe(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    try:
        completed = _run_process_group(
            list(command), cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
            timeout=RUNTIME_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise QualificationPostcompileError(
            f"pinned runtime preflight timed out: {' '.join(command)}"
        ) from error
    record = {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_sha256": _sha256(completed.stdout),
        "stderr_sha256": _sha256(completed.stderr),
    }
    if completed.returncode != 0:
        raise QualificationPostcompileError(
            f"pinned runtime preflight exited {completed.returncode}: "
            f"{' '.join(command)}"
        )
    return {**record, "stdout": completed.stdout}


def _validate_pinned_python_runtime(
    python_runtime: Path, repo_root: Path,
) -> dict[str, Any]:
    runtime = python_runtime.expanduser().absolute()
    if (
        runtime != PINNED_PYTHON
        or runtime.is_symlink()
        or runtime.resolve(strict=False) != PINNED_PYTHON
    ):
        raise QualificationPostcompileError(
            f"materialization requires pinned runtime {PINNED_PYTHON}"
        )
    executable = _file_record(runtime)
    if not os.access(runtime, os.X_OK):
        raise QualificationPostcompileError("pinned Python is not executable")
    if executable["sha256"] != PINNED_PYTHON_SHA256:
        raise QualificationPostcompileError("pinned Python executable digest differs")
    identity_script = (
        "import json,pathlib,sys,zstandard,zstandard.backend_c as backend;"
        "print(json.dumps({'python_version':list(sys.version_info[:3]),"
        "'zstandard_version':zstandard.__version__,"
        "'zstandard_init':str(pathlib.Path(zstandard.__file__).resolve()),"
        "'zstandard_backend':str(pathlib.Path(backend.__file__).resolve())},"
        "sort_keys=True,separators=(',',':')))"
    )
    identity_probe = _run_runtime_probe(
        [str(runtime), "-B", "-c", identity_script], cwd=repo_root
    )
    try:
        identity = json.loads(identity_probe.pop("stdout"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise QualificationPostcompileError(
            f"pinned runtime identity output is invalid: {error}"
        ) from error
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {
            "python_version", "zstandard_version", "zstandard_init",
            "zstandard_backend",
        }
        or identity["python_version"] != PINNED_PYTHON_VERSION
        or identity["zstandard_version"] != PINNED_ZSTANDARD_VERSION
    ):
        raise QualificationPostcompileError("pinned runtime identity differs")
    init_record = _file_record(Path(identity["zstandard_init"]))
    backend_record = _file_record(Path(identity["zstandard_backend"]))
    if (
        init_record["sha256"] != PINNED_ZSTANDARD_INIT_SHA256
        or backend_record["sha256"] != PINNED_ZSTANDARD_BACKEND_SHA256
    ):
        raise QualificationPostcompileError("pinned zstandard module digest differs")
    syntax = _run_runtime_probe(
        [
            str(runtime), "-B", str(repo_root / "tools/check_python_syntax_floor.py"),
            "--root", str(repo_root),
        ],
        cwd=repo_root,
    )
    help_probe = _run_runtime_probe(
        [
            str(runtime), "-B", str(repo_root / "tools/materialize_hook_claims.py"),
            "--help",
        ],
        cwd=repo_root,
    )
    import_probe = _run_runtime_probe(
        [
            str(runtime), "-B", "-c",
            "import harness.atlas_analyzer; import tools.materialize_hook_claims",
        ],
        cwd=repo_root,
    )
    for probe in (syntax, help_probe, import_probe):
        probe.pop("stdout")
    return {
        "python": executable,
        "python_path": str(runtime),
        "python_version": identity["python_version"],
        "zstandard_version": identity["zstandard_version"],
        "zstandard_init": {"path": identity["zstandard_init"], **init_record},
        "zstandard_backend": {
            "path": identity["zstandard_backend"], **backend_record,
        },
        "identity_probe": identity_probe,
        "syntax_preflight": syntax,
        "materializer_help_preflight": help_probe,
        "materializer_import_preflight": import_probe,
    }


def _real_materialize_map(
    *, map_id: str, stage_root: Path, log_root: Path,
    compiled_files: Mapping[str, Mapping[str, Any]], authorities: Mapping[str, Any],
    cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path, fall_oracle: Path,
    hook_attestation: Path, python_runtime: Path, timeout_seconds: int,
) -> None:
    materializer = ROOT / "tools/materialize_hook_claims.py"
    attestation = stage_root / f"{map_id}.hook-materialization.json"
    runtime_sidecar = stage_root / f"{map_id}.json"
    command = [
        str(python_runtime), "-B", str(materializer), "--bsp",
        str(stage_root / f"{map_id}.bsp"),
        "--meta", str(stage_root / f"{map_id}.meta.json"),
        "--runtime-sidecar", str(runtime_sidecar), "--output-attestation",
        str(attestation), "--cm-oracle", str(cm_oracle), "--pmove-oracle",
        str(pmove_oracle), "--hook-oracle", str(hook_oracle), "--fall-oracle",
        str(fall_oracle), "--hook-parity-attestation", str(hook_attestation),
    ]
    try:
        completed = _run_process_group(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise QualificationPostcompileError(
            f"{map_id} materializer timed out after {timeout_seconds} seconds"
        ) from error
    stdout_path = log_root / f"{map_id}.stdout.json"
    stderr_path = log_root / f"{map_id}.stderr.log"
    _exclusive_write(stdout_path, completed.stdout)
    _exclusive_write(stderr_path, completed.stderr)
    if completed.returncode != 0:
        raise QualificationPostcompileError(
            f"{map_id} materializer exited {completed.returncode}"
        )
    authority_sha256 = {
        name: authorities[name]["sha256"]
        for name in ("cm", "pmove", "hook", "fall", "hook_attestation")
    }
    try:
        _validate_result(
            completed.stdout, map_id=map_id, attestation=attestation,
            runtime_sidecar=runtime_sidecar, bsp=stage_root / f"{map_id}.bsp",
            compiled_files=compiled_files,
            authority_sha256=authority_sha256,
        )
    except MaterializeCohortError as error:
        raise QualificationPostcompileError(str(error)) from error


def _stage_report(
    *, declaration: Mapping[str, Any], declaration_sha256: str,
    implementation: Mapping[str, Any], input_report_sha256: str, stage: str,
    prior_passed: set[str], records: Mapping[str, Mapping[str, Any]] | None,
    suffixes: Sequence[str], infrastructure: Mapping[str, bool],
    global_failures: Sequence[str], map_failures: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    rows = []
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        files = {} if records is None else records.get(map_id, {})
        prior = map_id in prior_passed
        criteria = {
            "prior-stage-passed": prior,
            (
                "real-authority-materialization"
                if stage == "materialization" else "immutable-claims"
            ): not map_failures.get(map_id) and records is not None,
            (
                "materialized-membership"
                if stage == "materialization" else "claims-membership"
            ): records is not None,
            "input-stability": infrastructure.get("input-stability") is True,
        }
        if stage == "materialization":
            criteria["pinned-python-runtime"] = (
                infrastructure.get("pinned-python-runtime") is True
            )
        failures = list(map_failures.get(map_id, ()))
        if not prior:
            failures.append("prior stage did not pass this map")
        if records is None:
            failures.append(f"complete {stage} population was not published")
        evidence = (
            stage_evidence_sha256(map_id, prior, files)
            if map_id in (records or {})
            else _sha256(canonical_bytes({
                "map": map_id, "stage": stage, "failures": failures,
            }))
        )
        passed = all(criteria.values()) and not failures
        rows.append({
            "ordinal": declared["ordinal"], "map": map_id,
            "criteria": criteria, "evidence_sha256": evidence,
            "failures": failures, "passed": passed,
        })
    return {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": stage,
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": dict(implementation),
        "toolchain_authority_sha256": declaration[
            "toolchain_authority_sha256"
        ],
        "input_report_sha256": input_report_sha256,
        "infrastructure_checks": dict(infrastructure),
        "map_count": 28,
        "pass_count": sum(row["passed"] is True for row in rows),
        "maps": rows,
        "failures": list(global_failures),
    }


def _publish_stage_and_report(
    staging: Path, publication: Path, report_path: Path, report: Mapping[str, Any]
) -> None:
    _rename_noreplace(staging, publication)
    try:
        _exclusive_write(report_path, canonical_bytes(report))
    except Exception:
        _rename_noreplace(publication, staging)
        raise


def materialize_qualification(
    *, declaration_path: Path, prior_report_path: Path,
    prior_evidence_root: Path, compile_report_path: Path, compiled_root: Path,
    staging_root: Path, materialized_root: Path, log_root: Path,
    report_path: Path, cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path,
    fall_oracle: Path, hook_attestation: Path, python_runtime: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = _current_implementation,
    authority_provider: Callable[..., dict[str, Any]] = _authority_records,
    map_materializer: Callable[..., None] = _real_materialize_map,
    compiled_cm_validator: Callable[..., tuple[
        dict[str, Any], bytes, str, set[str]
    ]] = validate_published_qualification_compiled_cm,
    runtime_provider: Callable[[Path, Path], dict[str, Any]] = _validate_pinned_python_runtime,
) -> dict[str, Any]:
    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path, "prior_report": prior_report_path,
            "prior_evidence": prior_evidence_root,
            "compile_report": compile_report_path,
            "compiled": compiled_root, "staging": staging_root,
            "materialized": materialized_root, "logs": log_root,
            "report": report_path, "cm": cm_oracle, "pmove": pmove_oracle,
            "hook": hook_oracle, "fall": fall_oracle,
            "hook_attestation": hook_attestation, "python": python_runtime,
        }.items()
    }
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
        raise QualificationPostcompileError("timeout must be an integer in [1, 3600]")
    _fresh_outputs(
        [
            paths["compiled"], paths["declaration"], paths["prior_report"],
            paths["prior_evidence"], paths["compile_report"], paths["python"],
            paths["cm"], paths["pmove"], paths["hook"], paths["fall"],
            paths["hook_attestation"],
        ],
        [paths["staging"], paths["materialized"], paths["logs"], paths["report"]],
    )
    if paths["staging"].parent.stat().st_dev != paths["materialized"].parent.stat().st_dev:
        raise QualificationPostcompileError(
            "materialization staging/publication must share a filesystem"
        )
    implementation = _validate_implementation(implementation_provider(repo_root))
    declaration, declaration_raw, declaration_sha256 = _validate_declaration(
        paths["declaration"], implementation
    )
    try:
        _, prior_raw, prior_sha256, prior_passed = compiled_cm_validator(
            declaration_path=paths["declaration"],
            compile_report_path=paths["compile_report"],
            compiled_root=paths["compiled"], cm_oracle=paths["cm"],
            evidence_root=paths["prior_evidence"],
            report_path=paths["prior_report"], repo_root=repo_root,
        )
    except QualificationCompiledCmError as error:
        raise QualificationPostcompileError(str(error)) from error
    compiled_records = _selected_records(
        paths["compiled"], COMPILED_SUFFIXES, prior_passed
    )
    authorities = authority_provider(
        repo_root=repo_root, cm_oracle=paths["cm"], pmove_oracle=paths["pmove"],
        hook_oracle=paths["hook"], fall_oracle=paths["fall"],
        hook_attestation=paths["hook_attestation"],
    )
    runtime_identity = runtime_provider(paths["python"], repo_root)
    paths["staging"].mkdir()
    paths["logs"].mkdir()
    for declared in declaration["maps"]:
        if declared["map"] not in prior_passed:
            continue
        for suffix in COMPILED_SUFFIXES:
            shutil.copyfile(
                paths["compiled"] / f"{declared['map']}{suffix}",
                paths["staging"] / f"{declared['map']}{suffix}",
            )
    map_failures: dict[str, list[str]] = {}
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        if map_id not in prior_passed:
            continue
        try:
            map_materializer(
                map_id=map_id, stage_root=paths["staging"],
                log_root=paths["logs"], compiled_files=compiled_records[map_id],
                authorities=authorities, cm_oracle=paths["cm"],
                pmove_oracle=paths["pmove"], hook_oracle=paths["hook"],
                fall_oracle=paths["fall"],
                hook_attestation=paths["hook_attestation"],
                python_runtime=paths["python"],
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            map_failures[map_id] = [f"{type(error).__name__}: {error}"]
            for suffix in MATERIALIZED_SUFFIXES:
                (paths["staging"] / f"{map_id}{suffix}").unlink(missing_ok=True)
    records = None
    global_failures = []
    succeeded = prior_passed - set(map_failures)
    try:
        records = _sparse_records(
            declaration, paths["staging"], MATERIALIZED_SUFFIXES, succeeded
        )
        for map_id in succeeded:
            for suffix in COMPILED_SUFFIXES:
                if (
                    suffix != ".json"
                    and records[map_id][suffix] != compiled_records[map_id][suffix]
                ):
                    raise QualificationPostcompileError(
                        f"materializer changed immutable {map_id}{suffix}"
                    )
    except (QualificationCompileError, QualificationPostcompileError) as error:
        global_failures.append(str(error))
        records = None
    try:
        stable = (
            paths["declaration"].read_bytes() == declaration_raw
            and paths["prior_report"].read_bytes() == prior_raw
            and _selected_records(
                paths["compiled"], COMPILED_SUFFIXES, prior_passed
            ) == compiled_records
            and authority_provider(
                repo_root=repo_root, cm_oracle=paths["cm"],
                pmove_oracle=paths["pmove"], hook_oracle=paths["hook"],
                fall_oracle=paths["fall"],
                hook_attestation=paths["hook_attestation"],
            ) == authorities
            and _validate_implementation(implementation_provider(repo_root))
            == implementation
            and runtime_provider(paths["python"], repo_root) == runtime_identity
        )
    except Exception as error:
        stable = False
        global_failures.append(f"input stability: {error}")
    if not stable:
        records = None
        if not any("input stability" in failure for failure in global_failures):
            global_failures.append("input stability failed")
    published = records is not None and not global_failures
    infrastructure = {
        "authority-bound": True,
        "materialized-membership": published,
        "real-v4-materializer": True,
        "pinned-python-runtime": True,
        "exact-membership": published,
        "input-stability": stable,
        "exclusive-publication": published,
    }
    report = _stage_report(
        declaration=declaration, declaration_sha256=declaration_sha256,
        implementation=implementation, input_report_sha256=prior_sha256,
        stage="materialization", prior_passed=prior_passed,
        records=records if published else None, suffixes=MATERIALIZED_SUFFIXES,
        infrastructure=infrastructure, global_failures=global_failures,
        map_failures=map_failures,
    )
    if published:
        _publish_stage_and_report(
            paths["staging"], paths["materialized"], paths["report"], report
        )
        return report
    _exclusive_write(paths["report"], canonical_bytes(report))
    raise QualificationPostcompileError("; ".join(global_failures))


def claims_qualification(
    *, declaration_path: Path, prior_report_path: Path,
    upstream_report_path: Path,
    materialized_root: Path, staging_root: Path, claims_root: Path,
    report_path: Path, repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = _current_implementation,
    claim_builder: Callable[[Path], Mapping[str, Any]] = build_generator_claims,
) -> dict[str, Any]:
    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path, "prior_report": prior_report_path,
            "upstream_report": upstream_report_path,
            "materialized": materialized_root, "staging": staging_root,
            "claims": claims_root, "report": report_path,
        }.items()
    }
    _fresh_outputs(
        [
            paths["materialized"], paths["declaration"], paths["prior_report"],
            paths["upstream_report"],
        ],
        [paths["staging"], paths["claims"], paths["report"]],
    )
    if paths["staging"].parent.stat().st_dev != paths["claims"].parent.stat().st_dev:
        raise QualificationPostcompileError(
            "claims staging/publication must share a filesystem"
        )
    implementation = _validate_implementation(implementation_provider(repo_root))
    declaration, declaration_raw, declaration_sha256 = _validate_declaration(
        paths["declaration"], implementation
    )
    _, preliminary_raw, preliminary_sha256, preliminary_passed = _validate_prior_stage(
        paths["prior_report"], "materialization", declaration,
        declaration_sha256, implementation,
    )
    materialized_records = _sparse_records(
        declaration, paths["materialized"], MATERIALIZED_SUFFIXES,
        preliminary_passed,
    )
    _, prior_raw, prior_sha256, prior_passed = _validate_materialization_prior(
        report_path=paths["prior_report"],
        upstream_report_path=paths["upstream_report"], declaration=declaration,
        declaration_sha256=declaration_sha256, implementation=implementation,
        materialized_records=materialized_records,
    )
    if prior_raw != preliminary_raw or prior_sha256 != preliminary_sha256:
        raise QualificationPostcompileError("materialization report changed during validation")
    paths["staging"].mkdir()
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        if map_id not in prior_passed:
            continue
        for suffix in MATERIALIZED_SUFFIXES:
            shutil.copyfile(
                paths["materialized"] / f"{map_id}{suffix}",
                paths["staging"] / f"{map_id}{suffix}",
            )
    map_failures: dict[str, list[str]] = {}
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        if map_id not in prior_passed:
            continue
        try:
            payload = canonical_bytes(claim_builder(paths["staging"] / f"{map_id}.map"))
            _exclusive_write(
                paths["staging"] / f"{map_id}.generator-claims.json", payload
            )
        except (ClaimValidationError, OSError, ValueError, KeyError, TypeError) as error:
            map_failures[map_id] = [f"{type(error).__name__}: {error}"]
            for suffix in CLAIMS_SUFFIXES:
                (paths["staging"] / f"{map_id}{suffix}").unlink(missing_ok=True)
    records = None
    global_failures = []
    succeeded = prior_passed - set(map_failures)
    try:
        records = _sparse_records(
            declaration, paths["staging"], CLAIMS_SUFFIXES, succeeded
        )
        for map_id in succeeded:
            for suffix in MATERIALIZED_SUFFIXES:
                if records[map_id][suffix] != materialized_records[map_id][suffix]:
                    raise QualificationPostcompileError(
                        f"claims changed immutable {map_id}{suffix}"
                    )
    except (QualificationCompileError, QualificationPostcompileError) as error:
        global_failures.append(str(error))
        records = None
    try:
        stable = (
            paths["declaration"].read_bytes() == declaration_raw
            and paths["prior_report"].read_bytes() == prior_raw
            and _sparse_records(
                declaration, paths["materialized"], MATERIALIZED_SUFFIXES,
                prior_passed,
            ) == materialized_records
            and _validate_implementation(implementation_provider(repo_root))
            == implementation
        )
    except Exception as error:
        stable = False
        global_failures.append(f"input stability: {error}")
    if not stable:
        records = None
        if not any("input stability" in failure for failure in global_failures):
            global_failures.append("input stability failed")
    published = records is not None and not global_failures
    infrastructure = {
        "immutable-claims": not map_failures,
        "claims-membership": published,
        "exact-membership": published,
        "input-stability": stable,
        "exclusive-publication": published,
    }
    report = _stage_report(
        declaration=declaration, declaration_sha256=declaration_sha256,
        implementation=implementation, input_report_sha256=prior_sha256,
        stage="claims", prior_passed=prior_passed,
        records=records if published else None, suffixes=CLAIMS_SUFFIXES,
        infrastructure=infrastructure, global_failures=global_failures,
        map_failures=map_failures,
    )
    if published:
        _publish_stage_and_report(
            paths["staging"], paths["claims"], paths["report"], report
        )
        return report
    _exclusive_write(paths["report"], canonical_bytes(report))
    raise QualificationPostcompileError("; ".join(global_failures))


def validate_published_qualification_postcompile(
    *, declaration_path: Path, compile_report_path: Path, compiled_root: Path,
    compiled_cm_report_path: Path, compiled_cm_evidence_root: Path,
    materialization_report_path: Path, materialized_root: Path,
    materialization_log_root: Path, claims_report_path: Path, claims_root: Path,
    cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path, fall_oracle: Path,
    hook_attestation: Path, python_runtime: Path, repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = _current_implementation,
    authority_provider: Callable[..., dict[str, Any]] = _authority_records,
    runtime_provider: Callable[[Path, Path], dict[str, Any]] = _validate_pinned_python_runtime,
    result_validator: Callable[..., Any] = _validate_result,
    claim_builder: Callable[[Path], Mapping[str, Any]] = build_generator_claims,
    compiled_cm_validator: Callable[..., tuple[
        dict[str, Any], bytes, str, set[str]
    ]] = validate_published_qualification_compiled_cm,
) -> dict[str, Any]:
    """Replay sparse materialization and claims from real upstream evidence."""

    implementation = _validate_implementation(implementation_provider(repo_root))
    declaration, declaration_raw, declaration_sha256 = _validate_declaration(
        declaration_path, implementation
    )
    try:
        cm_report, cm_raw, cm_sha256, cm_passed = compiled_cm_validator(
            declaration_path=declaration_path,
            compile_report_path=compile_report_path,
            compiled_root=compiled_root, cm_oracle=cm_oracle,
            evidence_root=compiled_cm_evidence_root,
            report_path=compiled_cm_report_path, repo_root=repo_root,
        )
    except QualificationCompiledCmError as error:
        raise QualificationPostcompileError(str(error)) from error
    material_report, material_raw, material_sha256, material_passed = _validate_prior_stage(
        materialization_report_path, "materialization", declaration,
        declaration_sha256, implementation,
    )
    if material_report["input_report_sha256"] != cm_sha256:
        raise QualificationPostcompileError(
            "materialization raw hash chain differs from compiled-CM report"
        )
    material_records = _sparse_records(
        declaration, materialized_root, MATERIALIZED_SUFFIXES, material_passed
    )
    replayed_material, replayed_raw, replayed_sha, replayed_passed = (
        _validate_materialization_prior(
            report_path=materialization_report_path,
            upstream_report_path=compiled_cm_report_path,
            declaration=declaration, declaration_sha256=declaration_sha256,
            implementation=implementation, materialized_records=material_records,
        )
    )
    if (
        replayed_material != material_report or replayed_raw != material_raw
        or replayed_sha != material_sha256 or replayed_passed != material_passed
        or not material_passed.issubset(cm_passed)
    ):
        raise QualificationPostcompileError("materialization replay disposition differs")
    authorities = authority_provider(
        repo_root=repo_root, cm_oracle=cm_oracle, pmove_oracle=pmove_oracle,
        hook_oracle=hook_oracle, fall_oracle=fall_oracle,
        hook_attestation=hook_attestation,
    )
    runtime_identity = runtime_provider(python_runtime, repo_root)
    if materialization_log_root.is_symlink() or not materialization_log_root.is_dir():
        raise QualificationPostcompileError("materialization log root is absent or a symlink")
    actual_logs: set[str] = set()
    for path in materialization_log_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise QualificationPostcompileError(
                f"materialization logs contain non-regular entry: {path.name}"
            )
        actual_logs.add(path.name)
    required_logs = {
        f"{map_id}{suffix}" for map_id in material_passed
        for suffix in (".stdout.json", ".stderr.log")
    }
    allowed_logs = {
        f"{map_id}{suffix}" for map_id in cm_passed
        for suffix in (".stdout.json", ".stderr.log")
    }
    if not required_logs.issubset(actual_logs) or not actual_logs.issubset(allowed_logs):
        raise QualificationPostcompileError("materialization log membership differs")
    log_records = {
        name: _file_record(materialization_log_root / name)
        for name in actual_logs
    }
    authority_sha256 = {
        name: authorities[name]["sha256"]
        for name in ("cm", "pmove", "hook", "fall", "hook_attestation")
    }
    for map_id in sorted(material_passed):
        stdout = (materialization_log_root / f"{map_id}.stdout.json").read_bytes()
        try:
            result_validator(
                stdout, map_id=map_id,
                attestation=materialized_root / f"{map_id}.hook-materialization.json",
                runtime_sidecar=materialized_root / f"{map_id}.json",
                bsp=materialized_root / f"{map_id}.bsp",
                compiled_files={
                    suffix: material_records[map_id][suffix]
                    for suffix in COMPILED_SUFFIXES
                },
                authority_sha256=authority_sha256,
            )
        except MaterializeCohortError as error:
            raise QualificationPostcompileError(
                f"materialization result replay failed for {map_id}: {error}"
            ) from error
        for suffix in COMPILED_SUFFIXES:
            if suffix == ".json":
                continue
            if material_records[map_id][suffix] != _file_record(
                compiled_root / f"{map_id}{suffix}"
            ):
                raise QualificationPostcompileError(
                    f"materialization changed compiled input {map_id}{suffix}"
                )

    claims_report, claims_raw, claims_sha256, claims_passed = _validate_prior_stage(
        claims_report_path, "claims", declaration, declaration_sha256,
        implementation,
    )
    if claims_report["input_report_sha256"] != material_sha256:
        raise QualificationPostcompileError(
            "claims raw hash chain differs from materialization report"
        )
    claims_records = _sparse_records(
        declaration, claims_root, CLAIMS_SUFFIXES, claims_passed
    )
    expected_claim_criteria = {
        "prior-stage-passed", "immutable-claims", "claims-membership",
        "input-stability",
    }
    for declared, row in zip(declaration["maps"], claims_report["maps"]):
        map_id = str(declared["map"])
        prior = map_id in material_passed
        passing = map_id in claims_passed
        expected_row_criteria = {
            "prior-stage-passed": prior,
            "immutable-claims": not prior or passing,
            "claims-membership": True,
            "input-stability": True,
        }
        if (
            set(row["criteria"]) != expected_claim_criteria
            or row["criteria"] != expected_row_criteria
            or row["passed"] is not passing
        ):
            raise QualificationPostcompileError(
                f"claims eligibility differs for {map_id}"
            )
        if passing:
            if (
                not prior or row["evidence_sha256"] != stage_evidence_sha256(
                    map_id, prior, claims_records[map_id]
                )
                or not all(row["criteria"].values()) or row["failures"] != []
            ):
                raise QualificationPostcompileError(
                    f"claims raw evidence differs for {map_id}"
                )
            for suffix in MATERIALIZED_SUFFIXES:
                if claims_records[map_id][suffix] != material_records[map_id][suffix]:
                    raise QualificationPostcompileError(
                        f"claims changed materialized input {map_id}{suffix}"
                    )
            expected_claims = canonical_bytes(
                claim_builder(claims_root / f"{map_id}.map")
            )
            if (claims_root / f"{map_id}.generator-claims.json").read_bytes() != expected_claims:
                raise QualificationPostcompileError(
                    f"generator claims differ from independent rebuild for {map_id}"
                )
        else:
            expected_digest = _sha256(canonical_bytes({
                "map": map_id, "stage": "claims", "failures": row["failures"],
            }))
            if (
                row["evidence_sha256"] != expected_digest or not row["failures"]
                or (
                    not prior
                    and row["failures"]
                    != ["prior stage did not pass this map"]
                )
            ):
                raise QualificationPostcompileError(
                    f"claims failure evidence differs for {map_id}"
                )
    if not claims_passed.issubset(material_passed):
        raise QualificationPostcompileError("claims pass set exceeds materialization")
    stable = (
        declaration_path.read_bytes() == declaration_raw
        and compiled_cm_report_path.read_bytes() == cm_raw
        and materialization_report_path.read_bytes() == material_raw
        and claims_report_path.read_bytes() == claims_raw
        and _sparse_records(
            declaration, materialized_root, MATERIALIZED_SUFFIXES, material_passed
        ) == material_records
        and _sparse_records(
            declaration, claims_root, CLAIMS_SUFFIXES, claims_passed
        ) == claims_records
        and {
            name: _file_record(materialization_log_root / name)
            for name in actual_logs
        } == log_records
        and authority_provider(
            repo_root=repo_root, cm_oracle=cm_oracle, pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle, fall_oracle=fall_oracle,
            hook_attestation=hook_attestation,
        ) == authorities
        and runtime_provider(python_runtime, repo_root) == runtime_identity
        and _validate_implementation(implementation_provider(repo_root)) == implementation
    )
    if not stable:
        raise QualificationPostcompileError("postcompile replay inputs changed")
    return {
        "compiled_cm_sha256": cm_sha256,
        "materialization_sha256": material_sha256,
        "claims_sha256": claims_sha256,
        "compiled_cm_passed": sorted(cm_passed),
        "materialization_passed": sorted(material_passed),
        "claims_passed": sorted(claims_passed),
        "runtime": runtime_identity,
        "authorities": authorities,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--declaration", type=Path, required=True)
    materialize.add_argument("--prior-report", type=Path, required=True)
    materialize.add_argument("--prior-evidence-root", type=Path, required=True)
    materialize.add_argument("--compile-report", type=Path, required=True)
    materialize.add_argument("--compiled-root", type=Path, required=True)
    materialize.add_argument("--staging-root", type=Path, required=True)
    materialize.add_argument("--materialized-root", type=Path, required=True)
    materialize.add_argument("--log-root", type=Path, required=True)
    materialize.add_argument("--report", type=Path, required=True)
    materialize.add_argument("--cm-oracle", type=Path, required=True)
    materialize.add_argument("--pmove-oracle", type=Path, required=True)
    materialize.add_argument("--hook-oracle", type=Path, required=True)
    materialize.add_argument("--fall-oracle", type=Path, required=True)
    materialize.add_argument("--hook-attestation", type=Path, required=True)
    materialize.add_argument("--python", type=Path, required=True)
    materialize.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    claims = subparsers.add_parser("claims")
    claims.add_argument("--declaration", type=Path, required=True)
    claims.add_argument("--prior-report", type=Path, required=True)
    claims.add_argument("--upstream-report", type=Path, required=True)
    claims.add_argument("--materialized-root", type=Path, required=True)
    claims.add_argument("--staging-root", type=Path, required=True)
    claims.add_argument("--claims-root", type=Path, required=True)
    claims.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "materialize":
            report = materialize_qualification(
                declaration_path=args.declaration,
                prior_report_path=args.prior_report,
                prior_evidence_root=args.prior_evidence_root,
                compile_report_path=args.compile_report,
                compiled_root=args.compiled_root, staging_root=args.staging_root,
                materialized_root=args.materialized_root, log_root=args.log_root,
                report_path=args.report, cm_oracle=args.cm_oracle,
                pmove_oracle=args.pmove_oracle, hook_oracle=args.hook_oracle,
                fall_oracle=args.fall_oracle,
                hook_attestation=args.hook_attestation,
                python_runtime=args.python,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            report = claims_qualification(
                declaration_path=args.declaration,
                prior_report_path=args.prior_report,
                upstream_report_path=args.upstream_report,
                materialized_root=args.materialized_root,
                staging_root=args.staging_root, claims_root=args.claims_root,
                report_path=args.report,
            )
    except (QualificationPostcompileError, QualificationCompileError) as error:
        print(f"B2 qualification {args.command} failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
