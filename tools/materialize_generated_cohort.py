#!/usr/bin/env python3
"""Atomically materialize one fresh, declaration-bound generated cohort.

This producer has no resume, retry, subset, overwrite, or directory-discovery
mode.  It copies one exact compiled stage into a fresh unpublished directory,
invokes the single-map V4 materializer once per declared row, and publishes the
materialized directory only after all 28 rows and exact membership pass.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    STAGE_SUFFIXES,
    canonical_bytes,
    file_sha256,
    load_declaration,
    verify_stage_membership,
)
from tools.retired_cohort_registry import require_unretired_declaration  # noqa: E402
from harness.hook_claims_v4 import (  # noqa: E402
    HookClaimsV4Error,
    load_materialization,
    validate_runtime_sidecar,
)
from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    admit_hook_parity_attestation,
    load_b1_authority_gate,
)


REPORT_SCHEMA = "q2-generated-cohort-materialization-v1"
RESULT_SCHEMA = "q2-hook-materialization-result-v4"
COMPILED_FILE_COUNT = 168
MATERIALIZED_FILE_COUNT = 196
MATERIALIZED_RECORD_COUNT = 6
DEFAULT_MATERIALIZER_TIMEOUT_SECONDS = 900
MAX_MATERIALIZER_TIMEOUT_SECONDS = 3600
AT_FDCWD = -100
RENAME_NOREPLACE = 1
CURRENT_B1_GATE_SHA256 = (
    "b3f5ac1a22a2c07a7dfaef3e145c35e2fcbf9658fba7e2bf1a9d2501eebd5b6d"
)

# Compatibility constants used by the report/test helpers.  Final producer
# admission derives every expected digest from the validated current B1 gate
# and its admitted hook attestation in ``_expected_authority_sha256`` below.
EXPECTED_SHA256 = {
    "cm": "781edaee1b9317766dbf831ad5edc8b5fdebe696969ca1efe0e54e2f3e5c7d1e",
    "pmove": "66b481e924ec3d0a5e4eaf5458dd34cfe3c0927d5b7650455bceb368666718e4",
    "hook": "cd8bc4107ae2e9f4ac006fbe469b360832db80b96a5597c2e5dfe12c32dc9284",
    "fall": "dfdcf7ed74cc3ad7b8aa73df86986a8a4a31207da98ccffb4dd61673c324bef8",
    "hook_attestation": (
        "2e473d8face6b89f5b32798ddc5264bb8cc406e8dc29fd837e85bbd11b53d5ab"
    ),
}


class MaterializeCohortError(ValueError):
    """Raised when a cohort cannot be published without weakening scope."""

    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_error(error: BaseException) -> str:
    return " ".join(str(error).replace("\n", " ").split())[:4096]


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise MaterializeCohortError(
                "path-preflight", f"no existing ancestor for {path}"
            )
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing a racing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MaterializeCohortError(
            "publication", "libc has no renameat2; atomic no-replace is required"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise MaterializeCohortError(
            "publication",
            f"atomic no-replace publish failed: {os.strerror(error_number)}",
        )
    _fsync_directory(source.parent)
    if source.parent != destination.parent:
        _fsync_directory(destination.parent)


def _exclusive_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _stage_payload(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".stage", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _copy_exclusive(source: Path, destination: Path) -> None:
    descriptor = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _run_process_group(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout: int,
    stderr: int,
    check: bool,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    if stdout != subprocess.PIPE or stderr != subprocess.PIPE or check:
        raise ValueError("cohort runner requires captured output and check=false")
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        captured_stdout, captured_stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        captured_stdout, captured_stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout, output=captured_stdout, stderr=captured_stderr
        ) from error
    return subprocess.CompletedProcess(
        command, process.returncode, captured_stdout, captured_stderr
    )


def _membership_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "passed": report["passed"],
        "expected_map_count": report["expected_map_count"],
        "expected_file_count": report["expected_file_count"],
        "actual_file_count": report["actual_file_count"],
        "failures": list(report["failures"]),
        "report_sha256": _sha256_bytes(canonical_bytes(report)),
    }


def _require_fresh_topology(
    *,
    compiled_dir: Path,
    stage_dir: Path,
    materialized_dir: Path,
    log_dir: Path,
    report_path: Path,
) -> None:
    if not compiled_dir.is_dir() or compiled_dir.is_symlink():
        raise MaterializeCohortError(
            "path-preflight", "compiled stage is missing or is a symlink"
        )
    fresh = {
        "unpublished stage": stage_dir,
        "materialized publication": materialized_dir,
        "terminal log directory": log_dir,
        "materialization report": report_path,
    }
    for label, path in fresh.items():
        if _path_exists(path):
            raise MaterializeCohortError(
                "path-preflight", f"{label} already exists; refusing reuse"
            )

    directories = {
        "compiled": compiled_dir,
        "stage": stage_dir,
        "materialized": materialized_dir,
        "logs": log_dir,
    }
    names = list(directories)
    for index, first_name in enumerate(names):
        first = directories[first_name]
        for second_name in names[index + 1 :]:
            second = directories[second_name]
            if (
                first.resolve() == second.resolve()
                or _path_is_within(first, second)
                or _path_is_within(second, first)
            ):
                raise MaterializeCohortError(
                    "path-preflight",
                    f"{first_name} and {second_name} roots must be separate and non-nested",
                )
    if any(
        _path_is_within(report_path, directory)
        for directory in directories.values()
    ):
        raise MaterializeCohortError(
            "path-preflight", "report must be outside every stage and log root"
        )
    if stage_dir.parent.resolve() != materialized_dir.parent.resolve():
        raise MaterializeCohortError(
            "path-preflight", "stage and publication must be sibling directories"
        )

    device_paths = (
        compiled_dir,
        stage_dir.parent,
        materialized_dir.parent,
        log_dir.parent,
        report_path.parent,
    )
    devices = {
        _nearest_existing_directory(path).stat().st_dev for path in device_paths
    }
    if len(devices) != 1:
        raise MaterializeCohortError(
            "path-preflight",
            "compiled, stage, publication, logs, and report must share one filesystem",
        )


def _can_publish_failure_report(
    report_path: Path, *forbidden_roots: Path
) -> bool:
    return not _path_exists(report_path) and not any(
        _path_is_within(report_path, root) for root in forbidden_roots
    )


def _expected_authority_sha256(hook_attestation: Path) -> dict[str, str]:
    """Derive final-producer byte admission from the validated current B1 seal."""

    gate_path = ROOT / "docs/multires/B1-GATE.json"
    try:
        gate = load_b1_authority_gate(ROOT)
        parity = admit_hook_parity_attestation(
            hook_attestation, repo_root=ROOT
        )
    except B1AuthorityError as error:
        raise MaterializeCohortError(
            "authority-preflight", f"current B1 authority admission failed: {error}"
        ) from error
    gate_sha256 = file_sha256(gate_path)
    if gate_sha256 != CURRENT_B1_GATE_SHA256:
        raise MaterializeCohortError(
            "authority-preflight",
            "current B1 gate exact bytes differ: "
            f"expected {CURRENT_B1_GATE_SHA256}, got {gate_sha256}",
        )
    return {
        "b1_gate": gate_sha256,
        "cm": gate.cm_executable_sha256,
        "pmove": gate.pmove_executable_sha256,
        "hook": parity.hook_executable_sha256,
        "fall": gate.fall_executable_sha256,
        "hook_attestation": parity.attestation_sha256,
    }


def preflight_authority_records(
    *,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_attestation: Path,
) -> dict[str, dict[str, Any]]:
    expected_sha256 = _expected_authority_sha256(hook_attestation)
    paths = {
        "b1_gate": ROOT / "docs/multires/B1-GATE.json",
        "cm": cm_oracle,
        "pmove": pmove_oracle,
        "hook": hook_oracle,
        "fall": fall_oracle,
        "hook_attestation": hook_attestation,
    }
    records: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        expected = expected_sha256[label]
        if not path.is_file() or path.is_symlink():
            raise MaterializeCohortError(
                "authority-preflight", f"{label} authority is missing or is a symlink"
            )
        actual = file_sha256(path)
        records[label] = {
            "filename": path.name,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
        if actual != expected:
            raise MaterializeCohortError(
                "authority-preflight",
                f"{label} authority SHA-256 differs: expected {expected}, got {actual}",
            )
    return records


def _map_rows(declaration: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": row["ordinal"],
            "map": row["map"],
            "status": "not-attempted",
            "returncode": None,
            "result_sha256": None,
            "stdout": None,
            "stderr": None,
            "error": None,
        }
        for row in declaration["maps"]
    ]


def _base_report(
    *,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    rows: list[dict[str, Any]],
    timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "cohort_id": declaration["cohort_id"],
        "declaration_sha256": declaration_sha256,
        "selection_policy": "all-or-nothing",
        "expected_map_count": 28,
        "expected_compiled_file_count": COMPILED_FILE_COUNT,
        "expected_materialized_file_count": MATERIALIZED_FILE_COUNT,
        "materializer_timeout_seconds": timeout_seconds,
        "retry_allowed": False,
        "resume_allowed": False,
        "salvage_allowed": False,
        "replacement_allowed": False,
        "maps": rows,
    }


def _complete_report(
    base: Mapping[str, Any],
    *,
    phase: str,
    passed: bool,
    compiled_membership: Mapping[str, Any] | None,
    materialized_membership: Mapping[str, Any] | None,
    authorities: Mapping[str, Any],
    attempted_count: int,
    pass_count: int,
    materialized_published: bool,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        **base,
        "phase": phase,
        "passed": passed,
        "attempted_count": attempted_count,
        "pass_count": pass_count,
        "materialized_published": materialized_published,
        "compiled_membership": compiled_membership,
        "materialized_membership": materialized_membership,
        "authorities": dict(authorities),
        "failure": failure,
    }


def _log_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _validate_result(
    payload: bytes,
    *,
    map_id: str,
    attestation: Path,
    runtime_sidecar: Path,
    bsp: Path,
    compiled_files: Mapping[str, Mapping[str, Any]],
    authority_sha256: Mapping[str, str],
) -> str:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} result is not one JSON document: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} result must be an object"
        )
    expected_keys = {
        "schema",
        "map",
        "passed",
        "selected_count",
        "attestation_sha256",
        "runtime_sidecar_sha256",
    }
    if set(value) != expected_keys:
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} result keys differ"
        )
    if (
        value["schema"] != RESULT_SCHEMA
        or value["map"] != map_id
        or value["passed"] is not True
        or value["selected_count"] != MATERIALIZED_RECORD_COUNT
    ):
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} result identity or pass fields differ"
        )
    try:
        document, attestation_sha256 = load_materialization(attestation)
    except (HookClaimsV4Error, OSError, ValueError) as error:
        raise MaterializeCohortError(
            "map-result-validation",
            f"{map_id} attestation is not canonical V4: {_canonical_error(error)}",
        ) from error
    expected_bsp = compiled_files[".bsp"]
    expected_projection = compiled_files[".json"]["sha256"]
    if document["map"] != map_id:
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} attestation map differs"
        )
    if document["source_projection_sha256"] != expected_projection:
        raise MaterializeCohortError(
            "map-result-validation",
            f"{map_id} attestation source projection differs",
        )
    if document["bsp"] != {
        "sha256": expected_bsp["sha256"],
        "size_bytes": expected_bsp["bytes"],
    }:
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} attestation BSP binding differs"
        )
    if (
        not bsp.is_file()
        or bsp.is_symlink()
        or bsp.stat().st_size != expected_bsp["bytes"]
        or file_sha256(bsp) != expected_bsp["sha256"]
    ):
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} staged BSP differs"
        )
    expected_executables = {
        "collision": authority_sha256["cm"],
        "pmove": authority_sha256["pmove"],
        "hook": authority_sha256["hook"],
        "fall": authority_sha256["fall"],
    }
    if any(
        document["oracles"][name]["executable_sha256"] != digest
        for name, digest in expected_executables.items()
    ):
        raise MaterializeCohortError(
            "map-result-validation",
            f"{map_id} attestation executable authority differs",
        )
    if (
        document["oracles"]["hook_parity_attestation_sha256"]
        != authority_sha256["hook_attestation"]
    ):
        raise MaterializeCohortError(
            "map-result-validation",
            f"{map_id} attestation hook authority differs",
        )
    if not runtime_sidecar.is_file() or runtime_sidecar.is_symlink():
        raise MaterializeCohortError(
            "map-result-validation",
            f"{map_id} runtime sidecar is missing or is a symlink",
        )
    runtime_payload = runtime_sidecar.read_bytes()
    try:
        validate_runtime_sidecar(
            runtime_payload,
            map_id=map_id,
            bsp_sha256=expected_bsp["sha256"],
            materialization_sha256=attestation_sha256,
            records=document["selected_records"],
        )
    except HookClaimsV4Error as error:
        raise MaterializeCohortError(
            "map-result-validation",
            f"{map_id} runtime sidecar differs from V4 attestation: "
            f"{_canonical_error(error)}",
        ) from error
    if value["attestation_sha256"] != attestation_sha256:
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} attestation digest differs"
        )
    if value["runtime_sidecar_sha256"] != _sha256_bytes(runtime_payload):
        raise MaterializeCohortError(
            "map-result-validation", f"{map_id} runtime sidecar digest differs"
        )
    return _sha256_bytes(payload)


def _require_immutable_compiled_inputs(
    declaration: Mapping[str, Any],
    compiled_membership: Mapping[str, Any],
    stage_dir: Path,
) -> None:
    membership_rows = {
        row["map"]: row["files"] for row in compiled_membership["maps"]
    }
    for declared in declaration["maps"]:
        map_id = declared["map"]
        for suffix in STAGE_SUFFIXES["compiled"]:
            if suffix == ".json":
                # This is the sole compiled member that V4 deliberately
                # upgrades from a non-admissible projection to a sealed
                # runtime sidecar.
                continue
            path = stage_dir / f"{map_id}{suffix}"
            expected = membership_rows[map_id][suffix]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != expected["bytes"]
                or file_sha256(path) != expected["sha256"]
            ):
                raise MaterializeCohortError(
                    "materialized-input-finalization",
                    f"{map_id}{suffix} changed during materialization",
                )


def _atomic_publish_with_report(
    *,
    stage_dir: Path,
    materialized_dir: Path,
    report_path: Path,
    report_payload: bytes,
) -> None:
    staged_report = _stage_payload(report_path, report_payload)
    directory_published = False
    report_published = False
    try:
        if _path_exists(materialized_dir) or _path_exists(report_path):
            raise MaterializeCohortError(
                "publication", "publication or report appeared during materialization"
            )
        _rename_noreplace(stage_dir, materialized_dir)
        directory_published = True
        try:
            os.link(staged_report, report_path)
            report_published = True
            _fsync_directory(report_path.parent)
            try:
                staged_report.unlink()
                _fsync_directory(report_path.parent)
            except OSError:
                # The final report and directory are already durable. A
                # hidden same-inode staging link is harmless and is retried
                # by the unconditional cleanup below.
                pass
        except BaseException as error:
            if report_published:
                report_path.unlink(missing_ok=True)
                report_published = False
                _fsync_directory(report_path.parent)
            _rename_noreplace(materialized_dir, stage_dir)
            directory_published = False
            raise MaterializeCohortError(
                "publication", f"report publication failed and directory was rolled back: {error}"
            ) from error
    except BaseException:
        if directory_published:
            try:
                _rename_noreplace(materialized_dir, stage_dir)
            except BaseException as rollback_error:
                raise MaterializeCohortError(
                    "publication", f"materialized publication rollback failed: {rollback_error}"
                ) from rollback_error
        raise
    finally:
        staged_report.unlink(missing_ok=True)


def materialize_cohort(
    *,
    declaration_path: Path,
    compiled_dir: Path,
    stage_dir: Path,
    materialized_dir: Path,
    log_dir: Path,
    report_path: Path,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_attestation: Path,
    timeout_seconds: int = DEFAULT_MATERIALIZER_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = _run_process_group,
) -> dict[str, Any]:
    declaration_path = _absolute(declaration_path)
    compiled_dir = _absolute(compiled_dir)
    stage_dir = _absolute(stage_dir)
    materialized_dir = _absolute(materialized_dir)
    log_dir = _absolute(log_dir)
    report_path = _absolute(report_path)
    cm_oracle = _absolute(cm_oracle)
    pmove_oracle = _absolute(pmove_oracle)
    hook_oracle = _absolute(hook_oracle)
    fall_oracle = _absolute(fall_oracle)
    hook_attestation = _absolute(hook_attestation)
    declaration, declaration_sha256 = load_declaration(declaration_path)
    require_unretired_declaration(
        declaration_path, declaration, declaration_sha256
    )
    rows = _map_rows(declaration)
    base = _base_report(
        declaration=declaration,
        declaration_sha256=declaration_sha256,
        rows=rows,
        timeout_seconds=timeout_seconds,
    )
    compiled_projection: Mapping[str, Any] | None = None
    materialized_projection: Mapping[str, Any] | None = None
    authorities: Mapping[str, Any] = {}
    attempted_count = 0
    pass_count = 0
    current_map: Mapping[str, Any] | None = None

    try:
        _require_fresh_topology(
            compiled_dir=compiled_dir,
            stage_dir=stage_dir,
            materialized_dir=materialized_dir,
            log_dir=log_dir,
            report_path=report_path,
        )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= MAX_MATERIALIZER_TIMEOUT_SECONDS
        ):
            raise MaterializeCohortError(
                "timeout-preflight",
                "materializer timeout must be an integer from 1 through 3600 seconds",
            )
        log_dir.mkdir(parents=True, exist_ok=False)
        _fsync_directory(log_dir.parent)
        compiled_membership = verify_stage_membership(
            declaration, compiled_dir, "compiled"
        )
        compiled_projection = _membership_projection(compiled_membership)
        if (
            compiled_membership["passed"] is not True
            or compiled_membership["expected_file_count"] != COMPILED_FILE_COUNT
            or compiled_membership["actual_file_count"] != COMPILED_FILE_COUNT
        ):
            raise MaterializeCohortError(
                "compiled-membership", "compiled stage is not the exact 168-file declaration"
            )

        authorities = preflight_authority_records(
            cm_oracle=cm_oracle,
            pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle,
            fall_oracle=fall_oracle,
            hook_attestation=hook_attestation,
        )
        authority_sha256 = {
            name: record["actual_sha256"]
            for name, record in authorities.items()
        }
        compiled_files = {
            row["map"]: row["files"] for row in compiled_membership["maps"]
        }
        stage_dir.mkdir(parents=False, exist_ok=False)
        for declared in declaration["maps"]:
            for suffix in STAGE_SUFFIXES["compiled"]:
                _copy_exclusive(
                    compiled_dir / f"{declared['map']}{suffix}",
                    stage_dir / f"{declared['map']}{suffix}",
                )
        _fsync_directory(stage_dir)
        staged_compiled = verify_stage_membership(
            declaration, stage_dir, "compiled"
        )
        if staged_compiled != compiled_membership:
            raise MaterializeCohortError(
                "compiled-copy", "unpublished compiled copy differs from its input"
            )

        materializer = ROOT / "tools/materialize_hook_claims.py"
        for declared, row in zip(declaration["maps"], rows):
            current_map = declared
            attempted_count += 1
            map_id = declared["map"]
            attestation = stage_dir / f"{map_id}.hook-materialization.json"
            runtime_sidecar = stage_dir / f"{map_id}.json"
            command = [
                sys.executable,
                str(materializer),
                "--bsp",
                str(stage_dir / f"{map_id}.bsp"),
                "--meta",
                str(stage_dir / f"{map_id}.meta.json"),
                "--runtime-sidecar",
                str(runtime_sidecar),
                "--output-attestation",
                str(attestation),
                "--cm-oracle",
                str(cm_oracle),
                "--pmove-oracle",
                str(pmove_oracle),
                "--hook-oracle",
                str(hook_oracle),
                "--fall-oracle",
                str(fall_oracle),
                "--hook-parity-attestation",
                str(hook_attestation),
            ]
            stdout = b""
            stderr = b""
            returncode: int | None = None
            launch_error: BaseException | None = None
            try:
                completed = runner(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=timeout_seconds,
                )
                stdout = completed.stdout
                stderr = completed.stderr
                returncode = completed.returncode
            except subprocess.TimeoutExpired as error:
                timeout_message = (
                    f"materializer timed out after {timeout_seconds} seconds\n"
                ).encode("ascii")
                stdout = error.stdout if isinstance(error.stdout, bytes) else b""
                captured_stderr = (
                    error.stderr if isinstance(error.stderr, bytes) else b""
                )
                stderr = captured_stderr + timeout_message
                launch_error = MaterializeCohortError(
                    "map-materialization", timeout_message.decode("ascii").strip()
                )
            except BaseException as error:
                launch_error = error
                stderr = (
                    f"materializer launch failed: {_canonical_error(error)}\n"
                ).encode("utf-8")

            prefix = f"{declared['ordinal']:04d}-{map_id}.materialize"
            stdout_path = log_dir / f"{prefix}.stdout.json"
            stderr_path = log_dir / f"{prefix}.stderr.log"
            _exclusive_write(stdout_path, stdout)
            _exclusive_write(stderr_path, stderr)
            row["returncode"] = returncode
            row["stdout"] = _log_record(stdout_path, stdout)
            row["stderr"] = _log_record(stderr_path, stderr)

            if launch_error is not None:
                row["status"] = "failed"
                row["error"] = _canonical_error(launch_error)
                if isinstance(launch_error, MaterializeCohortError):
                    raise launch_error
                raise MaterializeCohortError(
                    "map-materialization", f"{map_id} materializer could not launch"
                ) from launch_error
            if returncode != 0:
                row["status"] = "failed"
                row["error"] = f"materializer exited {returncode}"
                raise MaterializeCohortError(
                    "map-materialization", f"{map_id} materializer exited {returncode}"
                )
            try:
                row["result_sha256"] = _validate_result(
                    stdout,
                    map_id=map_id,
                    attestation=attestation,
                    runtime_sidecar=runtime_sidecar,
                    bsp=stage_dir / f"{map_id}.bsp",
                    compiled_files=compiled_files[map_id],
                    authority_sha256=authority_sha256,
                )
            except MaterializeCohortError as error:
                row["status"] = "failed"
                row["error"] = _canonical_error(error)
                raise
            row["status"] = "passed"
            pass_count += 1

        # Refuse time-of-check/time-of-use substitution before publication.
        current_map = None
        try:
            final_declaration, final_declaration_sha256 = load_declaration(
                declaration_path
            )
        except GeneratorCohortError as error:
            raise MaterializeCohortError(
                "declaration-finalization",
                f"cohort declaration cannot be reread: {_canonical_error(error)}",
            ) from error
        if (
            final_declaration_sha256 != declaration_sha256
            or final_declaration != declaration
        ):
            raise MaterializeCohortError(
                "declaration-finalization",
                "cohort declaration changed during materialization",
            )
        final_authorities = preflight_authority_records(
            cm_oracle=cm_oracle,
            pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle,
            fall_oracle=fall_oracle,
            hook_attestation=hook_attestation,
        )
        if final_authorities != authorities:
            raise MaterializeCohortError(
                "authority-finalization", "authority records changed before publication"
            )
        final_compiled_membership = verify_stage_membership(
            declaration, compiled_dir, "compiled"
        )
        if final_compiled_membership != compiled_membership:
            raise MaterializeCohortError(
                "compiled-finalization",
                "compiled stage changed during materialization",
            )
        _require_immutable_compiled_inputs(
            declaration, compiled_membership, stage_dir
        )
        materialized_membership = verify_stage_membership(
            declaration, stage_dir, "materialized"
        )
        materialized_projection = _membership_projection(
            materialized_membership
        )
        if (
            materialized_membership["passed"] is not True
            or materialized_membership["expected_file_count"]
            != MATERIALIZED_FILE_COUNT
            or materialized_membership["actual_file_count"]
            != MATERIALIZED_FILE_COUNT
        ):
            raise MaterializeCohortError(
                "materialized-membership",
                "unpublished materialized stage is not the exact 196-file declaration",
            )
        _fsync_directory(stage_dir)
        success = _complete_report(
            base,
            phase="complete",
            passed=True,
            compiled_membership=compiled_projection,
            materialized_membership=materialized_projection,
            authorities=authorities,
            attempted_count=attempted_count,
            pass_count=pass_count,
            materialized_published=True,
            failure=None,
        )
        _atomic_publish_with_report(
            stage_dir=stage_dir,
            materialized_dir=materialized_dir,
            report_path=report_path,
            report_payload=canonical_bytes(success),
        )
        return success
    except BaseException as error:
        phase = (
            error.phase
            if isinstance(error, MaterializeCohortError)
            else "unexpected-failure"
        )
        failure = {
            "phase": phase,
            "map": None if current_map is None else current_map["map"],
            "ordinal": None if current_map is None else current_map["ordinal"],
            "error": _canonical_error(error),
        }
        failed = _complete_report(
            base,
            phase=phase,
            passed=False,
            compiled_membership=compiled_projection,
            materialized_membership=materialized_projection,
            authorities=authorities,
            attempted_count=attempted_count,
            pass_count=pass_count,
            materialized_published=False,
            failure=failure,
        )
        if _can_publish_failure_report(
            report_path, compiled_dir, stage_dir, materialized_dir, log_dir
        ):
            _exclusive_write(report_path, canonical_bytes(failed))
        if isinstance(error, MaterializeCohortError):
            raise
        raise MaterializeCohortError(phase, _canonical_error(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--materialized-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-parity-attestation", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_MATERIALIZER_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    try:
        report = materialize_cohort(
            declaration_path=args.declaration,
            compiled_dir=args.compiled_dir,
            stage_dir=args.stage_dir,
            materialized_dir=args.materialized_dir,
            log_dir=args.log_dir,
            report_path=args.report,
            cm_oracle=args.cm_oracle,
            pmove_oracle=args.pmove_oracle,
            hook_oracle=args.hook_oracle,
            fall_oracle=args.fall_oracle,
            hook_attestation=args.hook_parity_attestation,
            timeout_seconds=args.timeout_seconds,
        )
    except (GeneratorCohortError, MaterializeCohortError, OSError) as error:
        print(f"generated cohort materialization failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
