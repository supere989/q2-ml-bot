#!/usr/bin/env python3
"""Compile a disposable B2 qualification population with real q2tool.

This producer accepts only qualification-native declarations and source-stage
reports.  Its compiled bytes and report are retryable, explicitly
non-admissible evidence; they can never serve as a final-cohort stage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.ibsp38 import BspValidationError, parse_ibsp38  # noqa: E402
from tools.b2_qualification_toolchain import (  # noqa: E402
    ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
    Q2TOOL_FLAGS,
    ToolchainAuthority,
    ToolchainAuthorityError,
    inspect_baseq2_assets,
    inspect_q2tool,
    load_toolchain_authority,
)
from tools.assemble_b2_qualification import (  # noqa: E402
    DECLARATION_SCHEMA,
    EXPECTED_MAP_COUNT,
    STAGE_SCHEMA,
)
from tools.run_generator_cohort import (  # noqa: E402
    canonical_bytes,
    repository_binding,
)


SOURCE_SUFFIXES = (".map", ".json", ".meta.json", ".lattice.json", ".routes.json")
COMPILED_SUFFIXES = (*SOURCE_SUFFIXES, ".bsp")
CONCRETE_STYLES = (
    "open", "towers", "canyon", "pits", "arena_open", "arena_vertical",
    "arena_lanes",
)
DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_JOBS = min(4, os.cpu_count() or 1)
MAX_JOBS = 8
AT_FDCWD = -100
RENAME_NOREPLACE = 1
TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class QualificationCompileError(RuntimeError):
    """Qualification compilation failed without publishing a partial stage."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise QualificationCompileError(f"cannot stat {path}: {error}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise QualificationCompileError(f"required regular file is absent: {path}")
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationCompileError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                QualificationCompileError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationCompileError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise QualificationCompileError(f"JSON must be an object: {path}")
    document = dict(value)
    if raw != canonical_bytes(document):
        raise QualificationCompileError(f"JSON is not canonical: {path}")
    return document, raw


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise QualificationCompileError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _qualification_path(path: Path, label: str) -> Path:
    result = path.expanduser().absolute()
    lowered = [part.lower() for part in result.parts]
    if any("retired" in part or "generated-final" in part for part in lowered):
        raise QualificationCompileError(f"{label} is under a retired/final path")
    return result


def _overlap(first: Path, second: Path) -> bool:
    left = first.resolve(strict=False)
    right = second.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise QualificationCompileError("renameat2 is required for atomic publication")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination),
        RENAME_NOREPLACE,
    ) != 0:
        error = ctypes.get_errno()
        raise QualificationCompileError(
            f"atomic publish failed: {os.strerror(error)}"
        )
    _fsync_directory(source.parent)
    if source.parent != destination.parent:
        _fsync_directory(destination.parent)


def _current_implementation(repo_root: Path) -> dict[str, Any]:
    return repository_binding(repo_root)


def _validate_implementation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationCompileError("implementation binding must be an object")
    binding = dict(value)
    _exact_keys(
        binding,
        {
            "repository_commit", "repository_tree", "git_clean",
            "atlas_analyzer_authority_sha256",
            "atlas_analyzer_authority_file_count", "generator_sha256",
            "routes_sha256",
        },
        "implementation binding",
    )
    if (
        binding["git_clean"] is not True
        or not isinstance(binding["repository_commit"], str)
        or not isinstance(binding["repository_tree"], str)
        or re.fullmatch(r"[0-9a-f]{40}", binding["repository_commit"]) is None
        or re.fullmatch(r"[0-9a-f]{40}", binding["repository_tree"]) is None
        or not isinstance(binding["atlas_analyzer_authority_file_count"], int)
        or isinstance(binding["atlas_analyzer_authority_file_count"], bool)
        or binding["atlas_analyzer_authority_file_count"] < 1
        or any(
            not isinstance(binding[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", binding[name]) is None
            for name in (
                "atlas_analyzer_authority_sha256", "generator_sha256",
                "routes_sha256",
            )
        )
    ):
        raise QualificationCompileError("implementation binding is malformed or dirty")
    return binding


def _validate_declaration(
    path: Path, implementation: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, str]:
    declaration, raw = _load_json(path)
    _exact_keys(
        declaration,
        {
            "schema", "qualification_id", "mode", "non_admissible",
            "retryable", "final_cohort_authorized", "generator", "selection",
            "implementation", "toolchain_authority_sha256", "maps",
        },
        "qualification declaration",
    )
    if (
        declaration["schema"] != DECLARATION_SCHEMA
        or declaration["mode"] != "qualification"
        or declaration["non_admissible"] is not True
        or declaration["retryable"] is not True
        or declaration["final_cohort_authorized"] is not False
    ):
        raise QualificationCompileError("final-mode or admissible declaration rejected")
    qualification_id = declaration["qualification_id"]
    if (
        not isinstance(qualification_id, str)
        or TOKEN.fullmatch(qualification_id) is None
        or not qualification_id.startswith("b2q26_")
        or "final" in qualification_id
    ):
        raise QualificationCompileError("qualification ID is invalid or final-mode")
    if declaration["implementation"] != implementation:
        raise QualificationCompileError("declaration implementation binding differs")
    if (
        declaration["toolchain_authority_sha256"]
        != ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256
    ):
        raise QualificationCompileError(
            "declaration toolchain authority binding differs"
        )
    generator = declaration["generator"]
    if generator != {"version": "v6", "grid": 5, "gym": False, "observed_heat": None}:
        raise QualificationCompileError("qualification generator contract differs")
    selection = declaration["selection"]
    if selection != {
        "required_map_count": 28,
        "required_concrete_styles": list(CONCRETE_STYLES),
        "required_maps_per_style": 4,
    }:
        raise QualificationCompileError("qualification selection differs")
    maps = declaration["maps"]
    if not isinstance(maps, list) or len(maps) != EXPECTED_MAP_COUNT:
        raise QualificationCompileError("qualification declaration must contain 28 maps")
    names: set[str] = set()
    seeds: set[int] = set()
    style_counts = {style: 0 for style in CONCRETE_STYLES}
    for ordinal, item in enumerate(maps):
        if not isinstance(item, Mapping):
            raise QualificationCompileError("qualification map row must be an object")
        _exact_keys(
            item, {"ordinal", "map", "seed", "style", "grid", "observed_heat"},
            f"qualification map {ordinal}",
        )
        name = item["map"]
        seed = item["seed"]
        style = item["style"]
        if (
            item["ordinal"] != ordinal
            or not isinstance(name, str)
            or TOKEN.fullmatch(name) is None
            or not name.startswith("b2q26_")
            or "final" in name
            or name in names
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
            or seed in seeds
            or style not in style_counts
            or item["grid"] != 5
            or item["observed_heat"] is not None
        ):
            raise QualificationCompileError(f"qualification map {ordinal} differs")
        names.add(name)
        seeds.add(seed)
        style_counts[style] += 1
    if style_counts != {style: 4 for style in CONCRETE_STYLES}:
        raise QualificationCompileError("qualification style balance differs")
    return declaration, raw, _sha256(raw)


def _flat_records(
    declaration: Mapping[str, Any], root: Path, suffixes: Sequence[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    expected = {f"{row['map']}{suffix}" for row in declaration["maps"] for suffix in suffixes}
    if root.is_symlink() or not root.is_dir():
        raise QualificationCompileError(f"stage root is absent or a symlink: {root}")
    actual: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise QualificationCompileError(f"stage contains non-regular entry: {path.name}")
        actual.add(path.name)
    if actual != expected:
        raise QualificationCompileError(
            f"stage membership differs; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return {
        str(row["map"]): {
            suffix: _file_record(root / f"{row['map']}{suffix}")
            for suffix in suffixes
        }
        for row in declaration["maps"]
    }


def _sparse_compiled_records(
    declaration: Mapping[str, Any], root: Path, passed: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    declared = {str(row["map"]) for row in declaration["maps"]}
    if not passed.issubset(declared):
        raise QualificationCompileError("sparse compile set contains undeclared maps")
    expected = {
        f"{row['map']}{suffix}"
        for row in declaration["maps"] for suffix in SOURCE_SUFFIXES
    } | {f"{map_id}.bsp" for map_id in passed}
    if root.is_symlink() or not root.is_dir():
        raise QualificationCompileError(
            f"sparse compiled root is absent or a symlink: {root}"
        )
    actual: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise QualificationCompileError(
                f"sparse compiled root contains non-regular entry: {path.name}"
            )
        actual.add(path.name)
    if actual != expected:
        raise QualificationCompileError(
            "sparse compiled membership differs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return {
        str(row["map"]): {
            suffix: _file_record(root / f"{row['map']}{suffix}")
            for suffix in (
                COMPILED_SUFFIXES
                if row["map"] in passed else SOURCE_SUFFIXES
            )
        }
        for row in declaration["maps"]
    }


def _source_evidence_sha256(map_id: str, files: Mapping[str, Any]) -> str:
    return _sha256(canonical_bytes({"map": map_id, "files": dict(files)}))


def _validate_source_report(
    path: Path,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes, str]:
    report, raw = _load_json(path)
    _exact_keys(
        report,
        {
            "schema", "qualification_id", "mode", "stage", "non_admissible",
            "retryable", "final_cohort_authorized", "declaration_sha256",
            "implementation", "toolchain_authority_sha256",
            "input_report_sha256", "infrastructure_checks",
            "map_count", "pass_count", "maps", "failures",
        },
        "qualification source report",
    )
    if (
        report["schema"] != STAGE_SCHEMA
        or report["qualification_id"] != declaration["qualification_id"]
        or report["mode"] != "qualification"
        or report["stage"] != "source"
        or report["non_admissible"] is not True
        or report["retryable"] is not True
        or report["final_cohort_authorized"] is not False
        or report["declaration_sha256"] != declaration_sha256
        or report["implementation"] != implementation
        or report["toolchain_authority_sha256"]
        != declaration["toolchain_authority_sha256"]
        or report["input_report_sha256"] is not None
    ):
        raise QualificationCompileError("source-stage identity or binding differs")
    infrastructure = report["infrastructure_checks"]
    required_checks = {
        "source-static", "deterministic-cold-rebuild", "exact-membership",
        "bounded-parallel-workers", "input-stability",
    }
    if (
        not isinstance(infrastructure, Mapping)
        or set(infrastructure) != required_checks
        or not all(value is True for value in infrastructure.values())
        or report["failures"] != []
    ):
        raise QualificationCompileError("source-stage infrastructure is not green")
    rows = report["maps"]
    if (
        not isinstance(rows, list)
        or report["map_count"] != 28
        or report["pass_count"] != 28
        or len(rows) != 28
    ):
        raise QualificationCompileError("source-stage population is not 28/28")
    for declared, row in zip(declaration["maps"], rows):
        if not isinstance(row, Mapping):
            raise QualificationCompileError("source-stage row is not an object")
        _exact_keys(
            row, {"ordinal", "map", "criteria", "evidence_sha256", "failures", "passed"},
            "source-stage map row",
        )
        map_id = declared["map"]
        if (
            row["ordinal"] != declared["ordinal"]
            or row["map"] != map_id
            or not isinstance(row["criteria"], Mapping)
            or set(row["criteria"]) != {
                "source-files-complete", "cold-bytes-identical",
                "metadata-contract", "source-static", "route-contract",
                "spawn-origin-binding", "layout-unique",
            }
            or not all(value is True for value in row["criteria"].values())
            or row["failures"] != []
            or row["passed"] is not True
            or row["evidence_sha256"]
            != _source_evidence_sha256(map_id, source_records[map_id])
        ):
            raise QualificationCompileError(f"source-stage evidence differs for {map_id}")
    return report, raw, _sha256(raw)


def _inspect_basedir(
    basedir: Path, authority: ToolchainAuthority,
) -> dict[str, Any]:
    try:
        return inspect_baseq2_assets(basedir, authority)
    except ToolchainAuthorityError as error:
        raise QualificationCompileError(str(error)) from error


def _bsp_record(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    try:
        metadata = parse_ibsp38(path)
    except BspValidationError as error:
        raise QualificationCompileError(f"invalid IBSP-38 {path.name}: {error}") from error
    lighting = next(lump for lump in metadata.lumps if lump.name == "lighting")
    payload = path.read_bytes()[lighting.offset:lighting.offset + lighting.length]
    if not payload or len(payload) != metadata.lightmaps.byte_count:
        raise QualificationCompileError(f"compiled lightdata is absent: {path.name}")
    return {
        **record,
        "ibsp_version": metadata.version,
        "lightdata": {"bytes": len(payload), "sha256": _sha256(payload)},
    }


def _compile_one(
    row: Mapping[str, Any], staging: Path, logs: Path, q2tool: Path,
    basedir: Path, timeout_seconds: float,
) -> dict[str, Any]:
    ordinal = int(row["ordinal"])
    map_id = str(row["map"])
    stdout_path = logs / f"{ordinal:03d}-{map_id}.stdout.log"
    stderr_path = logs / f"{ordinal:03d}-{map_id}.stderr.log"
    map_path = staging / f"{map_id}.map"
    bsp_path = staging / f"{map_id}.bsp"
    command = [str(q2tool), *Q2TOOL_FLAGS, str(basedir), str(map_path)]
    exit_code: int | None = None
    timed_out = False
    invocation_error: str | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                command, cwd=staging, stdin=subprocess.DEVNULL, stdout=stdout,
                stderr=stderr, start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait()
        except OSError as error:
            invocation_error = f"{type(error).__name__}: {error}"
    failures = []
    bsp = None
    if timed_out:
        failures.append(f"q2tool exceeded {timeout_seconds:g}-second timeout")
    if invocation_error:
        failures.append(invocation_error)
    if exit_code != 0:
        failures.append(f"q2tool exit code {exit_code}")
    if not failures:
        try:
            bsp = _bsp_record(bsp_path)
        except QualificationCompileError as error:
            failures.append(str(error))
    prt = staging / f"{map_id}.prt"
    if prt.exists() or prt.is_symlink():
        if prt.is_symlink() or not prt.is_file():
            failures.append("q2tool PRT is not a regular file")
        else:
            prt.unlink()
    return {
        "ordinal": ordinal,
        "map": map_id,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "invocation_error": invocation_error,
        "stdout": _file_record(stdout_path),
        "stderr": _file_record(stderr_path),
        "bsp": bsp,
        "failures": failures,
        "compile_passed": not failures,
    }


def _map_stage_row(
    result: Mapping[str, Any], source_files: Mapping[str, Any], published: bool,
    logs: Path, q2tool: Mapping[str, Any], basedir: Mapping[str, Any],
    jobs: int, timeout_seconds: float,
) -> dict[str, Any]:
    map_id = str(result["map"])
    criteria = {
        "source-stage-bound": True,
        "q2tool-exit-zero": result["exit_code"] == 0,
        "q2tool-not-timed-out": result["timed_out"] is False,
        "ibsp38-lightdata": result["bsp"] is not None,
        "compiled-stage-published": published and result["compile_passed"] is True,
    }
    failures = list(result["failures"])
    if not published:
        failures.append("sparse compiled population was not published")
    evidence = {
        "schema": "q2-b2-qualification-compile-map-evidence-v1",
        "ordinal": result["ordinal"],
        "map": map_id,
        "source_files": dict(source_files),
        "q2tool": dict(q2tool),
        "basedir": dict(basedir),
        "execution": {
            "parallel_worker_limit": jobs,
            "q2tool_threads": 1,
            "per_map_timeout_milliseconds": round(timeout_seconds * 1000),
        },
        "command": result["command"],
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
        "invocation_error": result["invocation_error"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "bsp": result["bsp"],
        "criteria": criteria,
        "failures": failures,
        "passed": all(criteria.values()) and not failures,
    }
    payload = canonical_bytes(evidence)
    evidence_path = logs / f"{int(result['ordinal']):03d}-{map_id}.evidence.json"
    _exclusive_write(evidence_path, payload)
    return {
        "ordinal": result["ordinal"],
        "map": map_id,
        "criteria": criteria,
        "evidence_sha256": _sha256(payload),
        "failures": failures,
        "passed": evidence["passed"],
    }


def _failure_result(row: Mapping[str, Any], message: str) -> dict[str, Any]:
    return {
        "ordinal": row["ordinal"], "map": row["map"], "command": [],
        "exit_code": None, "timed_out": False, "invocation_error": message,
        "stdout": {"bytes": 0, "sha256": _sha256(b"")},
        "stderr": {"bytes": 0, "sha256": _sha256(b"")}, "bsp": None,
        "failures": [message], "compile_passed": False,
    }


def compile_qualification(
    *, declaration_path: Path, source_report_path: Path, source_root: Path,
    staging_root: Path, compiled_root: Path, log_root: Path, report_path: Path,
    q2tool: Path, basedir: Path, jobs: int = DEFAULT_JOBS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = _current_implementation,
    authority_provider: Callable[[Path], ToolchainAuthority] = load_toolchain_authority,
) -> dict[str, Any]:
    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path, "source_report": source_report_path,
            "source_root": source_root, "staging_root": staging_root,
            "compiled_root": compiled_root, "log_root": log_root,
            "report": report_path, "q2tool": q2tool, "basedir": basedir,
        }.items()
    }
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= MAX_JOBS:
        raise QualificationCompileError(f"jobs must be in [1, {MAX_JOBS}]")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 86400
    ):
        raise QualificationCompileError("timeout must be finite and in (0, 86400]")
    timeout_seconds = float(timeout_seconds)
    for output in (paths["staging_root"], paths["compiled_root"], paths["log_root"], paths["report"]):
        if output.exists() or output.is_symlink():
            raise QualificationCompileError(f"output must be fresh: {output}")
        if not output.parent.is_dir():
            raise QualificationCompileError(f"output parent must exist: {output.parent}")
    roots = [
        paths[name] for name in (
            "source_root", "staging_root", "compiled_root", "log_root"
        )
    ]
    if any(_overlap(left, right) for index, left in enumerate(roots) for right in roots[index + 1:]):
        raise QualificationCompileError("source and output roots must be disjoint")
    if any(_overlap(paths["report"], root) for root in roots):
        raise QualificationCompileError("report must be outside source/output roots")
    if paths["staging_root"].parent.stat().st_dev != paths["compiled_root"].parent.stat().st_dev:
        raise QualificationCompileError("staging and compiled roots must share a filesystem")

    initial_implementation = _validate_implementation(
        implementation_provider(repo_root)
    )
    try:
        toolchain_authority = authority_provider(repo_root)
    except ToolchainAuthorityError as error:
        raise QualificationCompileError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    declaration, declaration_raw, declaration_sha256 = _validate_declaration(
        paths["declaration"], initial_implementation
    )
    source_records = _flat_records(declaration, paths["source_root"], SOURCE_SUFFIXES)
    _, source_report_raw, source_report_sha256 = _validate_source_report(
        paths["source_report"], declaration, declaration_sha256,
        initial_implementation, source_records,
    )
    try:
        inspect_q2tool(paths["q2tool"], toolchain_authority)
    except ToolchainAuthorityError as error:
        raise QualificationCompileError(str(error)) from error
    q2tool_record = _file_record(paths["q2tool"])
    if not os.access(paths["q2tool"], os.X_OK):
        raise QualificationCompileError("q2tool is not executable")
    basedir_record = _inspect_basedir(paths["basedir"], toolchain_authority)

    paths["staging_root"].mkdir(mode=0o755)
    paths["log_root"].mkdir(mode=0o755)
    for row in declaration["maps"]:
        for suffix in SOURCE_SUFFIXES:
            name = f"{row['map']}{suffix}"
            shutil.copyfile(paths["source_root"] / name, paths["staging_root"] / name)
    if _flat_records(declaration, paths["staging_root"], SOURCE_SUFFIXES) != source_records:
        raise QualificationCompileError("staged source bytes differ")

    results_by_ordinal: dict[int, dict[str, Any]] = {}
    compile_error: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _compile_one, row, paths["staging_root"], paths["log_root"],
                    paths["q2tool"], paths["basedir"], timeout_seconds,
                ): int(row["ordinal"])
                for row in declaration["maps"]
            }
            for future in as_completed(futures):
                ordinal = futures[future]
                try:
                    results_by_ordinal[ordinal] = future.result()
                except Exception as error:  # defensive worker boundary
                    results_by_ordinal[ordinal] = _failure_result(
                        declaration["maps"][ordinal], f"worker {type(error).__name__}: {error}"
                    )
        results = [results_by_ordinal[index] for index in range(28)]
        systemic_failures = [
            f"{result['map']}: {message}"
            for result in results
            if result["timed_out"] is True or result["invocation_error"] is not None
            for message in result["failures"]
        ]
        if systemic_failures:
            compile_error = "; ".join(systemic_failures)
        passed_maps = {
            str(result["map"])
            for result in results if result["compile_passed"] is True
        }
        if compile_error is None:
            for result in results:
                if result["compile_passed"] is False:
                    (paths["staging_root"] / f"{result['map']}.bsp").unlink(
                        missing_ok=True
                    )
            try:
                compiled_records = _sparse_compiled_records(
                    declaration, paths["staging_root"], passed_maps
                )
                if any(
                    compiled_records[map_id][suffix]
                    != source_records[map_id][suffix]
                    for map_id in source_records for suffix in SOURCE_SUFFIXES
                ):
                    compile_error = "q2tool changed source artifacts"
            except QualificationCompileError as error:
                compile_error = str(error)
        if compile_error is None:
            try:
                stable = (
                    paths["declaration"].read_bytes() == declaration_raw
                    and paths["source_report"].read_bytes() == source_report_raw
                    and _flat_records(
                        declaration, paths["source_root"], SOURCE_SUFFIXES
                    ) == source_records
                    and _file_record(paths["q2tool"]) == q2tool_record
                    and _inspect_basedir(
                        paths["basedir"], toolchain_authority
                    ) == basedir_record
                    and authority_provider(repo_root) == toolchain_authority
                    and _validate_implementation(
                        implementation_provider(repo_root)
                    ) == initial_implementation
                )
            except (OSError, QualificationCompileError) as error:
                compile_error = f"input stability check failed: {error}"
            else:
                if not stable:
                    compile_error = (
                        "declaration/tool/source/implementation input changed "
                        "during compile"
                    )
        published = False
        if compile_error is None:
            _rename_noreplace(paths["staging_root"], paths["compiled_root"])
            published = True
        try:
            rows = [
                _map_stage_row(
                    result, source_records[str(result["map"])], published,
                    paths["log_root"], q2tool_record, basedir_record, jobs,
                    timeout_seconds,
                )
                for result in results
            ]
            report = {
                "schema": STAGE_SCHEMA,
                "qualification_id": declaration["qualification_id"],
                "mode": "qualification",
                "stage": "compile",
                "non_admissible": True,
                "retryable": True,
                "final_cohort_authorized": False,
                "declaration_sha256": declaration_sha256,
                "implementation": initial_implementation,
                "toolchain_authority_sha256": (
                    toolchain_authority.manifest_sha256
                ),
                "input_report_sha256": source_report_sha256,
                "infrastructure_checks": {
                    "real-q2tool": not systemic_failures,
                    "compiled-membership": published,
                    "bounded-parallel-workers": 1 <= jobs <= MAX_JOBS,
                    "input-stability": compile_error is None,
                    "ibsp38-validation-complete": not systemic_failures,
                    "exclusive-publication": published,
                },
                "map_count": 28,
                "pass_count": sum(row["passed"] is True for row in rows),
                "maps": rows,
                "failures": [] if published else [
                    compile_error or "sparse compiled population not published"
                ],
            }
            payload = canonical_bytes(report)
            _exclusive_write(paths["report"], payload)
        except Exception as error:
            if published:
                _rename_noreplace(paths["compiled_root"], paths["staging_root"])
            if isinstance(error, QualificationCompileError):
                raise
            raise QualificationCompileError(
                f"report/evidence publication failed: {error}"
            ) from error
        if not published:
            raise QualificationCompileError(report["failures"][0])
        return report
    except QualificationCompileError:
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compile_qualification(
            declaration_path=args.declaration,
            source_report_path=args.source_report,
            source_root=args.source_root,
            staging_root=args.staging_root,
            compiled_root=args.compiled_root,
            log_root=args.log_root,
            report_path=args.report,
            q2tool=args.q2tool,
            basedir=args.basedir,
            jobs=args.jobs,
            timeout_seconds=args.timeout_seconds,
        )
    except QualificationCompileError as error:
        print(f"B2 qualification compile failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes({
        "schema": STAGE_SCHEMA,
        "qualification_id": report["qualification_id"],
        "stage": "compile",
        "report": str(args.report.expanduser().absolute()),
        "report_sha256": _file_sha256(args.report),
        "map_count": report["map_count"],
        "pass_count": report["pass_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
