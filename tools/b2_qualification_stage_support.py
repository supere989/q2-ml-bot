#!/usr/bin/env python3
"""Shared fail-closed support for disposable B2 qualification producers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from collections import Counter
from typing import Any, Mapping, Sequence

from tools.assemble_b2_qualification import (
    DECLARATION_SCHEMA,
    EXPECTED_MAP_COUNT,
    STAGE_SCHEMA,
)
from tools.run_generator_cohort import CONCRETE_STYLES, canonical_bytes, repository_binding


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
TOKEN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
IMPLEMENTATION_KEYS = {
    "repository_commit", "repository_tree", "git_clean",
    "atlas_analyzer_authority_sha256", "atlas_analyzer_authority_file_count",
    "generator_sha256", "routes_sha256",
}
REPORT_KEYS = {
    "schema", "qualification_id", "mode", "stage", "non_admissible",
    "retryable", "final_cohort_authorized", "declaration_sha256",
    "implementation", "input_report_sha256", "infrastructure_checks",
    "map_count", "pass_count", "maps", "failures",
}
ROW_KEYS = {
    "ordinal", "map", "criteria", "evidence_sha256", "failures", "passed",
}
AT_FDCWD = -100
RENAME_NOREPLACE = 1
PINNED_PYTHON_VERSION = [3, 11, 4]
PINNED_PYTHON_SHA256 = "b25abf001748dc7ebb4b25013b2572d4e6913246b4c3b8e8b726b3da45494ff4"


class QualificationStageError(RuntimeError):
    """A qualification input or publication cannot be proved safe."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationStageError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise QualificationStageError(f"cannot stat {path}: {error}") from error
    require(stat.S_ISREG(mode) and not path.is_symlink(), f"not a regular file: {path}")
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def pinned_runtime_record() -> dict[str, Any]:
    record = file_record(Path(sys.executable).resolve())
    require(list(sys.version_info[:3]) == PINNED_PYTHON_VERSION,
            "producer is not running under pinned CPython 3.11.4")
    require(record["sha256"] == PINNED_PYTHON_SHA256,
            "producer interpreter bytes differ from pinned WSL runtime")
    return {**record, "executable": str(Path(sys.executable).resolve()),
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3])}


def _duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                QualificationStageError(f"non-finite JSON token {token}")
            ),
        )
    except QualificationStageError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationStageError(f"cannot load canonical JSON {path}: {error}") from error
    require(isinstance(value, Mapping), f"JSON must be an object: {path}")
    document = dict(value)
    require(raw == canonical_bytes(document), f"JSON is not canonical: {path}")
    return document, raw


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        f"{label} keys differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )


def qualification_path(path: Path, label: str) -> Path:
    result = path.expanduser().absolute()
    lowered = [part.lower() for part in result.parts]
    require(
        not any("retired" in part or "generated-final" in part for part in lowered),
        f"{label} is under a retired/final path",
    )
    return result


def validate_implementation(value: object) -> dict[str, Any]:
    require(isinstance(value, Mapping), "implementation must be an object")
    result = dict(value)
    exact_keys(result, IMPLEMENTATION_KEYS, "implementation")
    require(result["git_clean"] is True, "implementation must be clean")
    require(
        all(isinstance(result[name], str) and HEX40.fullmatch(result[name]) for name in (
            "repository_commit", "repository_tree",
        )),
        "implementation git identities are malformed",
    )
    require(
        all(isinstance(result[name], str) and HEX64.fullmatch(result[name]) for name in (
            "atlas_analyzer_authority_sha256", "generator_sha256", "routes_sha256",
        )),
        "implementation authority identities are malformed",
    )
    count = result["atlas_analyzer_authority_file_count"]
    require(isinstance(count, int) and not isinstance(count, bool) and count >= 1,
            "implementation authority file count is malformed")
    return result


def current_implementation(repo_root: Path) -> dict[str, Any]:
    return validate_implementation(repository_binding(repo_root))


def validate_declaration(
    path: Path, implementation: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, str]:
    path = qualification_path(path, "declaration")
    declaration, raw = load_canonical(path)
    exact_keys(declaration, {
        "schema", "qualification_id", "mode", "non_admissible", "retryable",
        "final_cohort_authorized", "generator", "selection", "implementation", "maps",
    }, "declaration")
    require(declaration["schema"] == DECLARATION_SCHEMA, "declaration schema differs")
    require(declaration["mode"] == "qualification", "final declaration rejected")
    require(declaration["non_admissible"] is True and declaration["retryable"] is True,
            "declaration qualification disposition differs")
    require(declaration["final_cohort_authorized"] is False,
            "declaration authorizes a final cohort")
    require(declaration["implementation"] == implementation,
            "declaration implementation differs from current clean repository")
    qualification_id = declaration["qualification_id"]
    require(isinstance(qualification_id, str) and qualification_id.startswith("b2q26_")
            and TOKEN.fullmatch(qualification_id) and "final" not in qualification_id,
            "qualification ID is malformed or final-mode")
    require(declaration["generator"] == {
        "version": "v6", "grid": 5, "gym": False, "observed_heat": None,
    }, "qualification generator contract differs")
    require(declaration["selection"] == {
        "required_map_count": 28,
        "required_concrete_styles": list(CONCRETE_STYLES),
        "required_maps_per_style": 4,
    }, "qualification selection contract differs")
    maps = declaration["maps"]
    require(isinstance(maps, list) and len(maps) == EXPECTED_MAP_COUNT,
            "declaration must contain exactly 28 maps")
    names: set[str] = set()
    seeds: set[int] = set()
    styles: Counter[str] = Counter()
    for ordinal, item in enumerate(maps):
        require(isinstance(item, Mapping), "declaration map must be an object")
        exact_keys(item, {"ordinal", "map", "seed", "style", "grid", "observed_heat"},
                   f"declaration map {ordinal}")
        require(item.get("ordinal") == ordinal, "declaration ordinal differs")
        name = item.get("map")
        require(isinstance(name, str) and TOKEN.fullmatch(name) and name.startswith("b2q26_")
                and "final" not in name and name not in names,
                "declaration map identity is malformed or duplicated")
        names.add(name)
        seed = item["seed"]
        require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
                and seed not in seeds, "declaration seed is malformed or duplicated")
        seeds.add(seed)
        style = item["style"]
        require(style in CONCRETE_STYLES, "declaration style differs")
        styles[str(style)] += 1
        require(item["grid"] == 5 and item["observed_heat"] is None,
                "declaration generator inputs differ")
    require(styles == Counter({style: 4 for style in CONCRETE_STYLES}),
            "qualification style balance differs")
    return declaration, raw, sha256_bytes(raw)


def validate_stage_report(
    path: Path, expected_stage: str, declaration: Mapping[str, Any],
    declaration_sha256: str, implementation: Mapping[str, Any],
    *, expected_input_sha256: str | None | object = ...,
) -> tuple[dict[str, Any], bytes, str, set[str]]:
    path = qualification_path(path, f"{expected_stage} report")
    report, raw = load_canonical(path)
    exact_keys(report, REPORT_KEYS, f"{expected_stage} report")
    require(report["schema"] == STAGE_SCHEMA and report["stage"] == expected_stage,
            f"{expected_stage} report schema/stage differs")
    require(report["qualification_id"] == declaration["qualification_id"],
            f"{expected_stage} qualification ID differs")
    require(report["mode"] == "qualification", f"{expected_stage} final report rejected")
    require(report["non_admissible"] is True and report["retryable"] is True,
            f"{expected_stage} disposition differs")
    require(report["final_cohort_authorized"] is False,
            f"{expected_stage} report authorizes final cohort")
    require(report["declaration_sha256"] == declaration_sha256,
            f"{expected_stage} declaration binding differs")
    require(report["implementation"] == implementation,
            f"{expected_stage} implementation binding differs")
    if expected_input_sha256 is not ...:
        require(report["input_report_sha256"] == expected_input_sha256,
                f"{expected_stage} input report hash differs")
    else:
        digest = report["input_report_sha256"]
        require(digest is None or isinstance(digest, str) and HEX64.fullmatch(digest),
                f"{expected_stage} input report hash is malformed")
    checks = report["infrastructure_checks"]
    require(isinstance(checks, Mapping) and checks and all(value is True for value in checks.values()),
            f"{expected_stage} infrastructure checks are not green")
    require(report["failures"] == [], f"{expected_stage} has infrastructure failures")
    rows = report["maps"]
    require(isinstance(rows, list) and len(rows) == EXPECTED_MAP_COUNT,
            f"{expected_stage} does not contain 28 rows")
    passed: set[str] = set()
    evidence: set[str] = set()
    for declared, item in zip(declaration["maps"], rows):
        require(isinstance(item, Mapping), f"{expected_stage} row must be an object")
        exact_keys(item, ROW_KEYS, f"{expected_stage} row")
        require(item["ordinal"] == declared["ordinal"] and item["map"] == declared["map"],
                f"{expected_stage} map membership/order differs")
        criteria = item["criteria"]
        failures = item["failures"]
        require(isinstance(criteria, Mapping) and criteria and
                all(isinstance(key, str) and isinstance(value, bool)
                    for key, value in criteria.items()),
                f"{expected_stage} criteria are malformed")
        require(isinstance(failures, list) and
                all(isinstance(value, str) and value for value in failures),
                f"{expected_stage} failures are malformed")
        recomputed = all(criteria.values()) and not failures
        require(item["passed"] is recomputed, f"{expected_stage} row result is self-declared")
        digest = item["evidence_sha256"]
        require(isinstance(digest, str) and HEX64.fullmatch(digest) and digest not in evidence,
                f"{expected_stage} evidence is malformed or reused")
        evidence.add(digest)
        if recomputed:
            passed.add(str(item["map"]))
    require(report["map_count"] == EXPECTED_MAP_COUNT and report["pass_count"] == len(passed),
            f"{expected_stage} counts differ")
    return report, raw, sha256_bytes(raw), passed


def expected_names(maps: Sequence[Mapping[str, Any]], suffixes: Sequence[str]) -> set[str]:
    return {f"{row['map']}{suffix}" for row in maps for suffix in suffixes}


def exact_flat_files(root: Path, expected: set[str], label: str) -> None:
    require(root.is_dir() and not root.is_symlink(), f"{label} root is absent or unsafe")
    actual: set[str] = set()
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"{label} contains non-regular {path.name}")
        actual.add(path.name)
    require(actual == expected,
            f"{label} membership differs; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def membership_records(root: Path, map_id: str, suffixes: Sequence[str]) -> dict[str, Any]:
    return {suffix: file_record(root / f"{map_id}{suffix}") for suffix in suffixes}


def evidence_sha256(map_id: str, files: Mapping[str, Any], **fields: Any) -> str:
    return sha256_bytes(canonical_bytes({"map": map_id, **fields, "files": dict(files)}))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(path.parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD,
                       os.fsencode(destination), RENAME_NOREPLACE)
    if result != 0:
        error = ctypes.get_errno()
        raise QualificationStageError(f"exclusive atomic publication failed: {os.strerror(error)}")
    fsync_directory(destination.parent)


def fresh_outputs(paths: Mapping[str, Path]) -> None:
    for label, path in paths.items():
        require(not path.exists() and not path.is_symlink(), f"{label} output must be fresh: {path}")
        require(path.parent.is_dir(), f"{label} output parent is absent: {path.parent}")


def stage_report(
    *, declaration: Mapping[str, Any], declaration_sha256: str,
    implementation: Mapping[str, Any], stage: str, input_sha256: str,
    checks: Mapping[str, bool], rows: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "input_report_sha256": input_sha256,
        "infrastructure_checks": dict(checks),
        "map_count": len(rows),
        "pass_count": sum(row["passed"] is True for row in rows),
        "maps": rows,
        "failures": [],
    }
