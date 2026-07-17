#!/usr/bin/env python3
"""Produce a disposable B2 qualification declaration and source stage.

This tool is qualification-only.  It generates one balanced 28-map population
and a cold rebuild through the real generator, independently validates the
source/static, route, metadata, and spawn-origin contracts, and emits the
qualification-native declaration and source-stage report consumed by
``assemble_b2_qualification.py``.  It never accepts or emits a final cohort.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assemble_b2_qualification import (  # noqa: E402
    DECLARATION_SCHEMA,
    QUALIFICATION_ID,
    REQUIRED_STAGE_CHECKS,
    STAGE_SCHEMA,
)
from tools.retired_cohort_registry import (  # noqa: E402
    RetiredCohortRegistryError,
    _load_registry,
)
from tools.run_generator_cohort import (  # noqa: E402
    CONCRETE_STYLES,
    MAP_ID_RE,
    SOURCE_SUFFIXES,
    GeneratorCohortError,
    _default_generator,
    _default_static_validator,
    _source_spawn_origin_binding,
    _validate_metadata,
    canonical_bytes,
    repository_binding,
)
from tools.source_route_contract import (  # noqa: E402
    SourceRouteContractError,
    load_source_route_contract,
)


MAP_COUNT = 28
MAX_WORKERS = 8
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMPLEMENTATION_KEYS = {
    "repository_commit",
    "repository_tree",
    "git_clean",
    "atlas_analyzer_authority_sha256",
    "atlas_analyzer_authority_file_count",
    "generator_sha256",
    "routes_sha256",
}


class QualificationSourceError(RuntimeError):
    """The disposable source producer failed before admissible publication."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationSourceError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exclusive_write(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise QualificationSourceError(f"output parent is absent or invalid: {path.parent}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_qualification_path(path: Path, label: str, repo_root: Path) -> None:
    absolute = _absolute(path)
    _require(not _within(absolute, repo_root), f"{label} must be outside the repository")
    for part in absolute.parts:
        lowered = part.lower()
        if (
            lowered in {"final", "retired"}
            or lowered.startswith("generated-final")
            or lowered.startswith("retired-")
            or lowered.endswith("-retired")
        ):
            raise QualificationSourceError(
                f"{label} is under a final/retired artifact path"
            )


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    for index, (first_name, first) in enumerate(items):
        for second_name, second in items[index + 1:]:
            _require(first != second, f"{first_name} and {second_name} paths coincide")
            if first_name.endswith("root") and second_name.endswith("root"):
                _require(
                    not _within(first, second) and not _within(second, first),
                    f"{first_name} and {second_name} roots overlap",
                )


def _validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(value)
    _require(set(binding) == IMPLEMENTATION_KEYS, "implementation binding keys differ")
    _require(binding.get("git_clean") is True, "implementation binding is not clean")
    for name in ("repository_commit", "repository_tree"):
        _require(
            isinstance(binding.get(name), str)
            and HEX40.fullmatch(binding[name]) is not None,
            f"implementation {name} is malformed",
        )
    for name in (
        "atlas_analyzer_authority_sha256", "generator_sha256", "routes_sha256"
    ):
        _require(
            isinstance(binding.get(name), str)
            and HEX64.fullmatch(binding[name]) is not None,
            f"implementation {name} is malformed",
        )
    _require(
        isinstance(binding.get("atlas_analyzer_authority_file_count"), int)
        and not isinstance(binding["atlas_analyzer_authority_file_count"], bool)
        and binding["atlas_analyzer_authority_file_count"] > 0,
        "implementation analyzer file count is malformed",
    )
    return binding


def _retired_identities() -> tuple[set[str], set[str], set[int]]:
    cohorts, _declarations, maps, seeds = _load_registry()
    return set(cohorts), set(maps), set(seeds)


def build_declaration(
    qualification_id: str,
    seed_base: int,
    implementation: Mapping[str, Any],
    *,
    retired: tuple[set[str], set[str], set[int]] | None = None,
) -> dict[str, Any]:
    _require(
        isinstance(qualification_id, str)
        and QUALIFICATION_ID.fullmatch(qualification_id) is not None
        and "final" not in qualification_id,
        "qualification ID is malformed or final-mode",
    )
    _require(
        isinstance(seed_base, int) and not isinstance(seed_base, bool)
        and 0 <= seed_base <= 2_147_482_943,
        "seed base must permit seven 100-seed style blocks in signed i32",
    )
    retired_cohorts, retired_maps, retired_seeds = (
        retired if retired is not None else _retired_identities()
    )
    _require(qualification_id not in retired_cohorts, "retired cohort ID reused")
    maps = []
    ordinal = 0
    for style_index, style in enumerate(CONCRETE_STYLES):
        for member in range(4):
            seed = seed_base + style_index * 100 + member
            name = f"{qualification_id}_{style}_{seed}"
            _require(
                MAP_ID_RE.fullmatch(name) is not None,
                f"qualification map name is invalid or too long: {name}",
            )
            _require(name not in retired_maps, f"retired map ID reused: {name}")
            _require(seed not in retired_seeds, f"retired map seed reused: {seed}")
            maps.append({
                "ordinal": ordinal,
                "map": name,
                "seed": seed,
                "style": style,
                "grid": 5,
                "observed_heat": None,
            })
            ordinal += 1
    _require(len({row["map"] for row in maps}) == MAP_COUNT, "map IDs are not unique")
    _require(len({row["seed"] for row in maps}) == MAP_COUNT, "map seeds are not unique")
    return {
        "schema": DECLARATION_SCHEMA,
        "qualification_id": qualification_id,
        "mode": "qualification",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "generator": {
            "version": "v6", "grid": 5, "gym": False,
            "observed_heat": None,
        },
        "selection": {
            "required_map_count": MAP_COUNT,
            "required_concrete_styles": list(CONCRETE_STYLES),
            "required_maps_per_style": 4,
        },
        "implementation": dict(implementation),
        "maps": maps,
    }


def _default_generation_task(row: Mapping[str, Any], output: str) -> None:
    # Generator summaries are useful interactively but the producer's stdout
    # is reserved for its one canonical report.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        _default_generator(
            str(row["map"]), int(row["seed"]), Path(output),
            grid=int(row["grid"]), style=str(row["style"]),
        )


def _generate_population(
    maps: Sequence[Mapping[str, Any]],
    output: Path,
    workers: int,
    generator: Callable[..., None] | None,
) -> None:
    failures: list[str] = []
    executor_type = ProcessPoolExecutor if generator is None else ThreadPoolExecutor
    with executor_type(max_workers=workers) as executor:
        futures = {}
        for row in maps:
            if generator is None:
                future = executor.submit(_default_generation_task, dict(row), str(output))
            else:
                future = executor.submit(
                    generator,
                    row["map"], row["seed"], output,
                    grid=row["grid"], style=row["style"],
                )
            futures[future] = (row["ordinal"], row["map"])
        for future in as_completed(futures):
            ordinal, name = futures[future]
            try:
                future.result()
            except Exception as error:  # executor/process boundary
                failures.append(
                    f"map {ordinal} {name}: {type(error).__name__}: {error}"
                )
    if failures:
        raise QualificationSourceError(
            "source generation failed: " + "; ".join(sorted(failures))
        )


def _source_membership(
    maps: Sequence[Mapping[str, Any]], directory: Path,
) -> dict[str, Any]:
    expected = {
        f"{row['map']}{suffix}" for row in maps for suffix in SOURCE_SUFFIXES
    }
    actual: set[str] = set()
    failures: list[str] = []
    if not directory.is_dir() or directory.is_symlink():
        failures.append("source root is absent, invalid, or a symlink")
    else:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                failures.append(f"source root contains symlink {relative}")
            elif path.is_file():
                actual.add(relative)
            elif path != directory and path.is_dir():
                failures.append(f"source root contains nested directory {relative}")
    failures.extend(f"missing {name}" for name in sorted(expected - actual))
    failures.extend(f"unexpected {name}" for name in sorted(actual - expected))
    return {
        "expected_file_count": len(expected),
        "actual_file_count": len(actual),
        "failures": sorted(set(failures)),
        "passed": not failures,
    }


def _source_files(
    row: Mapping[str, Any], primary: Path, cold: Path,
) -> tuple[dict[str, Any], bool]:
    files: dict[str, Any] = {}
    identical = True
    for suffix in SOURCE_SUFFIXES:
        first = primary / f"{row['map']}{suffix}"
        second = cold / f"{row['map']}{suffix}"
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
        identical = identical and first_bytes == second_bytes
        files[suffix] = {
            "bytes": len(first_bytes),
            "sha256": _sha256_bytes(first_bytes),
        }
    return files, identical


def source_evidence_sha256(map_id: str, files: Mapping[str, Any]) -> str:
    """Return the compile-stage-recomputable source evidence identity."""

    return _sha256_bytes(canonical_bytes({"map": map_id, "files": dict(files)}))


def _inspect_map(
    row: Mapping[str, Any],
    primary: Path,
    cold: Path,
    static_validator: Callable[[Path], Mapping[str, Any]],
    metadata_validator: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]],
    route_loader: Callable[[Path, str], Mapping[str, Any]],
    spawn_binding: Callable[[Path, Mapping[str, Any], str], Mapping[str, Any]],
) -> dict[str, Any]:
    name = str(row["map"])
    files, cold_identical = _source_files(row, primary, cold)
    criteria = {
        "source-files-complete": len(files) == len(SOURCE_SUFFIXES),
        "cold-bytes-identical": cold_identical,
        "metadata-contract": False,
        "source-static": False,
        "route-contract": False,
        "spawn-origin-binding": False,
        "layout-unique": False,  # assigned after every layout hash is known
    }
    failures: list[str] = []
    if not cold_identical:
        failures.append("primary and cold source bytes differ")
    try:
        primary_metadata = dict(
            metadata_validator(primary / f"{name}.meta.json", row)
        )
        cold_metadata = dict(metadata_validator(cold / f"{name}.meta.json", row))
        criteria["metadata-contract"] = primary_metadata == cold_metadata
        if not criteria["metadata-contract"]:
            failures.append("primary and cold metadata contracts differ")
    except Exception as error:
        failures.append(f"metadata contract: {type(error).__name__}: {error}")
    try:
        primary_static = dict(static_validator(primary / f"{name}.map"))
        cold_static = dict(static_validator(cold / f"{name}.map"))
        criteria["source-static"] = (
            primary_static.get("static_ok") is True
            and cold_static.get("static_ok") is True
            and canonical_bytes(primary_static) == canonical_bytes(cold_static)
        )
        if not criteria["source-static"]:
            failures.append("source/static validation failed or differed on cold rebuild")
    except Exception as error:
        failures.append(f"source/static validation: {type(error).__name__}: {error}")
    route: Mapping[str, Any] | None = None
    try:
        primary_route = dict(route_loader(primary / f"{name}.routes.json", name))
        cold_route = dict(route_loader(cold / f"{name}.routes.json", name))
        criteria["route-contract"] = primary_route == cold_route
        if criteria["route-contract"]:
            route = primary_route
        else:
            failures.append("primary and cold route contracts differ")
    except Exception as error:
        failures.append(f"route contract: {type(error).__name__}: {error}")
    if route is not None:
        try:
            primary_spawn = dict(
                spawn_binding(primary / f"{name}.map", route, name)
            )
            cold_spawn = dict(spawn_binding(cold / f"{name}.map", route, name))
            criteria["spawn-origin-binding"] = primary_spawn == cold_spawn
            if not criteria["spawn-origin-binding"]:
                failures.append("primary and cold spawn-origin bindings differ")
        except Exception as error:
            failures.append(
                f"spawn-origin binding: {type(error).__name__}: {error}"
            )
    return {
        "ordinal": row["ordinal"],
        "map": name,
        "criteria": criteria,
        "evidence_sha256": source_evidence_sha256(name, files),
        "failures": failures,
        "passed": False,
        "_layout_sha256": files[".map"]["sha256"],
    }


def run_source_qualification(
    *,
    qualification_id: str,
    seed_base: int,
    source_root: Path,
    cold_root: Path,
    declaration_path: Path,
    report_path: Path,
    workers: int,
    repo_root: Path = ROOT,
    _generator: Callable[..., None] | None = None,
    _static_validator: Callable[[Path], Mapping[str, Any]] = _default_static_validator,
    _metadata_validator: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] = _validate_metadata,
    _route_loader: Callable[[Path, str], Mapping[str, Any]] = load_source_route_contract,
    _spawn_binding: Callable[[Path, Mapping[str, Any], str], Mapping[str, Any]] = _source_spawn_origin_binding,
    _binding: Mapping[str, Any] | None = None,
    _retired: tuple[set[str], set[str], set[int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = _absolute(repo_root)
    _require(repo_root == ROOT, "repo root must be the repository containing this tool")
    _require(1 <= workers <= MAX_WORKERS, f"workers must be in [1,{MAX_WORKERS}]")
    paths = {
        "source-root": _absolute(source_root),
        "cold-root": _absolute(cold_root),
        "declaration": _absolute(declaration_path),
        "report": _absolute(report_path),
    }
    _require_distinct_paths(paths)
    for label, path in paths.items():
        _require_qualification_path(path, label, repo_root)
        _require(not path.exists() and not path.is_symlink(), f"{label} already exists")
        _require(path.parent.is_dir() and not path.parent.is_symlink(), f"{label} parent is absent or invalid")
    binding = _validate_binding(
        _binding if _binding is not None else repository_binding(repo_root)
    )
    declaration = build_declaration(
        qualification_id, seed_base, binding, retired=_retired
    )
    declaration_payload = canonical_bytes(declaration)
    declaration_sha256 = _sha256_bytes(declaration_payload)

    paths["source-root"].mkdir()
    paths["cold-root"].mkdir()
    _generate_population(declaration["maps"], paths["source-root"], workers, _generator)
    # A separate executor/process population makes the rebuild cold with respect
    # to generator instances and worker state.
    _generate_population(declaration["maps"], paths["cold-root"], workers, _generator)
    primary_membership = _source_membership(declaration["maps"], paths["source-root"])
    cold_membership = _source_membership(declaration["maps"], paths["cold-root"])
    _require(primary_membership["passed"] is True, "primary source membership differs: " + "; ".join(primary_membership["failures"]))
    _require(cold_membership["passed"] is True, "cold source membership differs: " + "; ".join(cold_membership["failures"]))

    by_ordinal: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _inspect_map, row, paths["source-root"], paths["cold-root"],
                _static_validator, _metadata_validator, _route_loader,
                _spawn_binding,
            ): int(row["ordinal"])
            for row in declaration["maps"]
        }
        for future in as_completed(futures):
            ordinal = futures[future]
            by_ordinal[ordinal] = future.result()
    rows = [by_ordinal[index] for index in range(MAP_COUNT)]
    layout_counts = Counter(row["_layout_sha256"] for row in rows)
    for row in rows:
        row["criteria"]["layout-unique"] = layout_counts[row["_layout_sha256"]] == 1
        if not row["criteria"]["layout-unique"]:
            row["failures"].append("source map layout duplicates another qualification member")
        row["failures"] = sorted(set(row["failures"]))
        row["passed"] = all(value is True for value in row["criteria"].values()) and not row["failures"]
        del row["_layout_sha256"]

    final_primary = _source_membership(declaration["maps"], paths["source-root"])
    final_cold = _source_membership(declaration["maps"], paths["cold-root"])
    _require(final_primary == primary_membership, "primary source root changed during validation")
    _require(final_cold == cold_membership, "cold source root changed during validation")
    # Membership equality is not content stability; re-read all source evidence.
    for declared, row in zip(declaration["maps"], rows):
        files, identical = _source_files(
            declared, paths["source-root"], paths["cold-root"]
        )
        _require(identical, f"{declared['map']} changed after cold comparison")
        _require(
            source_evidence_sha256(str(declared["map"]), files)
            == row["evidence_sha256"],
            f"{declared['map']} source evidence changed during validation",
        )
    evidence = [row["evidence_sha256"] for row in rows]
    _require(len(set(evidence)) == MAP_COUNT, "per-map source evidence digests are not unique")
    if _binding is None:
        _require(repository_binding(repo_root) == binding, "repository changed during source qualification")

    report = {
        "schema": STAGE_SCHEMA,
        "qualification_id": qualification_id,
        "mode": "qualification",
        "stage": "source",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": binding,
        "input_report_sha256": None,
        "infrastructure_checks": {
            **{name: True for name in REQUIRED_STAGE_CHECKS["source"]},
            "bounded-parallel-workers": True,
            "exact-membership": True,
            "input-stability": True,
        },
        "map_count": MAP_COUNT,
        "pass_count": sum(row["passed"] is True for row in rows),
        "maps": rows,
        "failures": [],
    }
    _exclusive_write(paths["declaration"], declaration_payload)
    _exclusive_write(paths["report"], canonical_bytes(report))
    return declaration, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-id", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cold-root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _declaration, report = run_source_qualification(
            qualification_id=args.qualification_id,
            seed_base=args.seed_base,
            source_root=args.source_root,
            cold_root=args.cold_root,
            declaration_path=args.declaration,
            report_path=args.report,
            workers=args.workers,
            repo_root=args.repo_root,
        )
        sys.stdout.buffer.write(canonical_bytes(report))
        return 0
    except (
        QualificationSourceError, GeneratorCohortError,
        RetiredCohortRegistryError, SourceRouteContractError, OSError,
    ) as error:
        print(f"B2 qualification source refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
