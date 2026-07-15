#!/usr/bin/env python3
"""Generate and audit the fixed 56-map development population.

This is pre-declaration development evidence only.  It never compiles maps,
builds Atlas artifacts, installs a runtime, or declares a final cohort.  The
canonical report is created exclusively and only after both fresh source
generations pass every gate; a partial or failed population produces no report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_generator_cohort as cohort  # noqa: E402
from tools.source_route_contract import (  # noqa: E402
    ROUTE_ARCHETYPES,
    SourceRouteContractError,
    load_source_route_contract,
)


REPORT_SCHEMA = "q2-b2-generator-development-population-audit-v1"
POPULATION_ID = "b2dev56_71425"
MAP_PREFIX = "b2dev"
GRID = 5
MAPS_PER_STYLE = 8
MAP_COUNT = len(cohort.CONCRETE_STYLES) * MAPS_PER_STYLE
FINAL_DECLARATION = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
)
STYLE_SEED_BASES = {
    style: 71_425_000 + style_index * 100
    for style_index, style in enumerate(cohort.CONCRETE_STYLES)
}


class DevelopmentPopulationError(ValueError):
    """Raised when the fixed development population fails closed."""


def development_matrix() -> list[dict[str, Any]]:
    """Return the immutable, balanced 56-member development matrix."""
    rows = []
    for style in cohort.CONCRETE_STYLES:
        for offset in range(MAPS_PER_STYLE):
            seed = STYLE_SEED_BASES[style] + offset
            rows.append({
                "ordinal": len(rows),
                "map": f"{MAP_PREFIX}_{style}_{seed}",
                "seed": seed,
                "style": style,
                "grid": GRID,
            })
    return rows


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=cohort._no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentPopulationError(
                    f"{label} contains non-finite JSON token {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentPopulationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DevelopmentPopulationError(f"{label} must be a JSON object")
    return value


def _final_reservations(
    declaration_path: Path,
) -> tuple[set[str], set[int], str, str]:
    try:
        declaration, digest = cohort.load_declaration(declaration_path)
    except cohort.GeneratorCohortError as exc:
        raise DevelopmentPopulationError(
            f"cannot bind final-cohort exclusions: {exc}"
        ) from exc
    return (
        {str(row["map"]) for row in declaration["maps"]},
        {int(row["seed"]) for row in declaration["maps"]},
        str(declaration["cohort_id"]),
        digest,
    )


def _validate_matrix(
    rows: Sequence[Mapping[str, Any]],
    final_ids: set[str],
    final_seeds: set[int],
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    for row in normalized:
        map_id = row.get("map")
        seed = row.get("seed")
        if (
            isinstance(map_id, str)
            and (map_id in final_ids or map_id.startswith("b2g26_"))
        ):
            raise DevelopmentPopulationError(
                f"development population rejects final cohort ID {map_id}"
            )
        if isinstance(seed, int) and not isinstance(seed, bool) and seed in final_seeds:
            raise DevelopmentPopulationError(
                f"development population rejects final cohort seed {seed}"
            )
    expected = development_matrix()
    if normalized != expected:
        raise DevelopmentPopulationError(
            "development matrix differs from the fixed disjoint 56-map contract; "
            "filtering, substitution, and reordering are forbidden"
        )
    if len({row["map"] for row in normalized}) != MAP_COUNT:
        raise DevelopmentPopulationError("development map IDs are not unique")
    if len({row["seed"] for row in normalized}) != MAP_COUNT:
        raise DevelopmentPopulationError("development seeds are not unique")
    return normalized


def _validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "repository_commit",
        "repository_tree",
        "git_clean",
        "atlas_analyzer_authority_sha256",
        "atlas_analyzer_authority_file_count",
        "generator_sha256",
        "routes_sha256",
    }
    normalized = dict(binding)
    if set(normalized) != expected_keys or normalized.get("git_clean") is not True:
        raise DevelopmentPopulationError(
            "implementation binding is incomplete or repository is not clean"
        )
    for key in (
        "atlas_analyzer_authority_sha256",
        "generator_sha256",
        "routes_sha256",
    ):
        value = normalized.get(key)
        if not isinstance(value, str) or not cohort.HEX_RE.fullmatch(value):
            raise DevelopmentPopulationError(
                f"implementation binding {key} is malformed"
            )
    for key in ("repository_commit", "repository_tree"):
        value = normalized.get(key)
        if not isinstance(value, str) or not cohort.GIT_HEX_RE.fullmatch(value):
            raise DevelopmentPopulationError(
                f"implementation binding {key} is malformed"
            )
    count = normalized.get("atlas_analyzer_authority_file_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise DevelopmentPopulationError(
            "implementation binding analyzer authority file count is malformed"
        )
    return normalized


def _default_generator(
    name: str,
    seed: int,
    output: Path,
    *,
    grid: int,
    style: str,
) -> None:
    from maps.generator import generate_map

    generate_map(
        name,
        seed,
        output,
        grid_n=grid,
        style=style,
        observed_heat=None,
        gym=False,
    )


def _default_static_validator(map_path: Path) -> Mapping[str, Any]:
    from tools.validate_maps import static_validate

    return static_validate(
        map_path,
        SimpleNamespace(
            min_spawns=8,
            min_spawn_distance=384.0,
            min_span=1024.0,
            min_spawn_area=1_000_000.0,
            min_weapons=4,
            min_pickups=8,
            min_hook_zones=6,
            min_light_coverage=0.98,
        ),
    )


def _generate_once(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    generator: Callable[..., None],
) -> None:
    for row in rows:
        try:
            generator(
                str(row["map"]),
                int(row["seed"]),
                output,
                grid=int(row["grid"]),
                style=str(row["style"]),
            )
        except Exception as exc:
            raise DevelopmentPopulationError(
                f"{row['map']} generation raised: {exc}"
            ) from exc


def _expected_files(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        f"{row['map']}{suffix}"
        for row in rows
        for suffix in cohort.SOURCE_SUFFIXES
    }


def _verify_exact_membership(
    rows: Sequence[Mapping[str, Any]], directory: Path, label: str
) -> None:
    expected = _expected_files(rows)
    actual: set[str] = set()
    failures = []
    if not directory.is_dir():
        raise DevelopmentPopulationError(f"{label} directory is missing")
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            failures.append(f"symlink {relative}")
        elif path.is_file():
            actual.add(relative)
    failures.extend(f"missing file {name}" for name in sorted(expected - actual))
    failures.extend(
        f"unexpected file {name}" for name in sorted(actual - expected)
    )
    if failures:
        raise DevelopmentPopulationError(
            f"{label} membership failed: {'; '.join(failures)}"
        )


def _metadata_identity(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _load_json(path, f"{row['map']} metadata")
    expected = {
        "name": row["map"],
        "seed": row["seed"],
        "style": row["style"],
        "generator": "v6",
    }
    for key, value in expected.items():
        actual = metadata.get(key)
        if actual != value or isinstance(actual, bool):
            raise DevelopmentPopulationError(
                f"{row['map']} metadata {key} differs from fixed matrix"
            )
    # Generator v6 does not duplicate grid in its metadata.  Bind grid to the
    # direct fixed invocation; if a future version emits it, it must agree.
    if "grid" in metadata and metadata["grid"] != row["grid"]:
        raise DevelopmentPopulationError(
            f"{row['map']} metadata grid differs from fixed matrix"
        )
    return {
        "name": metadata["name"],
        "seed": metadata["seed"],
        "style": metadata["style"],
        "grid": row["grid"],
        "grid_binding": "direct-generator-invocation",
        "generator": metadata["generator"],
    }


def _route_contract(path: Path, map_id: str) -> dict[str, Any]:
    try:
        return load_source_route_contract(path, map_id)
    except SourceRouteContractError as exc:
        raise DevelopmentPopulationError(str(exc)) from exc


def audit_development_population(
    primary_dir: Path,
    cold_dir: Path,
    report_path: Path,
    *,
    repo_root: Path = ROOT,
    final_declaration: Path = FINAL_DECLARATION,
    _generator: Callable[..., None] = _default_generator,
    _static_validator: Callable[[Path], Mapping[str, Any]] = (
        _default_static_validator
    ),
    _binding: Mapping[str, Any] | None = None,
    _matrix: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    final_ids, final_seeds, final_cohort_id, final_digest = _final_reservations(
        final_declaration
    )
    rows = _validate_matrix(
        development_matrix() if _matrix is None else _matrix,
        final_ids,
        final_seeds,
    )
    if primary_dir.resolve() == cold_dir.resolve():
        raise DevelopmentPopulationError(
            "primary and cold development directories must differ"
        )
    if cohort._path_is_within(primary_dir, cold_dir) or cohort._path_is_within(
        cold_dir, primary_dir
    ):
        raise DevelopmentPopulationError(
            "primary and cold development directories must not be nested"
        )
    if cohort._path_is_within(report_path, primary_dir) or cohort._path_is_within(
        report_path, cold_dir
    ):
        raise DevelopmentPopulationError(
            "development audit report must be outside source directories"
        )
    if report_path.exists():
        raise DevelopmentPopulationError(
            "development audit report already exists; exclusive publication refused"
        )

    try:
        raw_binding = (
            cohort.repository_binding(repo_root)
            if _binding is None
            else _binding
        )
    except cohort.GeneratorCohortError as exc:
        raise DevelopmentPopulationError(str(exc)) from exc
    binding = _validate_binding(raw_binding)

    try:
        cohort._require_empty_directory(primary_dir, "primary development directory")
        cohort._require_empty_directory(cold_dir, "cold development directory")
    except cohort.GeneratorCohortError as exc:
        raise DevelopmentPopulationError(str(exc)) from exc
    _generate_once(rows, primary_dir, _generator)
    _generate_once(rows, cold_dir, _generator)
    _verify_exact_membership(rows, primary_dir, "primary")
    _verify_exact_membership(rows, cold_dir, "cold")

    layouts: set[str] = set()
    map_reports = []
    total_items = 0
    total_routes = 0
    total_endpoints = 0
    for row in rows:
        map_id = str(row["map"])
        files = {}
        for suffix in cohort.SOURCE_SUFFIXES:
            primary_path = primary_dir / f"{map_id}{suffix}"
            cold_path = cold_dir / f"{map_id}{suffix}"
            primary = primary_path.read_bytes()
            cold = cold_path.read_bytes()
            if primary != cold:
                raise DevelopmentPopulationError(
                    f"{map_id}{suffix} differs across fresh generations"
                )
            files[suffix] = {
                "bytes": len(primary),
                "sha256": cohort.sha256_bytes(primary),
            }

        layout_sha256 = files[".map"]["sha256"]
        if layout_sha256 in layouts:
            raise DevelopmentPopulationError(
                f"{map_id} duplicates a prior development layout"
            )
        layouts.add(layout_sha256)

        metadata = _metadata_identity(
            primary_dir / f"{map_id}.meta.json", row
        )
        cold_metadata = _metadata_identity(
            cold_dir / f"{map_id}.meta.json", row
        )
        if metadata != cold_metadata:
            raise DevelopmentPopulationError(
                f"{map_id} cold metadata identity differs"
            )
        try:
            static = dict(_static_validator(primary_dir / f"{map_id}.map"))
            cold_static = dict(_static_validator(cold_dir / f"{map_id}.map"))
        except Exception as exc:
            raise DevelopmentPopulationError(
                f"{map_id} source/static validation raised: {exc}"
            ) from exc
        if static.get("static_ok") is not True or cold_static.get("static_ok") is not True:
            raise DevelopmentPopulationError(
                f"{map_id} source/static validation failed"
            )
        if cohort.canonical_bytes(static) != cohort.canonical_bytes(cold_static):
            raise DevelopmentPopulationError(
                f"{map_id} cold source/static report differs"
            )

        routes = _route_contract(
            primary_dir / f"{map_id}.routes.json", map_id
        )
        cold_routes = _route_contract(
            cold_dir / f"{map_id}.routes.json", map_id
        )
        if routes != cold_routes:
            raise DevelopmentPopulationError(f"{map_id} cold route audit differs")
        total_items += int(routes["item_origin_count"])
        total_routes += int(routes["route_count"])
        total_endpoints += int(routes["route_endpoint_count"])
        map_reports.append({
            **row,
            "metadata_identity": metadata,
            "layout_sha256": layout_sha256,
            "source_static": static,
            "route_contract": routes,
            "source_files": files,
        })

    style_counts = Counter(str(row["style"]) for row in rows)
    report = {
        "schema": REPORT_SCHEMA,
        "population_id": POPULATION_ID,
        "status": "development-pre-declaration-audit-passed",
        "evidence_scope": "development-only",
        "final_cohort_admissible": False,
        "bundle_admissible": False,
        "atlas_admissible": False,
        "compile_performed": False,
        "deployment_performed": False,
        "selection_policy": "fixed-all-or-nothing-no-substitution",
        "implementation": binding,
        "final_cohort_exclusion": {
            "reserved_cohort_id": final_cohort_id,
            "reserved_declaration_sha256": final_digest,
            "reserved_id_count": len(final_ids),
            "reserved_seed_count": len(final_seeds),
            "development_ids_disjoint": True,
            "development_seeds_disjoint": True,
        },
        "map_count": len(map_reports),
        "required_map_count": MAP_COUNT,
        "maps_per_style": MAPS_PER_STYLE,
        "style_counts": dict(sorted(style_counts.items())),
        "grid": GRID,
        "source_suffixes": list(cohort.SOURCE_SUFFIXES),
        "exact_source_file_count_per_directory": (
            MAP_COUNT * len(cohort.SOURCE_SUFFIXES)
        ),
        "cold_rebuild": {
            "distinct_empty_directories": True,
            "all_five_suffixes_byte_identical": True,
        },
        "unique_layout_count": len(layouts),
        "source_static_pass_count": len(map_reports),
        "metadata_identity_pass_count": len(map_reports),
        "route_contract_pass_count": len(map_reports),
        "route_archetypes": list(ROUTE_ARCHETYPES),
        "all_route_archetypes_exactly_once": True,
        "route_count": total_routes,
        "route_endpoint_count": total_endpoints,
        "globally_unique_map_scoped_item_origin_count": total_items,
        "globally_unique_item_origins": True,
        "duplicate_route_endpoints": 0,
        "zero_length_route_legs": 0,
        "minimum_distinct_item_endpoints_per_route": 2,
        "all_route_endpoints_are_items": True,
        "published_dist_matches_endpoint_loop": True,
        "all_item_nodes_floor_assigned": True,
        "item_spawn_origin_collisions": 0,
        "all_spawns_and_route_endpoints_floor_assigned": True,
        "all_selected_endpoints_share_source_standing_component": True,
        "maps": map_reports,
        "failures": [],
        "passed": True,
    }
    try:
        cohort._exclusive_write(report_path, cohort.canonical_bytes(report))
    except FileExistsError as exc:
        raise DevelopmentPopulationError(
            "development audit report already exists; exclusive publication refused"
        ) from exc
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--cold-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--final-declaration", type=Path, default=FINAL_DECLARATION,
        help="authoritative final declaration whose IDs/seeds are forbidden",
    )
    arguments = parser.parse_args()
    try:
        report = audit_development_population(
            arguments.primary_dir,
            arguments.cold_dir,
            arguments.report,
            final_declaration=arguments.final_declaration,
        )
    except DevelopmentPopulationError as exc:
        print(f"development population audit failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(cohort.canonical_bytes({
        "schema": REPORT_SCHEMA,
        "population_id": report["population_id"],
        "map_count": report["map_count"],
        "style_counts": report["style_counts"],
        "unique_layout_count": report["unique_layout_count"],
        "report_sha256": cohort.file_sha256(arguments.report),
        "passed": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
