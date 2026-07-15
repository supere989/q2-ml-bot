#!/usr/bin/env python3
"""Declare, generate, and verify an all-or-nothing generator-v6 cohort.

This tool deliberately stops at the source-freeze boundary.  It does not
compile maps, materialize hooks, run Atlas, install runtime files, or select a
passing subset.  The declaration fixes every map before generation.  A source
freeze is published only when both fresh generations are byte-identical and
every declared member passes the local source contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_source_closure import (  # noqa: E402
    atlas_analyzer_authority_inputs,
    atlas_analyzer_authority_sha256,
)
from tools.source_route_contract import (  # noqa: E402
    SourceRouteContractError,
    load_source_route_contract,
)


DECLARATION_SCHEMA = "q2-b2-generated-cohort-declaration-v1"
SOURCE_FREEZE_SCHEMA = "q2-b2-generated-source-freeze-v1"
STAGE_MEMBERSHIP_SCHEMA = "q2-b2-generator-stage-membership-v1"
CONCRETE_STYLES = (
    "open",
    "towers",
    "canyon",
    "pits",
    "arena_open",
    "arena_vertical",
    "arena_lanes",
)
SOURCE_SUFFIXES = (
    ".map",
    ".json",
    ".meta.json",
    ".lattice.json",
    ".routes.json",
)
STAGE_SUFFIXES = {
    "source": SOURCE_SUFFIXES,
    "compiled": (*SOURCE_SUFFIXES, ".bsp"),
    "materialized": (*SOURCE_SUFFIXES, ".bsp", ".hook-materialization.json"),
    "claims": (
        *SOURCE_SUFFIXES,
        ".bsp",
        ".hook-materialization.json",
        ".generator-claims.json",
    ),
    # Atlas output belongs in a separate directory because its .routes.json
    # artifact has the same name as the hash-bound generator route sidecar.
    "analysis": (
        ".analysis.manifest.json",
        ".atlas.bin",
        ".atlas.bin.zst",
        ".atlas.manifest.json",
        ".navigation.bin.zst",
        ".visibility.bin.zst",
        ".design-signature.json",
        ".routes.json",
    ),
}
MAP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


class GeneratorCohortError(ValueError):
    """Raised when an all-or-nothing cohort contract is not satisfied."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GeneratorCohortError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeneratorCohortError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise GeneratorCohortError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GeneratorCohortError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, expected: bool, label: str) -> None:
    if value is not expected:
        raise GeneratorCohortError(f"{label} must be {str(expected).lower()}")


def validate_declaration(value: object) -> dict[str, Any]:
    declaration = _mapping(value, "cohort declaration")
    _exact_keys(
        declaration,
        {
            "schema",
            "cohort_id",
            "mode",
            "generator",
            "selection",
            "implementation_binding",
            "source_suffixes",
            "maps",
        },
        "cohort declaration",
    )
    if declaration["schema"] != DECLARATION_SCHEMA:
        raise GeneratorCohortError("cohort declaration schema differs")
    cohort_id = declaration["cohort_id"]
    if not isinstance(cohort_id, str) or not MAP_ID_RE.fullmatch(cohort_id):
        raise GeneratorCohortError("cohort_id is not canonical")
    if declaration["mode"] != "final":
        raise GeneratorCohortError("source-freeze declarations must use final mode")

    generator = _mapping(declaration["generator"], "generator contract")
    _exact_keys(
        generator, {"version", "grid", "observed_heat", "gym"},
        "generator contract",
    )
    if generator["version"] != "v6":
        raise GeneratorCohortError("generator version must be v6")
    grid = _integer(generator["grid"], "generator grid", minimum=1)
    if grid != 5:
        raise GeneratorCohortError("the frozen generator grid must be 5")
    if generator["observed_heat"] is not None:
        raise GeneratorCohortError("final cohort cannot use observed heat")
    _boolean(generator["gym"], False, "generator gym")

    selection = _mapping(declaration["selection"], "selection contract")
    _exact_keys(
        selection,
        {
            "timing",
            "policy",
            "replacement_allowed",
            "salvage_allowed",
            "required_map_count",
            "required_concrete_styles",
            "required_maps_per_style",
        },
        "selection contract",
    )
    if selection["timing"] != "declared-before-generation":
        raise GeneratorCohortError("selection timing must precede generation")
    if selection["policy"] != "all-or-nothing":
        raise GeneratorCohortError("selection policy must be all-or-nothing")
    _boolean(selection["replacement_allowed"], False, "replacement_allowed")
    _boolean(selection["salvage_allowed"], False, "salvage_allowed")
    required_count = _integer(
        selection["required_map_count"], "required map count", minimum=1
    )
    per_style = _integer(
        selection["required_maps_per_style"], "required maps per style", minimum=1
    )
    styles = selection["required_concrete_styles"]
    if not isinstance(styles, list) or tuple(styles) != CONCRETE_STYLES:
        raise GeneratorCohortError(
            "required concrete styles must be the seven frozen explicit styles"
        )
    if required_count != len(CONCRETE_STYLES) * per_style or required_count != 28:
        raise GeneratorCohortError("final cohort must be balanced at 4/style and 28 maps")

    binding = _mapping(
        declaration["implementation_binding"], "implementation binding"
    )
    _exact_keys(
        binding,
        {
            "require_clean_git",
            "bind_repository_commit",
            "bind_repository_tree",
            "bind_atlas_analyzer_closure",
        },
        "implementation binding",
    )
    for key in binding:
        _boolean(binding[key], True, f"implementation binding {key}")

    suffixes = declaration["source_suffixes"]
    if not isinstance(suffixes, list) or tuple(suffixes) != SOURCE_SUFFIXES:
        raise GeneratorCohortError("source suffix contract differs")

    maps = declaration["maps"]
    if not isinstance(maps, list) or len(maps) != required_count:
        raise GeneratorCohortError(
            f"declared map count must be exactly {required_count}"
        )
    names: set[str] = set()
    seeds: set[int] = set()
    style_counts: Counter[str] = Counter()
    normalized_maps: list[dict[str, Any]] = []
    for expected_ordinal, item in enumerate(maps):
        row = _mapping(item, f"map declaration {expected_ordinal}")
        _exact_keys(
            row, {"ordinal", "map", "seed", "style", "grid", "observed_heat"},
            f"map declaration {expected_ordinal}",
        )
        ordinal = _integer(row["ordinal"], "map ordinal")
        if ordinal != expected_ordinal:
            raise GeneratorCohortError("map ordinals must be contiguous and ordered")
        name = row["map"]
        if not isinstance(name, str) or not MAP_ID_RE.fullmatch(name):
            raise GeneratorCohortError(f"map {ordinal} ID is not canonical")
        if name in names:
            raise GeneratorCohortError(f"duplicate map ID {name}")
        names.add(name)
        seed = _integer(row["seed"], f"map {name} seed")
        if seed in seeds:
            raise GeneratorCohortError(f"duplicate final seed {seed}")
        seeds.add(seed)
        style = row["style"]
        if style not in CONCRETE_STYLES:
            raise GeneratorCohortError(f"map {name} does not use a concrete style")
        style_counts[str(style)] += 1
        if _integer(row["grid"], f"map {name} grid", minimum=1) != grid:
            raise GeneratorCohortError(f"map {name} grid differs from generator contract")
        if row["observed_heat"] is not None:
            raise GeneratorCohortError(f"map {name} cannot use observed heat")
        normalized_maps.append(dict(row))
    expected_counts = {style: per_style for style in CONCRETE_STYLES}
    if dict(style_counts) != expected_counts:
        raise GeneratorCohortError(
            f"style balance differs: {dict(sorted(style_counts.items()))}"
        )

    normalized = dict(declaration)
    normalized["generator"] = dict(generator)
    normalized["selection"] = dict(selection)
    normalized["implementation_binding"] = dict(binding)
    normalized["maps"] = normalized_maps
    return normalized


def load_declaration(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GeneratorCohortError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratorCohortError(f"cannot read declaration: {exc}") from exc
    declaration = validate_declaration(value)
    if raw != canonical_bytes(declaration):
        raise GeneratorCohortError("cohort declaration is not canonical JSON")
    return declaration, sha256_bytes(raw)


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def repository_binding(repo_root: Path) -> dict[str, Any]:
    try:
        status = _git_output(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        commit = _git_output(repo_root, "rev-parse", "HEAD")
        tree = _git_output(repo_root, "rev-parse", "HEAD^{tree}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GeneratorCohortError(f"cannot bind Git implementation: {exc}") from exc
    if status:
        raise GeneratorCohortError("repository is not clean; refusing final generation")
    if not GIT_HEX_RE.fullmatch(commit) or not GIT_HEX_RE.fullmatch(tree):
        raise GeneratorCohortError("Git commit/tree identity is malformed")
    closure_inputs = atlas_analyzer_authority_inputs(repo_root)
    return {
        "repository_commit": commit,
        "repository_tree": tree,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": atlas_analyzer_authority_sha256(
            repo_root
        ),
        "atlas_analyzer_authority_file_count": len(closure_inputs),
        "generator_sha256": file_sha256(repo_root / "maps/generator.py"),
        "routes_sha256": file_sha256(repo_root / "maps/routes.py"),
    }


def _require_empty_directory(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise GeneratorCohortError(f"{label} is not a directory")
        if any(path.iterdir()):
            raise GeneratorCohortError(f"{label} must be empty")
    else:
        path.mkdir(parents=True)


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _expected_filenames(
    declaration: Mapping[str, Any], suffixes: Sequence[str]
) -> set[str]:
    return {
        f"{row['map']}{suffix}"
        for row in declaration["maps"]
        for suffix in suffixes
    }


def verify_stage_membership(
    declaration: Mapping[str, Any], directory: Path, stage: str
) -> dict[str, Any]:
    if stage not in STAGE_SUFFIXES:
        raise GeneratorCohortError(f"unknown cohort stage {stage!r}")
    declaration = validate_declaration(declaration)
    suffixes = STAGE_SUFFIXES[stage]
    expected = _expected_filenames(declaration, suffixes)
    failures: list[str] = []
    actual: set[str] = set()
    if not directory.is_dir():
        failures.append("stage directory is missing")
    else:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                failures.append(
                    f"stage contains symlink {path.relative_to(directory).as_posix()}"
                )
                continue
            if path.is_file():
                actual.add(path.relative_to(directory).as_posix())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    failures.extend(f"missing file {name}" for name in missing)
    failures.extend(f"unexpected file {name}" for name in unexpected)

    rows = []
    for row in declaration["maps"]:
        files = {}
        for suffix in suffixes:
            path = directory / f"{row['map']}{suffix}"
            if path.is_file() and not path.is_symlink():
                files[suffix] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
        rows.append({
            "ordinal": row["ordinal"],
            "map": row["map"],
            "files": files,
        })
    return {
        "schema": STAGE_MEMBERSHIP_SCHEMA,
        "cohort_id": declaration["cohort_id"],
        "stage": stage,
        "required_suffixes": list(suffixes),
        "expected_map_count": len(declaration["maps"]),
        "expected_file_count": len(expected),
        "actual_file_count": len(actual),
        "maps": rows,
        "failures": sorted(set(failures)),
        "passed": not failures,
    }


def _validate_metadata(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        meta = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GeneratorCohortError(f"non-finite metadata token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratorCohortError(f"{row['map']} metadata is unreadable: {exc}") from exc
    meta = _mapping(meta, f"{row['map']} metadata")
    expected = {
        "name": row["map"],
        "seed": row["seed"],
        "style": row["style"],
        "generator": "v6",
    }
    for key, value in expected.items():
        if meta.get(key) != value or isinstance(meta.get(key), bool):
            raise GeneratorCohortError(
                f"{row['map']} metadata {key} differs from declaration"
            )
    hooks = _mapping(
        meta.get("hook_claim_candidates_v4"),
        f"{row['map']} hook candidate metadata",
    )
    if (
        hooks.get("schema") != "q2-hook-claim-candidates-v4"
        or hooks.get("status") != "unproven"
        or hooks.get("bundle_admissible") is not False
        or not isinstance(hooks.get("records"), list)
        or len(hooks["records"]) < 6
    ):
        raise GeneratorCohortError(
            f"{row['map']} raw hook candidate contract is not fail-closed"
        )
    runtime_projection = path.with_name(f"{row['map']}.json").read_bytes()
    if b"# bundle_admissible: false" not in runtime_projection:
        raise GeneratorCohortError(
            f"{row['map']} raw hook projection is not explicitly non-admissible"
        )
    return {
        "name": meta["name"],
        "seed": meta["seed"],
        "style": meta["style"],
        "generator": meta["generator"],
        # Grid is an exact generator invocation input. Generator v6 does not
        # duplicate it in metadata, so the declaration and direct call bind it.
        "grid": row["grid"],
        "hook_candidate_count": len(hooks["records"]),
    }


def _default_static_validator(map_path: Path) -> Mapping[str, Any]:
    from tools.validate_maps import static_validate

    arguments = SimpleNamespace(
        min_spawns=8,
        min_spawn_distance=384.0,
        min_span=1024.0,
        min_spawn_area=1_000_000.0,
        min_weapons=4,
        min_pickups=8,
        min_hook_zones=6,
        min_light_coverage=0.98,
    )
    return static_validate(map_path, arguments)


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


def _generate_once(
    declaration: Mapping[str, Any],
    output: Path,
    generator: Callable[..., None],
) -> None:
    for row in declaration["maps"]:
        generator(
            row["map"],
            row["seed"],
            output,
            grid=row["grid"],
            style=row["style"],
        )


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def generate_source_freeze(
    declaration_path: Path,
    output_dir: Path,
    cold_dir: Path,
    report_path: Path,
    *,
    repo_root: Path = ROOT,
    _generator: Callable[..., None] = _default_generator,
    _static_validator: Callable[[Path], Mapping[str, Any]] = _default_static_validator,
    _binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    declaration, declaration_sha256 = load_declaration(declaration_path)
    if output_dir.resolve() == cold_dir.resolve():
        raise GeneratorCohortError("primary and cold directories must differ")
    if _path_is_within(report_path, output_dir) or _path_is_within(
        report_path, cold_dir
    ):
        raise GeneratorCohortError("source-freeze report must be outside stage directories")
    if report_path.exists():
        raise GeneratorCohortError("source-freeze report already exists")
    binding = dict(_binding) if _binding is not None else repository_binding(repo_root)
    expected_binding_keys = {
        "repository_commit",
        "repository_tree",
        "git_clean",
        "atlas_analyzer_authority_sha256",
        "atlas_analyzer_authority_file_count",
        "generator_sha256",
        "routes_sha256",
    }
    if set(binding) != expected_binding_keys or binding.get("git_clean") is not True:
        raise GeneratorCohortError("implementation binding is incomplete or not clean")
    for key in (
        "atlas_analyzer_authority_sha256", "generator_sha256", "routes_sha256"
    ):
        if not isinstance(binding.get(key), str) or not HEX_RE.fullmatch(binding[key]):
            raise GeneratorCohortError(f"implementation binding {key} is malformed")
    for key in ("repository_commit", "repository_tree"):
        if not isinstance(binding.get(key), str) or not GIT_HEX_RE.fullmatch(binding[key]):
            raise GeneratorCohortError(f"implementation binding {key} is malformed")
    _integer(
        binding.get("atlas_analyzer_authority_file_count"),
        "analyzer authority file count",
        minimum=1,
    )

    # The clean binding is captured before creating any artifact directory.
    _require_empty_directory(output_dir, "primary source directory")
    _require_empty_directory(cold_dir, "cold source directory")
    _generate_once(declaration, output_dir, _generator)
    _generate_once(declaration, cold_dir, _generator)

    primary_membership = verify_stage_membership(declaration, output_dir, "source")
    cold_membership = verify_stage_membership(declaration, cold_dir, "source")
    membership_failures = [
        *(f"primary: {item}" for item in primary_membership["failures"]),
        *(f"cold: {item}" for item in cold_membership["failures"]),
    ]
    if membership_failures:
        raise GeneratorCohortError("; ".join(sorted(membership_failures)))

    rows = []
    map_hashes: set[str] = set()
    for declared in declaration["maps"]:
        name = declared["map"]
        metadata = _validate_metadata(output_dir / f"{name}.meta.json", declared)
        cold_metadata = _validate_metadata(cold_dir / f"{name}.meta.json", declared)
        if metadata != cold_metadata:
            raise GeneratorCohortError(f"{name} cold metadata contract differs")
        try:
            static = dict(_static_validator(output_dir / f"{name}.map"))
            cold_static = dict(_static_validator(cold_dir / f"{name}.map"))
        except Exception as exc:
            raise GeneratorCohortError(
                f"{name} source/static validation raised: {exc}"
            ) from exc
        if static.get("static_ok") is not True or cold_static.get("static_ok") is not True:
            raise GeneratorCohortError(f"{name} source/static validation failed")
        if canonical_bytes(static) != canonical_bytes(cold_static):
            raise GeneratorCohortError(f"{name} cold source/static report differs")

        try:
            route_contract = load_source_route_contract(
                output_dir / f"{name}.routes.json", name
            )
            cold_route_contract = load_source_route_contract(
                cold_dir / f"{name}.routes.json", name
            )
        except SourceRouteContractError as exc:
            raise GeneratorCohortError(str(exc)) from exc
        if route_contract != cold_route_contract:
            raise GeneratorCohortError(f"{name} cold route contract differs")

        source_files = {}
        for suffix in SOURCE_SUFFIXES:
            primary = output_dir / f"{name}{suffix}"
            cold = cold_dir / f"{name}{suffix}"
            primary_bytes = primary.read_bytes()
            cold_bytes = cold.read_bytes()
            if primary_bytes != cold_bytes:
                raise GeneratorCohortError(
                    f"{name}{suffix} differs across fresh generations"
                )
            source_files[suffix] = {
                "bytes": len(primary_bytes),
                "sha256": sha256_bytes(primary_bytes),
            }
        layout_sha256 = source_files[".map"]["sha256"]
        if layout_sha256 in map_hashes:
            raise GeneratorCohortError(f"{name} duplicates a prior map layout")
        map_hashes.add(layout_sha256)
        rows.append({
            "ordinal": declared["ordinal"],
            "map": name,
            "seed": declared["seed"],
            "style": declared["style"],
            "grid": declared["grid"],
            "metadata": metadata,
            "source_static": static,
            "route_contract": route_contract,
            "source_files": source_files,
        })

    style_counts = Counter(row["style"] for row in rows)
    report = {
        "schema": SOURCE_FREEZE_SCHEMA,
        "cohort_id": declaration["cohort_id"],
        "status": "source-frozen-pre-compile",
        "bundle_admissible": False,
        "atlas_admissible": False,
        "selection_timing": "declared-before-generation",
        "selection_policy": "all-or-nothing",
        "replacement_allowed": False,
        "salvage_allowed": False,
        "declaration_sha256": declaration_sha256,
        "implementation": binding,
        "map_count": len(rows),
        "style_counts": dict(sorted(style_counts.items())),
        "unique_layout_count": len(map_hashes),
        "route_contract_pass_count": len(rows),
        "source_suffixes": list(SOURCE_SUFFIXES),
        "cold_rebuild": {
            "fresh_process_required": False,
            "independent_directory": True,
            "file_count": len(rows) * len(SOURCE_SUFFIXES),
            "all_file_bytes_match": True,
        },
        "maps": rows,
        "failures": [],
        "passed": True,
    }
    _exclusive_write(report_path, canonical_bytes(report))
    return report


def _main_generate(args: argparse.Namespace) -> int:
    report = generate_source_freeze(
        args.declaration,
        args.output_dir,
        args.cold_dir,
        args.report,
    )
    sys.stdout.buffer.write(canonical_bytes({
        "schema": SOURCE_FREEZE_SCHEMA,
        "cohort_id": report["cohort_id"],
        "map_count": report["map_count"],
        "style_counts": report["style_counts"],
        "unique_layout_count": report["unique_layout_count"],
        "report_sha256": file_sha256(args.report),
        "passed": True,
    }))
    return 0


def _main_verify_stage(args: argparse.Namespace) -> int:
    declaration, declaration_sha256 = load_declaration(args.declaration)
    report = verify_stage_membership(declaration, args.directory, args.stage)
    report["declaration_sha256"] = declaration_sha256
    payload = canonical_bytes(report)
    if args.output:
        if args.output.exists():
            raise GeneratorCohortError("stage membership report already exists")
        _exclusive_write(args.output, payload)
    sys.stdout.buffer.write(payload)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_declaration = ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"

    generate = subparsers.add_parser(
        "generate", help="generate and cold-verify the exact declared source cohort"
    )
    generate.add_argument("--declaration", type=Path, default=default_declaration)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--cold-dir", type=Path, required=True)
    generate.add_argument("--report", type=Path, required=True)
    generate.set_defaults(handler=_main_generate)

    verify = subparsers.add_parser(
        "verify-stage", help="verify exact declared membership for a later stage"
    )
    verify.add_argument("--declaration", type=Path, default=default_declaration)
    verify.add_argument("--stage", choices=tuple(STAGE_SUFFIXES), required=True)
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    verify.set_defaults(handler=_main_verify_stage)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except GeneratorCohortError as exc:
        print(f"generator cohort failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
