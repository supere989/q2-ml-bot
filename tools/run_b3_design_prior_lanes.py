#!/usr/bin/env python3
"""Run one frozen B3 baseline/treatment lane through the real B2 lifecycle.

This local-only orchestrator binds the B3 parameter plan to the existing
generator source freeze, source/static validator, q2tool compiler, compiled-CM
preflight, hook materializer, claims producer, Atlas builder, and compiled
promotion validator. It publishes a lane manifest only after every declared
map passes every stage. There is no resume, subset, replacement, remote, or
runtime-install mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compile_generated_cohort import (  # noqa: E402
    CompileCohortError,
    compile_generated_cohort,
)
from tools.materialize_generated_cohort import (  # noqa: E402
    MAX_MATERIALIZER_TIMEOUT_SECONDS,
    MaterializeCohortError,
    materialize_cohort,
)
from tools.run_compiled_cm_preflight import (  # noqa: E402
    CompiledCmPreflightError,
    MAX_JOBS,
    run as run_compiled_cm_preflight,
)
from tools.run_generated_atlas_campaign import (  # noqa: E402
    GeneratedAtlasCampaignError,
    build_atlas_campaign,
)
from tools.run_generator_claim_campaign import (  # noqa: E402
    ClaimCampaignError,
    prepare_claims,
    validate_campaign as validate_claim_campaign,
)
from tools.run_generator_cohort import (  # noqa: E402
    DECLARATION_SCHEMA,
    SOURCE_SUFFIXES,
    GeneratorCohortError,
    generate_source_freeze,
    repository_binding as cohort_repository_binding,
    validate_declaration,
)
from tools.run_b3_design_prior_campaign import (  # noqa: E402
    B3PriorError,
    DEFAULT_KNOBS,
    LANE_SCHEMA,
    PLAN_SCHEMA,
    STYLES,
    canonical_bytes,
    file_sha256,
    load_json,
    sha256_bytes,
    validate_design_signature,
    validate_plan,
)


PIPELINE_STAGES = (
    "declaration", "source_freeze", "compile", "compiled_cm_preflight",
    "materialization", "claims_prepare", "atlas_build", "claims_validation",
)
COMPILE_TIMEOUT_MAX_SECONDS = 86_400.0
ORACLE_BATCH_TIMEOUT_MAX_SECONDS = 60.0


class B3LaneError(ValueError):
    """Raised before lane publication when any real lifecycle stage fails."""


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise B3LaneError(f"pipeline report is missing or a symlink: {path}")
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _require_regular(path: Path, label: str, *, executable: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise B3LaneError(f"{label} must be an existing non-symlink regular file")
    if executable and not os.access(path, os.X_OK):
        raise B3LaneError(f"{label} must be executable")


def _require_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise B3LaneError(f"{label} must be an existing non-symlink directory")


def _overlaps(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _preflight_configuration(
    *,
    plan_path: Path,
    work_root: Path,
    output_lane: Path,
    q2tool: Path,
    packer: Path,
    verifier: Path,
    basedir: Path,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_attestation: Path,
    b1_gate: Path,
    client_root: Path,
    lithium_root: Path,
    compile_timeout_seconds: float,
    materialize_timeout_seconds: int,
    oracle_batch_timeout_seconds: float,
    jobs: int,
) -> None:
    if (
        isinstance(compile_timeout_seconds, bool)
        or not isinstance(compile_timeout_seconds, (int, float))
        or not math.isfinite(compile_timeout_seconds)
        or not 0 < compile_timeout_seconds <= COMPILE_TIMEOUT_MAX_SECONDS
    ):
        raise B3LaneError("compile timeout must be finite in (0,86400]")
    if (
        isinstance(materialize_timeout_seconds, bool)
        or not isinstance(materialize_timeout_seconds, int)
        or not 1 <= materialize_timeout_seconds <= MAX_MATERIALIZER_TIMEOUT_SECONDS
    ):
        raise B3LaneError(
            f"materialize timeout must be an integer in [1,{MAX_MATERIALIZER_TIMEOUT_SECONDS}]"
        )
    if (
        isinstance(oracle_batch_timeout_seconds, bool)
        or not isinstance(oracle_batch_timeout_seconds, (int, float))
        or not math.isfinite(oracle_batch_timeout_seconds)
        or not 0 < oracle_batch_timeout_seconds <= ORACLE_BATCH_TIMEOUT_MAX_SECONDS
    ):
        raise B3LaneError("compiled-CM oracle timeout must be finite in (0,60]")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= MAX_JOBS:
        raise B3LaneError(f"compiled-CM jobs must be an integer in [1,{MAX_JOBS}]")
    _require_regular(plan_path, "frozen campaign plan")
    _require_regular(q2tool, "q2tool", executable=True)
    for path, label in (
        (packer, "Atlas packer"), (verifier, "Atlas verifier"),
    ):
        if not path.is_absolute():
            raise B3LaneError(f"{label} must be an absolute path")
        _require_regular(path, label, executable=True)
    for path, label in (
        (cm_oracle, "collision oracle"), (pmove_oracle, "Pmove oracle"),
        (hook_oracle, "hook oracle"), (fall_oracle, "fall oracle"),
    ):
        _require_regular(path, label, executable=True)
    _require_regular(hook_attestation, "hook parity attestation")
    _require_regular(b1_gate, "B1 gate")
    _require_directory(basedir, "q2tool basedir")
    _require_directory(client_root, "client source root")
    _require_directory(lithium_root, "Lithium source root")
    if work_root.exists() or work_root.is_symlink():
        raise B3LaneError("lane work root must not exist")
    if output_lane.exists() or output_lane.is_symlink():
        raise B3LaneError("lane output already exists")
    if _overlaps(work_root, output_lane):
        raise B3LaneError("lane work root and output manifest must be disjoint")
    if not work_root.parent.is_dir() or work_root.parent.is_symlink():
        raise B3LaneError("lane work-root parent must be an existing non-symlink directory")
    if not output_lane.parent.is_dir() or output_lane.parent.is_symlink():
        raise B3LaneError("lane output parent must be an existing non-symlink directory")
    protected_inputs = (
        plan_path, q2tool, packer, verifier, basedir, cm_oracle, pmove_oracle,
        hook_oracle, fall_oracle, hook_attestation, b1_gate, client_root,
        lithium_root,
    )
    if any(_overlaps(work_root, path) or _overlaps(output_lane, path) for path in protected_inputs):
        raise B3LaneError("lane outputs overlap an authoritative input path")


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise B3LaneError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def lane_declaration(plan: Mapping[str, Any], lane: str) -> dict[str, Any]:
    if lane not in {"baseline", "treatment"}:
        raise B3LaneError("lane must be baseline or treatment")
    rows = []
    for pair in plan["pairs"]:
        expected = pair[lane]
        rows.append({
            "ordinal": pair["ordinal"], "map": expected["map"],
            "seed": pair["seed"], "style": expected["style"],
            "grid": 5, "observed_heat": None,
        })
    declaration = {
        "schema": DECLARATION_SCHEMA,
        "cohort_id": f"{plan['campaign_id']}_{lane}",
        "mode": "final",
        "generator": {"version": "v6", "grid": 5, "observed_heat": None, "gym": False},
        "selection": {
            "timing": "declared-before-generation", "policy": "all-or-nothing",
            "replacement_allowed": False, "salvage_allowed": False,
            "required_map_count": 28, "required_concrete_styles": list(STYLES),
            "required_maps_per_style": 4,
        },
        "implementation_binding": {
            "require_clean_git": True, "bind_repository_commit": True,
            "bind_repository_tree": True, "bind_atlas_analyzer_closure": True,
        },
        "source_suffixes": list(SOURCE_SUFFIXES),
        "maps": rows,
    }
    try:
        return validate_declaration(declaration)
    except GeneratorCohortError as exc:
        raise B3LaneError(f"B3 lane declaration is not accepted by the B2 lifecycle: {exc}") from exc


def _qknobs(generator: Any) -> dict[str, Any]:
    return {
        name: round(float(getattr(generator, name)) * 1_000_000)
        for name in (
            "occupied_density", "corridor_prob", "hallway_ratio", "tower_prob",
            "lane_prob", "lava_prob", "extra_arena_prob", "large_building_ratio",
        )
    } | {
        "terrace_levels": int(generator.terrace_levels),
        "arena_cover_range": list(generator.arena_cover_range),
        "corner_range": list(generator.corner_range),
    }


def _apply_qknobs(generator: Any, expected: Mapping[str, Any]) -> None:
    for name in (
        "occupied_density", "corridor_prob", "hallway_ratio", "tower_prob",
        "lane_prob", "lava_prob", "extra_arena_prob", "large_building_ratio",
    ):
        setattr(generator, name, int(expected[name]) / 1_000_000.0)
    generator.terrace_levels = int(expected["terrace_levels"])
    generator.arena_cover_range = tuple(int(value) for value in expected["arena_cover_range"])
    generator.corner_range = tuple(int(value) for value in expected["corner_range"])


def bound_generator(plan: Mapping[str, Any], lane: str):
    expected_by_name = {pair[lane]["map"]: pair for pair in plan["pairs"]}

    def generate(
        name: str, seed: int, output: Path, *, grid: int, style: str,
    ) -> None:
        pair = expected_by_name.get(name)
        if pair is None:
            raise B3LaneError(f"undeclared map reached the bound generator: {name}")
        expected = pair[lane]
        if (seed, grid, style) != (pair["seed"], 5, expected["style"]):
            raise B3LaneError(f"bound generator request differs for {name}")
        import maps.generator as generator_module

        original = generator_module.MapGenerator
        expected_knobs = expected["generator_knobs"]

        class ParameterBoundGenerator(original):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)
                if self.style != style or _qknobs(self) != DEFAULT_KNOBS[style]:
                    raise B3LaneError(
                        f"generator default identity drifted before applying B3 bias for {name}"
                    )
                _apply_qknobs(self, expected_knobs)
                if _qknobs(self) != expected_knobs:
                    raise B3LaneError(f"generator bias did not apply exactly for {name}")

        generator_module.MapGenerator = ParameterBoundGenerator
        try:
            generator_module.generate_map(
                name, seed, output, grid_n=grid, style=style,
                observed_heat=None, gym=False,
            )
        finally:
            generator_module.MapGenerator = original

    return generate


def _require_stage(report: Mapping[str, Any], label: str) -> None:
    if report.get("passed") is not True or report.get("failures") not in (None, []):
        raise B3LaneError(f"{label} did not produce a complete passing report")


def run_lane(
    *,
    plan_path: Path,
    lane: str,
    work_root: Path,
    output_lane: Path,
    q2tool: Path,
    packer: Path,
    verifier: Path,
    basedir: Path,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_attestation: Path,
    b1_gate: Path,
    client_root: Path,
    lithium_root: Path,
    compile_timeout_seconds: float = 900.0,
    materialize_timeout_seconds: int = 900,
    oracle_batch_timeout_seconds: float = ORACLE_BATCH_TIMEOUT_MAX_SECONDS,
    jobs: int = 4,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    if repo_root.resolve() != ROOT.resolve():
        raise B3LaneError("B3 lane orchestration requires the canonical repository root")
    _preflight_configuration(
        plan_path=plan_path, work_root=work_root, output_lane=output_lane,
        q2tool=q2tool, packer=packer, verifier=verifier, basedir=basedir,
        cm_oracle=cm_oracle,
        pmove_oracle=pmove_oracle, hook_oracle=hook_oracle,
        fall_oracle=fall_oracle, hook_attestation=hook_attestation,
        b1_gate=b1_gate, client_root=client_root, lithium_root=lithium_root,
        compile_timeout_seconds=compile_timeout_seconds,
        materialize_timeout_seconds=materialize_timeout_seconds,
        oracle_batch_timeout_seconds=oracle_batch_timeout_seconds, jobs=jobs,
    )
    plan = validate_plan(load_json(plan_path))
    if plan["schema"] != PLAN_SCHEMA:
        raise B3LaneError("campaign plan schema differs")
    if lane not in {"baseline", "treatment"}:
        raise B3LaneError("lane must be baseline or treatment")
    binding = cohort_repository_binding(repo_root)
    expected_implementation = plan["implementation"]
    comparisons = {
        "repository_commit": binding["repository_commit"],
        "repository_tree": binding["repository_tree"],
        "git_clean": binding["git_clean"],
        "generator_sha256": binding["generator_sha256"],
        "analyzer_authority_sha256": binding["atlas_analyzer_authority_sha256"],
        "analyzer_authority_file_count": binding["atlas_analyzer_authority_file_count"],
    }
    if comparisons != expected_implementation:
        raise B3LaneError("current implementation identity drifted from the frozen campaign plan")
    work_root.mkdir()
    declaration_path = work_root / "declaration.json"
    source_dir = work_root / "source"
    source_cold_dir = work_root / "source-cold"
    source_report_path = work_root / "source-freeze.json"
    compile_report_path = work_root / "compile.json"
    compiled_dir = work_root / "compiled"
    preflight_path = work_root / "compiled-cm-preflight.json"
    materialize_report_path = work_root / "materialize.json"
    materialized_dir = work_root / "materialized"
    claims_dir = work_root / "claims"
    claims_report_path = work_root / "claims-prepare.json"
    analysis_dir = work_root / "analysis"
    atlas_report_path = work_root / "atlas-build.json"
    validation_path = work_root / "claims-validation.json"
    declaration = lane_declaration(plan, lane)
    _write_new(declaration_path, declaration)
    try:
        source_report = generate_source_freeze(
            declaration_path, source_dir, source_cold_dir, source_report_path,
            repo_root=repo_root, _generator=bound_generator(plan, lane),
        )
        _require_stage(source_report, "source freeze")
        compile_report = compile_generated_cohort(
            declaration_path, source_dir, work_root / "compile-staging",
            compiled_dir, work_root / "compile-logs", compile_report_path,
            q2tool, basedir, timeout_seconds=compile_timeout_seconds,
        )
        _require_stage(compile_report, "q2tool compilation")
        preflight_report, _preflight_sha = run_compiled_cm_preflight(
            declaration_path=declaration_path, compiled_dir=compiled_dir,
            cm_oracle=cm_oracle, output=preflight_path, jobs=jobs,
            oracle_batch_timeout_seconds=oracle_batch_timeout_seconds,
        )
        _require_stage(preflight_report, "compiled-CM preflight")
        materialize_report = materialize_cohort(
            declaration_path=declaration_path, compiled_dir=compiled_dir,
            stage_dir=work_root / "materialize-staging",
            materialized_dir=materialized_dir,
            log_dir=work_root / "materialize-logs", report_path=materialize_report_path,
            cm_oracle=cm_oracle, pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle, fall_oracle=fall_oracle,
            hook_attestation=hook_attestation, timeout_seconds=materialize_timeout_seconds,
        )
        _require_stage(materialize_report, "hook materialization")
        claims_report = prepare_claims(declaration_path, materialized_dir, claims_dir)
        _require_stage(claims_report, "claims preparation")
        _write_new(claims_report_path, claims_report)
        atlas_report = build_atlas_campaign(
            declaration_path, claims_dir, analysis_dir, work_root / "atlas-diagnostics",
            atlas_report_path, repo_root=repo_root, client_root=client_root,
            lithium_root=lithium_root, hook_attestation=hook_attestation,
            fall_oracle=fall_oracle, packer=packer, verifier=verifier,
        )
        _require_stage(atlas_report, "Atlas construction")
        validation = validate_claim_campaign(declaration_path, claims_dir, analysis_dir, b1_gate)
        _require_stage(validation, "compiled promotion validation")
        _write_new(validation_path, validation)
    except (
        B3PriorError, GeneratorCohortError, CompileCohortError,
        CompiledCmPreflightError, MaterializeCohortError, ClaimCampaignError,
        GeneratedAtlasCampaignError, OSError, ValueError,
    ) as exc:
        raise B3LaneError(f"{lane} lifecycle failed: {exc}") from exc

    source_rows = {row["map"]: row for row in source_report["maps"]}
    rows = []
    for pair in plan["pairs"]:
        expected = pair[lane]
        name = expected["map"]
        signature_path = analysis_dir / f"{name}.design-signature.json"
        signature = validate_design_signature(
            load_json(signature_path, canonical=False), f"{lane} {name}",
        )
        bsp_path = claims_dir / f"{name}.bsp"
        if file_sha256(bsp_path) != signature["bsp_sha256"]:
            raise B3LaneError(f"{name} BSP identity differs from its design signature")
        source_row = source_rows[name]
        rows.append({
            "ordinal": pair["ordinal"], "map": name, "seed": pair["seed"],
            "style": expected["style"], "generator_knobs": expected["generator_knobs"],
            "source_static_passed": source_row["source_static"]["static_ok"] is True,
            "layout_sha256": source_row["source_files"][".map"]["sha256"],
            "bsp_sha256": signature["bsp_sha256"],
            "design_signature": _file_record(signature_path),
        })
    pipeline_paths = {
        "declaration": declaration_path,
        "source_freeze": source_report_path,
        "compile": compile_report_path,
        "compiled_cm_preflight": preflight_path,
        "materialization": materialize_report_path,
        "claims_prepare": claims_report_path,
        "atlas_build": atlas_report_path,
        "claims_validation": validation_path,
    }
    lane_report = {
        "schema": LANE_SCHEMA,
        "campaign_id": plan["campaign_id"], "lane": lane,
        "plan_sha256": file_sha256(plan_path),
        "evidence_kind": "measured-compiled-atlas", "synthetic_claims": False,
        "implementation": expected_implementation,
        "authorities": {
            "q2tool": _file_record(q2tool), "packer": _file_record(packer),
            "verifier": _file_record(verifier),
            "cm_oracle": _file_record(cm_oracle),
            "pmove_oracle": _file_record(pmove_oracle),
            "hook_oracle": _file_record(hook_oracle),
            "fall_oracle": _file_record(fall_oracle),
            "hook_attestation": _file_record(hook_attestation),
            "b1_gate": _file_record(b1_gate),
        },
        "pipeline": {name: _file_record(pipeline_paths[name]) for name in PIPELINE_STAGES},
        "maps": rows, "failures": [], "passed": True,
    }
    _write_new(output_lane, lane_report)
    return lane_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lane", choices=("baseline", "treatment"), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-lane", type=Path, required=True)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--lithium-root", type=Path, required=True)
    parser.add_argument("--compile-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--materialize-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--oracle-batch-timeout-seconds", type=float,
        default=ORACLE_BATCH_TIMEOUT_MAX_SECONDS,
    )
    parser.add_argument("--jobs", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_lane(
            plan_path=args.plan, lane=args.lane, work_root=args.work_root,
            output_lane=args.output_lane, q2tool=args.q2tool, basedir=args.basedir,
            packer=args.packer, verifier=args.verifier,
            cm_oracle=args.cm_oracle, pmove_oracle=args.pmove_oracle,
            hook_oracle=args.hook_oracle, fall_oracle=args.fall_oracle,
            hook_attestation=args.hook_attestation, b1_gate=args.b1_gate,
            client_root=args.client_root, lithium_root=args.lithium_root,
            compile_timeout_seconds=args.compile_timeout_seconds,
            materialize_timeout_seconds=args.materialize_timeout_seconds,
            oracle_batch_timeout_seconds=args.oracle_batch_timeout_seconds,
            jobs=args.jobs,
        )
    except B3LaneError as exc:
        print(f"B3 lane refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes({
        "schema": LANE_SCHEMA, "lane": report["lane"],
        "map_count": len(report["maps"]),
        "manifest_sha256": file_sha256(args.output_lane),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
