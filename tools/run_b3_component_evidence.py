#!/usr/bin/env python3
"""Produce measured recovery/guide and bundle evidence for the B3 gate.

Recovery/guide mode loads every declared Atlas through the real Rust
``AtlasRuntime`` (which re-solves and validates static costs), independently
counts descending/plateau cells from canonical bytes, and exercises objective
guide fixtures. Bundle mode runs the exact offline v2/v3 installer/farm suite.
Both modes run fixed component tests themselves and derive all booleans and
digests; callers cannot supply pass counts or claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assemble_b3_gate import (  # noqa: E402
    BUNDLE_SCHEMA,
    BUNDLE_SOURCE_PATHS,
    BUNDLE_TEST_COMMANDS,
    HAZARD_CLASSES,
    OBJECTIVE_CLASSES,
    RECOVERY_GUIDE_SCHEMA,
    RECOVERY_GUIDE_SOURCE_PATHS,
    RECOVERY_TEST_COMMANDS,
    _bundle_sections_from_claims,
    _source_closure,
    derive_bundle_claim_evidence,
    repository_identity,
    validate_bundle_evidence,
    validate_recovery_guide_evidence,
)
from tools.run_b3_design_prior_campaign import (  # noqa: E402
    canonical_bytes,
    file_sha256,
)
from tools.run_generator_cohort import (  # noqa: E402
    load_declaration,
    verify_stage_membership,
)


ATLAS_MAGIC = b"Q2ATL001"
ATLAS_HEADER_BYTES = 136
ATLAS_NODE_BYTES = 40
ATLAS_EDGE_BYTES = 28
COST_INFINITY = 0xFFFFFFFF
SAFE_TO_STAND = 1 << 2
MOVER_GATED_PLATEAU = 1 << 6
EDGE_MOVER = 9
EDGE_HOOK = 11
HOOK_NECESSITY_WALKING_BUDGET_TICKS = 15
HOOK_NECESSITY_GAME_TICK_HZ = 10
HOOK_NECESSITY_WALK_SPEED_Q8_PER_SECOND = 300 * 256
COMPONENT_COMMAND_TIMEOUT_SECONDS = 600.0
COMPONENT_TERMINATION_GRACE_SECONDS = 2.0
PASS_RE = re.compile(r"(?<![0-9])(\d+) passed")


class B3ComponentError(ValueError):
    """Raised before publication when concrete component evidence fails."""


def _write_new(path: Path, value: object) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise B3ComponentError("evidence output parent must already be a non-symlink directory")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise B3ComponentError(f"evidence output already exists or cannot be created: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _preflight_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise B3ComponentError(f"evidence output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise B3ComponentError("evidence output parent must already be a non-symlink directory")


def _run_commands(
    commands: Sequence[Sequence[str]],
    repo_root: Path = ROOT,
    *,
    timeout_seconds: float = COMPONENT_COMMAND_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= COMPONENT_COMMAND_TIMEOUT_SECONDS
    ):
        raise B3ComponentError("component command timeout must be finite in (0,600]")
    runs = []
    total_passed = 0
    for command in commands:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise B3ComponentError("fixed component test command is invalid")
        process = subprocess.Popen(
            list(command), cwd=repo_root, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=COMPONENT_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            raise B3ComponentError(
                f"fixed component test command timed out: {list(command)!r}"
            ) from exc
        combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
        passed = sum(int(value) for value in PASS_RE.findall(combined))
        if process.returncode != 0 or passed < 1:
            raise B3ComponentError(
                f"fixed component test command failed or reported no passing tests: {list(command)!r}"
            )
        total_passed += passed
        runs.append({
            "command": list(command), "exit_code": process.returncode,
            "passed_count": passed,
            "stdout": {"bytes": len(stdout), "sha256": hashlib.sha256(stdout).hexdigest()},
            "stderr": {"bytes": len(stderr), "sha256": hashlib.sha256(stderr).hexdigest()},
        })
    test_report = {"schema": "q2-b3-component-test-runs-v1", "runs": runs}
    return {
        "report_sha256": hashlib.sha256(canonical_bytes(test_report)).hexdigest(),
        "commands_sha256": hashlib.sha256(canonical_bytes([list(command) for command in commands])).hexdigest(),
        "passed_count": total_passed,
        "failed_count": 0, "runs": runs,
    }


def parse_atlas_recovery(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise B3ComponentError(f"raw Atlas is missing or a symlink: {path}")
    payload = path.read_bytes()
    if len(payload) > 32 * 1024 * 1024 or len(payload) < ATLAS_HEADER_BYTES:
        raise B3ComponentError("raw Atlas size is outside the admitted bound")
    if payload[:8] != ATLAS_MAGIC:
        raise B3ComponentError("raw Atlas magic differs")
    schema, byte_order, header_bytes = struct.unpack_from("<HHI", payload, 8)
    if (schema, byte_order, header_bytes) != (1, 0x454C, ATLAS_HEADER_BYTES):
        raise B3ComponentError("raw Atlas header schema differs")
    counts = struct.unpack_from("<5Q", payload, 56)
    lengths = struct.unpack_from("<5Q", payload, 96)
    if ATLAS_HEADER_BYTES + sum(lengths) != len(payload):
        raise B3ComponentError("raw Atlas section lengths do not close")
    l1_count, edge_count = counts[1], counts[2]
    if lengths[1] != l1_count * ATLAS_NODE_BYTES:
        raise B3ComponentError("raw Atlas L1 node section length differs")
    node_start = ATLAS_HEADER_BYTES + lengths[0]
    graph_start = node_start + lengths[1]
    if graph_start + lengths[2] > len(payload):
        raise B3ComponentError("raw Atlas graph section exceeds payload")
    offset_count = struct.unpack_from("<Q", payload, graph_start)[0]
    if offset_count != l1_count + 1:
        raise B3ComponentError("raw Atlas CSR offset count differs")
    expected_graph = 8 + offset_count * 4 + edge_count * ATLAS_EDGE_BYTES
    if lengths[2] != expected_graph:
        raise B3ComponentError("raw Atlas CSR byte count differs")
    offsets = struct.unpack_from(f"<{offset_count}I", payload, graph_start + 8)
    if offsets[0] != 0 or offsets[-1] != edge_count or any(
        offsets[index] > offsets[index + 1] for index in range(len(offsets) - 1)
    ):
        raise B3ComponentError("raw Atlas CSR offsets are not canonical")
    nodes = []
    for ordinal in range(l1_count):
        start = node_start + ordinal * ATLAS_NODE_BYTES
        index = struct.unpack_from("<iii", payload, start)
        flags = struct.unpack_from("<H", payload, start + 12)[0]
        cost = struct.unpack_from("<I", payload, start + 24)[0]
        nodes.append({"index": index, "flags": flags, "cost": cost})
    edge_start = graph_start + 8 + offset_count * 4
    edges = []
    for ordinal in range(edge_count):
        start = edge_start + ordinal * ATLAS_EDGE_BYTES
        target = struct.unpack_from("<I", payload, start)[0]
        edge_type = payload[start + 4]
        blocker = struct.unpack_from("<I", payload, start + 8)[0]
        if target >= l1_count:
            raise B3ComponentError("raw Atlas edge target exceeds L1 nodes")
        edges.append((target, edge_type, blocker))
    finite = descending = plateaus = hooks = 0
    for ordinal, node in enumerate(nodes):
        if node["cost"] == COST_INFINITY or node["flags"] & SAFE_TO_STAND:
            continue
        finite += 1
        if node["flags"] & MOVER_GATED_PLATEAU:
            plateaus += 1
            continue
        if any(
            edge_type not in (EDGE_HOOK, EDGE_MOVER)
            and blocker == 0
            and nodes[target]["cost"] < node["cost"]
            for target, edge_type, blocker in edges[offsets[ordinal]:offsets[ordinal + 1]]
        ):
            descending += 1
    hooks = sum(
        edge_type == EDGE_HOOK and blocker == 0
        for _target, edge_type, blocker in edges
    )
    hook_source_indices = [
        list(node["index"])
        for ordinal, node in enumerate(nodes)
        if any(
            edge_type == EDGE_HOOK and blocker == 0
            for _target, edge_type, blocker
            in edges[offsets[ordinal]:offsets[ordinal + 1]]
        )
    ]
    unresolved = finite - descending - plateaus
    return {
        "atlas_sha256": hashlib.sha256(payload).hexdigest(),
        "l1_nodes": l1_count, "finite_non_safe_cells": finite,
        "strict_descending_cells": descending, "mover_plateau_cells": plateaus,
        "unresolved_cells": unresolved, "hook_edge_count": hooks,
        "hook_source_indices": hook_source_indices,
        "first_l1_index": list(nodes[0]["index"]) if nodes else None,
    }


def _extension(extension_file: Path, repo_root: Path) -> Any:
    if extension_file.is_symlink() or not extension_file.is_file():
        raise B3ComponentError("extension file must be an exact non-symlink regular file")
    resolved = extension_file.resolve()
    admitted = {
        (repo_root / "target/debug/libq2_lattice_rs.so").resolve(),
        (repo_root / "target/debug/libq2_lattice_rs.dylib").resolve(),
        (repo_root / "target/debug/q2_lattice_rs.dll").resolve(),
    }
    if resolved not in admitted:
        raise B3ComponentError(
            "extension file must be the canonical repository target/debug artifact"
        )
    spec = importlib.util.spec_from_file_location("q2_lattice_rs", resolved)
    if spec is None or spec.loader is None:
        raise B3ComponentError("cannot construct the exact Rust extension loader")
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise B3ComponentError(f"cannot load exact Rust extension file: {exc}") from exc
    loaded = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded != resolved:
        raise B3ComponentError("loaded Rust extension identity differs from requested file")
    if not hasattr(module, "AtlasRuntime") or not hasattr(module.AtlasRuntime, "hook_necessity"):
        raise B3ComponentError("Rust extension lacks AtlasRuntime hook-necessity authority")
    return module


def produce_recovery_guide(
    *,
    declaration_path: Path,
    claims_dir: Path,
    analysis_dir: Path,
    output: Path,
    extension_file: Path,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    _preflight_output(output)
    repository = repository_identity(repo_root)
    declaration, _digest = load_declaration(declaration_path)
    claims_membership = verify_stage_membership(declaration, claims_dir, "claims")
    analysis_membership = verify_stage_membership(declaration, analysis_dir, "analysis")
    if claims_membership["passed"] is not True or analysis_membership["passed"] is not True:
        raise B3ComponentError("recovery producer requires exact claims and analysis stages")
    tests = _run_commands(RECOVERY_TEST_COMMANDS, repo_root)
    extension = _extension(extension_file, repo_root)
    extension_sha256 = file_sha256(extension_file)
    totals = CounterLike()
    artifact_records = []
    objective_fixtures = universal_zero = 0
    hook_queries = hook_paths = hook_positive = hook_evaluated = 0
    for map_epoch, declared in enumerate(declaration["maps"], start=1):
        name = declared["map"]
        atlas_path = analysis_dir / f"{name}.atlas.bin"
        manifest_path = analysis_dir / f"{name}.atlas.manifest.json"
        objectives_path = analysis_dir / f"{name}.objectives.json"
        bsp_path = claims_dir / f"{name}.bsp"
        stats = parse_atlas_recovery(atlas_path)
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_value.get("recovery_physics") != {
            "hook_walk_budget_ticks": HOOK_NECESSITY_WALKING_BUDGET_TICKS,
            "game_tick_hz": HOOK_NECESSITY_GAME_TICK_HZ,
            "walk_speed_q8_per_second": HOOK_NECESSITY_WALK_SPEED_Q8_PER_SECOND,
        }:
            raise B3ComponentError(f"Atlas recovery physics identity differs for {name}")
        try:
            runtime = extension.AtlasRuntime(
                manifest_path.read_bytes(), atlas_path.name, atlas_path.read_bytes(),
                objectives_path.name, objectives_path.read_bytes(), bsp_path.read_bytes(),
                name, map_epoch,
            )
        except Exception as exc:
            raise B3ComponentError(f"Rust AtlasRuntime rejected {name}: {exc}") from exc
        for l1_index in stats["hook_source_indices"]:
            try:
                hook = dict(runtime.hook_necessity(l1_index, map_epoch))
            except Exception as exc:
                raise B3ComponentError(
                    f"Rust hook-necessity evaluator rejected {name} {l1_index}: {exc}"
                ) from exc
            if set(hook) != {
                "walking_budget_ticks", "evaluated_hook_edges",
                "walking_reaches_safety_within_budget", "hook_path_reaches_safety",
                "hook_lowers_recovery_cost", "hook_was_necessary",
            }:
                raise B3ComponentError("Rust hook-necessity evidence keys differ")
            if hook["walking_budget_ticks"] != HOOK_NECESSITY_WALKING_BUDGET_TICKS:
                raise B3ComponentError("Rust hook-necessity walking budget differs")
            evaluated = hook["evaluated_hook_edges"]
            if isinstance(evaluated, bool) or not isinstance(evaluated, int) or evaluated < 1:
                raise B3ComponentError("Rust hook-necessity query evaluated no current hook edge")
            if any(
                not isinstance(hook[key], bool)
                for key in (
                    "walking_reaches_safety_within_budget", "hook_path_reaches_safety",
                    "hook_lowers_recovery_cost", "hook_was_necessary",
                )
            ):
                raise B3ComponentError("Rust hook-necessity booleans have invalid types")
            if hook["hook_was_necessary"] is not (
                not hook["walking_reaches_safety_within_budget"]
                and hook["hook_path_reaches_safety"]
                and hook["hook_lowers_recovery_cost"]
            ):
                raise B3ComponentError("Rust hook-necessity decision is internally inconsistent")
            hook_queries += 1
            hook_evaluated += evaluated
            hook_paths += int(hook["hook_path_reaches_safety"])
            hook_positive += int(hook["hook_was_necessary"])
        objectives = json.loads(objectives_path.read_text(encoding="utf-8"))["objectives"]
        if objectives:
            point = [value / 1000.0 for value in objectives[0]["world_milliunits"]]
            beliefs = [(item["objective_id"], 0.5) for item in objectives]
            vector = list(runtime.guide_features(point, 0.0, map_epoch, beliefs))
            if len(vector) != 60 or any(not math.isfinite(float(value)) for value in vector):
                raise B3ComponentError(f"Rust guide fixture is not finite width 60 for {name}")
            objective_fixtures += 1
            universal_zero += int(all(float(value) == 0.0 for value in vector))
        for key in (
            "finite_non_safe_cells", "strict_descending_cells",
            "mover_plateau_cells", "unresolved_cells", "hook_edge_count",
        ):
            totals[key] += int(stats[key])
        artifact_records.append({
            "map": name, "bsp_sha256": file_sha256(bsp_path),
            "atlas_sha256": stats["atlas_sha256"],
            "manifest_sha256": file_sha256(manifest_path),
            "objectives_sha256": file_sha256(objectives_path),
        })
    if totals["finite_non_safe_cells"] < 1 or totals["unresolved_cells"] != 0:
        raise B3ComponentError("measured Atlas set has no finite recovery cells or unresolved cells")
    if objective_fixtures < 1 or universal_zero != 0:
        raise B3ComponentError("measured Atlas set lacks a nonzero objective guide fixture")
    if hook_queries < 1 or hook_evaluated != totals["hook_edge_count"]:
        raise B3ComponentError(
            "measured Atlas set lacks complete non-vacuous current-edge hook evaluation"
        )
    repair_source = (repo_root / "crates/q2-lattice/src/atlas/recovery.rs").read_text()
    match = re.search(r"RECOVERY_REPAIR_NODE_LIMIT:\s*usize\s*=\s*(\d+)", repair_source)
    if match is None or int(match.group(1)) != 4096:
        raise B3ComponentError("cannot bind the Rust recovery repair-node ceiling")
    source_closure = _source_closure(repo_root, RECOVERY_GUIDE_SOURCE_PATHS)
    report = {
        "schema": RECOVERY_GUIDE_SCHEMA, "evidence_kind": "measured-offline-atlas",
        "synthetic_claims": False,
        "implementation": {**repository, "source_closure": source_closure},
        "rust_extension": {
            "path": str(extension_file.resolve()),
            "bytes": extension_file.stat().st_size,
            "sha256": extension_sha256,
            "repository_tree": repository["repository_tree"],
            "source_closure_sha256": source_closure["sha256"],
            "qualification_commands_sha256": tests["commands_sha256"],
        },
        "atlas_set_sha256": hashlib.sha256(canonical_bytes(artifact_records)).hexdigest(),
        "map_count": len(artifact_records),
        "recovery": {
            "finite_non_safe_cells": totals["finite_non_safe_cells"],
            "strict_descending_cells": totals["strict_descending_cells"],
            "mover_plateau_cells": totals["mover_plateau_cells"],
            "unresolved_cells": totals["unresolved_cells"],
            "max_local_repair_nodes": 4096, "hazard_classes": HAZARD_CLASSES,
            "hook_necessity_walking_budget_ticks": HOOK_NECESSITY_WALKING_BUDGET_TICKS,
            "hook_necessity_game_tick_hz": HOOK_NECESSITY_GAME_TICK_HZ,
            "hook_necessity_walk_speed_q8_per_second": HOOK_NECESSITY_WALK_SPEED_Q8_PER_SECOND,
            "hook_necessity_evaluated_edges": hook_evaluated,
            "hook_necessity_query_cells": hook_queries,
            "hook_necessity_path_to_safety_cases": hook_paths,
            "hook_necessity_positive_cases": hook_positive,
            "recovery_width": 16,
        },
        "guide": {
            "guide_width": 60, "candidate_count": 4, "candidate_width": 15,
            "objective_classes": OBJECTIVE_CLASSES,
            "objective_bearing_fixtures": objective_fixtures,
            "universal_zero_objective_fixtures": universal_zero,
        },
        "tests": tests, "failures": [], "passed": True,
    }
    validate_recovery_guide_evidence(report, repository, source_closure)
    _write_new(output, report)
    return report


class CounterLike(dict[str, int]):
    def __missing__(self, key: str) -> int:
        return 0


def _build_bundle_report(
    *,
    repo_root: Path,
    repository: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate an in-memory document without publishing evidence."""

    source_closure = _source_closure(repo_root, BUNDLE_SOURCE_PATHS)
    claim_tests = derive_bundle_claim_evidence(tests)
    sections = _bundle_sections_from_claims(claim_tests)
    report = {
        "schema": BUNDLE_SCHEMA, "evidence_kind": "measured-offline-installer",
        "synthetic_claims": False,
        "implementation": {**repository, "source_closure": source_closure},
        "claim_tests": claim_tests,
        "bundle_v2": sections["bundle_v2"],
        "bundle_v3": sections["bundle_v3"],
        "farm": sections["farm"],
        "tests": tests, "failures": [], "passed": True,
    }
    validate_bundle_evidence(report, repository, source_closure)
    return report


def produce_bundle(
    *,
    output: Path,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Run the fixed named-node campaign and publish measured bundle evidence."""

    _preflight_output(output)
    repository = repository_identity(repo_root)
    tests = _run_commands(BUNDLE_TEST_COMMANDS, repo_root)
    report = _build_bundle_report(
        repo_root=repo_root,
        repository=repository,
        tests=tests,
    )
    _write_new(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recovery = sub.add_parser("recovery-guide")
    recovery.add_argument("--declaration", type=Path, required=True)
    recovery.add_argument("--claims-dir", type=Path, required=True)
    recovery.add_argument("--analysis-dir", type=Path, required=True)
    recovery.add_argument("--extension-file", type=Path, required=True)
    recovery.add_argument("--output", type=Path, required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "recovery-guide":
            report = produce_recovery_guide(
                declaration_path=args.declaration, claims_dir=args.claims_dir,
                analysis_dir=args.analysis_dir, output=args.output,
                extension_file=args.extension_file,
            )
        else:
            report = produce_bundle(output=args.output)
    except (B3ComponentError, OSError, ValueError) as exc:
        print(f"B3 component evidence refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes({
        "schema": report["schema"], "output_sha256": file_sha256(args.output),
        "passed": report["passed"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
