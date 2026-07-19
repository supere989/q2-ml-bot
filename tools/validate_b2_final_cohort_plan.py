#!/usr/bin/env python3
"""Pre-authorization plan validator/orchestrator for the final B2 cohort lane.

Validates every later-stage CLI argument domain, tool/path identity, declaration
binding, stage order, retired-registry admission, and accepted timeout range
before any immutable final cohort source authorization can be consumed.  The
71446 operator defect that passed ``--oracle-batch-timeout-seconds 3600`` into a
command that accepts only ``(0, 60]`` must fail here as a runner/configuration
defect, without retiring a cohort or invoking source generation.

Dry-run is the default.  Mutating stages never execute until the full plan has
passed validation and an unambiguous mutating-execution acknowledgement is
supplied.  Validation itself never creates or authorizes a cohort.
Configuration defects rejected during pre-authorization do not consume
authorization. If source was already executed by an older or bypassed runner,
however, any later failure is terminal regardless of defect class; the
immutable no-retry contract cannot leave a consumed declaration dangling.

Retired cohort/map/seed identity is enforced via the authoritative
``tools.retired_cohort_registry`` used by the other B2 producers — not by
hard-coding any particular future cohort ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.materialize_generated_cohort import (  # noqa: E402
    MAX_MATERIALIZER_TIMEOUT_SECONDS,
)
from tools.compile_generated_cohort import MAX_MAP_TIMEOUT_SECONDS  # noqa: E402
import tools.assemble_b2_gate as b2_gate  # noqa: E402
from tools.final_execution_binding import (  # noqa: E402
    FinalExecutionBindingError,
    SourceAuthorizationJournalError,
    build_execution_binding,
    build_source_authorization_marker,
    open_secure_marker_journal,
    validate_execution_binding,
    verify_source_authorization_marker,
)
from tools.retired_cohort_registry import (  # noqa: E402
    RetiredCohortRegistryError,
    require_unretired_declaration,
)
from tools.run_compiled_cm_preflight import (  # noqa: E402
    MAX_JOBS as CM_PREFLIGHT_MAX_JOBS,
    MAX_ORACLE_BATCH_TIMEOUT_SECONDS,
)
from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    canonical_bytes,
    load_declaration,
    repository_binding,
)


PLAN_SCHEMA = "q2-b2-final-cohort-preauthorization-plan-v4"
EVIDENCE_SCHEMA = "q2-b2-final-cohort-preauthorization-evidence-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
VERSIONED_DECLARATION_NAME = re.compile(
    r"^B2-GENERATED-COHORT-([0-9]+)-DECLARATION\.json$"
)

# Complete final-lane order. Phase A is intentionally before the source
# authorization boundary; every later stage is terminal/no-retry once source
# begins.
DRIVER_STAGES = (
    "dyn-shape-preflight",
    "source",
    "compile",
    "compiled-membership",
    "compiled-static",
    "compiled-cm-preflight",
    "materialization",
    "materialized-membership",
    "claims-prepare",
    "atlas-build",
    "generated-promotion",
    "stock-campaign",
    "dyn-origin-binding",
    "dyn-execute",
    "test-suite",
    "assembly",
)
PREASSEMBLY_LIFECYCLE_SCHEMA = "q2-b2-final-lifecycle-preassembly-v1"
PREASSEMBLY_LIFECYCLE_STAGES = DRIVER_STAGES[:-1]

# Per-stage CLI argument domains accepted by the real final tools.
# Bounds that stage tools publish as module constants are imported and bound
# below so plan domains cannot silently drift from producer code.
ORACLE_BATCH_TIMEOUT_MAX = float(MAX_ORACLE_BATCH_TIMEOUT_SECONDS)
COMPILE_TIMEOUT_MAX = float(MAX_MAP_TIMEOUT_SECONDS)
MATERIALIZE_TIMEOUT_MIN = 1
MATERIALIZE_TIMEOUT_MAX = int(MAX_MATERIALIZER_TIMEOUT_SECONDS)
CM_JOBS_MIN = 1
CM_JOBS_MAX = int(CM_PREFLIGHT_MAX_JOBS)

# Explicit operator acknowledgement required before any mutating stage runs.
MUTATING_EXECUTION_ACK = "I_ACKNOWLEDGE_MUTATING_FINAL_COHORT_EXECUTION"

DEFECT_RUNNER_CONFIGURATION = "runner_configuration_defect"
DEFECT_COHORT_ARTIFACT = "cohort_artifact_failure"

TOOL_FILES = {
    "dyn-shape-preflight": "preflight_b2_dyn_invocation.py",
    "source": "run_generator_cohort.py",
    "compile": "compile_generated_cohort.py",
    "compiled-membership": "run_generator_cohort.py",
    "compiled-static": "run_compiled_static_campaign.py",
    "compiled-cm-preflight": "run_compiled_cm_preflight.py",
    "materialization": "materialize_generated_cohort.py",
    "materialized-membership": "run_generator_cohort.py",
    "claims-prepare": "run_generator_claim_campaign.py",
    "atlas-build": "run_generated_atlas_campaign.py",
    "generated-promotion": "run_generator_claim_campaign.py",
    "stock-campaign": "run_b2_stock_campaign.py",
    "dyn-origin-binding": "bind_b2_dyn_origin.py",
    "dyn-execute": "run_preflighted_b2_dyn.py",
    "test-suite": "run_b2_test_suite.py",
    "assembly": "assemble_b2_gate.py",
}

DECLARATION_STAGES = frozenset({
    "source", "compile", "compiled-membership", "compiled-static",
    "compiled-cm-preflight", "materialization", "materialized-membership",
    "claims-prepare", "atlas-build", "generated-promotion",
    "dyn-origin-binding", "dyn-execute", "assembly",
})

# Option names constructed for each stage; tested against live parsers.
STAGE_REQUIRED_OPTIONS: dict[str, tuple[str, ...]] = {
    "dyn-shape-preflight": (
        "--executable", "--repo-root", "--atlas", "--manifest", "--bsp",
        "--expected-map-id", "--expected-analyzer-authority",
        "--expected-crate-commit", "--map-epoch", "--environment-steps",
        "--samples", "--output", "--report",
    ),
    "source": (
        "--declaration", "--output-dir", "--cold-dir", "--report",
        "--final-source-authorization",
    ),
    "compile": (
        "--declaration",
        "--source-root",
        "--staging-root",
        "--publish-root",
        "--log-root",
        "--report",
        "--q2tool",
        "--basedir",
        "--timeout-seconds",
    ),
    "compiled-static": ("--declaration", "--compiled-dir", "--output"),
    "compiled-membership": (
        "--declaration", "--stage", "--directory", "--output",
    ),
    "compiled-cm-preflight": (
        "--declaration",
        "--compiled-dir",
        "--cm-oracle",
        "--output",
        "--jobs",
        "--oracle-batch-timeout-seconds",
    ),
    "materialization": (
        "--declaration",
        "--compiled-dir",
        "--stage-dir",
        "--materialized-dir",
        "--log-dir",
        "--report",
        "--cm-oracle",
        "--pmove-oracle",
        "--hook-oracle",
        "--fall-oracle",
        "--hook-parity-attestation",
        "--timeout-seconds",
    ),
    "materialized-membership": (
        "--declaration", "--stage", "--directory", "--output",
    ),
    "claims-prepare": (
        "--declaration",
        "--materialized-dir",
        "--claims-dir",
        "--output",
    ),
    "atlas-build": (
        "--declaration",
        "--claims-dir",
        "--analysis-dir",
        "--diagnostics-dir",
        "--output",
        "--client-root",
        "--lithium-root",
        "--hook-attestation",
        "--fall-oracle",
        "--packer",
        "--verifier",
    ),
    "generated-promotion": (
        "--declaration",
        "--claims-dir",
        "--analysis-dir",
        "--b1-gate",
        "--output",
    ),
    "stock-campaign": (
        "--repo-root", "--python", "--stock-pak", "--provenance",
        "--stock-inventory",
        "--b1-gate", "--client-root", "--lithium-root",
        "--hook-attestation", "--fall-oracle", "--packer", "--verifier",
        "--output-root", "--report",
    ),
    "dyn-origin-binding": (
        "--shape-preflight-report", "--generated-promotion-report",
        "--declaration", "--report",
    ),
    "dyn-execute": (
        "--shape-preflight-report", "--origin-binding-report", "--declaration",
    ),
    "test-suite": ("--output", "--python"),
    "assembly": tuple(
        f"--{name.replace('_', '-')}" for name in (
            "design", "plan", "repo_root", "b1_gate", "cm_oracle",
            "pmove_oracle", "hook_oracle", "fall_oracle", "hook_attestation",
            "atlas_verifier", "declaration", "source_dir", "source_cold_dir",
            "source_freeze_report", "compiled_dir", "compiled_membership_report",
            "compiled_static_report", "compiled_cm_preflight_report",
            "materialized_dir", "materialized_membership_report", "claims_dir",
            "claims_prepare_report", "analysis_dir", "generated_build_report",
            "generated_validation_report", "stock_provenance", "stock_inventory",
            "stock_bsp_dir", "stock_analysis_dir", "stock_validation_dir",
            "dyn_evidence_executable", "dyn_argv_preflight_report",
            "dyn_origin_binding_report", "dyn_evidence_report", "test_report",
            "preactivation_test_report", "qualification_report",
            "final_lifecycle_evidence", "output",
        )
    ),
}

STAGE_SUBCOMMANDS: dict[str, str | None] = {
    "dyn-shape-preflight": None,
    "source": "generate",
    "compile": None,
    "compiled-membership": "verify-stage",
    "compiled-static": None,
    "compiled-cm-preflight": None,
    "materialization": None,
    "materialized-membership": "verify-stage",
    "claims-prepare": "prepare",
    "atlas-build": None,
    "generated-promotion": "validate",
    "stock-campaign": None,
    "dyn-origin-binding": None,
    "dyn-execute": None,
    "test-suite": None,
    "assembly": None,
}


class FinalCohortPlanError(RuntimeError):
    """Raised when the pre-authorization plan is incomplete or invalid."""

    def __init__(
        self,
        message: str,
        *,
        defect_class: str = DEFECT_RUNNER_CONFIGURATION,
        check_id: str = "plan-validation",
        consumed: bool = False,
    ) -> None:
        super().__init__(message)
        self.defect_class = defect_class
        self.check_id = check_id
        self.consumed = consumed


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return canonical_bytes(value)


def _stock_pak_authority(
    stock_pak: Mapping[str, Any],
    provenance_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    """Bind the supplied stock PAK to both committed corpus authorities.

    This check intentionally runs while the immutable plan is being built and
    is replayed by every validation pass.  Merely pinning an arbitrary PAK
    digest is insufficient: selecting base ``pak0.pak`` instead of the retail
    deathmatch ``pak1.pak`` otherwise survives dry-run review and fails only
    after the one-shot source authorization has been consumed.
    """

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FinalCohortPlanError(
                    f"stock corpus authority contains duplicate key {key!r}",
                    check_id="stock-pak-authority",
                )
            result[key] = value
        return result

    def load(path: Path, label: str) -> Mapping[str, Any]:
        try:
            value = json.loads(
                path.read_bytes(),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise FinalCohortPlanError(
                f"cannot read {label}: {error}",
                check_id="stock-pak-authority",
            ) from error
        if not isinstance(value, Mapping):
            raise FinalCohortPlanError(
                f"{label} is not an object",
                check_id="stock-pak-authority",
            )
        return value

    provenance = load(provenance_path, "stock provenance")
    inventory = load(inventory_path, "stock inventory")
    records = provenance.get("records")
    maps = inventory.get("maps")
    if not isinstance(records, list) or not isinstance(maps, list):
        raise FinalCohortPlanError(
            "stock corpus authorities do not contain record/map lists",
            check_id="stock-pak-authority",
        )
    provenance_hashes = {
        row.get("archive_sha256")
        for row in records
        if isinstance(row, Mapping)
    }
    inventory_hash = inventory.get("archive_sha256")
    if (
        len(records) != 8
        or len(maps) != 8
        or len(provenance_hashes) != 1
        or not isinstance(inventory_hash, str)
        or provenance_hashes != {inventory_hash}
    ):
        raise FinalCohortPlanError(
            "stock PAK authorities disagree or do not bind exactly eight maps",
            check_id="stock-pak-authority",
        )
    expected = inventory_hash
    actual = stock_pak.get("sha256")
    if actual != expected:
        raise FinalCohortPlanError(
            f"stock PAK digest differs from corpus authority: expected={expected} actual={actual}",
            check_id="stock-pak-authority",
        )
    return {
        "schema": "q2-b2-stock-pak-authority-v1",
        "map_count": 8,
        "stock_pak_sha256": expected,
        "provenance_sha256": _sha256(provenance_path.read_bytes()),
        "inventory_sha256": _sha256(inventory_path.read_bytes()),
    }


def _execution_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the live-validated WSL execution/journal binding.

    The binding is part of the reviewed plan rather than an ambient home
    directory.  This makes an exact plan unusable on another LAN host or
    account before Phase A, and prevents a caller from redirecting the
    declaration tombstone to a fresh private journal.
    """

    authorization = plan.get("authorization")
    if not isinstance(authorization, Mapping):
        raise FinalCohortPlanError(
            "authorization contract is incomplete",
            check_id="execution-authorization-binding",
        )
    binding = authorization.get("execution_binding")
    if not isinstance(binding, Mapping):
        raise FinalCohortPlanError(
            "execution authorization binding is absent",
            check_id="execution-authorization-binding",
        )
    try:
        return validate_execution_binding(
            binding,
            repo_root=Path(str(plan.get("repo_root", ""))),
            workspace=Path(str(plan.get("workspace", ""))),
        )
    except FinalExecutionBindingError as error:
        raise FinalCohortPlanError(
            f"execution authorization binding rejected: {error}",
            check_id="execution-authorization-binding",
        ) from error


def _declaration_identity(plan: Mapping[str, Any]) -> tuple[str, str]:
    declaration = plan.get("declaration")
    declaration_sha256 = (
        declaration.get("sha256") if isinstance(declaration, Mapping) else None
    )
    cohort_id = plan.get("cohort_id")
    if not isinstance(declaration_sha256, str) or HEX64.fullmatch(
        declaration_sha256
    ) is None or not isinstance(cohort_id, str):
        raise FinalCohortPlanError(
            "source authorization declaration identity is malformed",
            check_id="source-authorization-journal",
        )
    return cohort_id, declaration_sha256


def _source_authorization_inputs(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, Path, Path, Path, Path, dict[str, Any]]:
    """Return the exact plan-bound inputs for the source capability marker."""

    binding = _execution_binding(plan)
    cohort_id, declaration_sha256 = _declaration_identity(plan)
    declaration = plan.get("declaration")
    roots = plan.get("stage_roots")
    implementation = plan.get("repository")
    source_step = next(
        (
            step for step in plan.get("commands", [])
            if isinstance(step, Mapping) and step.get("stage") == "source"
        ),
        None,
    )
    if (
        not isinstance(declaration, Mapping)
        or not isinstance(declaration.get("path"), str)
        or not isinstance(roots, Mapping)
        or not isinstance(roots.get("source"), str)
        or not isinstance(roots.get("source_cold"), str)
        or not isinstance(source_step, Mapping)
        or not isinstance(source_step.get("report"), str)
        or not isinstance(implementation, Mapping)
    ):
        raise FinalCohortPlanError(
            "source authorization contract is incomplete",
            check_id="source-authorization-journal",
        )
    return (
        binding,
        cohort_id,
        declaration_sha256,
        _absolute(Path(declaration["path"])),
        _absolute(Path(roots["source"])),
        _absolute(Path(roots["source_cold"])),
        _absolute(Path(source_step["report"])),
        dict(implementation),
    )


def _existing_source_authorization_marker(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Fail closed on an already-consumed declaration before Phase A runs."""

    (
        binding,
        cohort_id,
        declaration_sha256,
        _declaration,
        _source_root,
        _source_cold,
        _source_report,
        _implementation,
    ) = _source_authorization_inputs(plan)
    present = False
    try:
        with open_secure_marker_journal(
            binding,
            repo_root=Path(str(plan["repo_root"])),
            workspace=Path(str(plan["workspace"])),
        ) as journal:
            present = journal.exists(declaration_sha256)
            if not present:
                return None
            existing = journal.read(declaration_sha256)
            if existing is None:
                raise SourceAuthorizationJournalError(
                    "source authorization marker disappeared after existence check",
                    consumed=True,
                )
            verify_source_authorization_marker(
                existing,
                binding=binding,
                cohort_id=cohort_id,
                declaration_sha256=declaration_sha256,
                repo_root=Path(str(plan["repo_root"])),
                workspace=Path(str(plan["workspace"])),
            )
            journal.revalidate_path_binding()
            marker = journal.marker_path(declaration_sha256)
    except SourceAuthorizationJournalError as error:
        raise FinalCohortPlanError(
            f"declaration-scoped source authorization tombstone differs: {error}",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="source-authorization-already-consumed",
            consumed=True,
        ) from error
    except FinalExecutionBindingError as error:
        raise FinalCohortPlanError(
            f"declaration-scoped source authorization tombstone differs: {error}",
            defect_class=(
                DEFECT_COHORT_ARTIFACT if present else DEFECT_RUNNER_CONFIGURATION
            ),
            check_id=(
                "source-authorization-already-consumed"
                if present else "source-authorization-journal"
            ),
            consumed=present,
        ) from error
    return {
        "path": str(marker),
        "sha256": _sha256(existing),
        "bytes": len(existing),
    }


def _source_authorization_marker(plan: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Persist the one-shot source boundary before the source runner starts."""

    (
        binding,
        cohort_id,
        declaration_sha256,
        declaration_path,
        source_root,
        source_cold,
        source_report,
        implementation,
    ) = _source_authorization_inputs(plan)
    created: bool | None = None
    try:
        payload = build_source_authorization_marker(
            binding=binding,
            cohort_id=cohort_id,
            declaration_sha256=declaration_sha256,
            declaration_path=declaration_path,
            plan_sha256=_sha256(_canonical(plan)),
            workspace=Path(str(plan["workspace"])),
            repo_root=Path(str(plan["repo_root"])),
            implementation=implementation,
            source_output=source_root,
            source_cold=source_cold,
            source_report=source_report,
        )
    except FinalExecutionBindingError as error:
        raise FinalCohortPlanError(
            f"source authorization journal rejected: {error}",
            check_id="source-authorization-journal",
        ) from error
    try:
        with open_secure_marker_journal(
            binding,
            repo_root=Path(str(plan["repo_root"])),
            workspace=Path(str(plan["workspace"])),
        ) as journal:
            existing, created = journal.create(declaration_sha256, payload)
            verify_source_authorization_marker(
                existing,
                binding=binding,
                cohort_id=cohort_id,
                declaration_sha256=declaration_sha256,
                repo_root=Path(str(plan["repo_root"])),
                workspace=Path(str(plan["workspace"])),
            )
            journal.revalidate_path_binding()
            marker = journal.marker_path(declaration_sha256)
    except SourceAuthorizationJournalError as error:
        raise FinalCohortPlanError(
            f"source authorization journal creation was incomplete: {error}",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="source-authorization-journal-incomplete",
            consumed=True,
        ) from error
    except FinalExecutionBindingError as error:
        raise FinalCohortPlanError(
            (
                f"declaration-scoped source authorization tombstone differs: {error}"
                if created is False
                else f"source authorization journal rejected: {error}"
            ),
            defect_class=(
                DEFECT_COHORT_ARTIFACT
                if created is False else DEFECT_RUNNER_CONFIGURATION
            ),
            check_id=(
                "source-authorization-already-consumed"
                if created is False else "source-authorization-journal"
            ),
            consumed=created is False,
        ) from error
    return {
        "path": str(marker),
        "sha256": _sha256(existing),
        "bytes": len(existing),
    }, created


def _prepare_execution_workspace(plan: Mapping[str, Any]) -> None:
    """Create the only two driver-owned directories needed before Phase A."""

    workspace = _absolute(Path(str(plan.get("workspace", ""))))
    if not workspace.parent.is_dir() or workspace.parent.is_symlink():
        raise FinalCohortPlanError(
            "execution workspace parent is absent or a symlink",
            check_id="execution-workspace",
        )
    if workspace.exists() or workspace.is_symlink():
        if workspace.is_symlink() or not workspace.is_dir():
            raise FinalCohortPlanError(
                "execution workspace is not a plain directory",
                check_id="execution-workspace",
            )
    else:
        workspace.mkdir(mode=0o700)
    reports = workspace / "reports"
    unexpected = {
        path.name for path in workspace.iterdir() if path.name != reports.name
    }
    if unexpected:
        raise FinalCohortPlanError(
            f"execution workspace is not empty: {sorted(unexpected)!r}",
            check_id="execution-workspace",
        )
    if reports.exists() or reports.is_symlink():
        if reports.is_symlink() or not reports.is_dir() or any(reports.iterdir()):
            raise FinalCohortPlanError(
                "execution reports directory is not a plain empty directory",
                check_id="execution-workspace",
            )
    else:
        reports.mkdir(mode=0o700)
    directory_fd = os.open(reports, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    directory_fd = os.open(workspace, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _exclusive_canonical_write(path: Path, payload: bytes) -> None:
    """Publish a final-lane evidence record once, durably, without overwrite."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short canonical evidence write")
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_preassembly_lifecycle_evidence(
    plan: Mapping[str, Any],
    *,
    source_authorization_marker: Mapping[str, Any] | None,
    executed: Sequence[str],
    stage_executions: Sequence[Mapping[str, Any]],
    assembly_command: Sequence[str],
) -> dict[str, Any]:
    """Write the driver-only attestation consumed by the final gate assembler."""

    if list(executed) != list(PREASSEMBLY_LIFECYCLE_STAGES):
        raise FinalCohortPlanError(
            "cannot publish preassembly lifecycle evidence before every prior stage succeeds",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="preassembly-lifecycle-order",
            consumed=True,
        )
    if source_authorization_marker is None:
        raise FinalCohortPlanError(
            "cannot publish preassembly lifecycle evidence without the source marker",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="preassembly-lifecycle-marker",
            consumed=True,
        )
    roots = plan.get("stage_roots")
    declaration = plan.get("declaration")
    authorization = plan.get("authorization")
    implementation = plan.get("repository")
    assembly_invocation = plan.get("assembly_invocation")
    if (
        not isinstance(roots, Mapping)
        or not isinstance(roots.get("lifecycle_evidence"), str)
        or not isinstance(declaration, Mapping)
        or not isinstance(authorization, Mapping)
        or not isinstance(authorization.get("execution_binding"), Mapping)
        or not isinstance(implementation, Mapping)
        or not isinstance(assembly_invocation, Mapping)
    ):
        raise FinalCohortPlanError(
            "preassembly lifecycle evidence contract is incomplete",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="preassembly-lifecycle-contract",
            consumed=True,
        )
    try:
        current_assembly_invocation = _assembly_command_binding(
            assembly_command,
            repo=_direct_canonical_directory(
                Path(str(plan.get("repo_root", ""))),
                "final-cohort repository root",
            ),
        )
    except FinalCohortPlanError as error:
        raise FinalCohortPlanError(
            f"cannot publish preassembly lifecycle evidence: {error}",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="assembly-command-binding",
            consumed=True,
        ) from error
    if dict(assembly_invocation) != current_assembly_invocation:
        raise FinalCohortPlanError(
            "planned assembly command binding drifted before lifecycle publication",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="assembly-command-binding",
            consumed=True,
        )
    expected_marker_keys = {"path", "sha256", "bytes"}
    if set(source_authorization_marker) != expected_marker_keys:
        raise FinalCohortPlanError(
            "source marker evidence shape differs before assembly",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="preassembly-lifecycle-marker",
            consumed=True,
        )
    payload_value = {
        "schema": PREASSEMBLY_LIFECYCLE_SCHEMA,
        "status": "ready-for-assembly",
        "cohort_id": plan.get("cohort_id"),
        "declaration": {
            "path": declaration.get("path"),
            "sha256": declaration.get("sha256"),
        },
        "plan_sha256": _sha256(_canonical(plan)),
        "implementation": dict(implementation),
        "execution_binding": dict(authorization["execution_binding"]),
        "source_authorization_marker": dict(source_authorization_marker),
        "completed_stages": list(executed),
        "stage_executions": [dict(item) for item in stage_executions],
        "assembly_command_sha256": current_assembly_invocation[
            "command_sha256"
        ],
    }
    payload = _canonical(payload_value)
    path = _absolute(Path(roots["lifecycle_evidence"]))
    try:
        _exclusive_canonical_write(path, payload)
    except OSError as error:
        raise FinalCohortPlanError(
            f"cannot publish preassembly lifecycle evidence: {error}",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="preassembly-lifecycle-publish",
            consumed=True,
        ) from error
    return {
        "path": str(path),
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }


def _arg(command: list[str], name: str, value: Path | str | int | float) -> None:
    command.extend((name, str(value)))


def _tool(repo: Path, name: str) -> Path:
    return repo / "tools" / name


def _same_file_bytes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _require(condition: bool, message: str, *, check_id: str = "plan-validation") -> None:
    if not condition:
        raise FinalCohortPlanError(message, check_id=check_id)


def _regular_file(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    absolute = _absolute(path)
    if absolute.is_symlink() or not absolute.is_file():
        raise FinalCohortPlanError(
            f"{label} is absent or a symlink: {absolute}",
            check_id="path-identity",
        )
    if executable and not os.access(absolute, os.X_OK):
        raise FinalCohortPlanError(
            f"{label} is not executable: {absolute}",
            check_id="path-identity",
        )
    return {
        "path": str(absolute),
        "bytes": absolute.stat().st_size,
        "sha256": _file_sha256(absolute),
    }


def _direct_canonical_directory(path: Path, label: str) -> Path:
    """Require a direct spelling with no symlinked repo ancestry."""

    absolute = _absolute(path)
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FinalCohortPlanError(
            f"{label} cannot be resolved: {error}",
            check_id="path-identity",
        ) from error
    if absolute != resolved or absolute.is_symlink() or not absolute.is_dir():
        raise FinalCohortPlanError(
            f"{label} must be a direct canonical directory: {absolute}",
            check_id="path-identity",
        )
    return absolute


def _direct_assembly_tool(repo: Path) -> tuple[Path, dict[str, Any]]:
    """Bind the planned assembler to a direct repo leaf and loaded bytes."""

    tool = repo / "tools" / TOOL_FILES["assembly"]
    try:
        resolved = tool.resolve(strict=True)
    except OSError as error:
        raise FinalCohortPlanError(
            f"assembly tool cannot be resolved: {error}",
            check_id="assembly-command-binding",
        ) from error
    if tool != resolved or tool.is_symlink() or not tool.is_file():
        raise FinalCohortPlanError(
            f"assembly tool must be the direct canonical repository file: {tool}",
            check_id="assembly-command-binding",
        )
    record = _regular_file(tool, "assembly tool assemble_b2_gate.py")
    try:
        loaded = _regular_file(
            Path(b2_gate.__file__).resolve(strict=True),
            "loaded assembly gate tool",
        )
    except (OSError, TypeError) as error:
        raise FinalCohortPlanError(
            f"loaded assembly gate tool cannot be resolved: {error}",
            check_id="assembly-command-binding",
        ) from error
    if not _same_file_bytes(record, loaded):
        raise FinalCohortPlanError(
            "assembly tool bytes differ from the loaded gate assembler",
            check_id="assembly-command-binding",
        )
    return tool, record


def _assembly_command_binding(
    command: Sequence[str],
    *,
    repo: Path,
) -> dict[str, str]:
    """Compute the child assembler's exact argv digest before source exists."""

    if len(command) < 2:
        raise FinalCohortPlanError(
            "assembly command is too short",
            check_id="assembly-command-binding",
        )
    tool, _record = _direct_assembly_tool(repo)
    try:
        interpreter = Path(str(command[0])).resolve(strict=True)
    except OSError as error:
        raise FinalCohortPlanError(
            f"assembly interpreter cannot be resolved: {error}",
            check_id="assembly-command-binding",
        ) from error
    if (
        str(interpreter) != str(command[0])
        or str(tool) != str(command[1])
    ):
        raise FinalCohortPlanError(
            "assembly command must use canonical direct interpreter and tool paths",
            check_id="assembly-command-binding",
        )
    try:
        command_sha256 = b2_gate.assembly_command_sha256(
            interpreter, tool, [str(part) for part in command[2:]]
        )
    except b2_gate.B2GateError as error:
        raise FinalCohortPlanError(
            f"assembly command hash cannot be bound: {error}",
            check_id="assembly-command-binding",
        ) from error
    if command_sha256 != _sha256(_canonical(list(command))):
        raise FinalCohortPlanError(
            "planned assembly command hash differs from the assembler argv hash",
            check_id="assembly-command-binding",
        )
    return {
        "interpreter": str(interpreter),
        "tool": str(tool),
        "command_sha256": command_sha256,
    }


def _existing_dir(path: Path, label: str) -> Path:
    absolute = _absolute(path)
    if absolute.is_symlink() or not absolute.is_dir():
        raise FinalCohortPlanError(
            f"{label} is absent or a symlink: {absolute}",
            check_id="path-identity",
        )
    return absolute


def _require_versioned_final_declaration(
    repo: Path,
    declaration_path: Path,
    declaration: Mapping[str, Any],
) -> None:
    """Require the direct immutable declaration leaf, never a current alias."""

    expected_parent = _absolute(repo / "docs/multires")
    try:
        resolved = declaration_path.resolve(strict=True)
        resolved_parent = expected_parent.resolve(strict=True)
    except OSError as error:
        raise FinalCohortPlanError(
            f"versioned declaration path cannot be resolved: {error}",
            check_id="declaration-binding",
        ) from error
    matched = VERSIONED_DECLARATION_NAME.fullmatch(declaration_path.name)
    cohort_id = declaration.get("cohort_id")
    _require(
        declaration_path.is_absolute()
        and not declaration_path.is_symlink()
        and declaration_path.parent == expected_parent
        and resolved == declaration_path
        and resolved.parent == resolved_parent
        and matched is not None,
        "declaration must be the direct versioned immutable repository path",
        check_id="declaration-binding",
    )
    _require(
        isinstance(cohort_id, str)
        and matched is not None
        and cohort_id.endswith(f"_{matched.group(1)}"),
        "versioned declaration number differs from cohort identity",
        check_id="declaration-binding",
    )


def _format_bound(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _finite_open_interval(
    value: object,
    *,
    low_exclusive: float,
    high_inclusive: float,
    label: str,
    check_id: str,
) -> float:
    domain = f"({_format_bound(low_exclusive)}, {_format_bound(high_inclusive)}]"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalCohortPlanError(
            f"{label} must be a finite number in {domain}",
            check_id=check_id,
        )
    number = float(value)
    if not math.isfinite(number) or not (low_exclusive < number <= high_inclusive):
        raise FinalCohortPlanError(
            f"{label} must be finite and in {domain}; got {value!r}",
            check_id=check_id,
        )
    return number


def _integer_closed_interval(
    value: object,
    *,
    low: int,
    high: int,
    label: str,
    check_id: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalCohortPlanError(
            f"{label} must be an integer in [{low}, {high}]",
            check_id=check_id,
        )
    if not low <= value <= high:
        raise FinalCohortPlanError(
            f"{label} must be an integer in [{low}, {high}]; got {value!r}",
            check_id=check_id,
        )
    return value


def _disjoint_paths(paths: Sequence[Path], label: str) -> None:
    absolute = [_absolute(path) for path in paths]
    if len(absolute) != len(set(absolute)):
        raise FinalCohortPlanError(
            f"{label} paths are not unique",
            check_id="stage-input-binding",
        )
    for index, left in enumerate(absolute):
        for right in absolute[index + 1 :]:
            if left in right.parents or right in left.parents or left == right:
                raise FinalCohortPlanError(
                    f"{label} paths nest or collide: {left} vs {right}",
                    check_id="stage-input-binding",
                )


def _step(stage: str, command: Sequence[str], report: Path) -> dict[str, Any]:
    return {
        "stage": stage,
        "command": list(command),
        "report": str(report),
        "tool": str(Path(command[1])) if len(command) > 1 else "",
        "mutating": True,
    }


def _authorization_contract(*, consumed: bool = False) -> dict[str, Any]:
    """Authorization fields shared by plan and evidence payloads."""

    return {
        "source_authorization_consumed": consumed,
        "immutable_no_retry": True,
        "configuration_defect_does_not_retire_cohort": True,
        "passing_subset_allowed": False,
        "resume_allowed": False,
        "retired_registry_admission_required": True,
        "validation_does_not_create_or_authorize_cohort": True,
    }


def _assert_domain_constant_binding() -> None:
    """Fail closed if local domain bounds drift from stage-tool constants."""

    if MATERIALIZE_TIMEOUT_MAX != int(MAX_MATERIALIZER_TIMEOUT_SECONDS):
        raise FinalCohortPlanError(
            "materialize timeout domain drifted from materializer constant",
            check_id="domain-binding",
        )
    if CM_JOBS_MAX != int(CM_PREFLIGHT_MAX_JOBS):
        raise FinalCohortPlanError(
            "compiled-cm jobs domain drifted from CM preflight constant",
            check_id="domain-binding",
        )
    if COMPILE_TIMEOUT_MAX != float(MAX_MAP_TIMEOUT_SECONDS):
        raise FinalCohortPlanError(
            "compile timeout domain drifted from compiler constant",
            check_id="domain-binding",
        )
    if ORACLE_BATCH_TIMEOUT_MAX != float(MAX_ORACLE_BATCH_TIMEOUT_SECONDS):
        raise FinalCohortPlanError(
            "compiled-cm timeout domain drifted from preflight constant",
            check_id="domain-binding",
        )


def _stage_parser(stage: str) -> argparse.ArgumentParser:
    """Build a parse-only mirror of the live stage tool CLI surface.

    Full module introspection of inline ``main()`` parsers is impractical and
    would risk executing tool entrypoints.  These mirrors bind the option names
    and types used by the current stage tools; tests parse planned commands
    against them and re-hash tool digests so silent renames fail closed.
    """

    parser = argparse.ArgumentParser(exit_on_error=False, add_help=False)
    if stage == "dyn-shape-preflight":
        from tools.preflight_b2_dyn_invocation import _parser as dyn_shape_parser

        return dyn_shape_parser()
    if stage == "source":
        sub = parser.add_subparsers(dest="command", required=True)
        generate = sub.add_parser("generate", exit_on_error=False, add_help=False)
        generate.add_argument("--declaration", type=Path, required=True)
        generate.add_argument("--output-dir", type=Path, required=True)
        generate.add_argument("--cold-dir", type=Path, required=True)
        generate.add_argument("--report", type=Path, required=True)
        generate.add_argument("--final-source-authorization", type=Path, required=True)
        return parser
    if stage == "compile":
        parser.add_argument("--declaration", type=Path, required=True)
        parser.add_argument("--source-root", type=Path, required=True)
        parser.add_argument("--staging-root", type=Path, required=True)
        parser.add_argument("--publish-root", type=Path, required=True)
        parser.add_argument("--log-root", type=Path, required=True)
        parser.add_argument("--report", type=Path, required=True)
        parser.add_argument("--q2tool", type=Path, required=True)
        parser.add_argument("--basedir", type=Path, required=True)
        parser.add_argument("--timeout-seconds", type=float, required=True)
        return parser
    if stage in {"compiled-membership", "materialized-membership"}:
        sub = parser.add_subparsers(dest="command", required=True)
        verify = sub.add_parser(
            "verify-stage", exit_on_error=False, add_help=False
        )
        verify.add_argument("--declaration", type=Path, required=True)
        verify.add_argument(
            "--stage", choices=("compiled", "materialized"), required=True
        )
        verify.add_argument("--directory", type=Path, required=True)
        verify.add_argument("--output", type=Path, required=True)
        return parser
    if stage == "compiled-static":
        parser.add_argument("--declaration", type=Path, required=True)
        parser.add_argument("--compiled-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return parser
    if stage == "compiled-cm-preflight":
        # Prefer the real exported parser when available.
        from tools.run_compiled_cm_preflight import _parser as cm_parser

        return cm_parser()
    if stage == "materialization":
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
        parser.add_argument("--timeout-seconds", type=int, required=True)
        return parser
    if stage == "claims-prepare":
        sub = parser.add_subparsers(dest="phase", required=True)
        prepare = sub.add_parser("prepare", exit_on_error=False, add_help=False)
        prepare.add_argument("--declaration", type=Path, required=True)
        prepare.add_argument("--materialized-dir", type=Path, required=True)
        prepare.add_argument("--claims-dir", type=Path, required=True)
        prepare.add_argument("--output", type=Path, required=True)
        return parser
    if stage == "atlas-build":
        parser.add_argument("--declaration", type=Path, required=True)
        parser.add_argument("--claims-dir", type=Path, required=True)
        parser.add_argument("--analysis-dir", type=Path, required=True)
        parser.add_argument("--diagnostics-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--client-root", type=Path, required=True)
        parser.add_argument("--lithium-root", type=Path, required=True)
        parser.add_argument("--hook-attestation", type=Path, required=True)
        parser.add_argument("--fall-oracle", type=Path, required=True)
        parser.add_argument("--packer", type=Path, required=True)
        parser.add_argument("--verifier", type=Path, required=True)
        return parser
    if stage == "generated-promotion":
        sub = parser.add_subparsers(dest="phase", required=True)
        validate = sub.add_parser("validate", exit_on_error=False, add_help=False)
        validate.add_argument("--declaration", type=Path, required=True)
        validate.add_argument("--claims-dir", type=Path, required=True)
        validate.add_argument("--analysis-dir", type=Path, required=True)
        validate.add_argument("--b1-gate", type=Path, required=True)
        validate.add_argument("--output", type=Path, required=True)
        return parser
    if stage == "stock-campaign":
        from tools.run_b2_stock_campaign import _parser as stock_parser

        return stock_parser()
    if stage == "dyn-origin-binding":
        from tools.bind_b2_dyn_origin import _parser as dyn_binding_parser

        return dyn_binding_parser()
    if stage == "dyn-execute":
        from tools.run_preflighted_b2_dyn import _parser as dyn_execute_parser

        return dyn_execute_parser()
    if stage == "test-suite":
        from tools.run_b2_test_suite import _parser as test_parser

        return test_parser()
    if stage == "assembly":
        return b2_gate._parser()
    raise FinalCohortPlanError(
        f"no parser binding for stage {stage!r}",
        check_id="parser-conformance",
    )


def parse_stage_command(stage: str, command: Sequence[str]) -> argparse.Namespace:
    """Parse a planned stage command against the bound stage-tool CLI surface.

    Does not execute any stage tool.  Raises FinalCohortPlanError on option or
    subcommand drift.
    """

    if len(command) < 2:
        raise FinalCohortPlanError(
            f"{stage} command is too short for tool invocation",
            check_id="parser-conformance",
        )
    argv = [str(part) for part in command[2:]]
    expected_sub = STAGE_SUBCOMMANDS.get(stage)
    if expected_sub is not None:
        if not argv or argv[0] != expected_sub:
            raise FinalCohortPlanError(
                f"{stage} command missing subcommand {expected_sub!r}: {argv!r}",
                check_id="parser-conformance",
            )
    for option in STAGE_REQUIRED_OPTIONS[stage]:
        count = argv.count(option)
        if count != 1:
            raise FinalCohortPlanError(
                f"{stage} command must contain required option {option} exactly once; "
                f"found {count}",
                check_id="parser-conformance",
            )
    parser = _stage_parser(stage)
    try:
        return parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit) as error:
        raise FinalCohortPlanError(
            f"{stage} command failed stage-tool parser: {error}",
            check_id="parser-conformance",
        ) from error


def _require_unretired(
    declaration_path: Path,
    declaration_obj: Mapping[str, Any],
    declaration_sha256: str,
) -> None:
    try:
        require_unretired_declaration(
            declaration_path, declaration_obj, declaration_sha256
        )
    except RetiredCohortRegistryError as error:
        raise FinalCohortPlanError(
            f"retired registry admission denied: {error}",
            check_id="retired-registry",
        ) from error


def _require_active_declaration(
    declaration_path: Path,
    declaration_obj: Mapping[str, Any],
    declaration_sha256: str,
) -> None:
    """Bind the one-shot executor to the exact committed activation authority."""

    try:
        authority = b2_gate._require_active_final_authority()
    except b2_gate.B2GateError as error:
        raise FinalCohortPlanError(
            f"active final authority denied execution planning: {error}",
            check_id="active-authority",
        ) from error
    authority_path = _absolute(ROOT / authority.immutable_declaration_path)
    _require(
        declaration_path == authority_path
        and declaration_obj.get("cohort_id") == authority.cohort_id
        and declaration_sha256 == authority.declaration_sha256,
        "declaration path/cohort/digest differs from the active final authority",
        check_id="active-authority",
    )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the full final-lane command plan without executing any stage."""

    _assert_domain_constant_binding()

    repo = _direct_canonical_directory(
        args.repo_root, "final-cohort repository root"
    )
    workspace = _absolute(args.workspace)
    requested_python = _absolute(
        Path(args.python) if args.python else Path(sys.executable)
    )
    try:
        python = requested_python.resolve(strict=True)
    except OSError as error:
        raise FinalCohortPlanError(
            f"python runtime cannot be resolved: {error}",
            check_id="path-identity",
        ) from error
    declaration = _absolute(args.declaration)
    reports = workspace / "reports"
    try:
        implementation = repository_binding(repo)
    except GeneratorCohortError as error:
        raise FinalCohortPlanError(
            f"final-cohort repository binding failed: {error}",
            check_id="repository-binding",
        ) from error
    _require(
        implementation.get("git_clean") is True,
        "final-cohort repository must be clean",
        check_id="repository-binding",
    )

    # Argument domains first — catch the 71446 timeout defect before binding work.
    compile_timeout = _finite_open_interval(
        args.compile_timeout_seconds,
        low_exclusive=0.0,
        high_inclusive=COMPILE_TIMEOUT_MAX,
        label="compile timeout_seconds",
        check_id="timeout-domain",
    )
    oracle_batch_timeout = _finite_open_interval(
        args.oracle_batch_timeout_seconds,
        low_exclusive=0.0,
        high_inclusive=ORACLE_BATCH_TIMEOUT_MAX,
        label="compiled-cm oracle_batch_timeout_seconds",
        check_id="timeout-domain",
    )
    materialize_timeout = _integer_closed_interval(
        args.materialize_timeout_seconds,
        low=MATERIALIZE_TIMEOUT_MIN,
        high=MATERIALIZE_TIMEOUT_MAX,
        label="materialize timeout_seconds",
        check_id="timeout-domain",
    )
    cm_jobs = _integer_closed_interval(
        args.cm_jobs,
        low=CM_JOBS_MIN,
        high=CM_JOBS_MAX,
        label="compiled-cm jobs",
        check_id="timeout-domain",
    )
    dyn_map_epoch = _integer_closed_interval(
        args.dyn_map_epoch,
        low=1,
        high=2**63 - 1,
        label="Dyn map epoch",
        check_id="dyn-domain",
    )
    dyn_environment_steps = _integer_closed_interval(
        args.dyn_environment_steps,
        low=0,
        high=2**63 - 1,
        label="Dyn environment steps",
        check_id="dyn-domain",
    )
    dyn_samples = _integer_closed_interval(
        args.dyn_samples,
        low=2000,
        high=2**63 - 1,
        label="Dyn samples",
        check_id="dyn-domain",
    )

    resolved_repo = repo.resolve(strict=True)
    resolved_workspace = workspace.resolve(strict=False)
    if resolved_workspace != workspace:
        raise FinalCohortPlanError(
            "final-cohort workspace traverses a symlinked ancestor",
            check_id="path-identity",
        )
    try:
        resolved_workspace.relative_to(resolved_repo)
    except ValueError:
        pass
    else:
        raise FinalCohortPlanError(
            "final-cohort workspace must be outside the repository",
            check_id="path-identity",
        )
    if any("qualification" in part.lower() for part in workspace.parts):
        raise FinalCohortPlanError(
            "final-cohort workspace must not live under a qualification path",
            check_id="path-identity",
        )
    try:
        execution_binding = build_execution_binding(
            state_root=args.authorization_state_root,
            repo_root=repo,
            workspace=workspace,
        )
    except FinalExecutionBindingError as error:
        raise FinalCohortPlanError(
            f"execution authorization binding rejected: {error}",
            check_id="execution-authorization-binding",
        ) from error

    declaration_record = _regular_file(declaration, "declaration")
    try:
        declaration_obj, declaration_sha256 = load_declaration(declaration)
    except (GeneratorCohortError, OSError, ValueError) as error:
        raise FinalCohortPlanError(
            f"declaration binding failed: {error}",
            check_id="declaration-binding",
        ) from error
    _require(
        declaration_obj.get("mode") == "final",
        "declaration mode must be final",
        check_id="declaration-binding",
    )
    _require(
        declaration_obj.get("schema") == "q2-b2-generated-cohort-declaration-v1",
        "declaration schema differs",
        check_id="declaration-binding",
    )
    _require_versioned_final_declaration(repo, declaration, declaration_obj)
    maps = declaration_obj.get("maps")
    _require(
        isinstance(maps, list) and len(maps) == 28,
        "declaration must bind exactly 28 maps",
        check_id="declaration-binding",
    )
    cohort_id = declaration_obj.get("cohort_id")
    _require(
        isinstance(cohort_id, str) and cohort_id.startswith("b2g26_"),
        "declaration cohort_id is invalid",
        check_id="declaration-binding",
    )
    representative_map = str(maps[0]["map"])
    # Admission against the shared retired cohort/map/seed registry.  Validation
    # never invents a successor ID; it only refuses identities already retired.
    _require(
        declaration_sha256 == declaration_record["sha256"],
        "declaration digest disagrees with load_declaration",
        check_id="declaration-binding",
    )
    _require_unretired(declaration, declaration_obj, declaration_sha256)
    _require_active_declaration(declaration, declaration_obj, declaration_sha256)

    client_root = _existing_dir(args.client_root, "Atlas client root")
    lithium_root = _existing_dir(args.lithium_root, "Atlas lithium root")
    pinned_inputs = {
        "declaration": declaration_record,
        "python": _regular_file(python, "python runtime", executable=True),
        "q2tool": _regular_file(args.q2tool, "q2tool", executable=True),
        "base_pak": _regular_file(
            _existing_dir(args.basedir, "basedir") / "pak0.pak",
            "basedir/pak0.pak",
        ),
        "stock_pak": _regular_file(args.stock_pak, "stock PAK"),
        "design": _regular_file(args.design, "normative design"),
        "execution_plan": _regular_file(args.plan, "normative execution plan"),
        "qualification_report": _regular_file(
            args.qualification_report, "qualification report"
        ),
        "preactivation_test_report": _regular_file(
            args.preactivation_test_report, "preactivation test report"
        ),
        "stock_provenance": _regular_file(
            args.stock_provenance, "stock provenance"
        ),
        "stock_inventory": _regular_file(args.stock_inventory, "stock inventory"),
        "dyn_evidence_executable": _regular_file(
            args.dyn_evidence_executable,
            "Dyn evidence executable",
            executable=True,
        ),
        "cm_oracle": _regular_file(args.cm_oracle, "CM oracle", executable=True),
        "pmove_oracle": _regular_file(
            args.pmove_oracle, "Pmove oracle", executable=True
        ),
        "hook_oracle": _regular_file(args.hook_oracle, "hook oracle", executable=True),
        "fall_oracle": _regular_file(args.fall_oracle, "fall oracle", executable=True),
        "hook_attestation": _regular_file(
            args.hook_attestation, "hook parity attestation"
        ),
        "b1_gate": _regular_file(args.b1_gate, "B1 gate"),
        "packer": _regular_file(args.packer, "Atlas packer", executable=True),
        "verifier": _regular_file(
            args.verifier, "Atlas verifier", executable=True
        ),
        "atlas_cm_oracle": _regular_file(
            client_root / "release/q2-cm-oracle",
            "canonical Atlas CM oracle",
            executable=True,
        ),
        "atlas_pmove_oracle": _regular_file(
            client_root / "release/q2-pmove-oracle",
            "canonical Atlas Pmove oracle",
            executable=True,
        ),
        "atlas_hook_oracle": _regular_file(
            lithium_root / "tools/q2-hook-oracle",
            "canonical Atlas hook oracle",
            executable=True,
        ),
    }
    stock_pak_authority = _stock_pak_authority(
        pinned_inputs["stock_pak"],
        _absolute(args.stock_provenance),
        _absolute(args.stock_inventory),
    )
    for supplied, atlas in (
        ("cm_oracle", "atlas_cm_oracle"),
        ("pmove_oracle", "atlas_pmove_oracle"),
        ("hook_oracle", "atlas_hook_oracle"),
    ):
        if not _same_file_bytes(pinned_inputs[supplied], pinned_inputs[atlas]):
            raise FinalCohortPlanError(
                f"canonical Atlas {supplied} bytes differ from supplied authority",
                check_id="atlas-release-closure",
            )
    basedir = _existing_dir(args.basedir, "basedir")

    preactivation_path = _absolute(args.preactivation_test_report)
    try:
        preactivation_path.relative_to(workspace)
    except ValueError:
        pass
    else:
        raise FinalCohortPlanError(
            "preactivation test report must be outside the final-cohort workspace",
            check_id="path-identity",
        )
    if preactivation_path.resolve(strict=True) != preactivation_path:
        raise FinalCohortPlanError(
            "preactivation test report traverses a symlinked ancestor",
            check_id="path-identity",
        )
    try:
        preactivation_tests = b2_gate.validate_preactivation_test_binding(
            design=_absolute(args.design),
            plan=_absolute(args.plan),
            repo_root=repo,
            b1_gate=_absolute(args.b1_gate),
            qualification_report=_absolute(args.qualification_report),
            preactivation_test_report=preactivation_path,
            implementation=implementation,
        )
    except b2_gate.B2GateError as error:
        raise FinalCohortPlanError(
            f"preactivation test binding rejected: {error}",
            check_id="preactivation-test-binding",
        ) from error

    assembly_tool, assembly_tool_record = _direct_assembly_tool(repo)
    tool_records: dict[str, dict[str, Any]] = {}
    for stage, tool_name in TOOL_FILES.items():
        tool_records[stage] = (
            assembly_tool_record
            if stage == "assembly"
            else _regular_file(_tool(repo, tool_name), f"{stage} tool {tool_name}")
        )

    source_root = _absolute(args.source_root or workspace / "source")
    source_cold = _absolute(args.source_cold or workspace / "source-cold")
    compile_staging = _absolute(args.compile_staging or workspace / "compiled-staging")
    compiled = _absolute(args.compiled_root or workspace / "compiled")
    compile_logs = _absolute(args.compile_logs or workspace / "compile-logs")
    materialize_staging = _absolute(
        args.materialize_staging or workspace / "materialized-staging"
    )
    materialized = _absolute(args.materialized_root or workspace / "materialized")
    materialize_logs = _absolute(args.materialize_logs or workspace / "materialize-logs")
    claims = _absolute(args.claims_root or workspace / "claims")
    analysis = _absolute(args.analysis_root or workspace / "analysis")
    atlas_diagnostics = _absolute(
        args.atlas_diagnostics or workspace / "atlas-diagnostics"
    )
    stock = _absolute(args.stock_root or workspace / "stock")
    dyn_evidence = _absolute(args.dyn_evidence_root or workspace / "dyn-evidence")
    test_evidence = _absolute(args.test_evidence_root or workspace / "test-evidence")
    gate_output = _absolute(args.gate_output or workspace / "B2-GATE.json")

    report_paths = {
        "dyn-shape-preflight": reports / "dyn-argv-shape-preflight.json",
        "source": reports / "source-freeze.json",
        "compile": reports / "compile.json",
        "compiled-membership": reports / "compiled-membership.json",
        "compiled-static": reports / "compiled-static.json",
        "compiled-cm-preflight": reports / "compiled-cm-preflight.json",
        "materialization": reports / "materialize.json",
        "materialized-membership": reports / "materialized-membership.json",
        "claims-prepare": reports / "claims-prepare.json",
        "atlas-build": reports / "atlas-build.json",
        "generated-promotion": reports / "generated-promotion.json",
        "stock-campaign": reports / "stock-campaign.json",
        "dyn-origin-binding": reports / "dyn-origin-binding.json",
        "dyn-execute": dyn_evidence / "b2-dyn-evidence.json",
        "test-suite": test_evidence / "b2-test-report.json",
        "assembly": gate_output,
    }
    lifecycle_evidence = reports / "lifecycle-preassembly.json"
    # The source producer receives this spelling as a capability argument, but
    # the journal itself always opens and reads it relative to the bound root
    # descriptor.  No security-sensitive marker I/O follows this Path object.
    source_authorization_marker = (
        Path(str(execution_binding["state_root"]["path"]))
        / f"{declaration_sha256}.json"
    )

    # Output leaves must be exclusive and absent so a later stage cannot clobber
    # an earlier one under a mis-bound path.
    planned_outputs = [
        source_root,
        source_cold,
        compile_staging,
        compiled,
        compile_logs,
        materialize_staging,
        materialized,
        materialize_logs,
        claims,
        analysis,
        atlas_diagnostics,
        stock,
        dyn_evidence,
        test_evidence,
        gate_output,
        lifecycle_evidence,
        *(
            path
            for stage, path in report_paths.items()
            if stage not in {"dyn-execute", "test-suite", "assembly"}
        ),
    ]
    for path in planned_outputs:
        try:
            relative = path.relative_to(workspace)
        except ValueError as error:
            raise FinalCohortPlanError(
                f"planned output escapes the final-cohort workspace: {path}",
                check_id="path-identity",
            ) from error
        if not relative.parts:
            raise FinalCohortPlanError(
                "planned output cannot equal the final-cohort workspace",
                check_id="path-identity",
            )
        if path.resolve(strict=False) != path:
            raise FinalCohortPlanError(
                f"planned output traverses a symlinked ancestor: {path}",
                check_id="path-identity",
            )
    _disjoint_paths(planned_outputs, "planned output")
    for path, label in (
        (source_root, "source_root"),
        (source_cold, "source_cold"),
        (compile_staging, "compile_staging"),
        (compiled, "compiled_root"),
        (compile_logs, "compile_logs"),
        (materialize_staging, "materialize_staging"),
        (materialized, "materialized_root"),
        (materialize_logs, "materialize_logs"),
        (claims, "claims_root"),
        (analysis, "analysis_root"),
        (atlas_diagnostics, "atlas_diagnostics"),
        (stock, "stock_root"),
        (dyn_evidence, "dyn_evidence_root"),
        (test_evidence, "test_evidence_root"),
        (gate_output, "gate_output"),
        (lifecycle_evidence, "lifecycle_evidence"),
        *[
            (path, f"report:{stage}")
            for stage, path in report_paths.items()
            if stage not in {"dyn-execute", "test-suite", "assembly"}
        ],
    ):
        # Workspace itself may be fresh; parents of report leaves must exist or
        # be the workspace reports directory that dry-run does not create.
        absolute = _absolute(path)
        if absolute.exists() or absolute.is_symlink():
            raise FinalCohortPlanError(
                f"{label} must be absent before final-lane execution: {absolute}",
                check_id="path-identity",
            )

    if not workspace.parent.is_dir() or workspace.parent.is_symlink():
        raise FinalCohortPlanError(
            f"workspace parent is absent or a symlink: {workspace.parent}",
            check_id="path-identity",
        )
    if workspace.exists() and (workspace.is_symlink() or not workspace.is_dir()):
        raise FinalCohortPlanError(
            f"workspace is not a usable directory: {workspace}",
            check_id="path-identity",
        )

    commands: list[dict[str, Any]] = []

    command = [
        str(python),
        str(_tool(repo, TOOL_FILES["dyn-shape-preflight"])),
    ]
    for name, value in (
        ("--executable", _absolute(args.dyn_evidence_executable)),
        ("--repo-root", repo),
        ("--atlas", analysis / f"{representative_map}.atlas.bin"),
        ("--manifest", analysis / f"{representative_map}.atlas.manifest.json"),
        ("--bsp", claims / f"{representative_map}.bsp"),
        ("--expected-map-id", representative_map),
        (
            "--expected-analyzer-authority",
            implementation["atlas_analyzer_authority_sha256"],
        ),
        ("--expected-crate-commit", implementation["repository_commit"]),
        ("--map-epoch", dyn_map_epoch),
        ("--environment-steps", dyn_environment_steps),
        ("--samples", dyn_samples),
        ("--output", dyn_evidence),
        ("--report", report_paths["dyn-shape-preflight"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step(
            "dyn-shape-preflight",
            command,
            report_paths["dyn-shape-preflight"],
        )
    )

    command = [
        str(python),
        str(_tool(repo, TOOL_FILES["source"])),
        "generate",
    ]
    for name, value in (
        ("--declaration", declaration),
        ("--output-dir", source_root),
        ("--cold-dir", source_cold),
        ("--report", report_paths["source"]),
        ("--final-source-authorization", source_authorization_marker),
    ):
        _arg(command, name, value)
    commands.append(_step("source", command, report_paths["source"]))

    command = [str(python), str(_tool(repo, TOOL_FILES["compile"]))]
    for name, value in (
        ("--declaration", declaration),
        ("--source-root", source_root),
        ("--staging-root", compile_staging),
        ("--publish-root", compiled),
        ("--log-root", compile_logs),
        ("--report", report_paths["compile"]),
        ("--q2tool", _absolute(args.q2tool)),
        ("--basedir", basedir),
        ("--timeout-seconds", compile_timeout),
    ):
        _arg(command, name, value)
    commands.append(_step("compile", command, report_paths["compile"]))

    command = [
        str(python),
        str(_tool(repo, TOOL_FILES["compiled-membership"])),
        "verify-stage",
    ]
    for name, value in (
        ("--declaration", declaration),
        ("--stage", "compiled"),
        ("--directory", compiled),
        ("--output", report_paths["compiled-membership"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step(
            "compiled-membership", command, report_paths["compiled-membership"]
        )
    )

    command = [str(python), str(_tool(repo, TOOL_FILES["compiled-static"]))]
    for name, value in (
        ("--declaration", declaration),
        ("--compiled-dir", compiled),
        ("--output", report_paths["compiled-static"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step("compiled-static", command, report_paths["compiled-static"])
    )

    command = [str(python), str(_tool(repo, TOOL_FILES["compiled-cm-preflight"]))]
    for name, value in (
        ("--declaration", declaration),
        ("--compiled-dir", compiled),
        ("--cm-oracle", _absolute(args.cm_oracle)),
        ("--output", report_paths["compiled-cm-preflight"]),
        ("--jobs", cm_jobs),
        ("--oracle-batch-timeout-seconds", oracle_batch_timeout),
    ):
        _arg(command, name, value)
    commands.append(
        _step(
            "compiled-cm-preflight",
            command,
            report_paths["compiled-cm-preflight"],
        )
    )

    command = [str(python), str(_tool(repo, TOOL_FILES["materialization"]))]
    for name, value in (
        ("--declaration", declaration),
        ("--compiled-dir", compiled),
        ("--stage-dir", materialize_staging),
        ("--materialized-dir", materialized),
        ("--log-dir", materialize_logs),
        ("--report", report_paths["materialization"]),
        ("--cm-oracle", _absolute(args.cm_oracle)),
        ("--pmove-oracle", _absolute(args.pmove_oracle)),
        ("--hook-oracle", _absolute(args.hook_oracle)),
        ("--fall-oracle", _absolute(args.fall_oracle)),
        ("--hook-parity-attestation", _absolute(args.hook_attestation)),
        ("--timeout-seconds", materialize_timeout),
    ):
        _arg(command, name, value)
    commands.append(
        _step("materialization", command, report_paths["materialization"])
    )

    command = [
        str(python),
        str(_tool(repo, TOOL_FILES["materialized-membership"])),
        "verify-stage",
    ]
    for name, value in (
        ("--declaration", declaration),
        ("--stage", "materialized"),
        ("--directory", materialized),
        ("--output", report_paths["materialized-membership"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step(
            "materialized-membership",
            command,
            report_paths["materialized-membership"],
        )
    )

    command = [
        str(python),
        str(_tool(repo, TOOL_FILES["claims-prepare"])),
        "prepare",
    ]
    for name, value in (
        ("--declaration", declaration),
        ("--materialized-dir", materialized),
        ("--claims-dir", claims),
        ("--output", report_paths["claims-prepare"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step("claims-prepare", command, report_paths["claims-prepare"])
    )

    command = [str(python), str(_tool(repo, TOOL_FILES["atlas-build"]))]
    for name, value in (
        ("--declaration", declaration),
        ("--claims-dir", claims),
        ("--analysis-dir", analysis),
        ("--diagnostics-dir", atlas_diagnostics),
        ("--output", report_paths["atlas-build"]),
        ("--hook-attestation", _absolute(args.hook_attestation)),
        ("--fall-oracle", _absolute(args.fall_oracle)),
    ):
        _arg(command, name, value)
    _arg(command, "--client-root", client_root)
    _arg(command, "--lithium-root", lithium_root)
    _arg(command, "--packer", _absolute(args.packer))
    _arg(command, "--verifier", _absolute(args.verifier))
    commands.append(_step("atlas-build", command, report_paths["atlas-build"]))

    command = [
        str(python),
        str(_tool(repo, TOOL_FILES["generated-promotion"])),
        "validate",
    ]
    for name, value in (
        ("--declaration", declaration),
        ("--claims-dir", claims),
        ("--analysis-dir", analysis),
        ("--b1-gate", _absolute(args.b1_gate)),
        ("--output", report_paths["generated-promotion"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step("generated-promotion", command, report_paths["generated-promotion"])
    )

    command = [str(python), str(_tool(repo, TOOL_FILES["stock-campaign"]))]
    for name, value in (
        ("--repo-root", repo),
        ("--python", python),
        ("--stock-pak", _absolute(args.stock_pak)),
        ("--provenance", _absolute(args.stock_provenance)),
        ("--stock-inventory", _absolute(args.stock_inventory)),
        ("--b1-gate", _absolute(args.b1_gate)),
        ("--client-root", client_root),
        ("--lithium-root", lithium_root),
        ("--hook-attestation", _absolute(args.hook_attestation)),
        ("--fall-oracle", _absolute(args.fall_oracle)),
        ("--packer", _absolute(args.packer)),
        ("--verifier", _absolute(args.verifier)),
        ("--output-root", stock),
        ("--report", report_paths["stock-campaign"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step("stock-campaign", command, report_paths["stock-campaign"])
    )

    command = [
        str(python), str(_tool(repo, TOOL_FILES["dyn-origin-binding"]))
    ]
    for name, value in (
        ("--shape-preflight-report", report_paths["dyn-shape-preflight"]),
        ("--generated-promotion-report", report_paths["generated-promotion"]),
        ("--declaration", declaration),
        ("--report", report_paths["dyn-origin-binding"]),
    ):
        _arg(command, name, value)
    commands.append(
        _step(
            "dyn-origin-binding", command, report_paths["dyn-origin-binding"]
        )
    )

    command = [str(python), str(_tool(repo, TOOL_FILES["dyn-execute"]))]
    for name, value in (
        ("--shape-preflight-report", report_paths["dyn-shape-preflight"]),
        ("--origin-binding-report", report_paths["dyn-origin-binding"]),
        ("--declaration", declaration),
    ):
        _arg(command, name, value)
    commands.append(_step("dyn-execute", command, report_paths["dyn-execute"]))

    command = [str(python), str(_tool(repo, TOOL_FILES["test-suite"]))]
    for name, value in (
        ("--output", test_evidence),
        ("--python", python),
    ):
        _arg(command, name, value)
    commands.append(_step("test-suite", command, report_paths["test-suite"]))

    command = [str(python), str(assembly_tool)]
    for name, value in (
        ("--design", _absolute(args.design)),
        ("--plan", _absolute(args.plan)),
        ("--repo-root", repo),
        ("--b1-gate", _absolute(args.b1_gate)),
        ("--cm-oracle", _absolute(args.cm_oracle)),
        ("--pmove-oracle", _absolute(args.pmove_oracle)),
        ("--hook-oracle", _absolute(args.hook_oracle)),
        ("--fall-oracle", _absolute(args.fall_oracle)),
        ("--hook-attestation", _absolute(args.hook_attestation)),
        ("--atlas-verifier", _absolute(args.verifier)),
        ("--declaration", declaration),
        ("--source-dir", source_root),
        ("--source-cold-dir", source_cold),
        ("--source-freeze-report", report_paths["source"]),
        ("--compiled-dir", compiled),
        ("--compiled-membership-report", report_paths["compiled-membership"]),
        ("--compiled-static-report", report_paths["compiled-static"]),
        (
            "--compiled-cm-preflight-report",
            report_paths["compiled-cm-preflight"],
        ),
        ("--materialized-dir", materialized),
        (
            "--materialized-membership-report",
            report_paths["materialized-membership"],
        ),
        ("--claims-dir", claims),
        ("--claims-prepare-report", report_paths["claims-prepare"]),
        ("--analysis-dir", analysis),
        ("--generated-build-report", report_paths["atlas-build"]),
        (
            "--generated-validation-report",
            report_paths["generated-promotion"],
        ),
        ("--stock-provenance", _absolute(args.stock_provenance)),
        ("--stock-inventory", _absolute(args.stock_inventory)),
        ("--stock-bsp-dir", stock / "bsp"),
        ("--stock-analysis-dir", stock / "analysis"),
        ("--stock-validation-dir", stock / "validation"),
        ("--dyn-evidence-executable", _absolute(args.dyn_evidence_executable)),
        ("--dyn-argv-preflight-report", report_paths["dyn-shape-preflight"]),
        ("--dyn-origin-binding-report", report_paths["dyn-origin-binding"]),
        ("--dyn-evidence-report", report_paths["dyn-execute"]),
        ("--preactivation-test-report", preactivation_path),
        ("--test-report", report_paths["test-suite"]),
        ("--qualification-report", _absolute(args.qualification_report)),
        ("--final-lifecycle-evidence", lifecycle_evidence),
        ("--output", gate_output),
    ):
        _arg(command, name, value)
    commands.append(_step("assembly", command, report_paths["assembly"]))
    assembly_invocation = _assembly_command_binding(command, repo=repo)

    plan = {
        "schema": PLAN_SCHEMA,
        "cohort_id": cohort_id,
        "repo_root": str(repo),
        "workspace": str(workspace),
        "repository": implementation,
        "declaration": {
            "path": declaration_record["path"],
            "sha256": declaration_record["sha256"],
            "bytes": declaration_record["bytes"],
            "map_count": 28,
            "mode": "final",
        },
        "argument_domains": {
            "compile.timeout_seconds": {
                "domain": "(0, 86400]",
                "value": compile_timeout,
            },
            "compiled_cm_preflight.oracle_batch_timeout_seconds": {
                "domain": "(0, 60]",
                "value": oracle_batch_timeout,
            },
            "materialization.timeout_seconds": {
                "domain": "[1, 3600]",
                "value": materialize_timeout,
            },
            "compiled_cm_preflight.jobs": {
                "domain": f"[{CM_JOBS_MIN}, {CM_JOBS_MAX}]",
                "value": cm_jobs,
            },
            "dyn.map_epoch": {
                "domain": "[1, 2^63-1]",
                "value": dyn_map_epoch,
            },
            "dyn.environment_steps": {
                "domain": "[0, 2^63-1]",
                "value": dyn_environment_steps,
            },
            "dyn.samples": {
                "domain": "[2000, 2^63-1]",
                "value": dyn_samples,
            },
        },
        "pinned_inputs": pinned_inputs,
        "stock_pak_authority": stock_pak_authority,
        "preactivation_tests": preactivation_tests,
        "tools": tool_records,
        "assembly_invocation": assembly_invocation,
        "stage_roots": {
            "source": str(source_root),
            "source_cold": str(source_cold),
            "compile_staging": str(compile_staging),
            "compiled": str(compiled),
            "compile_logs": str(compile_logs),
            "materialize_staging": str(materialize_staging),
            "materialized": str(materialized),
            "materialize_logs": str(materialize_logs),
            "claims": str(claims),
            "analysis": str(analysis),
            "atlas_diagnostics": str(atlas_diagnostics),
            "stock": str(stock),
            "stock_bsp": str(stock / "bsp"),
            "stock_analysis": str(stock / "analysis"),
            "stock_validation": str(stock / "validation"),
            "dyn_evidence": str(dyn_evidence),
            "test_evidence": str(test_evidence),
            "gate_output": str(gate_output),
            "lifecycle_evidence": str(lifecycle_evidence),
        },
        "commands": commands,
        "authorization": {
            **_authorization_contract(consumed=False),
            "execution_binding": execution_binding,
        },
    }
    # Parse-only conformance against stage CLI surfaces before returning a plan.
    for step in commands:
        parse_stage_command(str(step["stage"]), step["command"])
    return plan


def validate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Re-check plan structure, stage order, domains, and declaration binding."""

    checks: list[dict[str, Any]] = []

    def record(check_id: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "passed": passed,
                "detail": detail,
                "defect_class": (
                    None if passed else DEFECT_RUNNER_CONFIGURATION
                ),
            }
        )
        if not passed:
            raise FinalCohortPlanError(detail, check_id=check_id)

    record(
        "schema",
        plan.get("schema") == PLAN_SCHEMA,
        f"plan schema differs: {plan.get('schema')!r}",
    )
    stages = [step.get("stage") for step in plan.get("commands", [])]
    record(
        "stage-order",
        stages == list(DRIVER_STAGES),
        f"final-lane stage order differs: {stages!r}",
    )
    record(
        "stage-mutation-contract",
        [step.get("mutating") for step in plan.get("commands", [])]
        == [True] * len(DRIVER_STAGES),
        "every lifecycle stage must remain explicitly mutating",
    )
    try:
        direct_repo = _direct_canonical_directory(
            Path(str(plan.get("repo_root", ""))),
            "final-cohort repository root",
        )
        record(
            "repository-path-identity",
            str(direct_repo) == plan.get("repo_root"),
            "final-cohort repository root is not a direct canonical path",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))
    try:
        live_repository = repository_binding(direct_repo)
    except GeneratorCohortError as error:
        raise FinalCohortPlanError(
            f"final-cohort repository binding failed: {error}",
            check_id="repository-binding",
        ) from error
    record(
        "repository-binding",
        isinstance(plan.get("repository"), Mapping)
        and dict(plan["repository"]) == live_repository
        and plan["repository"].get("git_clean") is True,
        "final-lane repository binding differs or is not clean",
    )
    declaration = plan.get("declaration")
    record(
        "declaration-binding",
        isinstance(declaration, Mapping)
        and isinstance(declaration.get("path"), str)
        and isinstance(declaration.get("sha256"), str)
        and HEX64.fullmatch(str(declaration.get("sha256"))) is not None
        and declaration.get("mode") == "final"
        and declaration.get("map_count") == 28,
        "declaration binding is incomplete",
    )
    declaration_path = Path(str(declaration["path"]))
    live = _regular_file(declaration_path, "declaration")
    record(
        "declaration-digest",
        live["sha256"] == declaration["sha256"]
        and live["bytes"] == declaration["bytes"],
        "declaration digest drifted after plan construction",
    )
    try:
        declaration_obj, declaration_sha256 = load_declaration(declaration_path)
    except (GeneratorCohortError, OSError, ValueError) as error:
        record("declaration-load", False, f"declaration reload failed: {error}")
        return checks
    try:
        _require_versioned_final_declaration(
            Path(str(plan["repo_root"])), declaration_path, declaration_obj
        )
        record(
            "declaration-versioned-path",
            True,
            "declaration is the direct versioned immutable repository path",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))
    record(
        "declaration-identity",
        declaration_obj.get("cohort_id") == plan.get("cohort_id")
        and declaration_obj.get("mode") == "final",
        "declaration cohort identity binding differs",
    )
    try:
        _require_unretired(declaration_path, declaration_obj, declaration_sha256)
        record(
            "retired-registry",
            True,
            "declaration is not retired by the authoritative registry",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    try:
        _require_active_declaration(
            declaration_path, declaration_obj, declaration_sha256
        )
        record(
            "active-authority",
            True,
            "declaration path/cohort/digest equals the active final authority",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    try:
        _assert_domain_constant_binding()
        record(
            "domain-binding",
            True,
            "timeout/jobs domains match stage-tool constants",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    domains = plan.get("argument_domains")
    record(
        "argument-domains-present",
        isinstance(domains, Mapping)
        and {
            "compile.timeout_seconds",
            "compiled_cm_preflight.oracle_batch_timeout_seconds",
            "materialization.timeout_seconds",
            "compiled_cm_preflight.jobs",
            "dyn.map_epoch",
            "dyn.environment_steps",
            "dyn.samples",
        }.issubset(domains),
        "argument domain table is incomplete",
    )
    oracle = domains["compiled_cm_preflight.oracle_batch_timeout_seconds"]
    try:
        _finite_open_interval(
            oracle.get("value"),
            low_exclusive=0.0,
            high_inclusive=ORACLE_BATCH_TIMEOUT_MAX,
            label="compiled-cm oracle_batch_timeout_seconds",
            check_id="timeout-domain",
        )
        record(
            "timeout-domain-oracle-batch",
            True,
            "oracle batch timeout is within (0, 60]",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    compile_domain = domains["compile.timeout_seconds"]
    _finite_open_interval(
        compile_domain.get("value"),
        low_exclusive=0.0,
        high_inclusive=COMPILE_TIMEOUT_MAX,
        label="compile timeout_seconds",
        check_id="timeout-domain",
    )
    record("timeout-domain-compile", True, "compile timeout is within (0, 86400]")

    materialize_domain = domains["materialization.timeout_seconds"]
    _integer_closed_interval(
        materialize_domain.get("value"),
        low=MATERIALIZE_TIMEOUT_MIN,
        high=MATERIALIZE_TIMEOUT_MAX,
        label="materialize timeout_seconds",
        check_id="timeout-domain",
    )
    record(
        "timeout-domain-materialize",
        True,
        "materialize timeout is within [1, 3600]",
    )

    for name, low in (
        ("dyn.map_epoch", 1),
        ("dyn.environment_steps", 0),
        ("dyn.samples", 2000),
    ):
        _integer_closed_interval(
            domains[name].get("value"),
            low=low,
            high=2**63 - 1,
            label=name,
            check_id="dyn-domain",
        )
        record(
            f"dyn-domain-{name.removeprefix('dyn.')}",
            True,
            f"{name} is within the admitted domain",
        )

    authorization = plan.get("authorization")
    expected_auth = _authorization_contract(consumed=False)
    record(
        "authorization-contract",
        isinstance(authorization, Mapping)
        and all(authorization.get(key) == value for key, value in expected_auth.items())
        and isinstance(authorization.get("execution_binding"), Mapping),
        "authorization contract weakened or incomplete",
    )
    try:
        _execution_binding(plan)
        record(
            "execution-authorization-binding",
            True,
            "final execution host, UID, machine identity, and journal root remain bound",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    tools = plan.get("tools")
    record(
        "tool-digest-table",
        isinstance(tools, Mapping) and set(DRIVER_STAGES).issubset(tools),
        "tool digest table incomplete",
    )
    for stage in DRIVER_STAGES:
        tool_record = tools[stage]
        live_tool = _regular_file(
            Path(str(tool_record["path"])), f"{stage} tool"
        )
        record(
            f"tool-digest:{stage}",
            live_tool["sha256"] == tool_record["sha256"]
            and live_tool["bytes"] == tool_record["bytes"]
            and Path(str(tool_record["path"])).name == TOOL_FILES[stage],
            f"tool digest drifted for {stage}",
        )
    try:
        direct_assembly_tool, direct_assembly_record = _direct_assembly_tool(
            direct_repo
        )
        record(
            "assembly-tool-identity",
            tools["assembly"] == direct_assembly_record,
            "assembly tool record differs from the direct canonical repository assembler",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    declaration_flag = str(declaration["path"])
    for step in plan["commands"]:
        stage = step["stage"]
        command = step["command"]
        record(
            f"tool-identity:{stage}",
            isinstance(command, list)
            and len(command) >= 2
            and Path(command[1]).name == TOOL_FILES[stage]
            and Path(command[1]).is_file()
            and not Path(command[1]).is_symlink(),
            f"{stage} tool identity differs",
        )
        if stage in DECLARATION_STAGES:
            if "--declaration" not in command:
                record(
                    f"stage-input-binding:{stage}",
                    False,
                    f"{stage} command lacks --declaration",
                )
            decl_index = command.index("--declaration")
            bound = command[decl_index + 1]
            record(
                f"stage-input-binding:{stage}",
                bound == declaration_flag,
                f"{stage} declaration binding differs: {bound!r}",
            )
        try:
            parse_stage_command(str(stage), command)
            record(
                f"parser-conformance:{stage}",
                True,
                f"{stage} command parses against bound stage-tool CLI",
            )
        except FinalCohortPlanError as error:
            record(error.check_id if error.check_id.startswith("parser") else f"parser-conformance:{stage}", False, str(error))
        if stage == "compiled-cm-preflight":
            flag = "--oracle-batch-timeout-seconds"
            record(
                "timeout-flag-oracle-batch",
                flag in command,
                "compiled-cm command missing oracle-batch timeout flag",
            )
            value = float(command[command.index(flag) + 1])
            record(
                "timeout-value-oracle-batch",
                0.0 < value <= ORACLE_BATCH_TIMEOUT_MAX,
                (
                    "compiled-cm oracle-batch-timeout-seconds out of range "
                    f"(0, 60]: {value!r}"
                ),
            )
        if stage == "compile":
            flag = "--timeout-seconds"
            value = float(command[command.index(flag) + 1])
            record(
                "timeout-value-compile",
                0.0 < value <= COMPILE_TIMEOUT_MAX,
                f"compile timeout out of range (0, 86400]: {value!r}",
            )
        if stage == "materialization":
            flag = "--timeout-seconds"
            value = int(command[command.index(flag) + 1])
            record(
                "timeout-value-materialize",
                MATERIALIZE_TIMEOUT_MIN <= value <= MATERIALIZE_TIMEOUT_MAX,
                f"materialize timeout out of range [1, {MATERIALIZE_TIMEOUT_MAX}]: {value!r}",
            )

    # Stage input chaining: later roots named in commands must match plan roots.
    roots = plan.get("stage_roots")
    record(
        "stage-roots-present",
        isinstance(roots, Mapping)
        and {
            "source",
            "source_cold",
            "compile_staging",
            "compiled",
            "compile_logs",
            "materialize_staging",
            "materialized",
            "materialize_logs",
            "claims",
            "analysis",
            "atlas_diagnostics",
            "stock",
            "stock_bsp",
            "stock_analysis",
            "stock_validation",
            "dyn_evidence",
            "test_evidence",
            "gate_output",
            "lifecycle_evidence",
        }.issubset(roots),
        "stage roots are incomplete",
    )
    compile_cmd = next(
        step["command"] for step in plan["commands"] if step["stage"] == "compile"
    )
    record(
        "stage-input-binding:compile-source",
        compile_cmd[compile_cmd.index("--source-root") + 1] == roots["source"],
        "compile source-root does not match source stage output",
    )
    compiled_membership_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "compiled-membership"
    )
    record(
        "stage-input-binding:compiled-membership",
        compiled_membership_cmd[
            compiled_membership_cmd.index("--directory") + 1
        ] == roots["compiled"]
        and compiled_membership_cmd[
            compiled_membership_cmd.index("--stage") + 1
        ] == "compiled",
        "compiled membership does not consume the compiled publication root",
    )
    cm_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "compiled-cm-preflight"
    )
    record(
        "stage-input-binding:cm-compiled",
        cm_cmd[cm_cmd.index("--compiled-dir") + 1] == roots["compiled"],
        "compiled-cm compiled-dir does not match compile publication root",
    )
    mat_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "materialization"
    )
    record(
        "stage-input-binding:materialize-compiled",
        mat_cmd[mat_cmd.index("--compiled-dir") + 1] == roots["compiled"],
        "materialization compiled-dir does not match compile publication root",
    )
    materialized_membership_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "materialized-membership"
    )
    record(
        "stage-input-binding:materialized-membership",
        materialized_membership_cmd[
            materialized_membership_cmd.index("--directory") + 1
        ] == roots["materialized"]
        and materialized_membership_cmd[
            materialized_membership_cmd.index("--stage") + 1
        ] == "materialized",
        "materialized membership does not consume the materialized root",
    )
    claims_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "claims-prepare"
    )
    record(
        "stage-input-binding:claims-materialized",
        claims_cmd[claims_cmd.index("--materialized-dir") + 1]
        == roots["materialized"],
        "claims prepare materialized-dir does not match materialization output",
    )
    atlas_cmd = next(
        step["command"] for step in plan["commands"] if step["stage"] == "atlas-build"
    )
    record(
        "stage-input-binding:atlas-claims",
        atlas_cmd[atlas_cmd.index("--claims-dir") + 1] == roots["claims"],
        "atlas claims-dir does not match claims stage output",
    )
    promo_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "generated-promotion"
    )
    record(
        "stage-input-binding:promotion-analysis",
        promo_cmd[promo_cmd.index("--analysis-dir") + 1] == roots["analysis"]
        and promo_cmd[promo_cmd.index("--claims-dir") + 1] == roots["claims"],
        "promotion claims/analysis binding differs from upstream stages",
    )

    pinned = plan.get("pinned_inputs")
    record(
        "pinned-inputs",
        isinstance(pinned, Mapping)
        and {
            "declaration",
            "python",
            "q2tool",
            "base_pak",
            "stock_pak",
            "design",
            "execution_plan",
            "qualification_report",
            "preactivation_test_report",
            "stock_provenance",
            "stock_inventory",
            "dyn_evidence_executable",
            "cm_oracle",
            "pmove_oracle",
            "hook_oracle",
            "fall_oracle",
            "hook_attestation",
            "b1_gate",
            "packer",
            "verifier",
            "atlas_cm_oracle",
            "atlas_pmove_oracle",
            "atlas_hook_oracle",
        }.issubset(pinned),
        "pinned inputs incomplete",
    )
    for name, record_value in pinned.items():
        path = Path(str(record_value["path"]))
        live_record = _regular_file(
            path,
            name,
            executable=name
            in {
                "python",
                "q2tool",
                "cm_oracle",
                "pmove_oracle",
                "hook_oracle",
                "fall_oracle",
                "packer",
                "verifier",
                "atlas_cm_oracle",
                "atlas_pmove_oracle",
                "atlas_hook_oracle",
                "dyn_evidence_executable",
            },
        )
        record(
            f"pinned-digest:{name}",
            live_record["sha256"] == record_value["sha256"],
            f"pinned input drifted: {name}",
        )

    try:
        stock_pak_authority = _stock_pak_authority(
            pinned["stock_pak"],
            Path(str(pinned["stock_provenance"]["path"])),
            Path(str(pinned["stock_inventory"]["path"])),
        )
        record(
            "stock-pak-authority",
            plan.get("stock_pak_authority") == stock_pak_authority,
            "stock PAK authority binding differs",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))

    pinned_paths = {
        name: str(_absolute(Path(str(value["path"]))))
        for name, value in pinned.items()
    }
    preactivation_path = Path(pinned_paths["preactivation_test_report"])
    workspace_path = _absolute(Path(str(plan["workspace"])))
    try:
        preactivation_path.relative_to(workspace_path)
    except ValueError:
        record(
            "preactivation-test-location",
            True,
            "preactivation test evidence remains outside the final workspace",
        )
    else:
        record(
            "preactivation-test-location",
            False,
            "preactivation test evidence must be outside the final workspace",
        )
    try:
        preactivation_binding = b2_gate.validate_preactivation_test_binding(
            design=Path(pinned_paths["design"]),
            plan=Path(pinned_paths["execution_plan"]),
            repo_root=Path(str(plan["repo_root"])),
            b1_gate=Path(pinned_paths["b1_gate"]),
            qualification_report=Path(pinned_paths["qualification_report"]),
            preactivation_test_report=Path(
                pinned_paths["preactivation_test_report"]
            ),
            implementation=live_repository,
        )
        record(
            "preactivation-test-binding",
            plan.get("preactivation_tests") == preactivation_binding,
            "preactivation test binding differs from the qualified implementation",
        )
    except b2_gate.B2GateError as error:
        record(
            "preactivation-test-binding",
            False,
            f"preactivation test binding rejected: {error}",
        )

    # A digest table is not sufficient by itself: every mutating command must
    # still consume the exact paths represented by that table when execution
    # revalidates the plan immediately before writing the source journal.
    for step in plan["commands"]:
        stage = str(step["stage"])
        command = step["command"]
        record(
            f"stage-input-binding:{stage}-python",
            command[0] == pinned_paths["python"],
            f"{stage} Python runtime does not match the pinned runtime",
        )
        record(
            f"stage-input-binding:{stage}-tool",
            command[1] == str(tools[stage]["path"]),
            f"{stage} command does not match its digest-pinned tool",
        )

    def record_flag_binding(
        check_id: str,
        command: Sequence[str],
        flag: str,
        expected: str,
        detail: str,
    ) -> None:
        record(
            check_id,
            flag in command and command[command.index(flag) + 1] == expected,
            detail,
        )

    commands_by_stage = {
        str(step["stage"]): step["command"] for step in plan["commands"]
    }
    reports_by_stage = {
        str(step["stage"]): str(step["report"]) for step in plan["commands"]
    }
    authorization_binding = plan["authorization"]["execution_binding"]
    expected_source_marker = str(
        Path(str(authorization_binding["state_root"]["path"]))
        / f"{declaration['sha256']}.json"
    )
    for stage, flag, expected in (
        ("source", "--output-dir", roots["source"]),
        ("source", "--cold-dir", roots["source_cold"]),
        ("source", "--final-source-authorization", expected_source_marker),
        ("compile", "--staging-root", roots["compile_staging"]),
        ("compile", "--publish-root", roots["compiled"]),
        ("compile", "--log-root", roots["compile_logs"]),
        ("compiled-static", "--compiled-dir", roots["compiled"]),
        ("compiled-cm-preflight", "--compiled-dir", roots["compiled"]),
        ("materialization", "--stage-dir", roots["materialize_staging"]),
        ("materialization", "--materialized-dir", roots["materialized"]),
        ("materialization", "--log-dir", roots["materialize_logs"]),
        ("claims-prepare", "--claims-dir", roots["claims"]),
        ("atlas-build", "--analysis-dir", roots["analysis"]),
        ("atlas-build", "--diagnostics-dir", roots["atlas_diagnostics"]),
        ("generated-promotion", "--claims-dir", roots["claims"]),
        ("generated-promotion", "--analysis-dir", roots["analysis"]),
    ):
        record_flag_binding(
            f"stage-output-binding:{stage}-{flag.removeprefix('--')}",
            commands_by_stage[stage],
            flag,
            expected,
            f"{stage} {flag} differs from the preauthorized stage root",
        )
    for stage, flag in (
        ("dyn-shape-preflight", "--report"),
        ("source", "--report"),
        ("compile", "--report"),
        ("compiled-membership", "--output"),
        ("compiled-static", "--output"),
        ("compiled-cm-preflight", "--output"),
        ("materialization", "--report"),
        ("materialized-membership", "--output"),
        ("claims-prepare", "--output"),
        ("atlas-build", "--output"),
        ("generated-promotion", "--output"),
        ("stock-campaign", "--report"),
        ("dyn-origin-binding", "--report"),
        ("assembly", "--output"),
    ):
        record_flag_binding(
            f"stage-report-binding:{stage}",
            commands_by_stage[stage],
            flag,
            reports_by_stage[stage],
            f"{stage} report differs from its preauthorized report path",
        )
    record(
        "stage-report-binding:dyn-execute",
        reports_by_stage["dyn-execute"]
        == str(Path(roots["dyn_evidence"]) / "b2-dyn-evidence.json"),
        "Dyn evidence report differs from the Dyn output contract",
    )
    record(
        "stage-report-binding:test-suite",
        reports_by_stage["test-suite"]
        == str(Path(roots["test_evidence"]) / "b2-test-report.json"),
        "test report differs from the atomic test publisher contract",
    )

    record_flag_binding(
        "stage-input-binding:compile-q2tool",
        compile_cmd,
        "--q2tool",
        pinned_paths["q2tool"],
        "compile q2tool does not match the pinned compiler",
    )
    record_flag_binding(
        "stage-input-binding:compile-basedir",
        compile_cmd,
        "--basedir",
        str(Path(pinned_paths["base_pak"]).parent),
        "compile basedir does not contain the pinned base PAK",
    )
    record_flag_binding(
        "stage-input-binding:cm-oracle",
        cm_cmd,
        "--cm-oracle",
        pinned_paths["cm_oracle"],
        "compiled-CM command does not match the pinned CM oracle",
    )
    for flag, pinned_name in (
        ("--cm-oracle", "cm_oracle"),
        ("--pmove-oracle", "pmove_oracle"),
        ("--hook-oracle", "hook_oracle"),
        ("--fall-oracle", "fall_oracle"),
        ("--hook-parity-attestation", "hook_attestation"),
    ):
        record_flag_binding(
            f"stage-input-binding:materialize-{pinned_name}",
            mat_cmd,
            flag,
            pinned_paths[pinned_name],
            f"materialization {flag} does not match pinned {pinned_name}",
        )

    atlas_client_root = str(Path(pinned_paths["atlas_cm_oracle"]).parent.parent)
    atlas_lithium_root = str(Path(pinned_paths["atlas_hook_oracle"]).parent.parent)
    for flag, expected, label in (
        ("--client-root", atlas_client_root, "client release root"),
        ("--lithium-root", atlas_lithium_root, "Lithium root"),
        ("--hook-attestation", pinned_paths["hook_attestation"], "hook attestation"),
        ("--fall-oracle", pinned_paths["fall_oracle"], "fall oracle"),
        ("--packer", pinned_paths["packer"], "packer"),
        ("--verifier", pinned_paths["verifier"], "verifier"),
    ):
        record_flag_binding(
            f"stage-input-binding:atlas-{flag.removeprefix('--')}",
            atlas_cmd,
            flag,
            expected,
            f"Atlas {label} does not match the pinned release closure",
        )
    record(
        "stage-input-binding:atlas-client-release-pair",
        Path(pinned_paths["atlas_pmove_oracle"]).parent.parent
        == Path(atlas_client_root),
        "pinned Atlas CM and Pmove oracles do not share one client release root",
    )
    record_flag_binding(
        "stage-input-binding:promotion-b1-gate",
        promo_cmd,
        "--b1-gate",
        pinned_paths["b1_gate"],
        "generated promotion does not match the pinned B1 gate",
    )

    stock_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "stock-campaign"
    )
    for flag, pinned_name in (
        ("--stock-pak", "stock_pak"),
        ("--provenance", "stock_provenance"),
        ("--stock-inventory", "stock_inventory"),
        ("--b1-gate", "b1_gate"),
        ("--hook-attestation", "hook_attestation"),
        ("--fall-oracle", "fall_oracle"),
        ("--packer", "packer"),
        ("--verifier", "verifier"),
    ):
        record_flag_binding(
            f"stage-input-binding:stock-{pinned_name}",
            stock_cmd,
            flag,
            pinned_paths[pinned_name],
            f"stock {flag} does not match pinned {pinned_name}",
        )
    for flag, expected in (
        ("--repo-root", str(_absolute(Path(str(plan["repo_root"]))))),
        ("--python", pinned_paths["python"]),
        ("--client-root", atlas_client_root),
        ("--lithium-root", atlas_lithium_root),
    ):
        record_flag_binding(
            f"stage-input-binding:stock-{flag.removeprefix('--')}",
            stock_cmd,
            flag,
            expected,
            f"stock {flag} differs from the preauthorized release closure",
        )
    record_flag_binding(
        "stage-input-binding:stock-output",
        stock_cmd,
        "--output-root",
        roots["stock"],
        "stock campaign publication root differs",
    )

    dyn_shape_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "dyn-shape-preflight"
    )
    record_flag_binding(
        "stage-input-binding:dyn-executable",
        dyn_shape_cmd,
        "--executable",
        pinned_paths["dyn_evidence_executable"],
        "Dyn Phase A executable differs from pinned bytes",
    )
    record_flag_binding(
        "stage-input-binding:dyn-repository",
        dyn_shape_cmd,
        "--expected-crate-commit",
        str(plan["repository"]["repository_commit"]),
        "Dyn Phase A commit differs from repository binding",
    )
    for flag, expected in (
        ("--repo-root", str(_absolute(Path(str(plan["repo_root"]))))),
        (
            "--expected-analyzer-authority",
            str(plan["repository"]["atlas_analyzer_authority_sha256"]),
        ),
        ("--output", roots["dyn_evidence"]),
    ):
        record_flag_binding(
            f"stage-input-binding:dyn-shape-{flag.removeprefix('--')}",
            dyn_shape_cmd,
            flag,
            expected,
            f"Dyn Phase A {flag} differs from the preauthorized binding",
        )
    dyn_binding_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "dyn-origin-binding"
    )
    dyn_execute_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "dyn-execute"
    )
    record(
        "stage-input-binding:dyn-authorities",
        dyn_binding_cmd[
            dyn_binding_cmd.index("--shape-preflight-report") + 1
        ]
        == dyn_execute_cmd[
            dyn_execute_cmd.index("--shape-preflight-report") + 1
        ]
        and dyn_binding_cmd[dyn_binding_cmd.index("--report") + 1]
        == dyn_execute_cmd[
            dyn_execute_cmd.index("--origin-binding-report") + 1
        ],
        "Dyn Phase A/B/execution report chain differs",
    )
    record(
        "stage-input-binding:dyn-promotion",
        dyn_binding_cmd[
            dyn_binding_cmd.index("--generated-promotion-report") + 1
        ]
        == reports_by_stage["generated-promotion"],
        "Dyn Phase B does not consume the preauthorized promotion report",
    )

    test_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "test-suite"
    )
    record_flag_binding(
        "stage-input-binding:test-output",
        test_cmd,
        "--output",
        roots["test_evidence"],
        "test suite output root differs",
    )

    assembly_cmd = next(
        step["command"]
        for step in plan["commands"]
        if step["stage"] == "assembly"
    )
    try:
        bound_assembly_invocation = _assembly_command_binding(
            assembly_cmd, repo=direct_repo
        )
        record(
            "assembly-command-binding",
            isinstance(plan.get("assembly_invocation"), Mapping)
            and dict(plan["assembly_invocation"]) == bound_assembly_invocation,
            "planned assembly command hash differs from the assembler argv hash",
        )
    except FinalCohortPlanError as error:
        record(error.check_id, False, str(error))
    for flag, expected in (
        ("--design", pinned_paths["design"]),
        ("--plan", pinned_paths["execution_plan"]),
        ("--repo-root", str(_absolute(Path(str(plan["repo_root"]))))),
        ("--b1-gate", pinned_paths["b1_gate"]),
        ("--cm-oracle", pinned_paths["cm_oracle"]),
        ("--pmove-oracle", pinned_paths["pmove_oracle"]),
        ("--hook-oracle", pinned_paths["hook_oracle"]),
        ("--fall-oracle", pinned_paths["fall_oracle"]),
        ("--hook-attestation", pinned_paths["hook_attestation"]),
        ("--atlas-verifier", pinned_paths["verifier"]),
        ("--qualification-report", pinned_paths["qualification_report"]),
        (
            "--preactivation-test-report",
            pinned_paths["preactivation_test_report"],
        ),
        ("--stock-provenance", pinned_paths["stock_provenance"]),
        ("--stock-inventory", pinned_paths["stock_inventory"]),
        ("--dyn-evidence-executable", pinned_paths["dyn_evidence_executable"]),
        ("--source-dir", roots["source"]),
        ("--source-cold-dir", roots["source_cold"]),
        ("--source-freeze-report", reports_by_stage["source"]),
        ("--compiled-dir", roots["compiled"]),
        (
            "--compiled-membership-report",
            reports_by_stage["compiled-membership"],
        ),
        ("--compiled-static-report", reports_by_stage["compiled-static"]),
        (
            "--compiled-cm-preflight-report",
            reports_by_stage["compiled-cm-preflight"],
        ),
        ("--materialized-dir", roots["materialized"]),
        (
            "--materialized-membership-report",
            reports_by_stage["materialized-membership"],
        ),
        ("--claims-dir", roots["claims"]),
        ("--claims-prepare-report", reports_by_stage["claims-prepare"]),
        ("--analysis-dir", roots["analysis"]),
        ("--generated-build-report", reports_by_stage["atlas-build"]),
        (
            "--generated-validation-report",
            reports_by_stage["generated-promotion"],
        ),
        ("--stock-bsp-dir", roots["stock_bsp"]),
        ("--stock-analysis-dir", roots["stock_analysis"]),
        ("--stock-validation-dir", roots["stock_validation"]),
        (
            "--dyn-argv-preflight-report",
            reports_by_stage["dyn-shape-preflight"],
        ),
        (
            "--dyn-origin-binding-report",
            reports_by_stage["dyn-origin-binding"],
        ),
        ("--dyn-evidence-report", reports_by_stage["dyn-execute"]),
        ("--test-report", reports_by_stage["test-suite"]),
        ("--final-lifecycle-evidence", roots["lifecycle_evidence"]),
        ("--output", roots["gate_output"]),
    ):
        record_flag_binding(
            f"stage-input-binding:assembly-{flag.removeprefix('--')}",
            assembly_cmd,
            flag,
            expected,
            f"assembly {flag} differs from the preauthorized input",
        )

    for supplied, atlas in (
        ("cm_oracle", "atlas_cm_oracle"),
        ("pmove_oracle", "atlas_pmove_oracle"),
        ("hook_oracle", "atlas_hook_oracle"),
    ):
        record(
            f"atlas-release-closure:{supplied}",
            _same_file_bytes(pinned[supplied], pinned[atlas]),
            f"canonical Atlas {supplied} bytes differ from supplied authority",
        )

    return checks


def build_evidence(
    plan: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    mutating_stages_executed: Sequence[str],
    failure: Mapping[str, Any] | None = None,
    source_authorization_marker: Mapping[str, Any] | None = None,
    stage_executions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Emit canonical pre-authorization evidence."""

    passed = failure is None and all(check.get("passed") is True for check in checks)
    defect_class = None
    if failure is not None:
        defect_class = failure.get("defect_class", DEFECT_RUNNER_CONFIGURATION)
    elif not passed:
        defect_class = DEFECT_RUNNER_CONFIGURATION

    # Pre-authorization configuration defects never consume the immutable
    # source authorization. Once source ran, the authorization was already
    # consumed and every failed execution path is terminal, including a
    # configuration defect that an older or bypassed runner allowed through.
    authorization_consumed = bool(mutating_stages_executed) and "source" in set(
        mutating_stages_executed
    )
    if failure is not None and authorization_consumed:
        cohort_retirement_triggered = True
        retry_under_same_declaration_allowed = False
    elif defect_class == DEFECT_RUNNER_CONFIGURATION:
        cohort_retirement_triggered = False
        retry_under_same_declaration_allowed = True
    elif defect_class == DEFECT_COHORT_ARTIFACT:
        cohort_retirement_triggered = authorization_consumed
        retry_under_same_declaration_allowed = False
    else:
        cohort_retirement_triggered = False
        retry_under_same_declaration_allowed = not authorization_consumed

    plan_authorization = plan.get("authorization")
    execution_binding = (
        plan_authorization.get("execution_binding")
        if isinstance(plan_authorization, Mapping)
        and isinstance(plan_authorization.get("execution_binding"), Mapping)
        else None
    )
    authorization_evidence = _authorization_contract(
        consumed=authorization_consumed
    )
    if execution_binding is not None:
        authorization_evidence["execution_binding"] = dict(execution_binding)

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "passed": passed,
        "dry_run": dry_run,
        "cohort_id": plan.get("cohort_id"),
        "plan_sha256": _sha256(_canonical(plan)),
        "plan_schema": plan.get("schema"),
        "stage_order": [step["stage"] for step in plan.get("commands", [])],
        "argument_domains": plan.get("argument_domains"),
        "declaration": plan.get("declaration"),
        "checks": list(checks),
        "mutating_stages_executed": list(mutating_stages_executed),
        "stage_executions": [dict(row) for row in stage_executions],
        "source_authorization_marker": (
            dict(source_authorization_marker)
            if source_authorization_marker is not None else None
        ),
        "failure": failure,
        "defect_class": defect_class,
        "authorization": {
            **authorization_evidence,
            "cohort_retirement_triggered": cohort_retirement_triggered,
            "retry_under_same_declaration_allowed": retry_under_same_declaration_allowed,
        },
    }
    return evidence


def _acknowledged_mutating_execution(value: object) -> bool:
    return value is True or value == MUTATING_EXECUTION_ACK


def run_plan(
    plan: Mapping[str, Any],
    *,
    dry_run: bool = True,
    runner: Callable[..., Any] | None = None,
    acknowledge_mutating_execution: object = False,
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the full plan, then optionally execute stages in order.

    Mutating stages never start until full-plan validation succeeds.  Dry-run
    (default) emits evidence only and never invokes ``runner``.  Mutating
    execution additionally requires an unambiguous acknowledgement
    (``True`` or ``MUTATING_EXECUTION_ACK``) plus an explicit runner.
    """

    checks: list[dict[str, Any]] = []
    try:
        checks = validate_plan(plan)
    except FinalCohortPlanError as error:
        checks.append(
            {
                "check_id": error.check_id,
                "passed": False,
                "detail": str(error),
                "defect_class": error.defect_class,
            }
        )
        return build_evidence(
            plan,
            checks,
            dry_run=dry_run,
            mutating_stages_executed=[],
            failure={
                "defect_class": error.defect_class,
                "check_id": error.check_id,
                "message": str(error),
            },
        )

    if dry_run:
        return build_evidence(
            plan,
            checks,
            dry_run=True,
            mutating_stages_executed=[],
        )

    plan_sha256 = _sha256(_canonical(plan))
    if (
        not isinstance(expected_plan_sha256, str)
        or HEX64.fullmatch(expected_plan_sha256) is None
        or expected_plan_sha256 != plan_sha256
    ):
        failure = {
            "defect_class": DEFECT_RUNNER_CONFIGURATION,
            "check_id": "expected-plan-sha256",
            "message": (
                "execute mode requires the exact reviewed dry-run plan SHA-256; "
                f"actual={plan_sha256}"
            ),
        }
        return build_evidence(
            plan,
            checks,
            dry_run=False,
            mutating_stages_executed=[],
            failure=failure,
        )

    if not _acknowledged_mutating_execution(acknowledge_mutating_execution):
        failure = {
            "defect_class": DEFECT_RUNNER_CONFIGURATION,
            "check_id": "execute-acknowledgement",
            "message": (
                "mutating execution requires unambiguous acknowledgement "
                f"{MUTATING_EXECUTION_ACK!r} (or True via library API)"
            ),
        }
        return build_evidence(
            plan,
            checks,
            dry_run=False,
            mutating_stages_executed=[],
            failure=failure,
        )

    if runner is None:
        failure = {
            "defect_class": DEFECT_RUNNER_CONFIGURATION,
            "check_id": "execute-runner",
            "message": "execute mode requires an explicit runner; refusing default subprocess",
        }
        return build_evidence(
            plan,
            checks,
            dry_run=False,
            mutating_stages_executed=[],
            failure=failure,
        )

    try:
        prior_authorization_marker = _existing_source_authorization_marker(plan)
    except FinalCohortPlanError as error:
        consumed = error.consumed
        return build_evidence(
            plan,
            checks,
            dry_run=False,
            mutating_stages_executed=["source"] if consumed else [],
            failure={
                "defect_class": (
                    DEFECT_COHORT_ARTIFACT if consumed else error.defect_class
                ),
                "check_id": error.check_id,
                "message": str(error),
                "stage": "source",
            },
        )
    if prior_authorization_marker is not None:
        return build_evidence(
            plan,
            checks,
            dry_run=False,
            mutating_stages_executed=["source"],
            failure={
                "defect_class": DEFECT_COHORT_ARTIFACT,
                "check_id": "source-authorization-already-consumed",
                "message": "source authorization marker already exists",
                "stage": "source",
            },
            source_authorization_marker=prior_authorization_marker,
        )
    try:
        _prepare_execution_workspace(plan)
    except (FinalCohortPlanError, OSError) as error:
        check_id = (
            error.check_id
            if isinstance(error, FinalCohortPlanError)
            else "execution-workspace"
        )
        return build_evidence(
            plan,
            checks,
            dry_run=False,
            mutating_stages_executed=[],
            failure={
                "defect_class": DEFECT_RUNNER_CONFIGURATION,
                "check_id": check_id,
                "message": str(error),
            },
        )

    # This list is deliberately write-ahead: entering a mutating stage consumes
    # its one-shot authorization before the runner can create its first byte.
    # Recording only after return would make a crash during source generation
    # appear pre-source and incorrectly reopen the immutable declaration.
    executed: list[str] = []
    stage_executions: list[dict[str, Any]] = []
    authorization_marker: Mapping[str, Any] | None = None
    for step in plan["commands"]:
        stage = str(step["stage"])
        try:
            validate_plan(plan)
        except FinalCohortPlanError as error:
            consumed = "source" in executed
            return build_evidence(
                plan,
                checks,
                dry_run=False,
                mutating_stages_executed=executed,
                failure={
                    "defect_class": (
                        DEFECT_COHORT_ARTIFACT
                        if consumed
                        else DEFECT_RUNNER_CONFIGURATION
                    ),
                    "check_id": f"stage-revalidation:{stage}:{error.check_id}",
                    "message": str(error),
                    "stage": stage,
                },
                source_authorization_marker=authorization_marker,
                stage_executions=stage_executions,
            )
        if stage == "source":
            try:
                authorization_marker, created = _source_authorization_marker(plan)
            except FinalCohortPlanError as error:
                current_declaration_consumed = error.consumed
                return build_evidence(
                    plan,
                    checks,
                    dry_run=False,
                    # Source has not been invoked yet. Claim consumption only
                    # when O_EXCL left a durable marker/tombstone on disk.
                    mutating_stages_executed=[
                        *executed,
                        *(["source"] if current_declaration_consumed else []),
                    ],
                    failure={
                        "defect_class": error.defect_class,
                        "check_id": error.check_id,
                        "message": str(error),
                        "stage": "source",
                    },
                    stage_executions=stage_executions,
                )
            if not created:
                return build_evidence(
                    plan,
                    checks,
                    dry_run=False,
                    mutating_stages_executed=[*executed, "source"],
                    failure={
                        "defect_class": DEFECT_COHORT_ARTIFACT,
                        "check_id": "source-authorization-already-consumed",
                        "message": "source authorization marker already exists",
                        "stage": "source",
                    },
                    source_authorization_marker=authorization_marker,
                    stage_executions=stage_executions,
                )
            try:
                validate_plan(plan)
            except FinalCohortPlanError as error:
                return build_evidence(
                    plan,
                    checks,
                    dry_run=False,
                    mutating_stages_executed=[*executed, "source"],
                    failure={
                        "defect_class": DEFECT_COHORT_ARTIFACT,
                        "check_id": f"stage-revalidation:source:{error.check_id}",
                        "message": str(error),
                        "stage": "source",
                    },
                    source_authorization_marker=authorization_marker,
                    stage_executions=stage_executions,
                )
        if stage == "assembly":
            try:
                _publish_preassembly_lifecycle_evidence(
                    plan,
                    source_authorization_marker=authorization_marker,
                    executed=executed,
                    stage_executions=stage_executions,
                    assembly_command=step["command"],
                )
            except FinalCohortPlanError as error:
                return build_evidence(
                    plan,
                    checks,
                    dry_run=False,
                    mutating_stages_executed=executed,
                    failure={
                        "defect_class": DEFECT_COHORT_ARTIFACT,
                        "check_id": error.check_id,
                        "message": str(error),
                        "stage": stage,
                    },
                    source_authorization_marker=authorization_marker,
                    stage_executions=stage_executions,
                )
        executed.append(stage)
        try:
            completed = runner(list(step["command"]), stage=stage, plan=plan)
        except BaseException as error:
            defect = (
                DEFECT_COHORT_ARTIFACT
                if "source" in executed
                else DEFECT_RUNNER_CONFIGURATION
            )
            return build_evidence(
                plan,
                checks,
                dry_run=False,
                mutating_stages_executed=executed,
                failure={
                    "defect_class": defect,
                    "check_id": f"stage-exception:{stage}",
                    "message": f"stage {stage} raised {type(error).__name__}",
                    "stage": stage,
                    "exception_type": type(error).__name__,
                },
                source_authorization_marker=authorization_marker,
                stage_executions=stage_executions,
            )
        returncode = getattr(completed, "returncode", completed)
        stdout = getattr(completed, "stdout", b"") or b""
        stderr = getattr(completed, "stderr", b"") or b""
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8")
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8")
        stage_executions.append({
            "stage": stage,
            "command_sha256": _sha256(_canonical(list(step["command"]))),
            "returncode": int(returncode),
            "stdout": {"bytes": len(stdout), "sha256": _sha256(stdout)},
            "stderr": {"bytes": len(stderr), "sha256": _sha256(stderr)},
        })
        if int(returncode) != 0:
            # After source has run, a non-zero exit is a consumed no-retry failure.
            defect = (
                DEFECT_COHORT_ARTIFACT
                if "source" in executed or stage == "source"
                else DEFECT_RUNNER_CONFIGURATION
            )
            stderr_excerpt = stderr[:4096].decode("utf-8", errors="replace").strip()
            return build_evidence(
                plan,
                checks,
                dry_run=False,
                mutating_stages_executed=executed,
                failure={
                    "defect_class": defect,
                    "check_id": f"stage-exit:{stage}",
                    "message": f"stage {stage} exited {returncode}",
                    "stage": stage,
                    "returncode": int(returncode),
                    "stderr_excerpt": stderr_excerpt,
                    "stderr_truncated": len(stderr) > 4096,
                },
                source_authorization_marker=authorization_marker,
                stage_executions=stage_executions,
            )

    return build_evidence(
        plan,
        checks,
        dry_run=False,
        mutating_stages_executed=executed,
        source_authorization_marker=authorization_marker,
        stage_executions=stage_executions,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--preactivation-test-report", type=Path, required=True)
    parser.add_argument(
        "--authorization-state-root",
        type=Path,
        required=True,
        help=(
            "pre-provisioned 0700 final-cohort source-authorization journal "
            "on DESKTOP-RTX2080 WSL"
        ),
    )
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--stock-pak", type=Path, required=True)
    parser.add_argument("--stock-provenance", type=Path, required=True)
    parser.add_argument("--stock-inventory", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--lithium-root", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--dyn-evidence-executable", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-cold", type=Path, default=None)
    parser.add_argument("--compile-staging", type=Path, default=None)
    parser.add_argument("--compiled-root", type=Path, default=None)
    parser.add_argument("--compile-logs", type=Path, default=None)
    parser.add_argument("--materialize-staging", type=Path, default=None)
    parser.add_argument("--materialized-root", type=Path, default=None)
    parser.add_argument("--materialize-logs", type=Path, default=None)
    parser.add_argument("--claims-root", type=Path, default=None)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--atlas-diagnostics", type=Path, default=None)
    parser.add_argument("--stock-root", type=Path, default=None)
    parser.add_argument("--dyn-evidence-root", type=Path, default=None)
    parser.add_argument("--test-evidence-root", type=Path, default=None)
    parser.add_argument("--gate-output", type=Path, default=None)
    parser.add_argument("--dyn-map-epoch", type=int, default=1)
    parser.add_argument("--dyn-environment-steps", type=int, default=4000)
    parser.add_argument("--dyn-samples", type=int, default=4000)
    parser.add_argument(
        "--compile-timeout-seconds",
        type=float,
        default=3600.0,
        help="per-map q2tool timeout domain (0, 86400]",
    )
    parser.add_argument(
        "--oracle-batch-timeout-seconds",
        type=float,
        default=10.0,
        help="compiled-CM oracle batch timeout domain (0, 60]",
    )
    parser.add_argument(
        "--materialize-timeout-seconds",
        type=int,
        default=900,
        help="materializer timeout domain [1, 3600]",
    )
    parser.add_argument("--cm-jobs", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="validate full plan only (default); never consumes source authorization",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "after full-plan validation, request mutating stage execution; "
            "requires --expected-plan-sha256 and "
            "--acknowledge-mutating-execution"
        ),
    )
    parser.add_argument(
        "--acknowledge-mutating-execution",
        default=None,
        metavar="TOKEN",
        help=(
            "unambiguous mutating-execution acknowledgement; must equal "
            f"{MUTATING_EXECUTION_ACK} when --execute is set"
        ),
    )
    parser.add_argument(
        "--expected-plan-sha256",
        default=None,
        metavar="HEX64",
        help="required in execute mode; must equal the reviewed dry-run plan hash",
    )
    return parser


def _subprocess_stage_runner(
    command: Sequence[str],
    *,
    stage: str,
    plan: Mapping[str, Any],
) -> subprocess.CompletedProcess[bytes]:
    """Execute one already-validated argv without a shell or implicit input."""

    del stage
    return subprocess.run(
        list(command),
        cwd=Path(str(plan["repo_root"])),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dry_run = not bool(args.execute)
    try:
        plan = build_plan(args)
        evidence = run_plan(
            plan,
            dry_run=dry_run,
            runner=_subprocess_stage_runner if args.execute else None,
            acknowledge_mutating_execution=args.acknowledge_mutating_execution,
            expected_plan_sha256=args.expected_plan_sha256,
        )
    except FinalCohortPlanError as error:
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "passed": False,
            "dry_run": dry_run,
            "cohort_id": None,
            "plan_sha256": None,
            "plan_schema": PLAN_SCHEMA,
            "stage_order": list(DRIVER_STAGES),
            "argument_domains": None,
            "declaration": None,
            "checks": [
                {
                    "check_id": error.check_id,
                    "passed": False,
                    "detail": str(error),
                    "defect_class": error.defect_class,
                }
            ],
            "mutating_stages_executed": [],
            "stage_executions": [],
            "source_authorization_marker": None,
            "failure": {
                "defect_class": error.defect_class,
                "check_id": error.check_id,
                "message": str(error),
            },
            "defect_class": error.defect_class,
            "authorization": {
                **_authorization_contract(consumed=False),
                "cohort_retirement_triggered": False,
                "retry_under_same_declaration_allowed": True,
            },
        }
    except OSError as error:
        print(f"final cohort plan refused: {error}", file=sys.stderr)
        return 1

    sys.stdout.buffer.write(_canonical(evidence))
    return 0 if evidence.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
