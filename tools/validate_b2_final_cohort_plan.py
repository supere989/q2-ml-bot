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
)


PLAN_SCHEMA = "q2-b2-final-cohort-preauthorization-plan-v1"
EVIDENCE_SCHEMA = "q2-b2-final-cohort-preauthorization-evidence-v1"
SOURCE_AUTHORIZATION_SCHEMA = "q2-b2-source-authorization-consumed-v1"
SOURCE_AUTHORIZATION_STATE_ROOT = (
    Path.home() / ".local/state/q2-ml-bot/final-cohort-authorizations"
)
HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")

# Normative final-lane order through compiled promotion (design/plan lifecycle).
DRIVER_STAGES = (
    "source",
    "compile",
    "compiled-static",
    "compiled-cm-preflight",
    "materialization",
    "claims-prepare",
    "atlas-build",
    "generated-promotion",
)

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
    "source": "run_generator_cohort.py",
    "compile": "compile_generated_cohort.py",
    "compiled-static": "run_compiled_static_campaign.py",
    "compiled-cm-preflight": "run_compiled_cm_preflight.py",
    "materialization": "materialize_generated_cohort.py",
    "claims-prepare": "run_generator_claim_campaign.py",
    "atlas-build": "run_generated_atlas_campaign.py",
    "generated-promotion": "run_generator_claim_campaign.py",
}

# Option names constructed for each stage; tested against live parsers.
STAGE_REQUIRED_OPTIONS: dict[str, tuple[str, ...]] = {
    "source": ("--declaration", "--output-dir", "--cold-dir", "--report"),
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
        "--hook-attestation",
        "--fall-oracle",
    ),
    "generated-promotion": (
        "--declaration",
        "--claims-dir",
        "--analysis-dir",
        "--b1-gate",
        "--output",
    ),
}

STAGE_SUBCOMMANDS: dict[str, str | None] = {
    "source": "generate",
    "compile": None,
    "compiled-static": None,
    "compiled-cm-preflight": None,
    "materialization": None,
    "claims-prepare": "prepare",
    "atlas-build": None,
    "generated-promotion": "validate",
}


class FinalCohortPlanError(RuntimeError):
    """Raised when the pre-authorization plan is incomplete or invalid."""

    def __init__(
        self,
        message: str,
        *,
        defect_class: str = DEFECT_RUNNER_CONFIGURATION,
        check_id: str = "plan-validation",
    ) -> None:
        super().__init__(message)
        self.defect_class = defect_class
        self.check_id = check_id


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


def _source_authorization_marker(plan: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Persist the one-shot source boundary before the source runner starts."""

    workspace = _absolute(Path(str(plan.get("workspace", ""))))
    if not workspace.parent.is_dir() or workspace.parent.is_symlink():
        raise FinalCohortPlanError(
            "source authorization workspace parent is absent or a symlink",
            check_id="source-authorization-journal",
        )
    if workspace.exists():
        if workspace.is_symlink() or not workspace.is_dir():
            raise FinalCohortPlanError(
                "source authorization workspace is invalid",
                check_id="source-authorization-journal",
            )
    else:
        try:
            workspace.mkdir(mode=0o700)
        except OSError as error:
            raise FinalCohortPlanError(
                "source authorization workspace could not be created",
                check_id="source-authorization-journal",
            ) from error
    declaration = plan.get("declaration")
    declaration_sha256 = (
        declaration.get("sha256") if isinstance(declaration, Mapping) else None
    )
    if not isinstance(declaration_sha256, str) or HEX64.fullmatch(
        declaration_sha256
    ) is None:
        raise FinalCohortPlanError(
            "source authorization declaration identity is malformed",
            check_id="source-authorization-journal",
        )
    state_root = _absolute(SOURCE_AUTHORIZATION_STATE_ROOT)
    try:
        state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        raise FinalCohortPlanError(
            "source authorization state root could not be created",
            check_id="source-authorization-journal",
        ) from error
    if state_root.is_symlink() or not state_root.is_dir():
        raise FinalCohortPlanError(
            "source authorization state root is invalid",
            check_id="source-authorization-journal",
        )
    marker = state_root / f"{declaration_sha256}.json"
    body = {
        "schema": SOURCE_AUTHORIZATION_SCHEMA,
        "status": "source-authorization-consumed",
        "cohort_id": plan.get("cohort_id"),
        "declaration_sha256": declaration_sha256,
        "plan_sha256": _sha256(_canonical(plan)),
        "workspace": str(workspace),
        "stage_started": "source",
        "immutable_no_retry": True,
    }
    payload = _canonical(body)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError:
        if marker.is_symlink() or not marker.is_file():
            raise FinalCohortPlanError(
                "declaration-scoped source authorization tombstone is invalid",
                defect_class=DEFECT_COHORT_ARTIFACT,
                check_id="source-authorization-already-consumed",
            )
        existing = marker.read_bytes()
        try:
            loaded = json.loads(existing)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise FinalCohortPlanError(
                "declaration-scoped source authorization tombstone is unreadable",
                defect_class=DEFECT_COHORT_ARTIFACT,
                check_id="source-authorization-already-consumed",
            ) from error
        if (
            existing != _canonical(loaded)
            or not isinstance(loaded, Mapping)
            or loaded.get("schema") != SOURCE_AUTHORIZATION_SCHEMA
            or loaded.get("status") != "source-authorization-consumed"
            or loaded.get("cohort_id") != plan.get("cohort_id")
            or loaded.get("declaration_sha256") != declaration_sha256
            or loaded.get("immutable_no_retry") is not True
        ):
            raise FinalCohortPlanError(
                "declaration-scoped source authorization tombstone differs",
                defect_class=DEFECT_COHORT_ARTIFACT,
                check_id="source-authorization-already-consumed",
            )
        return {
            "path": str(marker),
            "sha256": _sha256(existing),
            "bytes": len(existing),
        }, False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(marker, 0o600)
        directory_fd = os.open(state_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException as error:
        # O_EXCL has already consumed this declaration.  Never unlink a short,
        # partially written, or not-yet-directory-synced journal: its continued
        # presence is the fail-closed tombstone that prevents a second source
        # invocation after an I/O or process failure at this boundary.
        raise FinalCohortPlanError(
            "source authorization journal creation was incomplete; declaration remains consumed",
            defect_class=DEFECT_COHORT_ARTIFACT,
            check_id="source-authorization-journal-incomplete",
        ) from error
    return {
        "path": str(marker),
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }, True


def _arg(command: list[str], name: str, value: Path | str | int | float) -> None:
    command.extend((name, str(value)))


def _tool(repo: Path, name: str) -> Path:
    return repo / "tools" / name


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


def _existing_dir(path: Path, label: str) -> Path:
    absolute = _absolute(path)
    if absolute.is_symlink() or not absolute.is_dir():
        raise FinalCohortPlanError(
            f"{label} is absent or a symlink: {absolute}",
            check_id="path-identity",
        )
    return absolute


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
    if stage == "source":
        sub = parser.add_subparsers(dest="command", required=True)
        generate = sub.add_parser("generate", exit_on_error=False, add_help=False)
        generate.add_argument("--declaration", type=Path, required=True)
        generate.add_argument("--output-dir", type=Path, required=True)
        generate.add_argument("--cold-dir", type=Path, required=True)
        generate.add_argument("--report", type=Path, required=True)
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
        parser.add_argument("--client-root", type=Path, default=None)
        parser.add_argument("--lithium-root", type=Path, default=None)
        parser.add_argument("--hook-attestation", type=Path, default=None)
        parser.add_argument("--fall-oracle", type=Path, default=None)
        parser.add_argument("--packer", type=Path, default=None)
        parser.add_argument("--verifier", type=Path, default=None)
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
        if option not in argv:
            raise FinalCohortPlanError(
                f"{stage} command missing required option {option}",
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

    repo = _absolute(args.repo_root)
    workspace = _absolute(args.workspace)
    python = _absolute(Path(args.python) if args.python else Path(sys.executable))
    declaration = _absolute(args.declaration)
    reports = workspace / "reports"

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

    try:
        workspace.relative_to(repo)
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
    # Admission against the shared retired cohort/map/seed registry.  Validation
    # never invents a successor ID; it only refuses identities already retired.
    _require(
        declaration_sha256 == declaration_record["sha256"],
        "declaration digest disagrees with load_declaration",
        check_id="declaration-binding",
    )
    _require_unretired(declaration, declaration_obj, declaration_sha256)
    _require_active_declaration(declaration, declaration_obj, declaration_sha256)

    pinned_inputs = {
        "declaration": declaration_record,
        "python": _regular_file(python, "python runtime", executable=True),
        "q2tool": _regular_file(args.q2tool, "q2tool", executable=True),
        "base_pak": _regular_file(
            _existing_dir(args.basedir, "basedir") / "pak0.pak",
            "basedir/pak0.pak",
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
    }
    basedir = _existing_dir(args.basedir, "basedir")

    tool_records: dict[str, dict[str, Any]] = {}
    for stage, tool_name in TOOL_FILES.items():
        tool_records[stage] = _regular_file(
            _tool(repo, tool_name), f"{stage} tool {tool_name}"
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

    report_paths = {
        "source": reports / "source-freeze.json",
        "compile": reports / "compile.json",
        "compiled-static": reports / "compiled-static.json",
        "compiled-cm-preflight": reports / "compiled-cm-preflight.json",
        "materialization": reports / "materialize.json",
        "claims-prepare": reports / "claims-prepare.json",
        "atlas-build": reports / "atlas-build.json",
        "generated-promotion": reports / "generated-promotion.json",
    }

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
        *report_paths.values(),
    ]
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
        *[(path, f"report:{stage}") for stage, path in report_paths.items()],
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
        str(_tool(repo, TOOL_FILES["source"])),
        "generate",
    ]
    for name, value in (
        ("--declaration", declaration),
        ("--output-dir", source_root),
        ("--cold-dir", source_cold),
        ("--report", report_paths["source"]),
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
    if args.client_root is not None:
        _arg(command, "--client-root", _absolute(args.client_root))
    if args.lithium_root is not None:
        _arg(command, "--lithium-root", _absolute(args.lithium_root))
    if args.packer is not None:
        _arg(command, "--packer", _absolute(args.packer))
    if args.verifier is not None:
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

    plan = {
        "schema": PLAN_SCHEMA,
        "cohort_id": cohort_id,
        "repo_root": str(repo),
        "workspace": str(workspace),
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
        },
        "pinned_inputs": pinned_inputs,
        "tools": tool_records,
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
        },
        "commands": commands,
        "authorization": _authorization_contract(consumed=False),
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

    authorization = plan.get("authorization")
    expected_auth = _authorization_contract(consumed=False)
    record(
        "authorization-contract",
        isinstance(authorization, Mapping)
        and all(authorization.get(key) == value for key, value in expected_auth.items()),
        "authorization contract weakened or incomplete",
    )

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
        # Every mutating stage must bind the same declaration path.
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
            "compiled",
            "materialized",
            "claims",
            "analysis",
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
            "cm_oracle",
            "pmove_oracle",
            "hook_oracle",
            "fall_oracle",
            "hook_attestation",
            "b1_gate",
        }.issubset(pinned),
        "pinned inputs incomplete",
    )
    for name, record_value in pinned.items():
        path = Path(str(record_value["path"]))
        live_record = _regular_file(
            path,
            name,
            executable=name
            in {"python", "q2tool", "cm_oracle", "pmove_oracle", "hook_oracle", "fall_oracle"},
        )
        record(
            f"pinned-digest:{name}",
            live_record["sha256"] == record_value["sha256"],
            f"pinned input drifted: {name}",
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
        "source_authorization_marker": (
            dict(source_authorization_marker)
            if source_authorization_marker is not None else None
        ),
        "failure": failure,
        "defect_class": defect_class,
        "authorization": {
            **_authorization_contract(consumed=authorization_consumed),
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

    # This list is deliberately write-ahead: entering a mutating stage consumes
    # its one-shot authorization before the runner can create its first byte.
    # Recording only after return would make a crash during source generation
    # appear pre-source and incorrectly reopen the immutable declaration.
    executed: list[str] = []
    authorization_marker: Mapping[str, Any] | None = None
    for step in plan["commands"]:
        stage = str(step["stage"])
        if stage == "source":
            try:
                authorization_marker, created = _source_authorization_marker(plan)
            except FinalCohortPlanError as error:
                current_declaration_consumed = (
                    error.check_id == "source-authorization-journal-incomplete"
                )
                return build_evidence(
                    plan,
                    checks,
                    dry_run=False,
                    # Source has not been invoked yet. Claim consumption only
                    # when O_EXCL left a durable marker/tombstone on disk.
                    mutating_stages_executed=(
                        ["source"] if current_declaration_consumed else []
                    ),
                    failure={
                        "defect_class": error.defect_class,
                        "check_id": error.check_id,
                        "message": str(error),
                        "stage": "source",
                    },
                )
            if not created:
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
                    source_authorization_marker=authorization_marker,
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
            )
        returncode = getattr(completed, "returncode", completed)
        if int(returncode) != 0:
            # After source has run, a non-zero exit is a consumed no-retry failure.
            defect = (
                DEFECT_COHORT_ARTIFACT
                if "source" in executed or stage == "source"
                else DEFECT_RUNNER_CONFIGURATION
            )
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
                },
                source_authorization_marker=authorization_marker,
            )

    return build_evidence(
        plan,
        checks,
        dry_run=False,
        mutating_stages_executed=executed,
        source_authorization_marker=authorization_marker,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, default=None)
    parser.add_argument("--lithium-root", type=Path, default=None)
    parser.add_argument("--packer", type=Path, default=None)
    parser.add_argument("--verifier", type=Path, default=None)
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
            "requires --acknowledge-mutating-execution and an injected runner"
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dry_run = not bool(args.execute)
    try:
        plan = build_plan(args)
        evidence = run_plan(
            plan,
            dry_run=dry_run,
            runner=None,
            acknowledge_mutating_execution=args.acknowledge_mutating_execution,
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
