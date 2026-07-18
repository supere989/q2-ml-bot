#!/usr/bin/env python3
"""Produce a fresh B1 gate from exact historical evidence and live oracles.

This is a one-time, fail-closed requalification path for a normative-document
amendment.  It never edits the repository gate in place and never rebuilds or
discovers an oracle.  The caller supplies the exact historical gate, parity
attestation, four oracle binaries, and a real BSP used to challenge live CM and
Pmove identities.  Output is created only after every byte and identity check
passes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import atlas_b1_authority as authority  # noqa: E402


_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class B1RequalificationError(ValueError):
    """Raised before output when fresh B1 authority cannot be proven."""


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise B1RequalificationError(f"git probe failed: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise B1RequalificationError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def repository_binding(repo_root: Path) -> dict[str, Any]:
    """Require and bind a clean repository containing the amended documents."""

    status = _run_git(repo_root, ("status", "--porcelain", "--untracked-files=all"))
    if status:
        raise B1RequalificationError(
            "repository is dirty; commit the requalification implementation "
            "and write the gate outside the worktree"
        )
    commit = _run_git(repo_root, ("rev-parse", "HEAD"))
    tree = _run_git(repo_root, ("rev-parse", "HEAD^{tree}"))
    if _HEX40.fullmatch(commit) is None or _HEX40.fullmatch(tree) is None:
        raise B1RequalificationError("repository commit/tree identity is invalid")
    return {"commit": commit, "tree": tree, "clean": True}


def _load_historical_document(path: Path) -> dict[str, Any]:
    data = authority._regular_file_bytes(
        path, "historical B1 gate", limit=1024 * 1024
    )
    if hashlib.sha256(data).hexdigest() != authority.HISTORICAL_GATE_SHA256:
        raise B1RequalificationError(
            "historical B1 gate bytes differ from the accepted evidence root"
        )
    return authority._decode_json_object(data, "historical B1 gate")


def _reverify_live_inputs(
    *,
    historical_gate: authority.B1AuthorityGate,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_parity_attestation: Path,
    probe_bsp: Path,
    timeout_seconds: float,
) -> tuple[authority.B1RuntimeAuthoritySeal, dict[str, dict[str, Any]]]:
    parity, attestation = authority._load_hook_parity_attestation(
        authority._resolve_supplied_path(
            hook_parity_attestation, "supplied hook parity attestation"
        ),
        historical_gate,
    )
    map_path = authority._resolve_supplied_path(probe_bsp, "supplied probe BSP")
    map_digest = authority._file_sha256(map_path, "supplied probe BSP")

    cm = authority._resolve_supplied_path(cm_oracle, "supplied CM oracle")
    pmove = authority._resolve_supplied_path(pmove_oracle, "supplied Pmove oracle")
    hook = authority._resolve_supplied_path(hook_oracle, "supplied hook oracle")
    fall = authority._resolve_supplied_path(fall_oracle, "supplied fall oracle")
    cm_digest = authority._admit_executable(
        cm, historical_gate.cm_executable_sha256, "supplied CM oracle"
    )
    pmove_digest = authority._admit_executable(
        pmove, historical_gate.pmove_executable_sha256, "supplied Pmove oracle"
    )
    hook_digest = authority._admit_executable(
        hook, parity.hook_executable_sha256, "supplied hook oracle"
    )
    fall_digest = authority._admit_executable(
        fall, historical_gate.fall_executable_sha256, "supplied fall oracle"
    )

    cm_identity = authority._run_identity(
        cm,
        {"id": "b1-authority", "op": "identity"},
        arguments=("--map", str(map_path)),
        timeout_seconds=timeout_seconds,
    )
    pmove_identity = authority._run_identity(
        pmove,
        {
            "id": "b1-authority", "op": "identity", "gravity": 800,
            "airaccelerate": 0,
        },
        arguments=("--map", str(map_path)),
        timeout_seconds=timeout_seconds,
    )
    hook_identity = authority._run_identity(
        hook,
        {
            "id": "b1-authority", "op": "identity",
            "hook_speed": parity.hook_speed,
            "hook_pullspeed": parity.hook_pullspeed,
            "hook_sky": parity.hook_sky,
            "hook_maxtime": parity.hook_maxtime,
        },
        timeout_seconds=timeout_seconds,
    )
    fall_identity = authority._run_identity(
        fall,
        {
            "id": "b1-authority", "op": "identity",
            "fall_damagemod": 1, "deathmatch": True, "dmflags": 0,
        },
        timeout_seconds=timeout_seconds,
    )

    authority._validate_cm_identity(cm_identity, historical_gate, map_digest)
    authority._validate_pmove_identity(pmove_identity, historical_gate, map_digest)
    authority._validate_hook_identity(
        hook_identity, historical_gate, parity, attestation
    )
    authority._validate_fall_identity(fall_identity, historical_gate)

    for path, expected, label in (
        (cm, cm_digest, "supplied CM oracle"),
        (pmove, pmove_digest, "supplied Pmove oracle"),
        (hook, hook_digest, "supplied hook oracle"),
        (fall, fall_digest, "supplied fall oracle"),
        (map_path, map_digest, "supplied probe BSP"),
    ):
        if authority._file_sha256(path, label) != expected:
            raise B1RequalificationError(f"{label} changed during requalification")

    seal = authority.B1RuntimeAuthoritySeal(
        schema=authority.SEAL_SCHEMA,
        design_sha256=historical_gate.design_sha256,
        plan_sha256=historical_gate.plan_sha256,
        hook_parity_attestation_sha256=parity.attestation_sha256,
        fixture_bsp_sha256=parity.fixture_bsp_sha256,
        analysis_bsp_sha256=map_digest,
        cm_executable_sha256=cm_digest,
        pmove_executable_sha256=pmove_digest,
        hook_executable_sha256=hook_digest,
        fall_executable_sha256=fall_digest,
        collision_tool_identity=str(cm_identity["tool_identity"]),
        collision_physics_identity=str(cm_identity["physics_identity"]),
        pmove_tool_identity=str(pmove_identity["tool_identity"]),
        pmove_physics_identity=str(pmove_identity["physics_identity"]),
        hook_tool_identity=str(hook_identity["tool_identity"]),
        hook_physics_identity=str(hook_identity["physics_identity"]),
        fall_tool_identity=str(fall_identity["tool_identity"]),
        fall_physics_identity=str(fall_identity["physics_identity"]),
    )
    identities = {
        "collision": {
            "tool_identity": seal.collision_tool_identity,
            "physics_identity": seal.collision_physics_identity,
        },
        "pmove": {
            "tool_identity": seal.pmove_tool_identity,
            "physics_identity": seal.pmove_physics_identity,
            "parameters": deepcopy(pmove_identity["parameters"]),
        },
        "hook": {
            "tool_identity": seal.hook_tool_identity,
            "physics_identity": seal.hook_physics_identity,
        },
        "fall": {
            "tool_identity": seal.fall_tool_identity,
            "physics_identity": seal.fall_physics_identity,
        },
    }
    return seal, identities


def build_requalified_gate(
    *,
    historical_document: Mapping[str, Any],
    historical_gate: authority.B1AuthorityGate,
    historical_seal: authority.B1RuntimeAuthoritySeal,
    live_identities: Mapping[str, Any],
    repo_root: Path,
    repository: Mapping[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    """Build and self-validate a fresh gate from already verified evidence."""

    design_sha256 = authority._file_sha256(
        repo_root / authority.DESIGN_RELATIVE_PATH, "amended normative design"
    )
    plan_sha256 = authority._file_sha256(
        repo_root / authority.PLAN_RELATIVE_PATH, "amended normative plan"
    )
    if design_sha256 != authority.ACCEPTED_DESIGN_SHA256:
        raise B1RequalificationError("amended design bytes are not accepted")
    if plan_sha256 != authority.ACCEPTED_PLAN_SHA256:
        raise B1RequalificationError("amended plan bytes are not accepted")
    if (
        historical_seal.design_sha256 != authority.HISTORICAL_DESIGN_SHA256
        or historical_seal.plan_sha256 != authority.HISTORICAL_PLAN_SHA256
    ):
        raise B1RequalificationError("live verification was not rooted in historical B1")
    if (
        historical_seal.cm_executable_sha256
        != historical_gate.cm_executable_sha256
        or historical_seal.pmove_executable_sha256
        != historical_gate.pmove_executable_sha256
        or historical_seal.hook_executable_sha256
        != authority.HISTORICAL_HOOK_EXECUTABLE_SHA256
        or historical_seal.fall_executable_sha256
        != historical_gate.fall_executable_sha256
        or historical_seal.hook_parity_attestation_sha256
        != historical_gate.hook_attestation_sha256
    ):
        raise B1RequalificationError(
            "live verification inputs differ from historical B1"
        )
    if not isinstance(recorded_at, str) or not recorded_at:
        raise B1RequalificationError("recorded_at must be nonempty")

    candidate = deepcopy(dict(historical_document))
    candidate["normative_documents"] = {
        "design_sha256": design_sha256,
        "plan_sha256": plan_sha256,
    }
    candidate["amended_at"] = recorded_at
    probe_runtime_seal = historical_seal.as_dict()
    probe_runtime_seal["normative_documents"] = {
        "design_sha256": design_sha256,
        "plan_sha256": plan_sha256,
    }
    candidate["authority_requalification"] = {
        "schema": authority.REQUALIFICATION_SCHEMA,
        "status": "green",
        "recorded_at": recorded_at,
        "historical_gate_sha256": authority.HISTORICAL_GATE_SHA256,
        "historical_normative_documents": {
            "design_sha256": authority.HISTORICAL_DESIGN_SHA256,
            "plan_sha256": authority.HISTORICAL_PLAN_SHA256,
        },
        "current_normative_documents": {
            "design_sha256": design_sha256,
            "plan_sha256": plan_sha256,
        },
        "probe_bsp_sha256": historical_seal.analysis_bsp_sha256,
        "repository": dict(repository),
        "inputs": {
            "hook_parity_attestation_sha256": (
                historical_seal.hook_parity_attestation_sha256
            ),
            "executables": {
                "cm_sha256": historical_seal.cm_executable_sha256,
                "pmove_sha256": historical_seal.pmove_executable_sha256,
                "hook_sha256": historical_seal.hook_executable_sha256,
                "fall_sha256": historical_seal.fall_executable_sha256,
            },
        },
        "live_identities": deepcopy(dict(live_identities)),
        "probe_runtime_authority_seal": probe_runtime_seal,
        "checks": {
            "historical_gate_exact_bytes": True,
            "normative_documents_rehashed": True,
            "repository_clean": True,
            "executable_bytes_match_historical_gate": True,
            "hook_attestation_revalidated": True,
            "live_identities_recomputed": True,
            "live_identity_preimages_validated": True,
        },
        "failures": [],
    }
    tests = candidate.get("tests")
    if not isinstance(tests, dict):
        raise B1RequalificationError("historical B1 tests evidence is malformed")
    tests["authority_requalification"] = {
        "historical_gate_sha256": authority.HISTORICAL_GATE_SHA256,
        "historical_evidence_reexecuted": False,
        "byte_identical_oracles_reverified": 4,
        "live_identity_probes_passed": 4,
        "normative_documents_rehashed": 2,
    }

    try:
        authority._validate_gate_document(candidate, repo_root)
    except authority.B1AuthorityError as error:
        raise B1RequalificationError(
            f"fresh B1 candidate failed self-validation: {error}"
        ) from error
    return candidate


def requalify(
    *,
    repo_root: Path,
    historical_gate_path: Path,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_parity_attestation: Path,
    probe_bsp: Path,
    recorded_at: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    historical_document = _load_historical_document(historical_gate_path)
    historical_gate = authority.load_historical_b1_authority_gate(
        historical_gate_path, repo_root=repo_root
    )
    seal, identities = _reverify_live_inputs(
        historical_gate=historical_gate,
        cm_oracle=cm_oracle,
        pmove_oracle=pmove_oracle,
        hook_oracle=hook_oracle,
        fall_oracle=fall_oracle,
        hook_parity_attestation=hook_parity_attestation,
        probe_bsp=probe_bsp,
        timeout_seconds=timeout_seconds,
    )
    return build_requalified_gate(
        historical_document=historical_document,
        historical_gate=historical_gate,
        historical_seal=seal,
        live_identities=identities,
        repo_root=repo_root,
        repository=repository_binding(repo_root),
        recorded_at=recorded_at,
    )


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise B1RequalificationError(f"output already exists: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--historical-gate", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-parity-attestation", type=Path, required=True)
    parser.add_argument("--probe-bsp", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--recorded-at", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check_only == (args.output is not None):
        raise B1RequalificationError(
            "choose exactly one of --check-only or --output"
        )
    repo_root = args.repo_root.resolve()
    recorded_at = args.recorded_at or _timestamp_now()
    candidate = requalify(
        repo_root=repo_root,
        historical_gate_path=args.historical_gate,
        cm_oracle=args.cm_oracle,
        pmove_oracle=args.pmove_oracle,
        hook_oracle=args.hook_oracle,
        fall_oracle=args.fall_oracle,
        hook_parity_attestation=args.hook_parity_attestation,
        probe_bsp=args.probe_bsp,
        recorded_at=recorded_at,
        timeout_seconds=args.timeout_seconds,
    )
    payload = authority._canonical_json_bytes(candidate)
    if args.output is not None:
        _write_new(args.output.resolve(), payload)
    summary = {
        "schema": authority.REQUALIFICATION_SCHEMA,
        "status": "green",
        "check_only": bool(args.check_only),
        "gate_sha256": hashlib.sha256(payload).hexdigest(),
        "output": None if args.output is None else str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (B1RequalificationError, authority.B1AuthorityError) as error:
        print(f"B1 requalification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
