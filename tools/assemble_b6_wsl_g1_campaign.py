#!/usr/bin/env python3
"""Assemble the B6 WSL no-update G1 transport campaign evidence.

The 16,384-transition soak is produced by ``train.multires_one_run`` in
``b6-wsl-g1-no-update`` mode.  This tool never launches it.  It consumes the
sealed raw trajectory, operational fault probes, WSL B2 performance evidence,
public-service before/after probes, and the prior B4/B5/lineage/retirement
closure.  Every conclusion is recomputed before one exclusive output is
published.
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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.runtime_attestation import verify_runtime_manifest
from harness.causal_protocol import CAUSAL_TELEMETRY_SIZE
from harness.client_protocol import CLIENT_TELEMETRY_HEADER_SIZE, CLIENT_TELEMETRY_SIZE
from harness.protocol import ACT_SIZE, OBS_SIZE
from harness.multires_training_config import MultiresTrainingConfiguration
from tools import assemble_b2_gate as b2_gate
from tools.capture_b6_public_state import (
    CAMPAIGN_ID as ATTESTED_CAMPAIGN_ID,
    SCHEMA as PUBLIC_PROBE_SCHEMA,
    PublicProbeError,
    load_authority as load_public_authority,
    verify_probe,
)
from tools.run_b6_attested_campaign import (
    SCHEMA as CONTROLLER_LEDGER_SCHEMA,
    TOOL as CONTROLLER_TOOL,
    _remote_probe_argv,
    launch_id_for,
    load_plan as load_controller_plan,
)
from tools.assemble_b3_gate import (
    GATE_SCHEMA as B3_GATE_SCHEMA,
    validate_b3_gate,
)
from tools.assemble_b4_evidence import (
    B4EvidenceError,
    validate_b4_evidence,
)
from tools.assemble_b5_gate import (
    B5_PREDICATE_KEYS,
    gate_sha256 as b5_gate_sha256,
)
from tools.run_multires_pretraining_validation import (
    PretrainingValidationError,
    REQUIRED_CAMPAIGN_MODES,
    REQUIRED_REPLICATES,
    SUITE_SCHEMA as B5_SUITE_SCHEMA,
    TOOL_NAME as B5_SUITE_TOOL,
    canonical_bytes as b5_canonical_bytes,
    canonical_sha256 as b5_canonical_sha256,
    report_sha256 as b5_report_sha256,
    validate_runtime_input as validate_b5_runtime_input,
    validate_campaign_evidence as validate_b5_campaign_evidence,
    validate_campaign_set as validate_b5_campaign_set,
    validate_proof as validate_b5_proof,
)
from tools.run_multires_500_transition_proof import (
    COLLECTOR_CLASS_NAME,
    LATTICE_CRATE_NAME,
    ONE_RUN_PROTOCOL_VERSION,
    ONE_RUN_SCHEMA,
    PYTHON_COLLECTOR_SCHEMA,
    RUST_PROVIDER_SCHEMA,
    SPATIAL_PROVIDER_CLASS_NAME,
    parse_canonical_transition_record,
    trajectory_sha256,
)
from tools.qualify_network_client_frame_barrier import (
    QualificationError as NetworkQualificationError,
    _validate_execution_evidence as validate_network_execution_evidence,
)
from tools.verify_multires_integration import (
    _check_legacy_selector_deactivation,
    _check_lineage_attestation,
)
from train.multires_one_run import B6_CAMPAIGN_MODE


SCHEMA = "q2-multires-b6-wsl-g1-v1"
TOOL = "assemble_b6_wsl_g1_campaign"
FAULT_SCHEMA = "q2-multires-b6-g1-fault-probe-v1"
PUBLIC_HOST_AUTHORITY_PATH = ROOT / "docs/multires/B6-PUBLIC-HOST-AUTHORITY.json"
REQUIRED_TRANSITIONS = 16_384
CLIENT_COUNT = 4
MIN_ECHO_ACCEPT_RATE = 0.97
MIN_VERTICAL_MATCH_RATE = 0.99
MAX_FEATURE_P99_NS = 500_000
MAX_ATLAS_BYTES = 32 * 1024 * 1024
MAX_DYN_BYTES = 8 * 1024 * 1024
MAX_BUILD_RSS_BYTES = 512 * 1024 * 1024
MAX_DECLARED_RESYNC_LIMIT = 64
_SHA = re.compile(r"(?!0{64})[0-9a-f]{64}\Z")
_GIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class B6CampaignError(RuntimeError):
    """No green B6 campaign can be derived from the supplied evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B6CampaignError(message)


def _duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise B6CampaignError(f"non-finite JSON constant {value!r}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B6CampaignError("evidence is not canonical JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             f"{label} must be a non-placeholder SHA-256")
    return str(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum,
             f"{label} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value))
             and float(value) >= minimum, f"{label} must be finite >= {minimum}")
    return float(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(actual == expected,
             f"{label} keys differ; missing={sorted(expected-actual)} extra={sorted(actual-expected)}")


def _regular(path: Path, label: str) -> Path:
    source = Path(os.path.abspath(path.expanduser()))
    _require(source.is_absolute() and not source.is_symlink() and source.is_file(),
             f"{label} must be an absolute regular non-symlink file")
    return source


def file_record(path: Path) -> dict[str, Any]:
    data = _regular(path, str(path)).read_bytes()
    _require(bool(data), f"{path} is empty")
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load_json(
    path: Path, label: str, *, newline: bool | None = None,
    manifest_style: bool = False,
) -> dict[str, Any]:
    source = _regular(path, label)
    data = source.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise B6CampaignError(f"{label} is not strict JSON") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    encodings = {canonical_bytes(value), canonical_bytes(value) + b"\n"}
    if manifest_style:
        encodings.add(
            (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )
    if newline is True:
        encodings = {canonical_bytes(value) + b"\n"}
    elif newline is False:
        encodings = {canonical_bytes(value)}
    _require(data in encodings, f"{label} is not canonically encoded")
    return value


def _verify_simple_seal(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    digest = body.pop(field, None)
    _digest(digest, f"{label} seal")
    _require(digest == canonical_sha256(body), f"{label} seal differs")
    return str(digest)


def gate_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("gate_sha256", None)
    return canonical_sha256({"domain": SCHEMA, "gate": body})


def _git_identity(path: Path, label: str) -> dict[str, Any]:
    repo = Path(os.path.abspath(path.expanduser()))
    _require(repo.is_dir() and not repo.is_symlink(), f"{label} repository is invalid")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo, text=True, stderr=subprocess.STDOUT,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise B6CampaignError(f"{label} repository cannot be attested") from error
    _require(not status, f"{label} repository is dirty")
    _require(_GIT.fullmatch(commit) is not None and _GIT.fullmatch(tree) is not None,
             f"{label} Git identity is malformed")
    return {"commit": commit, "tree": tree, "clean": True}


def _validate_b4(value: Mapping[str, Any], atlas_sha256: str) -> dict[str, Any]:
    components = _mapping(
        value.get("component_evidence"), "B4 component evidence"
    )
    try:
        validate_b4_evidence(value, components)
    except B4EvidenceError as error:
        raise B6CampaignError(f"B4 evidence rejected: {error}") from error
    _require(value.get("schema") == "q2-multires-b4-evidence-v1"
             and value.get("milestone") == "B4" and value.get("status") == "green",
             "B4 evidence is not green")
    _require(value.get("atlas_sha256") == atlas_sha256, "B4 Atlas binding differs")
    sources = _mapping(value.get("source_repositories"), "B4 source repositories")
    _exact_keys(sources, {"bot", "client", "game"}, "B4 source repositories")
    for name, raw in sources.items():
        identity = _mapping(raw, f"B4 {name} source")
        _exact_keys(identity, {"commit", "tree", "clean"}, f"B4 {name} source")
        _require(identity.get("clean") is True, f"B4 {name} source was not clean")
    return dict(value)


def _validate_active_b2(value: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    try:
        validated = b2_gate.validate_gate(value)
        authority = b2_gate._require_active_final_authority()
    except b2_gate.B2GateError as error:
        raise B6CampaignError(f"B2 active authority rejected: {error}") from error
    generated = _mapping(validated.get("generated_cohort"), "B2 generated cohort")
    _require(
        generated.get("cohort_id") == authority.cohort_id
        and generated.get("cohort_id") != b2_gate.RETIRED_COHORT_71446
        and generated.get("declaration_sha256") == authority.declaration_sha256,
        "B2 gate does not bind the exact active final cohort/declaration",
    )
    return dict(validated), authority


def _validate_b5(
    value: Mapping[str, Any], *, objective_identity_sha256: str,
) -> dict[str, Any]:
    _exact_keys(value, {
        "schema", "tool", "status", "green", "source",
        "normative_documents", "runtime_bindings", "evidence",
        "component_evidence", "campaigns", "proof", "privilege",
        "predicates", "predecessor", "runtime_manifest_identity_sha256",
        "failures", "gate_sha256",
    }, "B5 gate")
    _require(value.get("schema") == "q2-multires-b5-gate-v1"
             and value.get("tool") == "assemble_b5_gate"
             and value.get("status") == "green" and value.get("green") is True
             and value.get("failures") == [],
             "B5 gate is not green")
    _digest(value.get("gate_sha256"), "B5 gate seal")
    expected = b5_gate_sha256(value)
    _require(value["gate_sha256"] == expected, "B5 gate seal differs")
    components = _mapping(
        value.get("component_evidence"), "B5 component evidence"
    )
    _exact_keys(
        components, {"validation_report", "proof_report"},
        "B5 component evidence",
    )
    validation = _mapping(
        components["validation_report"], "B5 validation report"
    )
    raw_proof = _mapping(components["proof_report"], "B5 proof report")
    evidence = _mapping(value.get("evidence"), "B5 evidence records")
    validation_bytes = b5_canonical_bytes(validation)
    proof_bytes = b5_canonical_bytes(raw_proof)[:-1]
    _require(
        evidence.get("validation_report") == {
            "bytes": len(validation_bytes),
            "sha256": hashlib.sha256(validation_bytes).hexdigest(),
        }
        and evidence.get("proof_report") == {
            "bytes": len(proof_bytes),
            "sha256": hashlib.sha256(proof_bytes).hexdigest(),
        },
        "B5 embedded component bytes differ from evidence records",
    )
    _require(
        validation.get("schema") == B5_SUITE_SCHEMA
        and validation.get("tool") == B5_SUITE_TOOL
        and validation.get("status") == "passed"
        and validation.get("passed") is True
        and validation.get("failures") == []
        and validation.get("report_sha256") == b5_report_sha256(validation),
        "B5 validation report identity/seal differs",
    )
    _require(
        validation.get("source") == value.get("source")
        and validation.get("bindings") == value.get("runtime_bindings")
        and validation.get("runtime_manifest_identity_sha256")
        == value.get("runtime_manifest_identity_sha256")
        and validation.get("b4_privilege_admission") == value.get("privilege"),
        "B5 validation source/runtime/privilege binding differs",
    )
    seeds = _mapping(validation.get("seeds"), "B5 validation seeds")
    transition_count = _integer(
        validation.get("campaign_transition_count"),
        "B5 campaign transition count", minimum=1,
    )
    expected_schedule = [
        {"campaign": mode, "replicate": replicate}
        for mode in REQUIRED_CAMPAIGN_MODES
        for replicate in REQUIRED_REPLICATES
    ]
    _require(validation.get("campaign_schedule") == expected_schedule,
             "B5 campaign schedule differs")
    source = _mapping(value.get("source"), "B5 source")
    source_pair = {"commit": source.get("commit"), "tree": source.get("tree")}
    bindings = _mapping(value.get("runtime_bindings"), "B5 runtime bindings")
    campaigns = []
    try:
        for raw in validation.get("campaigns", []):
            item = _mapping(raw, "B5 campaign evidence")
            campaigns.append(validate_b5_campaign_evidence(
                item, mode=item.get("campaign"),
                replicate=item.get("replicate"), seed=seeds.get("seed"),
                game_seed=seeds.get("game_seed"),
                transition_count=transition_count, source=source_pair,
                bindings=bindings,
                runtime_manifest_sha256=value[
                    "runtime_manifest_identity_sha256"
                ],
            ))
        summary = validate_b5_campaign_set(campaigns)
        proof = validate_b5_proof(
            raw_proof, seed=seeds.get("seed"),
            game_seed=seeds.get("game_seed"), bindings=bindings,
            runtime_manifest_sha256=value["runtime_manifest_identity_sha256"],
            objective_identity_sha256=objective_identity_sha256,
        )
    except PretrainingValidationError as error:
        raise B6CampaignError(f"B5 component evidence rejected: {error}") from error
    _require(
        summary == validation.get("campaign_summary")
        and summary == value.get("campaigns"),
        "B5 campaign summary does not derive from embedded campaigns",
    )
    proof_section = _mapping(validation.get("proof"), "B5 validation proof")
    _require(
        proof_section.get("record") == evidence["proof_report"]
        and proof_section.get("schema") == proof.get("schema")
        and proof_section.get("tool") == proof.get("tool")
        and proof_section.get("transition_count") == proof.get("transition_count")
        and proof_section.get("same_seed_match") == proof.get("same_seed_match")
        and proof_section.get("different_seed_diverges")
        == proof.get("different_seed_diverges")
        and proof_section.get("verifier_evidence_sha256")
        == b5_canonical_sha256(proof["verifier_evidence"])
        and value.get("proof") == {
            "transition_count": proof["transition_count"],
            "same_seed_match": True,
            "different_seed_diverges": True,
            "production_pass": True,
        },
        "B5 proof summary does not derive from embedded proof",
    )
    _require(validation.get("no_update") == {
        "checkpoint_unchanged": True,
        "policy_state_unchanged": True,
        "optimizer_state_unchanged": True,
        "optimizer_steps": 0,
        "backward_parameter_gradients": 0,
    }, "B5 no-update evidence differs")
    predicates = _mapping(value.get("predicates"), "B5 predicates")
    _require(
        set(predicates) == B5_PREDICATE_KEYS
        and all(item is True for item in predicates.values()),
        "B5 predicates differ from the producer contract",
    )
    return dict(value)


def _validate_runtime_files(
    *, b4: Mapping[str, Any], b5: Mapping[str, Any],
    runtime_evidence_path: Path, runtime_manifest_path: Path,
    atlas_sha256: str,
) -> dict[str, Any]:
    """Cross-bind B5's compact evidence to B4's full sealed manifest."""
    _require(
        runtime_evidence_path.resolve() != runtime_manifest_path.resolve(),
        "compact runtime evidence and full runtime manifest must be distinct files",
    )
    compact_record = file_record(runtime_evidence_path)
    try:
        compact_runtime_identity = validate_b5_runtime_input(
            runtime_evidence_path, atlas_sha256
        )
    except PretrainingValidationError as error:
        raise B6CampaignError(
            f"compact B4 runtime evidence is inadmissible: {error}"
        ) from error
    full_record = file_record(runtime_manifest_path)
    manifest = load_json(
        runtime_manifest_path, "runtime manifest", manifest_style=True
    )
    verified = verify_runtime_manifest(manifest)
    _require(
        verified.valid and _SHA.fullmatch(verified.digest) is not None,
        "runtime manifest is not sealed",
    )
    _require(
        compact_runtime_identity == verified.digest,
        "compact runtime evidence/full runtime manifest identity differs",
    )
    _require(
        b5.get("runtime_manifest_identity_sha256") == verified.digest,
        "B5 runtime semantic identity differs",
    )
    b4_manifest = _mapping(b4.get("runtime_manifest"), "B4 runtime manifest")
    _require(
        b4_manifest.get("sha256") == full_record["sha256"]
        and b4_manifest.get("size") == full_record["bytes"]
        and b4_manifest.get("manifest_sha256") == verified.digest,
        "B4 runtime manifest binding differs",
    )
    return {
        "compact_record": compact_record,
        "full_record": full_record,
        "manifest": manifest,
        "manifest_sha256": verified.digest,
    }


def _validate_bindings(
    *, b4: Mapping[str, Any], b5: Mapping[str, Any],
    runtime_evidence_path: Path, runtime_manifest_path: Path,
    checkpoint_path: Path, training_manifest_path: Path,
    objectives_path: Path, bundle_path: Path, atlas_path: Path,
    atlas_manifest_path: Path, b3_gate_path: Path, retirement_path: Path,
    b2_gate_path: Path, b4_evidence_path: Path, repo_root: Path,
) -> dict[str, Any]:
    # Preserve B5's historical ``runtime_manifest`` binding name.  At that
    # boundary it deliberately denotes the compact B4 wire-generation
    # evidence consumed by ``validate_runtime_evidence``.  B6 additionally
    # needs the full sealed runtime manifest consumed by one-run.  They are
    # distinct immutable files and must converge on one semantic manifest
    # digest; neither file can stand in for the other.
    b5_paths = {
        "b3_gate": b3_gate_path,
        "b4_evidence": b4_evidence_path,
        "runtime_manifest": runtime_evidence_path,
        "checkpoint": checkpoint_path,
        "training_manifest": training_manifest_path,
        "bundle_manifest": bundle_path,
        "objectives": objectives_path,
        "atlas": atlas_path,
    }
    current = {name: file_record(path) for name, path in b5_paths.items()}
    b5_bindings = _mapping(b5.get("runtime_bindings"), "B5 runtime bindings")
    for name, record in current.items():
        _require(b5_bindings.get(name) == record, f"B5 {name} binding differs")
    _require(
        set(b5_bindings) == set(b5_paths),
        "B5 runtime binding inventory differs",
    )
    runtime_files = _validate_runtime_files(
        b4=b4,
        b5=b5,
        runtime_evidence_path=runtime_evidence_path,
        runtime_manifest_path=runtime_manifest_path,
        atlas_sha256=current["atlas"]["sha256"],
    )
    _require(
        current["runtime_manifest"] == runtime_files["compact_record"],
        "B5 compact runtime evidence binding changed during validation",
    )
    full_runtime_record = runtime_files["full_record"]
    manifest = runtime_files["manifest"]
    runtime_manifest_sha256 = runtime_files["manifest_sha256"]

    retirement_record = file_record(retirement_path)
    retirement = load_json(retirement_path, "retirement manifest")
    _require(retirement.get("schema") == "q2-multires-runtime-retirement-v1"
             and retirement.get("status") == "legacy-runtime-retired"
             and retirement.get("fallback_allowed") is False,
             "retirement manifest does not enforce one-way retirement")
    try:
        runtime_retirement = manifest["semantic"]["runtime_config"][
            "retirement_manifest_sha256"
        ]
    except (KeyError, TypeError) as error:
        raise B6CampaignError("runtime manifest omits retirement binding") from error
    _require(runtime_retirement == retirement_record["sha256"],
             "runtime retirement binding differs")

    training = load_json(training_manifest_path, "training manifest")
    body = {
        key: training[key]
        for key in ("schema", "reward", "guide_dropout", "ppo")
        if key in training
    }
    _require(set(body) == {"schema", "reward", "guide_dropout", "ppo"},
             "training manifest body differs")
    training_configuration = MultiresTrainingConfiguration.create(
        reward=body["reward"], guide_dropout=body["guide_dropout"], ppo=body["ppo"]
    )
    _require(training.get("sha256", training_configuration.sha256)
             == training_configuration.sha256,
             "training manifest declared identity differs")
    try:
        runtime_config = manifest["semantic"]["runtime_config"]
        runtime_training = runtime_config["training_config_sha256"]
        network_barrier_execution = runtime_config[
            "network_barrier_execution_evidence_sha256"
        ]
    except (KeyError, TypeError) as error:
        raise B6CampaignError("runtime manifest omits training identity") from error
    _require(runtime_training == training_configuration.sha256,
             "runtime/training semantic identity differs")
    _digest(network_barrier_execution, "network barrier execution identity")

    objectives_record = file_record(objectives_path)
    load_json(objectives_path, "objectives")
    objective_identity = objectives_record["sha256"]
    _digest(objective_identity, "objective identity")
    bundle = load_json(bundle_path, "bundle manifest")
    _require(bundle.get("analysis_files", {}).get(Path(objectives_path).name)
             == objectives_record["sha256"],
             "bundle does not bind the exact objectives bytes")

    atlas_manifest_record = file_record(atlas_manifest_path)
    atlas_manifest = load_json(
        atlas_manifest_path, "Atlas manifest", manifest_style=True
    )
    _require(atlas_manifest.get("recovery_physics") == {
        "hook_walk_budget_ticks": 15,
        "game_tick_hz": 10,
        "walk_speed_q8_per_second": 76800,
    }, "Atlas hook-necessity budget/cadence identity differs")
    atlas_artifact = _mapping(
        atlas_manifest.get("artifacts"), "Atlas manifest artifacts"
    ).get(Path(atlas_path).name)
    _require(isinstance(atlas_artifact, Mapping)
             and atlas_artifact.get("sha256_uncompressed") == current["atlas"]["sha256"]
             and atlas_artifact.get("uncompressed_size") == current["atlas"]["bytes"],
             "Atlas manifest does not bind the exact raw Atlas")

    b3_gate_record = file_record(b3_gate_path)
    b3_gate = validate_b3_gate(load_json(b3_gate_path, "B3 gate"))
    _require(b3_gate.get("schema") == B3_GATE_SCHEMA
             and b3_gate.get("batch") == "B3"
             and b3_gate.get("status") == "green",
             "B3 predecessor gate is not green")
    b2_record = file_record(b2_gate_path)
    b2_value, authority = _validate_active_b2(
        load_json(b2_gate_path, "B2 gate", newline=False)
    )
    b2_generated = _mapping(
        b2_value.get("generated_cohort"), "B2 generated cohort"
    )
    expected_b3_predecessor = {
        "b2_gate": b2_record,
        "status": "green",
        "cohort_id": authority.cohort_id,
        "declaration_sha256": authority.declaration_sha256,
    }
    _require(
        b3_gate.get("predecessor") == expected_b3_predecessor,
        "B3 predecessor does not bind the exact supplied B2 gate/authority",
    )
    _require(
        b2_generated.get("cohort_id") == authority.cohort_id
        and b2_generated.get("declaration_sha256")
        == authority.declaration_sha256,
        "B2 generated cohort differs from active authority",
    )
    _require(b3_gate.get("recovery_guide", {}).get("hook_walk_budget_ticks") == 15
             and b3_gate.get("recovery_guide", {}).get("game_tick_hz") == 10
             and b3_gate.get("recovery_guide", {}).get(
                 "walk_speed_q8_per_second"
             ) == 76800,
             "B3 hook-necessity budget/cadence summary differs")
    b3_repository = _mapping(b3_gate.get("repository"), "B3 repository")
    b4_bot = _mapping(b4.get("source_repositories"), "B4 sources")["bot"]
    _require(b3_repository.get("repository_commit") == b4_bot.get("commit")
             and b3_repository.get("repository_tree") == b4_bot.get("tree")
             and b3_repository.get("git_clean") is True,
             "B3/B4 bot source identity differs")
    b3_recovery = _mapping(b3_gate.get("recovery_guide"), "B3 recovery guide")
    expected_b4_predecessor = {
        "b3_gate": {
            "name": "B3-gate",
            "sha256": b3_gate_record["sha256"],
            "size": b3_gate_record["bytes"],
        },
        "b3_gate_sha256": b3_gate["gate_sha256"],
        "status": "green",
        "cohort_id": authority.cohort_id,
        "declaration_sha256": authority.declaration_sha256,
        "atlas_set_sha256": b3_recovery.get("atlas_set_sha256"),
        "repository_commit": b4_bot.get("commit"),
        "repository_tree": b4_bot.get("tree"),
    }
    _require(
        b4.get("predecessor") == expected_b4_predecessor,
        "B4 predecessor does not bind the exact supplied B3/B2 chain",
    )
    expected_b5_predecessor = {
        "b3_gate": b3_gate_record,
        "b3_gate_sha256": b3_gate["gate_sha256"],
        "cohort_id": authority.cohort_id,
        "declaration_sha256": authority.declaration_sha256,
        "atlas_set_sha256": b3_recovery.get("atlas_set_sha256"),
    }
    _require(
        b5.get("predecessor") == expected_b5_predecessor,
        "B5 predecessor does not bind the exact supplied B3/B4/B2 chain",
    )
    rust_extension = _mapping(
        b3_gate.get("recovery_guide", {}).get("rust_extension"),
        "B3 Rust extension",
    )
    for name in (
        "sha256", "source_closure_sha256", "qualification_commands_sha256"
    ):
        _digest(rust_extension.get(name), f"B3 Rust extension {name}")
    _require(_integer(rust_extension.get("bytes"), "B3 Rust extension bytes", minimum=1) >= 1
             and rust_extension.get("repository_tree") == b4_bot.get("tree"),
             "B3 Rust extension source identity differs")
    try:
        runtime_rust = manifest["semantic"]["artifacts"]["rust_lattice"]
    except (KeyError, TypeError) as error:
        raise B6CampaignError("runtime omits Rust lattice artifact") from error
    _require(runtime_rust.get("enabled") is True
             and runtime_rust.get("sha256") == rust_extension.get("sha256")
             and runtime_rust.get("size") == rust_extension.get("bytes"),
             "B3/runtime Rust extension bytes differ")

    b4_sources = b4["source_repositories"]
    roots = {
        "bot": Path(repo_root),
        "client": Path(repo_root).resolve().parent / "q2-ml-client",
        "game": Path(repo_root).resolve().parent / "q2-lithium-3zb2",
    }
    for name, source in roots.items():
        _require(_git_identity(source, name) == b4_sources[name],
                 f"current {name} source differs from B4")
    _require(b5.get("source") == b4_sources["bot"],
             "B5/B4 bot source closure differs")
    return {
        **{
            name: record
            for name, record in current.items()
            if name != "runtime_manifest"
        },
        "runtime_evidence": current["runtime_manifest"],
        "runtime_manifest": full_runtime_record,
        "objectives": objectives_record,
        "atlas_manifest": atlas_manifest_record,
        "b3_gate": b3_gate_record,
        "b3_gate_sha256": b3_gate["gate_sha256"],
        "retirement_manifest": retirement_record,
        "runtime_manifest_identity_sha256": runtime_manifest_sha256,
        "training_config_sha256": training_configuration.sha256,
        "objective_identity_sha256": objective_identity,
        "network_barrier_execution_evidence_sha256": network_barrier_execution,
        "hook_necessity_runtime": {
            "hook_walk_budget_ticks": 15,
            "game_tick_hz": 10,
            "walk_speed_q8_per_second": 76800,
        },
        "rust_extension": dict(rust_extension),
        "b2_authority": {
            "cohort_id": authority.cohort_id,
            "declaration_sha256": authority.declaration_sha256,
        },
    }


def _validate_one_run(
    raw: Mapping[str, Any], *, bindings: Mapping[str, Any],
    atlas_sha256: str, runtime_manifest_sha256: str,
    retirement_sha256: str, expected_sources: Mapping[str, Any],
    expected_launch_id: str,
) -> dict[str, Any]:
    _verify_simple_seal(raw, "evidence_sha256", "one-run")
    expected_identity = {
        "schema": ONE_RUN_SCHEMA,
        "protocol_version": ONE_RUN_PROTOCOL_VERSION,
        "synthetic": False,
        "legacy": False,
        "collector": COLLECTOR_CLASS_NAME,
        "python_collector_schema": PYTHON_COLLECTOR_SCHEMA,
        "spatial_provider": SPATIAL_PROVIDER_CLASS_NAME,
        "rust_provider_schema": RUST_PROVIDER_SCHEMA,
        "lattice_crate": LATTICE_CRATE_NAME,
        "campaign_mode": B6_CAMPAIGN_MODE,
        "atlas_sha256": atlas_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "bundle_manifest_sha256": bindings["bundle_manifest"]["sha256"],
        "checkpoint_sha256": bindings["checkpoint"]["sha256"],
        "training_manifest_sha256": bindings["training_config_sha256"],
        "objective_identity_sha256": bindings["objective_identity_sha256"],
        "network_barrier_execution_evidence_sha256": bindings[
            "network_barrier_execution_evidence_sha256"
        ],
        "retirement_manifest_sha256": retirement_sha256,
        "transition_count": REQUIRED_TRANSITIONS,
    }
    for name, expected in expected_identity.items():
        _require(raw.get(name) == expected,
                 f"one-run {name}={raw.get(name)!r} expected {expected!r}")
    _require(raw.get("source_repositories") == expected_sources,
             "one-run source repositories differ from B4/fault closure")
    started_at = _integer(
        raw.get("started_at_unix_ns"), "one-run start time", minimum=1
    )
    completed_at = _integer(
        raw.get("completed_at_unix_ns"), "one-run completion time", minimum=1
    )
    _require(started_at < completed_at, "one-run wall-clock interval is invalid")
    received = _mapping(raw.get("received_inputs"), "one-run received inputs")
    _require(received.get("campaign_mode") == B6_CAMPAIGN_MODE
             and received.get("transition_count") == REQUIRED_TRANSITIONS
             and received.get("launch_id") == expected_launch_id,
             "one-run did not echo the B6 arbitrary-count launch")
    host = _mapping(raw.get("host"), "one-run host")
    _exact_keys(host, {
        "hostname", "kernel_release", "architecture", "machine_identity_sha256"
    }, "one-run host")
    _require(host.get("hostname") == "DESKTOP-RTX2080"
             and "microsoft-standard-WSL2" in str(host.get("kernel_release"))
             and host.get("architecture") == "x86_64",
             "B6 soak did not run on exact DESKTOP-RTX2080 WSL2")
    _digest(host.get("machine_identity_sha256"), "one-run machine identity")

    metrics = _mapping(raw.get("transport_metrics"), "one-run transport metrics")
    declared_boundary = _integer(metrics.get("boundary_rounds"), "boundary rounds")

    records = raw.get("records")
    _require(isinstance(records, list) and len(records) == REQUIRED_TRANSITIONS,
             "one-run does not contain exactly 16,384 raw records")
    built = []
    first_clients: list[str] = []
    prior_frame: int | None = None
    frame_discontinuities: list[dict[str, int]] = []
    vertical_matches = 0
    water_samples = land_samples = 0
    water_mismatches = land_mismatches = 0
    for index, item in enumerate(records):
        record = parse_canonical_transition_record(
            item, flat_index=index, expected_atlas_sha256=atlas_sha256,
            expected_runtime_manifest_sha256=runtime_manifest_sha256,
            expected_map_name=str(raw.get("map_name")),
            expected_map_epoch=int(raw.get("map_epoch", -1)),
            expected_policy_version=int(raw.get("policy_version", -1)),
            require_exact_layout=False,
        )
        round_id, client_index = divmod(index, CLIENT_COUNT)
        if round_id == 0:
            first_clients.append(record.client_id)
        _require(record.client_id == first_clients[client_index],
                 f"records[{index}] client ordering differs")
        _require(record.batch_round_id == round_id,
                 f"records[{index}] round identity differs")
        if client_index == 0:
            if prior_frame is not None:
                _require(record.server_frame > prior_frame,
                         "accepted one-run server frame regressed")
                if record.server_frame != prior_frame + 1:
                    frame_discontinuities.append({
                        "accepted_round": round_id,
                        "prior_frame": prior_frame,
                        "current_frame": record.server_frame,
                        "delta": record.server_frame - prior_frame,
                    })
            prior_frame = record.server_frame
        else:
            _require(record.server_frame == prior_frame,
                     "one-run accepted round has cross-client frame skew")
        audit = _mapping(item.get("transport_audit"), f"records[{index}] transport audit")
        _exact_keys(audit, {
            "authoritative_echo_valid", "trainable_transition",
            "requested_vertical", "echoed_vertical", "applied_upmove",
            "water_vertical_mode",
        }, f"records[{index}] transport audit")
        requested = _integer(audit["requested_vertical"], "requested vertical")
        echoed = _integer(audit["echoed_vertical"], "echoed vertical")
        _require(requested in (0, 1, 2) and echoed in (0, 1, 2),
                 "vertical intent is outside the frozen enum")
        expected_upmove = (-320, 0, 320)[requested]
        matched = bool(
            audit["authoritative_echo_valid"] is True
            and audit["trainable_transition"] is True
            and echoed == requested
            and audit["applied_upmove"] == expected_upmove
            and int(round(float(record.action[4]))) == requested
        )
        vertical_matches += int(matched)
        if audit["water_vertical_mode"] is True:
            water_samples += 1
            water_mismatches += int(not matched)
        elif audit["water_vertical_mode"] is False:
            land_samples += 1
            land_mismatches += int(not matched)
        else:
            raise B6CampaignError("water_vertical_mode is not boolean")
        built.append(record)
    _require(len(set(first_clients)) == CLIENT_COUNT,
             "one-run does not contain four unique clients")
    _require(trajectory_sha256(tuple(built)) == raw.get("trajectory_sha256"),
             "one-run trajectory digest differs from raw records")

    no_update = _mapping(raw.get("no_update"), "one-run no-update")
    _exact_keys(no_update, {
        "mode", "checkpoint_sha256_before", "checkpoint_sha256_after",
        "policy_state_sha256_before", "policy_state_sha256_after",
        "optimizer_state_sha256_before", "optimizer_state_sha256_after",
        "policy_updates", "optimizer_steps", "backward_calls",
    }, "one-run no-update")
    for prefix in ("checkpoint", "policy_state", "optimizer_state"):
        before = _digest(no_update[f"{prefix}_sha256_before"], f"{prefix} before")
        after = _digest(no_update[f"{prefix}_sha256_after"], f"{prefix} after")
        _require(before == after, f"{prefix} changed during B6 soak")
    _require(no_update["checkpoint_sha256_before"] == bindings["checkpoint"]["sha256"],
             "no-update checkpoint differs from exact B5 checkpoint")
    _require(no_update.get("mode") == "collect-only-no-update-v1"
             and no_update.get("policy_updates") == 0
             and no_update.get("optimizer_steps") == 0
             and no_update.get("backward_calls") == 0,
             "B6 soak executed a policy/optimizer/backward update")

    accepted = _integer(metrics.get("network_client/transitions_accepted"),
                        "accepted transitions")
    _require(accepted == REQUIRED_TRANSITIONS, "accepted counter differs from raw records")
    stale = _integer(metrics.get("network_client/stale_echoes_rejected"), "stale echoes")
    mismatched = _integer(metrics.get("network_client/mismatched_echoes_rejected"),
                          "mismatched echoes")
    failed = _integer(metrics.get("network_client/failed_rounds"), "failed rounds")
    timeouts = _integer(metrics.get("network_client/echo_timeouts"), "echo timeouts")
    _require(failed == 0 and timeouts == 0, "B6 soak recorded failure/echo timeout")
    attempts = accepted + stale + mismatched
    echo_rate = accepted / attempts if attempts else 0.0
    vertical_rate = vertical_matches / accepted
    _require(echo_rate >= MIN_ECHO_ACCEPT_RATE,
             f"authoritative echo accept rate {echo_rate:.6f} is below 0.97")
    _require(vertical_rate >= MIN_VERTICAL_MATCH_RATE,
             f"vertical match rate {vertical_rate:.6f} is below 0.99")
    _require(water_samples > 0 and land_samples > 0,
             "B6 soak did not measure both water and land projection")
    _require(water_mismatches == 0 and land_mismatches == 0,
             "water/land command projection skew is nonzero")
    boundary = declared_boundary
    limit = _integer(metrics.get("declared_resync_limit"), "declared resync limit")
    _require(limit <= MAX_DECLARED_RESYNC_LIMIT,
             "B6 declared resync limit exceeds the campaign cap")
    _require(boundary <= limit, "B6 soak exceeded its declared resync bound")
    _require(len(frame_discontinuities) <= boundary,
             "accepted-frame discontinuities exceed sealed boundary admissions")
    _require(raw.get("partial_admissions") == 0
             and raw.get("stale_admissions") == 0
             and raw.get("resync_admissions") == boundary,
             "one-run admission counters differ from the raw boundary evidence")

    payload = _mapping(
        metrics.get("protocol_payload_accounting"),
        "protocol payload accounting",
    )
    _exact_keys(payload, {
        "basis", "accepted_packet_samples", "client_telemetry_bytes",
        "action_packet_bytes", "telemetry_components", "atlas_wire_fields",
        "atlas_wire_bytes_per_frame", "dyn_wire_fields",
        "dyn_wire_bytes_per_frame",
    }, "protocol payload accounting")
    _require(payload == {
        "basis": "wire-abi-struct-calcsize-v1",
        "accepted_packet_samples": REQUIRED_TRANSITIONS,
        "client_telemetry_bytes": CLIENT_TELEMETRY_SIZE,
        "action_packet_bytes": ACT_SIZE,
        "telemetry_components": {
            "header_bytes": CLIENT_TELEMETRY_HEADER_SIZE,
            "engine_observation_bytes": OBS_SIZE,
            "causal_telemetry_bytes": CAUSAL_TELEMETRY_SIZE,
        },
        "atlas_wire_fields": 0,
        "atlas_wire_bytes_per_frame": 0,
        "dyn_wire_fields": 0,
        "dyn_wire_bytes_per_frame": 0,
    }, "one-run wire-ABI payload accounting differs")
    _require(
        CLIENT_TELEMETRY_SIZE
        == CLIENT_TELEMETRY_HEADER_SIZE + OBS_SIZE + CAUSAL_TELEMETRY_SIZE,
        "client telemetry ABI component sizes do not close",
    )

    processes = raw.get("process_records")
    terminated = raw.get("terminated_process_records")
    _require(isinstance(processes, list) and len(processes) == 5
             and terminated == processes
             and raw.get("orphan_processes_after_teardown") == 0,
             "one-run PID/start-tick teardown differs")
    expected_roles = ["q2ded", *(f"network-client-{index:02d}" for index in range(4))]
    seen_pids: set[int] = set()
    for record, role in zip(processes, expected_roles):
        item = _mapping(record, "process record")
        _exact_keys(item, {"role", "pid", "start_ticks"}, "process record")
        pid = _integer(item["pid"], "process pid", minimum=2)
        _require(item["role"] == role and pid not in seen_pids
                 and _integer(item["start_ticks"], "process start ticks", minimum=1) > 0,
                 "process role/PID/start-tick closure differs")
        seen_pids.add(pid)
    ports = raw.get("owned_ports")
    _require(isinstance(ports, list) and len(ports) == 6,
             "one-run owned port teardown cardinality differs")
    seen_ports: set[int] = set()
    for record in ports:
        item = _mapping(record, "owned port")
        _exact_keys(item, {
            "role", "address", "port", "transport", "available_after_teardown"
        }, "owned port")
        port = _integer(item["port"], "owned port", minimum=1)
        _require(item["address"] == "127.0.0.1" and item["transport"] == "udp"
                 and item["available_after_teardown"] is True
                 and port <= 65535 and port not in seen_ports,
                 "owned staging port was not proven released")
        seen_ports.add(port)
    _require(
        [item.get("role") for item in ports if isinstance(item, Mapping)]
        == ["q2ded", "telemetry", "harness-00", "harness-01", "harness-02", "harness-03"],
        "one-run socket ownership roles differ",
    )
    qports = raw.get("qport_identities")
    _require(isinstance(qports, list) and len(qports) == CLIENT_COUNT,
             "one-run protocol qport inventory differs")
    for index, item in enumerate(qports):
        identity = _mapping(item, "protocol qport")
        _exact_keys(identity, {"client_index", "qport"}, "protocol qport")
        _require(identity.get("client_index") == index
                 and type(identity.get("qport")) is int
                 and 1 <= identity["qport"] <= 65535,
                 "protocol qport identity differs")
    return {
        "accepted_transitions": accepted,
        "echo_attempts": attempts,
        "authoritative_echo_accept_rate": echo_rate,
        "vertical_samples": accepted,
        "vertical_matches": vertical_matches,
        "vertical_match_rate": vertical_rate,
        "water_samples": water_samples,
        "land_samples": land_samples,
        "water_projection_mismatches": water_mismatches,
        "land_projection_mismatches": land_mismatches,
        "water_land_projection_skew": 0,
        "failed_rounds": failed,
        "echo_timeouts": timeouts,
        "declared_resyncs": boundary,
        "declared_resync_limit": limit,
        "accepted_frame_discontinuities": frame_discontinuities,
        "payload": dict(payload),
        "no_update": dict(no_update),
        "host": dict(host),
        "process_records": processes,
        "owned_ports": ports,
        "qport_identities": qports,
        "trajectory_sha256": raw["trajectory_sha256"],
        "started_at_unix_ns": started_at,
        "completed_at_unix_ns": completed_at,
    }


def _validate_fault_probe(
    raw: Mapping[str, Any], *, expected_bindings: Mapping[str, Any],
    expected_sources: Mapping[str, Any], expected_host: Mapping[str, Any],
    execution_path: Path,
) -> dict[str, Any]:
    _verify_simple_seal(raw, "evidence_sha256", "B6 fault probe")
    _exact_keys(raw, {
        "schema", "tool", "synthetic", "host", "source_repositories",
        "bindings", "full_network_execution", "scenarios", "evidence_sha256",
    }, "B6 fault probe")
    _require(raw.get("schema") == FAULT_SCHEMA
             and raw.get("tool") == "assemble_b6_g1_fault_probe"
             and raw.get("synthetic") is False,
             "fault probe is synthetic or wrong schema")
    host = _mapping(raw.get("host"), "fault probe host")
    _require(host.get("hostname") == "DESKTOP-RTX2080"
             and "microsoft-standard-WSL2" in str(host.get("kernel_release")),
             "fault probe did not run on DESKTOP-RTX2080 WSL2")
    _require(dict(host) == dict(expected_host),
             "fault execution and soak machine identities differ")
    _require(raw.get("bindings") == expected_bindings,
             "fault probe runtime/Atlas/bundle/checkpoint/optimizer bindings differ")
    _require(raw.get("source_repositories") == expected_sources,
             "fault probe source repositories differ")
    execution = _mapping(raw.get("full_network_execution"), "fault execution")
    _exact_keys(execution, {"bytes", "sha256", "execution_evidence_sha256"},
                "fault execution")
    for name in execution:
        _integer(execution[name], name, minimum=1) if name == "bytes" else _digest(
            execution[name], name
        )
    execution_source = _regular(execution_path, "fault full-network execution")
    execution_raw = load_json(
        execution_source, "fault full-network execution", newline=True
    )
    try:
        validated_execution = validate_network_execution_evidence(
            execution_raw, execution_source
        )
    except NetworkQualificationError as error:
        raise B6CampaignError(
            f"fault full-network execution rejected: {error}"
        ) from error
    _require(
        validated_execution.get("test_mode") is False
        and validated_execution.get("full_network_executed") is True
        and execution == {
            **file_record(execution_source),
            "execution_evidence_sha256": validated_execution[
                "execution_evidence_sha256"
            ],
        },
        "fault probe does not bind the supplied real full-network execution",
    )
    scenarios = raw.get("scenarios")
    _require(isinstance(scenarios, list) and len(scenarios) == 3,
             "fault probe requires exactly three scenarios")
    by_name = {
        item.get("name"): item for item in scenarios if isinstance(item, Mapping)
    }
    _require(set(by_name) == {
        "map-epoch-recovery", "whole-batch-telemetry-gap-recovery",
        "partial-client-timeout-fatal",
    }, "fault probe scenario set differs")
    execution_scenarios = {
        item.get("scenario"): {
            "bytes": item.get("size"),
            "sha256": item.get("sha256"),
            "evidence_sha256": item.get("raw_evidence_sha256"),
        }
        for item in validated_execution.get("scenario_evidence", [])
        if isinstance(item, Mapping)
    }
    expected_sources = {
        "map-epoch-recovery": execution_scenarios.get("epoch-drain"),
        "whole-batch-telemetry-gap-recovery": execution_scenarios.get(
            "whole-batch-telemetry-gap-recovery"
        ),
        "partial-client-timeout-fatal": execution_scenarios.get(
            "partial-client-telemetry-timeout"
        ),
    }
    _require(
        all(
            by_name[name].get("source") == expected_sources[name]
            for name in expected_sources
        ),
        "fault scenario summaries do not bind the validated raw scenarios",
    )
    epoch = _mapping(by_name["map-epoch-recovery"], "map-epoch fault scenario")
    _exact_keys(epoch, {
        "name", "source_scenario", "source", "observed_outcome", "epoch_drains",
        "new_epoch_bootstrap_frames", "actions_dispatched_during_epoch_drain",
        "boundary_rounds", "accepted_after_recovery", "recovered_map_epoch",
    }, "map-epoch fault scenario")
    _require(epoch.get("source_scenario") == "epoch-drain"
             and epoch.get("observed_outcome") == "completed"
             and epoch.get("epoch_drains") == 1
             and epoch.get("new_epoch_bootstrap_frames") == 1
             and epoch.get("actions_dispatched_during_epoch_drain") == 0
             and _integer(epoch.get("boundary_rounds"), "epoch boundaries", minimum=1) >= 1
             and epoch.get("accepted_after_recovery") == CLIENT_COUNT
             and type(epoch.get("recovered_map_epoch")) is int,
             "map-epoch recovery was not exercised successfully")
    gap = _mapping(by_name["whole-batch-telemetry-gap-recovery"], "gap scenario")
    _exact_keys(gap, {
        "name", "source_scenario", "source", "observed_outcome",
        "boundary_rounds", "accepted_synchronized_frames",
        "fault_event_count", "accepted_after_recovery",
        "frame_discontinuities", "timeout_client_ids", "relay_event_count",
        "relay_events_sha256", "transport_metrics",
    }, "gap scenario")
    gap_metrics = _mapping(gap.get("transport_metrics"), "gap transport metrics")
    discontinuities = gap.get("frame_discontinuities")
    _require(gap.get("source_scenario") == "whole-batch-telemetry-gap-recovery"
             and gap.get("observed_outcome") == "completed"
             and gap.get("boundary_rounds") == 1
             and gap.get("accepted_synchronized_frames") == 32
             and gap.get("fault_event_count") == 0
             and gap.get("accepted_after_recovery") == CLIENT_COUNT
             and gap.get("timeout_client_ids")
             == [f"qual-{slot:02d}" for slot in range(CLIENT_COUNT)]
             and gap.get("relay_event_count") == CLIENT_COUNT
             and _SHA.fullmatch(str(gap.get("relay_events_sha256", ""))) is not None
             and isinstance(discontinuities, list)
             and len(discontinuities) == 1
             and discontinuities[0].get("delta") == 2
             and gap_metrics.get("telemetry_gap_resyncs") == 1
             and gap_metrics.get("failed_rounds") == 0
             and gap_metrics.get("echo_timeouts") == 0,
             "whole-batch telemetry-gap recovery was not exercised successfully")
    partial = _mapping(by_name["partial-client-timeout-fatal"], "partial scenario")
    _exact_keys(partial, {
        "name", "source_scenario", "source", "observed_outcome",
        "fault_event_count", "exception", "accepted_synchronized_frames",
        "timeout_client_ids", "relay_event_count", "relay_events_sha256",
        "transport_metrics",
    }, "partial timeout scenario")
    partial_metrics = _mapping(
        partial.get("transport_metrics"), "partial transport metrics"
    )
    _require(partial.get("source_scenario") == "partial-client-telemetry-timeout"
             and partial.get("observed_outcome") == "fatal"
             and partial.get("fault_event_count") == 0
             and partial.get("exception") == "AuthoritativeEchoError"
             and partial.get("timeout_client_ids") == ["qual-00"]
             and partial.get("relay_event_count") == 1
             and _SHA.fullmatch(
                 str(partial.get("relay_events_sha256", ""))
             ) is not None
             and partial_metrics.get("telemetry_gap_resyncs") == 0
             and partial_metrics.get("failed_rounds") == 1
             and partial_metrics.get("echo_timeouts") == 1,
             "partial-client timeout was not proven fatal")
    for item in (epoch, gap, partial):
        source = _mapping(item.get("source"), "fault raw scenario binding")
        _exact_keys(source, {"bytes", "sha256", "evidence_sha256"},
                    "fault raw scenario binding")
        _integer(source["bytes"], "fault raw bytes", minimum=1)
        _digest(source["sha256"], "fault raw file digest")
        _digest(source["evidence_sha256"], "fault raw evidence digest")
    return {
        "map_epoch_recovery_exercised": True,
        "telemetry_gap_recovery_exercised": True,
        "partial_client_timeout_fatal": True,
    }


def _validate_performance(
    raw: Mapping[str, Any], atlas_sha256: str, expected_host: Mapping[str, Any]
) -> dict[str, Any]:
    _require(raw.get("schema") == "q2-multires-b2-gate-v1"
             and raw.get("batch") == "B2" and raw.get("status") == "green",
             "B2 WSL performance gate is not green")
    gate = _mapping(raw.get("gate"), "B2 gate")
    _require(gate.get("green") is True and gate.get("feature_assembly_budget_passed") is True
             and gate.get("atlas_budgets_passed") is True
             and gate.get("dyn_snapshot_budget_passed") is True,
             "B2 WSL performance predicates are not green")
    dyn = _mapping(raw.get("dyn_evidence"), "B2 Dyn evidence")
    _require(dyn.get("host") == "DESKTOP-RTX2080"
             and "microsoft-standard-WSL2" in str(dyn.get("kernel_release")),
             "performance evidence is not from DESKTOP-RTX2080 WSL2")
    _require(dyn.get("machine_identity_sha256")
             == expected_host.get("machine_identity_sha256"),
             "performance and soak machine identities differ")
    budget = _mapping(raw.get("representative_budgets"), "B2 representative budgets")
    _require(budget.get("atlas_sha256") == atlas_sha256,
             "performance Atlas differs from B6 Atlas")
    p99 = _integer(budget.get("four_client_feature_assembly_p99_ns"), "feature p99", minimum=1)
    atlas = _integer(budget.get("atlas_resident_bytes"), "Atlas resident bytes", minimum=1)
    dyn_bytes = _integer(budget.get("four_dyn_resident_bytes"), "Dyn resident bytes", minimum=1)
    rss = _integer(budget.get("atlas_build_peak_rss_bytes"), "Atlas build RSS", minimum=1)
    _require(p99 < MAX_FEATURE_P99_NS and atlas <= MAX_ATLAS_BYTES
             and dyn_bytes < MAX_DYN_BYTES and rss <= MAX_BUILD_RSS_BYTES,
             "WSL p99/RSS/resident payload budget failed")
    return {
        "four_client_feature_assembly_p99_ns": p99,
        "atlas_resident_bytes": atlas,
        "four_dyn_resident_bytes": dyn_bytes,
        "atlas_build_peak_rss_bytes": rss,
        "host": dyn["host"],
        "kernel_release": dyn["kernel_release"],
        "machine_identity_sha256": dyn["machine_identity_sha256"],
    }


def _public_host_authority() -> dict[str, Any]:
    try:
        return load_public_authority(PUBLIC_HOST_AUTHORITY_PATH.resolve())
    except PublicProbeError as error:
        raise B6CampaignError(
            f"B6 public host authority rejected: {error}"
        ) from error


def _validate_public_probe(
    raw: Mapping[str, Any], label: str, *, run_nonce: str, phase: str,
    predecessor_evidence_sha256: str,
) -> dict[str, Any]:
    try:
        payload = verify_probe(
            raw,
            authority_path=PUBLIC_HOST_AUTHORITY_PATH.resolve(),
            expected_campaign_id=ATTESTED_CAMPAIGN_ID,
            expected_run_nonce=run_nonce,
            expected_phase=phase,
            expected_predecessor_evidence_sha256=predecessor_evidence_sha256,
        )
    except PublicProbeError as error:
        raise B6CampaignError(f"{label} rejected: {error}") from error
    return {
        "captured_at_unix_ns": payload["captured_at_unix_ns"],
        "host": dict(payload["host"]),
        "state": dict(payload["state"]),
        "state_sha256": payload["state_sha256"],
        "evidence_sha256": raw["evidence_sha256"],
        "run_binding": dict(payload["run_binding"]),
    }


def _validate_controller_ledger(
    raw: Mapping[str, Any], *, plan_path: Path, one_run_path: Path,
    public_pre_path: Path, public_post_path: Path,
    expected_host: Mapping[str, Any], authority: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(raw, {
        "schema", "tool", "status", "campaign_id", "run_nonce", "launch_id",
        "controller_host", "controller_tool", "plan", "authority", "stages",
        "ordering_basis", "ledger_sha256",
    }, "B6 controller ledger")
    _require(
        raw.get("schema") == CONTROLLER_LEDGER_SCHEMA
        and raw.get("tool") == CONTROLLER_TOOL
        and raw.get("status") == "green"
        and raw.get("campaign_id") == ATTESTED_CAMPAIGN_ID
        and raw.get("ordering_basis") == "single-controller-monotonic-ns-v1",
        "B6 controller ledger identity differs",
    )
    supplied = _digest(raw.get("ledger_sha256"), "B6 controller ledger seal")
    unsigned = dict(raw)
    unsigned.pop("ledger_sha256")
    _require(supplied == canonical_sha256(unsigned), "B6 controller ledger seal differs")

    plan = load_controller_plan(_regular(plan_path, "B6 controller plan"))
    _require(raw.get("plan") == file_record(plan_path),
             "B6 controller plan record differs")
    _require(raw.get("authority") == file_record(PUBLIC_HOST_AUTHORITY_PATH),
             "B6 controller authority record differs")
    controller_path = ROOT / "tools/run_b6_attested_campaign.py"
    _require(raw.get("controller_tool") == file_record(controller_path),
             "B6 controller tool bytes differ")

    controller_host = _mapping(raw.get("controller_host"), "B6 controller host")
    _exact_keys(
        controller_host, {"hostname", "kernel_release", "architecture"},
        "B6 controller host",
    )
    _require(
        controller_host.get("hostname") == expected_host.get("hostname")
        and controller_host.get("kernel_release") == expected_host.get("kernel_release")
        and controller_host.get("architecture") == expected_host.get("architecture"),
        "B6 controller and one-run hosts differ",
    )

    run_nonce = raw.get("run_nonce")
    pre_raw = load_json(public_pre_path, "public pre-probe", newline=True)
    one_raw = load_json(one_run_path, "B6 one-run", newline=False)
    post_raw = load_json(public_post_path, "public post-probe", newline=True)
    expected_launch = launch_id_for(str(run_nonce), pre_raw.get("evidence_sha256", ""))
    _require(raw.get("launch_id") == expected_launch,
             "B6 controller launch ID does not derive from signed pre-probe")

    stages = _mapping(raw.get("stages"), "B6 controller stages")
    _exact_keys(stages, {"public_pre", "one_run", "public_post"},
                "B6 controller stages")
    expected_outputs = {
        "public_pre": (public_pre_path, pre_raw.get("evidence_sha256")),
        "one_run": (one_run_path, one_raw.get("evidence_sha256")),
        "public_post": (public_post_path, post_raw.get("evidence_sha256")),
    }
    expected_argv = {
        "public_pre": _remote_probe_argv(
            plan, run_nonce=str(run_nonce), phase="pre",
            predecessor_evidence_sha256=authority["authority_sha256"],
        ),
        "one_run": [
            *plan["one_run_argv"], "--launch_id", expected_launch,
            "--out", str(_regular(one_run_path, "B6 one-run")),
        ],
        "public_post": _remote_probe_argv(
            plan, run_nonce=str(run_nonce), phase="post",
            predecessor_evidence_sha256=str(one_raw.get("evidence_sha256", "")),
        ),
    }
    intervals: dict[str, tuple[int, int]] = {}
    for name in ("public_pre", "one_run", "public_post"):
        stage = _mapping(stages.get(name), f"B6 controller {name} stage")
        _exact_keys(stage, {
            "argv", "argv_sha256", "started_monotonic_ns",
            "completed_monotonic_ns", "returncode", "stdout", "stderr",
            "output", "evidence_sha256",
        }, f"B6 controller {name} stage")
        argv = stage.get("argv")
        _require(
            argv == expected_argv[name]
            and stage.get("argv_sha256")
            == hashlib.sha256(canonical_bytes(argv)).hexdigest()
            and stage.get("returncode") == 0,
            f"B6 controller {name} invocation differs",
        )
        started = _integer(
            stage.get("started_monotonic_ns"), f"{name} start", minimum=1
        )
        completed = _integer(
            stage.get("completed_monotonic_ns"), f"{name} completion", minimum=1
        )
        _require(started <= completed, f"B6 controller {name} interval differs")
        intervals[name] = (started, completed)
        for stream in ("stdout", "stderr"):
            record = _mapping(stage.get(stream), f"B6 controller {name} {stream}")
            _exact_keys(record, {"bytes", "sha256"}, f"{name} {stream}")
            _integer(record.get("bytes"), f"{name} {stream} bytes")
            _digest(record.get("sha256"), f"{name} {stream} digest")
        path, evidence_sha256 = expected_outputs[name]
        _require(
            stage.get("output") == file_record(path)
            and stage.get("evidence_sha256") == evidence_sha256,
            f"B6 controller {name} output differs",
        )
    _require(
        intervals["public_pre"][1] <= intervals["one_run"][0]
        <= intervals["one_run"][1] <= intervals["public_post"][0],
        "B6 controller monotonic stage order differs",
    )
    return {
        "run_nonce": str(run_nonce),
        "launch_id": expected_launch,
        "ledger_sha256": supplied,
    }



def assemble_campaign(
    *, repo_root: Path, one_run_path: Path, fault_probe_path: Path,
    fault_execution_path: Path,
    controller_ledger_path: Path, controller_plan_path: Path,
    public_pre_path: Path, public_post_path: Path, b2_gate_path: Path,
    b4_evidence_path: Path, b5_gate_path: Path, lineage_evidence_path: Path,
    retirement_evidence_path: Path, runtime_evidence_path: Path,
    runtime_manifest_path: Path,
    checkpoint_path: Path, bundle_manifest_path: Path,
    training_manifest_path: Path, objectives_path: Path, atlas_path: Path,
    atlas_manifest_path: Path, b3_gate_path: Path,
    retirement_manifest_path: Path,
) -> dict[str, Any]:
    public_authority = _public_host_authority()
    ledger_raw = load_json(
        controller_ledger_path, "B6 controller ledger", newline=True
    )
    _require(
        ledger_raw.get("schema") == CONTROLLER_LEDGER_SCHEMA
        and ledger_raw.get("campaign_id") == ATTESTED_CAMPAIGN_ID,
        "B6 controller ledger preflight identity differs",
    )
    _verify_simple_seal(ledger_raw, "ledger_sha256", "B6 controller ledger")
    pre_raw = load_json(public_pre_path, "public pre-probe", newline=True)
    run_nonce = str(ledger_raw.get("run_nonce"))
    expected_launch_id = launch_id_for(
        run_nonce, str(pre_raw.get("evidence_sha256", ""))
    )
    _require(ledger_raw.get("launch_id") == expected_launch_id,
             "B6 controller launch preflight differs")
    atlas_sha256 = file_record(atlas_path)["sha256"]
    load_json(objectives_path, "objectives")
    objective_identity_preflight = file_record(objectives_path)["sha256"]
    _digest(objective_identity_preflight, "objective identity preflight")
    b4 = _validate_b4(load_json(b4_evidence_path, "B4 evidence", newline=True), atlas_sha256)
    b5 = _validate_b5(
        load_json(b5_gate_path, "B5 gate", newline=False),
        objective_identity_sha256=objective_identity_preflight,
    )
    bindings = _validate_bindings(
        b4=b4, b5=b5, runtime_evidence_path=runtime_evidence_path,
        runtime_manifest_path=runtime_manifest_path,
        checkpoint_path=checkpoint_path, training_manifest_path=training_manifest_path,
        objectives_path=objectives_path, bundle_path=bundle_manifest_path,
        atlas_path=atlas_path, atlas_manifest_path=atlas_manifest_path,
        b3_gate_path=b3_gate_path,
        b2_gate_path=b2_gate_path, b4_evidence_path=b4_evidence_path,
        retirement_path=retirement_manifest_path, repo_root=repo_root,
    )
    runtime_sha = bindings["runtime_manifest_identity_sha256"]
    lineage = load_json(lineage_evidence_path, "lineage evidence")
    context = {"atlas_sha256": atlas_sha256, "runtime_manifest_sha256": runtime_sha}
    try:
        _check_lineage_attestation(lineage, context)
    except Exception as error:
        raise B6CampaignError(f"prior lineage evidence differs: {error}") from error
    retirement_evidence = load_json(retirement_evidence_path, "retirement evidence")
    try:
        _check_legacy_selector_deactivation(retirement_evidence, context)
    except Exception as error:
        raise B6CampaignError(f"prior retirement evidence differs: {error}") from error
    _require(retirement_evidence.get("retirement_manifest_sha256")
             == bindings["retirement_manifest"]["sha256"],
             "retirement evidence does not bind the exact manifest")

    one_run = load_json(one_run_path, "B6 one-run", newline=False)
    soak = _validate_one_run(
        one_run, bindings=bindings, atlas_sha256=atlas_sha256,
        runtime_manifest_sha256=runtime_sha,
        retirement_sha256=bindings["retirement_manifest"]["sha256"],
        expected_sources=b4["source_repositories"],
        expected_launch_id=expected_launch_id,
    )
    controller = _validate_controller_ledger(
        ledger_raw, plan_path=controller_plan_path, one_run_path=one_run_path,
        public_pre_path=public_pre_path, public_post_path=public_post_path,
        expected_host=soak["host"], authority=public_authority,
    )
    fault_bindings = {
        name: bindings[name] for name in (
            "runtime_manifest", "training_manifest", "objectives", "checkpoint",
            "bundle_manifest", "atlas", "atlas_manifest", "b3_gate",
        )
    } | {
        "runtime_manifest_identity_sha256": runtime_sha,
        "training_config_sha256": bindings["training_config_sha256"],
        "objective_identity_sha256": bindings["objective_identity_sha256"],
        "network_barrier_execution_evidence_sha256": bindings[
            "network_barrier_execution_evidence_sha256"
        ],
        "b3_gate_sha256": bindings["b3_gate_sha256"],
        "hook_necessity_runtime": bindings["hook_necessity_runtime"],
        "rust_extension": bindings["rust_extension"],
    }
    faults = _validate_fault_probe(
        load_json(fault_probe_path, "B6 fault probe", newline=True),
        expected_bindings=fault_bindings,
        expected_sources=b4["source_repositories"],
        expected_host=soak["host"],
        execution_path=fault_execution_path,
    )
    performance = _validate_performance(
        load_json(b2_gate_path, "B2 performance gate", newline=False),
        atlas_sha256, soak["host"],
    )
    pre = _validate_public_probe(
        pre_raw, "public pre-probe", run_nonce=controller["run_nonce"],
        phase="pre",
        predecessor_evidence_sha256=public_authority["authority_sha256"],
    )
    post_raw = load_json(public_post_path, "public post-probe", newline=True)
    post = _validate_public_probe(
        post_raw, "public post-probe", run_nonce=controller["run_nonce"],
        phase="post", predecessor_evidence_sha256=one_run["evidence_sha256"],
    )
    _require(pre["host"] == post["host"] and pre["state"] == post["state"],
             "public host/service/port/queue/map/credential state changed")
    _require(
        pre["host"].get("hostname") == public_authority["hostname"]
        and pre["host"].get("architecture") == public_authority["architecture"]
        and pre["host"].get("machine_identity_sha256")
        == public_authority["machine_identity_sha256"],
        "public probes do not match the pinned public host identity",
    )
    _require(
        pre["run_binding"]["capture_nonce"]
        != post["run_binding"]["capture_nonce"],
        "public pre/post probes reused a signed capture nonce",
    )

    result = {
        "schema": SCHEMA,
        "tool": TOOL,
        "status": "green",
        "host": soak["host"],
        "source_repositories": b4["source_repositories"],
        "bindings": {
            **bindings,
            "atlas_sha256": atlas_sha256,
            "b2_gate": file_record(b2_gate_path),
            "b4_evidence": file_record(b4_evidence_path),
            "b4_evidence_sha256": b4["evidence_sha256"],
            "b5_gate": file_record(b5_gate_path),
            "b5_gate_sha256": b5["gate_sha256"],
            "lineage_evidence": file_record(lineage_evidence_path),
            "retirement_evidence": file_record(retirement_evidence_path),
        },
        "raw_evidence": {
            "one_run": file_record(one_run_path),
            "one_run_evidence_sha256": one_run["evidence_sha256"],
            "fault_probe": file_record(fault_probe_path),
            "fault_probe_evidence_sha256": load_json(
                fault_probe_path, "B6 fault probe", newline=True
            )["evidence_sha256"],
            "fault_execution": file_record(fault_execution_path),
            "public_pre_probe": file_record(public_pre_path),
            "public_post_probe": file_record(public_post_path),
            "controller_ledger": file_record(controller_ledger_path),
            "controller_plan": file_record(controller_plan_path),
            "controller_ledger_sha256": controller["ledger_sha256"],
        },
        "g1": {
            key: soak[key] for key in (
                "accepted_transitions", "echo_attempts",
                "authoritative_echo_accept_rate", "vertical_samples",
                "vertical_matches", "vertical_match_rate", "water_samples",
                "land_samples", "water_projection_mismatches",
                "land_projection_mismatches", "water_land_projection_skew",
                "failed_rounds", "echo_timeouts", "declared_resyncs",
                "declared_resync_limit", "accepted_frame_discontinuities",
                "trajectory_sha256",
            )
        } | faults,
        "no_update": soak["no_update"],
        "performance": {**performance, "lan_payload": soak["payload"]},
        "public_state": {
            "host": pre["host"],
            "authority": {
                **file_record(PUBLIC_HOST_AUTHORITY_PATH),
                "authority_sha256": public_authority["authority_sha256"],
            },
            "pre_state_sha256": canonical_sha256(pre["state"]),
            "post_state_sha256": canonical_sha256(post["state"]),
            "run_nonce": controller["run_nonce"],
            "launch_id": controller["launch_id"],
            "attestation_key_id": public_authority["probe_attestation"]["key_id"],
            "pre_evidence_sha256": pre["evidence_sha256"],
            "post_evidence_sha256": post["evidence_sha256"],
            "controller_ledger_sha256": controller["ledger_sha256"],
            "ordering_basis": "signed-hash-chain-plus-controller-monotonic-ns-v1",
            "modified": False,
        },
        "teardown": {
            "process_records": soak["process_records"],
            "terminated_process_records": soak["process_records"],
            "orphan_processes_after_teardown": 0,
            "owned_ports": soak["owned_ports"],
            "qport_identities": soak["qport_identities"],
        },
        "gate_sha256": "",
    }
    result["gate_sha256"] = gate_sha256(result)
    validate_campaign(result)
    return result


def validate_campaign(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == SCHEMA and value.get("tool") == TOOL
             and value.get("status") == "green", "B6 campaign identity differs")
    _digest(value.get("gate_sha256"), "B6 campaign seal")
    _require(value["gate_sha256"] == gate_sha256(value), "B6 campaign seal differs")
    g1 = _mapping(value.get("g1"), "B6 G1")
    _require(g1.get("accepted_transitions") == REQUIRED_TRANSITIONS
             and _number(g1.get("authoritative_echo_accept_rate"), "echo rate") >= 0.97
             and _number(g1.get("vertical_match_rate"), "vertical rate") >= 0.99
             and g1.get("water_land_projection_skew") == 0
             and g1.get("failed_rounds") == 0 and g1.get("echo_timeouts") == 0
             and g1.get("map_epoch_recovery_exercised") is True
             and g1.get("telemetry_gap_recovery_exercised") is True
             and g1.get("partial_client_timeout_fatal") is True,
             "B6 G1 conclusion differs")


def publish(output: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(output.expanduser()))
    _require(not destination.exists() and not destination.is_symlink(),
             "B6 output must be a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise B6CampaignError("B6 output already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    for name in (
        "one_run", "fault_probe", "fault_execution", "controller_ledger",
        "controller_plan",
        "public_pre", "public_post", "b2_gate",
        "b4_evidence", "b5_gate", "lineage_evidence", "retirement_evidence",
        "runtime_evidence", "runtime_manifest", "checkpoint",
        "training_manifest", "objectives",
        "bundle_manifest", "atlas", "atlas_manifest", "b3_gate",
        "retirement_manifest", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = assemble_campaign(
            repo_root=args.repo_root, one_run_path=args.one_run,
            fault_probe_path=args.fault_probe,
            fault_execution_path=args.fault_execution,
            public_pre_path=args.public_pre,
            controller_ledger_path=args.controller_ledger,
            controller_plan_path=args.controller_plan,
            public_post_path=args.public_post, b2_gate_path=args.b2_gate,
            b4_evidence_path=args.b4_evidence, b5_gate_path=args.b5_gate,
            lineage_evidence_path=args.lineage_evidence,
            retirement_evidence_path=args.retirement_evidence,
            runtime_evidence_path=args.runtime_evidence,
            runtime_manifest_path=args.runtime_manifest,
            checkpoint_path=args.checkpoint,
            training_manifest_path=args.training_manifest,
            objectives_path=args.objectives,
            bundle_manifest_path=args.bundle_manifest, atlas_path=args.atlas,
            atlas_manifest_path=args.atlas_manifest, b3_gate_path=args.b3_gate,
            retirement_manifest_path=args.retirement_manifest,
        )
        publish(args.output, value)
    except (B6CampaignError, OSError, ValueError, TypeError) as error:
        print(f"B6 campaign refused: {error}", file=sys.stderr)
        return 1
    print(canonical_bytes(value).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
