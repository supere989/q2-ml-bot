#!/usr/bin/env python3
"""Run the deterministic, no-update B5 pretraining qualification campaign.

This tool is deliberately an evidence producer, not a trainer.  It launches
each required campaign twice from the same immutable inputs, launches the
repository's exact production 500-transition proof, and publishes a report
only after every result has been validated and all model/optimizer bytes are
unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.multires_contract import FEATURE_SCHEMA_SHA256, POLICY_GENERATION  # noqa: E402
from harness.multires_lineage import CHECKPOINT_FORMAT, LineageError  # noqa: E402
from harness.multires_runtime import validate_runtime_evidence  # noqa: E402
from tools.assemble_b4_evidence import (  # noqa: E402
    B4EvidenceError,
    DOCUMENT_NAMES as B4_DOCUMENT_NAMES,
    _canonical_bytes as b4_canonical_bytes,
    validate_b4_evidence,
)
from tools.assemble_b3_gate import B3GateError, validate_b3_gate  # noqa: E402

SUITE_SCHEMA = "q2-multires-pretraining-validation-v1"
CAMPAIGN_SCHEMA = "q2-multires-pretraining-campaign-v1"
PROOF_SCHEMA = "q2-multires-500-transition-proof-v1"
TOOL_NAME = "run_multires_pretraining_validation"
PROOF_TOOL_NAME = "run_multires_500_transition_proof"
REQUIRED_PROOF_TRANSITIONS = 500
REQUIRED_CAMPAIGN_MODES = (
    "guide_on",
    "guide_off",
    "hazard_hook",
    "posture_water_crouch",
    "aim_combat_holdout",
)
REQUIRED_REPLICATES = (0, 1)
DESIGN_RELATIVE = "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"
PLAN_RELATIVE = "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md"
EXPECTED_DESIGN_SHA256 = (
    "c55fc7ffc32bd0e88410b8493b46c179f3333f3806632ff8e6530f1c717508e6"
)
EXPECTED_PLAN_SHA256 = (
    "371577feb8c40f542c90eec4b4aa91ef84c4a8e2019bf1614e59c46aedfec410"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
INPUT_NAMES = (
    "b3_gate",
    "b4_evidence",
    "runtime_manifest",
    "checkpoint",
    "training_manifest",
    "bundle_manifest",
    "objectives",
    "atlas",
)
ZERO_COUNTERS = (
    "optimizer_steps",
    "backward_parameter_gradients",
)


class PretrainingValidationError(RuntimeError):
    """Raised when qualification evidence is missing, malformed, or mismatched."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PretrainingValidationError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys differ")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result >= minimum, f"{label} is invalid")
    return result


def _digest(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and HEX64.fullmatch(value) is not None
        and value != "0" * 64,
        f"{label} must be a non-placeholder SHA-256",
    )
    return value


def _commit(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and HEX40.fullmatch(value) is not None
        and value != "0" * 40,
        f"{label} must be a non-placeholder Git object id",
    )
    return value


def _regular_file(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    _require(absolute.is_absolute(), f"{label} must be absolute")
    _require(not absolute.is_symlink(), f"{label} may not be a symlink")
    _require(absolute.is_file(), f"{label} must be a regular file")
    return absolute


def _executable_file(path: Path, label: str) -> Path:
    absolute = _regular_file(path, label)
    _require(os.access(absolute, os.X_OK), f"{label} must be executable")
    return absolute


def _directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    _require(not absolute.is_symlink() and absolute.is_dir(),
             f"{label} must be an exact directory")
    return absolute


def file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    _require(bool(payload), f"{path} is empty")
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PretrainingValidationError(f"{label} is not valid JSON: {error}") from error
    return _mapping(value, label)


def validate_runtime_input(path: Path, atlas_sha256: str) -> str:
    """Return the sealed semantic runtime identity carried by B4 evidence."""
    try:
        validated = validate_runtime_evidence(
            load_json(path, "runtime evidence"),
            expected_atlas_sha256=atlas_sha256,
        )
    except LineageError as error:
        raise PretrainingValidationError(f"runtime evidence is inadmissible: {error}") from error
    return _digest(validated.runtime_manifest_sha256, "runtime manifest identity")


def objective_identity(path: Path) -> str:
    load_json(path, "objectives")
    return file_record(path)["sha256"]


def validate_b4_privilege_input(
    path: Path, *, b3_gate_path: Path, atlas_sha256: str, runtime_manifest_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize the complete sealed B4 evidence directory.

    B5's offline fixture rows are not network packets.  The public privilege
    conclusion therefore comes from B4's non-vacuous real-datagram audit and
    exact teacher-packet negative probe, not a synthetic zero counter.
    """
    def canonical_b4(value: Any, label: str) -> bytes:
        try:
            return b4_canonical_bytes(value) + b"\n"
        except B4EvidenceError as error:
            raise PretrainingValidationError(
                f"{label} is not canonical JSON: {error}"
            ) from error

    direct_b3_path = _regular_file(b3_gate_path, "B3 predecessor gate")
    direct_b3 = dict(load_json(direct_b3_path, "B3 predecessor gate"))
    _require(
        direct_b3_path.read_bytes() == canonical_bytes(direct_b3),
        "B3 predecessor gate is not canonical JSON",
    )
    try:
        direct_b3 = validate_b3_gate(direct_b3)
    except B3GateError as error:
        raise PretrainingValidationError(
            f"B3 predecessor gate is inadmissible: {error}"
        ) from error
    b3_repository = _mapping(direct_b3.get("repository"), "B3 repository")
    _require(
        b3_repository == {
            "repository_commit": source["commit"],
            "repository_tree": source["tree"],
            "git_clean": True,
        },
        "B3 predecessor source differs from B5 source",
    )

    aggregate_path = _regular_file(path, "B4 aggregate evidence")
    aggregate = dict(load_json(aggregate_path, "B4 aggregate evidence"))
    _require(
        aggregate_path.read_bytes() == canonical_b4(aggregate, "B4 aggregate evidence"),
        "B4 aggregate evidence is not canonical JSON",
    )
    documents: dict[str, Mapping[str, Any]] = {}
    document_records: dict[str, dict[str, Any]] = {}
    for name, filename in B4_DOCUMENT_NAMES.items():
        document_path = _regular_file(
            aggregate_path.parent / filename, f"B4 {name} evidence"
        )
        document = load_json(document_path, f"B4 {name} evidence")
        _require(
            document_path.read_bytes() == canonical_b4(document, f"B4 {name} evidence"),
            f"B4 {name} evidence is not canonical JSON",
        )
        documents[name] = document
        document_records[name] = file_record(document_path)
    try:
        validate_b4_evidence(aggregate, documents)
    except B4EvidenceError as error:
        raise PretrainingValidationError(
            f"B4 privilege evidence is inadmissible: {error}"
        ) from error
    _require(
        aggregate.get("schema") == "q2-multires-b4-evidence-v1"
        and aggregate.get("milestone") == "B4"
        and aggregate.get("status") == "green",
        "B4 aggregate is not a green B4 gate",
    )
    _require(aggregate.get("atlas_sha256") == atlas_sha256,
             "B4 aggregate Atlas binding differs")
    b4_predecessor = _mapping(
        aggregate.get("predecessor"), "B4 predecessor chain"
    )
    direct_record = file_record(direct_b3_path)
    b3_predecessor = _mapping(
        direct_b3.get("predecessor"), "B3 B2 predecessor"
    )
    b3_recovery = _mapping(
        direct_b3.get("recovery_guide"), "B3 recovery/guide closure"
    )
    expected_b4_predecessor = {
        "b3_gate": {
            "name": "B3-gate",
            "sha256": direct_record["sha256"],
            "size": direct_record["bytes"],
        },
        "b3_gate_sha256": direct_b3.get("gate_sha256"),
        "status": "green",
        "cohort_id": b3_predecessor.get("cohort_id"),
        "declaration_sha256": b3_predecessor.get("declaration_sha256"),
        "atlas_set_sha256": b3_recovery.get("atlas_set_sha256"),
        "repository_commit": source["commit"],
        "repository_tree": source["tree"],
    }
    _require(
        dict(b4_predecessor) == expected_b4_predecessor,
        "B4 predecessor does not bind the exact supplied B3 gate",
    )
    repositories = _mapping(
        aggregate.get("source_repositories"), "B4 source repositories"
    )
    bot_source = _mapping(repositories.get("bot"), "B4 bot source")
    _require(
        bot_source == {
            "clean": True, "commit": source["commit"], "tree": source["tree"]
        },
        "B4 aggregate bot source differs from B5 source",
    )
    runtime = _mapping(aggregate.get("runtime_manifest"), "B4 runtime manifest")
    _require(
        runtime.get("manifest_sha256") == runtime_manifest_sha256,
        "B4 aggregate runtime manifest identity differs",
    )
    gate = _mapping(aggregate.get("gate"), "B4 gate")
    _require(
        gate.get("public_teacher_violations_zero") is True,
        "B4 public privilege gate is not green",
    )
    proof = _mapping(
        aggregate.get("public_privilege_proof"), "B4 public privilege proof"
    )
    datagrams = _integer(proof.get("datagrams_seen"), "B4 public datagrams", minimum=1)
    public_packets = _integer(
        proof.get("public_packets_decoded"), "B4 decoded public packets", minimum=1
    )
    _require(
        proof.get("teacher_packets_detected") == 0
        and proof.get("negative_probe_result")
        == "fatal-public-privilege-violation",
        "B4 public privilege proof did not exclude teacher packets",
    )
    qualification = _mapping(
        aggregate.get("network_qualification"), "B4 network qualification"
    )
    return {
        "basis": "sealed-b4-real-public-datagram-audit-v1",
        "aggregate_evidence_sha256": _digest(
            aggregate.get("evidence_sha256"), "B4 aggregate evidence seal"
        ),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "network_qualification_evidence_sha256": _digest(
            qualification.get("evidence_sha256"),
            "B4 network qualification evidence",
        ),
        "datagrams_seen": datagrams,
        "public_packets_decoded": public_packets,
        "teacher_packets_detected": 0,
        "negative_probe_result": "fatal-public-privilege-violation",
        "documents": document_records,
        "predecessor": {
            "b3_gate": direct_record,
            "b3_gate_sha256": _digest(
                direct_b3.get("gate_sha256"), "B3 gate seal"
            ),
            "cohort_id": b3_predecessor.get("cohort_id"),
            "declaration_sha256": _digest(
                b3_predecessor.get("declaration_sha256"),
                "B3 predecessor declaration",
            ),
            "atlas_set_sha256": _digest(
                b3_recovery.get("atlas_set_sha256"), "B3 Atlas-set identity"
            ),
        },
    }


def git_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(repo_root.expanduser()))
    _require((root / ".git").exists(), "repo root is not a Git worktree")

    def capture(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments], check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PretrainingValidationError(f"Git identity check failed: {error}") from error
        _require(result.returncode == 0, "Git identity check failed")
        return result.stdout.decode("ascii", errors="strict").strip()

    status = capture("status", "--porcelain=v1", "--untracked-files=all")
    _require(status == "", "repository must be clean before qualification")
    commit = _commit(capture("rev-parse", "HEAD"), "repository commit")
    tree = _commit(capture("rev-parse", "HEAD^{tree}"), "repository tree")
    return {"clean": True, "commit": commit, "tree": tree}


def campaign_result_sha256(evidence: Mapping[str, Any]) -> str:
    body = dict(evidence)
    body.pop("result_sha256", None)
    return canonical_sha256(
        {"domain": "q2-multires-pretraining-campaign-result-v1", "evidence": body}
    )


def _validate_campaign_results(mode: str, value: Any) -> None:
    results = _mapping(value, f"{mode} results")
    if mode in {"guide_on", "guide_off"}:
        keys = {
            "scenario_identity_sha256", "guide_enabled", "task_attempts",
            "task_successes", "guide_nonzero_samples", "guide_dropout_samples",
        }
        _exact_keys(results, keys, f"{mode} results")
        _digest(results["scenario_identity_sha256"], "guide scenario identity")
        _require(results["guide_enabled"] is (mode == "guide_on"), "guide mode differs")
        attempts = _integer(results["task_attempts"], "task attempts", minimum=1)
        successes = _integer(results["task_successes"], "task successes")
        _require(successes <= attempts, "task successes exceed attempts")
        nonzero = _integer(results["guide_nonzero_samples"], "nonzero guide samples")
        _integer(results["guide_dropout_samples"], "guide dropout samples")
        _require(
            (nonzero > 0 if mode == "guide_on" else nonzero == 0),
            "guide samples do not match guide mode",
        )
    elif mode == "hazard_hook":
        keys = {
            "hazard_scenarios", "hook_scenarios", "safe_arrivals",
            "environmental_deaths", "valid_hook_attachments",
            "invalid_hook_attempts", "reward_replay_violations",
            "rate_reward_violations", "hook_necessity_label_violations",
        }
        _exact_keys(results, keys, f"{mode} results")
        _integer(results["hazard_scenarios"], "hazard scenarios", minimum=1)
        _integer(results["hook_scenarios"], "hook scenarios", minimum=1)
        for key in keys - {"hazard_scenarios", "hook_scenarios"}:
            _integer(results[key], key.replace("_", " "))
        for key in (
            "reward_replay_violations", "rate_reward_violations",
            "hook_necessity_label_violations",
        ):
            _require(results[key] == 0, f"{key} must be zero")
    elif mode == "posture_water_crouch":
        keys = {
            "posture_fixtures", "water_fixtures", "crouch_fixtures",
            "fixtures_passed", "vertical_echo_mismatches",
            "standing_blocked_mismatches",
        }
        _exact_keys(results, keys, f"{mode} results")
        total = sum(
            _integer(results[key], key.replace("_", " "), minimum=1)
            for key in ("posture_fixtures", "water_fixtures", "crouch_fixtures")
        )
        _require(_integer(results["fixtures_passed"], "fixtures passed") == total,
                 "not every posture/water/crouch fixture passed")
        for key in ("vertical_echo_mismatches", "standing_blocked_mismatches"):
            _require(_integer(results[key], key.replace("_", " ")) == 0,
                     f"{key} must be zero")
    elif mode == "aim_combat_holdout":
        keys = {
            "holdout_samples", "visible_contacts", "actionable_exposures",
            "permitted_fire", "executed_fire", "hits", "repeat_hits", "kills",
            "hidden_fire", "yaw_mae_degrees", "pitch_mae_degrees",
        }
        _exact_keys(results, keys, f"{mode} results")
        _integer(results["holdout_samples"], "holdout samples", minimum=1)
        _integer(results["visible_contacts"], "visible contacts", minimum=1)
        for key in keys - {"holdout_samples", "visible_contacts",
                           "yaw_mae_degrees", "pitch_mae_degrees"}:
            _integer(results[key], key.replace("_", " "))
        _require(results["hidden_fire"] == 0, "holdout contains hidden fire")
        _number(results["yaw_mae_degrees"], "yaw MAE")
        _number(results["pitch_mae_degrees"], "pitch MAE")
    else:
        raise PretrainingValidationError(f"unknown campaign mode: {mode}")


def _validate_season_report(
    value: Any, *, mode: str, transition_count: int, atlas_sha256: str,
    results: Mapping[str, Any],
) -> None:
    report = _mapping(value, f"{mode} season report")
    _require(report.get("schema") == "multires-atlas-season-v1",
             "campaign season schema differs")
    _require(report.get("season_id") == f"b5-{mode}",
             "campaign season identity differs")
    _require(report.get("atlas_sha256") == atlas_sha256,
             "campaign season Atlas differs")
    _require(
        report.get("policy_start_version") == 1
        and report.get("policy_end_version") == 1,
        "campaign season policy interval differs",
    )
    _require(report.get("accepted_transitions") == transition_count,
             "campaign season transition count differs")
    transport = _mapping(report.get("transport"), "campaign season transport")
    _require(
        transport.get("command_echo_match_rate") == 1.0
        and transport.get("state_resyncs") == 0,
        "campaign season admitted a transport failure",
    )
    privilege = _mapping(report.get("privilege"), "campaign privilege scope")
    _exact_keys(
        privilege, {"scope", "upstream_evidence_required"},
        "campaign privilege scope",
    )
    _require(
        privilege == {
            "scope": "not-measured-offline-no-public-conduit",
            "upstream_evidence_required":
                "sealed-b4-real-public-datagram-audit-v1",
        },
        "offline campaign made or omitted a public privilege measurement claim",
    )
    guides = _mapping(report.get("guides"), "campaign season guides")
    if mode in {"guide_on", "guide_off"}:
        global_drop_rate = _number(
            guides.get("global_drop_rate"), "campaign global guide dropout"
        )
        candidate_drop_rate = _number(
            guides.get("candidate_drop_rate"), "campaign candidate guide dropout"
        )
        _require(global_drop_rate <= 1.0 and candidate_drop_rate <= 1.0,
                 "campaign guide dropout rate exceeds one")
        observed_drops = round(candidate_drop_rate * transition_count * 4)
        _require(observed_drops == results["guide_dropout_samples"],
                 "guide dropout result is disconnected from season report")
        if mode == "guide_off":
            _require(global_drop_rate == 1.0 and candidate_drop_rate == 1.0,
                     "guide-off season is not a complete ablation")
    if mode == "hazard_hook":
        hazard = _mapping(report.get("hazard"), "campaign season hazard")
        hook = _mapping(report.get("hook"), "campaign season hook")
        _require(hazard.get("safe_arrivals") == results["safe_arrivals"],
                 "hazard result is disconnected from season report")
        _require(
            hazard.get("environmental_deaths") == results["environmental_deaths"],
            "environmental death result is disconnected from season report",
        )
        _require(hook.get("invalid_attempts") == results["invalid_hook_attempts"],
                 "invalid-hook result is disconnected from season report")
    if mode == "aim_combat_holdout":
        combat = _mapping(report.get("combat"), "campaign season combat")
        links = {
            "actionable_exposures": "actionable_exposure",
            "permitted_fire": "fire_permission",
            "executed_fire": "executed_fire",
            "hits": "hits",
            "repeat_hits": "repeated_hits",
            "kills": "kills",
            "hidden_fire": "hidden_fire",
            "yaw_mae_degrees": "visible_contact_yaw_mae_deg",
            "pitch_mae_degrees": "visible_contact_pitch_mae_deg",
        }
        for result_name, report_name in links.items():
            _require(results[result_name] == combat.get(report_name),
                     f"combat result {result_name} is disconnected from season report")


def validate_campaign_evidence(
    value: Any, *, mode: str, replicate: int, seed: int, game_seed: int,
    transition_count: int, source: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]], runtime_manifest_sha256: str,
) -> dict[str, Any]:
    evidence = dict(_mapping(value, f"{mode}/{replicate} evidence"))
    _exact_keys(evidence, {
        "schema", "campaign", "replicate", "status", "no_update", "seed",
        "game_seed", "transition_count", "source", "bindings", "counters",
        "trajectory_sha256", "season_report", "season_report_sha256",
        "results", "result_sha256",
    }, f"{mode}/{replicate} evidence")
    _require(evidence["schema"] == CAMPAIGN_SCHEMA, "campaign schema differs")
    _require(evidence["campaign"] == mode, "campaign identity differs")
    _require(evidence["replicate"] == replicate, "campaign replicate differs")
    _require(evidence["status"] == "passed", "campaign did not pass")
    _require(evidence["no_update"] is True, "campaign was not no-update")
    _require(evidence["seed"] == seed and evidence["game_seed"] == game_seed,
             "campaign seeds differ")
    _require(evidence["transition_count"] == transition_count,
             "campaign transition count differs")

    actual_source = _mapping(evidence["source"], "campaign source")
    _exact_keys(actual_source, {"commit", "tree"}, "campaign source")
    _require(actual_source == {"commit": source["commit"], "tree": source["tree"]},
             "campaign source binding differs")

    actual_bindings = _mapping(evidence["bindings"], "campaign bindings")
    expected_binding_keys = {
        "runtime_manifest_sha256", "checkpoint_sha256_before",
        "checkpoint_sha256_after", "policy_state_sha256_before",
        "policy_state_sha256_after", "optimizer_state_sha256_before",
        "optimizer_state_sha256_after", "training_manifest_sha256",
        "bundle_manifest_sha256", "atlas_sha256", "objective_identity_sha256",
    }
    _exact_keys(actual_bindings, expected_binding_keys, "campaign bindings")
    expected_digests = {
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "checkpoint_sha256_before": bindings["checkpoint"]["sha256"],
        "checkpoint_sha256_after": bindings["checkpoint"]["sha256"],
        "training_manifest_sha256": bindings["training_manifest"]["sha256"],
        "bundle_manifest_sha256": bindings["bundle_manifest"]["sha256"],
        "atlas_sha256": bindings["atlas"]["sha256"],
        "objective_identity_sha256": bindings["objectives"]["sha256"],
    }
    for key, expected in expected_digests.items():
        _digest(actual_bindings.get(key), f"campaign {key}")
        _require(actual_bindings[key] == expected, f"campaign {key} differs")
    for state_name in ("policy_state", "optimizer_state"):
        before = _digest(actual_bindings[f"{state_name}_sha256_before"],
                         f"campaign {state_name} before")
        after = _digest(actual_bindings[f"{state_name}_sha256_after"],
                        f"campaign {state_name} after")
        _require(before == after, f"campaign {state_name} changed")

    counters = _mapping(evidence["counters"], "campaign counters")
    _exact_keys(counters, set(ZERO_COUNTERS), "campaign counters")
    for key in ZERO_COUNTERS:
        _require(_integer(counters[key], key.replace("_", " ")) == 0,
                 f"campaign {key} must be zero")
    _digest(evidence["trajectory_sha256"], "campaign trajectory")
    _validate_campaign_results(mode, evidence["results"])
    _validate_season_report(
        evidence["season_report"], mode=mode,
        transition_count=transition_count,
        atlas_sha256=bindings["atlas"]["sha256"],
        results=_mapping(evidence["results"], "campaign results"),
    )
    _digest(evidence["season_report_sha256"], "campaign season report")
    _require(
        evidence["season_report_sha256"]
        == canonical_sha256(evidence["season_report"]),
        "campaign season report digest is forged or stale",
    )
    _digest(evidence["result_sha256"], "campaign result")
    _require(evidence["result_sha256"] == campaign_result_sha256(evidence),
             "campaign result digest is forged or stale")
    return evidence


def validate_campaign_set(campaigns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(mode, replicate) for mode in REQUIRED_CAMPAIGN_MODES
                for replicate in REQUIRED_REPLICATES}
    indexed = {(row["campaign"], row["replicate"]): row for row in campaigns}
    _require(len(indexed) == len(campaigns) and set(indexed) == expected,
             "campaign membership differs")
    for mode in REQUIRED_CAMPAIGN_MODES:
        first, second = indexed[(mode, 0)], indexed[(mode, 1)]
        _require(first["trajectory_sha256"] == second["trajectory_sha256"],
                 f"{mode} trajectory is nondeterministic")
        _require(first["results"] == second["results"],
                 f"{mode} results are nondeterministic")
        _require(first["season_report"] == second["season_report"],
                 f"{mode} season report is nondeterministic")
        for state_name in ("policy_state", "optimizer_state"):
            _require(
                first["bindings"][f"{state_name}_sha256_before"]
                == second["bindings"][f"{state_name}_sha256_before"],
                f"{mode} loaded {state_name} identity differs",
            )
    for state_name in ("policy_state", "optimizer_state"):
        identities = {
            row["bindings"][f"{state_name}_sha256_before"] for row in campaigns
        }
        _require(len(identities) == 1,
                 f"campaigns did not load one exact {state_name}")
    guide_on = indexed[("guide_on", 0)]
    guide_off = indexed[("guide_off", 0)]
    _require(
        guide_on["results"]["scenario_identity_sha256"]
        == guide_off["results"]["scenario_identity_sha256"],
        "guide on/off campaigns did not use the same scenario",
    )
    return {
        "campaign_count": len(campaigns),
        "deterministic_pairs": len(REQUIRED_CAMPAIGN_MODES),
        "guide_scenario_identity_sha256":
            guide_on["results"]["scenario_identity_sha256"],
    }


def validate_proof(
    value: Any, *, seed: int, game_seed: int,
    bindings: Mapping[str, Mapping[str, Any]], runtime_manifest_sha256: str,
    objective_identity_sha256: str,
) -> dict[str, Any]:
    proof = dict(_mapping(value, "500-transition proof"))
    _require(proof.get("schema") == PROOF_SCHEMA, "proof schema differs")
    _require(proof.get("tool") == PROOF_TOOL_NAME, "proof tool differs")
    _require(proof.get("mode") == "production", "proof is not production mode")
    _require(proof.get("admissible") is True and proof.get("production_pass") is True,
             "proof is not admissible/pass")
    _require(proof.get("transition_count") == REQUIRED_PROOF_TRANSITIONS,
             "proof transition count differs")
    _require(proof.get("required_client_count") == 4
             and proof.get("transitions_per_client") == 125,
             "proof client partition differs")
    _require(proof.get("seed") == seed and proof.get("game_seed") == game_seed,
             "proof seeds differ")
    _require(proof.get("divergence_game_seed") == game_seed + 1,
             "proof divergence seed differs")
    _require(proof.get("same_seed_match") is True
             and proof.get("different_seed_diverges") is True,
             "proof determinism/divergence predicate failed")
    for key in ("partial_admissions", "stale_admissions", "resync_admissions",
                "orphan_processes_after_teardown"):
        _require(proof.get(key) == 0, f"proof {key} must be zero")
    _require(proof.get("failures") == [], "proof contains failures")
    _require(proof.get("records_required") is True
             and proof.get("digest_only_admissible") is False,
             "proof did not require canonical transition records")
    _require(proof.get("policy_generation") == POLICY_GENERATION,
             "proof policy generation differs")
    _require(proof.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256,
             "proof feature schema differs")
    _require(proof.get("checkpoint_format") == CHECKPOINT_FORMAT,
             "proof checkpoint format differs")
    _require(proof.get("collector") == "MultiresSynchronousCollector",
             "proof collector differs")
    _require(proof.get("spatial_provider") == "RustAtlasSpatialProvider",
             "proof spatial provider differs")
    _require(proof.get("lattice_crate") == "q2_lattice",
             "proof lattice crate differs")
    _require(proof.get("runtime_manifest_sha256") == runtime_manifest_sha256,
             "proof runtime manifest differs")
    _require(proof.get("atlas_sha256") == bindings["atlas"]["sha256"],
             "proof atlas differs")
    admission = _mapping(proof.get("production_admission"), "proof admission")
    _require(admission.get("bundle_version") == 3
             and admission.get("artifact_state") == "admitted",
             "proof bundle is not admitted v3")
    _require(admission.get("checkpoint_sha256") == bindings["checkpoint"]["sha256"],
             "proof checkpoint differs")
    _require(admission.get("training_manifest_sha256")
             == bindings["training_manifest"]["sha256"],
             "proof training manifest differs")
    _require(
        admission.get("objective_identity_sha256") == objective_identity_sha256,
        "proof objective identity differs",
    )
    _require(admission.get("checkpoint_initialization") == "random"
             and admission.get("checkpoint_training_step") == 0,
             "proof checkpoint is not fresh random step zero")
    _digest(admission.get("checkpoint_lineage_root_sha256"),
            "proof checkpoint lineage root")
    for key in ("q2ded", "client_binary", "runtime_root", "evidence_dir"):
        _require(isinstance(admission.get(key), str)
                 and Path(admission[key]).is_absolute(),
                 f"proof admission {key} is not absolute")
    trainer = admission.get("trainer_argv_prefix")
    expected_trainer = [
        str(Path(sys.executable).resolve()),
        str((ROOT / "train/multires_one_run.py").resolve()),
    ]
    _require(trainer == expected_trainer,
             "proof did not execute the exact repository one-run trainer")
    _require(isinstance(proof.get("verifier_evidence"), Mapping),
             "proof verifier evidence is missing")
    return proof


def report_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("report_sha256", None)
    return canonical_sha256(
        {"domain": "q2-multires-pretraining-validation-report-v1", "report": body}
    )


def _exclusive_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise PretrainingValidationError(f"refusing to overwrite {path}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _run_command(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as error:
        raise PretrainingValidationError(
            f"could not launch command {argv[0]}: {error}"
        ) from error
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise PretrainingValidationError(
            f"command timed out after {timeout:g}s: {argv[0]}"
        ) from error
    return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)


@dataclass(frozen=True)
class ValidationConfig:
    repo_root: Path
    b3_gate: Path
    b4_evidence: Path
    runtime_manifest: Path
    checkpoint: Path
    training_manifest: Path
    bundle_manifest: Path
    objectives: Path
    atlas: Path
    q2ded: Path
    client_binary: Path
    runtime_root: Path
    seed: int
    game_seed: int
    campaign_transitions: int
    jobs: int
    timeout_seconds: float
    work_dir: Path
    output: Path


def run_validation(config: ValidationConfig) -> dict[str, Any]:
    _require(config.seed >= 0 and config.game_seed >= 0, "seeds must be nonnegative")
    _require(config.campaign_transitions > 0, "campaign transitions must be positive")
    _require(1 <= config.jobs <= 10, "jobs must be in [1, 10]")
    _require(math.isfinite(config.timeout_seconds) and config.timeout_seconds > 0,
             "timeout must be finite and positive")
    repo = Path(os.path.abspath(config.repo_root.expanduser()))
    source = git_identity(repo)
    _require(source.get("clean") is True, "source provider did not prove clean Git")
    _commit(source.get("commit"), "source commit")
    _commit(source.get("tree"), "source tree")

    paths = {
        "b3_gate": _regular_file(config.b3_gate, "B3 predecessor gate"),
        "b4_evidence": _regular_file(config.b4_evidence, "B4 aggregate evidence"),
        "runtime_manifest": _regular_file(config.runtime_manifest, "runtime manifest"),
        "checkpoint": _regular_file(config.checkpoint, "checkpoint"),
        "training_manifest": _regular_file(config.training_manifest, "training manifest"),
        "bundle_manifest": _regular_file(config.bundle_manifest, "bundle manifest"),
        "objectives": _regular_file(config.objectives, "objectives"),
        "atlas": _regular_file(config.atlas, "atlas"),
    }
    q2ded = _executable_file(config.q2ded, "q2ded")
    client_binary = _executable_file(config.client_binary, "client binary")
    runtime_root = _directory(config.runtime_root, "runtime root")
    bindings = {name: file_record(path) for name, path in paths.items()}
    runtime_manifest_sha256 = validate_runtime_input(
        paths["runtime_manifest"], bindings["atlas"]["sha256"]
    )
    b4_privilege_admission = validate_b4_privilege_input(
        paths["b4_evidence"], b3_gate_path=paths["b3_gate"],
        atlas_sha256=bindings["atlas"]["sha256"],
        runtime_manifest_sha256=runtime_manifest_sha256, source=source,
    )
    design = _regular_file(repo / DESIGN_RELATIVE, "normative design")
    plan = _regular_file(repo / PLAN_RELATIVE, "normative plan")
    normative = {"design": file_record(design), "plan": file_record(plan)}
    _require(normative["design"]["sha256"] == EXPECTED_DESIGN_SHA256,
             "normative design digest differs")
    _require(normative["plan"]["sha256"] == EXPECTED_PLAN_SHA256,
             "normative plan digest differs")
    campaign_runner = _regular_file(
        repo / "tools/run_multires_pretraining_campaign.py",
        "repository campaign runner",
    )
    _require(os.access(campaign_runner, os.X_OK), "campaign runner is not executable")
    proof_tool = _regular_file(repo / "tools/run_multires_500_transition_proof.py",
                               "500-transition proof tool")
    one_run_tool = _regular_file(
        repo / "train/multires_one_run.py", "one-run trainer tool"
    )
    suite_tool = _regular_file(Path(__file__), "validation suite tool")
    tools = {
        "suite": file_record(suite_tool),
        "campaign_runner": file_record(campaign_runner),
        "proof": file_record(proof_tool),
        "one_run": file_record(one_run_tool),
    }

    work = Path(os.path.abspath(config.work_dir.expanduser()))
    _require(not work.is_relative_to(repo),
             "work directory must be outside the clean source repository")
    _require(not work.exists(), "work directory already exists")
    work.mkdir(parents=True, mode=0o700)
    output = Path(os.path.abspath(config.output.expanduser()))
    _require(not output.is_relative_to(repo),
             "validation output must be outside the clean source repository")
    _require(not output.exists(), "output already exists")
    source_pair = {"commit": source["commit"], "tree": source["tree"]}

    def campaign_job(mode: str, replicate: int) -> dict[str, Any]:
        result_path = work / f"campaign-{mode}-{replicate}.json"
        argv = [
            str(campaign_runner),
            "--protocol", CAMPAIGN_SCHEMA,
            "--campaign", mode, "--replicate", str(replicate),
            "--seed", str(config.seed), "--game-seed", str(config.game_seed),
            "--transition-count", str(config.campaign_transitions),
            "--repo-commit", source["commit"], "--repo-tree", source["tree"],
            "--runtime-manifest", str(paths["runtime_manifest"]),
            "--checkpoint", str(paths["checkpoint"]),
            "--training-manifest", str(paths["training_manifest"]),
            "--bundle-manifest", str(paths["bundle_manifest"]),
            "--atlas-bin", str(paths["atlas"]), "--no-update",
            "--q2ded", str(q2ded),
            "--client-binary", str(client_binary),
            "--runtime-root", str(runtime_root),
            "--objectives", str(paths["objectives"]),
            "--output", str(result_path),
        ]
        result = _run_command(argv, config.timeout_seconds)
        _require(result.returncode == 0,
                 f"campaign {mode}/{replicate} exited {result.returncode}")
        _require(result_path.is_file() and not result_path.is_symlink(),
                 f"campaign {mode}/{replicate} did not publish evidence")
        raw_evidence = load_json(result_path, f"campaign {mode}/{replicate}")
        _require(result_path.read_bytes() == canonical_bytes(raw_evidence),
                 f"campaign {mode}/{replicate} evidence is not canonical JSON")
        return validate_campaign_evidence(
            raw_evidence, mode=mode,
            replicate=replicate, seed=config.seed, game_seed=config.game_seed,
            transition_count=config.campaign_transitions, source=source_pair,
            bindings=bindings, runtime_manifest_sha256=runtime_manifest_sha256,
        )

    campaigns: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.jobs) as executor:
        futures = {
            executor.submit(campaign_job, mode, replicate): (mode, replicate)
            for mode in REQUIRED_CAMPAIGN_MODES for replicate in REQUIRED_REPLICATES
        }
        for future in as_completed(futures):
            campaigns.append(future.result())
    campaigns.sort(key=lambda row: (
        REQUIRED_CAMPAIGN_MODES.index(row["campaign"]), row["replicate"]
    ))
    campaign_summary = validate_campaign_set(campaigns)

    proof_path = work / "proof-500.json"
    proof_argv = [
        sys.executable, str(proof_tool), "--mode", "production",
        "--seed", str(config.seed), "--game_seed", str(config.game_seed),
        "--divergence_game_seed", str(config.game_seed + 1),
        "--transition_count", str(REQUIRED_PROOF_TRANSITIONS),
        "--policy_version", "1", "--map_epoch", "1",
        "--q2ded", str(q2ded), "--client_binary", str(client_binary),
        "--runtime_root", str(runtime_root),
        "--bundle_manifest", str(paths["bundle_manifest"]),
        "--objectives", str(paths["objectives"]),
        "--atlas_bin", str(paths["atlas"]),
        "--checkpoint", str(paths["checkpoint"]),
        "--training_manifest", str(paths["training_manifest"]),
        "--runtime_evidence", str(paths["runtime_manifest"]),
        "--evidence_dir", str(work),
        "--trainer_executable", str(Path(sys.executable).resolve()),
        "--trainer_arg", str(one_run_tool),
        "--out", str(proof_path),
    ]
    proof_result = _run_command(proof_argv, config.timeout_seconds)
    _require(proof_result.returncode == 0,
             f"500-transition proof exited {proof_result.returncode}")
    _require(proof_path.is_file() and not proof_path.is_symlink(),
             "500-transition proof did not publish evidence")
    raw_proof = load_json(proof_path, "500-transition proof")
    _require(proof_path.read_bytes() == canonical_bytes(raw_proof)[:-1],
             "500-transition proof is not canonical JSON")
    proof = validate_proof(
        raw_proof, seed=config.seed,
        game_seed=config.game_seed, bindings=bindings,
        runtime_manifest_sha256=runtime_manifest_sha256,
        objective_identity_sha256=objective_identity(paths["objectives"]),
    )

    after = {name: file_record(path) for name, path in paths.items()}
    _require(after == bindings, "an immutable qualification input changed")
    _require(
        validate_b4_privilege_input(
            paths["b4_evidence"], b3_gate_path=paths["b3_gate"],
            atlas_sha256=bindings["atlas"]["sha256"],
            runtime_manifest_sha256=runtime_manifest_sha256, source=source,
        ) == b4_privilege_admission,
        "B4 privilege evidence changed during qualification",
    )
    proof_record = file_record(proof_path)
    aggregate_counters = {
        key: sum(int(row["counters"][key]) for row in campaigns)
        for key in ZERO_COUNTERS
    }
    no_update = {
        "checkpoint_unchanged": all(
            row["bindings"]["checkpoint_sha256_before"]
            == row["bindings"]["checkpoint_sha256_after"]
            for row in campaigns
        ),
        "policy_state_unchanged": all(
            row["bindings"]["policy_state_sha256_before"]
            == row["bindings"]["policy_state_sha256_after"]
            for row in campaigns
        ),
        "optimizer_state_unchanged": all(
            row["bindings"]["optimizer_state_sha256_before"]
            == row["bindings"]["optimizer_state_sha256_after"]
            for row in campaigns
        ),
        **aggregate_counters,
    }
    _require(
        all(value is True for key, value in no_update.items() if key.endswith("unchanged"))
        and all(value == 0 for key, value in no_update.items() if not key.endswith("unchanged")),
        "campaign set did not prove no-update execution",
    )
    report: dict[str, Any] = {
        "schema": SUITE_SCHEMA,
        "tool": TOOL_NAME,
        "status": "passed",
        "passed": True,
        "source": source,
        "normative_documents": normative,
        "tools": tools,
        "bindings": bindings,
        "runtime_manifest_identity_sha256": runtime_manifest_sha256,
        "b4_privilege_admission": b4_privilege_admission,
        "seeds": {
            "seed": config.seed, "game_seed": config.game_seed,
            "divergence_game_seed": config.game_seed + 1,
        },
        "campaign_transition_count": config.campaign_transitions,
        "campaign_schedule": [
            {"campaign": mode, "replicate": replicate}
            for mode in REQUIRED_CAMPAIGN_MODES for replicate in REQUIRED_REPLICATES
        ],
        "campaigns": campaigns,
        "campaign_summary": campaign_summary,
        "proof": {
            "record": proof_record,
            "schema": proof["schema"], "tool": proof["tool"],
            "transition_count": proof["transition_count"],
            "same_seed_match": proof["same_seed_match"],
            "different_seed_diverges": proof["different_seed_diverges"],
            "verifier_evidence_sha256": canonical_sha256(proof["verifier_evidence"]),
        },
        "no_update": no_update,
        "failures": [],
    }
    report["report_sha256"] = report_sha256(report)
    _exclusive_write(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    for option in INPUT_NAMES:
        parser.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--q2ded", type=Path, required=True)
    parser.add_argument("--client-binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--game-seed", type=int, required=True)
    parser.add_argument("--campaign-transitions", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_validation(ValidationConfig(
            repo_root=args.repo_root, b3_gate=args.b3_gate,
            b4_evidence=args.b4_evidence,
            runtime_manifest=args.runtime_manifest,
            checkpoint=args.checkpoint,
            training_manifest=args.training_manifest,
            bundle_manifest=args.bundle_manifest, objectives=args.objectives,
            atlas=args.atlas, q2ded=args.q2ded,
            client_binary=args.client_binary, runtime_root=args.runtime_root,
            seed=args.seed,
            game_seed=args.game_seed,
            campaign_transitions=args.campaign_transitions, jobs=args.jobs,
            timeout_seconds=args.timeout_seconds, work_dir=args.work_dir,
            output=args.output,
        ))
    except PretrainingValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
