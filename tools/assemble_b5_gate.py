#!/usr/bin/env python3
"""Assemble the fail-closed B5 pretraining gate from independent evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_multires_pretraining_validation import (  # noqa: E402
    DESIGN_RELATIVE,
    EXPECTED_DESIGN_SHA256,
    EXPECTED_PLAN_SHA256,
    INPUT_NAMES,
    PLAN_RELATIVE,
    REQUIRED_CAMPAIGN_MODES,
    REQUIRED_REPLICATES,
    SUITE_SCHEMA,
    TOOL_NAME as SUITE_TOOL_NAME,
    PretrainingValidationError,
    _commit,
    _digest,
    _exact_keys,
    _exclusive_write,
    _mapping,
    _regular_file,
    _require,
    canonical_bytes,
    canonical_sha256,
    file_record,
    git_identity,
    load_json,
    objective_identity,
    report_sha256,
    validate_b4_privilege_input,
    validate_campaign_evidence,
    validate_campaign_set,
    validate_proof,
    validate_runtime_input,
)


GATE_SCHEMA = "q2-multires-b5-gate-v1"
GATE_TOOL_NAME = "assemble_b5_gate"
B5_PREDICATE_KEYS = frozenset({
    "repository_clean",
    "exact_source_and_runtime_bound",
    "sealed_b4_public_privilege_evidence_bound",
    "active_authority_b3_b4_predecessor_chain_bound",
    "all_campaigns_deterministic",
    "guide_on_off_exercised",
    "hazard_hook_exercised",
    "posture_water_crouch_exercised",
    "aim_combat_holdout_exercised",
    "checkpoint_unchanged",
    "policy_state_unchanged",
    "optimizer_state_unchanged",
    "no_rate_reward_violations",
    "exact_500_transition_proof_passed",
})


class B5GateError(RuntimeError):
    """Raised when no green B5 gate can be assembled."""


def gate_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("gate_sha256", None)
    return canonical_sha256({"domain": "q2-multires-b5-gate-v1", "gate": body})


def _same_record(actual: Any, expected: Mapping[str, Any], label: str) -> None:
    record = _mapping(actual, label)
    _exact_keys(record, {"bytes", "sha256"}, label)
    _digest(record.get("sha256"), f"{label} digest")
    _require(record == expected, f"{label} differs from current bytes")


def assemble_gate(
    *, repo_root: Path, validation_report: Path, proof_report: Path,
    b3_gate: Path, b4_evidence: Path, runtime_manifest: Path, checkpoint: Path,
    training_manifest: Path, bundle_manifest: Path, objectives: Path,
    atlas: Path, output: Path,
) -> dict[str, Any]:
    try:
        repo = Path(os.path.abspath(repo_root.expanduser()))
        source = git_identity(repo)
        validation_path = _regular_file(validation_report, "validation report")
        proof_path = _regular_file(proof_report, "proof report")
        runner_path = _regular_file(
            repo / "tools/run_multires_pretraining_campaign.py",
            "repository campaign runner",
        )
        paths = {
            "b3_gate": _regular_file(b3_gate, "B3 predecessor gate"),
            "b4_evidence": _regular_file(b4_evidence, "B4 aggregate evidence"),
            "runtime_manifest": _regular_file(runtime_manifest, "runtime manifest"),
            "checkpoint": _regular_file(checkpoint, "checkpoint"),
            "training_manifest": _regular_file(training_manifest, "training manifest"),
            "bundle_manifest": _regular_file(bundle_manifest, "bundle manifest"),
            "objectives": _regular_file(objectives, "objectives"),
            "atlas": _regular_file(atlas, "atlas"),
        }
        current_bindings = {name: file_record(path) for name, path in paths.items()}
        validation = dict(load_json(validation_path, "validation report"))
        _require(validation_path.read_bytes() == canonical_bytes(validation),
                 "validation report is not canonical JSON")
        _exact_keys(validation, {
            "schema", "tool", "status", "passed", "source",
            "normative_documents", "tools", "bindings", "seeds",
            "runtime_manifest_identity_sha256",
            "b4_privilege_admission",
            "campaign_transition_count", "campaign_schedule", "campaigns",
            "campaign_summary", "proof", "no_update", "failures",
            "report_sha256",
        }, "validation report")
        _require(validation["schema"] == SUITE_SCHEMA, "validation schema differs")
        _require(validation["tool"] == SUITE_TOOL_NAME, "validation tool differs")
        _require(validation["status"] == "passed" and validation["passed"] is True,
                 "validation report is not green")
        _require(validation["failures"] == [], "validation report contains failures")
        _digest(validation["report_sha256"], "validation report digest")
        _require(validation["report_sha256"] == report_sha256(validation),
                 "validation report body digest is forged or stale")

        reported_source = _mapping(validation["source"], "validation source")
        _exact_keys(reported_source, {"clean", "commit", "tree"}, "validation source")
        _require(reported_source == source, "validation source is not current clean Git")
        _commit(reported_source["commit"], "validation commit")
        _commit(reported_source["tree"], "validation tree")

        reported_bindings = _mapping(validation["bindings"], "validation bindings")
        _exact_keys(reported_bindings, set(INPUT_NAMES), "validation bindings")
        for name in INPUT_NAMES:
            _same_record(reported_bindings[name], current_bindings[name],
                         f"validation {name}")
        runtime_manifest_sha256 = validate_runtime_input(
            paths["runtime_manifest"], current_bindings["atlas"]["sha256"]
        )
        _digest(
            validation["runtime_manifest_identity_sha256"],
            "validation runtime manifest identity",
        )
        _require(
            validation["runtime_manifest_identity_sha256"] == runtime_manifest_sha256,
            "validation runtime manifest identity differs",
        )
        current_b4_privilege = validate_b4_privilege_input(
            paths["b4_evidence"],
            b3_gate_path=paths["b3_gate"],
            atlas_sha256=current_bindings["atlas"]["sha256"],
            runtime_manifest_sha256=runtime_manifest_sha256,
            source=source,
        )
        _require(
            validation["b4_privilege_admission"] == current_b4_privilege,
            "validation B4 privilege admission differs",
        )

        normative = _mapping(validation["normative_documents"], "normative documents")
        _exact_keys(normative, {"design", "plan"}, "normative documents")
        current_normative = {
            "design": file_record(_regular_file(repo / DESIGN_RELATIVE, "design")),
            "plan": file_record(_regular_file(repo / PLAN_RELATIVE, "plan")),
        }
        _require(current_normative["design"]["sha256"] == EXPECTED_DESIGN_SHA256,
                 "design is not the authoritative revision")
        _require(current_normative["plan"]["sha256"] == EXPECTED_PLAN_SHA256,
                 "plan is not the authoritative revision")
        for name in ("design", "plan"):
            _same_record(normative[name], current_normative[name], f"normative {name}")

        tools = _mapping(validation["tools"], "validation tools")
        _exact_keys(
            tools, {"suite", "campaign_runner", "proof", "one_run"},
            "validation tools",
        )
        current_tools = {
            "suite": file_record(_regular_file(
                repo / "tools/run_multires_pretraining_validation.py", "suite tool"
            )),
            "campaign_runner": file_record(runner_path),
            "proof": file_record(_regular_file(
                repo / "tools/run_multires_500_transition_proof.py", "proof tool"
            )),
            "one_run": file_record(_regular_file(
                repo / "train/multires_one_run.py", "one-run trainer tool"
            )),
        }
        for name in current_tools:
            _same_record(tools[name], current_tools[name], f"tool {name}")

        seeds = _mapping(validation["seeds"], "validation seeds")
        _exact_keys(seeds, {"seed", "game_seed", "divergence_game_seed"},
                    "validation seeds")
        _require(
            all(isinstance(seeds[key], int) and not isinstance(seeds[key], bool)
                and seeds[key] >= 0 for key in seeds),
            "validation seeds are invalid",
        )
        _require(seeds["divergence_game_seed"] == seeds["game_seed"] + 1,
                 "validation divergence seed differs")
        transition_count = validation["campaign_transition_count"]
        _require(isinstance(transition_count, int) and not isinstance(transition_count, bool)
                 and transition_count > 0, "campaign transition count is invalid")
        expected_schedule = [
            {"campaign": mode, "replicate": replicate}
            for mode in REQUIRED_CAMPAIGN_MODES for replicate in REQUIRED_REPLICATES
        ]
        _require(validation["campaign_schedule"] == expected_schedule,
                 "campaign schedule differs")

        source_pair = {"commit": source["commit"], "tree": source["tree"]}
        raw_campaigns = validation["campaigns"]
        _require(isinstance(raw_campaigns, list), "campaigns must be a list")
        campaigns = []
        for row in raw_campaigns:
            item = _mapping(row, "campaign evidence")
            mode = item.get("campaign")
            replicate = item.get("replicate")
            _require(mode in REQUIRED_CAMPAIGN_MODES and replicate in REQUIRED_REPLICATES,
                     "campaign identity is outside the required schedule")
            campaigns.append(validate_campaign_evidence(
                item, mode=mode, replicate=replicate, seed=seeds["seed"],
                game_seed=seeds["game_seed"], transition_count=transition_count,
                source=source_pair, bindings=current_bindings,
                runtime_manifest_sha256=runtime_manifest_sha256,
            ))
        summary = validate_campaign_set(campaigns)
        _require(validation["campaign_summary"] == summary,
                 "campaign summary is forged or stale")

        no_update = _mapping(validation["no_update"], "no-update attestation")
        _exact_keys(no_update, {
            "checkpoint_unchanged", "policy_state_unchanged",
            "optimizer_state_unchanged", "optimizer_steps",
            "backward_parameter_gradients",
        }, "no-update attestation")
        _require(no_update == {
            "checkpoint_unchanged": True, "policy_state_unchanged": True,
            "optimizer_state_unchanged": True,
            "optimizer_steps": 0, "backward_parameter_gradients": 0,
        }, "no-update attestation failed")

        proof_record = file_record(proof_path)
        proof_section = _mapping(validation["proof"], "validation proof")
        _exact_keys(proof_section, {
            "record", "schema", "tool", "transition_count", "same_seed_match",
            "different_seed_diverges", "verifier_evidence_sha256",
        }, "validation proof")
        _same_record(proof_section["record"], proof_record, "proof report")
        raw_proof = load_json(proof_path, "500-transition proof")
        _require(proof_path.read_bytes() == canonical_bytes(raw_proof)[:-1],
                 "500-transition proof is not canonical JSON")
        proof = validate_proof(
            raw_proof, seed=seeds["seed"],
            game_seed=seeds["game_seed"], bindings=current_bindings,
            runtime_manifest_sha256=runtime_manifest_sha256,
            objective_identity_sha256=objective_identity(paths["objectives"]),
        )
        expected_proof_section = {
            "record": proof_record,
            "schema": proof["schema"], "tool": proof["tool"],
            "transition_count": proof["transition_count"],
            "same_seed_match": proof["same_seed_match"],
            "different_seed_diverges": proof["different_seed_diverges"],
            "verifier_evidence_sha256": canonical_sha256(proof["verifier_evidence"]),
        }
        _require(dict(proof_section) == expected_proof_section,
                 "validation proof summary is forged or stale")

        output_path = Path(os.path.abspath(output.expanduser()))
        _require(not output_path.is_relative_to(repo),
                 "gate output must be outside the clean source repository")
        _require(not output_path.exists(), "gate output already exists")
        gate: dict[str, Any] = {
            "schema": GATE_SCHEMA,
            "tool": GATE_TOOL_NAME,
            "status": "green",
            "green": True,
            "source": source,
            "normative_documents": current_normative,
            "runtime_bindings": current_bindings,
            "runtime_manifest_identity_sha256": runtime_manifest_sha256,
            "privilege": current_b4_privilege,
            "predecessor": current_b4_privilege["predecessor"],
            "evidence": {
                "validation_report": file_record(validation_path),
                "proof_report": proof_record,
            },
            "component_evidence": {
                "validation_report": validation,
                "proof_report": raw_proof,
            },
            "campaigns": summary,
            "proof": {
                "transition_count": proof["transition_count"],
                "same_seed_match": True,
                "different_seed_diverges": True,
                "production_pass": True,
            },
            "predicates": {
                "repository_clean": True,
                "exact_source_and_runtime_bound": True,
                "sealed_b4_public_privilege_evidence_bound": True,
                "active_authority_b3_b4_predecessor_chain_bound": True,
                "all_campaigns_deterministic": True,
                "guide_on_off_exercised": True,
                "hazard_hook_exercised": True,
                "posture_water_crouch_exercised": True,
                "aim_combat_holdout_exercised": True,
                "checkpoint_unchanged": True,
                "policy_state_unchanged": True,
                "optimizer_state_unchanged": True,
                "no_rate_reward_violations": True,
                "exact_500_transition_proof_passed": True,
            },
            "failures": [],
        }
        _require(
            set(gate["predicates"]) == B5_PREDICATE_KEYS
            and all(value is True for value in gate["predicates"].values()),
            "B5 producer predicate set drifted",
        )
        gate["gate_sha256"] = gate_sha256(gate)
        _exclusive_write(output_path, gate)
        return gate
    except PretrainingValidationError as error:
        raise B5GateError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--proof-report", type=Path, required=True)
    for option in INPUT_NAMES:
        parser.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assemble_gate(
            repo_root=args.repo_root, validation_report=args.validation_report,
            proof_report=args.proof_report, b3_gate=args.b3_gate,
            b4_evidence=args.b4_evidence,
            runtime_manifest=args.runtime_manifest,
            checkpoint=args.checkpoint,
            training_manifest=args.training_manifest,
            bundle_manifest=args.bundle_manifest, objectives=args.objectives,
            atlas=args.atlas,
            output=args.output,
        )
    except B5GateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
