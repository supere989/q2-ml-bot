from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.atlas_analyzer import (
    AnalyzerLimits,
    AtlasAnalysisError,
    HookOracleProcess,
    OracleProcess,
    _oracle_atlas_origin,
    _replay_hook_record_exact,
    _validate_hook_materialization_binding,
    _validate_materialized_oracle_identities,
)
from harness.ibsp38 import parse_ibsp38
from harness.hook_claims_v4 import (
    HookClaimsV4Error,
    canonical_json,
    render_runtime_sidecar,
    runtime_records_sha256,
    selected_records_sha256,
    validate_candidates,
    validate_materialization,
    validate_runtime_sidecar,
    validate_trace,
    validation_trace_sha256,
    validation_traces_sha256,
)
from tools.materialize_hook_claims import (
    _measured_target_rejection,
    _publish_materialization_pair,
)


def _candidate(index: int = 0) -> dict:
    source = [index * 32_000, 0, 24_125]
    anchor = [source[0] + 16_000, 0, 200_000]
    eye = [source[0], source[1], source[2] + 22_000]
    return {
        "claim_id": f"hook:{index:04d}:candidate:0000",
        "source_milliunits": source,
        "trace_target_milliunits": anchor,
        "landing_milliunits": [source[0] + 16_000, 0, 24_000],
        "release_after_ticks": 2,
        "distance_milliunits": round(math.sqrt(sum(
            (anchor[axis] - eye[axis]) ** 2 for axis in range(3)
        ))),
        "flags": 1,
    }


def _record(index: int = 0) -> dict:
    candidate = _candidate(index)
    return candidate | {
        "measured_anchor_milliunits": list(candidate["trace_target_milliunits"]),
    }


def test_v4_candidate_schema_rejects_retired_v3_identity() -> None:
    value = {
        "schema": "q2-hook-claim-candidates-v4",
        "tick_msec": 100,
        "status": "unproven",
        "bundle_admissible": False,
        "records": [_candidate()],
    }
    assert validate_candidates(value) == value

    retired = deepcopy(value)
    retired["schema"] = "q2-hook-claim-candidates-v3"
    with pytest.raises(HookClaimsV4Error, match="not frozen v4"):
        validate_candidates(retired)


def _frame(origin_fixed: list[int], grounded: bool) -> dict:
    return {
        "origin_fixed": origin_fixed,
        "velocity_fixed": [0, 0, 0],
        "pm_type": 0,
        "pm_flags": 4 if grounded else 0,
        "pm_time": 0,
        "gravity": 800,
        "grounded": grounded,
    }


class _ExactCm:
    def __init__(self, record: dict):
        self.record = record

    def call(self, requests):
        output = []
        source = [value / 1000.0 for value in self.record["source_milliunits"]]
        anchor = [
            value / 1000.0
            for value in self.record.get(
                "measured_anchor_milliunits",
                self.record["trace_target_milliunits"],
            )
        ]
        for request in requests:
            if request["id"].endswith("source-clear"):
                output.append({
                    "startsolid": False, "allsolid": False, "fraction": 1.0,
                    "endpos": list(request["end"]),
                })
            elif request["id"].endswith("source-support"):
                support = list(source)
                support[2] -= 0.09375
                output.append({
                    "startsolid": False, "allsolid": False, "fraction": 0.01,
                    "endpos": support, "plane": {"normal": [0.0, 0.0, 1.0]},
                })
            else:
                output.append({
                    "startsolid": False, "allsolid": False, "fraction": 0.9,
                    "endpos": anchor, "surface": {"flags": 0}, "contents": 1,
                })
        return output


class _ExactHook:
    def call(self, request):
        if request["op"] == "touch":
            return {"attached": True, "action": "attach"}
        return {"full_velocity_overwrite": True, "velocity": [0.0, 0.0, 160.0]}


class _ExactPmove:
    def __init__(self, record: dict, required_release_ticks: int = 2):
        self.record = record
        self.required_release_ticks = required_release_ticks
        self.pull_count = 0
        self.identity = {"parameters": {"gravity": 800, "airaccelerate": 0.0}}

    def call(self, requests):
        request = requests[0]
        if request["id"].endswith(":source-ground"):
            assert request["gravity"] == self.identity["parameters"]["gravity"]
            assert request["airaccelerate"] == self.identity["parameters"]["airaccelerate"]
            fixed = [round(value * 8) for value in request["origin"]]
            frame = _frame(fixed, True)
            return [{"frames": [frame], "final": frame}]
        if ":pmove:" in request["id"]:
            self.pull_count += 1
            frame = _frame([self.pull_count * 8, 0, 200 + self.pull_count], False)
            return [{"frames": [frame], "final": frame}]
        landing = self.record["landing_milliunits"]
        if self.pull_count != self.required_release_ticks:
            landing = [0, 0, 24_000]
        landing_fixed = [value // 125 for value in landing]
        frames = [
            _frame([64, 0, 240], False),
            _frame(landing_fixed, True),
            _frame([999, 999, 999], True),
        ]
        return [{"frames": frames, "final": frames[-1]}]


def _replay(
    record: dict,
    *,
    expected_landing: bool = True,
    discover_landing: bool = False,
):
    authority = _record()
    return _replay_hook_record_exact(
        _ExactCm(authority), _ExactPmove(authority), _ExactHook(), record,
        request_prefix="test-hook", atlas_origin=(0, 0, 0),
        expected_landing=expected_landing,
        discover_landing=discover_landing,
    )


def test_exact_replay_uses_first_grounded_fixed_origin_and_seals_trace() -> None:
    record = _record()
    measured, reason, trace = _replay(record)
    assert reason is None and measured == record and trace is not None
    assert trace["origin_fixed_frames"][-1] == [128, 0, 192]
    assert trace["first_grounded_frame_index"] == len(trace["origin_fixed_frames"]) - 1
    assert validate_trace(trace, "trace") == trace


def test_discovery_preserves_trace_target_and_materializes_exact_cm_contact() -> None:
    candidate = _candidate()

    class ContactCm(_ExactCm):
        def call(self, requests):
            output = super().call(requests)
            for request, response in zip(requests, output):
                if request["id"].endswith(":anchor"):
                    response["endpos"][0] -= 0.04346
                    response["endpos"][2] -= 0.03125
            return output

    measured, reason, trace = _replay_hook_record_exact(
        ContactCm(candidate), _ExactPmove(_record()), _ExactHook(), candidate,
        request_prefix="test-hook", atlas_origin=(0, 0, 0),
        expected_landing=False, discover_landing=True,
    )
    assert reason is None and measured is not None and trace is not None
    assert measured["trace_target_milliunits"] == [16_000, 0, 200_000]
    assert measured["measured_anchor_milliunits"] == [15_957, 0, 199_969]
    eye = [
        measured["source_milliunits"][0], measured["source_milliunits"][1],
        measured["source_milliunits"][2] + 22_000,
    ]
    assert measured["distance_milliunits"] == round(math.sqrt(sum(
        (measured["measured_anchor_milliunits"][axis] - eye[axis]) ** 2
        for axis in range(3)
    )))


def test_source_and_release_tick_tampering_reject() -> None:
    source_tamper = _record()
    source_tamper["source_milliunits"][0] += 125
    anchor = source_tamper["measured_anchor_milliunits"]
    eye = [
        source_tamper["source_milliunits"][0], 0,
        source_tamper["source_milliunits"][2] + 22_000,
    ]
    source_tamper["distance_milliunits"] = round(math.sqrt(sum(
        (anchor[axis] - eye[axis]) ** 2 for axis in range(3)
    )))
    measured, reason, _ = _replay(source_tamper)
    assert measured is None and reason == "source_not_exactly_supported"

    release_tamper = _record()
    release_tamper["release_after_ticks"] = 3
    measured, reason, _ = _replay(release_tamper)
    assert measured is None and reason == "measured_landing_outside_desired_l1"


def test_source_requires_exact_cm_quantization_and_pmove_grounding() -> None:
    record = _record()

    class BadSupport(_ExactCm):
        def call(self, requests):
            output = super().call(requests)
            for request, response in zip(requests, output):
                if request["id"].endswith("source-support"):
                    response["endpos"][2] -= 0.125
            return output

    measured, reason, _ = _replay_hook_record_exact(
        BadSupport(record), _ExactPmove(record), _ExactHook(), record,
        request_prefix="test-hook", atlas_origin=(0, 0, 0),
        expected_landing=True,
    )
    assert measured is None and reason == "source_not_exactly_supported"

    class BadGround(_ExactPmove):
        def call(self, requests):
            if requests[0]["id"].endswith(":source-ground"):
                fixed = [round(value * 8) for value in requests[0]["origin"]]
                fixed[2] += 1
                frame = _frame(fixed, False)
                return [{"frames": [frame], "final": frame}]
            return super().call(requests)

    measured, reason, _ = _replay_hook_record_exact(
        _ExactCm(record), BadGround(record), _ExactHook(), record,
        request_prefix="test-hook", atlas_origin=(0, 0, 0),
        expected_landing=True,
    )
    assert measured is None and reason == "source_not_pmove_grounded"


def test_desired_l1_and_exact_measured_landing_tampering_reject() -> None:
    same_cell = _record()
    same_cell["landing_milliunits"][0] += 125
    measured, reason, _ = _replay(same_cell, expected_landing=True)
    assert measured is None and reason == "measured_landing_fixed_mismatch"

    other_cell = _record()
    other_cell["landing_milliunits"][0] += 16_000
    measured, reason, _ = _replay(other_cell, expected_landing=True)
    assert measured is None and reason == "measured_landing_outside_desired_l1"


def test_discovery_binds_measured_landing_then_exact_replay_remains_strict() -> None:
    proposal = _candidate()
    proposal["landing_milliunits"][0] += 16_000

    measured, reason, trace = _replay(
        proposal, expected_landing=False, discover_landing=True,
    )
    assert reason is None and measured is not None and trace is not None
    assert measured["landing_milliunits"] == _record()["landing_milliunits"]

    replayed, reason, replayed_trace = _replay(measured, expected_landing=True)
    assert reason is None and replayed == measured and replayed_trace == trace

    tampered = deepcopy(measured)
    tampered["landing_milliunits"][0] += 125
    replayed, reason, _ = _replay(tampered, expected_landing=True)
    assert replayed is None and reason == "measured_landing_fixed_mismatch"


def test_discovery_and_exact_landing_modes_are_mutually_exclusive() -> None:
    with pytest.raises(
        AtlasAnalysisError,
        match="exactly one of discovery or strict expectation",
    ):
        _replay(_record(), expected_landing=True, discover_landing=True)


def test_towers_71428103_preserved_trace_target_replays_exactly() -> None:
    """Regression for the 71428 collision-epsilon V3 failure."""

    root = Path(
        os.environ.get(
            "Q2_B2_71428_ARTIFACT_ROOT",
            "/home/raymondj/multires-artifacts/atlas-v1/B2/"
            "generated-final-71428-24780570/claims",
        )
    )
    map_id = "b2g26_towers_71428103"
    bsp = root / f"{map_id}.bsp"
    meta_path = root / f"{map_id}.meta.json"
    materialization_path = root / f"{map_id}.hook-materialization.json"
    cm_binary = Path(
        "/home/raymondj/multires-worktrees/integration/"
        "q2-ml-client/release/q2-cm-oracle"
    )
    pmove_binary = cm_binary.with_name("q2-pmove-oracle")
    hook_binary = Path(
        "/home/raymondj/multires-worktrees/integration/"
        "q2-lithium-3zb2/tools/q2-hook-oracle"
    )
    required = (bsp, meta_path, materialization_path, cm_binary, pmove_binary, hook_binary)
    if not all(path.is_file() for path in required):
        pytest.skip("71428 exact regression authorities are not installed")

    metadata = json.loads(meta_path.read_text())
    materialization = json.loads(materialization_path.read_text())
    claim_id = "hook:0001:candidate:0003"
    candidate_v3 = next(
        record for record in metadata["hook_claim_candidates_v3"]["records"]
        if record["claim_id"] == claim_id
    )
    selected_v3 = next(
        record for record in materialization["selected_records"]
        if record["claim_id"] == claim_id
    )
    selected_v4 = {
        "claim_id": claim_id,
        "source_milliunits": selected_v3["source_milliunits"],
        "trace_target_milliunits": candidate_v3["anchor_milliunits"],
        "measured_anchor_milliunits": selected_v3["anchor_milliunits"],
        "landing_milliunits": selected_v3["landing_milliunits"],
        "release_after_ticks": selected_v3["release_after_ticks"],
        "distance_milliunits": selected_v3["distance_milliunits"],
        "flags": selected_v3["flags"],
    }
    limits = AnalyzerLimits()
    with OracleProcess(cm_binary, bsp, "cm", limits) as cm:
        with OracleProcess(pmove_binary, bsp, "pmove", limits) as pmove:
            with HookOracleProcess(
                hook_binary, materialization["oracles"]["hook"]["physics_identity"],
                limits,
            ) as hook:
                origin = _oracle_atlas_origin(parse_ibsp38(bsp), cm.identity)
                measured, reason, trace = _replay_hook_record_exact(
                    cm, pmove, hook, selected_v4,
                    request_prefix="towers-71428103-v4",
                    atlas_origin=origin, expected_landing=True,
                )
                assert reason is None and measured == selected_v4 and trace is not None
                assert trace["first_grounded_frame_index"] == 13
                assert trace["sha256"] == (
                    "be28cf0116f16b10e07d07cd57a2f478"
                    "af1977c08355125eb1f87ba2c09dbb1f"
                )

                legacy_feedback = deepcopy(selected_v4)
                legacy_feedback["trace_target_milliunits"] = list(
                    legacy_feedback["measured_anchor_milliunits"]
                )
                measured, reason, _ = _replay_hook_record_exact(
                    cm, pmove, hook, legacy_feedback,
                    request_prefix="towers-71428103-v3-feedback",
                    atlas_origin=origin, expected_landing=True,
                )
                assert measured is None
                assert reason == "measured_anchor_fixed_mismatch"


def test_materializer_rejects_same_source_or_unsafe_measured_target() -> None:
    source = (1, 2, 3)
    clear = SimpleNamespace(
        supported=True, standing_clear=True, crouched_clear=False,
    )
    unsupported = SimpleNamespace(
        supported=False, standing_clear=True, crouched_clear=True,
    )
    blocked = SimpleNamespace(
        supported=True, standing_clear=False, crouched_clear=False,
    )

    assert _measured_target_rejection(source, source, clear) == (
        "measured_landing_shares_source_l1"
    )
    assert _measured_target_rejection(source, (4, 5, 6), None) == (
        "measured_landing_l1_unavailable"
    )
    assert _measured_target_rejection(source, (4, 5, 6), unsupported) == (
        "measured_landing_unsupported"
    )
    assert _measured_target_rejection(source, (4, 5, 6), blocked) == (
        "measured_landing_hull_blocked"
    )
    assert _measured_target_rejection(source, (4, 5, 6), clear) is None


def _materialization() -> dict:
    records = [_record(index) for index in range(6)]
    traces = []
    for record in records:
        fixed = [[value // 125 for value in record["landing_milliunits"]]]
        traces.append({
            "claim_id": record["claim_id"],
            "origin_fixed_frames": fixed,
            "first_grounded_frame_index": 0,
            "sha256": validation_trace_sha256(record["claim_id"], fixed, 0),
        })
    oracle_records = {
        name: {
            "executable_sha256": format(index + 5, "x") * 64,
            "tool_identity": format(index + 6, "x") * 64,
            "physics_identity": format(index + 7, "x") * 64,
            "requests": index + 1,
        }
        for index, name in enumerate(("collision", "pmove", "hook", "fall"))
    }
    seal = {
        "schema": "q2-b1-runtime-authority-seal-v1",
        "normative_documents": {
            "design_sha256": "a" * 64, "plan_sha256": "b" * 64,
        },
        "hook_parity_attestation_sha256": "c" * 64,
        "fixture_bsp_sha256": "d" * 64,
        "analysis_bsp_sha256": "1" * 64,
        "executables": {
            "cm_sha256": oracle_records["collision"]["executable_sha256"],
            "pmove_sha256": oracle_records["pmove"]["executable_sha256"],
            "hook_sha256": oracle_records["hook"]["executable_sha256"],
            "fall_sha256": oracle_records["fall"]["executable_sha256"],
        },
        "identities": {
            name: {
                field: oracle_records[name][field]
                for field in ("tool_identity", "physics_identity")
            }
            for name in ("collision", "pmove", "hook", "fall")
        },
    }
    return {
        "schema": "q2-hook-claim-materialization-v4",
        "map": "fixture", "passed": True,
        "landing_policy": "compiled-first-grounded-exact-v4",
        "bsp": {"sha256": "1" * 64, "size_bytes": 1234},
        "candidates": {
            "meta_sha256": "2" * 64, "records_sha256": "3" * 64,
            "record_count": 42,
        },
        "source_projection_sha256": "4" * 64,
        "runtime_records_sha256": runtime_records_sha256(records),
        "selected_records": records,
        "validation_traces": traces,
        "oracles": oracle_records | {
            "hook_parity_attestation_sha256": "c" * 64,
            "b1_runtime_authority_seal": seal,
        },
        "fresh_strict_replay": {
            "schema": "q2-hook-fresh-strict-replay-v4",
            "passed": True,
            "record_count": 6,
            "selected_records_sha256": selected_records_sha256(records),
            "validation_traces_sha256": validation_traces_sha256(traces),
            "oracles": {
                name: oracle_records[name] | {"requests": 1}
                for name in ("collision", "pmove", "hook")
            },
        },
        "replay": {
            "analyzer": "q2-hook-claim-materializer",
            "analyzer_version": "b2-c-v4",
            "verifier": "q2-atlas-analyzer-exact-hook-replay",
            "verifier_version": "b2-a-v4",
        },
        "request_count": 13,
    }


def test_trace_runtime_rows_and_six_unique_geometry_tampering_reject() -> None:
    document = _materialization()
    assert validate_materialization(document) == document

    retired = deepcopy(document)
    retired["schema"] = "q2-hook-claim-materialization-v3"
    with pytest.raises(HookClaimsV4Error, match="not a passed v4"):
        validate_materialization(retired)

    missing_fresh = deepcopy(document)
    del missing_fresh["fresh_strict_replay"]
    with pytest.raises(HookClaimsV4Error, match="keys differ"):
        validate_materialization(missing_fresh)

    strict_tamper = deepcopy(document)
    strict_tamper["fresh_strict_replay"]["validation_traces_sha256"] = "0" * 64
    with pytest.raises(HookClaimsV4Error, match="validation traces differ"):
        validate_materialization(strict_tamper)

    trace_tamper = deepcopy(document)
    trace_tamper["validation_traces"][0]["origin_fixed_frames"][0][0] += 1
    with pytest.raises(HookClaimsV4Error, match="sha256 differs"):
        validate_materialization(trace_tamper)

    fewer = deepcopy(document)
    fewer["selected_records"].pop()
    with pytest.raises(HookClaimsV4Error, match="exactly six"):
        validate_materialization(fewer)

    wrong_policy = deepcopy(document)
    wrong_policy["landing_policy"] = "generator-hint-v2"
    with pytest.raises(HookClaimsV4Error, match="landing policy differs"):
        validate_materialization(wrong_policy)

    duplicate = deepcopy(document)
    duplicate["selected_records"][1]["measured_anchor_milliunits"] = list(
        duplicate["selected_records"][0]["measured_anchor_milliunits"]
    )
    duplicate["selected_records"][1]["landing_milliunits"] = list(
        duplicate["selected_records"][0]["landing_milliunits"]
    )
    duplicate["selected_records"][1]["source_milliunits"] = list(
        duplicate["selected_records"][0]["source_milliunits"]
    )
    duplicate["selected_records"][1]["distance_milliunits"] = duplicate[
        "selected_records"
    ][0]["distance_milliunits"]
    duplicate["fresh_strict_replay"]["selected_records_sha256"] = (
        selected_records_sha256(duplicate["selected_records"])
    )
    with pytest.raises(HookClaimsV4Error, match="runtime geometries are not unique"):
        validate_materialization(duplicate)

    payload = render_runtime_sidecar(
        "fixture", "1" * 64, "9" * 64, document["selected_records"]
    )
    validate_runtime_sidecar(
        payload, map_id="fixture", bsp_sha256="1" * 64,
        materialization_sha256="9" * 64, records=document["selected_records"],
    )
    with pytest.raises(HookClaimsV4Error, match="header/rows differ"):
        validate_runtime_sidecar(
            payload[:-2] + b"9\n", map_id="fixture", bsp_sha256="1" * 64,
            materialization_sha256="9" * 64, records=document["selected_records"],
        )


def test_paired_publication_rolls_back_injected_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _materialization()
    attestation_bytes = canonical_json(document) + b"\n"
    attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
    source_projection = b"# source hook proposal\n# bundle_admissible: false\n"
    runtime_bytes = render_runtime_sidecar(
        "fixture", "1" * 64, attestation_sha256,
        document["selected_records"],
    )
    output_attestation = tmp_path / "fixture.hook-materialization.json"
    runtime_sidecar = tmp_path / "fixture.json"
    runtime_sidecar.write_bytes(source_projection)
    original_replace = os.replace

    def fail_runtime_publication(source, destination) -> None:
        if Path(destination) == runtime_sidecar:
            raise OSError("injected second-publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_runtime_publication)
    with pytest.raises(
        AtlasAnalysisError, match="failed and was rolled back",
    ):
        _publish_materialization_pair(
            output_attestation=output_attestation,
            runtime_sidecar=runtime_sidecar,
            attestation_bytes=attestation_bytes,
            runtime_bytes=runtime_bytes,
            source_projection=source_projection,
            source_projection_sha256=hashlib.sha256(source_projection).hexdigest(),
            map_id="fixture",
            bsp_sha256="1" * 64,
            records=document["selected_records"],
        )

    assert not output_attestation.exists()
    assert runtime_sidecar.read_bytes() == source_projection
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fixture.json"]


def test_paired_publication_is_cross_checked_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    document = _materialization()
    attestation_bytes = canonical_json(document) + b"\n"
    attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
    source_projection = b"# source hook proposal\n# bundle_admissible: false\n"
    runtime_bytes = render_runtime_sidecar(
        "fixture", "1" * 64, attestation_sha256,
        document["selected_records"],
    )
    output_attestation = tmp_path / "fixture.hook-materialization.json"
    runtime_sidecar = tmp_path / "fixture.json"
    runtime_sidecar.write_bytes(source_projection)
    arguments = {
        "output_attestation": output_attestation,
        "runtime_sidecar": runtime_sidecar,
        "attestation_bytes": attestation_bytes,
        "runtime_bytes": runtime_bytes,
        "source_projection": source_projection,
        "source_projection_sha256": hashlib.sha256(source_projection).hexdigest(),
        "map_id": "fixture",
        "bsp_sha256": "1" * 64,
        "records": document["selected_records"],
    }

    _publish_materialization_pair(**arguments)
    assert output_attestation.read_bytes() == attestation_bytes
    assert runtime_sidecar.read_bytes() == runtime_bytes
    with pytest.raises(AtlasAnalysisError, match="already exists"):
        _publish_materialization_pair(**arguments)


def _binding_fixture(tmp_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    document = _materialization()
    paths = {
        name: tmp_path / name
        for name in ("collision", "pmove", "hook", "fall", "parity")
    }
    for name, path in paths.items():
        path.write_bytes(f"{name} authority\n".encode())
    for name in ("collision", "pmove", "hook", "fall"):
        document["oracles"][name]["executable_sha256"] = hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
        if name != "fall":
            document["fresh_strict_replay"]["oracles"][name][
                "executable_sha256"
            ] = document["oracles"][name]["executable_sha256"]
        seal_name = "cm_sha256" if name == "collision" else f"{name}_sha256"
        document["oracles"]["b1_runtime_authority_seal"]["executables"][seal_name] = (
            document["oracles"][name]["executable_sha256"]
        )
    document["oracles"]["hook_parity_attestation_sha256"] = hashlib.sha256(
        paths["parity"].read_bytes()
    ).hexdigest()
    document["oracles"]["b1_runtime_authority_seal"][
        "hook_parity_attestation_sha256"
    ] = document["oracles"]["hook_parity_attestation_sha256"]
    validate_materialization(document)
    claims = {
        "schema": "q2-generator-claims-v3",
        "map": "fixture",
        "source_files": {
            "meta_sha256": document["candidates"]["meta_sha256"],
            "hook_materialization_sha256": hashlib.sha256(
                canonical_json(document) + b"\n"
            ).hexdigest(),
        },
        "hook_claims": document["selected_records"],
    }
    return document, claims, paths


def _bind(document: dict, claims: dict, paths: dict[str, Path]) -> dict:
    return _validate_hook_materialization_binding(
        document,
        canonical_map_id="fixture",
        generator_claims=claims,
        bsp_sha256="1" * 64,
        bsp_size=1234,
        cm_oracle=paths["collision"],
        pmove_oracle=paths["pmove"],
        hook_oracle=paths["hook"],
        fall_oracle=paths["fall"],
        hook_attestation=paths["parity"],
        b1_runtime_authority_seal=document["oracles"][
            "b1_runtime_authority_seal"
        ],
    )


def test_materialization_binds_canonical_bsp_meta_executables_and_parity(
    tmp_path: Path,
) -> None:
    document, claims, paths = _binding_fixture(tmp_path)
    assert _bind(document, claims, paths) == document

    changed = deepcopy(document)
    changed["request_count"] += 1
    with pytest.raises(AtlasAnalysisError, match="request count differs"):
        _bind(changed, claims, paths)

    changed_claims = deepcopy(claims)
    changed_claims["source_files"]["meta_sha256"] = "9" * 64
    with pytest.raises(AtlasAnalysisError, match="metadata identity"):
        _bind(document, changed_claims, paths)

    bsp_changed = deepcopy(document)
    bsp_changed["bsp"]["size_bytes"] += 1
    changed_claims = deepcopy(claims)
    changed_claims["source_files"]["hook_materialization_sha256"] = hashlib.sha256(
        canonical_json(bsp_changed) + b"\n"
    ).hexdigest()
    with pytest.raises(AtlasAnalysisError, match="BSP identity"):
        _bind(bsp_changed, changed_claims, paths)

    for name in ("collision", "pmove", "hook", "fall"):
        original = paths[name].read_bytes()
        paths[name].write_bytes(f"mutated {name} authority\n".encode())
        with pytest.raises(
            AtlasAnalysisError, match=f"{name} executable identity"
        ):
            _bind(document, claims, paths)
        paths[name].write_bytes(original)


def test_materialization_rejects_parity_and_live_tool_or_physics_mutation(
    tmp_path: Path,
) -> None:
    document, claims, paths = _binding_fixture(tmp_path)
    paths["parity"].write_bytes(b"mutated parity authority\n")
    with pytest.raises(AtlasAnalysisError, match="parity identity"):
        _bind(document, claims, paths)

    identities = {
        name: {
            "tool_identity": document["oracles"][name]["tool_identity"],
            "physics_identity": document["oracles"][name]["physics_identity"],
        }
        for name in ("collision", "pmove", "hook", "fall")
    }
    _validate_materialized_oracle_identities(document["oracles"], identities)
    for name in identities:
        for field in ("tool_identity", "physics_identity"):
            changed = deepcopy(identities)
            changed[name][field] = "0" * 64
            with pytest.raises(
                AtlasAnalysisError, match=f"{name} oracle identity differs"
            ):
                _validate_materialized_oracle_identities(
                    document["oracles"], changed
                )
