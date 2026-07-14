from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import pytest

from harness.atlas_analyzer import (
    AtlasAnalysisError,
    _replay_hook_record_exact,
    _validate_hook_materialization_binding,
    _validate_materialized_oracle_identities,
)
from harness.hook_claims_v2 import (
    HookClaimsV2Error,
    canonical_json,
    render_runtime_sidecar,
    runtime_records_sha256,
    validate_materialization,
    validate_runtime_sidecar,
    validate_trace,
    validation_trace_sha256,
)


def _record(index: int = 0) -> dict:
    source = [index * 32_000, 0, 24_125]
    anchor = [source[0] + 16_000, 0, 200_000]
    eye = [source[0], source[1], source[2] + 22_000]
    return {
        "claim_id": f"hook:{index:04d}:candidate:0000",
        "source_milliunits": source,
        "anchor_milliunits": anchor,
        "landing_milliunits": [source[0] + 16_000, 0, 24_000],
        "release_after_ticks": 2,
        "distance_milliunits": round(math.sqrt(sum(
            (anchor[axis] - eye[axis]) ** 2 for axis in range(3)
        ))),
        "flags": 1,
    }


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
        anchor = [value / 1000.0 for value in self.record["anchor_milliunits"]]
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


def _replay(record: dict, *, expected_landing: bool = True):
    authority = _record()
    return _replay_hook_record_exact(
        _ExactCm(authority), _ExactPmove(authority), _ExactHook(), record,
        request_prefix="test-hook", atlas_origin=(0, 0, 0),
        expected_landing=expected_landing,
    )


def test_exact_replay_uses_first_grounded_fixed_origin_and_seals_trace() -> None:
    record = _record()
    measured, reason, trace = _replay(record)
    assert reason is None and measured == record and trace is not None
    assert trace["origin_fixed_frames"][-1] == [128, 0, 192]
    assert trace["first_grounded_frame_index"] == len(trace["origin_fixed_frames"]) - 1
    assert validate_trace(trace, "trace") == trace


def test_anchor_materializes_the_exact_cm_contact_and_distance() -> None:
    record = _record()

    class ContactCm(_ExactCm):
        def call(self, requests):
            output = super().call(requests)
            for request, response in zip(requests, output):
                if request["id"].endswith(":anchor"):
                    response["endpos"][0] -= 0.04346
                    response["endpos"][2] -= 0.03125
            return output

    measured, reason, trace = _replay_hook_record_exact(
        ContactCm(record), _ExactPmove(record), _ExactHook(), record,
        request_prefix="test-hook", atlas_origin=(0, 0, 0),
        expected_landing=True,
    )
    assert reason is None and measured is not None and trace is not None
    assert measured["anchor_milliunits"] == [15_957, 0, 199_969]
    eye = [
        measured["source_milliunits"][0], measured["source_milliunits"][1],
        measured["source_milliunits"][2] + 22_000,
    ]
    assert measured["distance_milliunits"] == round(math.sqrt(sum(
        (measured["anchor_milliunits"][axis] - eye[axis]) ** 2
        for axis in range(3)
    )))


def test_source_and_release_tick_tampering_reject() -> None:
    source_tamper = _record()
    source_tamper["source_milliunits"][0] += 125
    anchor = source_tamper["anchor_milliunits"]
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
    measured, reason, _ = _replay(other_cell, expected_landing=False)
    assert measured is None and reason == "measured_landing_outside_desired_l1"


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
    return {
        "schema": "q2-hook-claim-materialization-v2",
        "map": "fixture", "passed": True,
        "bsp": {"sha256": "1" * 64, "size_bytes": 1234},
        "candidates": {
            "meta_sha256": "2" * 64, "records_sha256": "3" * 64,
            "record_count": 42,
        },
        "source_projection_sha256": "4" * 64,
        "runtime_records_sha256": runtime_records_sha256(records),
        "selected_records": records,
        "validation_traces": traces,
        "oracles": {
            name: {
                "executable_sha256": str(index + 5) * 64,
                "tool_identity": str(index + 6) * 64,
                "physics_identity": str(index + 7) * 64,
                "requests": index + 1,
            }
            for index, name in enumerate(("collision", "pmove", "hook"))
        } | {"hook_parity_attestation_sha256": "8" * 64},
        "replay": {
            "analyzer": "q2-hook-claim-materializer",
            "analyzer_version": "b2-c-v2",
            "verifier": "q2-atlas-analyzer-exact-hook-replay",
            "verifier_version": "b2-a-v2",
        },
        "request_count": 6,
    }


def test_trace_runtime_rows_and_six_unique_geometry_tampering_reject() -> None:
    document = _materialization()
    assert validate_materialization(document) == document

    trace_tamper = deepcopy(document)
    trace_tamper["validation_traces"][0]["origin_fixed_frames"][0][0] += 1
    with pytest.raises(HookClaimsV2Error, match="sha256 differs"):
        validate_materialization(trace_tamper)

    fewer = deepcopy(document)
    fewer["selected_records"].pop()
    with pytest.raises(HookClaimsV2Error, match="exactly six"):
        validate_materialization(fewer)

    duplicate = deepcopy(document)
    duplicate["selected_records"][1]["anchor_milliunits"] = list(
        duplicate["selected_records"][0]["anchor_milliunits"]
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
    with pytest.raises(HookClaimsV2Error, match="runtime geometries are not unique"):
        validate_materialization(duplicate)

    payload = render_runtime_sidecar(
        "fixture", "1" * 64, "9" * 64, document["selected_records"]
    )
    validate_runtime_sidecar(
        payload, map_id="fixture", bsp_sha256="1" * 64,
        materialization_sha256="9" * 64, records=document["selected_records"],
    )
    with pytest.raises(HookClaimsV2Error, match="header/rows differ"):
        validate_runtime_sidecar(
            payload[:-2] + b"9\n", map_id="fixture", bsp_sha256="1" * 64,
            materialization_sha256="9" * 64, records=document["selected_records"],
        )


def _binding_fixture(tmp_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    document = _materialization()
    paths = {
        name: tmp_path / name
        for name in ("collision", "pmove", "hook", "parity")
    }
    for name, path in paths.items():
        path.write_bytes(f"{name} authority\n".encode())
    for name in ("collision", "pmove", "hook"):
        document["oracles"][name]["executable_sha256"] = hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
    document["oracles"]["hook_parity_attestation_sha256"] = hashlib.sha256(
        paths["parity"].read_bytes()
    ).hexdigest()
    validate_materialization(document)
    claims = {
        "schema": "q2-generator-claims-v2",
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
        hook_attestation=paths["parity"],
    )


def test_materialization_binds_canonical_bsp_meta_executables_and_parity(
    tmp_path: Path,
) -> None:
    document, claims, paths = _binding_fixture(tmp_path)
    assert _bind(document, claims, paths) == document

    changed = deepcopy(document)
    changed["request_count"] += 1
    with pytest.raises(AtlasAnalysisError, match="canonical identity"):
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

    paths["collision"].write_bytes(b"mutated collision authority\n")
    with pytest.raises(AtlasAnalysisError, match="collision executable identity"):
        _bind(document, claims, paths)


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
        for name in ("collision", "pmove", "hook")
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
