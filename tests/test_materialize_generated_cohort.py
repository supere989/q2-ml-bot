from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from tools import materialize_generated_cohort as cohort
from harness.atlas_b1_authority import load_b1_authority_gate
from tools.run_generator_cohort import (
    STAGE_SUFFIXES,
    canonical_bytes,
    verify_stage_membership,
)
from harness.hook_claims_v4 import (
    canonical_json as hook_canonical_json,
    render_runtime_sidecar,
    runtime_records_sha256,
    selected_records_sha256,
    validation_trace_sha256,
    validation_traces_sha256,
)


STYLES = (
    "open",
    "towers",
    "canyon",
    "pits",
    "arena_open",
    "arena_vertical",
    "arena_lanes",
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(index: int) -> dict:
    source = [index * 32_000, 0, 24_125]
    anchor = [source[0] + 16_000, 0, 200_000]
    eye = [source[0], source[1], source[2] + 22_000]
    return {
        "claim_id": f"hook:{index:04d}:candidate:0000",
        "source_milliunits": source,
        "trace_target_milliunits": anchor,
        "measured_anchor_milliunits": list(anchor),
        "landing_milliunits": [source[0] + 16_000, 0, 24_000],
        "release_after_ticks": 2,
        "distance_milliunits": round(math.sqrt(sum(
            (anchor[axis] - eye[axis]) ** 2 for axis in range(3)
        ))),
        "flags": 1,
    }


def _materialization_pair(
    map_id: str, bsp_payload: bytes, source_projection: bytes
) -> tuple[bytes, bytes]:
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
    executable_labels = {
        "collision": "cm",
        "pmove": "pmove",
        "hook": "hook",
        "fall": "fall",
    }
    oracle_records = {
        name: {
            "executable_sha256": cohort.EXPECTED_SHA256[label],
            "tool_identity": format(index + 1, "x") * 64,
            "physics_identity": format(index + 5, "x") * 64,
            "requests": index + 1,
        }
        for index, (name, label) in enumerate(executable_labels.items())
    }
    bsp_sha256 = _sha(bsp_payload)
    hook_attestation = cohort.EXPECTED_SHA256["hook_attestation"]
    seal = {
        "schema": "q2-b1-runtime-authority-seal-v1",
        "normative_documents": {
            "design_sha256": "a" * 64,
            "plan_sha256": "b" * 64,
        },
        "hook_parity_attestation_sha256": hook_attestation,
        "fixture_bsp_sha256": "d" * 64,
        "analysis_bsp_sha256": bsp_sha256,
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
            for name in oracle_records
        },
    }
    document = {
        "schema": "q2-hook-claim-materialization-v4",
        "map": map_id,
        "passed": True,
        "landing_policy": "compiled-first-grounded-exact-v4",
        "bsp": {"sha256": bsp_sha256, "size_bytes": len(bsp_payload)},
        "candidates": {
            "meta_sha256": "2" * 64,
            "records_sha256": "3" * 64,
            "record_count": 42,
        },
        "source_projection_sha256": _sha(source_projection),
        "runtime_records_sha256": runtime_records_sha256(records),
        "selected_records": records,
        "validation_traces": traces,
        "oracles": oracle_records | {
            "hook_parity_attestation_sha256": hook_attestation,
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
    attestation = hook_canonical_json(document) + b"\n"
    runtime = render_runtime_sidecar(
        map_id, bsp_sha256, _sha(attestation), records
    )
    return attestation, runtime


def _write_declaration(root: Path) -> tuple[Path, dict]:
    maps = []
    ordinal = 0
    for style_index, style in enumerate(STYLES):
        for member in range(4):
            seed = 90000000 + style_index * 100 + member
            maps.append({
                "ordinal": ordinal,
                # Reverse lexical order so a glob/sort implementation cannot
                # accidentally satisfy the declaration-order test.
                "map": f"future_{27 - ordinal:02d}_{style}",
                "seed": seed,
                "style": style,
                "grid": 5,
                "observed_heat": None,
            })
            ordinal += 1
    declaration = {
        "schema": "q2-b2-generated-cohort-declaration-v1",
        "cohort_id": "b2g26_future_90000",
        "mode": "final",
        "generator": {
            "version": "v6",
            "grid": 5,
            "observed_heat": None,
            "gym": False,
        },
        "selection": {
            "timing": "declared-before-generation",
            "policy": "all-or-nothing",
            "replacement_allowed": False,
            "salvage_allowed": False,
            "required_map_count": 28,
            "required_concrete_styles": list(STYLES),
            "required_maps_per_style": 4,
        },
        "implementation_binding": {
            "require_clean_git": True,
            "bind_repository_commit": True,
            "bind_repository_tree": True,
            "bind_atlas_analyzer_closure": True,
        },
        "source_suffixes": [
            ".map",
            ".json",
            ".meta.json",
            ".lattice.json",
            ".routes.json",
        ],
        "maps": maps,
    }
    path = root / "future-declaration.json"
    path.write_bytes(canonical_bytes(declaration))
    return path, declaration


def _write_compiled(root: Path, declaration: dict) -> Path:
    compiled = root / "compiled"
    compiled.mkdir()
    for row in declaration["maps"]:
        for suffix in STAGE_SUFFIXES["compiled"]:
            if suffix == ".json":
                payload = b"# bundle_admissible: false\n"
            else:
                payload = f"{row['ordinal']}:{row['map']}:{suffix}\n".encode()
            (compiled / f"{row['map']}{suffix}").write_bytes(payload)
    return compiled


def _write_authorities(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    authority_root = root / "authorities"
    authority_root.mkdir()
    paths = {}
    expected = {
        "b1_gate": cohort.file_sha256(
            cohort.ROOT / "docs/multires/B1-GATE.json"
        )
    }
    for label in ("cm", "pmove", "hook", "fall", "hook_attestation"):
        path = authority_root / label
        payload = f"future authority {label}\n".encode()
        path.write_bytes(payload)
        paths[label] = path
        expected[label] = _sha(payload)
    monkeypatch.setattr(cohort, "EXPECTED_SHA256", expected)
    monkeypatch.setattr(
        cohort, "_expected_authority_sha256", lambda _attestation: expected
    )
    return paths


def _paths(root: Path, compiled: Path, declaration: Path, authorities: dict):
    return {
        "declaration_path": declaration,
        "compiled_dir": compiled,
        "stage_dir": root / "materialized-stage-attempt-1",
        "materialized_dir": root / "materialized",
        "log_dir": root / "logs" / "materialize",
        "report_path": root / "reports" / "materialization.json",
        "cm_oracle": authorities["cm"],
        "pmove_oracle": authorities["pmove"],
        "hook_oracle": authorities["hook"],
        "fall_oracle": authorities["fall"],
        "hook_attestation": authorities["hook_attestation"],
    }


class FakeMaterializer:
    def __init__(
        self,
        *,
        fail_at: int | None = None,
        bad_digest_at: int | None = None,
        tamper_first_bsp_at: int | None = None,
        bad_source_binding_at: int | None = None,
        tamper_runtime_at: int | None = None,
    ):
        self.fail_at = fail_at
        self.bad_digest_at = bad_digest_at
        self.tamper_first_bsp_at = tamper_first_bsp_at
        self.bad_source_binding_at = bad_source_binding_at
        self.tamper_runtime_at = tamper_runtime_at
        self.maps: list[str] = []

    @staticmethod
    def _argument(command: list[str], name: str) -> Path:
        return Path(command[command.index(name) + 1])

    def __call__(self, command: list[str], **kwargs):
        assert kwargs["check"] is False
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["timeout"] == cohort.DEFAULT_MATERIALIZER_TIMEOUT_SECONDS
        bsp = self._argument(command, "--bsp")
        map_id = bsp.stem
        index = len(self.maps)
        self.maps.append(map_id)
        if self.fail_at == index:
            return subprocess.CompletedProcess(
                command, 1, b"", b"compiled hook preflight proved 5/6\n"
            )

        attestation = self._argument(command, "--output-attestation")
        runtime = self._argument(command, "--runtime-sidecar")
        source_projection = runtime.read_bytes()
        attestation_payload, runtime_payload = _materialization_pair(
            map_id,
            bsp.read_bytes(),
            (
                b"different source projection\n"
                if self.bad_source_binding_at == index
                else source_projection
            ),
        )
        attestation.write_bytes(attestation_payload)
        runtime.write_bytes(runtime_payload)
        if self.tamper_runtime_at == index:
            runtime_payload += b"tampered runtime row\n"
            runtime.write_bytes(runtime_payload)
        if self.tamper_first_bsp_at == index:
            first_map = self.maps[0]
            bsp.with_name(f"{first_map}.bsp").write_bytes(b"substituted BSP\n")
        attestation_sha256 = _sha(attestation_payload)
        if self.bad_digest_at == index:
            attestation_sha256 = "0" * 64
        result = {
            "schema": cohort.RESULT_SCHEMA,
            "map": map_id,
            "passed": True,
            "selected_count": 6,
            "attestation_sha256": attestation_sha256,
            "runtime_sidecar_sha256": _sha(runtime_payload),
        }
        return subprocess.CompletedProcess(
            command, 0, json.dumps(result, sort_keys=True).encode() + b"\n", b""
        )


@pytest.fixture
def generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    declaration_path, declaration = _write_declaration(tmp_path)
    compiled = _write_compiled(tmp_path, declaration)
    authorities = _write_authorities(tmp_path, monkeypatch)
    return declaration, _paths(
        tmp_path, compiled, declaration_path, authorities
    )


def test_final_wrapper_compatibility_digests_match_current_validated_b1_gate() -> None:
    gate = load_b1_authority_gate(cohort.ROOT)

    assert cohort.file_sha256(cohort.ROOT / "docs/multires/B1-GATE.json") == (
        cohort.CURRENT_B1_GATE_SHA256
    )
    assert cohort.EXPECTED_SHA256["cm"] == gate.cm_executable_sha256
    assert cohort.EXPECTED_SHA256["pmove"] == gate.pmove_executable_sha256
    assert cohort.EXPECTED_SHA256["fall"] == gate.fall_executable_sha256
    assert (
        cohort.EXPECTED_SHA256["hook_attestation"]
        == gate.hook_attestation_sha256
    )


def test_final_wrapper_derives_all_six_exact_authorities_from_current_gate() -> None:
    attestation = (
        cohort.ROOT / "tests/fixtures/multires/hook-parity-pullspeed-1700.json"
    )

    assert cohort._expected_authority_sha256(attestation) == {
        "b1_gate": cohort.CURRENT_B1_GATE_SHA256,
        "cm": "781edaee1b9317766dbf831ad5edc8b5fdebe696969ca1efe0e54e2f3e5c7d1e",
        "pmove": "66b481e924ec3d0a5e4eaf5458dd34cfe3c0927d5b7650455bceb368666718e4",
        "hook": "cd8bc4107ae2e9f4ac006fbe469b360832db80b96a5597c2e5dfe12c32dc9284",
        "fall": "dfdcf7ed74cc3ad7b8aa73df86986a8a4a31207da98ccffb4dd61673c324bef8",
        "hook_attestation": (
            "2e473d8face6b89f5b32798ddc5264bb8cc406e8dc29fd837e85bbd11b53d5ab"
        ),
    }


def test_historical_b1_pin_rejects_current_gate_before_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cohort,
        "CURRENT_B1_GATE_SHA256",
        "909b1e46b4c3dca8adb6ab9017cd8716daa8c6cdd3eb106ae11aa09bee0572f8",
    )
    attestation = (
        cohort.ROOT / "tests/fixtures/multires/hook-parity-pullspeed-1700.json"
    )

    with pytest.raises(
        cohort.MaterializeCohortError,
        match="current B1 gate exact bytes differ.*909b1e46.*eb99e08e",
    ):
        cohort._expected_authority_sha256(attestation)


def test_success_runs_in_declaration_order_and_publishes_exact_membership(
    generated,
) -> None:
    declaration, arguments = generated
    runner = FakeMaterializer()

    report = cohort.materialize_cohort(**arguments, runner=runner)

    expected = [row["map"] for row in declaration["maps"]]
    assert runner.maps == expected
    assert report["passed"] is True
    assert report["attempted_count"] == report["pass_count"] == 28
    assert report["materialized_published"] is True
    assert report["materialized_membership"]["actual_file_count"] == 196
    assert not arguments["stage_dir"].exists()
    assert verify_stage_membership(
        declaration, arguments["materialized_dir"], "materialized"
    )["passed"] is True
    assert len(list(arguments["materialized_dir"].iterdir())) == 196
    assert len(list(arguments["log_dir"].iterdir())) == 56
    report_payload = arguments["report_path"].read_bytes()
    assert report_payload == canonical_bytes(report)
    assert all(row["status"] == "passed" for row in report["maps"])


def test_authority_hash_mismatch_fails_before_stage_or_subprocess(generated) -> None:
    _declaration, arguments = generated
    arguments["cm_oracle"].write_bytes(b"substituted authority\n")
    runner = FakeMaterializer()

    with pytest.raises(
        cohort.MaterializeCohortError, match="cm authority SHA-256 differs"
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert runner.maps == []
    assert not arguments["stage_dir"].exists()
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "authority-preflight"
    assert failure["passed"] is False
    assert failure["materialized_published"] is False
    assert failure["compiled_membership"]["actual_file_count"] == 168


def test_mid_cohort_failure_is_terminal_and_never_publishes_subset(
    generated,
) -> None:
    declaration, arguments = generated
    runner = FakeMaterializer(fail_at=5)

    with pytest.raises(
        cohort.MaterializeCohortError, match="materializer exited 1"
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert runner.maps == [row["map"] for row in declaration["maps"][:6]]
    assert arguments["stage_dir"].is_dir()
    assert not arguments["materialized_dir"].exists()
    failure_bytes = arguments["report_path"].read_bytes()
    failure = json.loads(failure_bytes)
    assert failure["phase"] == "map-materialization"
    assert failure["attempted_count"] == 6
    assert failure["pass_count"] == 5
    assert [row["status"] for row in failure["maps"][:7]] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
        "failed",
        "not-attempted",
    ]
    assert len(list(arguments["log_dir"].iterdir())) == 12

    with pytest.raises(cohort.MaterializeCohortError, match="refusing reuse"):
        cohort.materialize_cohort(**arguments, runner=runner)
    assert arguments["report_path"].read_bytes() == failure_bytes
    assert len(runner.maps) == 6


def test_result_digest_tamper_fails_closed_after_one_invocation(generated) -> None:
    declaration, arguments = generated
    runner = FakeMaterializer(bad_digest_at=0)

    with pytest.raises(
        cohort.MaterializeCohortError, match="attestation digest differs"
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert runner.maps == [declaration["maps"][0]["map"]]
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "map-result-validation"
    assert failure["attempted_count"] == 1
    assert failure["pass_count"] == 0
    assert failure["maps"][0]["status"] == "failed"
    assert all(
        row["status"] == "not-attempted" for row in failure["maps"][1:]
    )


def test_canonical_attestation_with_wrong_source_binding_is_rejected(
    generated,
) -> None:
    declaration, arguments = generated
    runner = FakeMaterializer(bad_source_binding_at=0)

    with pytest.raises(
        cohort.MaterializeCohortError, match="source projection differs"
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert runner.maps == [declaration["maps"][0]["map"]]
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "map-result-validation"
    assert failure["maps"][0]["status"] == "failed"


def test_runtime_bytes_must_exactly_match_canonical_attestation(generated) -> None:
    declaration, arguments = generated
    runner = FakeMaterializer(tamper_runtime_at=0)

    with pytest.raises(
        cohort.MaterializeCohortError,
        match="runtime sidecar differs from V4 attestation",
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert runner.maps == [declaration["maps"][0]["map"]]
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "map-result-validation"


def test_compiled_input_tamper_after_result_rejects_final_publication(
    generated,
) -> None:
    _declaration, arguments = generated
    runner = FakeMaterializer(tamper_first_bsp_at=9)

    with pytest.raises(
        cohort.MaterializeCohortError, match="changed during materialization"
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert len(runner.maps) == 28
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "materialized-input-finalization"
    assert failure["failure"]["map"] is None
    assert failure["pass_count"] == 28


def test_declaration_change_before_publication_rejects_cohort(generated) -> None:
    declaration, arguments = generated
    materializer = FakeMaterializer()

    def changing_runner(command, **kwargs):
        completed = materializer(command, **kwargs)
        if len(materializer.maps) == 28:
            changed = dict(declaration)
            changed["cohort_id"] = "b2g26_future_90001"
            arguments["declaration_path"].write_bytes(canonical_bytes(changed))
        return completed

    with pytest.raises(
        cohort.MaterializeCohortError, match="declaration changed"
    ):
        cohort.materialize_cohort(**arguments, runner=changing_runner)

    assert len(materializer.maps) == 28
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "declaration-finalization"
    assert failure["failure"]["map"] is None


def test_unexpected_compiled_member_fails_exactly_before_copy(generated) -> None:
    _declaration, arguments = generated
    (arguments["compiled_dir"] / "undeclared-replacement.bsp").write_bytes(
        b"replacement\n"
    )
    runner = FakeMaterializer()

    with pytest.raises(
        cohort.MaterializeCohortError, match="exact 168-file declaration"
    ):
        cohort.materialize_cohort(**arguments, runner=runner)

    assert runner.maps == []
    assert not arguments["stage_dir"].exists()
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "compiled-membership"
    assert failure["compiled_membership"]["actual_file_count"] == 169
    assert failure["compiled_membership"]["passed"] is False


def test_subprocess_timeout_is_bounded_logged_and_terminal(generated) -> None:
    declaration, arguments = generated
    calls = []

    def timeout_runner(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial stdout\n", stderr=b""
        )

    with pytest.raises(
        cohort.MaterializeCohortError,
        match="materializer timed out after 900 seconds",
    ):
        cohort.materialize_cohort(**arguments, runner=timeout_runner)

    assert len(calls) == 1
    assert not arguments["materialized_dir"].exists()
    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "map-materialization"
    assert failure["maps"][0]["status"] == "failed"
    prefix = f"0000-{declaration['maps'][0]['map']}.materialize"
    assert (arguments["log_dir"] / f"{prefix}.stdout.json").read_bytes() == (
        b"partial stdout\n"
    )
    assert (arguments["log_dir"] / f"{prefix}.stderr.log").read_bytes() == (
        b"materializer timed out after 900 seconds\n"
    )


def test_existing_stage_preflight_still_publishes_failure_provenance(
    generated,
) -> None:
    _declaration, arguments = generated
    arguments["stage_dir"].mkdir()

    with pytest.raises(cohort.MaterializeCohortError, match="refusing reuse"):
        cohort.materialize_cohort(**arguments, runner=FakeMaterializer())

    failure = json.loads(arguments["report_path"].read_text())
    assert failure["phase"] == "path-preflight"
    assert failure["attempted_count"] == 0
    assert failure["materialized_published"] is False


def test_atomic_rename_noreplace_preserves_racing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source-marker").write_bytes(b"source\n")
    (destination / "destination-marker").write_bytes(b"destination\n")

    with pytest.raises(cohort.MaterializeCohortError, match="File exists"):
        cohort._rename_noreplace(source, destination)

    assert (source / "source-marker").read_bytes() == b"source\n"
    assert (destination / "destination-marker").read_bytes() == b"destination\n"
