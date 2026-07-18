from __future__ import annotations

import json
import hashlib
import inspect
from pathlib import Path
import struct
import sys

import pytest

from tools.run_b3_design_prior_campaign import canonical_bytes

from tools.assemble_b3_gate import (
    B3GateError,
    BUNDLE_CLAIM_EXPECTED_VALUES,
    BUNDLE_CLAIM_NODE_IDS,
    BUNDLE_SOURCE_PATHS,
    BUNDLE_TEST_COMMANDS,
    _source_closure,
    derive_bundle_claim_evidence,
    validate_bundle_evidence,
)
from tools.run_b3_component_evidence import (
    ATLAS_HEADER_BYTES,
    ATLAS_MAGIC,
    B3ComponentError,
    _build_bundle_report,
    _run_commands,
    parse_atlas_recovery,
    produce_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = {"repository_commit": "1" * 40, "repository_tree": "2" * 40, "git_clean": True}


def _tests(commands: tuple[tuple[str, ...], ...]) -> dict:
    stdout = b"component passed\n"
    runs = [{
        "command": list(command), "exit_code": 0, "passed_count": 1,
        "stdout": {"bytes": len(stdout), "sha256": hashlib.sha256(stdout).hexdigest()},
        "stderr": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
    } for command in commands]
    body = {"schema": "q2-b3-component-test-runs-v1", "runs": runs}
    return {
        "report_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
        "commands_sha256": hashlib.sha256(
            canonical_bytes([list(command) for command in commands])
        ).hexdigest(),
        "passed_count": len(runs), "failed_count": 0, "runs": runs,
    }


TESTS = _tests(BUNDLE_TEST_COMMANDS)


def _node(flags: int, cost: int) -> bytes:
    return struct.pack("<iiiHBBHHiIIHHI", 0, 0, 0, flags, 0, 0, 56, 0, 0, cost, 1, 65535, 1, 0)


def _edge(target: int, edge_type: int) -> bytes:
    return struct.pack("<IBBHIIHHHHI", target, edge_type, 0, 0, 0, 1, 0, 65535, 1, 1, 0xFFFFFFFF)


def _atlas() -> bytes:
    nodes = _node(1 << 2, 0) + _node(0, 100) + _node(1 << 6, 100)
    offsets = [0, 0, 1, 2]
    graph = struct.pack("<Q", len(offsets)) + struct.pack("<4I", *offsets) + _edge(0, 0) + _edge(0, 11)
    counts = [0, 3, 2, 0, 0]
    lengths = [0, len(nodes), len(graph), 0, 0]
    header = (
        ATLAS_MAGIC + struct.pack("<HHI", 1, 0x454C, ATLAS_HEADER_BYTES)
        + struct.pack("<3q", 0, 0, 0) + struct.pack("<4I", 4, 16, 64, 256)
        + struct.pack("<5Q", *counts) + struct.pack("<5Q", *lengths)
    )
    assert len(header) == ATLAS_HEADER_BYTES
    return header + nodes + graph


def test_raw_atlas_recovery_inspector_counts_descents_plateaus_and_hooks(tmp_path: Path):
    path = tmp_path / "map.atlas.bin"
    path.write_bytes(_atlas())
    report = parse_atlas_recovery(path)
    assert report["finite_non_safe_cells"] == 2
    assert report["strict_descending_cells"] == 1
    assert report["mover_plateau_cells"] == 1
    assert report["unresolved_cells"] == 0
    assert report["hook_edge_count"] == 1
    assert report["hook_source_indices"] == [[0, 0, 0]]


def test_raw_atlas_inspector_rejects_csr_and_length_forgery(tmp_path: Path):
    payload = bytearray(_atlas())
    struct.pack_into("<Q", payload, 96 + 2 * 8, 1)
    path = tmp_path / "bad.atlas.bin"
    path.write_bytes(payload)
    with pytest.raises(B3ComponentError, match="section lengths"):
        parse_atlas_recovery(path)


def test_bundle_document_builder_derives_gate_document_without_caller_claims():
    report = _build_bundle_report(
        repo_root=ROOT,
        repository=REPOSITORY,
        tests=TESTS,
    )
    assert report["passed"] is True
    assert report["bundle_v3"]["public_enabled"] is False
    assert report["claim_tests"]["bundle_v3.public_enabled"] == {
        "required_node_ids": [
            "tests/test_map_farm.py::test_bundle_v3_is_isolated_by_default_and_explicitly_admitted"
        ],
        "node_outcomes": [{
            "node_id": "tests/test_map_farm.py::test_bundle_v3_is_isolated_by_default_and_explicitly_admitted",
            "outcome": "passed",
        }],
        "all_required_passed": True,
        "derived_value": False,
    }
    validate_bundle_evidence(
        report, REPOSITORY, _source_closure(ROOT, BUNDLE_SOURCE_PATHS),
    )


def test_production_bundle_api_has_no_repository_or_test_claim_bypass():
    signature = inspect.signature(produce_bundle)
    assert tuple(signature.parameters) == ("output", "repo_root")
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_bundle_schema_pins_exact_named_node_claims():
    schema = json.loads(
        (ROOT / "schemas/q2-b3-bundle-evidence-v1.schema.json").read_text()
    )
    claim_schemas = schema["properties"]["claim_tests"]["properties"]
    for claim, node_ids in BUNDLE_CLAIM_NODE_IDS.items():
        value_ref, exact_ref = claim_schemas[claim]["allOf"]
        expected_value_def = "true_claim" if BUNDLE_CLAIM_EXPECTED_VALUES[claim] else "false_claim"
        assert value_ref["$ref"] == f"#/$defs/{expected_value_def}"
        exact = schema["$defs"][exact_ref["$ref"].removeprefix("#/$defs/")]
        assert exact["properties"]["required_node_ids"]["const"] == list(node_ids)
        assert exact["properties"]["node_outcomes"]["const"] == [
            {"node_id": node_id, "outcome": "passed"}
            for node_id in node_ids
        ]


def test_bundle_schema_accepts_derived_report_and_rejects_forgery():
    jsonschema = pytest.importorskip("jsonschema")
    report = _build_bundle_report(
        repo_root=ROOT,
        repository=REPOSITORY,
        tests=TESTS,
    )
    schema = json.loads(
        (ROOT / "schemas/q2-b3-bundle-evidence-v1.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(report)

    forged = json.loads(json.dumps(report))
    forged["claim_tests"]["bundle_v3.public_enabled"]["required_node_ids"] = [
        "tests/test_map_farm.py::test_stock_rotation_shuffles_without_repeats_in_a_cycle"
    ]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(forged)


def test_bundle_producer_refuses_preexisting_output(tmp_path: Path):
    output = tmp_path / "bundle.json"
    output.write_text("owned")
    with pytest.raises(B3ComponentError, match="already exists"):
        produce_bundle(output=output, repo_root=ROOT)


def test_bundle_claim_derivation_rejects_generic_or_incomplete_test_runs():
    generic = _tests(((sys.executable, "-m", "pytest", "-q", "tests/test_map_farm.py"),))
    with pytest.raises(B3GateError, match="node ID is undeclared"):
        derive_bundle_claim_evidence(generic)

    incomplete = json.loads(json.dumps(TESTS))
    incomplete["runs"].pop()
    with pytest.raises(B3GateError, match="node membership differs"):
        derive_bundle_claim_evidence(incomplete)

    not_exact = json.loads(json.dumps(TESTS))
    not_exact["runs"][0]["passed_count"] = 2
    with pytest.raises(B3GateError, match="did not pass exactly once"):
        derive_bundle_claim_evidence(not_exact)


def test_component_commands_timeout_kills_process_group(tmp_path: Path):
    command = (sys.executable, "-c", "import time; time.sleep(30)")
    with pytest.raises(B3ComponentError, match="timed out"):
        _run_commands((command,), tmp_path, timeout_seconds=0.05)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("print('1 passed'); raise SystemExit(2)", "failed or reported no passing"),
        ("print('0 passed')", "failed or reported no passing"),
    ],
)
def test_component_commands_refuse_nonzero_and_zero_test_runs(
    tmp_path: Path, source: str, message: str,
):
    command = (sys.executable, "-c", source)
    with pytest.raises(B3ComponentError, match=message):
        _run_commands((command,), tmp_path, timeout_seconds=5.0)
