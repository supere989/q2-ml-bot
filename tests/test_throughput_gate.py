import hashlib
import hmac
import json
import sys

import pytest

from harness.runtime_attestation import MANIFEST_SCHEMA, semantic_digest
from harness.throughput_gate import (
    ThroughputGateError,
    evaluate_throughput_gate,
    parse_worker_throughput,
)


DIGEST = "a" * 64
POLICY = "b" * 64


def _record(worker, start_s, duration_s, transitions, *, digest=DIGEST, timeouts=0):
    return {
        "worker_id": worker,
        "producer": "q2",
        "policy_sha256": POLICY,
        "config_hash": "cfg",
        "runtime_manifest_sha256": digest,
        "collection": {
            "started_unix_ns": int(start_s * 1e9),
            "finished_unix_ns": int((start_s + duration_s) * 1e9),
            "elapsed_seconds": duration_s,
            "transitions": transitions,
            "timeouts": timeouts,
        },
    }


def test_concurrent_capacity_gate_reports_conservative_aggregate_sps():
    records = [
        _record("wsl", 1, 10, 200),
        _record("nobara", 2, 8, 160),
    ]
    result = evaluate_throughput_gate(
        records,
        baseline_sps=18.0,
        min_speedup=1.5,
        min_overlap_ratio=0.75,
        expected_runtime_manifest_sha256=DIGEST,
    )
    assert result.passed
    assert result.metrics["union_seconds"] == 10.0
    assert result.metrics["overlap_seconds"] == 8.0
    assert result.metrics["overlap_ratio"] == 1.0
    assert result.metrics["aggregate_sps"] == 36.0
    assert result.metrics["speedup_vs_baseline"] == 2.0


def test_gate_rejects_mismatched_runtime_low_overlap_and_timeouts():
    result = evaluate_throughput_gate(
        [
            _record("wsl", 1, 4, 40, timeouts=1),
            _record("nobara", 10, 4, 40, digest="c" * 64),
        ],
        min_aggregate_sps=50,
        min_overlap_ratio=0.5,
        max_timeouts=0,
        expected_runtime_manifest_sha256=DIGEST,
    )
    assert not result.passed
    text = "\n".join(result.reasons)
    assert "different runtime manifests" in text
    assert "timeouts" in text
    assert "overlap" in text
    assert "aggregate SPS" in text


def test_parser_rejects_inconsistent_wall_and_monotonic_intervals():
    record = _record("worker", 1, 2, 20)
    record["collection"]["elapsed_seconds"] = 9.0
    with pytest.raises(ThroughputGateError, match="duration mismatch"):
        parse_worker_throughput(record)


def test_gate_cli_requires_and_verifies_expected_manifest_signature(
    tmp_path, monkeypatch
):
    from tools.rollout_throughput_gate import main

    key = b"gate-test-secret"
    semantic = {"test": "signed-runtime"}
    digest = semantic_digest(semantic)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "semantic": semantic,
        "manifest_sha256": digest,
        "signature": {
            "algorithm": "hmac-sha256",
            "value": hmac.new(
                key, digest.encode("ascii"), hashlib.sha256
            ).hexdigest(),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record_paths = []
    for index, record in enumerate((
        _record("wsl", 1, 10, 200, digest=digest),
        _record("nobara", 2, 8, 160, digest=digest),
    )):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        record_paths.append(path)

    argv = [
        "rollout_throughput_gate.py",
        *(str(path) for path in record_paths),
        "--expected-manifest", str(manifest_path),
        "--attestation-key-env", "ATTEST_KEY",
        "--require-attestation-signature",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("ATTEST_KEY", raising=False)
    with pytest.raises(SystemExit) as missing:
        main()
    assert missing.value.code == 2

    monkeypatch.setenv("ATTEST_KEY", key.decode())
    assert main() == 0
