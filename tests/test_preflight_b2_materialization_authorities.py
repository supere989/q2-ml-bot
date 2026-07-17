from __future__ import annotations

import json
from pathlib import Path

from tools import materialize_generated_cohort as materializer
from tools import preflight_b2_materialization_authorities as preflight_tool


def test_preflight_is_read_only_and_reports_exact_records(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {}
    for label in ("cm", "pmove", "hook", "fall", "hook_attestation"):
        path = tmp_path / label
        path.write_bytes(label.encode())
        paths[label] = path
    expected = {
        "b1_gate": {"actual_sha256": "a" * 64, "passed": True},
        "cm": {"actual_sha256": "b" * 64, "passed": True},
    }
    calls = []

    def fake_records(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        preflight_tool, "preflight_authority_records", fake_records
    )
    report = preflight_tool.preflight(
        cm_oracle=paths["cm"],
        pmove_oracle=paths["pmove"],
        hook_oracle=paths["hook"],
        fall_oracle=paths["fall"],
        hook_parity_attestation=paths["hook_attestation"],
    )

    assert report == {
        "schema": preflight_tool.SCHEMA,
        "passed": True,
        "authorities": expected,
    }
    assert len(calls) == 1
    assert set(calls[0]) == {
        "cm_oracle", "pmove_oracle", "hook_oracle", "fall_oracle",
        "hook_attestation",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cm", "fall", "hook", "hook_attestation", "pmove",
    ]


def test_cli_emits_canonical_success_json(tmp_path: Path, monkeypatch, capsys) -> None:
    paths = []
    for label in ("cm", "pmove", "hook", "fall", "attestation"):
        path = tmp_path / label
        path.write_bytes(label.encode())
        paths.append(path)
    monkeypatch.setattr(
        preflight_tool,
        "preflight",
        lambda **_kwargs: {
            "schema": preflight_tool.SCHEMA,
            "passed": True,
            "authorities": {},
        },
    )

    assert preflight_tool.main([
        "--cm-oracle", str(paths[0]),
        "--pmove-oracle", str(paths[1]),
        "--hook-oracle", str(paths[2]),
        "--fall-oracle", str(paths[3]),
        "--hook-parity-attestation", str(paths[4]),
    ]) == 0
    payload = capsys.readouterr().out
    assert payload == json.dumps(
        {"schema": preflight_tool.SCHEMA, "passed": True, "authorities": {}},
        sort_keys=True, separators=(",", ":"),
    ) + "\n"


def test_preflight_exercises_shared_records_without_writes(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {}
    expected = {
        "b1_gate": materializer.CURRENT_B1_GATE_SHA256,
    }
    for label in ("cm", "pmove", "hook", "fall", "hook_attestation"):
        path = tmp_path / label
        path.write_bytes(f"exact:{label}\n".encode())
        paths[label] = path
        expected[label] = materializer.file_sha256(path)
    monkeypatch.setattr(
        materializer, "_expected_authority_sha256", lambda _path: expected
    )
    before = {
        path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())
    }

    report = preflight_tool.preflight(
        cm_oracle=paths["cm"], pmove_oracle=paths["pmove"],
        hook_oracle=paths["hook"], fall_oracle=paths["fall"],
        hook_parity_attestation=paths["hook_attestation"],
    )

    after = {path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())}
    assert report["passed"] is True
    assert report["authorities"]["b1_gate"]["actual_sha256"] == (
        materializer.CURRENT_B1_GATE_SHA256
    )
    assert before == after


def test_cli_handles_os_error_without_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        preflight_tool, "preflight",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read failed")),
    )

    assert preflight_tool.main([
        "--cm-oracle", "cm", "--pmove-oracle", "pmove",
        "--hook-oracle", "hook", "--fall-oracle", "fall",
        "--hook-parity-attestation", "attestation",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "materialization authority preflight failed: read failed" in captured.err
