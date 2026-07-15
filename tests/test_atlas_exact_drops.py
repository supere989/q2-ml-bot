from __future__ import annotations

from pathlib import Path

import pytest

from harness.atlas_exact_drops import (
    DropTrajectory,
    _decode_json_object,
    classify_drop_trajectories,
    summarize_drop_classifications,
)


class _NoCallPmove:
    def __init__(self, binary: Path, map_sha256: str) -> None:
        self.binary = binary
        self.requests = 0
        self.max_requests = 100
        self.identity = {
            "parameters": {"gravity": 800, "airaccelerate": 0},
            "tool_identity": "1" * 64,
            "physics_identity": "2" * 64,
            "map_sha256": map_sha256,
        }

    def call(self, requests):
        if not requests:
            return []
        raise AssertionError(f"dynamic-mover Unknown wrote Pmove: {requests}")


class _NoCallFall:
    def __init__(self, binary: Path) -> None:
        self.binary = binary
        self.requests = 0
        self.max_requests = 100
        self.identity = {
            "tool_identity": "3" * 64,
            "physics_identity": "4" * 64,
        }

    def call(self, requests):
        if not requests:
            return []
        raise AssertionError(f"dynamic-mover Unknown wrote fall: {requests}")


def test_dynamic_mover_unknown_has_no_exact_evidence_or_oracle_calls(
    tmp_path: Path,
) -> None:
    bsp = tmp_path / "fixture.bsp"
    pmove_binary = tmp_path / "pmove"
    fall_binary = tmp_path / "fall"
    bsp.write_bytes(b"bsp")
    pmove_binary.write_bytes(b"pmove")
    fall_binary.write_bytes(b"fall")
    import hashlib

    trajectory = DropTrajectory(
        identifier="mover-dependent",
        source_l1=(0, 0, 0),
        direction=(1, 0),
        mode="ground",
        request={
            "origin": [0.0, 0.0, 24.0],
            "velocity": [0.0, 0.0, 0.0],
            "pm_type": 0,
            "pm_flags": 4,
            "pm_time": 0,
            "gravity": 800,
            "airaccelerate": 0,
            "delta_angles_short": [0, 0, 0],
            "snapinitial": False,
            "commands": [{"msec": 50}, {"msec": 50}],
        },
        response={},
        dynamic_movers=True,
    )
    result = classify_drop_trajectories(
        [trajectory],
        bsp=bsp,
        pmove_process=_NoCallPmove(
            pmove_binary, hashlib.sha256(bsp.read_bytes()).hexdigest(),
        ),
        fall_process=_NoCallFall(fall_binary),
    )[0]
    assert result["classification"]["reason"] == "unsupported_dynamic_mover"
    assert result["evidence"] == 0
    assert result["validation_version"] == 0
    summary = summarize_drop_classifications([result])
    assert summary["unknown_omitted"] == 1
    assert summary["evidence"] == 0
    assert summary["validation_version"] == 0
    assert summary["evidence"] != 10


def test_empty_drop_summary_has_no_attempted_authority_evidence() -> None:
    assert summarize_drop_classifications([]) == {
        "classification_status": "oracle",
        "evidence": 0,
        "validation_version": 0,
        "candidate_count": 0,
        "exact_safe": 0,
        "exact_lethal": 0,
        "unknown_omitted": 0,
        "severity_counts": {},
    }


@pytest.mark.parametrize(
    "payload",
    [
        b'{"ok":true,"ok":false}',
        b'{"ok":NaN}',
        b'{"ok":Infinity}',
    ],
)
def test_persistent_fall_json_rejects_duplicate_keys_and_nonfinite(
    payload: bytes,
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _decode_json_object(payload)
