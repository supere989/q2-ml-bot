import copy
from pathlib import Path

import numpy as np
import pytest

from harness.runtime_attestation import (
    AttestationError,
    build_runtime_manifest,
    describe_observation,
    semantic_digest,
    verify_runtime_manifest,
)


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _fixture_tree(tmp_path: Path, label: str):
    root = tmp_path / label / "runtime"
    source = tmp_path / label / "source"
    _write(root / "q2ded", b"same-q2ded")
    _write(root / "lithium" / "game.so", b"same-game")
    _write(root / "baseq2" / "maps" / "mltrain_test.bsp", b"same-map")
    _write(root / "baseq2" / "maps" / "mltrain_test.json", b'{"hook": 1}')
    rust = tmp_path / label / "python" / "q2_lattice_rs.so"
    _write(rust, b"same-rust")
    _write(source / "harness" / "runtime.py", b'KEY = "R_KILL"\n')
    _write(source / "models" / "policy.py", b"POLICY = 2\n")
    _write(source / "train" / "ppo.py", b'GEN = "Q2_PROTOCOL_GENERATION"\n')
    return root, source, rust


def _policy(obs_dim=298):
    state = {
        "encoder.weight": np.zeros((4, obs_dim), dtype=np.float32),
        "actor.weight": np.zeros((8, 4), dtype=np.float32),
    }
    # Exercise the public descriptor shape without importing Torch.
    from harness.runtime_attestation import describe_policy_state

    return describe_policy_state(
        state, observation_dim=obs_dim, action_dim=8, hidden_dim=4,
        architecture="test.Policy",
    )


def _observation():
    return describe_observation({})


def _build(root, source, rust, *, environment=None, revision="abc123", policy=None):
    env = {
        "Q2_RUST_LATTICE": "1",
        "R_KILL": "2.0",
        "Q2_ROOT": str(root),  # operational paths are intentionally excluded
        "Q2_ROLLOUT_TOKEN": "secret",
    }
    if environment:
        env.update(environment)
    return build_runtime_manifest(
        q2_root=root,
        source_root=source,
        map_names=["mltrain_test"],
        rust_extension=rust,
        source_revision=revision,
        runtime_config={"n_bots": 4, "n_ml": 4, "timescale": 10.0},
        environment=env,
        observation_descriptor=_observation(),
        policy_descriptor=policy or _policy(),
    )


def test_retired_width_environment_cannot_select_an_observation_layout():
    baseline = describe_observation({})
    assert describe_observation({"Q2_EXT_OBS": "0"}) == baseline
    assert describe_observation({"Q2_EXT_OBS": "1"}) == baseline
    assert baseline["total_dim"] == 298


def test_manifest_is_path_independent_and_covers_semantic_inputs(tmp_path):
    first = _fixture_tree(tmp_path, "host-a")
    second = _fixture_tree(tmp_path, "host-b")
    left = _build(*first, environment={
        "Q2_SOURCE_REVISION": "host-a-value", "Q2_EXT_OBS": "0",
    })
    right = _build(*second, environment={
        "Q2_SOURCE_REVISION": "host-b-value", "Q2_EXT_OBS": "1",
    })

    assert left["manifest_sha256"] == right["manifest_sha256"]
    assert left["diagnostics"]["q2_root"] != right["diagnostics"]["q2_root"]
    semantic = left["semantic"]
    assert set(semantic) == {
        "artifacts", "maps", "observation", "policy", "environment",
        "runtime_config", "source",
    }
    assert semantic["artifacts"]["q2ded"]["sha256"]
    assert semantic["artifacts"]["game_module"]["sha256"]
    assert semantic["artifacts"]["rust_lattice"]["sha256"]
    assert len(semantic["maps"][0]["files"]) == 2
    assert semantic["observation"]["total_dim"] == 298
    assert semantic["observation"]["action_cardinalities"]["vertical_intent"] == 3
    assert semantic["policy"]["state_schema_sha256"]
    assert semantic["environment"]["R_KILL"] == "2.0"
    assert "Q2_ROOT" not in semantic["environment"]
    assert "Q2_ROLLOUT_TOKEN" not in semantic["environment"]
    assert "Q2_SOURCE_REVISION" not in semantic["environment"]
    assert "Q2_EXT_OBS" not in semantic["environment"]
    assert semantic["source"]["git_revision"] == "abc123"


def test_verification_reports_artifact_reward_policy_and_revision_drift(tmp_path):
    fixture = _fixture_tree(tmp_path, "host")
    expected = _build(*fixture)

    fixture[0].joinpath("lithium/game.so").write_bytes(b"different-game")
    current = _build(
        *fixture,
        environment={"R_KILL": "3.0"},
        revision="def456",
        policy=_policy(obs_dim=298) | {"architecture": "test.PolicyV3"},
    )
    result = verify_runtime_manifest(current, expected=expected)
    assert not result.valid
    paths = {difference["path"] for difference in result.differences}
    assert "semantic.artifacts.game_module.sha256" in paths
    assert "semantic.environment.R_KILL" in paths
    assert "semantic.policy.architecture" in paths
    assert "semantic.source.git_revision" in paths


def test_digest_and_hmac_reject_tampering(tmp_path):
    fixture = _fixture_tree(tmp_path, "host")
    manifest = build_runtime_manifest(
        q2_root=fixture[0],
        source_root=fixture[1],
        map_names=["mltrain_test"],
        rust_extension=fixture[2],
        source_revision="abc123",
        runtime_config={},
        environment={"Q2_RUST_LATTICE": "1"},
        observation_descriptor=_observation(),
        policy_descriptor=_policy(),
        hmac_key=b"shared-secret",
    )
    assert verify_runtime_manifest(
        manifest, hmac_key=b"shared-secret", require_signature=True
    ).valid
    no_key = verify_runtime_manifest(manifest, require_signature=True)
    assert not no_key.valid
    assert "manifest signature verification key is required" in no_key.errors
    tampered = copy.deepcopy(manifest)
    tampered["semantic"]["runtime_config"]["n_ml"] = 99
    result = verify_runtime_manifest(
        tampered, hmac_key=b"shared-secret", require_signature=True
    )
    assert not result.valid
    assert "manifest digest mismatch" in result.errors
    assert "manifest signature mismatch" in result.errors
    assert not verify_runtime_manifest(
        manifest, hmac_key=b"", require_signature=True
    ).valid
    with pytest.raises(AttestationError, match="must not be empty"):
        build_runtime_manifest(
            q2_root=fixture[0],
            source_root=fixture[1],
            map_names=["mltrain_test"],
            rust_extension=fixture[2],
            source_revision="abc123",
            runtime_config={},
            environment={"Q2_RUST_LATTICE": "1"},
            observation_descriptor=_observation(),
            policy_descriptor=_policy(),
            hmac_key=b"",
        )


def test_semantic_comparison_does_not_alias_json_scalar_types(tmp_path):
    fixture = _fixture_tree(tmp_path, "host")
    expected = _build(*fixture)
    expected["semantic"]["runtime_config"]["deterministic_actions"] = False
    expected["manifest_sha256"] = semantic_digest(expected["semantic"])
    for key, replacement in (
        ("deterministic_actions", 0),
        ("n_bots", 4.0),
    ):
        current = copy.deepcopy(expected)
        current.pop("signature", None)
        current["semantic"]["runtime_config"][key] = replacement
        current["manifest_sha256"] = semantic_digest(current["semantic"])
        result = verify_runtime_manifest(current, expected=expected)
        assert not result.valid
        assert any(
            difference["path"]
            == f"semantic.runtime_config.{key}"
            for difference in result.differences
        )
