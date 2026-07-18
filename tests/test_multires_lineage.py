from dataclasses import replace

import pytest

from harness.multires_contract import FEATURE_SCHEMA_SHA256, OBS_DIM
from harness.multires_lineage import (
    CHECKPOINT_FORMAT,
    LineageError,
    MultiresCheckpointManifest,
    select_attested_checkpoint,
)
from harness.multires_contract import POLICY_GENERATION
from harness.multires_runtime import (
    B4_CAUSAL_VERSION,
    B4_CLIENT_WIRE_VERSION,
    adapt_b4_observation_descriptor,
    validate_runtime_evidence,
)
from harness.multires_training_config import MultiresTrainingConfiguration


STATE = "1" * 64
ATLAS = "a" * 64
RUNTIME = "b" * 64
OPTIMIZER = "2" * 64
TRAINING_CONFIG = MultiresTrainingConfiguration.create(
    reward={"damage_reward": 0.003},
    guide_dropout={"global_probability": 0.1},
    ppo={"clip_coef": 0.2},
)


def manifest():
    return MultiresCheckpointManifest.create(
        state_schema_sha256=STATE,
        optimizer_identity_sha256=OPTIMIZER,
        atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        training_config=TRAINING_CONFIG,
        initialization="random",
        training_step=10,
    )


def test_manifest_binds_fresh_298_graph_atlas_and_runtime():
    value = manifest()
    value.validate(
        expected_state_schema_sha256=STATE,
        expected_optimizer_identity_sha256=OPTIMIZER,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
        expected_training_config=TRAINING_CONFIG,
    )
    assert value.checkpoint_format == CHECKPOINT_FORMAT
    assert value.observation_dim == OBS_DIM == 298
    assert value.posture_classes == 3
    assert value.feature_schema_sha256 == FEATURE_SCHEMA_SHA256


def test_manifest_rejects_legacy_dimensions_and_runtime_or_atlas_mismatch():
    legacy = replace(manifest(), observation_dim=219, posture_classes=2)
    with pytest.raises(LineageError, match="observation_dim"):
        legacy.validate(
            expected_state_schema_sha256=STATE,
            expected_optimizer_identity_sha256=OPTIMIZER,
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_training_config=TRAINING_CONFIG,
        )
    with pytest.raises(LineageError, match="atlas_sha256"):
        manifest().validate(
            expected_state_schema_sha256=STATE,
            expected_optimizer_identity_sha256=OPTIMIZER,
            expected_atlas_sha256="c" * 64,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_training_config=TRAINING_CONFIG,
        )


def test_manifest_parser_has_no_optional_legacy_tail_or_fallback():
    raw = manifest().to_mapping()
    raw["legacy_observation_dim"] = 219
    with pytest.raises(LineageError, match="fields differ"):
        MultiresCheckpointManifest.from_mapping(raw)
    with pytest.raises(LineageError, match="initialization"):
        MultiresCheckpointManifest.create(
            state_schema_sha256=STATE,
            optimizer_identity_sha256=OPTIMIZER,
            atlas_sha256=ATLAS,
            runtime_manifest_sha256=RUNTIME,
            training_config=TRAINING_CONFIG,
            initialization="legacy-warm-start",
            training_step=0,
        )


def test_checkpoint_selector_requires_one_explicit_non_symlink_file(tmp_path):
    checkpoint = tmp_path / "exact.pt"
    checkpoint.write_bytes(b"envelope checked by loader")
    assert select_attested_checkpoint(checkpoint) == checkpoint.resolve()
    with pytest.raises(LineageError, match="explicit"):
        select_attested_checkpoint(tmp_path)
    link = tmp_path / "latest.pt"
    link.symlink_to(checkpoint)
    with pytest.raises(LineageError, match="symbolic"):
        select_attested_checkpoint(link)


def runtime_evidence():
    return {
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_dim": 298,
        "action_dim": 8,
        "posture_classes": 3,
        "protocol_generation": 2,
        "observation_magic": 0x514D324F,
        "action_magic": 0x514D3241,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": 4,
        "rollout_schema": "ppo-telemetry-multires-v1",
        "atlas_sha256": ATLAS,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": 0x514D3343,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": 80,
    }


def test_runtime_admission_requires_atomic_b3_b4_evidence():
    validated = validate_runtime_evidence(
        runtime_evidence(), expected_atlas_sha256=ATLAS
    )
    assert validated.client_wire_version == B4_CLIENT_WIRE_VERSION
    assert len(validated.runtime_manifest_sha256) == 64


def test_b4_runtime_descriptor_adapts_without_redefining_endpoint_layout():
    from harness.runtime_attestation import describe_observation

    evidence = adapt_b4_observation_descriptor(
        describe_observation({}), atlas_sha256=ATLAS
    )
    validated = validate_runtime_evidence(evidence, expected_atlas_sha256=ATLAS)
    assert validated.observation_magic == 0x514D324F
    assert validated.action_magic == 0x514D3241
    assert validated.client_wire_version == B4_CLIENT_WIRE_VERSION
    assert validated.teacher_version == 4


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("observation_dim", 219, "observation_dim"),
        ("posture_classes", 2, "posture_classes"),
        ("protocol_generation", 1, "protocol_generation"),
        ("client_wire_version", 5, "client_wire_version"),
        ("causal_version", 1, "causal_version"),
        ("teacher_version", 3, "teacher_version"),
        ("rollout_schema", "ppo-telemetry-v8", "rollout_schema"),
        ("public_teacher_packing_separate", False, "public_teacher"),
        ("public_teacher_field_violations", 1, "public_teacher"),
    ),
)
def test_runtime_admission_rejects_legacy_or_privilege_leaking_evidence(
    field, value, match
):
    evidence = runtime_evidence()
    evidence[field] = value
    with pytest.raises(LineageError, match=match):
        validate_runtime_evidence(evidence, expected_atlas_sha256=ATLAS)
