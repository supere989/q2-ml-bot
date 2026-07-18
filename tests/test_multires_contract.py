import numpy as np
import pytest

from harness.multires_contract import (
    ALL_BLOCKS,
    DYN_DIM,
    FACTUAL_DIM,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    GUIDE_DIM,
    GUIDES,
    OBS_DIM,
    RECOVERY_DIM,
    GuideDropoutConfig,
    GuideDropoutIdentity,
    apply_seeded_guide_dropout,
    pack_policy_vector,
)


def test_frozen_298_layout_is_named_contiguous_and_unique():
    assert OBS_DIM == 298
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == OBS_DIM
    assert ALL_BLOCKS[0].start == 0
    assert all(left.stop == right.start for left, right in zip(ALL_BLOCKS, ALL_BLOCKS[1:]))
    assert ALL_BLOCKS[-1].stop == OBS_DIM
    assert len(FEATURE_SCHEMA_SHA256) == 64
    assert FEATURE_NAMES[224:227] == (
        "recovery_hazard_hurt",
        "recovery_hazard_void_or_lethal_drop",
        "recovery_hazard_crush_or_current",
    )
    assert FEATURE_NAMES[228:230] == (
        "recovery_hull_clearance",
        "recovery_cost_to_safety",
    )
    assert FEATURE_NAMES[238:241] == (
        "guide_0_forward", "guide_0_quake_right", "guide_0_up"
    )


def test_b5_contract_matches_b4_ordered_feature_schema_exactly():
    from harness.protocol import POLICY_FEATURE_NAMES, POLICY_FEATURE_SCHEMA_SHA256

    assert FEATURE_NAMES == POLICY_FEATURE_NAMES
    assert FEATURE_SCHEMA_SHA256 == POLICY_FEATURE_SCHEMA_SHA256


def test_policy_vector_packer_has_no_legacy_width_fallback():
    vector = pack_policy_vector(
        np.zeros(FACTUAL_DIM),
        np.ones(DYN_DIM),
        np.full(RECOVERY_DIM, 2.0),
        np.full(GUIDE_DIM, 3.0),
    )
    assert vector.shape == (OBS_DIM,)
    assert np.all(vector[GUIDES.slice] == 3.0)

    with pytest.raises(ValueError, match="exactly 198"):
        pack_policy_vector(
            np.zeros(195), np.zeros(DYN_DIM), np.zeros(RECOVERY_DIM), np.zeros(GUIDE_DIM)
        )
    with pytest.raises(ValueError, match="non-finite"):
        bad = np.zeros(GUIDE_DIM)
        bad[0] = np.nan
        pack_policy_vector(
            np.zeros(FACTUAL_DIM), np.zeros(DYN_DIM), np.zeros(RECOVERY_DIM), bad
        )


def _guides():
    guides = np.arange(GUIDE_DIM, dtype=np.float32).reshape(4, 15)
    guides[:, 7:15] = 0.0
    guides[0, 7] = 1.0
    guides[1, 8] = 1.0
    guides[2, 9] = 1.0
    guides[3, 10] = 1.0
    return guides.reshape(-1)


def test_guide_dropout_is_deterministic_and_advisory_only():
    identity = GuideDropoutIdentity("q2dm1", 12, "client-a", 101)
    config = GuideDropoutConfig(global_probability=0.45, tick_bucket_size=10)
    first = apply_seeded_guide_dropout(_guides(), identity, config)
    second = apply_seeded_guide_dropout(_guides(), identity, config)
    assert np.array_equal(first.guides, second.guides)
    assert first.dropped_candidates == second.dropped_candidates

    # A forced global dropout proves the function owns only the 60-guide block.
    forced = apply_seeded_guide_dropout(
        _guides(), identity, GuideDropoutConfig(global_probability=1.0)
    )
    assert forced.global_drop
    assert forced.dropped_candidates == (True, True, True, True)
    assert np.count_nonzero(forced.guides) == 0


def test_guide_class_dropout_uses_frozen_eight_class_slots():
    config = GuideDropoutConfig(
        global_probability=0.0,
        class_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    result = apply_seeded_guide_dropout(
        _guides(), GuideDropoutIdentity("arena", 1, "client", 0), config
    )
    assert result.dropped_candidates == (True, False, False, False)
    assert np.count_nonzero(result.guides[:15]) == 0
    assert np.count_nonzero(result.guides[15:]) > 0
