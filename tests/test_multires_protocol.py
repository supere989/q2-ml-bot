import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

import numpy as np
import pytest

from harness.client_batch import decode_policy_action
from harness.protocol import (
    ACT_SIZE,
    DYN_FEATURE_NAMES,
    DYN_SLICE,
    FACTUAL_SLICE,
    ML_ACT_MAGIC,
    ML_OBS_MAGIC,
    OBJECTIVE_FEATURE_NAMES,
    OBJECTIVE_SLICES,
    OBJECTIVES_SLICE,
    OBS_DIM,
    OBS_DYN_DIM,
    OBS_FACTUAL_DIM,
    OBS_FMT,
    OBS_OBJECTIVE_DIM,
    OBS_RECOVERY_DIM,
    OBS_SIZE,
    POLICY_FEATURE_NAMES,
    RECOVERY_FEATURE_NAMES,
    RECOVERY_SLICE,
    SPATIAL_FEATURE_SCHEMA_SHA256,
    Action,
    SpatialPolicyFeatures,
    VerticalIntent,
    pack_action,
    parse_obs,
)


ROOT = Path(__file__).resolve().parents[1]


def _observation():
    values = list(struct.unpack(OBS_FMT, bytes(OBS_SIZE)))
    values[0:3] = [ML_OBS_MAGIC, 77, 2]
    # These are the three mandatory factual stance values at C byte offset 836.
    values[209:212] = [1.0, 0.0, 1.0]
    return parse_obs(struct.pack(OBS_FMT, *values))


def _json_name_digest(names) -> str:
    encoded = json.dumps(
        tuple(names), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_frozen_named_298_layout_has_no_optional_or_legacy_tail():
    assert OBS_DIM == 298
    assert (OBS_FACTUAL_DIM, OBS_DYN_DIM, OBS_RECOVERY_DIM) == (198, 24, 16)
    assert FACTUAL_SLICE == slice(0, 198)
    assert DYN_SLICE == slice(198, 222)
    assert RECOVERY_SLICE == slice(222, 238)
    assert OBJECTIVES_SLICE == slice(238, 298)
    assert OBJECTIVE_SLICES == (
        slice(238, 253), slice(253, 268), slice(268, 283), slice(283, 298)
    )
    assert all(value.stop - value.start == OBS_OBJECTIVE_DIM for value in OBJECTIVE_SLICES)
    assert len(POLICY_FEATURE_NAMES) == len(set(POLICY_FEATURE_NAMES)) == OBS_DIM

    observation = _observation()
    assert observation is not None
    assert observation.factual_vector()[-3:].tolist() == [1.0, 0.0, 1.0]
    spatial = SpatialPolicyFeatures(
        dyn=np.arange(24, dtype=np.float32),
        recovery=np.arange(16, dtype=np.float32) + 100,
        objectives=np.arange(60, dtype=np.float32).reshape(4, 15) + 200,
    )
    vector = observation.to_vector(spatial)
    assert vector.shape == (298,)
    assert np.array_equal(vector[DYN_SLICE], spatial.dyn)
    assert np.array_equal(vector[RECOVERY_SLICE], spatial.recovery)
    assert np.array_equal(vector[OBJECTIVES_SLICE], spatial.objectives.reshape(-1))
    with pytest.raises(TypeError):
        observation.to_vector(None)
    with pytest.raises(ValueError, match="exact|shape"):
        observation.to_vector(SpatialPolicyFeatures(
            dyn=np.zeros(23), recovery=np.zeros(16), objectives=np.zeros((4, 15))
        ))
    with pytest.raises(ValueError, match="NaN|infinity"):
        observation.to_vector(SpatialPolicyFeatures(
            dyn=np.zeros(24), recovery=np.full(16, np.nan), objectives=np.zeros((4, 15))
        ))


@pytest.mark.parametrize("vertical", list(VerticalIntent))
def test_three_way_vertical_action_has_one_frozen_byte(vertical):
    encoded = pack_action(Action(
        move_forward=0.5,
        move_right=-0.25,
        look_yaw=12.0,
        look_pitch=-4.0,
        vertical_intent=vertical,
        fire=True,
        hook=3,
        weapon=9,
    ), 77)
    assert len(encoded) == ACT_SIZE == 28
    unpacked = struct.unpack("<IIffff4B", encoded)
    assert unpacked[:2] == (ML_ACT_MAGIC, 77)
    assert unpacked[-4:] == (int(vertical), 1, 3, 9)


def test_action_and_observation_reject_legacy_or_malformed_generations():
    assert parse_obs(bytes(1032)) is None
    values = list(struct.unpack(OBS_FMT, bytes(OBS_SIZE)))
    values[0] = 0x514D4C50
    assert parse_obs(struct.pack(OBS_FMT, *values)) is None
    for invalid in (-1, 3, 255):
        with pytest.raises(ValueError, match="vertical_intent"):
            pack_action(Action(vertical_intent=invalid), 1)
    with pytest.raises(ValueError, match="binary"):
        pack_action(Action(fire=2), 1)
    with pytest.raises(ValueError, match="exact integers"):
        decode_policy_action([0, 0, 0, 0, 1.5, 0, 0, 0])
    with pytest.raises(ValueError, match="cardinality"):
        decode_policy_action([0, 0, 0, 0, 3, 0, 0, 0])


def test_rust_and_python_spatial_names_and_digest_are_exact():
    source = ROOT / "crates" / "q2-lattice" / "src"
    rust_names = (
        tuple(re.findall(r'"(dyn_[^"]+)"', (source / "dynstate.rs").read_text()))
        + tuple(re.findall(
            r'"(recovery_[^"]+)"', (source / "atlas" / "recovery.rs").read_text()
        ))
        + tuple(re.findall(
            r'"(guide_[0-3]_[^"]+)"', (source / "atlas" / "guide.rs").read_text()
        ))
    )
    python_names = (
        DYN_FEATURE_NAMES + RECOVERY_FEATURE_NAMES + OBJECTIVE_FEATURE_NAMES
    )
    assert len(rust_names) == len(set(rust_names)) == 100
    assert rust_names == python_names
    assert _json_name_digest(rust_names) == SPATIAL_FEATURE_SCHEMA_SHA256


def test_c_server_and_client_sources_match_python_wire_generation(tmp_path):
    cc = shutil.which("cc")
    lithium = ROOT.parent / "q2-lithium-3zb2"
    client = ROOT.parent / "q2-ml-client" / "src" / "client" / "cl_ml_harness.c"
    if not cc or not (lithium / "ml_bridge.h").is_file() or not client.is_file():
        pytest.skip("cross-repository B4 sources/compiler are not present")

    probe = tmp_path / "probe.c"
    probe.write_text(
        '#include <stddef.h>\n#include <stdio.h>\n'
        '#include "ml_bridge.h"\n#include "ml_client_wire.h"\n'
        'int main(void) { printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %u %u %u %u\\n", '
        'sizeof(ml_obs_t), sizeof(ml_action_t), sizeof(ml_action_debug_t), '
        'sizeof(ml_teacher_sample_t), sizeof(ml_client_register_t), '
        'sizeof(ml_client_ack_t), sizeof(ml_causal_telemetry_t), '
        'sizeof(ml_client_telemetry_t), offsetof(ml_obs_t, actual_ducked), '
        'offsetof(ml_obs_t, action_debug), ML_OBS_MAGIC, ML_ACT_MAGIC, '
        'ML_CLIENT_WIRE_VERSION, ML_TEACHER_VERSION); }\n',
        encoding="ascii",
    )
    executable = tmp_path / "probe"
    subprocess.run(
        [cc, "-std=c11", f"-I{lithium}", str(probe), "-o", str(executable)],
        check=True,
    )
    values = tuple(map(int, subprocess.check_output([executable], text=True).split()))
    assert values == (
        OBS_SIZE, ACT_SIZE, 60, 1224, 148, 100, 80, 1248, 836, 996,
        ML_OBS_MAGIC, ML_ACT_MAGIC, 8, 4,
    )

    client_source = client.read_text(encoding="utf-8")
    assert "#define ML_CLIENT_WIRE_VERSION 8u" in client_source
    assert "#define ML_OBSERVATION_MAGIC     0x514d324fu" in client_source
    assert "#define ML_ACTION_MAGIC          0x514d3241u" in client_source
    assert "#define ML_CLIENT_TELEMETRY_SIZE 1248u" in client_source
    assert "? 320 : latest_action.vertical_intent == ML_VERTICAL_DOWN_OR_CROUCH" in client_source
    assert "? -320 : 0" in client_source
