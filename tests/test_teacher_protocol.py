import struct

from harness.protocol import (
    ACT_FMT,
    ML_ACT_MAGIC,
    ML_CONTROL_LEGACY_BOT,
    ML_OBS_MAGIC,
    OBS_FMT,
    OBS_SIZE,
)
from harness.teacher_protocol import (
    ML_TEACHER_MAGIC,
    ML_TEACHER_VERSION,
    TEACHER_HEADER_FMT,
    TEACHER_PACKET_SIZE,
    parse_teacher_sample,
)


def test_target_solution_teacher_semantics_are_versioned():
    assert ML_TEACHER_VERSION == 2


def _packet(*, source=ML_CONTROL_LEGACY_BOT, yaw=12.0):
    values = list(struct.unpack(OBS_FMT, bytes(OBS_SIZE)))
    values[0:3] = [ML_OBS_MAGIC, 77, 2]
    values[215] = source
    obs = struct.pack(OBS_FMT, *values)
    action = struct.pack(
        ACT_FMT, ML_ACT_MAGIC, 77,
        0.5, -0.25, yaw, -4.0, 1, 1, 0, 7,
    )
    header = struct.pack(
        TEACHER_HEADER_FMT,
        ML_TEACHER_MAGIC,
        ML_TEACHER_VERSION,
        TEACHER_PACKET_SIZE,
        9,
        77,
        2,
        1,
        b"mllive_12345678",
    )
    return header + obs + action


def test_teacher_packet_round_trip():
    sample = parse_teacher_sample(_packet())

    assert sample is not None
    assert sample.sequence == 9
    assert sample.tick == 77
    assert sample.bot_slot == 2
    assert sample.grounded
    assert sample.map_name == "mllive_12345678"
    assert sample.action.tolist() == [0.5, -0.25, 12.0, -4.0, 1.0, 1.0, 0.0, 7.0]


def test_teacher_packet_rejects_nonlegacy_source_and_bad_action():
    assert parse_teacher_sample(_packet(source=0)) is None
    assert parse_teacher_sample(_packet(yaw=90.0)) is None


def test_teacher_packet_size_stays_below_tailscale_mtu():
    assert TEACHER_PACKET_SIZE == 1120
    assert TEACHER_PACKET_SIZE < 1280
