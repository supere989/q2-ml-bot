import struct

from harness.causal_protocol import (
    CAUSAL_TELEMETRY_FMT,
    CAUSAL_TELEMETRY_SIZE,
    ML_CAUSAL_MAGIC,
    ML_CAUSAL_VERSION,
    CausalFlags,
)
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
    assert ML_TEACHER_VERSION == 4


def _packet(*, source=ML_CONTROL_LEGACY_BOT, yaw=12.0):
    values = list(struct.unpack(OBS_FMT, bytes(OBS_SIZE)))
    values[0:3] = [ML_OBS_MAGIC, 77, 2]
    values[218] = source
    obs = struct.pack(OBS_FMT, *values)
    action = struct.pack(
        ACT_FMT, ML_ACT_MAGIC, 77,
        0.5, -0.25, yaw, -4.0, 1, 1, 1, 7,
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
    causal = struct.pack(
        CAUSAL_TELEMETRY_FMT,
        ML_CAUSAL_MAGIC, ML_CAUSAL_VERSION, CAUSAL_TELEMETRY_SIZE,
        int(CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_ATTACHED
            | CausalFlags.HOOK_VALID | CausalFlags.HOOK_NECESSITY_KNOWN
            | CausalFlags.HOOK_WAS_NECESSARY | CausalFlags.ECHO_VALID
            | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE),
        77, 6, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 2, 77, 0, 0,
    )
    return header + obs + causal + action


def test_teacher_packet_round_trip():
    sample = parse_teacher_sample(_packet())

    assert sample is not None
    assert sample.sequence == 9
    assert sample.tick == 77
    assert sample.bot_slot == 2
    assert sample.grounded
    assert sample.map_name == "mllive_12345678"
    assert sample.hook_was_necessary
    assert sample.action.tolist() == [0.5, -0.25, 12.0, -4.0, 1.0, 1.0, 1.0, 7.0]


def test_teacher_packet_rejects_nonlegacy_source_and_bad_action():
    assert parse_teacher_sample(_packet(source=0)) is None
    assert parse_teacher_sample(_packet(yaw=90.0)) is None


def test_teacher_packet_size_stays_below_tailscale_mtu():
    assert TEACHER_PACKET_SIZE == 1224
    assert TEACHER_PACKET_SIZE < 1280


def test_teacher_causal_bytes_are_not_part_of_public_observation():
    packet = _packet()
    causal_offset = struct.calcsize(TEACHER_HEADER_FMT) + OBS_SIZE
    magic = struct.unpack_from("<I", packet, causal_offset)[0]
    assert magic == ML_CAUSAL_MAGIC
    assert parse_teacher_sample(packet[:causal_offset]) is None


def test_teacher_unresolved_hook_necessity_is_admissible_without_label():
    packet = bytearray(_packet())
    causal_offset = struct.calcsize(TEACHER_HEADER_FMT) + OBS_SIZE
    incomplete = int(
        CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_ATTACHED
        | CausalFlags.HOOK_VALID | CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    )
    struct.pack_into("<I", packet, causal_offset + 12, incomplete)
    sample = parse_teacher_sample(bytes(packet))
    assert sample is not None
    assert not sample.hook_was_necessary


def test_teacher_delayed_hook_outcome_does_not_require_current_fire_action():
    packet = bytearray(_packet())
    causal_offset = struct.calcsize(TEACHER_HEADER_FMT) + OBS_SIZE
    struct.pack_into("<I", packet, causal_offset + 68, 72)
    action_offset = causal_offset + CAUSAL_TELEMETRY_SIZE
    packet[action_offset + 26] = 2  # current action holds; origin fired at tick 72
    sample = parse_teacher_sample(bytes(packet))
    assert sample is not None
    assert sample.causal.hook_attempt_tick == 72
    assert sample.action[6] == 2
