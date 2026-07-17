import struct

from harness.causal_protocol import (
    CAUSAL_TELEMETRY_FMT,
    CAUSAL_TELEMETRY_SIZE,
    ML_CAUSAL_MAGIC,
    ML_CAUSAL_VERSION,
    CausalFlags,
)
from harness.client_protocol import (
    CLIENT_TELEMETRY_HEADER_FMT,
    CLIENT_TELEMETRY_SIZE,
    ML_CLIENT_TELEM_MAGIC,
    ML_CLIENT_WIRE_VERSION,
    parse_client_telemetry,
)
from harness.protocol import OBS_FMT, OBS_SIZE, ML_OBS_MAGIC


def test_target_solution_wire_semantics_are_versioned():
    assert ML_CLIENT_WIRE_VERSION == 8
    assert ML_OBS_MAGIC == 0x514D324F


def _packet(client_id="client-a", slot=2, frame=77, sequence=4):
    obs_values = [0] * len(struct.unpack(OBS_FMT, bytes(OBS_SIZE)))
    obs_values[0] = ML_OBS_MAGIC
    obs_values[1] = frame
    obs_values[2] = slot
    obs = struct.pack(OBS_FMT, *obs_values)
    encoded_id = client_id.encode().ljust(40, b"\0")
    encoded_map = b"q2dm1".ljust(32, b"\0")
    header = struct.pack(
        CLIENT_TELEMETRY_HEADER_FMT,
        ML_CLIENT_TELEM_MAGIC,
        ML_CLIENT_WIRE_VERSION,
        CLIENT_TELEMETRY_SIZE,
        sequence,
        slot,
        frame,
        1,
        1,
        3,
        frame - 1,
        encoded_id,
        encoded_map,
    )
    causal = struct.pack(
        CAUSAL_TELEMETRY_FMT,
        ML_CAUSAL_MAGIC, ML_CAUSAL_VERSION, CAUSAL_TELEMETRY_SIZE,
        int(CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
            | CausalFlags.TRANSITION_TRAINABLE
            | CausalFlags.ROLE_PLAYING
            | CausalFlags.ROLE_PUBLIC_PM_NORMAL),
        frame, 3, 0, 0, 0, 0, 0, 0, 0, 0, frame - 1, 5, 0, 0, 0, 0,
    )
    return header + obs + causal


def test_parses_matching_client_envelope():
    parsed = parse_client_telemetry(_packet())
    assert parsed is not None
    assert parsed.client_id == "client-a"
    assert parsed.client_slot == 2
    assert parsed.server_frame == 77
    assert parsed.map_name == "q2dm1"
    assert parsed.observation.tick == 77
    assert parsed.causal.client_life_epoch == 3
    assert parsed.causal.transition_trainable


def test_rejects_cross_slot_observation():
    packet = bytearray(_packet())
    header_size = struct.calcsize(CLIENT_TELEMETRY_HEADER_FMT)
    struct.pack_into("<I", packet, header_size + 8, 3)
    assert parse_client_telemetry(bytes(packet)) is None


def test_rejects_legacy_wire_version_and_observation_magic():
    legacy_wire = bytearray(_packet())
    struct.pack_into("<I", legacy_wire, 4, 5)
    assert parse_client_telemetry(bytes(legacy_wire)) is None

    legacy_obs = bytearray(_packet())
    header_size = struct.calcsize(CLIENT_TELEMETRY_HEADER_FMT)
    struct.pack_into("<I", legacy_obs, header_size, 0x514D4C50)
    assert parse_client_telemetry(bytes(legacy_obs)) is None


def test_rejects_missing_or_malformed_private_causal_tail():
    packet = _packet()
    assert parse_client_telemetry(packet[:-CAUSAL_TELEMETRY_SIZE]) is None
    malformed = bytearray(packet)
    causal_offset = len(packet) - CAUSAL_TELEMETRY_SIZE
    struct.pack_into("<I", malformed, causal_offset, 0)
    assert parse_client_telemetry(bytes(malformed)) is None
