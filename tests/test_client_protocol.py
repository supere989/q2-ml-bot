import struct

from harness.client_protocol import (
    CLIENT_TELEMETRY_HEADER_FMT,
    CLIENT_TELEMETRY_SIZE,
    ML_CLIENT_TELEM_MAGIC,
    ML_CLIENT_WIRE_VERSION,
    parse_client_telemetry,
)
from harness.protocol import OBS_FMT, OBS_SIZE, ML_OBS_MAGIC


def test_target_solution_wire_semantics_are_versioned():
    assert ML_CLIENT_WIRE_VERSION == 2
    assert ML_OBS_MAGIC == 0x514D4C50


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
        encoded_id,
        encoded_map,
    )
    return header + obs


def test_parses_matching_client_envelope():
    parsed = parse_client_telemetry(_packet())
    assert parsed is not None
    assert parsed.client_id == "client-a"
    assert parsed.client_slot == 2
    assert parsed.server_frame == 77
    assert parsed.map_name == "q2dm1"
    assert parsed.observation.tick == 77


def test_rejects_cross_slot_observation():
    packet = bytearray(_packet())
    header_size = struct.calcsize(CLIENT_TELEMETRY_HEADER_FMT)
    struct.pack_into("<I", packet, header_size + 8, 3)
    assert parse_client_telemetry(bytes(packet)) is None
