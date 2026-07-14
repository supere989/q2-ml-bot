"""Wire framing for network-native clients with privileged observations.

The embedded Yamagi client is the router: it connects to q2ded through the
normal protocol-34 netchan, receives only its authenticated client_id stream
from game.so, and forwards that stream to this loopback endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Optional

from .protocol import OBS_SIZE, Observation, parse_obs

ML_CLIENT_WIRE_VERSION = 4
ML_CLIENT_TELEM_MAGIC = 0x54434D51
ML_CLIENT_ID_SIZE = 40

CLIENT_TELEMETRY_HEADER_FMT = "<6I40s32s"
CLIENT_TELEMETRY_HEADER_SIZE = struct.calcsize(CLIENT_TELEMETRY_HEADER_FMT)
CLIENT_TELEMETRY_SIZE = CLIENT_TELEMETRY_HEADER_SIZE + OBS_SIZE


@dataclass(frozen=True)
class ClientTelemetry:
    sequence: int
    client_slot: int
    server_frame: int
    client_id: str
    map_name: str
    observation: Observation


def parse_client_telemetry(data: bytes) -> Optional[ClientTelemetry]:
    if len(data) != CLIENT_TELEMETRY_SIZE:
        return None
    (magic, version, packet_size, sequence, client_slot, server_frame,
     raw_id, raw_map) = (
        struct.unpack_from(CLIENT_TELEMETRY_HEADER_FMT, data)
    )
    if (
        magic != ML_CLIENT_TELEM_MAGIC
        or version != ML_CLIENT_WIRE_VERSION
        or packet_size != len(data)
    ):
        return None
    raw_id = raw_id.split(b"\0", 1)[0]
    try:
        client_id = raw_id.decode("ascii")
    except UnicodeDecodeError:
        return None
    if not client_id:
        return None
    try:
        map_name = raw_map.split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not map_name:
        return None
    observation = parse_obs(data[CLIENT_TELEMETRY_HEADER_SIZE:])
    if observation is None:
        return None
    if observation.tick != server_frame or observation.bot_slot != client_slot:
        return None
    return ClientTelemetry(
        sequence=sequence,
        client_slot=client_slot,
        server_frame=server_frame,
        client_id=client_id,
        map_name=map_name,
        observation=observation,
    )
