"""Wire framing for network-native clients with privileged observations.

The embedded Yamagi client is the router: it connects to q2ded through the
normal protocol-34 netchan, receives only its authenticated client_id stream
from game.so, and forwards that stream to this loopback endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Optional

from .causal_protocol import (
    CAUSAL_TELEMETRY_SIZE,
    CausalTelemetry,
    parse_causal_telemetry,
)
from .protocol import OBS_SIZE, Observation, parse_obs

ML_CLIENT_WIRE_VERSION = 8
ML_CLIENT_FRAME_BARRIER_VERSION = 1
ML_CLIENT_FRAME_BARRIER_CAPABILITY = 0x00000001
ML_CLIENT_TELEM_MAGIC = 0x54434D51
ML_CLIENT_ID_SIZE = 40

# Passive demonstration samples are a physically different datagram emitted
# by game.so to the teacher receiver.  They must never be treated as merely a
# malformed public packet: seeing this identity on a public harness socket is
# a privilege-boundary failure and terminates collection.
ML_TEACHER_MAGIC = 0x5154345A
ML_TEACHER_VERSION = 4
TEACHER_SAMPLE_SIZE = 1224

CLIENT_TELEMETRY_HEADER_FMT = "<10I40s32s"
CLIENT_TELEMETRY_HEADER_SIZE = struct.calcsize(CLIENT_TELEMETRY_HEADER_FMT)
CLIENT_TELEMETRY_SIZE = (
    CLIENT_TELEMETRY_HEADER_SIZE + OBS_SIZE + CAUSAL_TELEMETRY_SIZE
)


class PublicTelemetryPrivilegeViolation(RuntimeError):
    """Teacher-formatted bytes were delivered to the public policy conduit."""


def _reject_teacher_datagram(data: bytes) -> None:
    """Fail closed on the teacher packet identity before public parsing.

    Magic is sufficient to make this fatal.  Version and size are included in
    the error only as bounded diagnostics; a truncated or future teacher packet
    must not evade the privilege fence by failing its own schema validation.
    """
    if len(data) < 4:
        return
    magic = struct.unpack_from("<I", data)[0]
    if magic != ML_TEACHER_MAGIC:
        return
    version = struct.unpack_from("<I", data, 4)[0] if len(data) >= 8 else None
    packet_size = struct.unpack_from("<I", data, 8)[0] if len(data) >= 12 else None
    raise PublicTelemetryPrivilegeViolation(
        "teacher datagram on public conduit "
        f"(version={version!r}, declared_size={packet_size!r}, bytes={len(data)})"
    )


@dataclass(frozen=True)
class ClientTelemetry:
    sequence: int
    client_slot: int
    server_frame: int
    client_id: str
    map_name: str
    observation: Observation
    causal: CausalTelemetry
    barrier_version: int = ML_CLIENT_FRAME_BARRIER_VERSION
    barrier_capabilities: int = ML_CLIENT_FRAME_BARRIER_CAPABILITY
    map_epoch: int = 1
    applied_action_tick: int = 0


def parse_client_telemetry(data: bytes) -> Optional[ClientTelemetry]:
    _reject_teacher_datagram(data)
    if len(data) != CLIENT_TELEMETRY_SIZE:
        return None
    (magic, version, packet_size, sequence, client_slot, server_frame,
     barrier_version, barrier_capabilities, map_epoch, applied_action_tick,
     raw_id, raw_map) = (
        struct.unpack_from(CLIENT_TELEMETRY_HEADER_FMT, data)
    )
    if (
        magic != ML_CLIENT_TELEM_MAGIC
        or version != ML_CLIENT_WIRE_VERSION
        or packet_size != len(data)
        or barrier_version != ML_CLIENT_FRAME_BARRIER_VERSION
        or not (barrier_capabilities & ML_CLIENT_FRAME_BARRIER_CAPABILITY)
        or map_epoch <= 0
        or applied_action_tick > server_frame
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
    obs_start = CLIENT_TELEMETRY_HEADER_SIZE
    obs_end = obs_start + OBS_SIZE
    observation = parse_obs(data[obs_start:obs_end])
    if observation is None:
        return None
    if observation.tick != server_frame or observation.bot_slot != client_slot:
        return None
    causal = parse_causal_telemetry(
        data[obs_end:], expected_tick=server_frame,
        require_action_generation=True,
    )
    if causal is None:
        return None
    return ClientTelemetry(
        sequence=sequence,
        client_slot=client_slot,
        server_frame=server_frame,
        client_id=client_id,
        map_name=map_name,
        observation=observation,
        causal=causal,
        barrier_version=barrier_version,
        barrier_capabilities=barrier_capabilities,
        map_epoch=map_epoch,
        applied_action_tick=applied_action_tick,
    )
