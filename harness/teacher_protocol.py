"""Wire format for passive 3ZB2 demonstration samples."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .causal_protocol import (
    CAUSAL_TELEMETRY_SIZE,
    CausalFlags,
    CausalTelemetry,
    parse_causal_telemetry,
)
from .protocol import (
    ACT_FMT,
    ACT_SIZE,
    ML_ACT_MAGIC,
    ML_CONTROL_LEGACY_BOT,
    OBS_SIZE,
    Action,
    Observation,
    parse_obs,
)

ML_TEACHER_MAGIC = 0x5154345A
ML_TEACHER_VERSION = 4
TEACHER_HEADER_FMT = "<7I32s"
TEACHER_HEADER_SIZE = struct.calcsize(TEACHER_HEADER_FMT)
TEACHER_PACKET_SIZE = (
    TEACHER_HEADER_SIZE + OBS_SIZE + CAUSAL_TELEMETRY_SIZE + ACT_SIZE
)


@dataclass(frozen=True)
class TeacherSample:
    sequence: int
    tick: int
    bot_slot: int
    grounded: bool
    map_name: str
    observation: Observation
    causal: CausalTelemetry
    action: np.ndarray

    @property
    def hook_was_necessary(self) -> bool:
        return self.causal.hook_was_necessary


def parse_teacher_sample(data: bytes) -> TeacherSample | None:
    if len(data) != TEACHER_PACKET_SIZE:
        return None
    header = struct.unpack_from(TEACHER_HEADER_FMT, data)
    magic, version, packet_size, sequence, tick, slot, flags, map_raw = header
    if (
        magic != ML_TEACHER_MAGIC
        or version != ML_TEACHER_VERSION
        or packet_size != TEACHER_PACKET_SIZE
        or flags & ~1
    ):
        return None
    try:
        map_name = map_raw.split(b"\0", 1)[0].decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    if not map_name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in map_name):
        return None
    obs_start = TEACHER_HEADER_SIZE
    observation = parse_obs(data[obs_start:obs_start + OBS_SIZE])
    if observation is None:
        return None
    causal_start = obs_start + OBS_SIZE
    causal_end = causal_start + CAUSAL_TELEMETRY_SIZE
    causal = parse_causal_telemetry(
        data[causal_start:causal_end], expected_tick=tick,
        require_action_generation=False,
    )
    # Passive samples are admitted only when their actual action and every
    # required causal fact are authoritative. Unknown hook necessity is not a
    # legacy false label; it makes that demonstration non-admissible.
    if causal is None or not causal.transition_trainable:
        return None
    action_values = struct.unpack_from(
        ACT_FMT, data, causal_end
    )
    action_magic, action_tick = action_values[:2]
    if (
        action_magic != ML_ACT_MAGIC
        or action_tick != tick
        or observation.tick != tick
        or observation.bot_slot != slot
        or int(observation.self_debug[2]) != ML_CONTROL_LEGACY_BOT
    ):
        return None
    action = np.asarray(action_values[2:], dtype=np.float32)
    if action.shape != (8,) or not np.isfinite(action).all():
        return None
    if not (
        np.all(np.abs(action[:2]) <= 1.0001)
        and abs(float(action[2])) <= 45.0001
        and abs(float(action[3])) <= 30.0001
        and action[4] in (0, 1, 2)
        and 0 <= action[5] <= 1
        and 0 <= action[6] <= 3
        and 0 <= action[7] <= 9
    ):
        return None
    if (
        causal.has(CausalFlags.HOOK_ATTEMPTED)
        and causal.hook_attempt_tick == tick
        and int(action[6]) != 1
    ):
        return None
    return TeacherSample(
        sequence=int(sequence),
        tick=int(tick),
        bot_slot=int(slot),
        grounded=bool(flags & 1),
        map_name=map_name,
        causal=causal,
        observation=observation,
        action=action,
    )
