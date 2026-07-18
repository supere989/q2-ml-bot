"""Strict private causal/debug wire contract for multires training.

The block is physically separate from :class:`Observation`.  It is suitable
for event attribution and transition admission, never policy composition.
``environmental_source_id`` names C damage-source provenance; Atlas owns the
separate pre-damage reward hazard-component identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import struct


ML_CAUSAL_MAGIC = 0x514D3343
ML_CAUSAL_VERSION = 2
CAUSAL_TELEMETRY_FMT = "<20I"
CAUSAL_TELEMETRY_SIZE = struct.calcsize(CAUSAL_TELEMETRY_FMT)
CAUSAL_FIELD_NAMES = (
    "magic",
    "version",
    "packet_size",
    "flags",
    "tick",
    "client_life_epoch",
    "target_id",
    "target_epoch",
    "environmental_source_id",
    "environmental_source_epoch",
    "environmental_mod",
    "environmental_damage",
    "crouch_edge_id",
    "crouch_edge_epoch",
    "echo_tick",
    "action_generation",
    "hook_zone_id",
    "hook_attempt_tick",
    "hook_action_generation",
    "reserved",
)
ML_ACTION_GENERATION_COUNT = 192


class CausalFlags(IntFlag):
    TARGET_VALID = 1 << 0
    ENV_SOURCE_ACTIVE = 1 << 1
    ENV_SOURCE_EVIDENCE = 1 << 2
    ENV_DAMAGE = 1 << 3
    ENV_DEATH = 1 << 4
    ENV_SOURCE_CLEARED = 1 << 5
    CROUCH_EDGE_ACTIVE = 1 << 6
    CROUCH_EDGE_ENTERED = 1 << 7
    CROUCH_EDGE_COMPLETED = 1 << 8
    HOOK_ATTEMPTED = 1 << 9
    HOOK_ATTACHED = 1 << 10
    HOOK_VALID = 1 << 11
    HOOK_NECESSITY_KNOWN = 1 << 12
    HOOK_WAS_NECESSARY = 1 << 13
    ECHO_VALID = 1 << 14
    FACTS_COMPLETE = 1 << 15
    TRANSITION_TRAINABLE = 1 << 16
    TARGET_HIT = 1 << 17
    TARGET_KILLED = 1 << 18
    HOOK_INVALID = 1 << 19
    ROLE_PLAYING = 1 << 20
    ROLE_PUBLIC_PM_NORMAL = 1 << 21


CAUSAL_FLAGS_MASK = (1 << 22) - 1


@dataclass(frozen=True)
class CausalTelemetry:
    tick: int
    client_life_epoch: int
    target_id: int
    target_epoch: int
    environmental_source_id: int
    environmental_source_epoch: int
    environmental_mod: int
    environmental_damage: int
    crouch_edge_id: int
    crouch_edge_epoch: int
    echo_tick: int
    action_generation: int
    hook_zone_id: int
    hook_attempt_tick: int
    hook_action_generation: int
    flags: CausalFlags

    def has(self, flag: CausalFlags) -> bool:
        return bool(self.flags & flag)

    @property
    def hook_was_necessary(self) -> bool:
        return self.has(CausalFlags.HOOK_WAS_NECESSARY)

    @property
    def echo_valid(self) -> bool:
        return self.has(CausalFlags.ECHO_VALID)

    @property
    def facts_complete(self) -> bool:
        return self.has(CausalFlags.FACTS_COMPLETE)

    @property
    def transition_trainable(self) -> bool:
        return self.has(CausalFlags.TRANSITION_TRAINABLE)

    @property
    def role_playing(self) -> bool:
        return self.has(CausalFlags.ROLE_PLAYING)

    @property
    def role_public_pm_normal(self) -> bool:
        return self.has(CausalFlags.ROLE_PUBLIC_PM_NORMAL)


def parse_causal_telemetry(
    data: bytes,
    *,
    expected_tick: int,
    require_action_generation: bool,
) -> CausalTelemetry | None:
    """Parse one complete causal block, rejecting inconsistent provenance.

    ``require_action_generation`` is true on protocol-34 client telemetry.
    Passive 3ZB2 demonstrations have a directly observed action instead and
    therefore require generation zero.
    """

    if len(data) != CAUSAL_TELEMETRY_SIZE:
        return None
    values = struct.unpack(CAUSAL_TELEMETRY_FMT, data)
    (
        magic, version, packet_size, raw_flags, tick, client_life_epoch,
        target_id, target_epoch, environmental_source_id,
        environmental_source_epoch,
        environmental_mod, environmental_damage, crouch_edge_id,
        crouch_edge_epoch, echo_tick, action_generation, hook_zone_id,
        hook_attempt_tick, hook_action_generation, reserved,
    ) = values
    if (
        magic != ML_CAUSAL_MAGIC
        or version != ML_CAUSAL_VERSION
        or packet_size != CAUSAL_TELEMETRY_SIZE
        or tick != expected_tick
        or client_life_epoch == 0
        or raw_flags & ~CAUSAL_FLAGS_MASK
        or reserved
    ):
        return None
    flags = CausalFlags(raw_flags)

    # These role facts are private game authority.  Every routed protocol-34
    # packet must belong to a real Lithium player; passive teacher samples do
    # not use this client-route precondition.  Public-normal is the exact
    # cross-protocol fence used by the client and can never stand alone.
    if (
        flags & CausalFlags.ROLE_PUBLIC_PM_NORMAL
        and not flags & CausalFlags.ROLE_PLAYING
    ):
        return None
    if require_action_generation and not flags & CausalFlags.ROLE_PLAYING:
        return None

    target_valid = bool(flags & CausalFlags.TARGET_VALID)
    if target_valid != bool(target_id and target_epoch):
        return None
    if bool(target_id) != bool(target_epoch):
        return None
    if flags & (CausalFlags.TARGET_HIT | CausalFlags.TARGET_KILLED) and not target_valid:
        return None
    if flags & CausalFlags.TARGET_KILLED and not flags & CausalFlags.TARGET_HIT:
        return None

    environmental_flags = flags & (
        CausalFlags.ENV_SOURCE_ACTIVE
        | CausalFlags.ENV_SOURCE_EVIDENCE
        | CausalFlags.ENV_DAMAGE
        | CausalFlags.ENV_DEATH
        | CausalFlags.ENV_SOURCE_CLEARED
    )
    if environmental_flags and (
        not environmental_source_id or not environmental_source_epoch
    ):
        return None
    if bool(environmental_source_id) != bool(environmental_source_epoch):
        return None
    if bool(environmental_source_id) != bool(environmental_mod):
        return None
    if flags & (
        CausalFlags.ENV_SOURCE_EVIDENCE
        | CausalFlags.ENV_DAMAGE
        | CausalFlags.ENV_DEATH
        | CausalFlags.ENV_SOURCE_CLEARED
    ) and not environmental_mod:
        return None
    if bool(flags & CausalFlags.ENV_DAMAGE) != bool(environmental_damage):
        return None
    if (
        flags & CausalFlags.ENV_SOURCE_ACTIVE
        and not flags & CausalFlags.ENV_SOURCE_EVIDENCE
    ):
        return None
    if (
        flags & CausalFlags.ENV_DEATH
        and not flags & CausalFlags.ENV_SOURCE_EVIDENCE
    ):
        return None
    if flags & CausalFlags.ENV_DEATH and not flags & CausalFlags.ENV_DAMAGE:
        return None
    if (
        flags & CausalFlags.ENV_SOURCE_CLEARED
        and not flags & CausalFlags.ENV_SOURCE_EVIDENCE
    ):
        return None
    if (
        flags & CausalFlags.ENV_SOURCE_CLEARED
        and flags & CausalFlags.ENV_SOURCE_ACTIVE
    ):
        return None
    if flags & CausalFlags.ENV_SOURCE_CLEARED and flags & (
        CausalFlags.ENV_DAMAGE | CausalFlags.ENV_DEATH
    ):
        return None

    crouch_flags = flags & (
        CausalFlags.CROUCH_EDGE_ACTIVE
        | CausalFlags.CROUCH_EDGE_ENTERED
        | CausalFlags.CROUCH_EDGE_COMPLETED
    )
    if crouch_flags and (not crouch_edge_id or not crouch_edge_epoch):
        return None
    if (
        flags & CausalFlags.CROUCH_EDGE_ENTERED
        and not flags & CausalFlags.CROUCH_EDGE_ACTIVE
    ):
        return None
    if (
        flags & CausalFlags.CROUCH_EDGE_COMPLETED
        and flags & CausalFlags.CROUCH_EDGE_ACTIVE
    ):
        return None

    if flags & CausalFlags.HOOK_ATTACHED and not flags & CausalFlags.HOOK_VALID:
        return None
    if flags & CausalFlags.HOOK_VALID and not flags & CausalFlags.HOOK_ATTACHED:
        return None
    if flags & CausalFlags.HOOK_INVALID and flags & (
        CausalFlags.HOOK_ATTACHED
        | CausalFlags.HOOK_VALID
        | CausalFlags.HOOK_WAS_NECESSARY
    ):
        return None
    if flags & CausalFlags.HOOK_WAS_NECESSARY and not (
        flags & CausalFlags.HOOK_NECESSITY_KNOWN
        and flags & CausalFlags.HOOK_VALID
        and hook_zone_id
    ):
        return None
    if (
        flags & CausalFlags.HOOK_VALID
        and flags & CausalFlags.HOOK_NECESSITY_KNOWN
        and not hook_zone_id
    ):
        return None
    if (
        flags & CausalFlags.HOOK_INVALID
        and not flags & CausalFlags.HOOK_NECESSITY_KNOWN
    ):
        return None
    if flags & CausalFlags.HOOK_NECESSITY_KNOWN and not flags & (
        CausalFlags.HOOK_ATTACHED | CausalFlags.HOOK_INVALID
    ):
        return None
    hook_flags = flags & (
        CausalFlags.HOOK_ATTEMPTED
        | CausalFlags.HOOK_ATTACHED
        | CausalFlags.HOOK_VALID
        | CausalFlags.HOOK_NECESSITY_KNOWN
        | CausalFlags.HOOK_WAS_NECESSARY
        | CausalFlags.HOOK_INVALID
    )
    if hook_flags:
        if not flags & CausalFlags.HOOK_ATTEMPTED:
            return None
        if not hook_attempt_tick or hook_attempt_tick > tick:
            return None
        if require_action_generation:
            if not 1 <= hook_action_generation <= ML_ACTION_GENERATION_COUNT:
                return None
        elif hook_action_generation:
            return None
    elif hook_attempt_tick or hook_action_generation:
        return None

    echo_valid = bool(flags & CausalFlags.ECHO_VALID)
    facts_complete = bool(flags & CausalFlags.FACTS_COMPLETE)
    trainable = bool(flags & CausalFlags.TRANSITION_TRAINABLE)
    if echo_valid and not echo_tick:
        return None
    # Trainability is a stricter admission claim, not an alias for complete
    # attribution.  Stock engine-owned lifecycle transitions (for example,
    # PMF_TIME_TELEPORT settling) deliberately carry E=1/F=1/T=0 so the
    # lifecycle FSM can consume their exact action echo without admitting a
    # PPO transition.  The reverse combinations remain forged: T may only be
    # asserted when both its prerequisites are true.
    if trainable and not (echo_valid and facts_complete):
        return None
    if trainable and require_action_generation and not (
        flags & CausalFlags.ROLE_PUBLIC_PM_NORMAL
    ):
        return None
    if action_generation > ML_ACTION_GENERATION_COUNT:
        return None
    if require_action_generation:
        if echo_valid and not 1 <= action_generation <= ML_ACTION_GENERATION_COUNT:
            return None
    elif action_generation:
        return None
    if (
        require_action_generation
        and hook_flags
        and hook_attempt_tick == tick
        and hook_action_generation != action_generation
    ):
        return None

    return CausalTelemetry(
        tick=int(tick),
        client_life_epoch=int(client_life_epoch),
        target_id=int(target_id),
        target_epoch=int(target_epoch),
        environmental_source_id=int(environmental_source_id),
        environmental_source_epoch=int(environmental_source_epoch),
        environmental_mod=int(environmental_mod),
        environmental_damage=int(environmental_damage),
        crouch_edge_id=int(crouch_edge_id),
        crouch_edge_epoch=int(crouch_edge_epoch),
        echo_tick=int(echo_tick),
        action_generation=int(action_generation),
        hook_zone_id=int(hook_zone_id),
        hook_attempt_tick=int(hook_attempt_tick),
        hook_action_generation=int(hook_action_generation),
        flags=flags,
    )
