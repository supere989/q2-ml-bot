"""Synchronous batched collection for network-native Quake II clients.

Actions for every client are dispatched before collection begins.  A sample
is admitted only after game.so echoes a usercmd newer than that client's
action tick and the echoed movement/buttons match the requested action.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .client_env import (
    ClientActionDispatch,
    ClientTelemetryDrain,
    Q2NetworkClientEnv,
)
from .client_protocol import ClientTelemetry
from .multires_admission import SPATIAL_ATTESTATION_INFO_KEY
from .multires_admission import SpatialFeatureProviderFactory
from .protocol import (
    Action,
    ActionDebugIndex,
    ML_FIRE_GATE_PROTECTED,
    ML_FIRE_GATE_SUPPRESSED,
    ML_FIRE_GATE_TARGET,
    ML_ACTION_GENERATION_COUNT,
    ML_ACTION_GENERATION_MASK,
    ML_ACTION_GENERATION_SHIFT,
    ML_TERMINAL_DEATH,
    ML_TERMINAL_INTERMISSION,
    VerticalIntent,
)


class StalePolicyVersionError(ValueError):
    """Raised before dispatch when a caller attempts policy rollback."""


class AuthoritativeEchoError(RuntimeError):
    """Raised when one client cannot produce a valid echo for its action."""

    def __init__(
        self,
        message: str,
        *,
        stale_echoes: int = 0,
        mismatched_echoes: int = 0,
        timed_out: bool = False,
    ):
        super().__init__(message)
        self.stale_echoes = int(stale_echoes)
        self.mismatched_echoes = int(mismatched_echoes)
        self.timed_out = bool(timed_out)


@dataclass(frozen=True)
class BatchActionTag:
    """Provenance attached to one admitted transition."""

    round_id: int
    policy_version: int
    client_index: int
    client_id: str
    client_slot: int
    action_tick: int


@dataclass(frozen=True)
class BatchMetrics:
    """Cumulative collector counters suitable for logs/TensorBoard."""

    latest_policy_version: int | None
    rounds_dispatched: int
    rounds_accepted: int
    failed_rounds: int
    actions_dispatched: int
    transitions_accepted: int
    stale_policy_rounds_rejected: int
    stale_echoes_rejected: int
    mismatched_echoes_rejected: int
    echo_timeouts: int
    map_epoch_resyncs: int
    telemetry_gap_resyncs: int
    realtime_catchup_resyncs: int
    action_state_resyncs: int
    death_lifecycle_resyncs: int
    preflight_packets_drained: int
    max_observed_frame_span: int
    fire_gate_suppressions: int

    def as_dict(self, prefix: str = "network_client") -> dict[str, int | float]:
        attempted = self.transitions_accepted + self.stale_echoes_rejected + \
            self.mismatched_echoes_rejected
        return {
            f"{prefix}/rounds_dispatched": self.rounds_dispatched,
            f"{prefix}/rounds_accepted": self.rounds_accepted,
            f"{prefix}/failed_rounds": self.failed_rounds,
            f"{prefix}/actions_dispatched": self.actions_dispatched,
            f"{prefix}/transitions_accepted": self.transitions_accepted,
            f"{prefix}/stale_policy_rounds_rejected": (
                self.stale_policy_rounds_rejected
            ),
            f"{prefix}/stale_echoes_rejected": self.stale_echoes_rejected,
            f"{prefix}/mismatched_echoes_rejected": (
                self.mismatched_echoes_rejected
            ),
            f"{prefix}/echo_timeouts": self.echo_timeouts,
            f"{prefix}/map_epoch_resyncs": self.map_epoch_resyncs,
            f"{prefix}/telemetry_gap_resyncs": self.telemetry_gap_resyncs,
            f"{prefix}/realtime_catchup_resyncs": self.realtime_catchup_resyncs,
            f"{prefix}/action_state_resyncs": self.action_state_resyncs,
            f"{prefix}/death_lifecycle_resyncs": (
                self.death_lifecycle_resyncs
            ),
            f"{prefix}/preflight_packets_drained": self.preflight_packets_drained,
            f"{prefix}/max_observed_frame_span": self.max_observed_frame_span,
            f"{prefix}/fire_gate_suppressions": self.fire_gate_suppressions,
            f"{prefix}/authoritative_echo_accept_rate": (
                self.transitions_accepted / attempted if attempted else 1.0
            ),
        }


@dataclass(frozen=True)
class BatchRound:
    """One same-policy action round admitted by authoritative echo."""

    round_id: int
    policy_version: int
    observations: Any
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: tuple[dict[str, Any], ...]
    tags: tuple[BatchActionTag, ...]

    def gym_result(self):
        return (
            self.observations,
            self.rewards,
            self.terminated,
            self.truncated,
            list(self.infos),
        )


@dataclass(frozen=True)
class _MatchedEcho:
    telemetry: ClientTelemetry
    echo_tick: int
    stale_echoes: int
    mismatched_echoes: int


@dataclass(frozen=True)
class _MapEpoch:
    telemetry: ClientTelemetry
    stale_echoes: int
    mismatched_echoes: int


@dataclass(frozen=True)
class _ActionStateResync:
    """Causal action altered by a newer engine-owned view/lifecycle state."""

    telemetry: ClientTelemetry
    echo_tick: int
    stale_echoes: int
    mismatched_echoes: int
    phase: str = "action_state_resync"
    hold_life_epoch: int | None = None


@dataclass(frozen=True)
class _DeathLifecycleResync:
    """Exact action echo that belongs to a non-trainable death boundary."""

    telemetry: ClientTelemetry
    echo_tick: int
    stale_echoes: int
    mismatched_echoes: int
    previous_life_epoch: int
    next_life_epoch: int
    death_terminal: bool = False
    phase: str = "corpse"


@dataclass(frozen=True)
class _DeathLifecyclePending:
    """Per-client death boundary retained until one actionable prime."""

    previous_life_epoch: int
    new_life_seen: bool = False


class Q2NetworkClientBatch:
    """Own and synchronously collect N :class:`Q2NetworkClientEnv` objects.

    ``collect_round`` is the provenance-rich API.  ``step`` returns the usual
    vectorized Gym tuple while preserving the tags in every info dictionary.
    Policy versions must be monotonic; a lower version is rejected before any
    client action is sent.
    """

    def __init__(
        self,
        envs: Iterable[Q2NetworkClientEnv],
        *,
        vector: bool = True,
        round_timeout: float = 2.0,
        max_rejected_echoes: int = 16,
        movement_tolerance: float = 0.05,
        look_tolerance: float = 0.25,
        deterministic_frame_barrier: bool = False,
    ):
        self.envs = tuple(envs)
        if not self.envs:
            raise ValueError("at least one network client environment is required")
        if round_timeout <= 0:
            raise ValueError("round_timeout must be positive")
        if max_rejected_echoes < 0:
            raise ValueError("max_rejected_echoes cannot be negative")
        if movement_tolerance < 0 or not math.isfinite(movement_tolerance):
            raise ValueError("movement_tolerance must be finite and non-negative")
        if look_tolerance < 0 or not math.isfinite(look_tolerance):
            raise ValueError("look_tolerance must be finite and non-negative")
        client_ids = [env.client_id for env in self.envs]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("network client IDs must be unique within a batch")

        self.vector = bool(vector)
        self.round_timeout = float(round_timeout)
        self.max_rejected_echoes = int(max_rejected_echoes)
        self.movement_tolerance = float(movement_tolerance)
        self.look_tolerance = float(look_tolerance)
        self.deterministic_frame_barrier = bool(deterministic_frame_barrier)
        if self.deterministic_frame_barrier:
            if any(not getattr(env, "deterministic_frame_barrier", False)
                   for env in self.envs):
                raise ValueError("every client must enable deterministic_frame_barrier")
            self.max_rejected_echoes = 0
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.envs), thread_name_prefix="q2-client"
        )
        self._round_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._next_round_id = 0
        self._latest_policy_version: int | None = None
        self._rounds_dispatched = 0
        self._rounds_accepted = 0
        self._failed_rounds = 0
        self._actions_dispatched = 0
        self._transitions_accepted = 0
        self._stale_policy_rounds_rejected = 0
        self._stale_echoes_rejected = 0
        self._mismatched_echoes_rejected = 0
        self._echo_timeouts = 0
        self._map_epoch_resyncs = 0
        self._telemetry_gap_resyncs = 0
        self._realtime_catchup_resyncs = 0
        self._action_state_resyncs = 0
        self._death_lifecycle_resyncs = 0
        self._preflight_packets_drained = 0
        self._max_observed_frame_span = 0
        self._fire_gate_suppressions = 0
        # Generated-map downloads can outlive one action round.  Once a map
        # boundary is observed, do not dispatch again until every client is
        # on the same playable map after this source epoch.
        self._map_epoch_source: str | None = None
        # A generated-map download may silence every conduit before any client
        # emits the intermission/new-map packet that normally starts the map
        # barrier. Treat only a whole-batch timeout as this fail-closed gap;
        # partial timeouts remain fatal transport failures.
        self._telemetry_gap_pending = False
        self._telemetry_gap_resumed_clients: set[str] = set()
        # Death closes the current trajectory.  Deterministic clients still
        # submit commands while the corpse/respawn clock advances, but those
        # commands are not policy transitions.  The causal life epoch is the
        # sole authority for ending this boundary; visual health/dead flags
        # are deliberately not used as transport identity.
        self._client_life_epochs: dict[str, int] = {}
        self._death_life_epochs: dict[str, _DeathLifecyclePending] = {}
        # A stock spawn/teleporter hold may occur without a life-epoch change.
        # Only a public-normal causal nontrainable bit opens a same-life
        # boundary.  Intermission PM_FREEZE is also E=1/F=1/T=0, but its
        # ROLE_PUBLIC_PM_NORMAL=0 fence must never arm this state machine.
        # The hold closes only after one exact actionable command is discarded
        # as a prime, preventing reward or accumulated view state from
        # migrating into the first admitted policy transition.
        self._action_hold_epochs: dict[str, int] = {}

    @staticmethod
    def _policy_version(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("policy_version must be an integer")
        version = int(value)
        if version < 0:
            raise ValueError("policy_version cannot be negative")
        return version

    @staticmethod
    def _clamp_movement(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _echo_state(
        self,
        telemetry: ClientTelemetry,
        dispatch: ClientActionDispatch,
        *,
        allow_lifecycle_action_resync: bool = False,
    ) -> tuple[str, int]:
        debug = telemetry.observation.action_debug
        if len(debug) != len(ActionDebugIndex):
            return "mismatch", -1
        echo_tick = int(debug[ActionDebugIndex.TICK])
        accepted = int(debug[ActionDebugIndex.ACCEPTED]) == 1
        if self.deterministic_frame_barrier and (
            telemetry.applied_action_tick != dispatch.action_tick
            or telemetry.server_frame != dispatch.action_tick + 1
            or echo_tick != dispatch.action_tick
        ):
            return "mismatch", echo_tick
        if not accepted or echo_tick < dispatch.action_tick:
            return "stale", echo_tick
        if echo_tick > telemetry.server_frame:
            return "mismatch", echo_tick
        gate_flags = int(debug[ActionDebugIndex.FLAGS])
        echoed_generation = (
            gate_flags & ML_ACTION_GENERATION_MASK
        ) >> ML_ACTION_GENERATION_SHIFT
        if (
            echoed_generation == 0
            or echoed_generation - 1
            != dispatch.action_tick % ML_ACTION_GENERATION_COUNT
        ):
            return "stale", echo_tick
        causal = telemetry.causal
        if (
            causal.echo_tick != echo_tick
            or causal.action_generation != echoed_generation
        ):
            return "mismatch", echo_tick
        expected_forward = self._clamp_movement(dispatch.action.move_forward)
        expected_right = self._clamp_movement(dispatch.action.move_right)
        movement_matches = (
            abs(float(debug[ActionDebugIndex.MOVE_FORWARD]) - expected_forward)
            <= self.movement_tolerance
            and abs(float(debug[ActionDebugIndex.MOVE_RIGHT]) - expected_right)
            <= self.movement_tolerance
        )
        yaw_matches = (
            abs(float(debug[ActionDebugIndex.LOOK_YAW]) - float(dispatch.action.look_yaw))
            <= self.look_tolerance
        )
        pitch_matches = (
            abs(float(debug[ActionDebugIndex.LOOK_PITCH]) - float(dispatch.action.look_pitch))
            <= self.look_tolerance
        )
        echoed_fire = bool(int(debug[ActionDebugIndex.FIRE]))
        requested_fire = bool(dispatch.action.fire)
        fire_suppressed = (
            requested_fire
            and not echoed_fire
            and bool(gate_flags & ML_FIRE_GATE_SUPPRESSED)
        )
        fire_matches = (
            echoed_fire == requested_fire or fire_suppressed
        )
        requested_vertical = VerticalIntent(int(dispatch.action.vertical_intent))
        expected_upmove = {
            VerticalIntent.CROUCH_OR_DOWN: -320,
            VerticalIntent.NEUTRAL: 0,
            VerticalIntent.JUMP_OR_UP: 320,
        }[requested_vertical]
        vertical_matches = (
            int(debug[ActionDebugIndex.VERTICAL_INTENT]) == int(requested_vertical)
            and int(debug[ActionDebugIndex.APPLIED_UPMOVE]) == expected_upmove
        )
        reliable_command_matches = (
            int(debug[ActionDebugIndex.HOOK]) == int(dispatch.action.hook)
            and int(debug[ActionDebugIndex.WEAPON]) == int(dispatch.action.weapon)
        )
        # The public server continues running between policy inference and
        # dispatch. Death/respawn, delta-angle updates, or a pitch clamp can
        # alter engine-owned look/buttons even though the ordinary usercmd was
        # delivered. A matching same-tick generation plus movement and reliable
        # commands proves causality; discard the whole synchronized round while
        # retaining fatal checks for movement/hook/weapon corruption.
        action_state_resync = (
            not (yaw_matches and pitch_matches) or not fire_matches
        )
        engine_hold = (
            causal.echo_valid
            and causal.facts_complete
            and not causal.transition_trainable
            and causal.role_public_pm_normal
            and movement_matches
            and vertical_matches
            and reliable_command_matches
        )
        if engine_hold:
            return "engine_hold", echo_tick
        if (
            self.deterministic_frame_barrier
            and action_state_resync
            and not allow_lifecycle_action_resync
        ):
            return "mismatch", echo_tick
        if (
            action_state_resync
            and movement_matches
            and vertical_matches
            and reliable_command_matches
        ):
            return "state_resync", echo_tick
        command_matched = (
            movement_matches
            and yaw_matches
            and pitch_matches
            and vertical_matches
            and fire_matches
            and reliable_command_matches
        )
        if command_matched and not (
            causal.echo_valid
            and causal.facts_complete
            and causal.transition_trainable
        ):
            return "state_resync", echo_tick
        return ("matched" if command_matched else "mismatch"), echo_tick

    def _collect_echo(
        self,
        env: Q2NetworkClientEnv,
        dispatch: ClientActionDispatch,
        deadline: float,
        expected_life_epoch: int,
        death_pending: _DeathLifecyclePending | None,
        hold_life_epoch: int | None,
    ) -> _MatchedEcho | _MapEpoch | _ActionStateResync | _DeathLifecycleResync:
        sequence = dispatch.after_sequence
        stale_echoes = 0
        mismatched_echoes = 0
        reward_fields = (
            "reward_damage_dealt",
            "reward_damage_taken",
            "reward_kill",
            "reward_death",
            "reward_item_pickup",
            "reward_hook_traversal",
            "reward_damage_taken_prox",
            "reward_offense",
            "reward_survival",
        )
        reward_sums = {field: 0.0 for field in reward_fields}
        terminal = False
        terminal_reason = 0
        while stale_echoes + mismatched_echoes <= self.max_rejected_echoes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AuthoritativeEchoError(
                    f"authoritative echo timed out for {dispatch.client_id}",
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                    timed_out=True,
                )
            try:
                telemetry = env.receive_telemetry(
                    after_sequence=sequence,
                    timeout=remaining,
                )
            except TimeoutError as error:
                raise AuthoritativeEchoError(
                    f"authoritative echo timed out for {dispatch.client_id}",
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                    timed_out=True,
                ) from error
            sequence = telemetry.sequence
            # A level change resets server_frame, so neither echo tick ordering
            # nor reward deltas from this packet belong to the dispatched
            # action's epoch. Hand control back before accumulating either.
            if telemetry.map_name != dispatch.map_name:
                return _MapEpoch(
                    telemetry=telemetry,
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                )
            observation = telemetry.observation
            if not telemetry.causal.role_playing:
                raise AuthoritativeEchoError(
                    "routed causal role is not an authoritative Lithium "
                    f"player for {dispatch.client_id}",
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes + 1,
                )
            if (
                telemetry.causal.transition_trainable
                and not telemetry.causal.role_public_pm_normal
            ):
                raise AuthoritativeEchoError(
                    "trainable routed causal role lacks an exact public "
                    f"PM_NORMAL witness for {dispatch.client_id}",
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes + 1,
                )
            death_terminal = (
                bool(getattr(observation, "is_terminal", False))
                and int(getattr(observation, "terminal_reason", 0))
                == ML_TERMINAL_DEATH
            )
            # During intermission ClientThink intentionally stops applying
            # usercmds, so action_debug remains on the last playable frame.
            # Treat that authoritative terminal marker as a map boundary
            # before the normal stale/mismatch budget can terminate PPO.
            if (
                bool(getattr(observation, "is_terminal", False))
                and int(getattr(observation, "terminal_reason", 0))
                == ML_TERMINAL_INTERMISSION
            ):
                return _MapEpoch(
                    telemetry=telemetry,
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                )
            for field in reward_fields:
                reward_sums[field] += float(getattr(observation, field, 0.0))
            if bool(getattr(observation, "is_terminal", False)):
                terminal = True
                terminal_reason = int(
                    getattr(observation, "terminal_reason", terminal_reason)
                )
            observed_life_epoch = int(telemetry.causal.client_life_epoch)
            death_life_epoch = (
                None if death_pending is None
                else death_pending.previous_life_epoch
            )
            if death_life_epoch is None:
                life_epoch_valid = observed_life_epoch == expected_life_epoch
            elif death_pending.new_life_seen:
                life_epoch_valid = observed_life_epoch == death_life_epoch + 1
            else:
                life_epoch_valid = observed_life_epoch in (
                    death_life_epoch, death_life_epoch + 1
                )
            if not life_epoch_valid:
                raise AuthoritativeEchoError(
                    "causal client life epoch violated the synchronized "
                    f"death boundary for {dispatch.client_id}: expected "
                    f"{expected_life_epoch if death_life_epoch is None else death_life_epoch}"
                    f"{'' if death_life_epoch is None else ' or ' + str(death_life_epoch + 1)}, "
                    f"received {observed_life_epoch}",
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes + 1,
                )
            state, echo_tick = self._echo_state(
                telemetry,
                dispatch,
                allow_lifecycle_action_resync=(
                    death_life_epoch is not None or death_terminal
                ),
            )
            if death_terminal:
                if death_pending is not None:
                    if observed_life_epoch == death_life_epoch:
                        raise AuthoritativeEchoError(
                            f"duplicate death terminal for {dispatch.client_id}",
                            stale_echoes=stale_echoes,
                            mismatched_echoes=mismatched_echoes + 1,
                        )
                    # The player can die again on E+1 before E's lifecycle has
                    # produced its actionable prime.  Advance the pending
                    # identity to E+1; a later respawn must be exactly E+2.
                    return _DeathLifecycleResync(
                        telemetry=telemetry,
                        echo_tick=echo_tick,
                        stale_echoes=stale_echoes,
                        mismatched_echoes=mismatched_echoes,
                        previous_life_epoch=observed_life_epoch,
                        next_life_epoch=observed_life_epoch,
                        death_terminal=True,
                        phase="rapid_redeath_terminal",
                    )
                terminal_state = state in (
                    "matched", "state_resync", "engine_hold"
                )
                realtime_stale = (
                    state == "stale" and not self.deterministic_frame_barrier
                )
                if terminal_state or realtime_stale:
                    # Death has precedence over a same-life hold and over a
                    # command that would otherwise be admissible.  It creates
                    # the new lifecycle boundary before generic hold handling,
                    # so its reward can never migrate across the respawn.
                    return _DeathLifecycleResync(
                        telemetry=telemetry,
                        echo_tick=echo_tick,
                        stale_echoes=stale_echoes + int(realtime_stale),
                        mismatched_echoes=mismatched_echoes,
                        previous_life_epoch=expected_life_epoch,
                        next_life_epoch=observed_life_epoch,
                        death_terminal=True,
                        phase="death_terminal",
                    )
            if death_pending is not None and state in (
                "matched", "state_resync", "engine_hold"
            ):
                if observed_life_epoch == death_life_epoch:
                    phase = "corpse"
                elif not death_pending.new_life_seen:
                    # PutClientInServer changes the causal identity and the
                    # client rebases its command-space view.  Even an exact
                    # first packet is lifecycle state, never policy reward.
                    phase = "new_life_rebase"
                elif state == "matched":
                    # _echo_state calls a command matched only when the full
                    # causal triad is actionable.  This first exact new-life
                    # action closes the lifecycle but remains a zero-reward
                    # priming boundary; only the following action may train.
                    phase = "new_life_actionable_prime"
                elif state == "engine_hold":
                    phase = "new_life_teleport_settling"
                else:
                    raise AuthoritativeEchoError(
                        "new-life altered action was not marked as an "
                        f"engine-owned nontrainable transition for {dispatch.client_id}",
                        stale_echoes=stale_echoes,
                        mismatched_echoes=mismatched_echoes + 1,
                    )
                return _DeathLifecycleResync(
                    telemetry=telemetry,
                    echo_tick=echo_tick,
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                    previous_life_epoch=death_life_epoch,
                    next_life_epoch=observed_life_epoch,
                    phase=phase,
                )
            if hold_life_epoch is not None:
                if observed_life_epoch != hold_life_epoch:
                    raise AuthoritativeEchoError(
                        "same-life hold crossed a client life epoch without a "
                        f"death boundary for {dispatch.client_id}",
                        stale_echoes=stale_echoes,
                        mismatched_echoes=mismatched_echoes + 1,
                    )
                if state == "engine_hold":
                    return _ActionStateResync(
                        telemetry=telemetry,
                        echo_tick=echo_tick,
                        stale_echoes=stale_echoes,
                        mismatched_echoes=mismatched_echoes,
                        phase="same_life_hold_settling",
                        hold_life_epoch=hold_life_epoch,
                    )
                if state == "matched":
                    return _ActionStateResync(
                        telemetry=telemetry,
                        echo_tick=echo_tick,
                        stale_echoes=stale_echoes,
                        mismatched_echoes=mismatched_echoes,
                        phase="same_life_actionable_prime",
                        hold_life_epoch=hold_life_epoch,
                    )
                raise AuthoritativeEchoError(
                    "same-life hold did not settle to an exact actionable "
                    f"transition for {dispatch.client_id}",
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes + 1,
                )
            if state == "state_resync":
                return _ActionStateResync(
                    telemetry=telemetry,
                    echo_tick=echo_tick,
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                )
            if state == "engine_hold":
                return _ActionStateResync(
                    telemetry=telemetry,
                    echo_tick=echo_tick,
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                    phase="same_life_hold_settling",
                    hold_life_epoch=observed_life_epoch,
                )
            if state == "matched":
                replacements = dict(reward_sums)
                replacements.update(
                    is_terminal=terminal,
                    terminal_reason=terminal_reason,
                )
                if dataclasses.is_dataclass(observation):
                    observation = dataclasses.replace(observation, **replacements)
                else:  # Enables lightweight protocol fakes in focused tests.
                    observation = copy.copy(observation)
                    for field, value in replacements.items():
                        setattr(observation, field, value)
                telemetry = dataclasses.replace(
                    telemetry, observation=observation
                )
                return _MatchedEcho(
                    telemetry=telemetry,
                    echo_tick=echo_tick,
                    stale_echoes=stale_echoes,
                    mismatched_echoes=mismatched_echoes,
                )
            if state == "stale":
                stale_echoes += 1
            else:
                mismatched_echoes += 1
        raise AuthoritativeEchoError(
            f"too many rejected authoritative echoes for {dispatch.client_id}",
            stale_echoes=stale_echoes,
            mismatched_echoes=mismatched_echoes,
        )

    @staticmethod
    def _result_rejections(
        result: (
            _MatchedEcho | _MapEpoch | _ActionStateResync |
            _DeathLifecycleResync | None
        ),
    ) -> tuple[int, int]:
        if result is None:
            return 0, 0
        return result.stale_echoes, result.mismatched_echoes

    @staticmethod
    def _is_intermission(telemetry: ClientTelemetry) -> bool:
        observation = telemetry.observation
        return bool(getattr(observation, "is_terminal", False)) and int(
            getattr(observation, "terminal_reason", 0)
        ) == ML_TERMINAL_INTERMISSION

    def _wait_for_map_progress(
        self,
        env: Q2NetworkClientEnv,
        current: ClientTelemetry,
        source_map: str,
        deadline: float,
    ) -> ClientTelemetry:
        """Wait for a playable post-source packet without failing on timeout."""
        latest = current
        if latest.map_name != source_map and not self._is_intermission(latest):
            return latest
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return latest
            try:
                latest = env.receive_telemetry(
                    after_sequence=latest.sequence,
                    timeout=remaining,
                )
            except TimeoutError:
                # A generated-map download pause is not an action-echo
                # failure. Keep the barrier active for the next call.
                return latest
            if latest.map_name != source_map and not self._is_intermission(latest):
                return latest

    def _wait_for_telemetry_progress(
        self,
        env: Q2NetworkClientEnv,
        current: ClientTelemetry,
        deadline: float,
    ) -> ClientTelemetryDrain:
        """Wait once for a conduit to resume, preserving drain provenance."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ClientTelemetryDrain(current, current, 0, ())
        try:
            first = env.receive_telemetry(
                after_sequence=current.sequence,
                timeout=remaining,
            )
        except TimeoutError:
            return ClientTelemetryDrain(current, current, 0, ())
        extra = env.drain_latest_telemetry()
        latest = extra.latest
        names = (first.map_name,) + extra.map_names
        return ClientTelemetryDrain(
            previous=current,
            latest=latest,
            packet_count=1 + extra.packet_count,
            map_names=names,
        )

    def _poll_telemetry_gap(
        self,
        drains: Sequence[ClientTelemetryDrain],
        *,
        round_id: int,
        policy_version: int,
    ) -> BatchRound:
        """Hold dispatch while a synchronized conduit/download gap persists."""
        samples = list(drains)
        self._telemetry_gap_resumed_clients.update(
            drain.latest.client_id for drain in samples if drain.advanced
        )
        pending_indices = [
            index for index, drain in enumerate(samples)
            if drain.latest.client_id not in self._telemetry_gap_resumed_clients
        ]
        if pending_indices:
            deadline = time.monotonic() + self.round_timeout
            futures = {
                index: self._executor.submit(
                    self._wait_for_telemetry_progress,
                    self.envs[index],
                    samples[index].latest,
                    deadline,
                )
                for index in pending_indices
            }
            for index, future in futures.items():
                samples[index] = future.result()
            self._telemetry_gap_resumed_clients.update(
                drain.latest.client_id for drain in samples if drain.advanced
            )
        map_boundary_seen = any(
            drain.map_changed or self._is_intermission(drain.latest)
            for drain in samples
        )
        all_clients_resumed = self._telemetry_gap_resumed_clients == {
            env.client_id for env in self.envs
        }
        if all_clients_resumed or map_boundary_seen:
            self._telemetry_gap_pending = False
            self._telemetry_gap_resumed_clients.clear()
        return self._preflight_boundary(
            samples,
            round_id=round_id,
            policy_version=policy_version,
            telemetry_gap_resync=True,
        )

    def _begin_map_epoch(self, source_map: str) -> None:
        if self._map_epoch_source is None:
            self._map_epoch_source = source_map
            self._map_epoch_resyncs += 1
            return
        if self._map_epoch_source != source_map:
            self._failed_rounds += 1
            raise AuthoritativeEchoError(
                "map epoch source changed while synchronization was pending: "
                f"{self._map_epoch_source} -> {source_map}"
            )

    def _accept_map_life_epochs(
        self, samples: Sequence[ClientTelemetry]
    ) -> None:
        """Re-prime transport identities at a separately proven map boundary."""
        if any(
            not sample.causal.role_playing
            or not sample.causal.role_public_pm_normal
            for sample in samples
        ):
            self._failed_rounds += 1
            raise AuthoritativeEchoError(
                "map bootstrap lacks an exact playable public PM_NORMAL role"
            )
        observed = {
            sample.client_id: int(sample.causal.client_life_epoch)
            for sample in samples
        }
        if set(observed) != set(self._client_life_epochs):
            self._failed_rounds += 1
            raise AuthoritativeEchoError(
                "map boundary changed the deterministic client roster"
            )
        if any(epoch <= 0 for epoch in observed.values()):
            self._failed_rounds += 1
            raise AuthoritativeEchoError(
                "map boundary produced a zero causal client life epoch"
            )
        if any(
            observed[client_id] < previous_epoch
            for client_id, previous_epoch in self._client_life_epochs.items()
        ):
            self._failed_rounds += 1
            raise AuthoritativeEchoError(
                "map boundary rolled back a causal client life epoch"
            )
        self._client_life_epochs = observed
        self._death_life_epochs.clear()
        self._action_hold_epochs.clear()

    def _pending_map_boundary(
        self,
        drains: Sequence[ClientTelemetryDrain],
        *,
        round_id: int,
        policy_version: int,
        poll: bool,
        tags: tuple[BatchActionTag, ...] | None = None,
        action_dispatched: bool = False,
        rejections: Sequence[tuple[int, int]] | None = None,
    ) -> BatchRound:
        """Return a non-trainable barrier until one new map is playable."""
        source_map = self._map_epoch_source
        if source_map is None:
            raise RuntimeError("map epoch boundary requested without a source map")

        samples = [drain.latest for drain in drains]
        if poll:
            deadline = time.monotonic() + self.round_timeout
            futures = [
                self._executor.submit(
                    self._wait_for_map_progress,
                    env,
                    sample,
                    source_map,
                    deadline,
                )
                for env, sample in zip(self.envs, samples)
            ]
            samples = [future.result() for future in futures]

        post_source_maps = {
            sample.map_name for sample in samples
            if sample.map_name != source_map
        }
        if len(post_source_maps) > 1:
            self._failed_rounds += 1
            raise AuthoritativeEchoError(
                "clients crossed multiple map epochs in one batch: "
                + ", ".join(sorted(post_source_maps))
            )
        target_map = (
            next(iter(post_source_maps)) if post_source_maps else source_map
        )
        ready = bool(post_source_maps) and all(
            sample.map_name == target_map and not self._is_intermission(sample)
            for sample in samples
        )

        if tags is None:
            tags = tuple(
                BatchActionTag(
                    round_id=round_id,
                    policy_version=policy_version,
                    client_index=index,
                    client_id=drain.latest.client_id,
                    client_slot=drain.latest.client_slot,
                    action_tick=drain.previous.server_frame,
                )
                for index, drain in enumerate(drains)
            )
        if rejections is None:
            rejections = [(0, 0)] * len(samples)

        initial_results = [
            env.initial_result(sample, vector=self.vector)
            for env, sample in zip(self.envs, samples)
        ]
        infos = []
        for initial, tag, sample, drain, rejection in zip(
            initial_results, tags, samples, drains, rejections
        ):
            stale, mismatched = rejection
            info = dict(initial[1])
            info.update({
                "batch_round_id": tag.round_id,
                "policy_version": tag.policy_version,
                "action_tick": tag.action_tick,
                "action_dispatched": action_dispatched,
                "authoritative_echo_tick": int(
                    sample.observation.action_debug[ActionDebugIndex.TICK]
                ),
                "authoritative_echo_valid": False,
                "authoritative_client_life_epoch": int(
                    sample.causal.client_life_epoch
                ),
                "authoritative_echo_stale": True,
                "trainable_transition": False,
                "realtime_catchup_resync": True,
                "preflight_packets_drained": drain.packet_count,
                "preflight_from_frame": drain.previous.server_frame,
                "preflight_to_frame": sample.server_frame,
                "map_epoch_resync": True,
                "map_epoch_pending": not ready,
                "map_epoch_from": source_map,
                "map_epoch_target": target_map,
                "terminal_reason": ML_TERMINAL_INTERMISSION,
                "stale_echoes_rejected": stale,
                "mismatched_echoes_rejected": mismatched,
            })
            infos.append(info)

        frame_values = [sample.server_frame for sample in samples]
        self._max_observed_frame_span = max(
            self._max_observed_frame_span,
            max(frame_values) - min(frame_values),
        )
        drained_count = sum(drain.packet_count for drain in drains)
        self._preflight_packets_drained += drained_count
        self._realtime_catchup_resyncs += 1
        if ready:
            self._accept_map_life_epochs(samples)
            self._map_epoch_source = None
        return BatchRound(
            round_id=round_id,
            policy_version=policy_version,
            observations=self._collate_observations(
                [initial[0] for initial in initial_results], self.vector
            ),
            rewards=np.zeros(len(self.envs), dtype=np.float32),
            terminated=np.ones(len(self.envs), dtype=np.bool_),
            truncated=np.zeros(len(self.envs), dtype=np.bool_),
            infos=tuple(infos),
            tags=tags,
        )

    @staticmethod
    def _collate_observations(values: Sequence[Any], vector: bool):
        if vector:
            return np.stack(values, axis=0)
        return tuple(values)

    def _preflight_boundary(
        self,
        drains: Sequence[ClientTelemetryDrain],
        *,
        round_id: int,
        policy_version: int,
        action_dispatched: bool = False,
        tags: tuple[BatchActionTag, ...] | None = None,
        rejections: Sequence[tuple[int, int]] | None = None,
        action_state_resync: bool = False,
        action_state_phases: dict[str, str] | None = None,
        death_lifecycle_resync: bool = False,
        death_lifecycle_phases: dict[str, str] | None = None,
        telemetry_gap_resync: bool = False,
    ) -> BatchRound:
        map_changed = any(drain.map_changed for drain in drains)
        intermission = any(
            bool(getattr(drain.latest.observation, "is_terminal", False))
            and int(
                getattr(drain.latest.observation, "terminal_reason", 0)
            ) == ML_TERMINAL_INTERMISSION
            for drain in drains
        )
        map_boundary = map_changed or intermission
        if map_boundary:
            source_maps = {drain.previous.map_name for drain in drains}
            if len(source_maps) != 1:
                self._failed_rounds += 1
                raise AuthoritativeEchoError(
                    "preflight crossed multiple source maps: "
                    + ", ".join(sorted(source_maps))
                )
            # The explicit map-epoch barrier now owns synchronization; do not
            # leave the earlier pre-boundary gap latched after it completes.
            self._telemetry_gap_pending = False
            self._telemetry_gap_resumed_clients.clear()
            self._begin_map_epoch(next(iter(source_maps)))
            return self._pending_map_boundary(
                drains,
                round_id=round_id,
                policy_version=policy_version,
                poll=False,
                tags=tags,
                action_dispatched=action_dispatched,
                rejections=rejections,
            )

        samples = [drain.latest for drain in drains]
        target_map = samples[0].map_name

        initial_results = [
            env.initial_result(sample, vector=self.vector)
            for env, sample in zip(self.envs, samples)
        ]
        if tags is None:
            tags = tuple(
                BatchActionTag(
                    round_id=round_id,
                    policy_version=policy_version,
                    client_index=index,
                    client_id=drain.latest.client_id,
                    client_slot=drain.latest.client_slot,
                    action_tick=drain.previous.server_frame,
                )
                for index, drain in enumerate(drains)
            )
        if rejections is None:
            rejections = [(0, 0)] * len(drains)
        infos = []
        lifecycle_phases = death_lifecycle_phases or {}
        action_phases = action_state_phases or {}
        for initial, tag, sample, drain, rejection in zip(
            initial_results, tags, samples, drains, rejections
        ):
            stale, mismatched = rejection
            info = dict(initial[1])
            info.update({
                "batch_round_id": tag.round_id,
                "policy_version": tag.policy_version,
                "action_tick": tag.action_tick,
                "action_dispatched": action_dispatched,
                "authoritative_echo_tick": int(
                    sample.observation.action_debug[ActionDebugIndex.TICK]
                ),
                "authoritative_echo_valid": False,
                "authoritative_client_life_epoch": int(
                    sample.causal.client_life_epoch
                ),
                "authoritative_echo_stale": True,
                "trainable_transition": False,
                "realtime_catchup_resync": True,
                "action_state_resync": action_state_resync,
                "action_state_phase": action_phases.get(
                    sample.client_id, "peer" if action_state_resync else "none"
                ),
                "death_lifecycle_resync": death_lifecycle_resync,
                "death_lifecycle_phase": lifecycle_phases.get(
                    sample.client_id, "peer" if death_lifecycle_resync else "none"
                ),
                "causal_echo_valid": sample.causal.echo_valid,
                "causal_facts_complete": sample.causal.facts_complete,
                "causal_transition_trainable": (
                    sample.causal.transition_trainable
                ),
                "causal_role_playing": sample.causal.role_playing,
                "causal_role_public_pm_normal": (
                    sample.causal.role_public_pm_normal
                ),
                "authoritative_action_generation": (
                    sample.causal.action_generation
                ),
                "telemetry_gap_resync": telemetry_gap_resync,
                "preflight_packets_drained": drain.packet_count,
                "preflight_from_frame": drain.previous.server_frame,
                "preflight_to_frame": sample.server_frame,
                "map_epoch_resync": map_boundary,
                "map_epoch_pending": intermission and not map_changed,
                "map_epoch_from": drain.previous.map_name,
                "map_epoch_target": target_map,
                "terminal_reason": (
                    ML_TERMINAL_INTERMISSION if map_boundary else
                    ML_TERMINAL_DEATH if death_lifecycle_resync else 0
                ),
                "observed_terminal_reason": int(
                    getattr(sample.observation, "terminal_reason", 0)
                ),
                "stale_echoes_rejected": stale,
                "mismatched_echoes_rejected": mismatched,
            })
            infos.append(info)

        frame_values = [sample.server_frame for sample in samples]
        self._max_observed_frame_span = max(
            self._max_observed_frame_span,
            max(frame_values) - min(frame_values),
        )
        drained_count = sum(drain.packet_count for drain in drains)
        self._preflight_packets_drained += drained_count
        self._realtime_catchup_resyncs += 1
        self._action_state_resyncs += int(action_state_resync)
        self._death_lifecycle_resyncs += int(death_lifecycle_resync)
        return BatchRound(
            round_id=round_id,
            policy_version=policy_version,
            observations=self._collate_observations(
                [initial[0] for initial in initial_results], self.vector
            ),
            rewards=np.zeros(len(self.envs), dtype=np.float32),
            terminated=np.ones(len(self.envs), dtype=np.bool_),
            truncated=np.zeros(len(self.envs), dtype=np.bool_),
            infos=tuple(infos),
            tags=tags,
        )

    def reset(self):
        """Start all clients concurrently and return their initial observations."""
        with self._round_lock:
            if self._closed:
                raise RuntimeError("network client batch is closed")
            if self._started:
                raise RuntimeError("network client batch is already started")
            futures = [self._executor.submit(env.start) for env in self.envs]
            telemetry = [future.result() for future in futures]
            if any(
                not sample.causal.role_playing
                or not sample.causal.role_public_pm_normal
                for sample in telemetry
            ):
                raise AuthoritativeEchoError(
                    "initial bootstrap lacks an exact playable public "
                    "PM_NORMAL role"
                )
            results = [
                env.initial_result(sample, vector=self.vector)
                for env, sample in zip(self.envs, telemetry)
            ]
            initial_life_epochs = {
                sample.client_id: int(sample.causal.client_life_epoch)
                for sample in telemetry
            }
            if any(epoch <= 0 for epoch in initial_life_epochs.values()):
                raise AuthoritativeEchoError(
                    "initial causal client life epoch must be positive"
                )
            self._client_life_epochs = initial_life_epochs
            self._death_life_epochs.clear()
            self._action_hold_epochs.clear()
            self._started = True
            observations = self._collate_observations(
                [result[0] for result in results], self.vector
            )
            return observations, [result[1] for result in results]

    def collect_round(
        self,
        actions: Sequence[Action],
        *,
        policy_version: int,
    ) -> BatchRound:
        """Dispatch all actions, then admit only their authoritative echoes."""
        with self._round_lock:
            if self._closed:
                raise RuntimeError("network client batch is closed")
            if not self._started:
                raise RuntimeError("call reset() before collect_round()")
            if len(actions) != len(self.envs):
                raise ValueError(
                    f"expected {len(self.envs)} actions, received {len(actions)}"
                )
            version = self._policy_version(policy_version)
            if (
                self._latest_policy_version is not None
                and version < self._latest_policy_version
            ):
                self._stale_policy_rounds_rejected += 1
                raise StalePolicyVersionError(
                    f"policy {version} is older than active policy "
                    f"{self._latest_policy_version}"
                )

            round_id = self._next_round_id
            self._next_round_id += 1
            self._latest_policy_version = version
            drain_futures = [
                self._executor.submit(env.drain_latest_telemetry)
                for env in self.envs
            ]
            drains = [future.result() for future in drain_futures]
            if self._map_epoch_source is not None:
                return self._pending_map_boundary(
                    drains,
                    round_id=round_id,
                    policy_version=version,
                    poll=True,
                )
            if self._telemetry_gap_pending:
                return self._poll_telemetry_gap(
                    drains,
                    round_id=round_id,
                    policy_version=version,
                )
            if any(drain.advanced for drain in drains):
                barrier_epoch_boundary = any(
                    drain.map_changed or self._is_intermission(drain.latest)
                    for drain in drains
                )
                if self.deterministic_frame_barrier and not barrier_epoch_boundary:
                    self._failed_rounds += 1
                    raise AuthoritativeEchoError(
                        "deterministic frame barrier observed unsolicited telemetry"
                    )
                return self._preflight_boundary(
                    drains,
                    round_id=round_id,
                    policy_version=version,
                )
            dispatches = [
                env.dispatch_action(action)
                for env, action in zip(self.envs, actions)
            ]
            lifecycle_epochs = [
                self._death_life_epochs.get(dispatch.client_id)
                for dispatch in dispatches
            ]
            hold_epochs = [
                self._action_hold_epochs.get(dispatch.client_id)
                for dispatch in dispatches
            ]
            self._rounds_dispatched += 1
            self._actions_dispatched += len(dispatches)
            deadline = time.monotonic() + self.round_timeout
            futures = [
                self._executor.submit(
                    self._collect_echo,
                    env,
                    dispatch,
                    deadline,
                    self._client_life_epochs[dispatch.client_id],
                    lifecycle_epoch,
                    hold_epoch,
                )
                for env, dispatch, lifecycle_epoch, hold_epoch in zip(
                    self.envs, dispatches, lifecycle_epochs, hold_epochs
                )
            ]
            echo_results: list[
                _MatchedEcho | _MapEpoch | _ActionStateResync |
                _DeathLifecycleResync | None
            ] = [None] * len(
                futures
            )
            errors: list[AuthoritativeEchoError] = []
            for index, future in enumerate(futures):
                try:
                    echo_results[index] = future.result()
                except AuthoritativeEchoError as error:
                    errors.append(error)

            synchronized_gap = (
                len(self.envs) > 1
                and len(errors) == len(self.envs)
                and all(error.timed_out for error in errors)
                and all(result is None for result in echo_results)
            )

            for result in echo_results:
                stale, mismatched = self._result_rejections(result)
                self._stale_echoes_rejected += stale
                self._mismatched_echoes_rejected += mismatched
            for error in errors:
                self._stale_echoes_rejected += error.stale_echoes
                self._mismatched_echoes_rejected += error.mismatched_echoes
                if not synchronized_gap:
                    self._echo_timeouts += int(error.timed_out)

            tags = tuple(
                BatchActionTag(
                    round_id=round_id,
                    policy_version=version,
                    client_index=index,
                    client_id=dispatch.client_id,
                    client_slot=dispatch.client_slot,
                    action_tick=dispatch.action_tick,
                )
                for index, dispatch in enumerate(dispatches)
            )
            map_epochs = [
                result for result in echo_results if isinstance(result, _MapEpoch)
            ]
            if map_epochs:
                source_maps = {dispatch.map_name for dispatch in dispatches}
                if len(source_maps) != 1:
                    self._failed_rounds += 1
                    raise AuthoritativeEchoError(
                        "map boundary crossed multiple source maps: "
                        + ", ".join(sorted(source_maps))
                    )
                self._begin_map_epoch(next(iter(source_maps)))
                boundary_drains = [
                    env.drain_latest_telemetry() for env in self.envs
                ]
                return self._pending_map_boundary(
                    boundary_drains,
                    round_id=round_id,
                    policy_version=version,
                    poll=False,
                    tags=tags,
                    action_dispatched=True,
                    rejections=[
                        self._result_rejections(result)
                        for result in echo_results
                    ],
                )

            if synchronized_gap:
                self._telemetry_gap_pending = True
                self._telemetry_gap_resumed_clients.clear()
                self._telemetry_gap_resyncs += 1
                boundary_drains = [
                    env.drain_latest_telemetry() for env in self.envs
                ]
                self._telemetry_gap_resumed_clients.update(
                    drain.latest.client_id
                    for drain in boundary_drains if drain.advanced
                )
                if self._telemetry_gap_resumed_clients == {
                    env.client_id for env in self.envs
                }:
                    self._telemetry_gap_pending = False
                    self._telemetry_gap_resumed_clients.clear()
                return self._preflight_boundary(
                    boundary_drains,
                    round_id=round_id,
                    policy_version=version,
                    action_dispatched=True,
                    tags=tags,
                    rejections=[
                        (error.stale_echoes, error.mismatched_echoes)
                        for error in errors
                    ],
                    telemetry_gap_resync=True,
                )

            if errors:
                self._failed_rounds += 1
                raise AuthoritativeEchoError(
                    f"batch round {round_id} rejected for {len(errors)} client(s)",
                    stale_echoes=sum(error.stale_echoes for error in errors),
                    mismatched_echoes=sum(
                        error.mismatched_echoes for error in errors
                    ),
                    timed_out=any(error.timed_out for error in errors),
                ) from errors[0]

            # A different client may already require a whole-batch boundary.
            # Register every independently exact death terminal before taking
            # that early return so overlapping deaths cannot be lost and then
            # surface as an unexplained life-epoch jump at respawn.
            for result in echo_results:
                if not isinstance(result, _MatchedEcho):
                    continue
                observation = result.telemetry.observation
                if not (
                    bool(getattr(observation, "is_terminal", False))
                    and int(getattr(observation, "terminal_reason", 0))
                    == ML_TERMINAL_DEATH
                ):
                    continue
                client_id = result.telemetry.client_id
                if client_id in self._death_life_epochs:
                    self._failed_rounds += 1
                    raise AuthoritativeEchoError(
                        f"duplicate death terminal for {client_id}"
                    )
                self._death_life_epochs[client_id] = _DeathLifecyclePending(
                    int(result.telemetry.causal.client_life_epoch)
                )

            state_resyncs = [
                result for result in echo_results
                if isinstance(result, _ActionStateResync)
            ]
            death_lifecycle_resyncs = [
                result for result in echo_results
                if isinstance(result, _DeathLifecycleResync)
            ]
            if state_resyncs or death_lifecycle_resyncs:
                action_phases = {
                    result.telemetry.client_id: result.phase
                    for result in state_resyncs
                }
                for result in state_resyncs:
                    client_id = result.telemetry.client_id
                    if result.phase == "same_life_hold_settling":
                        if result.hold_life_epoch is None:
                            self._failed_rounds += 1
                            raise AuthoritativeEchoError(
                                f"same-life hold omitted epoch for {client_id}"
                            )
                        self._action_hold_epochs[client_id] = (
                            result.hold_life_epoch
                        )
                    elif result.phase == "same_life_actionable_prime":
                        self._action_hold_epochs.pop(client_id, None)
                lifecycle_phases = {
                    result.telemetry.client_id: result.phase
                    for result in death_lifecycle_resyncs
                }
                for result in death_lifecycle_resyncs:
                    client_id = result.telemetry.client_id
                    if result.death_terminal:
                        self._action_hold_epochs.pop(client_id, None)
                        self._death_life_epochs[client_id] = (
                            _DeathLifecyclePending(result.previous_life_epoch)
                        )
                    elif result.phase == "new_life_rebase":
                        self._death_life_epochs[client_id] = (
                            _DeathLifecyclePending(
                                result.previous_life_epoch,
                                new_life_seen=True,
                            )
                        )
                    elif result.phase == "new_life_actionable_prime":
                        self._client_life_epochs[client_id] = (
                            result.next_life_epoch
                        )
                        self._death_life_epochs.pop(client_id, None)
                boundary_drains = [
                    env.drain_latest_telemetry() for env in self.envs
                ]
                return self._preflight_boundary(
                    boundary_drains,
                    round_id=round_id,
                    policy_version=version,
                    action_dispatched=True,
                    tags=tags,
                    rejections=[
                        self._result_rejections(result)
                        for result in echo_results
                    ],
                    action_state_resync=bool(state_resyncs),
                    action_state_phases=action_phases,
                    death_lifecycle_resync=bool(death_lifecycle_resyncs),
                    death_lifecycle_phases=lifecycle_phases,
                )

            accepted = [
                result
                for result in echo_results
                if isinstance(result, _MatchedEcho)
            ]
            results = [
                env.transition_result(match.telemetry, vector=self.vector)
                for env, match in zip(self.envs, accepted)
            ]
            spatial_epochs = []
            for result in results:
                attestation = result[4].get(SPATIAL_ATTESTATION_INFO_KEY)
                if isinstance(attestation, dict) and type(
                    attestation.get("map_epoch")
                ) is int:
                    spatial_epochs.append(attestation["map_epoch"])
            if spatial_epochs and (
                len(spatial_epochs) != len(results)
                or len(set(spatial_epochs)) != 1
            ):
                self._failed_rounds += 1
                raise AuthoritativeEchoError(
                    "spatial provider map epochs differ across synchronized clients"
                )
            authoritative_map_epoch = (
                spatial_epochs[0] if spatial_epochs else None
            )
            infos = []
            for result, tag, match, dispatch in zip(
                results, tags, accepted, dispatches
            ):
                info = dict(result[4])
                debug = match.telemetry.observation.action_debug
                gate_flags = int(debug[ActionDebugIndex.FLAGS])
                fire_suppressed = bool(
                    gate_flags & ML_FIRE_GATE_SUPPRESSED
                )
                info.update({
                    "batch_round_id": tag.round_id,
                    "policy_version": tag.policy_version,
                    "action_tick": tag.action_tick,
                    "authoritative_echo_tick": match.echo_tick,
                    "authoritative_echo_valid": True,
                    "authoritative_client_life_epoch": int(
                        match.telemetry.causal.client_life_epoch
                    ),
                    "authoritative_echo_stale": False,
                    "trainable_transition": True,
                    "causal_echo_valid": match.telemetry.causal.echo_valid,
                    "causal_facts_complete": match.telemetry.causal.facts_complete,
                    "causal_transition_trainable": (
                        match.telemetry.causal.transition_trainable
                    ),
                    "causal_role_playing": (
                        match.telemetry.causal.role_playing
                    ),
                    "causal_role_public_pm_normal": (
                        match.telemetry.causal.role_public_pm_normal
                    ),
                    "authoritative_action_generation": (
                        match.telemetry.causal.action_generation
                    ),
                    "authoritative_map_epoch": authoritative_map_epoch,
                    "stale_echoes_rejected": match.stale_echoes,
                    "mismatched_echoes_rejected": match.mismatched_echoes,
                    "fire_gate_protected": bool(
                        gate_flags & ML_FIRE_GATE_PROTECTED
                    ),
                    "fire_gate_target": bool(
                        gate_flags & ML_FIRE_GATE_TARGET
                    ),
                    "fire_gate_suppressed": fire_suppressed,
                    "requested_action_fire": bool(dispatch.action.fire),
                    "effective_action_fire": bool(
                        int(debug[ActionDebugIndex.FIRE])
                    ),
                    "effective_action_look_yaw": float(
                        debug[ActionDebugIndex.LOOK_YAW]
                    ),
                    "effective_action_look_pitch": float(
                        debug[ActionDebugIndex.LOOK_PITCH]
                    ),
                })
                infos.append(info)

            self._fire_gate_suppressions += sum(
                int(info["fire_gate_suppressed"]) for info in infos
            )

            frame_values = [match.telemetry.server_frame for match in accepted]
            frame_span = max(frame_values) - min(frame_values)
            if self.deterministic_frame_barrier and frame_span != 0:
                self._failed_rounds += 1
                raise AuthoritativeEchoError(
                    "deterministic frame barrier returned mixed client frames"
                )
            self._max_observed_frame_span = max(
                self._max_observed_frame_span, frame_span
            )
            self._rounds_accepted += 1
            self._transitions_accepted += len(results)
            return BatchRound(
                round_id=round_id,
                policy_version=version,
                observations=self._collate_observations(
                    [result[0] for result in results], self.vector
                ),
                rewards=np.asarray([result[1] for result in results], dtype=np.float32),
                terminated=np.asarray(
                    [result[2] for result in results], dtype=np.bool_
                ),
                truncated=np.asarray(
                    [result[3] for result in results], dtype=np.bool_
                ),
                infos=tuple(infos),
                tags=tags,
            )

    def step(self, actions: Sequence[Action], *, policy_version: int):
        return self.collect_round(
            actions, policy_version=policy_version
        ).gym_result()

    @property
    def metrics(self) -> BatchMetrics:
        return BatchMetrics(
            latest_policy_version=self._latest_policy_version,
            rounds_dispatched=self._rounds_dispatched,
            rounds_accepted=self._rounds_accepted,
            failed_rounds=self._failed_rounds,
            actions_dispatched=self._actions_dispatched,
            transitions_accepted=self._transitions_accepted,
            stale_policy_rounds_rejected=self._stale_policy_rounds_rejected,
            stale_echoes_rejected=self._stale_echoes_rejected,
            mismatched_echoes_rejected=self._mismatched_echoes_rejected,
            echo_timeouts=self._echo_timeouts,
            map_epoch_resyncs=self._map_epoch_resyncs,
            telemetry_gap_resyncs=self._telemetry_gap_resyncs,
            realtime_catchup_resyncs=self._realtime_catchup_resyncs,
            action_state_resyncs=self._action_state_resyncs,
            death_lifecycle_resyncs=self._death_lifecycle_resyncs,
            preflight_packets_drained=self._preflight_packets_drained,
            max_observed_frame_span=self._max_observed_frame_span,
            fire_gate_suppressions=self._fire_gate_suppressions,
        )

    def close(self):
        with self._round_lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
            first_error: BaseException | None = None
            for env in self.envs:
                try:
                    env.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            if first_error is not None:
                raise first_error

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def decode_policy_action(raw: Sequence[float]) -> Action:
    """Decode the frozen eight-value, three-way-posture action representation."""
    if len(raw) != 8:
        raise ValueError(f"expected eight action values, received {len(raw)}")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("policy action contains NaN or infinity")
    categorical = tuple(int(values[index]) for index in range(4, 8))
    if any(float(value) != values[index] for index, value in zip(range(4, 8), categorical)):
        raise ValueError("policy categorical actions must be exact integers")
    vertical, fire, hook, weapon = categorical
    if vertical not in (0, 1, 2) or fire not in (0, 1) or not 0 <= hook <= 3 or not 0 <= weapon <= 9:
        raise ValueError("policy categorical action is outside the frozen cardinality")
    return Action(
        move_forward=float(np.clip(values[0], -1, 1)),
        move_right=float(np.clip(values[1], -1, 1)),
        look_yaw=float(np.clip(values[2], -45, 45)),
        look_pitch=float(np.clip(values[3], -30, 30)),
        vertical_intent=VerticalIntent(vertical),
        fire=bool(fire),
        hook=hook,
        weapon=weapon,
    )


class Q2NetworkClientMultiEnv:
    """``Q2MultiEnv``-compatible adapter around a client batch.

    It intentionally does not own or restart q2ded: ``reset_slot`` is an
    episode-memory boundary around the latest live frame, matching the public
    server's continuous lifecycle.
    """

    def __init__(
        self,
        envs: Iterable[Q2NetworkClientEnv],
        *,
        max_ep_steps: int = 1000,
        initial_policy_version: int = 0,
        round_timeout: float = 2.0,
        max_rejected_echoes: int = 16,
        movement_tolerance: float = 0.05,
        look_tolerance: float = 0.25,
    ):
        env_tuple = tuple(envs)
        self._batch = Q2NetworkClientBatch(
            env_tuple,
            vector=True,
            round_timeout=round_timeout,
            max_rejected_echoes=max_rejected_echoes,
            movement_tolerance=movement_tolerance,
            look_tolerance=look_tolerance,
        )
        self.n_ml = len(env_tuple)
        self.max_ep_steps = max(1, int(max_ep_steps))
        self._policy_version = Q2NetworkClientBatch._policy_version(
            initial_policy_version
        )
        self._ep_steps = [0] * self.n_ml
        self._last_vectors: list[np.ndarray | None] = [None] * self.n_ml
        self._started = False
        self.active_map = "unknown"
        # Kept for PPO lattice checkpointing and tactical-sidecar wiring.
        self._spatial_rewards = [env._spatial for env in env_tuple]
        self._multires_clients = tuple(
            bool(getattr(env, "uses_multires_spatial", False))
            for env in env_tuple
        )

    def set_policy_version(self, policy_version: int) -> None:
        version = Q2NetworkClientBatch._policy_version(policy_version)
        if version < self._policy_version:
            raise StalePolicyVersionError(
                f"policy {version} is older than active policy "
                f"{self._policy_version}"
            )
        self._policy_version = version

    def reset_all(self) -> list[np.ndarray]:
        if self._started:
            return [self.reset_slot(index) for index in range(self.n_ml)]
        observations, infos = self._batch.reset()
        self._started = True
        vectors = [np.asarray(value, dtype=np.float32) for value in observations]
        self._last_vectors = [value.copy() for value in vectors]
        self._ep_steps = [0] * self.n_ml
        if infos:
            self.active_map = str(infos[0].get("map", "unknown"))
        return vectors

    def step_all(
        self,
        actions: Sequence[Sequence[float]],
        policy_version: int | None = None,
    ) -> list[tuple]:
        if policy_version is not None:
            self.set_policy_version(policy_version)
        decoded = [decode_policy_action(action) for action in actions]
        round_result = self._batch.collect_round(
            decoded, policy_version=self._policy_version
        )
        results = []
        for index in range(self.n_ml):
            observation = np.asarray(
                round_result.observations[index], dtype=np.float32
            )
            reward = float(round_result.rewards[index])
            terminated = bool(round_result.terminated[index])
            truncated = bool(round_result.truncated[index])
            info = dict(round_result.infos[index])
            map_epoch_resync = bool(info.get("map_epoch_resync", False))
            synchronization_boundary = bool(
                info.get("realtime_catchup_resync", False)
            ) or map_epoch_resync
            if synchronization_boundary:
                self._ep_steps[index] = 0
            else:
                self._ep_steps[index] += 1
            if (
                not synchronization_boundary
                and not terminated
                and self._ep_steps[index] >= self.max_ep_steps
            ):
                truncated = True
            if (
                (terminated or truncated)
                and not synchronization_boundary
                and not self._multires_clients[index]
            ):
                outcome_bonus, outcome_info = self._spatial_rewards[
                    index
                ].finalize_episode(
                    terminal_reason=int(info.get("terminal_reason", 0)),
                    truncated=truncated,
                )
                reward += outcome_bonus
                info["spatial_bonus"] = (
                    float(info.get("spatial_bonus", 0.0)) + outcome_bonus
                )
                info.update(outcome_info)
            self._last_vectors[index] = observation.copy()
            self.active_map = str(info.get("map", self.active_map))
            results.append((observation, reward, terminated, truncated, info))
        return results

    def reset_slot(self, slot_idx: int) -> np.ndarray:
        if not 0 <= slot_idx < self.n_ml:
            raise IndexError(f"client slot index {slot_idx} is out of range")
        if not self._started:
            raise RuntimeError("call reset_all() before reset_slot()")
        cached = self._last_vectors[slot_idx]
        if cached is None:
            raise RuntimeError("admitted policy vector is unavailable")
        self._ep_steps[slot_idx] = 0
        # Episode reset is a policy-memory boundary only. Sampling the live
        # provider here would advance Dyn/source state without an admitted
        # authoritative transition.
        return cached.copy()

    @property
    def metrics(self) -> BatchMetrics:
        return self._batch.metrics

    def close(self):
        try:
            self._batch.close()
        finally:
            self._started = False


def _build_network_client_envs(
    *,
    n_clients: int,
    server: str,
    telemetry_server: str,
    telemetry_token: str,
    client_binary: str,
    client_root: str,
    client_data_root: str | None = None,
    harness_host: str = "127.0.0.1",
    harness_port_base: int = 39000,
    qport_base: int = 49000,
    client_id_prefix: str = "trainer",
    name_prefix: str = "ml",
    game: str = "lithium",
    client_timeout: float = 8.0,
    spatial_seed: int | None = None,
    debug: bool = False,
    deterministic_frame_barrier: bool = False,
    start_observer: bool = False,
    extra_args: tuple[str, ...] = (),
    multires_spatial_provider_factories: Sequence[
        SpatialFeatureProviderFactory
    ] | None = None,
    expected_runtime_manifest_sha256: str | None = None,
) -> tuple[Q2NetworkClientEnv, ...]:
    """Build strict one-client/one-factory production bindings."""
    count = int(n_clients)
    if count <= 0:
        raise ValueError("n_clients must be positive")
    if deterministic_frame_barrier and start_observer:
        raise ValueError(
            "deterministic frame-barrier clients cannot start as observers"
        )
    if (
        multires_spatial_provider_factories is None
        or len(multires_spatial_provider_factories) != count
        or not expected_runtime_manifest_sha256
    ):
        raise ValueError(
            "production network builder requires one multires provider factory "
            "per client and an attested runtime digest; legacy selection is retired"
        )
    factories = tuple(multires_spatial_provider_factories)
    if len({id(factory) for factory in factories}) != count:
        raise ValueError(
            "each network client requires its own provider factory instance"
        )
    if any(not callable(getattr(factory, "create", None)) for factory in factories):
        raise TypeError("every multires provider factory must implement create()")
    prefix = client_id_prefix.encode("ascii", errors="strict").decode("ascii")
    suffix_size = len(f"-{count - 1:02d}")
    if not prefix or len(prefix.encode("ascii")) + suffix_size >= 40:
        raise ValueError("client_id_prefix is too long for the 39-byte wire ID")
    return tuple(
        Q2NetworkClientEnv(
            server=server,
            telemetry_server=telemetry_server,
            telemetry_token=telemetry_token,
            client_binary=client_binary,
            client_root=client_root,
            client_data_root=client_data_root,
            harness_host=harness_host,
            harness_port=int(harness_port_base) + index,
            qport=int(qport_base) + index,
            client_id=f"{prefix}-{index:02d}",
            name=f"{name_prefix}-{index:02d}",
            game=game,
            timeout=client_timeout,
            spatial_seed=(
                None if spatial_seed is None else int(spatial_seed) + index * 1009
            ),
            debug=debug,
            deterministic_frame_barrier=deterministic_frame_barrier,
            extra_args=extra_args,
            multires_spatial_provider_factory=(
                factories[index]
            ),
            expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
        )
        for index in range(count)
    )


def build_network_client_batch(
    *,
    n_clients: int,
    server: str,
    telemetry_server: str,
    telemetry_token: str,
    client_binary: str,
    client_root: str,
    client_data_root: str | None = None,
    harness_host: str = "127.0.0.1",
    harness_port_base: int = 39000,
    qport_base: int = 49000,
    client_id_prefix: str = "trainer",
    name_prefix: str = "ml",
    game: str = "lithium",
    client_timeout: float = 8.0,
    round_timeout: float = 2.0,
    max_rejected_echoes: int = 16,
    movement_tolerance: float = 0.05,
    look_tolerance: float = 0.25,
    spatial_seed: int | None = None,
    debug: bool = False,
    deterministic_frame_barrier: bool = False,
    start_observer: bool = False,
    extra_args: tuple[str, ...] = (),
    multires_spatial_provider_factories: Sequence[
        SpatialFeatureProviderFactory
    ] | None = None,
    expected_runtime_manifest_sha256: str | None = None,
) -> Q2NetworkClientBatch:
    """Build the direct synchronous multires collector surface.

    The returned batch is the public production object consumed by
    ``MultiresSynchronousCollector``; callers do not need the Gym adapter or a
    private ``_batch`` escape hatch.
    """
    envs = _build_network_client_envs(
        n_clients=n_clients,
        server=server,
        telemetry_server=telemetry_server,
        telemetry_token=telemetry_token,
        client_binary=client_binary,
        client_root=client_root,
        client_data_root=client_data_root,
        harness_host=harness_host,
        harness_port_base=harness_port_base,
        qport_base=qport_base,
        client_id_prefix=client_id_prefix,
        name_prefix=name_prefix,
        game=game,
        client_timeout=client_timeout,
        spatial_seed=spatial_seed,
        debug=debug,
        deterministic_frame_barrier=deterministic_frame_barrier,
        start_observer=start_observer,
        extra_args=extra_args,
        multires_spatial_provider_factories=multires_spatial_provider_factories,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
    )
    return Q2NetworkClientBatch(
        envs,
        vector=True,
        round_timeout=round_timeout,
        max_rejected_echoes=max_rejected_echoes,
        movement_tolerance=movement_tolerance,
        look_tolerance=look_tolerance,
        deterministic_frame_barrier=deterministic_frame_barrier,
    )


def build_network_client_multi_env(
    *,
    n_clients: int,
    server: str,
    telemetry_server: str,
    telemetry_token: str,
    client_binary: str,
    client_root: str,
    client_data_root: str | None = None,
    harness_host: str = "127.0.0.1",
    harness_port_base: int = 39000,
    qport_base: int = 49000,
    client_id_prefix: str = "trainer",
    name_prefix: str = "ml",
    game: str = "lithium",
    client_timeout: float = 8.0,
    round_timeout: float = 2.0,
    max_rejected_echoes: int = 16,
    movement_tolerance: float = 0.05,
    look_tolerance: float = 0.25,
    max_ep_steps: int = 1000,
    initial_policy_version: int = 0,
    spatial_seed: int | None = None,
    debug: bool = False,
    extra_args: tuple[str, ...] = (),
    multires_spatial_provider_factories: Sequence[
        SpatialFeatureProviderFactory
    ] | None = None,
    expected_runtime_manifest_sha256: str | None = None,
) -> Q2NetworkClientMultiEnv:
    """Build the compatibility Gym adapter around strict multires clients."""
    envs = _build_network_client_envs(
        n_clients=n_clients,
        server=server,
        telemetry_server=telemetry_server,
        telemetry_token=telemetry_token,
        client_binary=client_binary,
        client_root=client_root,
        client_data_root=client_data_root,
        harness_host=harness_host,
        harness_port_base=harness_port_base,
        qport_base=qport_base,
        client_id_prefix=client_id_prefix,
        name_prefix=name_prefix,
        game=game,
        client_timeout=client_timeout,
        spatial_seed=spatial_seed,
        debug=debug,
        extra_args=extra_args,
        multires_spatial_provider_factories=multires_spatial_provider_factories,
        expected_runtime_manifest_sha256=expected_runtime_manifest_sha256,
    )
    return Q2NetworkClientMultiEnv(
        envs,
        max_ep_steps=max_ep_steps,
        initial_policy_version=initial_policy_version,
        round_timeout=round_timeout,
        max_rejected_echoes=max_rejected_echoes,
        movement_tolerance=movement_tolerance,
        look_tolerance=look_tolerance,
    )
