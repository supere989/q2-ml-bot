"""Event-causal reward reducer for the multires Atlas policy generation.

The reducer consumes admitted authoritative facts.  It does not inspect the
298-float policy vector, so advisory Atlas/Dyn/guide fields cannot become aim or
fire rewards.  Positive credit is attached to one-shot outcomes or monotonic
episode milestones; actuator frequency never pays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

from .multires_contract import (
    POSTURE_CROUCH_OR_DOWN,
    POSTURE_JUMP_OR_UP,
    POSTURE_NEUTRAL,
)

FIRE_SUPPRESSION_NONE = 0
FIRE_SUPPRESSION_PROTECTED = 1
FIRE_SUPPRESSION_UNALIGNED = 2


class RewardAdmissionError(ValueError):
    """Raised when causal reward attribution cannot be proven."""


@dataclass(frozen=True)
class CausalRewardConfig:
    alignment_acquisition_reward: float = 0.02
    damage_reward: float = 0.003
    repeated_hit_damage_reward: float = 0.001
    repeated_hit_cap: int = 6
    hit_continuity_ticks: int = 30
    kill_reward: float = 1.0
    hidden_fire_penalty: float = 0.025
    environmental_damage_penalty: float = 0.004
    environmental_death_penalty: float = 1.0
    hazard_progress_reward: float = 0.04
    hazard_progress_milestones: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    safe_arrival_reward: float = 0.15
    necessary_hook_arrival_reward: float = 0.08
    invalid_hook_penalty: float = 0.02
    hazard_timeout_ticks: int = 100
    safe_clearance_threshold: float = 0.5
    safe_rearm_ticks: int = 30
    crouch_edge_entry_reward: float = 0.03
    crouch_edge_completion_reward: float = 0.06
    unnecessary_crouch_penalty: float = 0.01

    def validate(self) -> None:
        scalar_values = {
            name: value
            for name, value in self.__dict__.items()
            if name != "hazard_progress_milestones"
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in scalar_values.values()
        ):
            raise ValueError("reward configuration contains a non-finite value")
        positive_names = (
            "alignment_acquisition_reward",
            "damage_reward",
            "repeated_hit_damage_reward",
            "kill_reward",
            "hidden_fire_penalty",
            "environmental_damage_penalty",
            "environmental_death_penalty",
            "hazard_progress_reward",
            "safe_arrival_reward",
            "necessary_hook_arrival_reward",
            "invalid_hook_penalty",
            "crouch_edge_entry_reward",
            "crouch_edge_completion_reward",
            "unnecessary_crouch_penalty",
        )
        if any(float(getattr(self, name)) < 0.0 for name in positive_names):
            raise ValueError("reward magnitudes must be nonnegative")
        if self.repeated_hit_cap < 0 or self.hit_continuity_ticks < 1:
            raise ValueError("hit streak bounds must be positive")
        if self.hazard_timeout_ticks < 1 or self.safe_rearm_ticks < 1:
            raise ValueError("hazard timeout/rearm ticks must be positive")
        if self.safe_clearance_threshold < 0.0:
            raise ValueError("safe_clearance_threshold must be nonnegative")
        milestones = tuple(float(value) for value in self.hazard_progress_milestones)
        if (
            not milestones
            or tuple(sorted(set(milestones))) != milestones
            or milestones[0] <= 0.0
            or milestones[-1] > 1.0
        ):
            raise ValueError("hazard milestones must be unique ascending values in (0, 1]")


@dataclass(frozen=True)
class CausalRewardFrame:
    tick: int
    client_life_epoch: int
    authoritative_echo_valid: bool
    trainable_transition: bool
    state_resync: bool = False
    teacher_field_violations: int = 0

    # Combat ladder. target_id/epoch are debug attribution, never policy input.
    target_id: int = 0
    target_epoch: int = 0
    actionable_exposure: bool = False
    post_command_aligned: bool = False
    fire_permitted: bool = False
    fire_requested: bool = False
    fire_suppressed: bool = False
    fire_suppression_reason: int = FIRE_SUPPRESSION_NONE
    fire_executed: bool = False
    damage_dealt: float = 0.0
    target_killed: bool = False
    aim_yaw_error_deg: Optional[float] = None
    aim_pitch_error_deg: Optional[float] = None

    # Atlas hazard component identity must remain stable through the
    # safe-clearance rearm interval, even after the immediate hazard is absent.
    hazard_component_id: int = 0
    hazard_component_epoch: int = 0
    environmental_source_id: int = 0
    environmental_source_epoch: int = 0
    environmental_mod: int = 0
    environmental_hazard_evidence: bool = False
    cost_to_safety: Optional[float] = None
    safe_clearance: float = 0.0
    environmental_source_cleared: bool = False
    environmental_damage: float = 0.0
    environmental_death: bool = False
    hook_attempted: bool = False
    hook_pending: bool = False
    hook_attached: bool = False
    hook_valid: bool = False
    hook_invalid: bool = False
    hook_was_necessary: bool = False
    hook_necessity_known: bool = False
    hook_zone_id: int = 0
    hook_attempt_tick: int = 0
    hook_action_generation: int = 0
    action_generation: int = 0

    # Posture/crouch outcomes. requested_vertical is the three-way enum.
    requested_vertical: int = POSTURE_NEUTRAL
    actual_ducked: bool = False
    standing_blocked: bool = False
    water_vertical_mode: bool = False
    crouch_edge_id: int = 0
    crouch_edge_epoch: int = 0
    crouch_edge_active: bool = False
    crouch_edge_entered: bool = False
    crouch_edge_completed: bool = False

    def validate(self) -> None:
        if self.tick < 0 or self.client_life_epoch < 0:
            raise RewardAdmissionError("tick and client_life_epoch must be nonnegative")
        if any(value < 0 for value in (
            self.target_id, self.target_epoch, self.hazard_component_id,
            self.hazard_component_epoch, self.environmental_source_id,
            self.environmental_source_epoch, self.environmental_mod, self.crouch_edge_id,
            self.crouch_edge_epoch, self.hook_zone_id, self.hook_attempt_tick,
            self.hook_action_generation, self.action_generation,
        )):
            raise RewardAdmissionError("persistent identities must be nonnegative")
        if (
            not self.authoritative_echo_valid
            or not self.trainable_transition
            or self.state_resync
        ):
            raise RewardAdmissionError(
                "reward requires a trainable authoritative-echo transition"
            )
        if self.action_generation <= 0:
            raise RewardAdmissionError(
                "network causal reward requires authoritative action_generation"
            )
        if self.teacher_field_violations != 0:
            raise RewardAdmissionError(
                "teacher-only fields cannot enter the public reward reducer"
            )
        if self.requested_vertical not in (
            POSTURE_CROUCH_OR_DOWN, POSTURE_NEUTRAL, POSTURE_JUMP_OR_UP
        ):
            raise RewardAdmissionError("requested_vertical is not a valid posture enum")
        damage_values = (self.damage_dealt, self.environmental_damage)
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in damage_values
        ):
            raise RewardAdmissionError("damage values must be finite and nonnegative")
        if not math.isfinite(float(self.safe_clearance)):
            raise RewardAdmissionError("signed safe clearance must be finite")
        if self.cost_to_safety is not None and (
            not math.isfinite(float(self.cost_to_safety))
            or float(self.cost_to_safety) < 0.0
        ):
            raise RewardAdmissionError("cost_to_safety must be finite and nonnegative")
        for value in (self.aim_yaw_error_deg, self.aim_pitch_error_deg):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise RewardAdmissionError("aim errors must be finite and nonnegative")
        if (self.damage_dealt > 0.0 or self.target_killed) and (
            self.target_id <= 0 or self.target_epoch <= 0
        ):
            raise RewardAdmissionError(
                "hit/kill reward requires persistent target_id and target_epoch"
            )
        if (
            self.actionable_exposure
            or self.post_command_aligned
            or self.aim_yaw_error_deg is not None
            or self.aim_pitch_error_deg is not None
        ) and (self.target_id <= 0 or self.target_epoch <= 0):
            raise RewardAdmissionError(
                "target geometry requires persistent target_id and target_epoch"
            )
        if self.post_command_aligned and not self.actionable_exposure:
            raise RewardAdmissionError(
                "post-command alignment requires actionable exposure"
            )
        if self.fire_suppression_reason not in (
            FIRE_SUPPRESSION_NONE,
            FIRE_SUPPRESSION_PROTECTED,
            FIRE_SUPPRESSION_UNALIGNED,
        ):
            raise RewardAdmissionError("unknown fire suppression reason")
        if self.fire_suppressed != (
            self.fire_suppression_reason != FIRE_SUPPRESSION_NONE
        ):
            raise RewardAdmissionError(
                "fire suppression flag and reason must agree"
            )
        if self.fire_suppressed and (
            not self.fire_requested or self.fire_executed
        ):
            raise RewardAdmissionError(
                "suppressed fire requires an unexecuted fire request"
            )
        if not self.fire_suppressed and (
            self.fire_requested != self.fire_executed
        ):
            raise RewardAdmissionError(
                "unsuppressed fire request must equal executed fire"
            )
        if self.fire_permitted and not self.actionable_exposure:
            raise RewardAdmissionError(
                "fire permission requires actionable exposure"
            )
        if (
            self.environmental_hazard_evidence
            or (
                self.cost_to_safety is not None
                and self.cost_to_safety > 0.0
            )
        ) and self.hazard_component_id <= 0:
            raise RewardAdmissionError(
                "Atlas recovery evidence requires a stable hazard_component_id"
            )
        if self.hazard_component_id > 0 and self.hazard_component_epoch <= 0:
            raise RewardAdmissionError(
                "hazard component identity requires component epoch"
            )
        if (
            self.environmental_source_cleared
            or self.environmental_damage > 0.0
            or self.environmental_death
        ) and (
            self.environmental_source_id <= 0
            or self.environmental_source_epoch <= 0
            or self.environmental_mod <= 0
        ):
            raise RewardAdmissionError(
                "environmental source event requires source id/epoch and MOD"
            )
        if (
            self.crouch_edge_active
            or self.crouch_edge_entered
            or self.crouch_edge_completed
        ) and self.crouch_edge_id <= 0:
            raise RewardAdmissionError("crouch outcome requires crouch_edge_id")
        if self.crouch_edge_id > 0 and self.crouch_edge_epoch <= 0:
            raise RewardAdmissionError("crouch identity requires crouch_edge_epoch")
        hook_event = bool(
            self.hook_attempted
            or self.hook_pending
            or self.hook_attached
            or self.hook_valid
            or self.hook_invalid
            or self.hook_necessity_known
            or self.hook_was_necessary
        )
        if hook_event != bool(
            self.hook_attempt_tick > 0 and self.hook_action_generation > 0
        ):
            raise RewardAdmissionError(
                "hook events require one exact attempt tick/generation origin"
            )
        if self.hook_pending and (
            self.hook_attached or self.hook_invalid or self.hook_necessity_known
        ):
            raise RewardAdmissionError(
                "pending hook cannot already have a terminal outcome"
            )
        if self.hook_attached and not self.hook_valid:
            raise RewardAdmissionError("attached hook must be valid")
        if self.hook_invalid and (
            self.hook_attached
            or self.hook_valid
            or self.hook_necessity_known
            or self.hook_was_necessary
        ):
            raise RewardAdmissionError(
                "invalid hook cannot also carry a valid or necessity outcome"
            )
        if self.hook_necessity_known and not self.hook_valid:
            raise RewardAdmissionError(
                "hook necessity verdict requires a valid hook outcome"
            )
        if self.hook_was_necessary and (
            not self.hook_necessity_known or not self.hook_valid or self.hook_zone_id <= 0
        ):
            raise RewardAdmissionError(
                "necessary hook requires known verdict, valid hook, and hook zone"
            )


@dataclass(frozen=True)
class CausalRewardResult:
    reward: float
    metrics: dict[str, float]


@dataclass
class _HazardEpisode:
    opened_tick: int
    initial_cost: float
    best_cost: float
    open: bool = True
    rearm_blocked: bool = False
    safe_ticks: int = 0
    credited_milestones: set[float] = field(default_factory=set)
    arrival_paid: bool = False
    hook_used: bool = False
    necessary_hook_used: bool = False
    necessary_hook_credit_paid: bool = False


@dataclass
class _PendingHook:
    life_epoch: int
    attempt_tick: int
    action_generation: int
    hazard_key: Optional[tuple[int, int, int]]
    valid: bool = False
    attached: bool = False
    zone_id: int = 0


class CausalRewardReducer:
    """Reduce one strictly increasing authoritative frame stream to rewards."""

    # The empty declaration is intentionally testable: no positive coefficient
    # is indexed by strafe/jump/crouch/hook action frequency.
    positive_actuator_rate_rewards: tuple[str, ...] = ()

    def __init__(self, config: CausalRewardConfig = CausalRewardConfig()):
        config.validate()
        self.config = config
        self._last_tick = -1
        self._life_epoch: Optional[int] = None
        self._alignment_paid: set[tuple[int, int, int]] = set()
        self._hit_target: Optional[tuple[int, int, int]] = None
        self._hit_streak = 0
        self._last_hit_tick = -1
        self._hazards: dict[tuple[int, int, int], _HazardEpisode] = {}
        self._active_hazard_key: Optional[tuple[int, int, int]] = None
        self._pending_hook: Optional[_PendingHook] = None
        self._crouch_entered: set[tuple[int, int, int]] = set()
        self._crouch_completed: set[tuple[int, int, int]] = set()
        self._previous_actual_ducked = False

    def reset_stream(self) -> None:
        self.__init__(self.config)

    @staticmethod
    def _target_key(frame: CausalRewardFrame) -> Optional[tuple[int, int, int]]:
        if frame.target_id <= 0 or frame.target_epoch <= 0:
            return None
        return (frame.client_life_epoch, frame.target_id, frame.target_epoch)

    def _life_boundary(self, frame: CausalRewardFrame) -> None:
        if self._life_epoch == frame.client_life_epoch:
            return
        if self._life_epoch is not None and frame.client_life_epoch < self._life_epoch:
            raise RewardAdmissionError("client_life_epoch cannot roll back")
        self._life_epoch = frame.client_life_epoch
        # Reducers are one-client streams.  A new life makes every prior-life
        # target, hazard, and crouch key permanently ineligible, so retaining
        # them only creates unbounded season-long memory growth.
        self._alignment_paid.clear()
        self._hit_target = None
        self._hit_streak = 0
        self._last_hit_tick = -1
        self._hazards.clear()
        self._active_hazard_key = None
        self._pending_hook = None
        self._crouch_entered.clear()
        self._crouch_completed.clear()
        self._previous_actual_ducked = False

    def _combat(self, frame: CausalRewardFrame, metrics: dict[str, float]) -> float:
        reward = 0.0
        target_key = self._target_key(frame)
        aligned_event = bool(
            target_key is not None
            and frame.actionable_exposure
            and frame.post_command_aligned
            and target_key not in self._alignment_paid
        )
        if aligned_event:
            self._alignment_paid.add(target_key)
            reward += self.config.alignment_acquisition_reward

        hidden_fire = bool(frame.fire_requested and not frame.fire_permitted)
        if hidden_fire:
            reward -= self.config.hidden_fire_penalty

        hit = frame.damage_dealt > 0.0
        repeated_hit = False
        streak = 0
        if hit:
            assert target_key is not None  # frame validation established this
            if (
                target_key == self._hit_target
                and 0 <= frame.tick - self._last_hit_tick
                <= self.config.hit_continuity_ticks
            ):
                self._hit_streak += 1
            else:
                self._hit_target = target_key
                self._hit_streak = 1
            self._last_hit_tick = frame.tick
            streak = self._hit_streak
            repeated_hit = streak > 1
            reward += frame.damage_dealt * self.config.damage_reward
            reward += (
                frame.damage_dealt
                * self.config.repeated_hit_damage_reward
                * min(max(0, streak - 1), self.config.repeated_hit_cap)
            )
        if frame.target_killed:
            reward += self.config.kill_reward
            self._hit_target = None
            self._hit_streak = 0
            self._last_hit_tick = -1

        metrics.update({
            "combat/actionable_exposure": float(frame.actionable_exposure),
            "combat/post_command_alignment": float(frame.post_command_aligned),
            "combat/alignment_acquisition": float(aligned_event),
            "combat/fire_permission": float(frame.fire_permitted),
            "combat/fire_requested": float(frame.fire_requested),
            "combat/fire_suppressed": float(frame.fire_suppressed),
            "combat/fire_suppression_reason": float(frame.fire_suppression_reason),
            "combat/executed_fire": float(frame.fire_executed),
            "combat/hidden_fire": float(hidden_fire),
            "combat/hit": float(hit),
            "combat/repeated_hit": float(repeated_hit),
            "combat/hit_streak": float(streak),
            "combat/kill": float(frame.target_killed),
            "combat/aim_yaw_error_deg": float(frame.aim_yaw_error_deg or 0.0),
            "combat/aim_pitch_error_deg": float(frame.aim_pitch_error_deg or 0.0),
        })
        return reward

    def _hazard(self, frame: CausalRewardFrame, metrics: dict[str, float]) -> float:
        reward = 0.0
        progress_credits = 0
        arrival_credit = False
        necessary_hook_credit = False
        hook_arrival = False
        recovery_ticks = 0
        timeout = False
        rearmed = False

        reward -= frame.environmental_damage * self.config.environmental_damage_penalty
        if frame.environmental_death:
            reward -= self.config.environmental_death_penalty

        key: Optional[tuple[int, int, int]] = None
        episode: Optional[_HazardEpisode] = None
        if frame.hazard_component_id > 0:
            key = (
                frame.client_life_epoch,
                frame.hazard_component_id,
                frame.hazard_component_epoch,
            )
            episode = self._hazards.get(key)
            self._active_hazard_key = key
        elif self._active_hazard_key is not None:
            key = self._active_hazard_key
            episode = self._hazards.get(key)

        # A closed region only rearms after explicit evidence that this client
        # stayed beyond the safe-clearance threshold for 30 consecutive ticks.
        if episode is not None and episode.rearm_blocked:
            safely_clear = bool(
                not frame.environmental_hazard_evidence
                and frame.safe_clearance >= self.config.safe_clearance_threshold
            )
            episode.safe_ticks = episode.safe_ticks + 1 if safely_clear else 0
            if episode.safe_ticks >= self.config.safe_rearm_ticks:
                assert key is not None
                del self._hazards[key]
                episode = None
                if self._active_hazard_key == key:
                    self._active_hazard_key = None
                rearmed = True

        if (
            frame.environmental_hazard_evidence
            and episode is None
        ):
            initial_cost = frame.cost_to_safety
            assert key is not None
            episode = _HazardEpisode(
                opened_tick=frame.tick,
                initial_cost=float(initial_cost or 0.0),
                best_cost=float(initial_cost or 0.0),
            )
            self._hazards[key] = episode

        hook_event = bool(
            frame.hook_attempted
            or frame.hook_pending
            or frame.hook_attached
            or frame.hook_valid
            or frame.hook_invalid
            or frame.hook_necessity_known
            or frame.hook_was_necessary
        )
        origin = (
            frame.client_life_epoch,
            frame.hook_attempt_tick,
            frame.hook_action_generation,
        )
        if frame.hook_attempted:
            if self._pending_hook is not None:
                raise RewardAdmissionError(
                    "new hook attempt overlaps an unresolved pending attempt"
                )
            self._pending_hook = _PendingHook(
                life_epoch=frame.client_life_epoch,
                attempt_tick=frame.hook_attempt_tick,
                action_generation=frame.hook_action_generation,
                hazard_key=key,
                valid=frame.hook_valid,
                attached=frame.hook_attached,
                zone_id=frame.hook_zone_id,
            )
        elif hook_event:
            pending = self._pending_hook
            if pending is None or origin != (
                pending.life_epoch,
                pending.attempt_tick,
                pending.action_generation,
            ):
                raise RewardAdmissionError("orphan or stale delayed hook outcome")
            if pending.hazard_key != key:
                raise RewardAdmissionError(
                    "delayed hook outcome crossed its Atlas hazard episode"
                )

        if hook_event:
            pending = self._pending_hook
            assert pending is not None
            if pending.zone_id and frame.hook_zone_id and (
                pending.zone_id != frame.hook_zone_id
            ):
                raise RewardAdmissionError("delayed hook outcome changed hook zone")
            if frame.hook_zone_id:
                pending.zone_id = frame.hook_zone_id
            pending.valid = bool(pending.valid or frame.hook_valid)
            pending.attached = bool(pending.attached or frame.hook_attached)
            if pending.valid and episode is not None:
                episode.hook_used = True
            if frame.hook_invalid:
                reward -= self.config.invalid_hook_penalty
                self._pending_hook = None
            elif frame.hook_necessity_known:
                if frame.hook_was_necessary:
                    if episode is None or not pending.valid:
                        raise RewardAdmissionError(
                            "necessary hook outcome lacks its Atlas hazard episode"
                        )
                    episode.necessary_hook_used = True
                self._pending_hook = None
            elif frame.hook_attached:
                self._pending_hook = None

        safe_arrival = bool(
            episode is not None
            and episode.open
            and not frame.environmental_hazard_evidence
            and frame.cost_to_safety == 0.0
            and frame.safe_clearance >= 0.0
        )
        if episode is not None and episode.open:
            if frame.cost_to_safety is not None and episode.initial_cost > 0.0:
                episode.best_cost = min(episode.best_cost, float(frame.cost_to_safety))
                progress = max(
                    0.0,
                    min(1.0, 1.0 - episode.best_cost / episode.initial_cost),
                )
                for milestone in self.config.hazard_progress_milestones:
                    if progress >= milestone and milestone not in episode.credited_milestones:
                        episode.credited_milestones.add(milestone)
                        reward += self.config.hazard_progress_reward
                        progress_credits += 1

            if safe_arrival:
                if not episode.arrival_paid:
                    episode.arrival_paid = True
                    reward += self.config.safe_arrival_reward
                    arrival_credit = True
                    recovery_ticks = frame.tick - episode.opened_tick
                    hook_arrival = episode.hook_used
                    if episode.hook_used and episode.necessary_hook_used:
                        reward += self.config.necessary_hook_arrival_reward
                        necessary_hook_credit = True
                        episode.necessary_hook_credit_paid = True
                episode.open = False
                episode.rearm_blocked = True
                episode.safe_ticks = 0
            elif frame.environmental_death:
                episode.open = False
                episode.rearm_blocked = True
                episode.safe_ticks = 0
            elif frame.tick - episode.opened_tick >= self.config.hazard_timeout_ticks:
                # Do not reset best_cost or credited milestones. Re-entry into
                # this region cannot replay them before a real safe rearm.
                episode.open = False
                episode.rearm_blocked = True
                episode.safe_ticks = 0
                timeout = True

        # A necessity oracle may resolve after safe arrival.  The pending
        # origin still binds it to this exact Atlas episode, so credit the
        # already-admitted arrival once and never transfer it to another one.
        if (
            episode is not None
            and episode.arrival_paid
            and episode.necessary_hook_used
            and not episode.necessary_hook_credit_paid
        ):
            reward += self.config.necessary_hook_arrival_reward
            necessary_hook_credit = True
            episode.necessary_hook_credit_paid = True

        metrics.update({
            "hazard/evidence": float(frame.environmental_hazard_evidence),
            "hazard/progress_credits": float(progress_credits),
            "hazard/safe_arrival": float(safe_arrival),
            "hazard/environmental_source_cleared": float(
                frame.environmental_source_cleared
            ),
            "hazard/safe_arrival_credit": float(arrival_credit),
            "hazard/recovery_ticks": float(recovery_ticks),
            "hazard/environmental_damage": float(frame.environmental_damage),
            "hazard/environmental_death": float(frame.environmental_death),
            "hazard/timeout": float(timeout),
            "hazard/rearmed": float(rearmed),
            "hook/attempt": float(frame.hook_attempted),
            "hook/pending": float(frame.hook_pending),
            "hook/attached": float(frame.hook_attached),
            "hook/invalid_attempt": float(frame.hook_invalid),
            "hook/necessity_known": float(frame.hook_necessity_known),
            "hook/necessary": float(frame.hook_was_necessary),
            "hook/necessary_safe_arrival_credit": float(necessary_hook_credit),
            "hook/safe_arrival_with_hook": float(hook_arrival),
        })
        return reward

    def _posture(self, frame: CausalRewardFrame, metrics: dict[str, float]) -> float:
        reward = 0.0
        entry_credit = False
        completion_credit = False
        unnecessary_entry = False
        key = (
            frame.client_life_epoch, frame.crouch_edge_id, frame.crouch_edge_epoch
        )
        if frame.crouch_edge_entered and key not in self._crouch_entered:
            if not frame.actual_ducked:
                raise RewardAdmissionError(
                    "crouch-edge entry credit requires authoritative actual_ducked"
                )
            self._crouch_entered.add(key)
            reward += self.config.crouch_edge_entry_reward
            entry_credit = True
        if frame.crouch_edge_completed and key not in self._crouch_completed:
            if key not in self._crouch_entered:
                raise RewardAdmissionError(
                    "crouch-edge completion requires a prior admitted entry"
                )
            self._crouch_completed.add(key)
            reward += self.config.crouch_edge_completion_reward
            completion_credit = True

        actual_duck_entry = bool(frame.actual_ducked and not self._previous_actual_ducked)
        if (
            actual_duck_entry
            and not frame.crouch_edge_active
            and not frame.standing_blocked
            and not frame.water_vertical_mode
        ):
            reward -= self.config.unnecessary_crouch_penalty
            unnecessary_entry = True
        self._previous_actual_ducked = frame.actual_ducked

        metrics.update({
            "posture/requested_crouch_or_down": float(
                frame.requested_vertical == POSTURE_CROUCH_OR_DOWN
            ),
            "posture/requested_neutral": float(
                frame.requested_vertical == POSTURE_NEUTRAL
            ),
            "posture/requested_jump_or_up": float(
                frame.requested_vertical == POSTURE_JUMP_OR_UP
            ),
            "posture/actual_ducked": float(frame.actual_ducked),
            "posture/standing_blocked": float(frame.standing_blocked),
            "posture/water_vertical_mode": float(frame.water_vertical_mode),
            "posture/crouch_edge_active": float(frame.crouch_edge_active),
            "posture/crouch_edge_entry_credit": float(entry_credit),
            "posture/crouch_edge_completion_credit": float(completion_credit),
            "posture/unnecessary_crouch_entry": float(unnecessary_entry),
        })
        return reward

    def step(self, frame: CausalRewardFrame) -> CausalRewardResult:
        frame.validate()
        if frame.tick <= self._last_tick:
            raise RewardAdmissionError(
                f"reward tick {frame.tick} does not advance past {self._last_tick}"
            )
        self._last_tick = frame.tick
        self._life_boundary(frame)
        metrics: dict[str, float] = {"reward/admitted_transition": 1.0}
        combat = self._combat(frame, metrics)
        hazard = self._hazard(frame, metrics)
        posture = self._posture(frame, metrics)
        reward = combat + hazard + posture
        metrics.update({
            "reward/combat": float(combat),
            "reward/hazard": float(hazard),
            "reward/posture": float(posture),
            "reward/total": float(reward),
        })
        return CausalRewardResult(reward=float(reward), metrics=metrics)
