"""Required B5 telemetry aggregation for one multires quality season."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Mapping, Optional, Sequence

from .multires_contract import GUIDE_CLASS_NAMES
from .multires_lineage import LineageError
from .multires_reward import CausalRewardFrame, CausalRewardResult


SEASON_SCHEMA = "multires-atlas-season-v1"
QUERY_TIMING_COMPONENTS = (
    "dyn_query_us",
    "atlas_lookup_us",
    "recovery_query_us",
    "guide_query_us",
)


@dataclass
class MultiresSeasonMetrics:
    season_id: str
    atlas_sha256: str
    policy_start_version: int
    counters: Counter = field(default_factory=Counter)
    sums: Counter = field(default_factory=Counter)
    runtime_maxima: dict[str, float] = field(default_factory=dict)
    query_timings: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in QUERY_TIMING_COMPONENTS}
    )

    def __post_init__(self) -> None:
        if not self.season_id:
            raise ValueError("season_id is required")
        if len(self.atlas_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.atlas_sha256
        ):
            raise ValueError("atlas_sha256 must be lowercase SHA-256")
        if self.policy_start_version < 0:
            raise ValueError("policy_start_version must be nonnegative")

    def observe(
        self,
        frame: CausalRewardFrame,
        result: CausalRewardResult,
        *,
        command_echo_match: bool,
        state_resync: bool = False,
        guide_dropped: Sequence[bool] = (False, False, False, False),
        guide_classes: Sequence[Optional[int]] = (None, None, None, None),
        global_guide_drop: bool = False,
        teacher_field_violations: int = 0,
        forward_command: float = 0.0,
        backward_command: float = 0.0,
        movement_speed: float = 0.0,
        true_view_pitch_deg: Optional[float] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        if teacher_field_violations:
            raise LineageError("teacher-only field observed on public trainer conduit")
        if not command_echo_match or state_resync:
            raise LineageError("noncausal/resync transition cannot enter season metrics")
        if len(guide_dropped) != 4 or len(guide_classes) != 4:
            raise ValueError("guide telemetry requires four candidate slots")
        if not all(math.isfinite(float(value)) for value in (
            forward_command, backward_command, movement_speed
        )):
            raise ValueError("movement telemetry must be finite")
        if true_view_pitch_deg is not None and not math.isfinite(
            float(true_view_pitch_deg)
        ):
            raise ValueError("true-view pitch telemetry must be finite")

        self.counters["transitions"] += 1
        self.counters["command_echo_match"] += 1
        self.counters["state_resync"] += int(state_resync)
        self.counters["teacher_field_violations"] += int(teacher_field_violations)
        self.counters["requested_crouch_or_down"] += int(
            result.metrics["posture/requested_crouch_or_down"]
        )
        self.counters["requested_neutral"] += int(
            result.metrics["posture/requested_neutral"]
        )
        self.counters["requested_jump_or_up"] += int(
            result.metrics["posture/requested_jump_or_up"]
        )
        for source, destination in (
            ("posture/actual_ducked", "actual_ducked"),
            ("posture/standing_blocked", "standing_blocked"),
            ("posture/water_vertical_mode", "water_vertical_mode"),
            ("hazard/evidence", "hazard_evidence"),
            ("hazard/progress_credits", "hazard_progress_credits"),
            ("hazard/safe_arrival", "safe_arrivals"),
            ("hazard/environmental_death", "environmental_deaths"),
            ("hook/attempt", "hook_attempts"),
            ("hook/invalid_attempt", "invalid_hook_attempts"),
            ("hook/necessary_safe_arrival_credit", "necessary_hook_arrivals"),
            ("hook/safe_arrival_with_hook", "hook_recovery_arrivals"),
            ("combat/actionable_exposure", "actionable_exposure"),
            ("combat/post_command_alignment", "post_command_alignment"),
            ("combat/fire_permission", "fire_permission"),
            ("combat/executed_fire", "executed_fire"),
            ("combat/hit", "hits"),
            ("combat/repeated_hit", "repeated_hits"),
            ("combat/kill", "kills"),
            ("combat/hidden_fire", "hidden_fire"),
        ):
            self.counters[destination] += int(result.metrics[source])
        self.sums["environmental_damage"] += result.metrics[
            "hazard/environmental_damage"
        ]
        self.sums["reward"] += result.reward
        self.sums["forward_command"] += float(forward_command)
        self.sums["backward_command"] += float(backward_command)
        self.sums["movement_speed"] += float(movement_speed)
        recovery_ticks = float(result.metrics["hazard/recovery_ticks"])
        if recovery_ticks > 0.0:
            self.sums["hazard_recovery_ticks"] += recovery_ticks
            self.counters["hazard_recoveries"] += 1
        if true_view_pitch_deg is not None:
            pitch = float(true_view_pitch_deg)
            self.sums["true_view_pitch_deg"] += pitch
            self.sums["true_view_pitch_abs_deg"] += abs(pitch)
            self.counters["true_view_pitch_samples"] += 1
            self.counters["downlook_over_15deg"] += int(pitch > 15.0)
        if frame.actionable_exposure and frame.aim_yaw_error_deg is not None:
            self.sums["aim_yaw_error_deg"] += float(frame.aim_yaw_error_deg)
            self.counters["aim_yaw_samples"] += 1
        if frame.actionable_exposure and frame.aim_pitch_error_deg is not None:
            self.sums["aim_pitch_error_deg"] += float(frame.aim_pitch_error_deg)
            self.counters["aim_pitch_samples"] += 1
        if frame.fire_executed and frame.post_command_aligned:
            self.counters["aligned_executed_fire"] += 1
            if frame.damage_dealt > 0.0:
                self.counters["aligned_fire_hits"] += 1

        self.counters["guide_global_drop"] += int(global_guide_drop)
        for dropped, class_index in zip(guide_dropped, guide_classes):
            self.counters["guide_candidates"] += 1
            self.counters["guide_candidate_drop"] += int(dropped)
            if class_index is None:
                continue
            if not 0 <= int(class_index) < len(GUIDE_CLASS_NAMES):
                raise ValueError("guide class index is outside the frozen eight classes")
            class_name = GUIDE_CLASS_NAMES[int(class_index)]
            self.counters[f"guide_class_{class_name}_seen"] += 1
            self.counters[f"guide_class_{class_name}_drop"] += int(dropped)

        if extra_metrics:
            for name, value in extra_metrics.items():
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"extra metric {name!r} is not finite")
                self.sums[f"extra/{name}"] += number

    def observe_runtime_snapshot(
        self,
        *,
        atlas_loaded: bool,
        atlas_hash_match: bool,
        atlas_resident_bytes: int,
        atlas_build_peak_rss_bytes: int,
        atlas_cell_count: int,
        atlas_chunk_count: int,
        atlas_deserialize_ms: float,
        query_timings_us: Mapping[str, float],
        dyn_cell_count: int,
        live_thermal_tracks: int,
        expired_thermal_tracks: int,
        dyn_snapshot_bytes: int,
        thermal_checkpoint_fields: int = 0,
    ) -> None:
        """Record the Atlas/Dyn evidence required by the season gate."""
        integer_values = {
            "atlas_resident_bytes": atlas_resident_bytes,
            "atlas_build_peak_rss_bytes": atlas_build_peak_rss_bytes,
            "atlas_cell_count": atlas_cell_count,
            "atlas_chunk_count": atlas_chunk_count,
            "dyn_cell_count": dyn_cell_count,
            "live_thermal_tracks": live_thermal_tracks,
            "expired_thermal_tracks": expired_thermal_tracks,
            "dyn_snapshot_bytes": dyn_snapshot_bytes,
            "thermal_checkpoint_fields": thermal_checkpoint_fields,
        }
        if any(int(value) != value or int(value) < 0 for value in integer_values.values()):
            raise ValueError("Atlas/Dyn runtime counters must be nonnegative integers")
        if not math.isfinite(float(atlas_deserialize_ms)) or atlas_deserialize_ms < 0:
            raise ValueError("Atlas deserialize time must be finite and nonnegative")
        if set(query_timings_us) != set(QUERY_TIMING_COMPONENTS):
            raise ValueError("runtime query timings must contain the four frozen components")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in query_timings_us.values()
        ):
            raise ValueError("runtime query timings must be finite and nonnegative")

        self.counters["runtime_snapshots"] += 1
        self.counters["atlas_load_failures"] += int(not atlas_loaded)
        self.counters["atlas_hash_failures"] += int(not atlas_hash_match)
        self.counters["expired_thermal_tracks"] += int(expired_thermal_tracks)
        self.counters["thermal_checkpoint_fields"] += int(
            thermal_checkpoint_fields
        )
        maxima = {
            **{name: float(value) for name, value in integer_values.items()},
            "atlas_deserialize_ms": float(atlas_deserialize_ms),
        }
        for name, value in maxima.items():
            self.runtime_maxima[name] = max(
                self.runtime_maxima.get(name, 0.0), value
            )
        for name in QUERY_TIMING_COMPONENTS:
            self.query_timings[name].append(float(query_timings_us[name]))

    @staticmethod
    def _rate(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    @staticmethod
    def _p99(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        index = max(0, math.ceil(0.99 * len(ordered)) - 1)
        return ordered[index]

    def report(self, *, policy_end_version: int) -> dict:
        transitions = int(self.counters["transitions"])
        if policy_end_version < self.policy_start_version:
            raise ValueError("policy version interval cannot go backwards")
        combat_counts = {
            name: int(self.counters[name])
            for name in (
                "actionable_exposure",
                "post_command_alignment",
                "fire_permission",
                "executed_fire",
                "hits",
                "repeated_hits",
                "kills",
                "hidden_fire",
            )
        }
        per_class_dropout = {}
        for class_name in GUIDE_CLASS_NAMES:
            seen = self.counters[f"guide_class_{class_name}_seen"]
            dropped = self.counters[f"guide_class_{class_name}_drop"]
            per_class_dropout[class_name] = self._rate(dropped, seen)
        aligned_fire = self.counters["aligned_executed_fire"]
        return {
            "schema": SEASON_SCHEMA,
            "season_id": self.season_id,
            "atlas_sha256": self.atlas_sha256,
            "policy_start_version": self.policy_start_version,
            "policy_end_version": int(policy_end_version),
            "accepted_transitions": transitions,
            "transport": {
                "command_echo_match_rate": self._rate(
                    self.counters["command_echo_match"], transitions
                ),
                "state_resyncs": int(self.counters["state_resync"]),
            },
            "posture": {
                name: self._rate(self.counters[name], transitions)
                for name in (
                    "requested_crouch_or_down",
                    "requested_neutral",
                    "requested_jump_or_up",
                    "actual_ducked",
                    "standing_blocked",
                    "water_vertical_mode",
                )
            },
            "movement": {
                "forward_command_mean": self._rate(
                    self.sums["forward_command"], transitions
                ),
                "backward_command_mean": self._rate(
                    self.sums["backward_command"], transitions
                ),
                "speed_mean": self._rate(self.sums["movement_speed"], transitions),
                "true_view_pitch_mean_deg": self._rate(
                    self.sums["true_view_pitch_deg"],
                    self.counters["true_view_pitch_samples"],
                ),
                "true_view_pitch_abs_mean_deg": self._rate(
                    self.sums["true_view_pitch_abs_deg"],
                    self.counters["true_view_pitch_samples"],
                ),
                "downlook_over_15deg_rate": self._rate(
                    self.counters["downlook_over_15deg"],
                    self.counters["true_view_pitch_samples"],
                ),
            },
            "hazard": {
                "evidence": int(self.counters["hazard_evidence"]),
                "bounded_progress_credits": int(
                    self.counters["hazard_progress_credits"]
                ),
                "safe_arrivals": int(self.counters["safe_arrivals"]),
                "environmental_damage": float(self.sums["environmental_damage"]),
                "environmental_deaths": int(self.counters["environmental_deaths"]),
                "recovery_time_mean_ticks": self._rate(
                    self.sums["hazard_recovery_ticks"],
                    self.counters["hazard_recoveries"],
                ),
            },
            "hook": {
                "raw_attempt_rate": self._rate(
                    self.counters["hook_attempts"], transitions
                ),
                "invalid_attempts": int(self.counters["invalid_hook_attempts"]),
                "necessary_safe_arrivals": int(
                    self.counters["necessary_hook_arrivals"]
                ),
                "recovery_safe_arrivals": int(
                    self.counters["hook_recovery_arrivals"]
                ),
            },
            "guides": {
                "global_drop_rate": self._rate(
                    self.counters["guide_global_drop"], transitions
                ),
                "candidate_drop_rate": self._rate(
                    self.counters["guide_candidate_drop"],
                    self.counters["guide_candidates"],
                ),
                "per_class_drop_rate": per_class_dropout,
            },
            "privilege": {
                "teacher_field_violations": int(
                    self.counters["teacher_field_violations"]
                ),
            },
            "combat": {
                **combat_counts,
                "aligned_fire_precision": self._rate(
                    self.counters["aligned_fire_hits"], aligned_fire
                ),
                "visible_contact_yaw_mae_deg": self._rate(
                    self.sums["aim_yaw_error_deg"],
                    self.counters["aim_yaw_samples"],
                ),
                "visible_contact_pitch_mae_deg": self._rate(
                    self.sums["aim_pitch_error_deg"],
                    self.counters["aim_pitch_samples"],
                ),
            },
            "atlas": {
                "runtime_snapshots": int(self.counters["runtime_snapshots"]),
                "load_failures": int(self.counters["atlas_load_failures"]),
                "hash_failures": int(self.counters["atlas_hash_failures"]),
                "resident_bytes_max": int(
                    self.runtime_maxima.get("atlas_resident_bytes", 0.0)
                ),
                "build_peak_rss_bytes_max": int(
                    self.runtime_maxima.get("atlas_build_peak_rss_bytes", 0.0)
                ),
                "cell_count_max": int(
                    self.runtime_maxima.get("atlas_cell_count", 0.0)
                ),
                "chunk_count_max": int(
                    self.runtime_maxima.get("atlas_chunk_count", 0.0)
                ),
                "deserialize_ms_max": self.runtime_maxima.get(
                    "atlas_deserialize_ms", 0.0
                ),
                "query_p99_us": {
                    name: self._p99(self.query_timings[name])
                    for name in QUERY_TIMING_COMPONENTS
                },
            },
            "dyn": {
                "cell_count_max": int(
                    self.runtime_maxima.get("dyn_cell_count", 0.0)
                ),
                "live_thermal_tracks_max": int(
                    self.runtime_maxima.get("live_thermal_tracks", 0.0)
                ),
                "expired_thermal_tracks": int(
                    self.counters["expired_thermal_tracks"]
                ),
                "snapshot_bytes_max": int(
                    self.runtime_maxima.get("dyn_snapshot_bytes", 0.0)
                ),
                "thermal_checkpoint_fields": int(
                    self.counters["thermal_checkpoint_fields"]
                ),
            },
            "reward_total": float(self.sums["reward"]),
            "extra_sums": {
                name.removeprefix("extra/"): float(value)
                for name, value in sorted(self.sums.items())
                if name.startswith("extra/")
            },
        }
