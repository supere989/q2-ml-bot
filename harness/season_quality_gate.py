"""Metric- and coverage-based promotion gate for distributed training seasons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


REQUIRED_REGRESSION_CHECKS = (
    "aim",
    "reward",
    "kd",
    "lattice_memory",
)


@dataclass(frozen=True)
class SeasonGateConfig:
    min_successful_seasons: int = 3
    min_generations: int = 100
    min_env_steps: int = 1_000_000
    min_map_episodes: int = 100
    min_speedup: float = 1.25
    max_approx_kl_p95: float = 0.03
    max_clip_fraction_p95: float = 0.20


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_season(report: Mapping, config: SeasonGateConfig) -> dict:
    """Evaluate one season report and return an evidence-rich decision."""
    failures: list[str] = []
    season_id = str(report.get("season_id", "")).strip()
    if not season_id:
        failures.append("season_id is required")
    start_policy_version = int(report.get("start_policy_version", -1))
    end_policy_version = int(report.get("end_policy_version", -1))
    if start_policy_version < 0 or end_policy_version <= start_policy_version:
        failures.append("policy version interval must be positive and increasing")
    generations = int(report.get("generations", -1))
    env_steps = int(report.get("env_steps", -1))
    if generations < config.min_generations:
        failures.append(
            f"generations {generations} < required {config.min_generations}"
        )
    if env_steps < config.min_env_steps:
        failures.append(f"env_steps {env_steps} < required {config.min_env_steps}")

    maps = report.get("maps", {})
    if not isinstance(maps, Mapping) or not maps:
        failures.append("map coverage is required")
    else:
        for map_name, evidence in sorted(maps.items()):
            episodes = int(evidence.get("episodes", -1)) if isinstance(evidence, Mapping) else -1
            if episodes < config.min_map_episodes:
                failures.append(
                    f"map {map_name} episodes {episodes} < required "
                    f"{config.min_map_episodes}"
                )

    recovery = report.get("recovery", {})
    if int(recovery.get("unrecovered_failures", -1)) != 0:
        failures.append("season has unrecovered worker/learner failures")
    if int(recovery.get("duplicate_optimizer_updates", -1)) != 0:
        failures.append("season has duplicate optimizer updates")

    stability = report.get("stability", {})
    if int(stability.get("nonfinite_updates", -1)) != 0:
        failures.append("season has non-finite optimizer updates")
    try:
        kl = _finite_number(stability.get("approx_kl_p95"), "approx_kl_p95")
        if kl > config.max_approx_kl_p95:
            failures.append(f"approx_kl_p95 {kl:g} > {config.max_approx_kl_p95:g}")
        clip = _finite_number(
            stability.get("clip_fraction_p95"), "clip_fraction_p95"
        )
        if clip > config.max_clip_fraction_p95:
            failures.append(
                f"clip_fraction_p95 {clip:g} > {config.max_clip_fraction_p95:g}"
            )
    except ValueError as error:
        failures.append(str(error))

    performance = report.get("performance", {})
    try:
        distributed_sps = _finite_number(
            performance.get("distributed_sps"), "distributed_sps"
        )
        baseline_sps = _finite_number(
            performance.get("baseline_sps"), "baseline_sps"
        )
        speedup = distributed_sps / baseline_sps if baseline_sps > 0 else 0.0
        if baseline_sps <= 0 or speedup < config.min_speedup:
            failures.append(
                f"speedup {speedup:.3f} < required {config.min_speedup:.3f}"
            )
    except ValueError as error:
        speedup = 0.0
        failures.append(str(error))

    regressions = report.get("regressions", {})
    for name in REQUIRED_REGRESSION_CHECKS:
        if regressions.get(name) is not False:
            failures.append(f"regression check {name} must explicitly be false")
    if report.get("cpu_deterministic_audit") is not True:
        failures.append("CPU deterministic audit did not pass")

    return {
        "season_id": season_id,
        "passed": not failures,
        "failures": failures,
        "generations": generations,
        "env_steps": env_steps,
        "start_policy_version": start_policy_version,
        "end_policy_version": end_policy_version,
        "speedup": speedup,
    }


def evaluate_promotion(reports: Sequence[Mapping], config: SeasonGateConfig) -> dict:
    decisions = [evaluate_season(report, config) for report in reports]
    identifiers = [decision["season_id"] for decision in decisions]
    duplicate_ids = sorted({value for value in identifiers if identifiers.count(value) > 1})
    passed = sum(decision["passed"] for decision in decisions)
    failures = []
    if duplicate_ids:
        failures.append(f"duplicate season IDs: {', '.join(duplicate_ids)}")
    if passed < config.min_successful_seasons:
        failures.append(
            f"successful seasons {passed} < required {config.min_successful_seasons}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "successful_seasons": passed,
        "required_successful_seasons": config.min_successful_seasons,
        "seasons": decisions,
    }
