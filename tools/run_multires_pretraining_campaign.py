#!/usr/bin/env python3
"""Run one admitted, deterministic, no-update B5 evaluator campaign.

This is the repository implementation of the
``q2-multires-pretraining-campaign-v1`` producer protocol.  It never accepts
result counters from its caller.  Production admission is delegated to the
exact one-run preflight, the random step-zero checkpoint is loaded into the
real multires trainer runtime, guides come from the admitted Rust Atlas/Dyn
objects, actions pass through the frozen decoder/fire gate, and every reward
row is reduced and reported by the causal B5 reward/season implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.client_batch import decode_policy_action  # noqa: E402
from harness.multires_contract import (  # noqa: E402
    DYN_DIM,
    ENTITIES,
    FACTUAL_DIM,
    GUIDE_CANDIDATE_DIM,
    GUIDE_CANDIDATES,
    GUIDE_DIM,
    GuideDropoutConfig,
    GuideDropoutIdentity,
    POSTURE_CROUCH_OR_DOWN,
    POSTURE_JUMP_OR_UP,
    POSTURE_NEUTRAL,
    RECOVERY_DIM,
)
from harness.multires_metrics import MultiresSeasonMetrics  # noqa: E402
from harness.multires_reward import (  # noqa: E402
    CausalRewardConfig,
    CausalRewardFrame,
    CausalRewardReducer,
    RewardAdmissionError,
)
from harness.multires_runtime import validate_runtime_evidence  # noqa: E402
from harness.rust_multires_provider import RustAtlasSpatialProvider  # noqa: E402
from models.multires_policy import target_fire_allowed  # noqa: E402
from tools.run_multires_pretraining_validation import (  # noqa: E402
    CAMPAIGN_SCHEMA,
    REQUIRED_CAMPAIGN_MODES,
    campaign_result_sha256,
    canonical_bytes,
)
from train.multires_one_run import (  # noqa: E402
    DETERMINISM_MODE,
    OneRunAdmission,
    _load_extension,
    _state_sha256,
    preflight,
)
from train.multires_ppo import MultiresPPOConfig  # noqa: E402
from train.multires_runtime import MultiresTrainerRuntime  # noqa: E402


TOOL_NAME = "run_multires_pretraining_campaign"
POLICY_VERSION = 1
MAP_EPOCH = 1
ZERO_COUNTERS = {
    "optimizer_steps": 0,
    "backward_parameter_gradients": 0,
}


class CampaignError(RuntimeError):
    """Raised before publication when an evaluator cannot prove its result."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _file(path: Path, label: str, *, executable: bool = False) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise CampaignError(
            f"{label} must be an absolute regular non-symlink file"
        )
    resolved = source.resolve()
    if executable and not os.access(resolved, os.X_OK):
        raise CampaignError(f"{label} is not executable")
    return resolved


def _git_identity(repo_commit: str, repo_tree: str) -> None:
    import subprocess

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    if status or commit != repo_commit or tree != repo_tree:
        raise CampaignError("campaign source is not the requested clean Git identity")


@dataclass
class CampaignContext:
    admission: OneRunAdmission
    runtime: MultiresTrainerRuntime
    optimizer: Any
    provider: Any
    objectives: tuple[tuple[int, tuple[float, float, float]], ...]
    policy_state_before: str
    optimizer_state_before: str


class _NoUpdateProbe:
    """Instrument the only mutation paths available to the offline evaluator."""

    def __init__(self, context: CampaignContext) -> None:
        self.context = context
        self.counters = dict(ZERO_COUNTERS)
        self._original_optimizer_step = context.optimizer.step
        self._gradient_hooks: list[Any] = []

    def __enter__(self) -> "_NoUpdateProbe":
        def counted_optimizer_step(*args: Any, **kwargs: Any) -> Any:
            self.counters["optimizer_steps"] += 1
            return self._original_optimizer_step(*args, **kwargs)

        self.context.optimizer.step = counted_optimizer_step
        self._gradient_hooks = [
            parameter.register_hook(self._count_parameter_gradient)
            for parameter in self.context.runtime.policy.parameters()
            if parameter.requires_grad
        ]
        return self

    def _count_parameter_gradient(self, gradient: Any) -> Any:
        self.counters["backward_parameter_gradients"] += 1
        return gradient

    def __exit__(self, *_exc: Any) -> None:
        for handle in self._gradient_hooks:
            handle.remove()
        self.context.optimizer.step = self._original_optimizer_step

    def validate(self) -> None:
        if self.counters != ZERO_COUNTERS:
            raise CampaignError(
                f"no-update evaluator executed a mutation path: {self.counters!r}"
            )


class _QueryInputSentinel:
    def sample(self, *_args: Any, **_kwargs: Any) -> None:
        raise CampaignError("B5 direct Atlas evaluation may not sample live telemetry")


def _objective_points(path: Path) -> tuple[tuple[int, tuple[float, float, float]], ...]:
    document = _json(path, "objectives")
    rows = document.get("objectives")
    if not isinstance(rows, list) or not rows:
        raise CampaignError("objectives artifact contains no evaluator targets")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CampaignError("objective row is malformed")
        objective_id = row.get("objective_id")
        milli = row.get("world_milliunits")
        if (
            type(objective_id) is not int
            or objective_id < 0
            or not isinstance(milli, list)
            or len(milli) != 3
            or not all(type(axis) is int for axis in milli)
        ):
            raise CampaignError("objective evaluator coordinates are malformed")
        result.append((objective_id, tuple(float(axis) / 1000.0 for axis in milli)))
    result.sort(key=lambda item: item[0])
    return tuple(result)


def _one_run_namespace(args: argparse.Namespace, *, atlas_sha256: str,
                       runtime_sha256: str, map_name: str) -> argparse.Namespace:
    return SimpleNamespace(
        seed=args.seed,
        game_seed=args.game_seed,
        q2ded=args.q2ded,
        client_binary=args.client_binary,
        runtime_root=args.runtime_root,
        bundle_manifest=args.bundle_manifest,
        objectives=args.objectives,
        atlas_bin=args.atlas_bin,
        checkpoint=args.checkpoint,
        training_manifest=args.training_manifest,
        runtime_evidence=args.runtime_manifest,
        transition_count=args.transition_count,
        policy_version=POLICY_VERSION,
        map_epoch=MAP_EPOCH,
        map_name=map_name,
        out=args.output,
        launch_id=f"b5-{args.campaign}-{args.seed}",
        expected_atlas_sha256=atlas_sha256,
        expected_runtime_manifest_sha256=runtime_sha256,
        campaign_mode=DETERMINISM_MODE,
    )


def build_context(args: argparse.Namespace) -> CampaignContext:
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    atlas = _file(args.atlas_bin, "Atlas")
    runtime_path = _file(args.runtime_manifest, "runtime evidence")
    bundle_path = _file(args.bundle_manifest, "bundle manifest")
    runtime_document = _json(runtime_path, "runtime evidence")
    atlas_sha256 = _sha256(atlas)
    validated = validate_runtime_evidence(
        runtime_document, expected_atlas_sha256=atlas_sha256
    )
    bundle = _json(bundle_path, "bundle manifest")
    map_name = bundle.get("name")
    if not isinstance(map_name, str) or not map_name:
        raise CampaignError("bundle manifest lacks its canonical map name")
    admission = preflight(_one_run_namespace(
        args,
        atlas_sha256=atlas_sha256,
        runtime_sha256=validated.runtime_manifest_sha256,
        map_name=map_name,
    ))

    training = admission.training_configuration
    reward_config = CausalRewardConfig(**dict(training.reward))
    guide_dropout = GuideDropoutConfig(**dict(training.guide_dropout))
    ppo_config = MultiresPPOConfig(**dict(training.ppo))
    optimizer_config = admission.runtime_config["optimizer"]

    def optimizer_factory(parameters: Any) -> Any:
        return torch.optim.Adam(
            parameters,
            lr=float(optimizer_config["learning_rate"]),
            **dict(optimizer_config["kwargs"]),
        )

    runtime, optimizer, checkpoint_manifest = MultiresTrainerRuntime.resume(
        admission.args.checkpoint,
        admission.runtime_evidence,
        expected_atlas_sha256=admission.args.expected_atlas_sha256,
        device=torch.device("cpu"),
        optimizer_factory=optimizer_factory,
        reward_config=reward_config,
        guide_dropout=guide_dropout,
        ppo_config=ppo_config,
    )
    if (
        checkpoint_manifest.initialization != "random"
        or checkpoint_manifest.training_step != 0
        or optimizer.state_dict().get("state") != {}
    ):
        raise CampaignError("B5 requires the admitted random step-zero state")

    atlas_manifest = _json(
        bundle_path.parent / f"{map_name}.atlas.manifest.json",
        "Atlas manifest",
    )
    try:
        origin = tuple(int(axis) for axis in atlas_manifest["grid"]["origin"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("Atlas manifest origin is malformed") from error
    if len(origin) != 3:
        raise CampaignError("Atlas manifest origin must have three axes")
    extension = _load_extension(admission.rust_extension)
    provider = RustAtlasSpatialProvider.from_admitted_bundle(
        extension_module=extension,
        bundle_manifest_path=admission.bundle_manifest,
        uncompressed_atlas_path=admission.atlas_bin,
        dyn_snapshot_path=admission.dyn_snapshots[0],
        input_source=_QueryInputSentinel(),
        expected_atlas_sha256=admission.args.expected_atlas_sha256,
        runtime_manifest_sha256=admission.runtime_manifest_sha256,
        map_epoch=MAP_EPOCH,
        rust_client_id=0,
        client_count=int(admission.runtime_config["client_count"]),
        environment_steps=0,
        atlas_origin=origin,  # type: ignore[arg-type]
    )
    return CampaignContext(
        admission=admission,
        runtime=runtime,
        optimizer=optimizer,
        provider=provider,
        objectives=_objective_points(admission.objectives),
        policy_state_before=_state_sha256(runtime.policy.state_dict()),
        optimizer_state_before=_state_sha256(optimizer.state_dict()),
    )


def _frame(tick: int, **values: Any) -> CausalRewardFrame:
    return CausalRewardFrame(
        tick=tick,
        client_life_epoch=int(values.pop("client_life_epoch", 1)),
        authoritative_echo_valid=True,
        trainable_transition=True,
        action_generation=tick + 1000,
        **values,
    )


def _season(mode: str, atlas_sha256: str) -> MultiresSeasonMetrics:
    return MultiresSeasonMetrics(
        season_id=f"b5-{mode}", atlas_sha256=atlas_sha256,
        policy_start_version=POLICY_VERSION,
    )


def _offline_season_report(season: MultiresSeasonMetrics) -> dict[str, Any]:
    """Publish an honest offline report without inventing conduit telemetry."""
    report = season.report(policy_end_version=POLICY_VERSION)
    report["privilege"] = {
        "scope": "not-measured-offline-no-public-conduit",
        "upstream_evidence_required": "sealed-b4-real-public-datagram-audit-v1",
    }
    return report


def _observe(season: MultiresSeasonMetrics, frame: CausalRewardFrame,
             result: Any, *, dropped: Sequence[bool] = (False,) * 4,
             classes: Sequence[int | None] = (None,) * 4,
             global_drop: bool = False, forward: float = 0.0,
             backward: float = 0.0, pitch: float = 0.0) -> None:
    season.observe(
        frame, result, command_echo_match=True,
        guide_dropped=dropped, guide_classes=classes,
        global_guide_drop=global_drop,
        forward_command=forward, backward_command=backward,
        movement_speed=abs(forward) + abs(backward),
        true_view_pitch_deg=pitch,
    )


def _guide_classes(guides: np.ndarray) -> tuple[int | None, ...]:
    rows = np.asarray(guides, dtype=np.float32).reshape(
        GUIDE_CANDIDATES, GUIDE_CANDIDATE_DIM
    )
    result: list[int | None] = []
    for row in rows:
        bits = row[7:15]
        result.append(int(np.argmax(bits)) if float(bits.max()) > 0.0 else None)
    return tuple(result)


def evaluate_guides(context: CampaignContext, count: int, *, enabled: bool) -> tuple[dict, str, dict]:
    runtime = context.runtime
    points = context.objectives
    beliefs = [(objective_id, 1.0) for objective_id, _point in points]
    vectors: list[np.ndarray] = []
    raw_rows: list[list[float]] = []
    classes: list[tuple[int | None, ...]] = []
    policy_guides: list[np.ndarray] = []
    drop_rows: list[tuple[bool, ...]] = []
    for index in range(count):
        _objective_id, target = points[index % len(points)]
        position = (target[0] - 192.0, target[1] + ((index % 3) - 1) * 32.0,
                    target[2] - 16.0)
        raw = np.asarray(context.provider.atlas_runtime.guide_features(
            position, float((index * 37) % 360), MAP_EPOCH, beliefs
        ), dtype=np.float32)
        if raw.shape != (GUIDE_DIM,) or not np.isfinite(raw).all():
            raise CampaignError("Rust guide evaluator returned malformed Guide60")
        raw_rows.append([float(value) for value in raw])
        classes.append(_guide_classes(raw))
        guides = raw if enabled else np.zeros(GUIDE_DIM, dtype=np.float32)
        factual = np.zeros(FACTUAL_DIM, dtype=np.float32)
        factual[6] = 100.0
        vector, dropped = runtime.prepare_observation(
            factual, np.zeros(DYN_DIM, dtype=np.float32),
            np.zeros(RECOVERY_DIM, dtype=np.float32), guides,
            dropout_identity=GuideDropoutIdentity(
                context.admission.args.map_name, POLICY_VERSION, "b5-guide", index
            ),
            # Guide-on exercises the real identity-seeded training dropout.
            # Guide-off is a separate explicit ablation of the same raw query.
            training=enabled,
        )
        if not enabled:
            dropped = (True,) * GUIDE_CANDIDATES
        policy_guides.append(vector[-GUIDE_DIM:].copy())
        drop_rows.append(tuple(bool(value) for value in dropped))
        vectors.append(vector)
    batch = np.stack(vectors)
    hidden = [runtime.policy.init_hidden() for _ in range(count)]
    actions, _values, _log_probs, _states = runtime.policy.act_batch(
        batch, hidden, deterministic=True, gate_fire=True,
    )
    season = _season("guide_on" if enabled else "guide_off",
                     context.admission.args.expected_atlas_sha256)
    trace = []
    successes = 0
    nonzero = 0
    dropout_samples = 0
    for index, (raw, action_values) in enumerate(zip(raw_rows, actions)):
        action = decode_policy_action(action_values)
        policy_guide = policy_guides[index]
        guide_rows = policy_guide.reshape(GUIDE_CANDIDATES, GUIDE_CANDIDATE_DIM)
        available = [row for row in guide_rows if float(np.linalg.norm(row[:3])) > 0.0]
        # Frozen measurable criterion: commanded planar movement has positive
        # projection onto the first available admitted guide direction.
        if available:
            direction = available[0]
            successes += int(
                action.move_forward * float(direction[0])
                + action.move_right * float(direction[1]) > 0.0
            )
        nonzero += int(np.any(policy_guide != 0.0))
        dropout_samples += sum(drop_rows[index])
        frame = _frame(index + 1, requested_vertical=int(action.vertical_intent))
        result = runtime.reward("b5-guide", frame)
        dropped = drop_rows[index]
        _observe(
            season, frame, result, dropped=dropped, classes=classes[index],
            global_drop=bool(all(dropped)),
            forward=max(action.move_forward, 0.0),
            backward=max(-action.move_forward, 0.0),
            pitch=action.look_pitch,
        )
        trace.append({
            "index": index,
            "raw_guide_sha256": _canonical_sha256(raw),
            "policy_guide_sha256": _canonical_sha256(
                [float(value) for value in policy_guide]
            ),
            "guide_dropped": list(dropped),
            "policy_guide_nonzero": bool(np.any(policy_guide != 0.0)),
            "action": [float(value) for value in action_values],
        })
    scenario = _canonical_sha256({
        "domain": "b5-guide-scenario-v1",
        "map": context.admission.args.map_name,
        "objective_identity_sha256": context.admission.objective_identity_sha256,
        "raw_guides": raw_rows,
    })
    results = {
        "scenario_identity_sha256": scenario,
        "guide_enabled": enabled,
        "task_attempts": count,
        "task_successes": successes,
        "guide_nonzero_samples": nonzero,
        "guide_dropout_samples": dropout_samples,
    }
    return results, _canonical_sha256(trace), _offline_season_report(season)


def _run_frames(config: CausalRewardConfig, frames: Sequence[CausalRewardFrame]) -> tuple[list[Any], list[dict]]:
    reducer = CausalRewardReducer(config)
    results = []
    rows = []
    for frame in frames:
        result = reducer.step(frame)
        results.append(result)
        rows.append({
            "frame": dict(frame.__dict__),
            "reward": result.reward,
            "metrics": result.metrics,
        })
    return results, rows


def evaluate_hazard_hook(context: CampaignContext, count: int) -> tuple[dict, str, dict]:
    config = context.runtime.reward_config
    frames = [
        _frame(1, hazard_component_id=1, hazard_component_epoch=1,
               environmental_hazard_evidence=True, cost_to_safety=8.0,
               hook_attempted=True, hook_pending=True, hook_attempt_tick=1,
               hook_action_generation=1001),
        _frame(2, hazard_component_id=1, hazard_component_epoch=1,
               environmental_hazard_evidence=True, cost_to_safety=4.0,
               hook_attached=True, hook_valid=True, hook_necessity_known=True,
               hook_was_necessary=True, hook_zone_id=7, hook_attempt_tick=1,
               hook_action_generation=1001),
        _frame(3, hazard_component_id=1, hazard_component_epoch=1,
               cost_to_safety=0.0, safe_clearance=1.0),
        _frame(4, client_life_epoch=2, hazard_component_id=2,
               hazard_component_epoch=1, environmental_hazard_evidence=True,
               cost_to_safety=3.0, hook_attempted=True, hook_invalid=True,
               hook_attempt_tick=4, hook_action_generation=1004),
        _frame(5, client_life_epoch=3, hazard_component_id=3,
               hazard_component_epoch=1, environmental_hazard_evidence=True,
               cost_to_safety=2.0, environmental_source_id=9,
               environmental_source_epoch=1, environmental_mod=22,
               environmental_damage=10.0, environmental_death=True),
    ]
    while len(frames) < count:
        tick = len(frames) + 1
        frames.append(_frame(tick, client_life_epoch=4, safe_clearance=1.0))
    frames = frames[:count]
    first, trace = _run_frames(config, frames)
    second, replay = _run_frames(config, frames)
    replay_violations = int(_canonical_sha256(trace) != _canonical_sha256(replay))
    label_violations = 0
    try:
        _frame(
            count + 10, client_life_epoch=99,
            hook_attempted=True, hook_valid=True, hook_was_necessary=True,
            hook_zone_id=1, hook_attempt_tick=count + 10,
            hook_action_generation=count + 1010,
        ).validate()
        label_violations = 1
    except RewardAdmissionError:
        pass
    season = _season("hazard_hook", context.admission.args.expected_atlas_sha256)
    for frame, result in zip(frames, first):
        _observe(season, frame, result)
    report = _offline_season_report(season)
    results = {
        "hazard_scenarios": int(report["hazard"]["evidence"]),
        "hook_scenarios": int(sum(result.metrics["hook/attempt"] +
                                  result.metrics["hook/attached"] +
                                  result.metrics["hook/invalid_attempt"]
                                  for result in first)),
        "safe_arrivals": int(report["hazard"]["safe_arrivals"]),
        "environmental_deaths": int(report["hazard"]["environmental_deaths"]),
        "valid_hook_attachments": int(sum(
            result.metrics["hook/attached"] for result in first
        )),
        "invalid_hook_attempts": int(report["hook"]["invalid_attempts"]),
        "reward_replay_violations": replay_violations,
        "rate_reward_violations": len(CausalRewardReducer.positive_actuator_rate_rewards),
        "hook_necessity_label_violations": label_violations,
    }
    return results, _canonical_sha256(trace), report


def evaluate_posture(context: CampaignContext, count: int) -> tuple[dict, str, dict]:
    raw = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    )
    actions = [decode_policy_action(value) for value in raw]
    frames = [
        _frame(1, requested_vertical=int(actions[0].vertical_intent),
               actual_ducked=True, crouch_edge_id=4, crouch_edge_epoch=1,
               crouch_edge_active=True, crouch_edge_entered=True),
        _frame(2, requested_vertical=int(actions[1].vertical_intent),
               actual_ducked=True, crouch_edge_id=4, crouch_edge_epoch=1,
               crouch_edge_active=True, crouch_edge_completed=True),
        _frame(3, requested_vertical=int(actions[2].vertical_intent),
               water_vertical_mode=True),
        _frame(4, requested_vertical=int(actions[0].vertical_intent),
               water_vertical_mode=True),
        _frame(5, requested_vertical=int(actions[1].vertical_intent),
               standing_blocked=True),
        _frame(6, requested_vertical=int(actions[1].vertical_intent)),
        _frame(7, requested_vertical=int(actions[2].vertical_intent)),
    ]
    while len(frames) < count:
        tick = len(frames) + 1
        frames.append(_frame(tick, requested_vertical=POSTURE_NEUTRAL))
    frames = frames[:count]
    reduced, trace = _run_frames(context.runtime.reward_config, frames)
    season = _season(
        "posture_water_crouch", context.admission.args.expected_atlas_sha256
    )
    for frame, result in zip(frames, reduced):
        _observe(season, frame, result)
    posture_expectations = (
        (0, actions[0], "posture/requested_crouch_or_down"),
        (1, actions[1], "posture/requested_neutral"),
        (2, actions[2], "posture/requested_jump_or_up"),
    )
    posture_passes = 0
    vertical_mismatches = 0
    for frame_index, action, metric in posture_expectations:
        observed = reduced[frame_index].metrics[metric] == 1.0
        echoed = frames[frame_index].requested_vertical == int(action.vertical_intent)
        posture_passes += int(observed and echoed)
        vertical_mismatches += int(not (observed and echoed))
    water_passes = sum(
        int(reduced[index].metrics["posture/water_vertical_mode"] == 1.0)
        for index in (2, 3)
    )
    crouch_passes = int(
        reduced[0].metrics["posture/crouch_edge_entry_credit"] == 1.0
    ) + int(reduced[1].metrics["posture/crouch_edge_completion_credit"] == 1.0)
    standing_mismatches = int(
        reduced[4].metrics["posture/standing_blocked"] != 1.0
    )
    posture_fixtures = 3
    water_fixtures = 2
    crouch_fixtures = 2
    results = {
        "posture_fixtures": posture_fixtures,
        "water_fixtures": water_fixtures,
        "crouch_fixtures": crouch_fixtures,
        "fixtures_passed": posture_passes + water_passes + crouch_passes,
        "vertical_echo_mismatches": vertical_mismatches,
        "standing_blocked_mismatches": standing_mismatches,
    }
    return results, _canonical_sha256(trace), _offline_season_report(season)


def evaluate_aim(context: CampaignContext, count: int) -> tuple[dict, str, dict]:
    vectors = []
    geometry = []
    for index in range(count):
        objective_id, target = context.objectives[index % len(context.objectives)]
        dx = 512.0
        dy = float(((index % 5) - 2) * 48)
        dz = float(((index % 3) - 1) * 32)
        factual = np.zeros(FACTUAL_DIM, dtype=np.float32)
        factual[6] = 100.0
        entity = np.zeros(9, dtype=np.float32)
        entity[:3] = (dx / 4096.0, dy / 4096.0, dz / 4096.0)
        entity[6:] = (100.0, 1.0, 1.0)
        factual[ENTITIES.slice.start:ENTITIES.slice.start + 9] = entity
        vector, _ = context.runtime.prepare_observation(
            factual, np.zeros(DYN_DIM, dtype=np.float32),
            np.zeros(RECOVERY_DIM, dtype=np.float32),
            np.zeros(GUIDE_DIM, dtype=np.float32),
            dropout_identity=GuideDropoutIdentity(
                context.admission.args.map_name, POLICY_VERSION, "b5-aim", index
            ), training=False,
        )
        vectors.append(vector)
        geometry.append((objective_id, target, dx, dy, dz))
    batch = np.stack(vectors)
    hidden = [context.runtime.policy.init_hidden() for _ in range(count)]
    actions, _values, _log_prob, _states, metadata = context.runtime.policy.act_batch(
        batch, hidden, deterministic=True, gate_fire=True,
        return_fire_metadata=True,
    )
    allowed_again = target_fire_allowed(
        torch.from_numpy(batch), torch.from_numpy(actions[:, 2:4])
    ).cpu().numpy()
    if not np.array_equal(metadata["fire_allowed"], allowed_again):
        raise CampaignError("policy fire metadata differs from frozen factual fire gate")
    season = _season(
        "aim_combat_holdout", context.admission.args.expected_atlas_sha256
    )
    trace = []
    for index, (action_values, item) in enumerate(zip(actions, geometry)):
        objective_id, target, dx, dy, dz = item
        action = decode_policy_action(action_values)
        desired_yaw = -math.degrees(math.atan2(dy, dx))
        desired_pitch = -math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        yaw_error = abs(((action.look_yaw - desired_yaw + 180.0) % 360.0) - 180.0)
        pitch_error = abs(action.look_pitch - desired_pitch)
        allowed = bool(allowed_again[index])
        frame = _frame(
            index + 1, target_id=int(objective_id) + 1, target_epoch=1,
            actionable_exposure=True, post_command_aligned=allowed,
            fire_permitted=allowed, fire_requested=bool(action.fire),
            fire_executed=bool(action.fire and allowed),
            aim_yaw_error_deg=yaw_error, aim_pitch_error_deg=pitch_error,
            requested_vertical=int(action.vertical_intent),
        )
        result = context.runtime.reward("b5-aim", frame)
        _observe(
            season, frame, result,
            forward=max(action.move_forward, 0.0),
            backward=max(-action.move_forward, 0.0), pitch=action.look_pitch,
        )
        trace.append({
            "index": index,
            "objective_id": objective_id,
            "target_world": list(target),
            "action": [float(value) for value in action_values],
            "fire_allowed": allowed,
            "yaw_error": yaw_error,
            "pitch_error": pitch_error,
            "reward": result.reward,
        })
    report = _offline_season_report(season)
    combat = report["combat"]
    results = {
        "holdout_samples": count,
        "visible_contacts": count,
        "actionable_exposures": int(combat["actionable_exposure"]),
        "permitted_fire": int(combat["fire_permission"]),
        "executed_fire": int(combat["executed_fire"]),
        "hits": int(combat["hits"]),
        "repeat_hits": int(combat["repeated_hits"]),
        "kills": int(combat["kills"]),
        "hidden_fire": int(combat["hidden_fire"]),
        "yaw_mae_degrees": float(combat["visible_contact_yaw_mae_deg"]),
        "pitch_mae_degrees": float(combat["visible_contact_pitch_mae_deg"]),
    }
    return results, _canonical_sha256(trace), report


Evaluator = Callable[[CampaignContext, int], tuple[dict, str, dict]]


def _evaluate(context: CampaignContext, mode: str, count: int) -> tuple[dict, str, dict]:
    if mode == "guide_on":
        return evaluate_guides(context, count, enabled=True)
    if mode == "guide_off":
        return evaluate_guides(context, count, enabled=False)
    if mode == "hazard_hook":
        return evaluate_hazard_hook(context, count)
    if mode == "posture_water_crouch":
        return evaluate_posture(context, count)
    if mode == "aim_combat_holdout":
        return evaluate_aim(context, count)
    raise CampaignError(f"unknown campaign mode {mode!r}")


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol != CAMPAIGN_SCHEMA or args.campaign not in REQUIRED_CAMPAIGN_MODES:
        raise CampaignError("campaign protocol or mode differs")
    if args.replicate not in (0, 1) or args.seed < 0 or args.game_seed < 0:
        raise CampaignError("replicate and seeds are invalid")
    if args.transition_count < 8:
        raise CampaignError("production evaluator requires at least eight transitions")
    if not args.no_update:
        raise CampaignError("campaign runner is no-update only")
    _git_identity(args.repo_commit, args.repo_tree)
    input_paths = {
        "runtime_manifest": _file(args.runtime_manifest, "runtime evidence"),
        "checkpoint": _file(args.checkpoint, "checkpoint"),
        "training_manifest": _file(args.training_manifest, "training manifest"),
        "bundle_manifest": _file(args.bundle_manifest, "bundle manifest"),
        "atlas": _file(args.atlas_bin, "Atlas"),
        "q2ded": _file(args.q2ded, "q2ded", executable=True),
        "client_binary": _file(args.client_binary, "client binary", executable=True),
        "objectives": _file(args.objectives, "objectives"),
    }
    before = {name: _sha256(path) for name, path in input_paths.items()}
    context = build_context(args)
    with _NoUpdateProbe(context) as no_update_probe:
        results, trajectory, season_report = _evaluate(
            context, args.campaign, args.transition_count
        )
    no_update_probe.validate()
    policy_after = _state_sha256(context.runtime.policy.state_dict())
    optimizer_after = _state_sha256(context.optimizer.state_dict())
    if (
        policy_after != context.policy_state_before
        or optimizer_after != context.optimizer_state_before
    ):
        raise CampaignError("evaluator modified policy or optimizer state")
    after = {name: _sha256(path) for name, path in input_paths.items()}
    if after != before:
        raise CampaignError("an immutable campaign input changed")
    if (
        season_report.get("accepted_transitions") != args.transition_count
        or season_report.get("transport", {}).get("command_echo_match_rate") != 1.0
        or season_report.get("transport", {}).get("state_resyncs") != 0
    ):
        raise CampaignError("season report did not admit the complete causal campaign")
    evidence: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": args.campaign,
        "replicate": args.replicate,
        "status": "passed",
        "no_update": True,
        "seed": args.seed,
        "game_seed": args.game_seed,
        "transition_count": args.transition_count,
        "source": {"commit": args.repo_commit, "tree": args.repo_tree},
        "bindings": {
            "runtime_manifest_sha256": context.admission.runtime_manifest_sha256,
            "checkpoint_sha256_before": before["checkpoint"],
            "checkpoint_sha256_after": after["checkpoint"],
            "policy_state_sha256_before": context.policy_state_before,
            "policy_state_sha256_after": policy_after,
            "optimizer_state_sha256_before": context.optimizer_state_before,
            "optimizer_state_sha256_after": optimizer_after,
            "training_manifest_sha256": before["training_manifest"],
            "bundle_manifest_sha256": before["bundle_manifest"],
            "atlas_sha256": before["atlas"],
            "objective_identity_sha256": (
                context.admission.objective_identity_sha256
            ),
        },
        "counters": dict(no_update_probe.counters),
        "trajectory_sha256": trajectory,
        "season_report": season_report,
        "season_report_sha256": _canonical_sha256(season_report),
        "results": results,
    }
    evidence["result_sha256"] = campaign_result_sha256(evidence)
    return evidence


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(os.path.abspath(path.expanduser()))
    if destination.exists() or destination.is_symlink():
        raise CampaignError("campaign output already exists")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--campaign", choices=REQUIRED_CAMPAIGN_MODES, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--game-seed", type=int, required=True)
    parser.add_argument("--transition-count", type=int, required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--repo-tree", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--atlas-bin", type=Path, required=True)
    parser.add_argument("--q2ded", type=Path, required=True)
    parser.add_argument("--client-binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--objectives", type=Path, required=True)
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = run_campaign(args)
        _write_exclusive(args.output, evidence)
    except Exception as error:
        print(f"B5 campaign failed: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
