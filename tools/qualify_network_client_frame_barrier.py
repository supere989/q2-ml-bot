#!/usr/bin/env python3
"""Cold full-network qualification for the isolated frame-barrier runtime.

The production CLI has no result-injection switch.  It launches the exact
q2ded/game/client files on loopback, drives ordinary harness action packets,
and derives evidence from routed v8 telemetry plus structured engine events.
Tests may inject an executor only through :func:`produce_execution_evidence` with
``test_mode=True``; such an artifact is permanently non-passing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Protocol, Sequence


SCHEMA = "q2-network-client-frame-barrier-qualification-v1"
EXECUTION_SCHEMA = "q2-network-client-frame-barrier-execution-v1"
RAW_SCHEMA = "q2-network-client-frame-barrier-scenario-v1"
MODE = "client-telemetry-frame-ack-v1"
PROTOCOL_VERSION = 1
CLIENT_COUNT = 4
FRAME_COUNT = 32
WIRE_VERSION = 8
BARRIER_VERSION = 1
BARRIER_CAPABILITY = 1
TELEMETRY_BYTES = 1248
ACK_BYTES = 100
ACTION_TABLE_SCHEMA = "q2-network-client-frame-barrier-actions-v1"
EVENT_PREFIX = "ML_FRAME_BARRIER_EVENT "
# Debug identity is control-plane evidence emitted by game.so, not policy
# input.  Qualification uses it to prove that a routed identity is an
# ordinary playable client before the first action can be dispatched.
ML_CONTROL_HUMAN = 1
ML_ENTITY_CLIENT = 0x01
ML_ENTITY_BOT = 0x02
ML_ENTITY_ML = 0x04
ML_ENTITY_VISIBLE = 0x08
ML_ENTITY_DEAD = 0x10
ML_ENTITY_OBSERVER = 0x20
ML_ENTITY_SOLID_NOT = 0x40
ML_ENTITY_NOCLIP = 0x80
ML_ENTITY_NOCLIENT = 0x100
ML_ENTITY_SPECTATOR = 0x200
ML_ENTITY_PM_SPECTATOR = 0x1000
ML_ENTITY_PM_FREEZE = 0x2000
ML_ENTITY_EPOCH_SHIFT = 18
ML_ENTITY_EPOCH_MASK = 0xFFFC0000
ML_PLAYING_REQUIRED_FLAGS = ML_ENTITY_CLIENT | ML_ENTITY_VISIBLE
ML_PLAYING_FORBIDDEN_FLAGS = (
    ML_ENTITY_BOT
    | ML_ENTITY_ML
    | ML_ENTITY_DEAD
    | ML_ENTITY_OBSERVER
    | ML_ENTITY_SOLID_NOT
    | ML_ENTITY_NOCLIP
    | ML_ENTITY_NOCLIENT
    | ML_ENTITY_SPECTATOR
    | ML_ENTITY_PM_SPECTATOR
    | ML_ENTITY_PM_FREEZE
)
TEST_FAULT_VOCABULARY = (
    "ML_FRAME_BARRIER_TEST_FAULTS_V1=baseline,duplicate,stale,future,"
    "brief-drop,sustained-drop,conflict,death,same-life-hold,epoch-drain,"
    "drain-sigkill,load-delay,old-telemetry"
)
ROOT = Path(__file__).resolve().parents[1]
# Direct execution (``python /path/to/tools/qualify_....py``) otherwise places
# only ``tools/`` at sys.path[0].  Pin the resolved repository root ahead of
# every lazy harness import; qualification must not depend on cwd or PYTHONPATH.
_ROOT_IMPORT_PATH = os.fspath(ROOT)
if sys.path[:1] != [_ROOT_IMPORT_PATH]:
    try:
        sys.path.remove(_ROOT_IMPORT_PATH)
    except ValueError:
        pass
    sys.path.insert(0, _ROOT_IMPORT_PATH)

_SHA_CHARS = frozenset("0123456789abcdef")
_TOKEN = re.compile(r"^[A-Za-z0-9._~-]+$")
_EVENT = re.compile(r"^ML_FRAME_BARRIER_EVENT(?: [a-z_]+=[^\s]*)+$")
_REQUIRED_Q2DED_MARKERS = (
    b"sv_ml_frame_barrier_test_mode",
    b"sv_ml_frame_barrier_test_fault",
    b"sv_ml_frame_barrier_test_tick",
    b"sv_ml_frame_barrier_test_map",
    EVENT_PREFIX.encode("ascii"),
    TEST_FAULT_VOCABULARY.encode("ascii"),
)
_REQUIRED_CLIENT_MARKERS = (
    b"ml_frame_barrier",
    b"ml_frame_barrier_version",
    b"ml_frame_barrier_capability",
    b"ml_barrier_bootstrap_ready",
    b"ml_barrier_ready",
)
_REQUIRED_GAME_MARKERS = (
    b"ml_client_telemetry",
    b"ml_client_telemetry_token",
)


class QualificationError(RuntimeError):
    """Fail-closed full-network qualification error."""


def _bounded_failure_cause(error: BaseException) -> str:
    """Return useful one-line diagnostics without reproducing bearer secrets."""
    detail = " ".join(str(error).split())
    detail = re.sub(
        r"(?i)\b(token|password|secret|credential|authorization)\b"
        r"\s*(?:=|:)\s*[^\s,;]+",
        r"\1=<redacted>",
        detail,
    )
    # Qualification bearer values are long URL-safe atoms.  Redact any such
    # atom even when a lower layer omitted the field name.
    detail = re.sub(r"(?<![\w.~/-])[A-Za-z0-9_-]{24,}(?![\w.~/-])", "<redacted>", detail)
    if not detail:
        detail = "no detail"
    return f"{type(error).__name__}: {detail[:240]}"


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in _SHA_CHARS for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QualificationError("qualification value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, name: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise QualificationError(f"{name} must be an exact regular file")
    return {
        "name": name,
        "sha256": _sha256_file(source),
        "size": source.stat().st_size,
    }


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    if "evidence_sha256" in value:
        raise QualificationError("unsealed evidence contains a digest")
    payload = dict(value)
    return {**payload, "evidence_sha256": _sha256_bytes(_canonical_bytes(payload))}


def _atomic_write(path: Path, data: bytes, *, replace: bool = False) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise QualificationError(f"output is a symbolic link: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and destination.exists():
            raise QualificationError(f"immutable output already exists: {destination}")
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class QualificationIdentity:
    q2ded: Mapping[str, Any]
    game_module: Mapping[str, Any]
    client_binary: Mapping[str, Any]
    design_sha256: str
    source_repositories: Mapping[str, Mapping[str, str]]
    source_closure_sha256: str

    def validate(self) -> None:
        for name in ("q2ded", "game_module", "client_binary"):
            record = getattr(self, name)
            if (
                not isinstance(record, Mapping)
                or not _valid_sha256(record.get("sha256"))
                or type(record.get("size")) is not int
                or record["size"] < 1
                or not isinstance(record.get("name"), str)
            ):
                raise QualificationError(f"{name} identity is invalid")
        if not _valid_sha256(self.design_sha256):
            raise QualificationError("design digest is invalid")
        if set(self.source_repositories) != {"bot", "client", "game"}:
            raise QualificationError("source repository closure is incomplete")
        for name, record in self.source_repositories.items():
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("commit"), str)
                or len(record["commit"]) not in (40, 64)
                or any(character not in _SHA_CHARS for character in record["commit"])
                or not isinstance(record.get("tree"), str)
                or len(record["tree"]) not in (40, 64)
                or any(character not in _SHA_CHARS for character in record["tree"])
                or record.get("clean") is not True
            ):
                raise QualificationError(f"source repository {name} is invalid")
        source_payload = {
            name: dict(record) for name, record in sorted(self.source_repositories.items())
        }
        if (
            not _valid_sha256(self.source_closure_sha256)
            or self.source_closure_sha256
            != _sha256_bytes(_canonical_bytes(source_payload))
        ):
            raise QualificationError("source closure digest differs")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    seed: int
    cold_launch: int
    expected_outcome: str
    injected_fault: str | None = None


@dataclass(frozen=True)
class ExecutorConfiguration:
    q2ded: Path
    game_module: Path
    client_binary: Path
    data_root: Path
    game: str
    map_name: str
    epoch_map_name: str
    timeout_ms: int
    server_warmup_seconds: float


class ScenarioExecutor(Protocol):
    def run(
        self,
        spec: ScenarioSpec,
        action_table: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


def scenario_plan(seed: int) -> tuple[ScenarioSpec, ...]:
    if type(seed) is not int or seed < 0:
        raise QualificationError("qualification seed must be nonnegative")
    return (
        ScenarioSpec("baseline-cold-1", seed, 1, "completed"),
        ScenarioSpec("baseline-cold-2", seed, 2, "completed"),
        ScenarioSpec("different-seed", seed + 1, 1, "completed"),
        ScenarioSpec("duplicate-idempotence", seed, 1, "completed", "duplicate"),
        ScenarioSpec("stale-action-recovery", seed, 1, "completed", "stale"),
        ScenarioSpec("future-action", seed, 1, "fatal", "future"),
        ScenarioSpec("conflicting-action", seed, 1, "fatal", "conflict"),
        ScenarioSpec("brief-drop-recovery", seed, 1, "completed", "brief-drop"),
        ScenarioSpec(
            "load-delay-before-bootstrap", seed, 1, "completed", "load-delay"
        ),
        ScenarioSpec(
            "old-telemetry-regression", seed, 1, "completed", "old-telemetry"
        ),
        ScenarioSpec("death-respawn-lifecycle", seed, 1, "completed", "death"),
        ScenarioSpec("same-life-teleport-hold", seed, 1, "completed", "same-life-hold"),
        ScenarioSpec("sustained-drop-timeout", seed, 1, "fatal", "sustained-drop"),
        ScenarioSpec("one-client-sigkill", seed, 1, "fatal", "sigkill"),
        ScenarioSpec(
            "disconnect-during-drain", seed, 1, "fatal", "drain-sigkill"
        ),
        ScenarioSpec("fifth-client-rejection", seed, 1, "rejected", "fifth"),
        ScenarioSpec("human-client-rejection", seed, 1, "rejected", "human"),
        ScenarioSpec("mixed-cvar-rejection", seed, 1, "rejected", "mixed-cvar"),
        ScenarioSpec("epoch-drain", seed, 1, "completed", "epoch-drain"),
    )


def _server_fault_for(spec: ScenarioSpec) -> str:
    fault = spec.injected_fault
    if fault in {
        "duplicate", "stale", "future", "brief-drop", "sustained-drop",
        "conflict", "death", "same-life-hold", "epoch-drain", "drain-sigkill",
        "load-delay", "old-telemetry",
    }:
        return str(fault)
    return ""


def build_action_table(seed: int) -> dict[str, Any]:
    """Return a quantized fixed 32-frame/four-client script, never PPO."""
    generator = random.Random(seed)
    rows = []
    for frame in range(FRAME_COUNT):
        clients = []
        for client in range(CLIENT_COUNT):
            action = {
                "client_id": f"qual-{client:02d}",
                "move_forward": generator.choice((-0.75, -0.25, 0.25, 0.75)),
                "move_right": generator.choice((-0.5, 0.0, 0.5)),
                "look_yaw": generator.choice((-6.0, -2.0, 0.0, 2.0, 6.0)),
                "look_pitch": generator.choice((-2.0, 0.0, 2.0)),
                "vertical_intent": generator.choice((0, 1, 1, 1, 2)),
                "fire": False,
                "hook": 0,
                "weapon": 0,
            }
            # Script reliable controls explicitly; zero-only actions cannot
            # prove deferred hook/weapon ordering through the real netchan.
            if frame == 6 and client == 0:
                action["hook"] = 1
            elif frame == 7 and client == 0:
                action["hook"] = 3
            if frame == 8 and client == 1:
                action["weapon"] = 3
            if frame == 11 and client == 2:
                action["hook"] = 1
                action["weapon"] = 2
            if client == 0:
                # Faults trigger on an absolute server tick after startup,
                # independent of the admitted trajectory ordinal.  Keep a
                # deterministic nonzero probe on slot 0 throughout the script
                # so stock PMF's clamp can never pass vacuously.
                action.update({
                    "move_forward": 0.75,
                    "move_right": 0.5,
                    "look_yaw": 6.0,
                    "look_pitch": 2.0,
                })
            clients.append(action)
        rows.append({"ordinal": frame + 1, "clients": clients})
    table = {
        "schema": ACTION_TABLE_SCHEMA,
        "seed": seed,
        "client_count": CLIENT_COUNT,
        "frame_count": FRAME_COUNT,
        "rows": rows,
    }
    return {**table, "sha256": _sha256_bytes(_canonical_bytes(table))}


def _parse_events(log: str) -> list[dict[str, str]]:
    events = []
    for line in log.splitlines():
        line = line.strip()
        if not line.startswith(EVENT_PREFIX):
            continue
        if _EVENT.fullmatch(line) is None:
            raise QualificationError(f"malformed structured barrier event: {line!r}")
        values = {}
        for item in line.removeprefix(EVENT_PREFIX).split(" "):
            key, value = item.split("=", 1)
            if key in values:
                raise QualificationError(f"duplicate barrier event key {key!r}")
            values[key] = value
        if not values.get("event"):
            raise QualificationError("barrier event omits event name")
        events.append(values)
    return events


def _event_count(
    events: Sequence[Mapping[str, str]], name: str, **wanted: str
) -> int:
    return sum(
        event.get("event") == name
        and all(event.get(key) == value for key, value in wanted.items())
        for event in events
    )


def _boundary_action_matches(
    item: Mapping[str, Any], *, require_look_fire: bool
) -> bool:
    expected = item.get("expected_action")
    applied = item.get("applied_action")
    if not isinstance(expected, Mapping) or not isinstance(applied, Mapping):
        return False
    float_names = ["move_forward", "move_right"]
    if require_look_fire:
        float_names += ["look_yaw", "look_pitch"]
    for name in float_names:
        if name not in applied or name not in expected or not math.isclose(
            float(applied[name]), float(expected[name]), rel_tol=0.0,
            abs_tol=0.25 if name.startswith("look_") else 0.05,
        ):
            return False
    exact_names = ["vertical_intent", "hook", "weapon"]
    if require_look_fire:
        exact_names.append("fire")
    return all(applied.get(name) == expected.get(name) for name in exact_names)


def _bootstrap_role_record(sample: Any) -> dict[str, Any]:
    """Return an exact, action-free playing-role certificate or fail closed."""
    client_id = str(sample.client_id)
    slot = int(sample.client_slot)
    observation = sample.observation
    debug = [int(value) for value in observation.self_debug]
    if len(debug) != 4:
        raise QualificationError(
            f"routed role proof for {client_id} is not four debug fields"
        )
    edict_index, debug_slot, control_source, flags = debug
    life_epoch = int(sample.causal.client_life_epoch)
    encoded_life = (flags & ML_ENTITY_EPOCH_MASK) >> ML_ENTITY_EPOCH_SHIFT
    health = float(observation.self_state[6])
    causal_role_playing = bool(sample.causal.role_playing)
    causal_role_public_pm_normal = bool(
        sample.causal.role_public_pm_normal
    )
    role_valid = (
        re.fullmatch(r"qual-0[0-3]", client_id) is not None
        and slot == int(client_id.removeprefix("qual-"))
        and edict_index == slot + 1
        and debug_slot == slot
        and control_source == ML_CONTROL_HUMAN
        and flags & ML_PLAYING_REQUIRED_FLAGS == ML_PLAYING_REQUIRED_FLAGS
        and flags & ML_PLAYING_FORBIDDEN_FLAGS == 0
        and life_epoch > 0
        and encoded_life == (life_epoch & 0x3FFF)
        and math.isfinite(health)
        and health > 0.0
        and not bool(observation.is_terminal)
        and int(observation.terminal_reason) == 0
        and causal_role_playing
        and causal_role_public_pm_normal
    )
    record = {
        "client_id": client_id,
        "client_slot": slot,
        "server_frame": int(sample.server_frame),
        "map_epoch": int(sample.map_epoch),
        "client_life_epoch": life_epoch,
        "edict_index": edict_index,
        "debug_client_slot": debug_slot,
        "control_source": control_source,
        "debug_flags": flags,
        "encoded_life_epoch": encoded_life,
        "health": health,
        "terminal_reason": int(observation.terminal_reason),
        "causal_role_playing": causal_role_playing,
        "causal_role_public_pm_normal": causal_role_public_pm_normal,
        "role": "playing" if role_valid else "invalid",
    }
    if not role_valid:
        raise QualificationError(
            f"routed identity is not a playable non-spectator client: {record}"
        )
    return record


def _validate_bootstrap_roles(value: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Recompute the four-client role seal from raw execution evidence."""
    mismatches: dict[str, tuple[Any, Any]] = {}
    runtime = value.get("routed_role_runtime")
    expected_runtime = {
        "use_startobserver": 0,
        "use_startchasecam": 0,
    }
    if runtime != expected_runtime:
        mismatches["routed_role_runtime"] = (runtime, expected_runtime)
    records = value.get("routed_role_preflight")
    if not isinstance(records, list) or len(records) != CLIENT_COUNT:
        return {
            **mismatches,
            "routed_role_preflight": (records, "four playing certificates"),
        }
    by_slot = {
        record.get("client_slot"): record
        for record in records if isinstance(record, Mapping)
    }
    if set(by_slot) != set(range(CLIENT_COUNT)):
        mismatches["routed_role_slots"] = (
            sorted(slot for slot in by_slot if isinstance(slot, int)),
            list(range(CLIENT_COUNT)),
        )
        return mismatches
    for slot, record in sorted(by_slot.items()):
        client_id = f"qual-{slot:02d}"
        flags = record.get("debug_flags")
        life_epoch = record.get("client_life_epoch")
        valid = (
            record.get("client_id") == client_id
            and record.get("edict_index") == slot + 1
            and record.get("debug_client_slot") == slot
            and record.get("control_source") == ML_CONTROL_HUMAN
            and type(flags) is int
            and flags & ML_PLAYING_REQUIRED_FLAGS == ML_PLAYING_REQUIRED_FLAGS
            and flags & ML_PLAYING_FORBIDDEN_FLAGS == 0
            and type(life_epoch) is int
            and life_epoch > 0
            and record.get("encoded_life_epoch") == (life_epoch & 0x3FFF)
            and type(record.get("server_frame")) is int
            and record["server_frame"] > 0
            and type(record.get("map_epoch")) is int
            and record["map_epoch"] > 0
            and isinstance(record.get("health"), (int, float))
            and math.isfinite(float(record["health"]))
            and float(record["health"]) > 0.0
            and record.get("terminal_reason") == 0
            and record.get("causal_role_playing") is True
            and record.get("causal_role_public_pm_normal") is True
            and record.get("role") == "playing"
        )
        if not valid:
            mismatches[f"routed_role_{slot}"] = (
                record, "exact playable non-spectator role proof",
            )
    return mismatches


def _validate_stock_lifecycle_boundaries(
    spec: ScenarioSpec,
    value: Mapping[str, Any],
    action_table: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Validate variable-length stock PMF lifecycle/hold evidence."""
    mismatches: dict[str, tuple[Any, Any]] = {}
    rows = value.get("trajectory_rows")
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        return {"lifecycle_trajectory": (rows, f"{FRAME_COUNT} rows")}
    death = spec.injected_fault == "death"
    key = (
        "death_lifecycle_boundary_rows" if death
        else "action_hold_boundary_rows"
    )
    boundaries = value.get(key)
    if not isinstance(boundaries, list) or len(boundaries) < 4:
        return {f"{key}_count": (boundaries, ">=4 stock boundaries")}
    ordinals = {
        boundary.get("trajectory_ordinal")
        for boundary in boundaries if isinstance(boundary, Mapping)
    }
    if (
        len(ordinals) != 1 or type(next(iter(ordinals), None)) is not int
        or not 2 <= next(iter(ordinals)) <= FRAME_COUNT
    ):
        return {
            "lifecycle_trajectory_ordinal": (
                ordinals, f"one constant ordinal within 2..{FRAME_COUNT}"
            )
        }
    trajectory_index = next(iter(ordinals)) - 1
    metric = (
        value.get("death_lifecycle_resyncs") if death
        else value.get("action_state_resyncs")
    )
    if metric != len(boundaries) or value.get("boundary_rounds") != len(boundaries):
        mismatches["lifecycle_boundary_counts"] = (
            (metric, value.get("boundary_rounds"), len(boundaries)),
            "equal exact counts",
        )
    other = value.get(
        "action_hold_boundary_rows" if death
        else "death_lifecycle_boundary_rows"
    )
    other_metric = value.get(
        "action_state_resyncs" if death else "death_lifecycle_resyncs"
    )
    if other not in ([], None) or other_metric not in (0, None):
        mismatches["lifecycle_unexpected_other_boundary"] = (
            (other, other_metric), ([], 0),
        )

    trajectory_by_slot = [
        {item["client_slot"]: item for item in row["clients"]}
        for row in rows
    ]
    if death:
        admitted_terminals = [
            item for row in rows for item in row.get("clients", ())
            if item.get("observation", {}).get("terminal_reason") == 1
        ]
        if admitted_terminals:
            mismatches["death_admitted_terminal_count"] = (
                len(admitted_terminals), 0,
            )
        changes = [
            index for index in range(1, len(rows))
            if trajectory_by_slot[index][0]["causal"]["client_life_epoch"]
            != trajectory_by_slot[index - 1][0]["causal"]["client_life_epoch"]
        ]
        if changes != [trajectory_index]:
            mismatches["death_admitted_life_change"] = (
                changes, [trajectory_index],
            )
            return mismatches
        next_index = trajectory_index
        old_life = trajectory_by_slot[next_index - 1][0]["causal"][
            "client_life_epoch"
        ]
        new_life = trajectory_by_slot[next_index][0]["causal"][
            "client_life_epoch"
        ]
        if new_life != old_life + 1:
            mismatches["death_life_increment"] = ((old_life, new_life), "+1")
    else:
        next_index = trajectory_index
        old_life = trajectory_by_slot[next_index - 1][0]["causal"][
            "client_life_epoch"
        ]
        new_life = old_life
        if any(
            item["causal"]["client_life_epoch"] != old_life
            for row in trajectory_by_slot for item in row.values()
        ):
            mismatches["same_life_hold_epoch"] = ("changed", "stable")

    previous_row = trajectory_by_slot[next_index - 1]
    next_row = trajectory_by_slot[next_index]
    expected_actions = {
        slot: item for slot, item in enumerate(
            action_table["rows"][next_index]["clients"]
        )
    }
    prior_frame = previous_row[0]["server_frame"]
    next_frame = next_row[0]["server_frame"]
    prior_sequences = {
        slot: item["sequence"] for slot, item in previous_row.items()
    }
    phases = []
    rebase_count = 0
    settle_count = 0
    prime_count = 0
    prior_round = -1
    for boundary_index, boundary in enumerate(boundaries):
        clients = boundary.get("clients") if isinstance(boundary, Mapping) else None
        round_id = boundary.get("round_id") if isinstance(boundary, Mapping) else None
        if (
            type(round_id) is not int or round_id <= prior_round
            or not isinstance(clients, list) or len(clients) != CLIENT_COUNT
        ):
            mismatches[f"lifecycle_boundary_shape_{boundary_index}"] = (
                boundary, "monotonic four-client boundary",
            )
            continue
        prior_round = round_id
        by_slot = {
            item.get("client_slot"): item
            for item in clients if isinstance(item, Mapping)
        }
        frames = {item.get("server_frame") for item in clients}
        if set(by_slot) != set(range(CLIENT_COUNT)) or len(frames) != 1:
            mismatches[f"lifecycle_boundary_roster_{boundary_index}"] = (
                (set(by_slot), frames), "exact synchronized roster",
            )
            continue
        frame = next(iter(frames))
        if type(frame) is not int or not prior_frame < frame < next_frame:
            mismatches[f"lifecycle_boundary_frame_{boundary_index}"] = (
                frame, f"{prior_frame} < frame < {next_frame}",
            )
        target = by_slot[0]
        phase = target.get("phase")
        phases.append(phase)
        if phase == "new_life_rebase":
            rebase_count += 1
        elif phase in (
            "new_life_teleport_settling", "same_life_hold_settling"
        ):
            settle_count += 1
        elif phase in (
            "new_life_actionable_prime", "same_life_actionable_prime"
        ):
            prime_count += 1
        expected_target_life = (
            old_life if death and phase in ("death_terminal", "corpse")
            else new_life
        )
        for slot, item in by_slot.items():
            action_tick = item.get("action_tick")
            sequence = item.get("sequence")
            expected_life = expected_target_life if slot == 0 else (
                previous_row[slot]["causal"]["client_life_epoch"]
            )
            expected_public_normal = not (
                death and slot == 0
                and phase in ("death_terminal", "corpse")
            )
            exact_identity = (
                item.get("client_id") == f"qual-{slot:02d}"
                and item.get("map_name") == previous_row[slot]["map_name"]
                and item.get("map_epoch") == previous_row[slot]["map_epoch"]
                and item.get("client_life_epoch") == expected_life
                and item.get("trainable_transition") is False
                and item.get("echo_tick") == action_tick
                and type(action_tick) is int
                and item.get("action_generation") == action_tick % 192 + 1
                and type(sequence) is int
                and sequence > prior_sequences.get(slot, -1)
                and item.get("expected_action") == expected_actions[slot]
                and item.get("causal_role_playing") is True
                and item.get("causal_role_public_pm_normal")
                is expected_public_normal
            )
            if not exact_identity:
                mismatches[f"lifecycle_boundary_identity_{boundary_index}_{slot}"] = (
                    item, "exact identity/action/generation",
                )
            prior_sequences[slot] = sequence if type(sequence) is int else -1
            require_full_action = slot != 0 or phase in (
                "new_life_actionable_prime", "same_life_actionable_prime"
            )
            if not _boundary_action_matches(
                item, require_look_fire=require_full_action
            ):
                mismatches[f"lifecycle_boundary_action_{boundary_index}_{slot}"] = (
                    item.get("applied_action"), "exact causal action envelope",
                )
        if target.get("causal_echo_valid") is not True or target.get(
            "causal_facts_complete"
        ) is not True:
            mismatches[f"lifecycle_causal_complete_{boundary_index}"] = (
                target, "echo/facts true",
            )
        should_train = phase in (
            "new_life_actionable_prime", "same_life_actionable_prime"
        )
        if phase in (
            "new_life_rebase", "new_life_teleport_settling",
            "same_life_hold_settling",
        ) and target.get("causal_transition_trainable") is not False:
            mismatches[f"lifecycle_settle_trainable_{boundary_index}"] = (
                target.get("causal_transition_trainable"), False,
            )
        if should_train and target.get("causal_transition_trainable") is not True:
            mismatches[f"lifecycle_prime_trainable_{boundary_index}"] = (
                target.get("causal_transition_trainable"), True,
            )
        if boundary_index == len(boundaries) - 2 and phase in (
            "new_life_teleport_settling", "same_life_hold_settling"
        ):
            expected = target.get("expected_action", {})
            applied = target.get("applied_action", {})
            if not (
                expected.get("look_pitch") == 2.0
                and math.isclose(
                    float(applied.get("look_pitch", math.inf)), 0.0,
                    rel_tol=0.0, abs_tol=0.25,
                )
                and math.isclose(
                    float(applied.get("look_yaw", math.inf)),
                    float(expected.get("look_yaw", math.inf)),
                    rel_tol=0.0, abs_tol=0.25,
                )
            ):
                mismatches["lifecycle_final_stock_pitch_clamp"] = (
                    (expected, applied), "nonzero expected pitch, actual 0, yaw exact",
                )

    wanted_prime = (
        "new_life_actionable_prime" if death
        else "same_life_actionable_prime"
    )
    if prime_count != 1 or not phases or phases[-1] != wanted_prime:
        mismatches["lifecycle_final_prime"] = (
            (prime_count, phases[-1] if phases else None), (1, wanted_prime),
        )
    if settle_count < 2:
        mismatches["lifecycle_settle_count"] = (settle_count, ">=2")
    if death:
        if (
            not phases or phases[0] != "death_terminal"
            or phases.count("death_terminal") != 1 or rebase_count != 1
        ):
            mismatches["death_phase_sequence"] = (
                (phases, rebase_count), "terminal ... one rebase ... settle ... prime",
            )
        if boundaries[0]["clients"][0].get("observed_terminal_reason") != 1:
            mismatches["death_terminal_boundary"] = (
                boundaries[0]["clients"][0], "slot-0 death terminal",
            )
        rebase_index = phases.index("new_life_rebase") if rebase_count == 1 else -1
        if not (
            rebase_index > 0
            and all(phase == "corpse" for phase in phases[1:rebase_index])
            and all(
                phase == "new_life_teleport_settling"
                for phase in phases[rebase_index + 1:-1]
            )
        ):
            mismatches["death_exact_phase_order"] = (
                phases,
                "death_terminal, corpse*, new_life_rebase, settle+, prime",
            )
    elif rebase_count:
        mismatches["same_life_unexpected_rebase"] = (rebase_count, 0)
    elif not all(
        phase == "same_life_hold_settling" for phase in phases[:-1]
    ):
        mismatches["same_life_exact_phase_order"] = (
            phases, "same_life_hold_settling+, same_life_actionable_prime",
        )
    for slot, sequence in prior_sequences.items():
        if next_row[slot]["sequence"] <= sequence:
            mismatches[f"lifecycle_post_prime_sequence_{slot}"] = (
                next_row[slot]["sequence"], f"> {sequence}",
            )
    if next_row[0]["observation"]["terminal_reason"] != 0:
        mismatches["lifecycle_next_admitted_alive"] = (
            next_row[0]["observation"], "alive",
        )
    return mismatches


def _validate_stock_lifecycle_events(
    spec: ScenarioSpec, value: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    mismatches: dict[str, tuple[Any, Any]] = {}
    events = value.get("structured_events")
    if not isinstance(events, list):
        return {"lifecycle_events": (events, "structured event list")}
    if value.get("exception") is not None or any(
        isinstance(event, Mapping)
        and event.get("event") in ("fatal", "disconnect")
        for event in events
    ):
        mismatches["lifecycle_fatal_or_disconnect"] = (
            (value.get("exception"), value.get("fault_event_count")),
            (None, 0),
        )
    boundary_key = (
        "death_lifecycle_boundary_rows"
        if spec.injected_fault == "death" else "action_hold_boundary_rows"
    )
    boundaries = value.get(boundary_key)
    final_settle = None
    if isinstance(boundaries, list) and len(boundaries) >= 2:
        final_settle = next(
            (
                item for item in boundaries[-2].get("clients", ())
                if item.get("client_slot") == 0
            ),
            None,
        )
    clear_events = [
        (index, event) for index, event in enumerate(events)
        if isinstance(event, Mapping)
        and event.get("event") == "ml_respawn_settling_action"
    ]
    clear_index = -1
    if len(clear_events) != 1:
        mismatches["lifecycle_clear_latch_count"] = (len(clear_events), 1)
    else:
        clear_index, clear = clear_events[0]
        applied = (
            final_settle.get("applied_action", {})
            if isinstance(final_settle, Mapping) else {}
        )
        if not (
            isinstance(final_settle, Mapping)
            and clear.get("slot") == "0"
            and clear.get("client_id") == "qual-00"
            and clear.get("client_life_epoch")
            == str(final_settle.get("client_life_epoch"))
            and clear.get("server_frame")
            == str(final_settle.get("server_frame"))
            and clear.get("action_tick")
            == str(final_settle.get("action_tick"))
            and clear.get("entry_latched") == "1"
            and clear.get("live_pmf_time_teleport") == "0"
            and clear.get("active") == "1"
            and clear.get("post_pmove_active") == "0"
            and clear.get("echo_valid") == "1"
            and clear.get("facts_complete") == "1"
            and clear.get("transition_trainable") == "0"
            and math.isclose(
                float(clear.get("actual_look_yaw", math.inf)),
                float(applied.get("look_yaw", math.inf)),
                rel_tol=0.0, abs_tol=0.000001,
            )
            and math.isclose(
                float(clear.get("actual_look_pitch", math.inf)),
                float(applied.get("look_pitch", math.inf)),
                rel_tol=0.0, abs_tol=0.000001,
            )
        ):
            mismatches["lifecycle_clear_latch"] = (
                clear, "exact final settle E1/F1/T0 entry latch",
            )
    if spec.injected_fault == "same-life-hold":
        injected = [
            (index, event) for index, event in enumerate(events)
            if isinstance(event, Mapping)
            and event.get("event") == "same_life_hold_injected"
        ]
        if len(injected) != 1:
            mismatches["same_life_hold_injection_count"] = (len(injected), 1)
        else:
            event = injected[0][1]
            if not (
                event.get("slot") == "0"
                and event.get("action_tick") == "5"
                and event.get("client_life_epoch") == "1"
                and event.get("active") == "1"
                and event.get("pm_time") == "14"
                and event.get("law") == "stock"
            ):
                mismatches["same_life_hold_injection"] = (
                    event, "slot0 tick5 life1 active stock pm_time14",
                )
            if clear_index != -1 and injected[0][0] >= clear_index:
                mismatches["same_life_hold_event_order"] = (
                    (injected[0][0], clear_index), "injection < clear latch",
                )
        return mismatches

    terminal_item = None
    if isinstance(boundaries, list) and boundaries:
        terminal_item = next(
            (
                item for item in boundaries[0].get("clients", ())
                if item.get("client_slot") == 0
            ),
            None,
        )
    old_life = (
        terminal_item.get("client_life_epoch")
        if isinstance(terminal_item, Mapping) else None
    )
    selected = {}
    for name in (
        "death_injected", "telemetry", "respawn_action_restore",
        "respawn_teleport_hold",
    ):
        matches = [
            (index, event) for index, event in enumerate(events)
            if isinstance(event, Mapping) and event.get("event") == name
            and (
                name != "telemetry"
                or event.get("slot") == "0"
                and event.get("terminal_reason") == "1"
            )
        ]
        if len(matches) != 1:
            mismatches[f"death_event_{name}_count"] = (len(matches), 1)
        else:
            selected[name] = matches[0]
    if len(selected) == 4:
        death_index, death_event = selected["death_injected"]
        terminal_index, terminal = selected["telemetry"]
        restore_index, restore = selected["respawn_action_restore"]
        hold_index, hold = selected["respawn_teleport_hold"]
        old_text = str(old_life) if type(old_life) is int else None
        new_text = str(old_life + 1) if type(old_life) is int else None
        if not (
            death_event.get("slot") == "0"
            and death_event.get("action_tick") == "5"
            and terminal.get("client_id") == "qual-00"
            and terminal.get("client_life_epoch") == old_text
            and terminal.get("alive") == "0"
            and restore.get("slot") == "0"
            and restore.get("client_id") == "qual-00"
            and restore.get("prior_life_epoch") == old_text
            and restore.get("life_epoch") == new_text
            and restore.get("route_preserved") == "1"
            and restore.get("attribution") == "exact"
            and hold.get("slot") == "0"
            and hold.get("client_life_epoch") == new_text
            and hold.get("active") == "1"
            and hold.get("pm_time") == "14"
            and hold.get("law") == "stock"
        ):
            mismatches["death_stock_event_identity"] = (
                (death_event, terminal, restore, hold), "exact stock lifecycle",
            )
        if not (
            death_index < terminal_index < restore_index < hold_index
            and (clear_index == -1 or hold_index < clear_index)
        ):
            mismatches["death_stock_event_order"] = (
                (
                    death_index, terminal_index, restore_index, hold_index,
                    clear_index,
                ),
                "death < terminal < restore < stock hold < clear latch",
            )
    return mismatches


class _PortAllocator:
    _lock = threading.Lock()
    _leased: set[int] = set()

    @classmethod
    def reserve(cls, count: int) -> tuple[int, ...]:
        """Lease process-unique loopback ports until explicit release.

        The probe sockets must close before q2ded and the clients bind, but
        the port identities remain leased in-process for the complete
        scenario.  Concurrent qualification jobs therefore cannot receive a
        port that an earlier job has probed but not yet bound.
        """
        if type(count) is not int or count < 1:
            raise QualificationError("loopback port lease count is invalid")
        with cls._lock:
            sockets = []
            ports = []
            try:
                while len(ports) < count:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.bind(("127.0.0.1", 0))
                    port = int(sock.getsockname()[1])
                    if port in cls._leased:
                        sock.close()
                        continue
                    sockets.append(sock)
                    ports.append(port)
                if len(set(ports)) != count:
                    raise QualificationError("loopback port allocation collided")
                cls._leased.update(ports)
                return tuple(ports)
            finally:
                for sock in sockets:
                    sock.close()

    @classmethod
    def release(cls, ports: Sequence[int]) -> None:
        values = tuple(ports)
        if len(values) != len(set(values)) or any(
            type(port) is not int for port in values
        ):
            raise QualificationError("loopback port lease release is invalid")
        with cls._lock:
            if not set(values).issubset(cls._leased):
                raise QualificationError("loopback port lease was not active")
            cls._leased.difference_update(values)


class FullNetworkExecutor:
    """Real loopback process executor; no child-result injection surface."""

    def __init__(self, configuration: ExecutorConfiguration):
        self.configuration = configuration
        if (
            re.fullmatch(r"[A-Za-z0-9_-]+", configuration.game) is None
            or re.fullmatch(r"[A-Za-z0-9_-]+", configuration.map_name) is None
            or re.fullmatch(r"[A-Za-z0-9_-]+", configuration.epoch_map_name) is None
            or configuration.map_name == configuration.epoch_map_name
        ):
            raise QualificationError(
                "game/map names must be safe and epoch map must be distinct"
            )
        self._preflight_controls()

    def _preflight_controls(self) -> None:
        config = self.configuration
        for path, markers, label in (
            (config.q2ded, _REQUIRED_Q2DED_MARKERS, "q2ded"),
            (config.client_binary, _REQUIRED_CLIENT_MARKERS, "client"),
            (config.game_module, _REQUIRED_GAME_MARKERS, "game module"),
        ):
            data = path.read_bytes()
            missing = [value.decode("ascii") for value in markers if value not in data]
            if missing:
                raise QualificationError(
                    f"{label} lacks production qualification controls: {missing!r}"
                )

    @staticmethod
    def _link_tree(source: Path, destination: Path, *, exclude: set[str] = set()) -> None:
        if not source.is_dir():
            raise QualificationError(f"asset directory is absent: {source}")
        destination.mkdir(parents=True, exist_ok=False)
        for child in source.iterdir():
            if child.name in exclude:
                continue
            (destination / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())

    def _runtime(self, root: Path) -> tuple[Path, Path]:
        config = self.configuration
        q2ded = root / "q2ded"
        client = root / "yquake2"
        shutil.copy2(config.q2ded, q2ded)
        shutil.copy2(config.client_binary, client)
        os.chmod(q2ded, 0o700)
        os.chmod(client, 0o700)
        self._link_tree(config.data_root / "baseq2", root / "baseq2")
        self._link_tree(
            config.data_root / config.game,
            root / config.game,
            exclude={"game.so"},
        )
        shutil.copy2(config.game_module, root / config.game / "game.so")
        return q2ded, client

    @staticmethod
    def _actions(row: Mapping[str, Any]) -> list[Any]:
        from harness.protocol import Action, VerticalIntent

        return [Action(
            move_forward=float(value["move_forward"]),
            move_right=float(value["move_right"]),
            look_yaw=float(value["look_yaw"]),
            look_pitch=float(value["look_pitch"]),
            vertical_intent=VerticalIntent(int(value["vertical_intent"])),
            fire=bool(value["fire"]),
            hook=int(value["hook"]),
            weapon=int(value["weapon"]),
        ) for value in row["clients"]]

    @staticmethod
    def _trajectory_row(
        round_result: Any,
        *,
        ordinal: int,
        action_row: Mapping[str, Any],
        map_epochs: Mapping[str, int],
    ) -> dict[str, Any]:
        clients = []
        for observation, info, expected_action in zip(
            round_result.observations,
            round_result.infos,
            action_row["clients"],
        ):
            causal = {
                "client_life_epoch": int(
                    info["authoritative_client_life_epoch"]
                ),
                "echo_tick": int(info["authoritative_echo_tick"]),
                "action_generation": int(
                    info["authoritative_action_generation"]
                ),
                "echo_valid": bool(info["causal_echo_valid"]),
                "facts_complete": bool(info["causal_facts_complete"]),
                "transition_trainable": bool(
                    info["causal_transition_trainable"]
                ),
                "role_playing": bool(info["causal_role_playing"]),
                "role_public_pm_normal": bool(
                    info["causal_role_public_pm_normal"]
                ),
            }
            map_name = str(info["map"])
            if map_name not in map_epochs:
                raise QualificationError(
                    f"trajectory reported unexpected map {map_name!r}"
                )
            clients.append({
                "client_id": str(info["client_id"]),
                "client_slot": int(info["client_slot"]),
                "sequence": int(info["sequence"]),
                "server_frame": int(info["server_frame"]),
                "map_name": map_name,
                # The batch's optional privileged epoch source is deliberately
                # absent in this network-only qualification.  Bind the two
                # distinct, sealed maps to their server-owned epoch instead of
                # attempting to coerce that nullable diagnostic field.
                "map_epoch": int(map_epochs[map_name]),
                "action_tick": int(info["action_tick"]),
                "applied_action_tick": int(info["action_tick"]),
                "echo_tick": int(info["authoritative_echo_tick"]),
                "action_generation": int(info["authoritative_action_generation"]),
                "causal": causal,
                "expected_action": dict(expected_action),
                "applied_action": {
                    "move_forward": float(info["action_debug_movement"][0]),
                    "move_right": float(info["action_debug_movement"][1]),
                    "look_yaw": float(info["effective_action_look_yaw"]),
                    "look_pitch": float(info["effective_action_look_pitch"]),
                    "vertical_intent": int(info["action_debug_vertical_intent"]),
                    "fire": bool(info["effective_action_fire"]),
                    "hook": int(info["action_debug_hook"]),
                    "weapon": int(info["action_debug_weapon"]),
                },
                "observation": {
                    "tick": int(observation.tick),
                    "slot": int(observation.bot_slot),
                    "yaw": float(observation.yaw),
                    "pitch": float(observation.pitch),
                    "self_state": [float(value) for value in observation.self_state],
                    "action_debug": [float(value) for value in observation.action_debug],
                    "terminal_reason": int(observation.terminal_reason),
                },
            })
        return {"ordinal": ordinal, "clients": clients}

    def _server_command(
        self,
        q2ded: Path,
        root: Path,
        server_port: int,
        telemetry_port: int,
        token: str,
        spec: ScenarioSpec,
    ) -> list[str]:
        config = self.configuration
        fault = spec.injected_fault
        server_fault = _server_fault_for(spec)
        values = {
            "game": config.game,
            "port": str(server_port),
            "dedicated": "1",
            "deathmatch": "1",
            # Keep real liveness shorter than the barrier timeout so SIGKILL
            # proves disconnect handling, while sustained-drop (live client,
            # missing readiness) separately proves the barrier timeout path.
            "timeout": "1",
            "cheats": "1",
            "maxclients": str(CLIENT_COUNT),
            "autospawn": "0",
            "allow_client_bot_controls": "0",
            "ml_enabled": "1",
            "ml_async": "0",
            "sv_ml_frame_barrier": "1",
            "sv_ml_frame_barrier_clients": str(CLIENT_COUNT),
            "sv_ml_frame_barrier_timeout_ms": str(config.timeout_ms),
            "sv_ml_frame_barrier_test_mode": "1",
            "sv_ml_frame_barrier_test_fault": server_fault,
            "sv_ml_frame_barrier_test_tick": "5",
            "sv_ml_frame_barrier_test_map": config.epoch_map_name,
            "ml_bot_slot": "99",
            "ml_teacher_enabled": "0",
            "ml_game_seed": str(spec.seed),
            "ml_client_telemetry": "1",
            "ml_client_telemetry_port": str(telemetry_port),
            "ml_client_telemetry_token": token,
            "use_mapqueue": "0",
            # Lithium defaults to observer-first operation.  Qualification is
            # a playable-client barrier proof, so both observer entry paths
            # are explicitly sealed off instead of inheriting host config.
            "use_startobserver": "0",
            "use_startchasecam": "0",
        }
        # The telemetry bearer token must never appear in argv/procfs.  The
        # isolated runtime is private and the cfg is created mode 0600.
        cfg_name = "frame_barrier_qualification.cfg"
        cfg_path = root / config.game / cfg_name
        lines = []
        for name, value in values.items():
            if value != "" and _TOKEN.fullmatch(str(value)) is None:
                raise QualificationError(f"unsafe server cvar value for {name}")
            lines.append(f'set {name} "{value}"')
        lines.append(f'map "{config.map_name}"')
        _atomic_write(cfg_path, ("\n".join(lines) + "\n").encode("utf-8"))
        if cfg_path.stat().st_mode & 0o077:
            raise QualificationError("qualification server config is not private")
        return [
            str(q2ded),
            "+set", "game", config.game,
            "+set", "dedicated", "1",
            "+exec", cfg_name,
        ]

    @staticmethod
    def _stop(process: subprocess.Popen[Any] | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    @staticmethod
    def _wait_process_launch(env: Any, client_name: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process = env._process
            if process is not None:
                if process.poll() is not None:
                    raise QualificationError(
                        f"client {client_name} exited during launch"
                    )
                # Only this client process exists during the grace period, so
                # its first accepted connect necessarily owns the next slot.
                time.sleep(0.25)
                return
            time.sleep(0.005)
        raise QualificationError(
            f"client {client_name} process did not launch"
        )

    def _start_exact_roster(
        self,
        batch: Any,
        envs: Sequence[Any],
        *,
        delay_final: bool,
        before_final: Any = None,
    ) -> list[dict[str, Any]]:
        # The engine slot is allocated by connect arrival, while qualification
        # binds client-id suffix to that slot. Launch in suffix order and wait
        # for the engine-owned connected witness before releasing the next
        # process; telemetry waits remain concurrent.
        futures = []
        for slot, env in enumerate(envs):
            if slot == CLIENT_COUNT - 1 and before_final is not None:
                before_final()
            if delay_final and slot == CLIENT_COUNT - 1:
                time.sleep(self.configuration.timeout_ms / 1000.0 + 0.25)
            futures.append(batch._executor.submit(env.start))
            self._wait_process_launch(env, f"qual-{slot:02d}")
        telemetry = [future.result() for future in futures]
        for env, sample in zip(envs, telemetry):
            env.initial_result(sample, vector=batch.vector)
        initial_life_epochs = {
            sample.client_id: int(sample.causal.client_life_epoch)
            for sample in telemetry
        }
        if (
            set(initial_life_epochs)
            != {f"qual-{slot:02d}" for slot in range(CLIENT_COUNT)}
            or any(epoch <= 0 for epoch in initial_life_epochs.values())
            or any(
                sample.client_id != f"qual-{sample.client_slot:02d}"
                for sample in telemetry
            )
        ):
            raise QualificationError(
                "ordered roster bootstrap life-epoch identity differs"
            )
        # This is deliberately checked before batch._started is set.  The
        # initial telemetry is action-free, and an observer/spectator/noclip
        # identity must never be allowed to produce a trajectory action.
        role_preflight = [_bootstrap_role_record(sample) for sample in telemetry]
        # reset() normally owns this initialization. Qualification stages
        # env.start() calls in suffix order to prove deterministic slot
        # admission, so close that same public-telemetry invariant explicitly.
        batch._client_life_epochs = initial_life_epochs
        batch._death_life_epochs.clear()
        batch._action_hold_epochs.clear()
        batch._started = True
        return role_preflight

    def run(
        self,
        spec: ScenarioSpec,
        action_table: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ports = _PortAllocator.reserve(CLIENT_COUNT * 2 + 4)
        try:
            return self._run_with_ports(spec, action_table, ports)
        finally:
            _PortAllocator.release(ports)

    def _run_with_ports(
        self,
        spec: ScenarioSpec,
        action_table: Mapping[str, Any],
        ports: Sequence[int],
    ) -> Mapping[str, Any]:
        from harness.client_batch import Q2NetworkClientBatch
        from harness.client_env import Q2NetworkClientEnv

        config = self.configuration
        if len(ports) != CLIENT_COUNT * 2 + 4:
            raise QualificationError("scenario loopback port lease is incomplete")
        server_port, telemetry_port = ports[:2]
        harness_ports = ports[2:2 + CLIENT_COUNT + 1]
        qports = ports[3 + CLIENT_COUNT:4 + CLIENT_COUNT * 2]
        token = secrets.token_urlsafe(32)
        if _TOKEN.fullmatch(token) is None:
            raise QualificationError("generated conduit token is malformed")
        server = None
        batch = None
        extra_env = None
        trajectory = []
        death_lifecycle_boundary_rows = []
        action_hold_boundary_rows = []
        routed_role_preflight: list[dict[str, Any]] = []
        accepted_rounds = 0
        boundary_rounds = 0
        actions_dispatched_during_epoch_drain = 0
        outcome = "fatal"
        exception_name = None
        exception_detail = None
        with tempfile.TemporaryDirectory(prefix=f"q2-barrier-{spec.name}-") as temporary:
            runtime_root = Path(temporary)
            q2ded, client = self._runtime(runtime_root)
            log_path = runtime_root / "server.log"
            log_stream = log_path.open("w+b")
            try:
                server = subprocess.Popen(
                    self._server_command(
                        q2ded, runtime_root, server_port, telemetry_port, token, spec
                    ),
                    cwd=runtime_root,
                    stdin=subprocess.PIPE,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
                time.sleep(config.server_warmup_seconds)
                if server.poll() is not None:
                    raise QualificationError("q2ded exited before roster launch")
                envs = []
                for index in range(CLIENT_COUNT):
                    deterministic = not (
                        spec.injected_fault == "mixed-cvar" and index == 3
                    )
                    envs.append(Q2NetworkClientEnv(
                        server=f"127.0.0.1:{server_port}",
                        telemetry_server=f"127.0.0.1:{telemetry_port}",
                        telemetry_token=token,
                        client_binary=str(client),
                        client_root=str(runtime_root),
                        client_data_root=str(runtime_root / "clients"),
                        harness_host="127.0.0.1",
                        harness_port=harness_ports[index],
                        qport=qports[index],
                        client_id=f"qual-{index:02d}",
                        name=f"qual-{index:02d}",
                        game=config.game,
                        timeout=max(5.0, config.timeout_ms / 1000.0 + 3.0),
                        deterministic_frame_barrier=deterministic,
                        extra_args=(
                            "+set", "ml_frame_barrier_version", "1",
                            "+set", "ml_frame_barrier_capability", "1",
                        ),
                    ))
                batch = Q2NetworkClientBatch(
                    envs,
                    vector=False,
                    round_timeout=max(1.0, config.timeout_ms / 1000.0 + 0.5),
                    deterministic_frame_barrier=(spec.injected_fault != "mixed-cvar"),
                )
                probe_rejected = False
                if spec.injected_fault in ("fifth", "human"):
                    extra_env = Q2NetworkClientEnv(
                        server=f"127.0.0.1:{server_port}",
                        telemetry_server=f"127.0.0.1:{telemetry_port}",
                        telemetry_token=token,
                        client_binary=str(client),
                        client_root=str(runtime_root),
                        client_data_root=str(runtime_root / "clients"),
                        harness_host="127.0.0.1",
                        harness_port=harness_ports[-1],
                        qport=qports[-1],
                        client_id=(
                            "qual-04" if spec.injected_fault == "fifth"
                            else "human"
                        ),
                        name="unqualified",
                        game=config.game,
                        timeout=2.0,
                        deterministic_frame_barrier=(
                            spec.injected_fault == "fifth"
                        ),
                        extra_args=(
                            "+set", "ml_frame_barrier_version", "1",
                            "+set", "ml_frame_barrier_capability", "1",
                        ),
                    )

                def probe_unqualified() -> None:
                    nonlocal probe_rejected
                    if extra_env is None:
                        raise QualificationError("unqualified probe is absent")
                    try:
                        extra_env.start()
                    except Exception:
                        probe_rejected = True
                    else:
                        raise QualificationError(
                            "unqualified fifth/human client connected"
                        )
                    finally:
                        try:
                            extra_env.close()
                        except Exception:
                            pass

                routed_role_preflight = self._start_exact_roster(
                    batch,
                    envs,
                    delay_final=(spec.injected_fault == "load-delay"),
                    before_final=(
                        probe_unqualified
                        if spec.injected_fault in ("fifth", "human")
                        else None
                    ),
                )

                if spec.injected_fault in ("fifth", "human"):
                    if probe_rejected:
                        outcome = "rejected"
                    else:
                        raise QualificationError(
                            "unqualified fifth/human probe produced no rejection"
                        )
                else:
                    attempt = 0
                    epoch_drain_observed = False
                    drain_victim_killed = False
                    while accepted_rounds < FRAME_COUNT and attempt < FRAME_COUNT + 64:
                        row = action_table["rows"][accepted_rounds]
                        if spec.injected_fault == "sigkill" and accepted_rounds == 4:
                            victim = envs[0]._process
                            if victim is None:
                                raise QualificationError("SIGKILL victim was not started")
                            victim.kill()
                        if (
                            spec.injected_fault == "drain-sigkill"
                            and epoch_drain_observed
                            and not drain_victim_killed
                        ):
                            victim = envs[0]._process
                            if victim is None:
                                raise QualificationError(
                                    "drain SIGKILL victim was not started"
                                )
                            victim.kill()
                            drain_victim_killed = True
                        before_actions = batch.metrics.actions_dispatched
                        try:
                            result = batch.collect_round(
                                self._actions(row), policy_version=attempt
                            )
                        except Exception as error:
                            exception_name = type(error).__name__
                            exception_detail = str(error)[:512]
                            outcome = "fatal"
                            if spec.injected_fault in (
                                "sigkill", "drain-sigkill", "sustained-drop"
                            ):
                                time.sleep(
                                    3.0 if spec.injected_fault in (
                                        "sigkill", "drain-sigkill"
                                    ) else 1.25
                                )
                            break
                        dispatched = (
                            batch.metrics.actions_dispatched - before_actions
                        )
                        if all(
                            info.get("trainable_transition") is True
                            for info in result.infos
                        ):
                            trajectory.append(self._trajectory_row(
                                result,
                                ordinal=accepted_rounds + 1,
                                action_row=row,
                                map_epochs={
                                    config.map_name: 1,
                                    config.epoch_map_name: 2,
                                },
                            ))
                            accepted_rounds += 1
                        else:
                            death_flags = [
                                info.get("death_lifecycle_resync") is True
                                for info in result.infos
                            ]
                            if any(death_flags):
                                if not all(death_flags) or any(
                                    info.get("trainable_transition") is not False
                                    for info in result.infos
                                ):
                                    raise QualificationError(
                                        "death lifecycle boundary was not whole-batch nontrainable"
                                    )
                                death_lifecycle_boundary_rows.append({
                                    "round_id": int(result.round_id),
                                    "trajectory_ordinal": accepted_rounds + 1,
                                    "clients": [
                                        {
                                            "client_id": str(info["client_id"]),
                                            "client_slot": int(info["client_slot"]),
                                            "sequence": int(info["sequence"]),
                                            "server_frame": int(info["server_frame"]),
                                            "map_name": str(info["map"]),
                                            "map_epoch": int(info["map_epoch"]),
                                            "action_tick": int(info["action_tick"]),
                                            "echo_tick": int(
                                                info["authoritative_echo_tick"]
                                            ),
                                            "action_generation": (
                                                int(info["action_debug_flags"])
                                                >> 16
                                            ) & 0xFF,
                                            "client_life_epoch": int(
                                                info["authoritative_client_life_epoch"]
                                            ),
                                            "phase": str(
                                                info["death_lifecycle_phase"]
                                            ),
                                            "observed_terminal_reason": int(
                                                info["observed_terminal_reason"]
                                            ),
                                            "causal_echo_valid": bool(
                                                info["causal_echo_valid"]
                                            ),
                                            "causal_facts_complete": bool(
                                                info["causal_facts_complete"]
                                            ),
                                            "causal_transition_trainable": bool(
                                                info["causal_transition_trainable"]
                                            ),
                                            "causal_role_playing": bool(
                                                info["causal_role_playing"]
                                            ),
                                            "causal_role_public_pm_normal": bool(
                                                info[
                                                    "causal_role_public_pm_normal"
                                                ]
                                            ),
                                            "expected_action": dict(expected),
                                            "applied_action": {
                                                "move_forward": float(
                                                    info["action_debug_movement"][0]
                                                ),
                                                "move_right": float(
                                                    info["action_debug_movement"][1]
                                                ),
                                                "look_yaw": float(
                                                    info["action_debug_movement"][2]
                                                ),
                                                "look_pitch": float(
                                                    info["action_debug_movement"][3]
                                                ),
                                                "vertical_intent": int(
                                                    info["action_debug_vertical_intent"]
                                                ),
                                                "fire": bool(
                                                    info["action_debug_fire"]
                                                ),
                                                "hook": int(
                                                    info["action_debug_hook"]
                                                ),
                                                "weapon": int(
                                                    info["action_debug_weapon"]
                                                ),
                                            },
                                            "trainable_transition": False,
                                        }
                                        for info, expected in zip(
                                            result.infos, row["clients"]
                                        )
                                    ],
                                })
                            action_flags = [
                                info.get("action_state_resync") is True
                                for info in result.infos
                            ]
                            if any(action_flags):
                                if not all(action_flags) or any(
                                    info.get("trainable_transition") is not False
                                    for info in result.infos
                                ):
                                    raise QualificationError(
                                        "action hold boundary was not whole-batch nontrainable"
                                    )
                                action_hold_boundary_rows.append({
                                    "round_id": int(result.round_id),
                                    "trajectory_ordinal": accepted_rounds + 1,
                                    "clients": [
                                        {
                                            "client_id": str(info["client_id"]),
                                            "client_slot": int(info["client_slot"]),
                                            "sequence": int(info["sequence"]),
                                            "server_frame": int(info["server_frame"]),
                                            "map_name": str(info["map"]),
                                            "map_epoch": int(info["map_epoch"]),
                                            "action_tick": int(info["action_tick"]),
                                            "echo_tick": int(
                                                info["authoritative_echo_tick"]
                                            ),
                                            "action_generation": int(
                                                info["authoritative_action_generation"]
                                            ),
                                            "client_life_epoch": int(
                                                info["authoritative_client_life_epoch"]
                                            ),
                                            "phase": str(info["action_state_phase"]),
                                            "causal_echo_valid": bool(
                                                info["causal_echo_valid"]
                                            ),
                                            "causal_facts_complete": bool(
                                                info["causal_facts_complete"]
                                            ),
                                            "causal_transition_trainable": bool(
                                                info["causal_transition_trainable"]
                                            ),
                                            "causal_role_playing": bool(
                                                info["causal_role_playing"]
                                            ),
                                            "causal_role_public_pm_normal": bool(
                                                info[
                                                    "causal_role_public_pm_normal"
                                                ]
                                            ),
                                            "expected_action": dict(expected),
                                            "applied_action": {
                                                "move_forward": float(
                                                    info["action_debug_movement"][0]
                                                ),
                                                "move_right": float(
                                                    info["action_debug_movement"][1]
                                                ),
                                                "look_yaw": float(
                                                    info["action_debug_movement"][2]
                                                ),
                                                "look_pitch": float(
                                                    info["action_debug_movement"][3]
                                                ),
                                                "vertical_intent": int(
                                                    info[
                                                        "action_debug_vertical_intent"
                                                    ]
                                                ),
                                                "fire": bool(
                                                    info["action_debug_fire"]
                                                ),
                                                "hook": int(
                                                    info["action_debug_hook"]
                                                ),
                                                "weapon": int(
                                                    info["action_debug_weapon"]
                                                ),
                                            },
                                            "trainable_transition": False,
                                        }
                                        for info, expected in zip(
                                            result.infos, row["clients"]
                                        )
                                    ],
                                })
                            if epoch_drain_observed:
                                actions_dispatched_during_epoch_drain += dispatched
                            epoch_drain_observed = True
                            boundary_rounds += 1
                        attempt += 1
                    if accepted_rounds == FRAME_COUNT:
                        outcome = "completed"
            except Exception as error:
                if isinstance(error, QualificationError):
                    raise
                exception_name = type(error).__name__
                exception_detail = str(error)[:512]
                outcome = "rejected" if spec.expected_outcome == "rejected" else "fatal"
            finally:
                # End q2ded before closing healthy clients; otherwise normal
                # qualification teardown fabricates a barrier disconnect fault.
                self._stop(server)
                if extra_env is not None:
                    try:
                        extra_env.close()
                    except Exception:
                        pass
                if batch is not None:
                    try:
                        batch.close()
                    except Exception:
                        pass
                log_stream.flush()
                log_stream.close()

            log = log_path.read_text(encoding="utf-8", errors="replace")
            events = _parse_events(log)
            accepted_acks = {
                int(event["slot"])
                for event in events
                if event.get("event") == "route_ack"
                and event.get("slot", "").isdigit()
            }
            bootstrap_count = _event_count(events, "bootstrap_commit")
            fault_events = [
                event for event in events if event.get("event") == "fatal"
            ]
            epoch_drains = _event_count(events, "epoch_drain_enter")
            new_bootstraps = max(0, bootstrap_count - 1)
            if spec.injected_fault != "epoch-drain":
                actions_dispatched_during_epoch_drain = 0
            raw = {
                "schema": RAW_SCHEMA,
                "scenario": spec.name,
                "seed": spec.seed,
                "cold_launch": spec.cold_launch,
                "expected_outcome": spec.expected_outcome,
                "observed_outcome": outcome,
                "injected_fault": spec.injected_fault,
                "client_ids": [f"qual-{index:02d}" for index in range(CLIENT_COUNT)],
                "accepted_route_ack_slots": sorted(accepted_acks),
                "routed_role_runtime": {
                    "use_startobserver": 0,
                    "use_startchasecam": 0,
                },
                "routed_role_preflight": routed_role_preflight,
                "action_free_bootstrap_frames": bootstrap_count,
                "accepted_synchronized_frames": accepted_rounds,
                "boundary_rounds": boundary_rounds,
                "trajectory_sha256": _sha256_bytes(_canonical_bytes(trajectory)),
                "trajectory_rows": trajectory,
                "death_lifecycle_boundary_rows": death_lifecycle_boundary_rows,
                "action_hold_boundary_rows": action_hold_boundary_rows,
                "death_lifecycle_resyncs": (
                    0 if batch is None else batch.metrics.death_lifecycle_resyncs
                ),
                "action_state_resyncs": (
                    0 if batch is None else batch.metrics.action_state_resyncs
                ),
                "structured_events": events,
                "structured_events_sha256": _sha256_bytes(
                    _canonical_bytes(events)
                ),
                "fault_event_count": len(fault_events),
                "exception": exception_name,
                "exception_detail": exception_detail,
                "epoch_drains": epoch_drains,
                "new_epoch_bootstrap_frames": new_bootstraps,
                "actions_dispatched_during_epoch_drain": (
                    actions_dispatched_during_epoch_drain
                ),
                "server_exit_code": None if server is None else server.returncode,
                "barrier_timeout_ms": config.timeout_ms,
                "test_mode": False,
            }
            return _seal(raw)


def _require_raw(
    spec: ScenarioSpec,
    raw: Mapping[str, Any],
    action_table: Mapping[str, Any],
    *,
    test_mode: bool,
    timeout_ms: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise QualificationError(f"scenario {spec.name} returned no evidence")
    value = dict(raw)
    digest = value.pop("evidence_sha256", None)
    if not _valid_sha256(digest) or digest != _sha256_bytes(_canonical_bytes(value)):
        raise QualificationError(f"scenario {spec.name} digest differs")
    expected = {
        "schema": RAW_SCHEMA,
        "scenario": spec.name,
        "seed": spec.seed,
        "cold_launch": spec.cold_launch,
        "expected_outcome": spec.expected_outcome,
        "observed_outcome": spec.expected_outcome,
        "injected_fault": spec.injected_fault,
        "client_ids": [f"qual-{index:02d}" for index in range(CLIENT_COUNT)],
        "barrier_timeout_ms": timeout_ms,
        "test_mode": test_mode,
    }
    mismatches = {
        name: (value.get(name), wanted)
        for name, wanted in expected.items()
        if value.get(name) != wanted
    }
    # Rejection cases may fail before the exact four-client roster exists.
    # Every completed/fatal scenario, however, can dispatch only after this
    # action-free role seal has passed.
    if spec.expected_outcome in ("completed", "fatal"):
        mismatches.update(_validate_bootstrap_roles(value))
    if spec.expected_outcome == "completed":
        if value.get("fault_event_count") != 0:
            mismatches["fault_event_count"] = (
                value.get("fault_event_count"), 0,
            )
        if value.get("accepted_route_ack_slots") != [0, 1, 2, 3]:
            mismatches["accepted_route_ack_slots"] = (
                value.get("accepted_route_ack_slots"), [0, 1, 2, 3]
            )
        expected_bootstraps = 2 if spec.injected_fault == "epoch-drain" else 1
        if value.get("action_free_bootstrap_frames") != expected_bootstraps:
            mismatches["action_free_bootstrap_frames"] = (
                value.get("action_free_bootstrap_frames"), expected_bootstraps,
            )
        if value.get("accepted_synchronized_frames") != FRAME_COUNT:
            mismatches["accepted_synchronized_frames"] = (
                value.get("accepted_synchronized_frames"), FRAME_COUNT,
            )
        if (
            not _valid_sha256(value.get("trajectory_sha256"))
            or not isinstance(value.get("trajectory_rows"), list)
            or value.get("trajectory_sha256")
            != _sha256_bytes(_canonical_bytes(value["trajectory_rows"]))
        ):
            mismatches["trajectory_sha256"] = (
                value.get("trajectory_sha256"), "exact trajectory digest",
            )
        rows = value.get("trajectory_rows")
        if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
            mismatches["trajectory_rows"] = (
                len(rows) if isinstance(rows, list) else rows, FRAME_COUNT,
            )
        else:
            prior: tuple[int, int] | None = None
            prior_clients: dict[str, Mapping[str, Any]] = {}
            trajectory_epochs: set[int] = set()
            life_epochs: dict[str, tuple[int, int]] = {}
            client_sequences: dict[str, int] = {}
            map_names_by_epoch: dict[int, str] = {}
            life_epoch_changes: list[tuple[str, int, int, int]] = []
            death_frame_gaps = 0
            lifecycle_rows = value.get(
                "death_lifecycle_boundary_rows"
                if spec.injected_fault == "death"
                else "action_hold_boundary_rows"
            )
            lifecycle_gap_index = (
                lifecycle_rows[0].get("trajectory_ordinal") - 1
                if isinstance(lifecycle_rows, list) and lifecycle_rows
                and isinstance(lifecycle_rows[0], Mapping)
                and type(lifecycle_rows[0].get("trajectory_ordinal")) is int
                else None
            )
            for index, row in enumerate(rows):
                clients = row.get("clients") if isinstance(row, Mapping) else None
                frames = {
                    item.get("server_frame")
                    for item in clients or () if isinstance(item, Mapping)
                }
                if len(clients or ()) != CLIENT_COUNT or len(frames) != 1:
                    mismatches[f"trajectory_row_{index}"] = (row, "four one-frame clients")
                    continue
                frame = next(iter(frames))
                epochs = {
                    item.get("map_epoch")
                    for item in clients if isinstance(item, Mapping)
                }
                if len(epochs) != 1 or type(next(iter(epochs))) is not int:
                    mismatches[f"map_epoch_{index}"] = (epochs, "one integer")
                    continue
                epoch = next(iter(epochs))
                trajectory_epochs.add(epoch)
                map_names = {
                    item.get("map_name")
                    for item in clients if isinstance(item, Mapping)
                }
                if len(map_names) != 1 or not isinstance(
                    next(iter(map_names)), str
                ):
                    mismatches[f"map_name_{index}"] = (
                        map_names, "one exact map name",
                    )
                else:
                    map_name = next(iter(map_names))
                    prior_map_name = map_names_by_epoch.setdefault(epoch, map_name)
                    if prior_map_name != map_name:
                        mismatches[f"map_name_epoch_{index}"] = (
                            map_name, prior_map_name,
                        )
                if prior is not None:
                    prior_epoch, prior_frame = prior
                    if epoch == prior_epoch and frame != prior_frame + 1:
                        prior_slot_zero = prior_clients.get("qual-00", {})
                        current_slot_zero = next(
                            (
                                item for item in clients
                                if isinstance(item, Mapping)
                                and item.get("client_id") == "qual-00"
                            ),
                            {},
                        )
                        prior_life = prior_slot_zero.get("causal", {}).get(
                            "client_life_epoch"
                        )
                        current_life = current_slot_zero.get("causal", {}).get(
                            "client_life_epoch"
                        )
                        allowed_death_gap = (
                            spec.injected_fault in ("death", "same-life-hold")
                            and index == lifecycle_gap_index
                            and frame > prior_frame + 1
                            and prior_slot_zero.get("observation", {}).get(
                                "terminal_reason"
                            ) == 0
                            and current_slot_zero.get("observation", {}).get(
                                "terminal_reason"
                            ) == 0
                            and type(prior_life) is int
                            and current_life == prior_life + (
                                1 if spec.injected_fault == "death" else 0
                            )
                        )
                        if allowed_death_gap:
                            death_frame_gaps += 1
                        else:
                            mismatches[f"frame_delta_{index}"] = (
                                frame - prior_frame, 1,
                            )
                    elif epoch != prior_epoch and epoch != prior_epoch + 1:
                        mismatches[f"epoch_delta_{index}"] = (
                            epoch - prior_epoch, 1,
                        )
                prior = (epoch, frame)
                prior_clients = {
                    item.get("client_id"): item
                    for item in clients if isinstance(item, Mapping)
                    and isinstance(item.get("client_id"), str)
                }
                expected_clients = {
                    item["client_id"]: item
                    for item in action_table["rows"][index]["clients"]
                }
                for item in clients:
                    action_tick = item.get("action_tick")
                    client_id = item.get("client_id")
                    expected_slot = (
                        int(str(client_id).removeprefix("qual-"))
                        if isinstance(client_id, str)
                        and re.fullmatch(r"qual-0[0-3]", client_id)
                        else None
                    )
                    if expected_slot is None or item.get("client_slot") != expected_slot:
                        mismatches[f"client_identity_{index}_{client_id}"] = (
                            (client_id, item.get("client_slot")),
                            "qual suffix equals slot",
                        )
                    sequence = item.get("sequence")
                    if type(sequence) is not int or sequence <= 0:
                        mismatches[f"sequence_{index}_{client_id}"] = (
                            sequence, "positive integer",
                        )
                    elif client_id in client_sequences and sequence <= client_sequences[
                        client_id
                    ]:
                        mismatches[f"sequence_monotonic_{index}_{client_id}"] = (
                            sequence, f"> {client_sequences[client_id]}",
                        )
                    if type(sequence) is int and sequence > 0:
                        client_sequences[str(client_id)] = sequence
                    causal_value = item.get("causal")
                    life_epoch = (
                        causal_value.get("client_life_epoch")
                        if isinstance(causal_value, Mapping) else None
                    )
                    if type(life_epoch) is not int or life_epoch <= 0:
                        mismatches[f"life_epoch_{index}_{client_id}"] = (
                            life_epoch, "positive integer",
                        )
                    elif client_id in life_epochs:
                        previous_map_epoch, previous_life_epoch = life_epochs[
                            client_id
                        ]
                        if epoch == previous_map_epoch and life_epoch != previous_life_epoch:
                            life_epoch_changes.append((
                                str(client_id), previous_life_epoch,
                                life_epoch, index,
                            ))
                    if type(life_epoch) is int and life_epoch > 0:
                        life_epochs[str(client_id)] = (epoch, life_epoch)
                    expected_generation = (
                        action_tick % 192 + 1
                        if type(action_tick) is int else None
                    )
                    if (
                        item.get("echo_tick") != item.get("action_tick")
                        or item.get("applied_action_tick") != item.get("action_tick")
                        or item.get("action_generation") != expected_generation
                        or item.get("causal", {}).get("echo_tick") != item.get("action_tick")
                        or item.get("causal", {}).get("action_generation")
                        != expected_generation
                        or item.get("causal", {}).get("echo_valid") is not True
                        or item.get("causal", {}).get("facts_complete") is not True
                        or item.get("causal", {}).get("transition_trainable")
                        is not True
                        or item.get("causal", {}).get("role_playing") is not True
                        or item.get("causal", {}).get("role_public_pm_normal")
                        is not True
                    ):
                        mismatches[f"echo_{index}_{item.get('client_id')}"] = (
                            item, "exact action/applied/causal echo",
                        )
                    expected_action = expected_clients.get(client_id)
                    applied_action = item.get("applied_action")
                    if item.get("expected_action") != expected_action or not isinstance(
                        applied_action, Mapping
                    ):
                        mismatches[f"action_{index}_{client_id}"] = (
                            (item.get("expected_action"), applied_action),
                            expected_action,
                        )
                        continue
                    for name, tolerance in (
                        ("move_forward", 0.05), ("move_right", 0.05),
                        ("look_yaw", 0.25), ("look_pitch", 0.25),
                    ):
                        if not math.isclose(
                            float(applied_action.get(name, math.inf)),
                            float(expected_action[name]),
                            rel_tol=0.0,
                            abs_tol=tolerance,
                        ):
                            mismatches[f"applied_{index}_{client_id}_{name}"] = (
                                applied_action.get(name), expected_action[name],
                            )
                    for name in ("vertical_intent", "fire", "hook", "weapon"):
                        if applied_action.get(name) != expected_action[name]:
                            mismatches[f"applied_{index}_{client_id}_{name}"] = (
                                applied_action.get(name), expected_action[name],
                            )
            if spec.injected_fault in ("death", "same-life-hold"):
                mismatches.update(_validate_stock_lifecycle_boundaries(
                    spec, value, action_table
                ))
                if death_frame_gaps != 1:
                    mismatches["lifecycle_frame_gap_count"] = (
                        death_frame_gaps, 1,
                    )
            if (
                spec.injected_fault not in ("death", "same-life-hold")
                and life_epoch_changes
            ):
                mismatches["unexpected_life_epoch_changes"] = (
                    life_epoch_changes, [],
                )
            elif spec.injected_fault not in ("death", "same-life-hold") and (
                value.get("death_lifecycle_boundary_rows") not in ([], None)
                or value.get("death_lifecycle_resyncs") not in (0, None)
            ):
                mismatches["unexpected_death_lifecycle_boundaries"] = (
                    (
                        value.get("death_lifecycle_boundary_rows"),
                        value.get("death_lifecycle_resyncs"),
                    ),
                    ([], 0),
                )
            if spec.injected_fault not in ("death", "same-life-hold") and (
                value.get("action_hold_boundary_rows") not in ([], None)
                or value.get("action_state_resyncs") not in (0, None)
            ):
                mismatches["unexpected_action_hold_boundaries"] = (
                    (
                        value.get("action_hold_boundary_rows"),
                        value.get("action_state_resyncs"),
                    ),
                    ([], 0),
                )
            if not test_mode:
                wanted_epochs = {1, 2} if spec.injected_fault == "epoch-drain" else {1}
                if trajectory_epochs != wanted_epochs:
                    mismatches["trajectory_map_epochs"] = (
                        sorted(trajectory_epochs), sorted(wanted_epochs),
                    )
        if spec.injected_fault == "epoch-drain" and (
            value.get("epoch_drains") != 1
            or value.get("new_epoch_bootstrap_frames") != 1
            or value.get("actions_dispatched_during_epoch_drain") != 0
        ):
            mismatches["epoch_drain"] = (
                (
                    value.get("epoch_drains"),
                    value.get("new_epoch_bootstrap_frames"),
                    value.get("actions_dispatched_during_epoch_drain"),
                ),
                (1, 1, 0),
            )
    elif spec.expected_outcome == "fatal":
        if value.get("accepted_route_ack_slots") != [0, 1, 2, 3]:
            mismatches["accepted_route_ack_slots"] = (
                value.get("accepted_route_ack_slots"), [0, 1, 2, 3]
            )
        if value.get("action_free_bootstrap_frames") != 1:
            mismatches["action_free_bootstrap_frames"] = (
                value.get("action_free_bootstrap_frames"), 1,
            )
        if type(value.get("fault_event_count")) is not int or value[
            "fault_event_count"
        ] < 1:
            mismatches["fault_event_count"] = (
                value.get("fault_event_count"), ">=1",
            )
    events = value.get("structured_events")
    if not isinstance(events, list) or value.get("structured_events_sha256") != (
        _sha256_bytes(_canonical_bytes(events)) if isinstance(events, list) else None
    ):
        mismatches["structured_events"] = (
            value.get("structured_events_sha256"), "exact event-list digest",
        )
        events = []

    def has_event(name: str, **wanted: str) -> bool:
        return any(
            isinstance(event, Mapping)
            and event.get("event") == name
            and all(event.get(key) == expected for key, expected in wanted.items())
            for event in events
        )

    if not has_event(
        "test_mode_sealed",
        value="1",
        dedicated="1",
        roster="exact",
        controls_immutable="1",
        fault_vocabulary="v1",
    ):
        mismatches["test_mode_startup_seal"] = (False, True)
    if not has_event("startup_clock_sync", settle_frames="2"):
        mismatches["startup_clock_sync"] = (False, "settle_frames=2")
    map_resets = [
        event for event in events if event.get("event") == "map_reset"
    ]
    if not map_resets or any(
        event.get("test_controls_sealed") != "1"
        or event.get("test_mode") != "1"
        or event.get("test_fault") != _server_fault_for(spec)
        or event.get("test_tick") != "5"
        or event.get("dedicated") != "1"
        or not event.get("test_map")
        for event in map_resets
    ):
        mismatches["map_reset_test_control_seal"] = (map_resets, "sealed")
    elif map_resets[0].get("map") == map_resets[0].get("test_map"):
        mismatches["map_reset_distinct_test_map"] = (
            map_resets[0].get("map"), map_resets[0].get("test_map")
        )

    if spec.expected_outcome == "completed" and events:
        bootstrap_indices = [
            index for index, event in enumerate(events)
            if event.get("event") == "bootstrap_commit"
        ]
        if not bootstrap_indices:
            mismatches["bootstrap_event"] = (False, True)
        else:
            first_bootstrap = bootstrap_indices[0]
            ready_slots = {
                int(event["slot"])
                for event in events[:first_bootstrap]
                if event.get("event") == "bootstrap_ready"
                and event.get("slot", "").isdigit()
            }
            if ready_slots != set(range(CLIENT_COUNT)):
                mismatches["bootstrap_ready_before_commit"] = (
                    sorted(ready_slots), list(range(CLIENT_COUNT)),
                )
        route_ack_events = [
            event for event in events if event.get("event") == "route_ack"
        ]
        for slot in range(CLIENT_COUNT):
            if not any(
                event.get("slot") == str(slot)
                and event.get("client_id") == f"qual-{slot:02d}"
                and event.get("wire") == str(WIRE_VERSION)
                and event.get("barrier") == str(BARRIER_VERSION)
                and event.get("capability") == str(BARRIER_CAPABILITY)
                for event in route_ack_events
            ):
                mismatches[f"route_ack_{slot}"] = (False, "exact accepted ACK")
        event_epochs = []
        current_event_epoch = None
        for event in events:
            if event.get("event") == "map_reset" and event.get(
                "map_epoch", ""
            ).isdigit():
                current_event_epoch = int(event["map_epoch"])
            event_epochs.append((event, current_event_epoch))
        for row in value.get("trajectory_rows", ()):
            clients = row.get("clients", ()) if isinstance(row, Mapping) else ()
            if not clients:
                continue
            action_tick = clients[0].get("action_tick")
            map_epoch = clients[0].get("map_epoch")
            apply_slots = [
                int(event["slot"])
                for event, event_epoch in event_epochs
                if event.get("event") == "apply"
                and event.get("action_tick") == str(action_tick)
                and event_epoch == map_epoch
                and event.get("slot", "").isdigit()
            ]
            if apply_slots != [0, 1, 2, 3]:
                mismatches[f"apply_order_{action_tick}"] = (
                    apply_slots, [0, 1, 2, 3],
                )
        if not has_event("deferred_control", order="hook_then_weapon"):
            mismatches["deferred_control_order"] = (False, "hook_then_weapon")
        if spec.injected_fault == "epoch-drain":
            resets = [
                event for event in events if event.get("event") == "map_reset"
            ]
            if (
                len(resets) < 2
                or resets[0].get("map_epoch") != "1"
                or resets[-1].get("map_epoch") != "2"
                or resets[0].get("map") == resets[-1].get("map")
            ):
                mismatches["distinct_epoch_map_reset"] = (
                    resets, "distinct epoch 1 -> epoch 2 maps",
                )

    required_events: list[tuple[str, dict[str, str]]] = []
    if spec.injected_fault == "duplicate":
        required_events = [
            ("duplicate_command", {"result": "idempotent"}),
            ("duplicate_ready", {"result": "idempotent"}),
        ]
    elif spec.injected_fault == "stale":
        required_events = [
            ("stale_command", {"result": "stale"}),
            ("stale_ready", {"result": "stale"}),
        ]
    elif spec.injected_fault == "brief-drop":
        required_events = [
            ("brief_drop_recovery", {
                "source": "command_triple", "result": "accepted"
            }),
        ]
    elif spec.injected_fault == "load-delay":
        required_events = [
            ("bootstrap_load_delay", {
                "result": "recovered", "timeout_started": "0"
            }),
        ]
    elif spec.injected_fault == "old-telemetry":
        required_events = [
            ("telemetry_replay", {}),
            ("old_telemetry", {"result": "discarded", "clock_regressed": "0"}),
        ]
    elif spec.injected_fault == "death":
        required_events = [
            ("death_injected", {"slot": "0", "action_tick": "5"}),
        ]
    elif spec.injected_fault == "sustained-drop":
        required_events = [
            ("sustained_drop", {}),
            ("sustained_drop_ready", {}),
            ("fatal", {"fault": "timeout"}),
        ]
    elif spec.injected_fault == "future":
        required_events = [
            ("future_ready", {"result": "rejected"}),
            ("fatal", {"fault": "future-ready"}),
        ]
    elif spec.injected_fault == "conflict":
        required_events = [
            ("conflicting_command", {"result": "rejected"}),
            ("fatal", {"fault": "command-conflict"}),
        ]
    elif spec.injected_fault == "sigkill":
        required_events = [
            ("disconnect", {"reason": "liveness"}),
            ("fatal", {"fault": "disconnect"}),
        ]
    elif spec.injected_fault == "drain-sigkill":
        required_events = [
            ("intermission_injected", {"drain_hold_ms": "3000"}),
            ("epoch_drain_enter", {}),
            ("disconnect", {"reason": "liveness"}),
            ("fatal", {"fault": "disconnect"}),
        ]
    elif spec.injected_fault in ("fifth", "human", "mixed-cvar"):
        required_events = [("admission_reject", {"reason": "roster_or_capability"})]
    elif spec.injected_fault == "epoch-drain":
        required_events = [
            ("epoch_drain_enter", {}),
            ("intermission_injected", {"drain_hold_ms": "0"}),
            ("map_reset", {"map_epoch": "2"}),
        ]
    for name, wanted in required_events:
        if not has_event(name, **wanted):
            mismatches[f"required_event_{name}"] = (False, wanted or True)
    if spec.injected_fault in ("death", "same-life-hold"):
        mismatches.update(_validate_stock_lifecycle_events(spec, value))
    if spec.injected_fault == "load-delay":
        load_values = [
            int(event["load_ms"])
            for event in events
            if event.get("event") == "bootstrap_load_delay"
            and event.get("result") == "recovered"
            and event.get("load_ms", "").isdigit()
        ]
        if not load_values or max(load_values) <= timeout_ms:
            mismatches["bootstrap_load_delay_duration"] = (
                load_values, f"> {timeout_ms}",
            )
    if spec.injected_fault in ("epoch-drain", "drain-sigkill"):
        enter_indices = [
            index for index, event in enumerate(events)
            if event.get("event") == "epoch_drain_enter"
        ]
        if len(enter_indices) != 1:
            mismatches["epoch_drain_enter_count"] = (len(enter_indices), 1)
        else:
            enter = enter_indices[0]
            injected = [
                index for index, event in enumerate(events[:enter])
                if event.get("event") == "intermission_injected"
            ]
            held = [
                index for index, event in enumerate(events[:enter])
                if event.get("event") == "epoch_drain_exit_held"
                and event.get("terminals_complete") == "0"
            ]
            if not injected or not held or injected[-1] >= held[-1]:
                mismatches["epoch_drain_exit_gate"] = (
                    {"intermission": injected, "held": held, "enter": enter},
                    "intermission < held(terminals_complete=0) < enter",
                )
            start = enter + 1
            end = len(events)
            for index in range(start, len(events)):
                if events[index].get("event") in ("map_reset", "fatal"):
                    end = index
                    break
            drain_events = events[start:end]
            forbidden = [
                event for event in drain_events
                if event.get("event") in ("apply", "telemetry")
            ]
            if forbidden:
                mismatches["epoch_drain_ml_activity"] = (
                    forbidden, "zero apply/telemetry events",
                )
            entry_event = events[enter]
            injected_events = [
                (index, events[index]) for index in injected
            ]
            commit_window_start = (
                injected_events[0][0]
                if len(injected_events) == 1 else enter
            )
            commit_events = [
                (commit_window_start + offset, event)
                for offset, event in enumerate(
                    events[commit_window_start:end]
                )
                if event.get("event") == "action_commit"
            ]
            injected_action_tick = (
                injected_events[0][1].get("action_tick")
                if len(injected_events) == 1 else None
            )
            injected_server_frame = (
                injected_events[0][1].get("server_frame")
                if len(injected_events) == 1 else None
            )
            enter_server_frame = entry_event.get("server_frame")
            active_map_epoch = None
            if len(injected_events) == 1:
                preceding_resets = [
                    event for event in events[:injected_events[0][0] + 1]
                    if event.get("event") == "map_reset"
                ]
                if preceding_resets:
                    active_map_epoch = preceding_resets[-1].get("map_epoch")
            entry_commit_valid = (
                len(injected_events) == 1
                and len(commit_events) == 1
                and commit_events[0][0] == start
                and isinstance(injected_action_tick, str)
                and injected_action_tick.isdigit()
                and isinstance(injected_server_frame, str)
                and injected_server_frame.isdigit()
                and isinstance(enter_server_frame, str)
                and enter_server_frame.isdigit()
                and isinstance(active_map_epoch, str)
                and active_map_epoch.isdigit()
                and commit_events[0][1].get("action_tick")
                == injected_action_tick
                and commit_events[0][1].get("server_frame")
                == injected_server_frame
                == enter_server_frame
                and commit_events[0][1].get("map_epoch")
                == active_map_epoch
            )
            if not entry_commit_valid:
                mismatches["epoch_drain_entry_commit"] = (
                    {
                        "injected": injected_events,
                        "enter": (enter, entry_event),
                        "active_map_epoch": active_map_epoch,
                        "commits": commit_events,
                    },
                    "the sole commit from intermission injection through "
                    "the drain boundary, immediately after drain entry and "
                    "matching action tick, server frame, and active map epoch",
                )
            clock_events = [
                (start + offset, int(event["server_frame"]))
                for offset, event in enumerate(drain_events)
                if event.get("event") == "epoch_drain_clock"
                and event.get("server_frame", "").isdigit()
            ]
            clocks = [frame for _index, frame in clock_events]
            if len(clocks) < 2 or any(
                current <= previous
                for previous, current in zip(clocks, clocks[1:])
            ):
                mismatches["epoch_drain_clock_progress"] = (
                    clocks, "at least two strictly increasing frames",
                )
            if spec.injected_fault == "drain-sigkill":
                global_disconnects = [
                    event for event in events
                    if event.get("event") == "disconnect"
                    and event.get("reason") == "liveness"
                ]
                global_disconnect_fatals = [
                    event for event in events
                    if event.get("event") == "fatal"
                    and event.get("fault") == "disconnect"
                ]
                disconnects = [
                    (start + offset, event)
                    for offset, event in enumerate(drain_events)
                    if event.get("event") == "disconnect"
                    and event.get("reason") == "liveness"
                ]
                closing = events[end] if end < len(events) else None
                boundary_events = [
                    event for event in drain_events
                    if event.get("event") in (
                        "map_reset", "startup_clock_sync",
                        "bootstrap_ready", "bootstrap_commit",
                    )
                ]
                valid_disconnect = (
                    len(global_disconnects) == 1
                    and global_disconnects[0].get("slot") == "0"
                    and len(global_disconnect_fatals) == 1
                    and len(disconnects) == 1
                    and disconnects[0][1] == global_disconnects[0]
                    and disconnects[0][1].get("slot") == "0"
                    and disconnects[0][1].get("frame", "").isdigit()
                    and len(clock_events) >= 2
                    and all(
                        clock_index < disconnects[0][0]
                        for clock_index, _frame in clock_events
                    )
                    and int(disconnects[0][1]["frame"]) >= clocks[-1]
                    and isinstance(closing, Mapping)
                    and closing.get("event") == "fatal"
                    and closing.get("fault") == "disconnect"
                    and disconnects[0][0] < end
                    and not boundary_events
                )
                if not valid_disconnect:
                    mismatches["drain_disconnect_same_window"] = (
                        {
                            "clocks": clocks,
                            "disconnects": [
                                event for _index, event in disconnects
                            ],
                            "global_disconnects": global_disconnects,
                            "global_disconnect_fatals": (
                                global_disconnect_fatals
                            ),
                            "closing": closing,
                            "boundary_events": boundary_events,
                        },
                        "drain_enter < two increasing clocks < slot-0 "
                        "liveness disconnect < fatal, without map reset",
                    )
            if (
                spec.injected_fault == "epoch-drain"
                and (end >= len(events) or events[end].get("event") != "map_reset")
            ):
                mismatches["epoch_drain_enter_before_map_exit"] = (
                    events[end] if end < len(events) else None,
                    "map_reset after drain enter",
                )
    if mismatches:
        raise QualificationError(
            f"scenario {spec.name} evidence differs: {mismatches!r}"
        )
    return {**value, "evidence_sha256": digest}


def _publish_raw_evidence(
    output: Path, results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    evidence_dir = output.parent / f"{output.stem}.evidence"
    if evidence_dir.exists() or evidence_dir.is_symlink():
        raise QualificationError(f"scenario evidence directory exists: {evidence_dir}")
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{evidence_dir.name}.", suffix=".tmp", dir=output.parent
    ))
    records = []
    try:
        for index, result in enumerate(results):
            name = f"{index:02d}-{result['scenario']}.json"
            path = temporary / name
            data = _canonical_bytes(result) + b"\n"
            _atomic_write(path, data)
            records.append({
                "scenario": result["scenario"],
                "path": f"{evidence_dir.name}/{name}",
                "sha256": _sha256_bytes(data),
                "size": len(data),
                "raw_evidence_sha256": result["evidence_sha256"],
                "trajectory_sha256": result.get("trajectory_sha256"),
                "observed_outcome": result["observed_outcome"],
            })
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, evidence_dir)
        parent = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return records


def produce_execution_evidence(
    *,
    identity: QualificationIdentity,
    output: Path,
    seed: int,
    timeout_ms: int,
    executor: ScenarioExecutor,
    jobs: int = 4,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Run scenarios and publish runtime-independent execution evidence."""
    identity.validate()
    destination = Path(output).expanduser()
    if not destination.is_absolute():
        raise QualificationError("qualification output must be absolute")
    if destination.exists() or destination.is_symlink():
        raise QualificationError("qualification output must be a new regular path")
    if type(timeout_ms) is not int or not 1500 <= timeout_ms <= 30000:
        raise QualificationError("barrier timeout must be 1500..30000 ms")
    if type(jobs) is not int or not 1 <= jobs <= 8:
        raise QualificationError("qualification jobs must be within 1..8")
    if executor is None:
        raise QualificationError("qualification executor is required")
    if not test_mode and type(executor) is not FullNetworkExecutor:
        raise QualificationError(
            "production qualification rejects injected scenario executors"
        )

    plan = scenario_plan(seed)
    tables = {spec.seed: build_action_table(spec.seed) for spec in plan}
    results_by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(jobs, len(plan))) as pool:
        futures = {
            pool.submit(executor.run, spec, tables[spec.seed]): spec
            for spec in plan
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                raw = future.result()
            except Exception as error:
                raise QualificationError(
                    f"full-network scenario {spec.name} failed closed: "
                    f"{_bounded_failure_cause(error)}"
                ) from error
            results_by_name[spec.name] = _require_raw(
                spec, raw, tables[spec.seed], test_mode=test_mode,
                timeout_ms=timeout_ms,
            )
    results = [results_by_name[spec.name] for spec in plan]

    baseline_one = results_by_name["baseline-cold-1"]["trajectory_sha256"]
    baseline_two = results_by_name["baseline-cold-2"]["trajectory_sha256"]
    different = results_by_name["different-seed"]["trajectory_sha256"]
    if baseline_one != baseline_two:
        raise QualificationError("same-seed cold trajectories differ")
    if baseline_one == different:
        raise QualificationError("different-seed trajectory did not diverge")
    if tables[seed]["sha256"] == tables[seed + 1]["sha256"]:
        raise QualificationError("different-seed action tables did not diverge")

    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_records = _publish_raw_evidence(destination, results)
    payload = {
        "schema": EXECUTION_SCHEMA,
        "mode": MODE,
        "protocol_version": PROTOCOL_VERSION,
        "test_mode": test_mode,
        "full_network_executed": not test_mode,
        "core_only": False,
        "non_admissible_for_training": True,
        "client_count": CLIENT_COUNT,
        "frames_per_successful_scenario": FRAME_COUNT,
        "design_sha256": identity.design_sha256,
        "source_repositories": {
            name: dict(record)
            for name, record in sorted(identity.source_repositories.items())
        },
        "source_closure_sha256": identity.source_closure_sha256,
        "q2ded_sha256": identity.q2ded["sha256"],
        "q2ded_size": identity.q2ded["size"],
        "game_module_sha256": identity.game_module["sha256"],
        "game_module_size": identity.game_module["size"],
        "client_binary_sha256": identity.client_binary["sha256"],
        "client_binary_size": identity.client_binary["size"],
        "runtime_binaries": {
            "q2ded": dict(identity.q2ded),
            "game_module": dict(identity.game_module),
            "client_binary": dict(identity.client_binary),
        },
        "wire": {
            "client_wire_version": WIRE_VERSION,
            "barrier_version": BARRIER_VERSION,
            "barrier_capability": BARRIER_CAPABILITY,
            "telemetry_bytes": TELEMETRY_BYTES,
            "ack_bytes": ACK_BYTES,
        },
        "launch_cvars": {
            "sv_ml_frame_barrier": 1,
            "sv_ml_frame_barrier_clients": CLIENT_COUNT,
            "sv_ml_frame_barrier_timeout_ms": timeout_ms,
            "ml_async": 0,
            "ml_frame_barrier": 1,
            "ml_frame_barrier_version": BARRIER_VERSION,
            "ml_frame_barrier_capability": BARRIER_CAPABILITY,
        },
        "qualification_fault_runtime": {
            "sv_ml_frame_barrier_test_mode": 1,
            "startup_sealed": True,
            "scenario_faults_only": True,
            "fault_vocabulary": "v1",
            "fault_vocabulary_marker": TEST_FAULT_VOCABULARY,
        },
        "required_training_runtime": {
            "sv_ml_frame_barrier_test_mode": 0,
            "sv_ml_frame_barrier_test_fault": "",
            "sv_ml_frame_barrier_test_tick": 0,
            "fault_injection_allowed": False,
        },
        "roster": [f"qual-{index:02d}" for index in range(CLIENT_COUNT)],
        # Embed the canonical scripts so finalization can independently replay
        # every raw action/telemetry assertion without trusting this producer.
        "action_tables": {
            str(value): tables[value] for value in sorted(tables)
        },
        "cold_launches_same_seed": 2,
        "same_seed_trajectory_sha256": baseline_one,
        "different_seed_trajectory_sha256": different,
        "same_seed_byte_identical": True,
        "different_seed_diverged": True,
        "scenario_evidence": raw_records,
        "enabled_cvar": "ml_client_frame_barrier",
        "fault_injection_passed": not test_mode,
        "ack_timeout_rejection_passed": not test_mode,
        "unsupported_mode_rejected": not test_mode,
        "deterministic_client_id_slot_admission": not test_mode,
        "all_clients_registered_before_bootstrap": not test_mode,
        "action_free_bootstrap_frames": 1,
        "fresh_usercmd_per_client_frame": not test_mode,
        "modulo_generation_enforced": not test_mode,
        "usercmd_application_order": "client-id-then-slot",
        "reliable_hook_weapon_deferred_ordered": not test_mode,
        "automatic_promotion": False,
    }
    artifact = {
        **payload,
        "execution_evidence_sha256": _sha256_bytes(_canonical_bytes(payload)),
    }
    _atomic_write(destination, _canonical_bytes(artifact) + b"\n")
    return artifact


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise QualificationError(f"{label} must be an exact regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _validate_execution_evidence(
    document: Mapping[str, Any], source: Path
) -> dict[str, Any]:
    execution = dict(document)
    digest = execution.pop("execution_evidence_sha256", None)
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or not _valid_sha256(digest)
        or digest != _sha256_bytes(_canonical_bytes(execution))
    ):
        raise QualificationError("execution evidence seal differs")
    test_mode = execution.get("test_mode")
    if type(test_mode) is not bool:
        raise QualificationError("execution evidence test mode is invalid")
    if (
        execution.get("mode") != MODE
        or execution.get("protocol_version") != PROTOCOL_VERSION
        or execution.get("core_only") is not False
        or execution.get("non_admissible_for_training") is not True
        or execution.get("full_network_executed") is not (not test_mode)
        or execution.get("client_count") != CLIENT_COUNT
        or execution.get("frames_per_successful_scenario") != FRAME_COUNT
    ):
        raise QualificationError("execution evidence mode or topology differs")
    launch_cvars = execution.get("launch_cvars")
    if (
        not isinstance(launch_cvars, Mapping)
        or type(launch_cvars.get("sv_ml_frame_barrier_timeout_ms")) is not int
        or not 1500 <= launch_cvars["sv_ml_frame_barrier_timeout_ms"] <= 30000
    ):
        raise QualificationError("execution barrier timeout differs")
    expected_launch = {
        "sv_ml_frame_barrier": 1,
        "sv_ml_frame_barrier_clients": CLIENT_COUNT,
        "sv_ml_frame_barrier_timeout_ms": launch_cvars[
            "sv_ml_frame_barrier_timeout_ms"
        ],
        "ml_async": 0,
        "ml_frame_barrier": 1,
        "ml_frame_barrier_version": BARRIER_VERSION,
        "ml_frame_barrier_capability": BARRIER_CAPABILITY,
    }
    if dict(launch_cvars) != expected_launch:
        raise QualificationError("execution launch cvar closure differs")
    if execution.get("wire") != {
        "client_wire_version": WIRE_VERSION,
        "barrier_version": BARRIER_VERSION,
        "barrier_capability": BARRIER_CAPABILITY,
        "telemetry_bytes": TELEMETRY_BYTES,
        "ack_bytes": ACK_BYTES,
    }:
        raise QualificationError("execution wire ABI closure differs")
    if execution.get("qualification_fault_runtime") != {
        "sv_ml_frame_barrier_test_mode": 1,
        "startup_sealed": True,
        "scenario_faults_only": True,
        "fault_vocabulary": "v1",
        "fault_vocabulary_marker": TEST_FAULT_VOCABULARY,
    }:
        raise QualificationError("qualification fault-control closure differs")
    if execution.get("required_training_runtime") != {
        "sv_ml_frame_barrier_test_mode": 0,
        "sv_ml_frame_barrier_test_fault": "",
        "sv_ml_frame_barrier_test_tick": 0,
        "fault_injection_allowed": False,
    }:
        raise QualificationError("required training fault controls differ")

    binaries = execution.get("runtime_binaries")
    if not isinstance(binaries, Mapping):
        raise QualificationError("execution binary closure is absent")
    identity = QualificationIdentity(
        q2ded=binaries.get("q2ded", {}),
        game_module=binaries.get("game_module", {}),
        client_binary=binaries.get("client_binary", {}),
        design_sha256=execution.get("design_sha256", ""),
        source_repositories=execution.get("source_repositories", {}),
        source_closure_sha256=execution.get("source_closure_sha256", ""),
    )
    identity.validate()
    for prefix, record in (
        ("q2ded", identity.q2ded),
        ("game_module", identity.game_module),
        ("client_binary", identity.client_binary),
    ):
        if (
            execution.get(f"{prefix}_sha256") != record["sha256"]
            or execution.get(f"{prefix}_size") != record["size"]
        ):
            raise QualificationError(f"execution {prefix} identity differs")

    tables_value = execution.get("action_tables")
    if not isinstance(tables_value, Mapping):
        raise QualificationError("execution action tables are absent")
    records = execution.get("scenario_evidence")
    base_seed = None
    if isinstance(tables_value, Mapping):
        seeds = sorted(
            int(value) for value in tables_value
            if isinstance(value, str) and value.isdigit()
        )
        if len(seeds) == 2 and seeds[1] == seeds[0] + 1:
            base_seed = seeds[0]
    if base_seed is None:
        raise QualificationError("execution action-table seeds differ")
    tables = {}
    for seed in (base_seed, base_seed + 1):
        table = tables_value.get(str(seed))
        expected = build_action_table(seed)
        if table != expected:
            raise QualificationError(f"execution action table {seed} differs")
        tables[seed] = expected
    plan = scenario_plan(base_seed)
    if not isinstance(records, list) or len(records) != len(plan):
        raise QualificationError("execution scenario record cardinality differs")

    parent = Path(source).resolve().parent
    validated_results: dict[str, dict[str, Any]] = {}
    for index, (spec, record) in enumerate(zip(plan, records)):
        if not isinstance(record, Mapping) or record.get("scenario") != spec.name:
            raise QualificationError("execution scenario order differs")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise QualificationError("execution scenario path is unsafe")
        path = parent / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(parent)
        except (OSError, ValueError) as error:
            raise QualificationError("execution scenario escapes evidence root") from error
        if path.is_symlink() or not path.is_file():
            raise QualificationError("execution scenario is not an exact regular file")
        data = path.read_bytes()
        if (
            type(record.get("size")) is not int
            or record["size"] != len(data)
            or record.get("sha256") != _sha256_bytes(data)
        ):
            raise QualificationError(f"scenario {spec.name} file identity differs")
        raw = _load_json_file(path, f"scenario {spec.name}")
        try:
            validated = _require_raw(
                spec, raw, tables[spec.seed], test_mode=test_mode,
                timeout_ms=launch_cvars["sv_ml_frame_barrier_timeout_ms"],
            )
        except QualificationError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise QualificationError(
                f"scenario {spec.name} raw structure is invalid"
            ) from error
        if (
            record.get("raw_evidence_sha256") != validated["evidence_sha256"]
            or record.get("trajectory_sha256")
            != validated.get("trajectory_sha256")
            or record.get("observed_outcome") != spec.expected_outcome
        ):
            raise QualificationError(f"scenario {spec.name} root binding differs")
        validated_results[spec.name] = validated

    baseline = validated_results["baseline-cold-1"]["trajectory_sha256"]
    if (
        baseline != validated_results["baseline-cold-2"]["trajectory_sha256"]
        or baseline == validated_results["different-seed"]["trajectory_sha256"]
        or execution.get("same_seed_trajectory_sha256") != baseline
        or execution.get("different_seed_trajectory_sha256")
        != validated_results["different-seed"]["trajectory_sha256"]
        or execution.get("same_seed_byte_identical") is not True
        or execution.get("different_seed_diverged") is not True
    ):
        raise QualificationError("execution deterministic trajectory closure differs")
    return {**execution, "execution_evidence_sha256": digest}


def finalize_qualification(
    *, execution: Path, runtime_manifest: Path, output: Path
) -> dict[str, Any]:
    """Bind validated execution evidence to one exact sealed runtime manifest."""
    from harness.runtime_attestation import (
        load_runtime_manifest,
        verify_runtime_manifest,
    )

    execution_path = Path(execution).expanduser()
    destination = Path(output).expanduser()
    if not execution_path.is_absolute() or not destination.is_absolute():
        raise QualificationError("execution and qualification outputs must be absolute")
    if execution_path.resolve().parent != destination.resolve().parent:
        raise QualificationError("execution and qualification must share an evidence root")
    if destination.exists() or destination.is_symlink():
        raise QualificationError("qualification output must be a new regular path")
    execution_document = _validate_execution_evidence(
        _load_json_file(execution_path, "execution evidence"), execution_path
    )
    manifest_path = Path(runtime_manifest).expanduser()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise QualificationError("runtime manifest must be an exact regular file")
    manifest = load_runtime_manifest(manifest_path)
    verified = verify_runtime_manifest(manifest)
    if not verified.valid or not _valid_sha256(verified.digest):
        raise QualificationError(
            "runtime manifest is not sealed: " + "; ".join(verified.errors)
        )
    try:
        semantic = manifest["semantic"]
        runtime_config = semantic["runtime_config"]
        artifacts = semantic["artifacts"]
        q2ded = artifacts["q2ded"]
        game = artifacts["game_module"]
    except (KeyError, TypeError) as error:
        raise QualificationError("runtime manifest omits barrier closure") from error
    execution_digest = execution_document["execution_evidence_sha256"]
    if runtime_config.get(
        "network_barrier_execution_evidence_sha256"
    ) != execution_digest:
        raise QualificationError("runtime semantic execution binding differs")
    binaries = execution_document["runtime_binaries"]
    for label, manifest_record, execution_record in (
        ("q2ded", q2ded, binaries["q2ded"]),
        ("game module", game, binaries["game_module"]),
    ):
        if (
            manifest_record.get("sha256") != execution_record["sha256"]
            or manifest_record.get("size") != execution_record["size"]
        ):
            raise QualificationError(f"runtime manifest {label} identity differs")
    client = binaries["client_binary"]
    if (
        runtime_config.get("client_binary_sha256") != client["sha256"]
        or runtime_config.get("client_binary_size") != client["size"]
    ):
        raise QualificationError("runtime manifest client identity differs")

    runtime_closure_payload = {
        "runtime_manifest_sha256": verified.digest,
        "execution_evidence_sha256": execution_digest,
    }
    test_mode = execution_document["test_mode"]
    artifact = _seal({
        "schema": SCHEMA,
        "passed": not test_mode,
        "mode": MODE,
        "protocol_version": PROTOCOL_VERSION,
        "test_mode": test_mode,
        "non_admissible_for_training": True,
        "runtime_manifest_sha256": verified.digest,
        "execution_evidence_sha256": execution_digest,
        "runtime_closure_sha256": _sha256_bytes(
            _canonical_bytes(runtime_closure_payload)
        ),
        "execution_evidence": execution_document,
    })
    _atomic_write(destination, _canonical_bytes(artifact) + b"\n")
    return artifact


def _git_identity(repo: Path) -> dict[str, Any]:
    source = Path(repo).resolve()
    if not source.is_dir() or Path(repo).is_symlink():
        raise QualificationError(f"source repository is invalid: {repo}")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=source,
            text=True,
            stderr=subprocess.STDOUT,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise QualificationError(f"source repository cannot be attested: {repo}") from error
    if status:
        raise QualificationError(f"source repository is dirty: {repo}")
    return {"commit": commit, "tree": tree, "clean": True}


def _execution_identity(
    q2ded: Path,
    game: Path,
    client: Path,
    design: Path,
) -> QualificationIdentity:
    design_record = _file_record(design, "frame-barrier-design")
    q2ded_record = _file_record(q2ded, "q2ded")
    game_record = _file_record(game, "game.so")
    client_record = _file_record(client, "yquake2")
    repositories = {
        "bot": _git_identity(ROOT),
        "client": _git_identity(ROOT.parent / "q2-ml-client"),
        "game": _git_identity(ROOT.parent / "q2-lithium-3zb2"),
    }
    source_closure = _sha256_bytes(_canonical_bytes(repositories))
    return QualificationIdentity(
        q2ded=q2ded_record,
        game_module=game_record,
        client_binary=client_record,
        design_sha256=design_record["sha256"],
        source_repositories=repositories,
        source_closure_sha256=source_closure,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--q2ded", type=Path, required=True)
    run_parser.add_argument("--game-module", type=Path, required=True)
    run_parser.add_argument("--client-binary", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument(
        "--design", type=Path,
        default=ROOT / "docs/NETWORK-CLIENT-FRAME-BARRIER.md",
    )
    run_parser.add_argument("--execution-output", type=Path, required=True)
    run_parser.add_argument("--game", default="lithium")
    run_parser.add_argument("--map", dest="map_name", default="q2dm1")
    run_parser.add_argument("--epoch-map", dest="epoch_map_name", default="q2dm2")
    run_parser.add_argument("--seed", type=int, default=7142026)
    run_parser.add_argument("--timeout-ms", type=int, default=1500)
    run_parser.add_argument("--server-warmup-seconds", type=float, default=0.25)
    run_parser.add_argument("--jobs", type=int, default=4)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--execution", type=Path, required=True)
    finalize_parser.add_argument("--runtime-manifest", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            identity = _execution_identity(
                args.q2ded, args.game_module, args.client_binary, args.design
            )
            data_root = args.data_root.expanduser().resolve()
            if not data_root.is_dir() or data_root.is_symlink():
                raise QualificationError("data root must be a non-symlink directory")
            if not math.isfinite(args.server_warmup_seconds) or not (
                0.0 <= args.server_warmup_seconds <= 10.0
            ):
                raise QualificationError("server warmup must be within 0..10 seconds")
            config = ExecutorConfiguration(
                q2ded=args.q2ded.expanduser().resolve(),
                game_module=args.game_module.expanduser().resolve(),
                client_binary=args.client_binary.expanduser().resolve(),
                data_root=data_root,
                game=args.game,
                map_name=args.map_name,
                epoch_map_name=args.epoch_map_name,
                timeout_ms=args.timeout_ms,
                server_warmup_seconds=args.server_warmup_seconds,
            )
            produce_execution_evidence(
                identity=identity,
                output=args.execution_output.expanduser().resolve(),
                seed=args.seed,
                timeout_ms=args.timeout_ms,
                executor=FullNetworkExecutor(config),
                jobs=args.jobs,
                test_mode=False,
            )
            printed_output = args.execution_output
        else:
            finalize_qualification(
                execution=args.execution.expanduser().resolve(),
                runtime_manifest=args.runtime_manifest.expanduser().resolve(),
                output=args.output.expanduser().resolve(),
            )
            printed_output = args.output
    except QualificationError as error:
        parser.error(str(error))
    print(printed_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
