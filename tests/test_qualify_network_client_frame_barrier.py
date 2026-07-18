from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import socket
import time
from types import SimpleNamespace

import pytest

from harness.runtime_attestation import MANIFEST_SCHEMA, semantic_digest
from tools import qualify_network_client_frame_barrier as qualification


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _record(name, byte):
    payload = byte * 8
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _identity():
    repositories = {
        name: {"commit": character * 40, "tree": character.upper() * 40,
               "clean": True}
        for name, character in (("bot", "a"), ("client", "b"), ("game", "c"))
    }
    # Git object IDs are lowercase hexadecimal.
    repositories = {
        name: {**value, "tree": value["tree"].lower()}
        for name, value in repositories.items()
    }
    source = hashlib.sha256(_canonical(repositories)).hexdigest()
    values = {
        "q2ded": _record("q2ded", b"q"),
        "game_module": _record("game.so", b"g"),
        "client_binary": _record("yquake2", b"c"),
        "design_sha256": "e" * 64,
        "source_repositories": repositories,
        "source_closure_sha256": source,
    }
    return qualification.QualificationIdentity(**values)


def _manifest(path, identity, execution_sha256):
    semantic = {
        "artifacts": {
            "q2ded": dict(identity.q2ded),
            "game_module": dict(identity.game_module),
        },
        "runtime_config": {
            "network_barrier_execution_evidence_sha256": execution_sha256,
            "client_binary_sha256": identity.client_binary["sha256"],
            "client_binary_size": identity.client_binary["size"],
        },
    }
    digest = semantic_digest(semantic)
    path.write_text(json.dumps({
        "schema": MANIFEST_SCHEMA,
        "semantic": semantic,
        "manifest_sha256": digest,
    }, sort_keys=True), encoding="utf-8")
    return digest


def _common_events(rows, *, epoch=False, fault=""):
    events = [
        {
            "event": "test_mode_sealed", "value": "1", "dedicated": "1",
            "roster": "exact", "controls_immutable": "1",
            "fault_vocabulary": "v1",
        },
        {"event": "startup_clock_sync", "settle_frames": "2"},
        {
        "event": "map_reset", "map": "q2dm1", "map_epoch": "1",
        "clients": "4", "timeout_ms": "1500", "wire": "8",
        "barrier": "1", "capability": "1", "test_controls_sealed": "1",
        "test_mode": "1", "test_fault": fault, "test_tick": "5",
        "test_map": "q2dm2", "dedicated": "1",
        },
    ]
    for slot in range(4):
        events.append({
            "event": "connected", "slot": str(slot)
        })
        events.append({
            "event": "bootstrap_ready", "slot": str(slot),
            "client_id": f"qual-{slot:02d}",
        })
    events.append({
        "event": "bootstrap_commit", "action_tick": "0",
        "server_frame": "1", "map_epoch": "1",
    })
    for slot in range(4):
        events.append({
            "event": "route_ack", "slot": str(slot),
            "client_id": f"qual-{slot:02d}", "server_frame": "1",
            "wire": "8", "barrier": "1", "capability": "1",
        })
    for row in rows:
        tick = str(row["clients"][0]["action_tick"])
        frame = str(row["clients"][0]["server_frame"])
        for slot in range(4):
            events.append({
                "event": "apply", "slot": str(slot),
                "action_tick": tick, "server_frame": frame, "msec": "100",
            })
        events.append({
            "event": "action_commit", "action_tick": tick,
            "server_frame": frame, "map_epoch": "1",
        })
        for slot in range(4):
            client = row["clients"][slot]
            observation = client["observation"]
            causal = client["causal"]
            events.append({
                "event": "telemetry", "slot": str(slot),
                "server_frame": frame, "applied_action_tick": tick,
                "map_epoch": "1", "sequence": str(client["sequence"]),
                "causal_echo_tick": tick,
                "causal_generation": str(int(tick) % 192 + 1),
                "client_id": client["client_id"],
                "client_life_epoch": str(causal["client_life_epoch"]),
                "terminal_reason": str(observation["terminal_reason"]),
                "alive": "1" if observation["self_state"][6] > 0 else "0",
                "action_tick": tick,
                "action_generation": str(int(tick) % 192 + 1),
            })
    events.append({
        "event": "deferred_control", "slot": "2", "action_tick": "12",
        "hook": "1", "weapon": "2", "order": "hook_then_weapon",
    })
    if epoch:
        events.extend((
            {"event": "intermission_injected", "action_tick": "5",
             "server_frame": "6", "target_map": "q2dm2",
             "drain_hold_ms": "0"},
            {"event": "epoch_drain_exit_held", "server_frame": "6",
             "terminals_complete": "0"},
            {"event": "epoch_drain_enter", "source": "intermission",
             "server_frame": "6"},
            {"event": "action_commit", "action_tick": "5",
             "server_frame": "6", "map_epoch": "1"},
            {"event": "epoch_drain_clock", "server_frame": "7"},
            {"event": "epoch_drain_clock", "server_frame": "8"},
            {"event": "map_reset", "map": "q2dm2", "map_epoch": "2",
             "clients": "4", "timeout_ms": "1500", "wire": "8",
             "barrier": "1", "capability": "1", "test_controls_sealed": "1",
             "test_mode": "1", "test_fault": fault, "test_tick": "5",
             "test_map": "q2dm2", "dedicated": "1"},
        ))
        for slot in range(4):
            events.append({
                "event": "bootstrap_ready", "slot": str(slot),
                "client_id": f"qual-{slot:02d}",
            })
            events.append({
                "event": "route_ack", "slot": str(slot),
                "client_id": f"qual-{slot:02d}", "server_frame": "1",
                "wire": "8", "barrier": "1", "capability": "1",
            })
        events.append({
            "event": "bootstrap_commit", "action_tick": "0",
            "server_frame": "1", "map_epoch": "2",
        })
    return events


def _trajectory(
    table, *, lifecycle_boundaries=0, lifecycle_ordinal=3, death=False,
    transport_gap_at=None,
):
    rows = []
    for row in table["rows"]:
        tick = row["ordinal"]
        action_tick = tick + int(
            transport_gap_at is not None and tick >= transport_gap_at
        )
        server_frame = action_tick + 1
        if lifecycle_boundaries and tick >= lifecycle_ordinal:
            server_frame += lifecycle_boundaries + 2
        clients = []
        for slot, action in enumerate(row["clients"]):
            generation = action_tick % 192 + 1
            clients.append({
                "client_id": f"qual-{slot:02d}",
                "client_slot": slot,
                "sequence": server_frame - 1,
                "server_frame": server_frame,
                "map_name": "q2dm1",
                "map_epoch": 1,
                "action_tick": action_tick,
                "applied_action_tick": action_tick,
                "echo_tick": action_tick,
                "action_generation": generation,
                "causal": {
                    "client_life_epoch": (
                        2 if death and slot == 0 and tick >= lifecycle_ordinal else 1
                    ),
                    "echo_tick": action_tick, "action_generation": generation,
                    "echo_valid": True, "facts_complete": True,
                    "transition_trainable": True,
                    "role_playing": True,
                    "role_public_pm_normal": True,
                },
                "expected_action": dict(action),
                "applied_action": {
                    name: action[name] for name in (
                        "move_forward", "move_right", "look_yaw", "look_pitch",
                        "vertical_intent", "fire", "hook", "weapon",
                    )
                },
                "observation": {
                    "slot": slot,
                    "tick": server_frame,
                    "terminal_reason": 0,
                    "self_debug": [
                        slot + 1,
                        slot,
                        qualification.ML_CONTROL_HUMAN,
                        (
                            qualification.ML_PLAYING_REQUIRED_FLAGS
                            | ((2 if death and slot == 0 and
                               tick >= lifecycle_ordinal else 1)
                               << qualification.ML_ENTITY_EPOCH_SHIFT)
                        ),
                    ],
                    "self_state": [
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        1.0,
                    ],
                },
            })
        rows.append({"ordinal": tick, "clients": clients})
    return rows


def _playing_role_preflight():
    return [
        {
            "client_id": f"qual-{slot:02d}",
            "client_slot": slot,
            "server_frame": 1,
            "map_epoch": 1,
            "client_life_epoch": 1,
            "edict_index": slot + 1,
            "debug_client_slot": slot,
            "control_source": qualification.ML_CONTROL_HUMAN,
            "debug_flags": (
                qualification.ML_PLAYING_REQUIRED_FLAGS
                | (1 << qualification.ML_ENTITY_EPOCH_SHIFT)
            ),
            "encoded_life_epoch": 1,
            "health": 100.0,
            "terminal_reason": 0,
            "causal_role_playing": True,
            "causal_role_public_pm_normal": True,
            "role": "playing",
        }
        for slot in range(4)
    ]


def _relay_fault_event(slot, *, timeout_ms=1500):
    held_ns = 1_000_000_000 + slot * 10_000
    scheduled_ns = held_ns + timeout_ms * 600_000
    released_ns = scheduled_ns + 1_000_000
    digest = hashlib.sha256(f"relay-{slot}".encode()).hexdigest()
    return {
        "event": "udp_telemetry_held_and_released",
        "client_id": f"qual-{slot:02d}",
        "client_slot": slot,
        "server_frame": 6,
        "applied_action_tick": 5,
        "sequence": 5,
        "datagram_bytes": qualification.TELEMETRY_BYTES,
        "held_datagram_sha256": digest,
        "released_datagram_sha256": digest,
        "relay_endpoint": f"127.0.0.1:{41000 + slot}",
        "upstream_endpoint": f"127.0.0.1:{42000 + slot}",
        "downstream_endpoint": f"127.0.0.1:{43000 + slot}",
        "held_monotonic_ns": held_ns,
        "scheduled_release_monotonic_ns": scheduled_ns,
        "released_monotonic_ns": released_ns,
        "hold_duration_ns": released_ns - held_ns,
        "queue_limit": 1,
        "queue_high_water": 1,
        "queue_overflows": 0,
        "held_packets": 1,
        "released_packets": 1,
        "additional_suppressed_packets": 0,
        "additional_suppressed_bytes": 0,
        "additional_suppressed_rolling_sha256": hashlib.sha256(b"").hexdigest(),
        "additional_first_sequence": None,
        "additional_last_sequence": None,
        "additional_first_server_frame": None,
        "additional_last_server_frame": None,
        "forwarded_to_harness_before_hold": 1,
        "forwarded_to_harness_during_hold": 0,
        "forwarded_to_harness": 2,
        "forwarded_to_client": 1,
        "relay_thread_alive": True,
    }


def _fault_process_liveness():
    return [{
        "role": "server" if index == 0 else "client",
        "identity": "q2ded" if index == 0 else f"qual-{index - 1:02d}",
        "pid": 1000 + index,
        "start_ticks": 2000 + index,
        "alive": True,
    } for index in range(5)]


class FakeExecutor:
    def __init__(self, *, tamper=None):
        self.tamper = tamper

    def run(self, spec, table):
        lifecycle_count = (
            6 if spec.injected_fault == "death"
            else 4 if spec.injected_fault == "same-life-hold" else 0
        )
        rows = (
            _trajectory(
                table, lifecycle_boundaries=lifecycle_count,
                death=spec.injected_fault == "death",
                transport_gap_at=(
                    5 if spec.injected_fault == "whole-telemetry-gap" else None
                ),
            )
            if spec.expected_outcome == "completed" else []
        )
        events = _common_events(
            rows,
            epoch=spec.injected_fault == "epoch-drain",
            fault=qualification._server_fault_for(spec),
        )
        if spec.injected_fault == "duplicate":
            events.extend((
                {"event": "duplicate_command", "result": "idempotent"},
                {"event": "duplicate_ready", "result": "idempotent"},
            ))
        elif spec.injected_fault == "stale":
            events.extend((
                {"event": "stale_command", "result": "stale"},
                {"event": "stale_ready", "result": "stale"},
            ))
        elif spec.injected_fault == "brief-drop":
            events.append({
                "event": "brief_drop_recovery", "source": "command_triple",
                "result": "accepted",
            })
        elif spec.injected_fault == "load-delay":
            events.append({
                "event": "bootstrap_load_delay", "result": "recovered",
                "timeout_started": "0", "load_ms": "1750",
            })
        elif spec.injected_fault == "old-telemetry":
            events.append({
                "event": "telemetry_replay", "slot": "0",
                "replay_frame": "4", "current_frame": "5",
            })
            events.append({
                "event": "old_telemetry", "result": "discarded",
                "clock_regressed": "0", "slot": "0",
                "replay_frame": "4", "accepted_frame": "5",
            })
        elif spec.injected_fault == "death":
            events.extend(({
                "event": "death_injected", "slot": "0",
                "action_tick": "5", "server_frame": "6",
            }, {
                "event": "telemetry", "slot": "0",
                "client_id": "qual-00", "client_life_epoch": "1",
                "terminal_reason": "1", "alive": "0",
            }, {
                "event": "respawn_action_restore", "slot": "0",
                "client_id": "qual-00", "prior_life_epoch": "1",
                "life_epoch": "2", "action_tick": "8",
                "action_generation": "9", "route_preserved": "1",
                "attribution": "exact", "alive": "1",
            }, {
                "event": "respawn_teleport_hold", "slot": "0",
                "client_life_epoch": "2", "active": "1",
                "pm_time": "14", "law": "stock",
            }, {
                "event": "ml_respawn_settling_action", "slot": "0",
                "client_id": "qual-00", "client_life_epoch": "2",
                "server_frame": "10", "action_tick": "9",
                "entry_latched": "1", "live_pmf_time_teleport": "0",
                "active": "1", "post_pmove_active": "0",
                "echo_valid": "1", "facts_complete": "1",
                "transition_trainable": "0", "actual_look_yaw": "6.000000",
                "actual_look_pitch": "0.000000",
            }))
        elif spec.injected_fault == "same-life-hold":
            events.append({
                "event": "same_life_hold_injected", "slot": "0",
                "action_tick": "5", "server_frame": "6",
                "client_life_epoch": "1", "active": "1",
                "pm_time": "14", "law": "stock",
            })
            events.append({
                "event": "ml_respawn_settling_action", "slot": "0",
                "client_id": "qual-00", "client_life_epoch": "1",
                "server_frame": "8", "action_tick": "7",
                "entry_latched": "1", "live_pmf_time_teleport": "0",
                "active": "1", "post_pmove_active": "0",
                "echo_valid": "1", "facts_complete": "1",
                "transition_trainable": "0", "actual_look_yaw": "6.000000",
                "actual_look_pitch": "0.000000",
            })
        elif spec.injected_fault == "sustained-drop":
            events.extend((
                {"event": "sustained_drop"},
                {"event": "sustained_drop_ready"},
                {"event": "fatal", "fault": "timeout"},
            ))
        elif spec.injected_fault == "future":
            events.extend((
                {"event": "future_ready", "result": "rejected"},
                {"event": "fatal", "fault": "future-ready"},
            ))
        elif spec.injected_fault == "conflict":
            events.extend((
                {"event": "conflicting_command", "result": "rejected"},
                {"event": "fatal", "fault": "command-conflict"},
            ))
        elif spec.injected_fault == "sigkill":
            events.extend((
                {"event": "disconnect", "slot": "0", "frame": "5",
                 "reason": "liveness"},
                {"event": "fatal", "fault": "disconnect"},
            ))
        elif spec.injected_fault == "drain-sigkill":
            events.extend((
                {"event": "intermission_injected", "action_tick": "5",
                 "server_frame": "6", "target_map": "q2dm2",
                 "drain_hold_ms": "3000"},
                {"event": "epoch_drain_exit_held", "server_frame": "6",
                 "terminals_complete": "0"},
                {"event": "epoch_drain_enter", "source": "intermission",
                 "server_frame": "6"},
                {"event": "action_commit", "action_tick": "5",
                 "server_frame": "6", "map_epoch": "1"},
                {"event": "epoch_drain_clock", "server_frame": "7"},
                {"event": "epoch_drain_clock", "server_frame": "8"},
                {"event": "disconnect", "slot": "0", "frame": "8",
                 "reason": "liveness"},
                {"event": "fatal", "fault": "disconnect"},
            ))
        elif spec.injected_fault in ("fifth", "human", "mixed-cvar"):
            events.append({
                "event": "admission_reject", "slot": "4",
                "reason": "roster_or_capability",
            })
        transport_fault_events = []
        if spec.injected_fault == "whole-telemetry-gap":
            transport_fault_events = [
                _relay_fault_event(slot)
                for slot in range(4)
            ]
        elif spec.injected_fault == "partial-telemetry-timeout":
            transport_fault_events = [_relay_fault_event(0)]
        death_boundaries = []
        action_hold_boundaries = []
        phases = (
            (
                "death_terminal", "corpse", "new_life_rebase",
                "new_life_teleport_settling",
                "new_life_teleport_settling",
                "new_life_actionable_prime",
            ) if spec.injected_fault == "death" else (
                "same_life_hold_settling",
                "same_life_hold_settling",
                "same_life_hold_settling",
                "same_life_actionable_prime",
            ) if spec.injected_fault == "same-life-hold" else ()
        )
        target_boundaries = (
            death_boundaries if spec.injected_fault == "death"
            else action_hold_boundaries
        )
        expected_row = table["rows"][2]
        for offset, phase in enumerate(phases):
            server_frame = 6 + offset
            action_tick = server_frame - 1
            target_boundaries.append({
                    "round_id": server_frame,
                    "trajectory_ordinal": 3,
                    "clients": [
                        {
                            "client_id": f"qual-{slot:02d}",
                            "client_slot": slot,
                            "sequence": server_frame - 1,
                            "server_frame": server_frame,
                            "map_name": "q2dm1",
                            "map_epoch": 1,
                            "action_tick": action_tick,
                            "echo_tick": action_tick,
                            "action_generation": action_tick % 192 + 1,
                            "client_life_epoch": (
                                2 if (
                                    spec.injected_fault == "death" and slot == 0
                                    and phase not in ("death_terminal", "corpse")
                                ) else 1
                            ),
                            "phase": phase if slot == 0 else "peer",
                            "observed_terminal_reason": int(
                                spec.injected_fault == "death" and slot == 0
                                and phase == "death_terminal"
                            ),
                            "causal_echo_valid": True,
                            "causal_facts_complete": True,
                            "causal_transition_trainable": (
                                False if slot == 0 and phase in (
                                    "new_life_rebase",
                                    "new_life_teleport_settling",
                                    "same_life_hold_settling",
                                ) else True
                            ),
                            "causal_role_playing": True,
                            "causal_role_public_pm_normal": not (
                                spec.injected_fault == "death" and slot == 0
                                and phase in ("death_terminal", "corpse")
                            ),
                            "expected_action": dict(
                                expected_row["clients"][slot]
                            ),
                            "applied_action": {
                                **dict(expected_row["clients"][slot]),
                                "look_pitch": (
                                    0.0 if slot == 0 and offset == len(phases) - 2
                                    else expected_row["clients"][slot]["look_pitch"]
                                ),
                            },
                            "trainable_transition": False,
                        }
                        for slot in range(4)
                    ],
                })
        raw = {
            "schema": qualification.RAW_SCHEMA,
            "scenario": spec.name,
            "seed": spec.seed,
            "cold_launch": spec.cold_launch,
            "expected_outcome": spec.expected_outcome,
            "observed_outcome": spec.expected_outcome,
            "injected_fault": spec.injected_fault,
            "client_ids": [f"qual-{index:02d}" for index in range(4)],
            "accepted_route_ack_slots": [0, 1, 2, 3],
            "routed_role_runtime": {
                "use_startobserver": 0,
                "use_startchasecam": 0,
            },
            "routed_role_preflight": _playing_role_preflight(),
            "action_free_bootstrap_frames": (
                2 if spec.injected_fault == "epoch-drain" else 1
            ),
            "accepted_synchronized_frames": 32 if rows else 4,
            "boundary_rounds": (
                len(death_boundaries) if spec.injected_fault == "death"
                else len(action_hold_boundaries)
                if spec.injected_fault == "same-life-hold"
                else 1 if spec.injected_fault in (
                    "epoch-drain", "whole-telemetry-gap"
                ) else 0
            ),
            "trajectory_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
            "trajectory_rows": rows,
            "death_lifecycle_boundary_rows": death_boundaries,
            "action_hold_boundary_rows": action_hold_boundaries,
            "death_lifecycle_resyncs": len(death_boundaries),
            "action_state_resyncs": len(action_hold_boundaries),
            "public_telemetry_audit": [
                {
                    "client_id": f"qual-{slot:02d}",
                    "datagrams_seen": 33,
                    "public_packets_decoded": 33,
                    "routed_packets_accepted": 33,
                    "malformed_packets_rejected": 0,
                    "foreign_client_packets_rejected": 0,
                    "stale_packets_rejected": 0,
                    "teacher_packets_detected": 0,
                }
                for slot in range(4)
            ],
            "transport_fault_events": transport_fault_events,
            "fault_process_liveness": (
                _fault_process_liveness()
                if spec.injected_fault in (
                    "whole-telemetry-gap", "partial-telemetry-timeout"
                ) else []
            ),
            "transport_metrics": {
                "rounds_dispatched": (
                    33 if spec.injected_fault in (
                        "whole-telemetry-gap", "epoch-drain"
                    ) else 32 + lifecycle_count if rows else 5
                ),
                "actions_dispatched": (
                    132 if spec.injected_fault in (
                        "whole-telemetry-gap", "epoch-drain"
                    ) else (32 + lifecycle_count) * 4 if rows else 20
                ),
                "transitions_accepted": 128 if rows else 16,
                "failed_rounds": int(
                    spec.injected_fault == "partial-telemetry-timeout"
                ),
                "echo_timeouts": int(
                    spec.injected_fault == "partial-telemetry-timeout"
                ),
                "telemetry_gap_resyncs": int(
                    spec.injected_fault == "whole-telemetry-gap"
                ),
                "realtime_catchup_resyncs": (
                    1 if spec.injected_fault in (
                        "whole-telemetry-gap", "epoch-drain"
                    ) else lifecycle_count
                ),
            },
            "structured_events": events,
            "structured_events_sha256": hashlib.sha256(_canonical(events)).hexdigest(),
            "fault_event_count": sum(
                event.get("event") == "fatal" for event in events
            ),
            "exception": (
                "AuthoritativeEchoError"
                if spec.injected_fault == "partial-telemetry-timeout" else None
            ),
            "epoch_drains": 1 if spec.injected_fault == "epoch-drain" else 0,
            "new_epoch_bootstrap_frames": (
                1 if spec.injected_fault == "epoch-drain" else 0
            ),
            "actions_dispatched_during_epoch_drain": 0,
            "server_exit_code": 0,
            "barrier_timeout_ms": 1500,
            "test_mode": True,
        }
        if self.tamper == spec.name:
            raw["accepted_synchronized_frames"] = 31
        return qualification._seal(raw)


def test_action_table_is_fixed_for_seed_and_diverges_for_new_seed():
    first = qualification.build_action_table(41)
    second = qualification.build_action_table(41)
    different = qualification.build_action_table(42)
    assert first == second
    assert first["sha256"] != different["sha256"]
    assert len(first["rows"]) == 32
    assert all(len(row["clients"]) == 4 for row in first["rows"])
    assert first["rows"][11]["clients"][2]["hook"] == 1
    assert first["rows"][11]["clients"][2]["weapon"] == 2


def test_matrix_requires_real_death_respawn_lifecycle_evidence():
    spec = next(
        item for item in qualification.scenario_plan(41)
        if item.injected_fault == "death"
    )
    table = qualification.build_action_table(spec.seed)
    raw = FakeExecutor().run(spec, table)
    payload = dict(raw)
    payload.pop("evidence_sha256")
    payload["death_lifecycle_boundary_rows"][0]["clients"][0]["phase"] = (
        "corpse"
    )
    with pytest.raises(qualification.QualificationError, match="death_phase"):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


def test_routed_role_preflight_rejects_observer_before_trajectory():
    spec = qualification.scenario_plan(41)[0]
    table = qualification.build_action_table(spec.seed)
    raw = FakeExecutor().run(spec, table)
    payload = dict(raw)
    payload.pop("evidence_sha256")
    record = payload["routed_role_preflight"][0]
    record["debug_flags"] |= (
        qualification.ML_ENTITY_OBSERVER
        | qualification.ML_ENTITY_NOCLIP
        | qualification.ML_ENTITY_PM_SPECTATOR
    )
    record["role"] = "invalid"
    with pytest.raises(qualification.QualificationError, match="routed_role_0"):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


def test_death_lifecycle_requires_one_final_new_life_priming_boundary():
    spec = next(
        item for item in qualification.scenario_plan(41)
        if item.injected_fault == "death"
    )
    table = qualification.build_action_table(spec.seed)
    raw = FakeExecutor().run(spec, table)
    payload = dict(raw)
    payload.pop("evidence_sha256")
    final = payload["death_lifecycle_boundary_rows"][-1]["clients"][0]
    final["client_life_epoch"] = 1
    with pytest.raises(
        qualification.QualificationError,
        match="lifecycle_boundary_identity",
    ):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


def _validated_fault_raw(fault):
    spec = next(
        item for item in qualification.scenario_plan(41)
        if item.injected_fault == fault
    )
    table = qualification.build_action_table(spec.seed)
    raw = FakeExecutor().run(spec, table)
    qualification._require_raw(
        spec, raw, table, test_mode=True, timeout_ms=1500
    )
    return spec, table, raw


def test_whole_gap_is_a_distinct_real_transport_scenario():
    names = {spec.name: spec.injected_fault for spec in qualification.scenario_plan(41)}
    assert names["brief-drop-recovery"] == "brief-drop"
    assert names["whole-batch-telemetry-gap-recovery"] == "whole-telemetry-gap"
    assert names["partial-client-telemetry-timeout"] == "partial-telemetry-timeout"
    assert names["one-client-sigkill"] == "sigkill"


def test_fault_relay_holds_the_real_datagram_and_releases_identical_bytes():
    downstream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    downstream.bind(("127.0.0.1", 0))
    downstream.settimeout(0.04)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    relay_port = probe.getsockname()[1]
    probe.close()
    upstream = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    upstream.bind(("127.0.0.1", 0))
    relay = qualification._TelemetryFaultRelay(
        relay_port=relay_port,
        downstream_port=downstream.getsockname()[1],
        client_id="qual-00",
        hold_seconds=0.12,
    )
    parsed = iter((
        SimpleNamespace(
            client_id="qual-00", client_slot=0, server_frame=1,
            applied_action_tick=0, sequence=1,
        ),
        SimpleNamespace(
            client_id="qual-00", client_slot=0, server_frame=2,
            applied_action_tick=1, sequence=2,
        ),
        SimpleNamespace(
            client_id="qual-00", client_slot=0, server_frame=2,
            applied_action_tick=1, sequence=3,
        ),
        SimpleNamespace(
            client_id="qual-00", client_slot=0, server_frame=2,
            applied_action_tick=1, sequence=4,
        ),
    ))
    relay._parse = lambda data: next(parsed)
    try:
        upstream.sendto(b"bootstrap", ("127.0.0.1", relay_port))
        assert downstream.recvfrom(65535)[0] == b"bootstrap"
        relay.arm()
        held = b"exact-held-telemetry-bytes"
        upstream.sendto(held, ("127.0.0.1", relay_port))
        upstream.sendto(b"suppressed-telemetry-one", ("127.0.0.1", relay_port))
        upstream.sendto(b"suppressed-telemetry-two", ("127.0.0.1", relay_port))
        with pytest.raises(socket.timeout):
            downstream.recvfrom(65535)
        relay.wait_released(0.5)
        downstream.settimeout(0.5)
        assert downstream.recvfrom(65535)[0] == held
        audit = relay.audit()
        digest = hashlib.sha256(held).hexdigest()
        assert audit["held_datagram_sha256"] == digest
        assert audit["released_datagram_sha256"] == digest
        assert audit["forwarded_to_harness_during_hold"] == 0
        assert audit["queue_high_water"] == 1
        assert audit["additional_suppressed_packets"] == 2
        assert audit["additional_suppressed_bytes"] == (
            len(b"suppressed-telemetry-one") + len(b"suppressed-telemetry-two")
        )
        assert audit["additional_first_sequence"] == 3
        assert audit["additional_last_sequence"] == 4
    finally:
        relay.close()
        upstream.close()
        downstream.close()


def test_whole_gap_rejects_forged_resync_metric():
    spec, table, raw = _validated_fault_raw("whole-telemetry-gap")
    payload = dict(raw)
    payload.pop("evidence_sha256")
    payload["transport_metrics"] = dict(payload["transport_metrics"])
    payload["transport_metrics"]["telemetry_gap_resyncs"] = 0
    with pytest.raises(qualification.QualificationError, match="whole_gap_metric"):
        qualification._require_raw(
            spec, qualification._seal(payload), table,
            test_mode=True, timeout_ms=1500,
        )


def test_partial_timeout_rejects_disconnect_relabel():
    spec, table, raw = _validated_fault_raw("partial-telemetry-timeout")
    payload = dict(raw)
    payload.pop("evidence_sha256")
    payload["transport_fault_events"] = [{
        "event": "disconnect", "client_id": "qual-00",
        "scope": "partial-client", "after_sequence": 5,
    }]
    with pytest.raises(qualification.QualificationError, match="relay_fault_event"):
        qualification._require_raw(
            spec, qualification._seal(payload), table,
            test_mode=True, timeout_ms=1500,
        )


def test_public_teacher_detection_cannot_be_relabelled_as_malformed():
    spec = next(
        item for item in qualification.scenario_plan(41)
        if item.name == "baseline-cold-1"
    )
    table = qualification.build_action_table(spec.seed)
    raw = FakeExecutor().run(spec, table)
    payload = dict(raw)
    payload.pop("evidence_sha256")
    payload["public_telemetry_audit"] = [
        dict(record) for record in payload["public_telemetry_audit"]
    ]
    payload["public_telemetry_audit"][0]["teacher_packets_detected"] = 1
    payload["public_telemetry_audit"][0]["datagrams_seen"] += 1
    with pytest.raises(
        qualification.QualificationError, match="public_privilege_boundary"
    ):
        qualification._require_raw(
            spec, qualification._seal(payload), table,
            test_mode=True, timeout_ms=1500,
        )


def test_same_life_hold_fixture_proves_settle_prime_and_startup_offset():
    _spec, _table, raw = _validated_fault_raw("same-life-hold")
    boundaries = raw["action_hold_boundary_rows"]
    assert {row["trajectory_ordinal"] for row in boundaries} == {3}
    assert [row["clients"][0]["phase"] for row in boundaries] == [
        "same_life_hold_settling",
        "same_life_hold_settling",
        "same_life_hold_settling",
        "same_life_actionable_prime",
    ]
    assert raw["trajectory_rows"][1]["clients"][0]["server_frame"] == 3
    assert raw["trajectory_rows"][2]["clients"][0]["server_frame"] == 10


@pytest.mark.parametrize("location", ("peer", "prime"))
@pytest.mark.parametrize(
    "field",
    (
        "move_forward",
        "move_right",
        "look_yaw",
        "look_pitch",
        "vertical_intent",
        "fire",
        "hook",
        "weapon",
    ),
)
def test_same_life_full_boundary_action_field_tamper_fails_closed(
    location, field
):
    spec, table, raw = _validated_fault_raw("same-life-hold")
    payload = json.loads(json.dumps(raw))
    payload.pop("evidence_sha256")
    if location == "peer":
        item = payload["action_hold_boundary_rows"][0]["clients"][1]
    else:
        item = payload["action_hold_boundary_rows"][-1]["clients"][0]
    item["applied_action"].pop(field)
    with pytest.raises(
        qualification.QualificationError,
        match="lifecycle_boundary_action",
    ):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


def test_drain_disconnect_fixture_proves_one_same_window_liveness_fault():
    _spec, _table, raw = _validated_fault_raw("drain-sigkill")
    events = raw["structured_events"]
    enter = next(
        index for index, event in enumerate(events)
        if event.get("event") == "epoch_drain_enter"
    )
    clocks = [
        index for index, event in enumerate(events)
        if event.get("event") == "epoch_drain_clock"
    ]
    disconnect = next(
        index for index, event in enumerate(events)
        if event.get("event") == "disconnect"
    )
    fatal = next(
        index for index, event in enumerate(events)
        if event.get("event") == "fatal"
    )
    assert enter < clocks[-2] < clocks[-1] < disconnect < fatal
    assert not any(
        event.get("event") == "map_reset"
        for event in events[enter + 1:disconnect]
    )
    injected = next(
        event for event in events
        if event.get("event") == "intermission_injected"
    )
    assert injected["drain_hold_ms"] == "3000"


@pytest.mark.parametrize(
    "fault,wanted",
    (("drain-sigkill", "3000"), ("epoch-drain", "0")),
)
def test_epoch_fault_fixture_binds_exact_sealed_drain_hold(fault, wanted):
    _spec, _table, raw = _validated_fault_raw(fault)
    injected = [
        event for event in raw["structured_events"]
        if event.get("event") == "intermission_injected"
    ]
    assert len(injected) == 1
    assert injected[0]["drain_hold_ms"] == wanted


@pytest.mark.parametrize(
    "fault,wanted",
    (("drain-sigkill", "3000"), ("epoch-drain", "0")),
)
@pytest.mark.parametrize("mutation", ("missing", "wrong"))
def test_epoch_fault_drain_hold_tamper_fails_closed(
    fault, wanted, mutation
):
    spec, table, raw = _validated_fault_raw(fault)
    payload = json.loads(json.dumps(raw))
    payload.pop("evidence_sha256")
    event = next(
        item for item in payload["structured_events"]
        if item.get("event") == "intermission_injected"
    )
    if mutation == "missing":
        event.pop("drain_hold_ms")
    else:
        event["drain_hold_ms"] = "0" if wanted == "3000" else "3000"
    payload["structured_events_sha256"] = hashlib.sha256(
        _canonical(payload["structured_events"])
    ).hexdigest()
    with pytest.raises(
        qualification.QualificationError,
        match="required_event_intermission_injected",
    ):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


@pytest.mark.parametrize("fault", ("drain-sigkill", "epoch-drain"))
def test_epoch_drain_fixture_has_one_exact_entry_frame_commit(fault):
    _spec, _table, raw = _validated_fault_raw(fault)
    events = raw["structured_events"]
    injected_index = next(
        index for index, event in enumerate(events)
        if event.get("event") == "intermission_injected"
    )
    injected = events[injected_index]
    enter = next(
        index for index, event in enumerate(events)
        if event.get("event") == "epoch_drain_enter"
    )
    boundary = next(
        index for index, event in enumerate(events[enter + 1:],
                                            start=enter + 1)
        if event.get("event") in ("map_reset", "fatal")
    )
    commits = [
        (index, event) for index, event in enumerate(
            events[injected_index:boundary], start=injected_index
        )
        if event.get("event") == "action_commit"
    ]
    assert len(commits) == 1
    assert commits[0][0] == enter + 1
    assert commits[0][1]["action_tick"] == injected["action_tick"]
    assert commits[0][1]["server_frame"] == injected["server_frame"]
    assert commits[0][1]["server_frame"] == events[enter]["server_frame"]
    assert commits[0][1]["map_epoch"] == "1"


@pytest.mark.parametrize("fault", ("drain-sigkill", "epoch-drain"))
@pytest.mark.parametrize(
    "mutation",
    (
        "missing", "extra_before_enter", "extra_later", "mismatched",
        "wrong_map_epoch",
    ),
)
def test_epoch_drain_entry_commit_tamper_fails_closed(fault, mutation):
    spec, table, raw = _validated_fault_raw(fault)
    payload = json.loads(json.dumps(raw))
    payload.pop("evidence_sha256")
    events = payload["structured_events"]
    enter = next(
        index for index, event in enumerate(events)
        if event.get("event") == "epoch_drain_enter"
    )
    commit = next(
        index for index, event in enumerate(events[enter + 1:],
                                            start=enter + 1)
        if event.get("event") == "action_commit"
    )
    if mutation == "missing":
        events.pop(commit)
    elif mutation == "extra_before_enter":
        events.insert(enter, {
            "event": "action_commit",
            "action_tick": "5",
            "server_frame": "6",
            "map_epoch": "1",
        })
    elif mutation == "extra_later":
        boundary = next(
            index for index, event in enumerate(events[enter + 1:],
                                                start=enter + 1)
            if event.get("event") in ("map_reset", "fatal")
        )
        events.insert(boundary, {
            "event": "action_commit",
            "action_tick": "6",
            "server_frame": "7",
            "map_epoch": "1",
        })
    elif mutation == "mismatched":
        events[commit]["server_frame"] = "7"
    else:
        events[commit]["map_epoch"] = "2"
    payload["structured_events_sha256"] = hashlib.sha256(
        _canonical(events)
    ).hexdigest()
    with pytest.raises(
        qualification.QualificationError,
        match="epoch_drain_entry_commit",
    ):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "map_reset_before_disconnect",
        "disconnect_outside_window",
        "insufficient_clocks",
        "wrong_reason",
        "duplicate_disconnect",
        "duplicate_fatal",
    ),
)
def test_drain_disconnect_window_tamper_fails_closed(mutation):
    spec, table, raw = _validated_fault_raw("drain-sigkill")
    payload = json.loads(json.dumps(raw))
    payload.pop("evidence_sha256")
    events = payload["structured_events"]
    disconnect_index = next(
        index for index, event in enumerate(events)
        if event.get("event") == "disconnect"
    )
    if mutation == "map_reset_before_disconnect":
        events.insert(disconnect_index, {
            "event": "map_reset",
            "map": "q2dm2",
            "map_epoch": "2",
        })
    elif mutation == "disconnect_outside_window":
        disconnect = events.pop(disconnect_index)
        events.append(disconnect)
    elif mutation == "insufficient_clocks":
        clock_index = next(
            index for index, event in enumerate(events)
            if event.get("event") == "epoch_drain_clock"
        )
        events.pop(clock_index)
    else:
        if mutation == "duplicate_disconnect":
            events.append(dict(events[disconnect_index]))
        elif mutation == "duplicate_fatal":
            fatal = next(
                event for event in events if event.get("event") == "fatal"
            )
            events.append(dict(fatal))
        else:
            events[disconnect_index]["reason"] = "engine-drop"
    payload["structured_events_sha256"] = hashlib.sha256(
        _canonical(events)
    ).hexdigest()
    with pytest.raises(
        qualification.QualificationError,
        match=(
            "drain_disconnect_same_window|epoch_drain_clock_progress|"
            "required_event_disconnect"
        ),
    ):
        qualification._require_raw(
            spec,
            qualification._seal(payload),
            table,
            test_mode=True,
            timeout_ms=1500,
        )


@pytest.mark.parametrize(
    "mutation",
    ("unknown", "duplicate_terminal", "reordered", "ordinal"),
)
def test_death_phase_and_trajectory_ordinal_tamper_fail_closed(mutation):
    spec, table, raw = _validated_fault_raw("death")
    payload = dict(raw)
    payload.pop("evidence_sha256")
    boundaries = payload["death_lifecycle_boundary_rows"]
    if mutation == "unknown":
        boundaries[1]["clients"][0]["phase"] = "unknown_lifecycle_phase"
    elif mutation == "duplicate_terminal":
        boundaries[1]["clients"][0]["phase"] = "death_terminal"
    elif mutation == "reordered":
        boundaries[2]["clients"][0]["phase"] = "new_life_teleport_settling"
        boundaries[3]["clients"][0]["phase"] = "new_life_rebase"
    else:
        boundaries[2]["trajectory_ordinal"] = 4
    with pytest.raises(qualification.QualificationError, match="lifecycle|death"):
        qualification._require_raw(
            spec, qualification._seal(payload), table,
            test_mode=True, timeout_ms=1500,
        )


@pytest.mark.parametrize(
    "mutation", ("missing", "duplicate", "wrong_trainable", "wrong_frame")
)
def test_clear_latch_event_tamper_fails_closed(mutation):
    spec, table, raw = _validated_fault_raw("same-life-hold")
    payload = dict(raw)
    payload.pop("evidence_sha256")
    events = payload["structured_events"]
    index = next(
        i for i, event in enumerate(events)
        if event.get("event") == "ml_respawn_settling_action"
    )
    if mutation == "missing":
        events.pop(index)
    elif mutation == "duplicate":
        events.append(dict(events[index]))
    elif mutation == "wrong_trainable":
        events[index]["transition_trainable"] = "1"
    else:
        events[index]["server_frame"] = "999"
    payload["structured_events_sha256"] = hashlib.sha256(
        _canonical(events)
    ).hexdigest()
    with pytest.raises(
        qualification.QualificationError, match="lifecycle_clear_latch"
    ):
        qualification._require_raw(
            spec, qualification._seal(payload), table,
            test_mode=True, timeout_ms=1500,
        )


def test_structured_event_parser_preserves_sealed_empty_fault_value():
    events = qualification._parse_events(
        "ML_FRAME_BARRIER_EVENT event=map_reset test_fault= test_mode=1\n"
    )
    assert events == [{
        "event": "map_reset", "test_fault": "", "test_mode": "1"
    }]


def test_test_only_executor_exercises_schema_but_cannot_pass(tmp_path):
    identity = _identity()
    execution_output = (tmp_path / "execution.json").resolve()
    execution = qualification.produce_execution_evidence(
        identity=identity,
        output=execution_output,
        seed=41,
        timeout_ms=1500,
        executor=FakeExecutor(),
        jobs=4,
        test_mode=True,
    )
    assert execution["schema"] == qualification.EXECUTION_SCHEMA
    assert execution["test_mode"] is True
    assert execution["full_network_executed"] is False
    assert len(execution["scenario_evidence"]) == 21
    manifest = tmp_path / "runtime-manifest.json"
    runtime_digest = _manifest(
        manifest, identity, execution["execution_evidence_sha256"]
    )
    output = (tmp_path / "qualification.json").resolve()
    artifact = qualification.finalize_qualification(
        execution=execution_output,
        runtime_manifest=manifest,
        output=output,
    )
    assert artifact["schema"] == qualification.SCHEMA
    assert artifact["passed"] is False
    assert artifact["test_mode"] is True
    assert artifact["non_admissible_for_training"] is True
    assert artifact["runtime_manifest_sha256"] == runtime_digest
    assert artifact["execution_evidence_sha256"] == execution[
        "execution_evidence_sha256"
    ]
    assert artifact["execution_evidence"]["qualification_fault_runtime"][
        "sv_ml_frame_barrier_test_mode"
    ] == 1
    assert artifact["execution_evidence"]["required_training_runtime"] == {
        "sv_ml_frame_barrier_test_mode": 0,
        "sv_ml_frame_barrier_test_fault": "",
        "sv_ml_frame_barrier_test_tick": 0,
        "fault_injection_allowed": False,
    }
    assert artifact["execution_evidence"]["same_seed_byte_identical"] is True
    assert artifact["execution_evidence"]["different_seed_diverged"] is True
    assert output.is_file()
    assert len(list((tmp_path / "execution.evidence").glob("*.json"))) == 21
    assert list(tmp_path.rglob("*.tmp")) == []
    decoded = json.loads(output.read_text(encoding="utf-8"))
    digest = decoded.pop("evidence_sha256")
    assert digest == hashlib.sha256(_canonical(decoded)).hexdigest()


def test_finalizer_revalidates_raw_execution_and_manifest_bindings(tmp_path):
    identity = _identity()
    execution_path = (tmp_path / "execution.json").resolve()
    execution = qualification.produce_execution_evidence(
        identity=identity, output=execution_path, seed=41, timeout_ms=1500,
        executor=FakeExecutor(), jobs=4, test_mode=True,
    )
    manifest = tmp_path / "runtime-manifest.json"
    _manifest(manifest, identity, execution["execution_evidence_sha256"])
    raw_path = next((tmp_path / "execution.evidence").glob("*.json"))
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b" ")
    with pytest.raises(qualification.QualificationError, match="file identity"):
        qualification.finalize_qualification(
            execution=execution_path, runtime_manifest=manifest,
            output=(tmp_path / "raw-tamper.json").resolve(),
        )
    raw_path.write_bytes(original)
    wrong_manifest = tmp_path / "wrong-runtime-manifest.json"
    _manifest(wrong_manifest, identity, "f" * 64)
    with pytest.raises(qualification.QualificationError, match="binding differs"):
        qualification.finalize_qualification(
            execution=execution_path, runtime_manifest=wrong_manifest,
            output=(tmp_path / "manifest-tamper.json").resolve(),
        )


def test_server_token_is_private_cfg_only_and_absent_from_argv(tmp_path):
    game = tmp_path / "lithium"
    game.mkdir()
    configuration = qualification.ExecutorConfiguration(
        q2ded=tmp_path / "q2ded", game_module=tmp_path / "game.so",
        client_binary=tmp_path / "yquake2", data_root=tmp_path,
        game="lithium", map_name="q2dm1", epoch_map_name="q2dm2",
        timeout_ms=1500, server_warmup_seconds=0.0,
    )
    executor = object.__new__(qualification.FullNetworkExecutor)
    executor.configuration = configuration
    token = "secret-qualification-token-1234567890"
    command = executor._server_command(
        configuration.q2ded, tmp_path, 28000, 28001, token,
        qualification.scenario_plan(41)[0],
    )
    assert token not in command
    cfg = game / "frame_barrier_qualification.cfg"
    cfg_text = cfg.read_text(encoding="utf-8")
    assert token in cfg_text
    assert 'set use_startobserver "0"' in cfg_text
    assert 'set use_startchasecam "0"' in cfg_text
    assert cfg.stat().st_mode & 0o077 == 0


def test_concurrent_port_leases_are_disjoint_and_released():
    workers = 8
    ports_per_worker = 12
    rendezvous = threading.Barrier(workers)
    with qualification._PortAllocator._lock:
        before = set(qualification._PortAllocator._leased)

    def lease_once():
        ports = qualification._PortAllocator.reserve(ports_per_worker)
        try:
            rendezvous.wait(timeout=5.0)
            return ports
        finally:
            qualification._PortAllocator.release(ports)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        leases = list(pool.map(lambda _index: lease_once(), range(workers)))
    assert all(len(set(ports)) == ports_per_worker for ports in leases)
    flattened = [port for ports in leases for port in ports]
    assert len(flattened) == len(set(flattened))
    with qualification._PortAllocator._lock:
        assert qualification._PortAllocator._leased == before


def test_full_network_executor_releases_port_lease_on_failure():
    executor = object.__new__(qualification.FullNetworkExecutor)
    with qualification._PortAllocator._lock:
        before = set(qualification._PortAllocator._leased)

    def fail(_spec, _action_table, ports):
        assert len(ports) == qualification.CLIENT_COUNT * 3 + 4
        raise RuntimeError("synthetic executor failure")

    executor._run_with_ports = fail
    with pytest.raises(RuntimeError, match="synthetic executor failure"):
        executor.run(qualification.scenario_plan(41)[0], {})
    with qualification._PortAllocator._lock:
        assert qualification._PortAllocator._leased == before


def test_absolute_script_from_outside_repo_imports_repo_harness(tmp_path):
    script = (
        Path(qualification.__file__).resolve().parent
        / "qualify_network_client_frame_barrier.py"
    )
    outside = tmp_path / "outside-repository"
    outside.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable, str(script), "finalize",
            "--execution", str(outside / "absent-execution.json"),
            "--runtime-manifest", str(outside / "absent-manifest.json"),
            "--output", str(outside / "qualification.json"),
        ],
        cwd=outside,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "execution evidence must be an exact regular file" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


def test_scenario_failure_reports_bounded_redacted_cause(tmp_path):
    class FailingExecutor:
        def run(self, spec, table):
            del spec, table
            raise ModuleNotFoundError(
                "No module named 'harness'; token="
                "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
            )

    with pytest.raises(qualification.QualificationError) as raised:
        qualification.produce_execution_evidence(
            identity=_identity(), output=(tmp_path / "failure.json").resolve(),
            seed=41, timeout_ms=1500, executor=FailingExecutor(), jobs=1,
            test_mode=True,
        )
    diagnostic = str(raised.value)
    assert "full-network scenario baseline-cold-1 failed closed" in diagnostic
    assert "ModuleNotFoundError: No module named 'harness'" in diagnostic
    assert "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG" not in diagnostic
    assert "token=<redacted>" in diagnostic
    assert len(diagnostic) < 400


def test_production_rejects_injected_executor_and_tampered_scenario(tmp_path):
    with pytest.raises(qualification.QualificationError, match="rejects injected"):
        qualification.produce_execution_evidence(
            identity=_identity(), output=(tmp_path / "prod.json").resolve(),
            seed=41, timeout_ms=1500, executor=FakeExecutor(), test_mode=False,
        )
    with pytest.raises(qualification.QualificationError, match="evidence differs"):
        qualification.produce_execution_evidence(
            identity=_identity(), output=(tmp_path / "bad.json").resolve(),
            seed=41, timeout_ms=1500,
            executor=FakeExecutor(tamper="baseline-cold-1"), test_mode=True,
        )
    assert not (tmp_path / "prod.json").exists()
    assert not (tmp_path / "bad.json").exists()


def test_production_control_preflight_fails_before_launch(tmp_path):
    paths = []
    for name in ("q2ded", "game.so", "yquake2"):
        path = tmp_path / name
        path.write_bytes(b"not a qualified executable")
        paths.append(path)
    config = qualification.ExecutorConfiguration(
        q2ded=paths[0], game_module=paths[1], client_binary=paths[2],
        data_root=tmp_path, game="lithium", map_name="q2dm1",
        epoch_map_name="q2dm2", timeout_ms=1500,
        server_warmup_seconds=0.0,
    )
    with pytest.raises(qualification.QualificationError, match="lacks production"):
        qualification.FullNetworkExecutor(config)


def test_preflight_requires_exact_known_fault_vocabulary_not_loose_strings(tmp_path):
    q2ded = tmp_path / "q2ded"
    q2ded.write_bytes(
        b"\0".join(
            marker
            for marker in qualification._REQUIRED_Q2DED_MARKERS
            if marker != qualification.TEST_FAULT_VOCABULARY.encode("ascii")
        )
        + b"\0load-delay\0old-telemetry\0"
    )
    client = tmp_path / "yquake2"
    client.write_bytes(b"\0".join(qualification._REQUIRED_CLIENT_MARKERS))
    game = tmp_path / "game.so"
    game.write_bytes(b"\0".join(qualification._REQUIRED_GAME_MARKERS))
    config = qualification.ExecutorConfiguration(
        q2ded=q2ded, game_module=game, client_binary=client,
        data_root=tmp_path, game="lithium", map_name="q2dm1",
        epoch_map_name="q2dm2", timeout_ms=1500,
        server_warmup_seconds=0.0,
    )
    with pytest.raises(qualification.QualificationError, match="TEST_FAULTS_V1"):
        qualification.FullNetworkExecutor(config)


def test_git_identity_requires_clean_commit_and_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    source = repo / "source.txt"
    source.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    identity = qualification._git_identity(repo)
    assert identity["clean"] is True
    assert len(identity["commit"]) == 40
    assert len(identity["tree"]) == 40
    source.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(qualification.QualificationError, match="dirty"):
        qualification._git_identity(repo)
