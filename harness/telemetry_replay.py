"""Record and replay network-client telemetry sessions.

``SocketRecordingEnv`` wraps a live :class:`Q2NetworkClientEnv` and logs
every ``start``/``dispatch_action``/``receive_telemetry``/
``drain_latest_telemetry`` call with its arguments and results (raw packet
bytes) to a JSONL file.  ``ReplayEnv`` serves the same per-env call sequence
back, re-parsing raw packets through the real wire decoder, so two collector
implementations can be driven over byte-identical sessions and their emitted
rounds compared for equivalence.

This is verification tooling: it changes no production semantics.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .client_env import ClientActionDispatch, ClientTelemetryDrain
from .client_protocol import parse_client_telemetry
from .protocol import Action


def action_fields(action: Action) -> dict[str, Any]:
    return {
        "move_forward": float(action.move_forward),
        "move_right": float(action.move_right),
        "look_yaw": float(action.look_yaw),
        "look_pitch": float(action.look_pitch),
        "jump": bool(action.jump),
        "fire": bool(action.fire),
        "hook": int(action.hook),
        "weapon": int(action.weapon),
    }


def action_from_fields(fields: dict[str, Any]) -> Action:
    return Action(
        move_forward=float(fields["move_forward"]),
        move_right=float(fields["move_right"]),
        look_yaw=float(fields["look_yaw"]),
        look_pitch=float(fields["look_pitch"]),
        jump=bool(fields["jump"]),
        fire=bool(fields["fire"]),
        hook=int(fields["hook"]),
        weapon=int(fields["weapon"]),
    )


class SocketRecordingEnv:
    """Proxy that records raw packets and dispatch metadata per call."""

    def __init__(self, env, client_index: int, handle):
        self._env = env
        self._index = int(client_index)
        self._handle = handle
        self._spatial = env._spatial
        self.client_id = env.client_id

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _log(self, event: dict) -> None:
        event["client"] = self._index
        self._handle.write(json.dumps(event) + "\n")

    def start(self):
        telemetry = self._env.start()
        self._log({
            "kind": "start",
            "sequence": int(telemetry.sequence),
            "raw": self._env._last_raw.hex(),
        })
        return telemetry

    def dispatch_action(self, action: Action) -> ClientActionDispatch:
        dispatch = self._env.dispatch_action(action)
        self._log({
            "kind": "dispatch",
            "action": action_fields(action),
            "after_sequence": int(dispatch.after_sequence),
            "action_tick": int(dispatch.action_tick),
            "map_name": dispatch.map_name,
            "client_slot": int(dispatch.client_slot),
        })
        return dispatch

    def receive_telemetry(self, *, after_sequence: int,
                          timeout: float | None = None):
        started = time.monotonic()
        try:
            telemetry = self._env.receive_telemetry(
                after_sequence=after_sequence, timeout=timeout
            )
        except TimeoutError:
            self._log({
                "kind": "receive",
                "after_sequence": int(after_sequence),
                "timeout": True,
                "wait_ms": (time.monotonic() - started) * 1000.0,
            })
            raise
        self._log({
            "kind": "receive",
            "after_sequence": int(after_sequence),
            "timeout": False,
            "sequence": int(telemetry.sequence),
            "raw": self._env._last_raw.hex(),
            "wait_ms": (time.monotonic() - started) * 1000.0,
        })
        return telemetry

    def drain_latest_telemetry(self) -> ClientTelemetryDrain:
        drain = self._env.drain_latest_telemetry()
        self._log({
            "kind": "drain",
            "packet_count": int(drain.packet_count),
            "raws": [raw.hex() for raw in self._env._drain_raws],
        })
        return drain


class ReplayEnv:
    """Serve one recorded per-env call sequence back to a collector.

    Network-facing methods validate the call shape against the recording and
    return objects rebuilt through the real wire parser.  Everything else
    (spatial reward, feature extraction) runs the real production code on a
    detached :class:`Q2NetworkClientEnv` instance, so replay exercises the
    same post-processing path as live collection.
    """

    def __init__(self, env, events: list[dict]):
        self._env = env  # detached Q2NetworkClientEnv (never started)
        self._events = list(events)
        self._cursor = 0
        self._spatial = env._spatial
        self.client_id = env.client_id
        self._last = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _next(self, kind: str) -> dict:
        if self._cursor >= len(self._events):
            raise AssertionError(
                f"replay exhausted for {self.client_id}: expected {kind}"
            )
        event = self._events[self._cursor]
        self._cursor += 1
        if event["kind"] != kind:
            raise AssertionError(
                f"replay divergence for {self.client_id} at event "
                f"{self._cursor - 1}: expected {kind}, got {event['kind']}"
            )
        return event

    @staticmethod
    def _telemetry_from_raw(raw_hex: str):
        telemetry = parse_client_telemetry(bytes.fromhex(raw_hex))
        if telemetry is None:
            raise AssertionError("recorded packet failed to parse on replay")
        return telemetry

    def start(self):
        event = self._next("start")
        telemetry = self._telemetry_from_raw(event["raw"])
        self._last = telemetry
        return telemetry

    def initial_result(self, current, *, vector=False):
        return self._env.initial_result(current, vector=vector)

    def transition_result(self, current, *, vector=False):
        return self._env.transition_result(current, vector=vector)

    def reset_episode_vector(self):
        return self._env.reset_episode_vector()

    def dispatch_action(self, action: Action) -> ClientActionDispatch:
        event = self._next("dispatch")
        recorded = event["action"]
        if action_fields(action) != recorded:
            raise AssertionError(
                f"replay divergence for {self.client_id}: dispatched action "
                f"{action_fields(action)} != recorded {recorded}"
            )
        return ClientActionDispatch(
            client_id=self.client_id,
            client_slot=int(event["client_slot"]),
            after_sequence=int(event["after_sequence"]),
            action_tick=int(event["action_tick"]),
            map_name=event["map_name"],
            action=action,
        )

    def receive_telemetry(self, *, after_sequence: int,
                          timeout: float | None = None):
        event = self._next("receive")
        if int(event["after_sequence"]) != int(after_sequence):
            raise AssertionError(
                f"replay divergence for {self.client_id}: receive after "
                f"sequence {after_sequence} != recorded "
                f"{event['after_sequence']}"
            )
        if event["timeout"]:
            raise TimeoutError("recorded receive timeout")
        telemetry = self._telemetry_from_raw(event["raw"])
        self._last = telemetry
        return telemetry

    def drain_latest_telemetry(self) -> ClientTelemetryDrain:
        event = self._next("drain")
        previous = self._last
        latest = previous
        map_names = []
        raws = event.get("raws") or []
        if raws:
            decoded = [self._telemetry_from_raw(raw) for raw in raws]
            latest = decoded[-1]
            map_names = [telemetry.map_name for telemetry in decoded]
        self._last = latest
        return ClientTelemetryDrain(
            previous=previous,
            latest=latest,
            packet_count=int(event["packet_count"]),
            map_names=tuple(map_names),
        )

    def close(self):
        pass


def read_recording(path: str) -> list[list[dict]]:
    by_client: dict[int, list[dict]] = {}
    with open(path) as handle:
        for line in handle:
            event = json.loads(line)
            by_client.setdefault(int(event["client"]), []).append(event)
    return [by_client[index] for index in sorted(by_client)]
