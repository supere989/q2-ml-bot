"""Gym-like environment backed by a real Yamagi network client."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

from .client_protocol import ClientTelemetry, parse_client_telemetry
from .protocol import (
    Action,
    R_DAMAGE_DEALT,
    R_DAMAGE_TAKEN,
    R_DEATH,
    R_HOOK,
    R_ITEM,
    R_KILL,
    pack_action,
)
from .spatial import VoxelSpatialReward


def authoritative_reward(obs) -> float:
    """Reward channels emitted by game.so for this exact client_id."""
    return (
        obs.reward_damage_dealt * R_DAMAGE_DEALT
        - obs.reward_damage_taken * R_DAMAGE_TAKEN
        + obs.reward_kill * R_KILL
        - obs.reward_death * R_DEATH
        + obs.reward_item_pickup * R_ITEM
        + obs.reward_hook_traversal * R_HOOK
    )


class Q2NetworkClientEnv:
    """One ordinary player connection plus one private telemetry route.

    This backend is intentionally real-time and pipelined.  It never pauses
    q2ded waiting for inference; the newest policy action is held by the client
    until a newer authoritative telemetry frame is answered.
    """

    def __init__(
        self,
        *,
        server: str,
        telemetry_server: str,
        telemetry_token: str,
        client_binary: str,
        client_root: str,
        harness_host: str = "127.0.0.1",
        harness_port: int = 39000,
        qport: int | None = None,
        client_id: str | None = None,
        name: str | None = None,
        game: str = "lithium",
        timeout: float = 8.0,
        spatial_seed: int | None = None,
        debug: bool = False,
        extra_args: tuple[str, ...] = (),
    ):
        self.server = server
        self.telemetry_server = telemetry_server
        self.telemetry_token = telemetry_token
        self.client_binary = Path(client_binary).resolve()
        self.client_root = Path(client_root).resolve()
        self.harness_host = harness_host
        self.harness_port = int(harness_port)
        self.qport = int(qport if qport is not None else 10000 + harness_port % 50000)
        self.client_id = client_id or uuid.uuid4().hex
        self.name = name or f"ml-{self.client_id[:8]}"
        self.game = game
        self.timeout = timeout
        self.extra_args = tuple(extra_args)
        self.debug = bool(debug)
        self._spatial = VoxelSpatialReward.from_env(seed=spatial_seed)
        self._spatial_map: str | None = None
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._client_address: tuple[str, int] | None = None
        self._last: ClientTelemetry | None = None

    def start(self) -> ClientTelemetry:
        if self._process is not None:
            raise RuntimeError("network client is already running")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.harness_host, self.harness_port))
        sock.settimeout(0.25)
        self._socket = sock

        env = os.environ.copy()
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        env.setdefault("SDL_AUDIODRIVER", "dummy")
        args = [
            str(self.client_binary),
            "+set", "game", self.game,
            "+set", "name", self.name,
            "+set", "qport", str(self.qport),
            "+set", "ml_harness", "1",
            "+set", "ml_headless", "1",
            "+set", "ml_harness_debug", "1" if self.debug else "0",
            "+set", "ml_client_id", self.client_id,
            "+set", "ml_telemetry_server", self.telemetry_server,
            "+set", "ml_telemetry_token", self.telemetry_token,
            "+set", "ml_harness_addr", f"{self.harness_host}:{self.harness_port}",
            "+set", "s_enable", "0",
            "+set", "vid_renderer", "soft",
            "+connect", self.server,
            *self.extra_args,
        ]
        self._process = subprocess.Popen(
            args,
            cwd=self.client_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self._receive(after_sequence=None)

    def _receive(self, after_sequence: int | None) -> ClientTelemetry:
        if self._socket is None:
            raise RuntimeError("network client is not started")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                output = self._process.stdout.read() if self._process.stdout else ""
                raise RuntimeError(
                    f"network client exited with {self._process.returncode}:\n{output[-4000:]}"
                )
            try:
                data, address = self._socket.recvfrom(65535)
            except socket.timeout:
                continue
            telemetry = parse_client_telemetry(data)
            if telemetry is None or telemetry.client_id != self.client_id:
                continue
            if after_sequence is not None and telemetry.sequence <= after_sequence:
                continue
            self._client_address = address
            self._last = telemetry
            return telemetry
        raise TimeoutError(
            f"no telemetry for client_id={self.client_id} from {self.telemetry_server}"
        )

    def step(self, action: Action):
        if self._socket is None or self._client_address is None or self._last is None:
            raise RuntimeError("call start() before step()")
        previous_sequence = self._last.sequence
        self._socket.sendto(
            pack_action(action, self._last.server_frame), self._client_address
        )
        current = self._receive(after_sequence=previous_sequence)
        obs = current.observation
        return obs, authoritative_reward(obs), obs.is_terminal, False, {
            "client_id": current.client_id,
            "client_slot": current.client_slot,
            "sequence": current.sequence,
            "server_frame": current.server_frame,
            "network_native": True,
            "map": current.map_name,
        }

    def reset(self):
        """Start the real client and return the existing policy input shape."""
        current = self.start()
        self._spatial.reset(current.map_name, current.observation)
        self._spatial_map = current.map_name
        memory = self._spatial.memory_features(current.observation)
        return current.observation.to_vector(memory), {
            "client_id": current.client_id,
            "client_slot": current.client_slot,
            "server_frame": current.server_frame,
            "map": current.map_name,
            "network_native": True,
        }

    def step_vector(self, action: Action):
        """Gym-like policy step with the same lattice-extended observation."""
        obs, reward, terminal, truncated, info = self.step(action)
        map_name = str(info["map"])
        if map_name != self._spatial_map:
            self._spatial.reset(map_name, obs)
            self._spatial_map = map_name
        spatial_bonus, spatial_info = self._spatial.update(obs)
        memory = self._spatial.memory_features(obs)
        info.update(spatial_info)
        return (
            obs.to_vector(memory),
            reward + spatial_bonus,
            terminal,
            truncated,
            info,
        )

    def close(self):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            self._process = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._client_address = None
        self._last = None
        self._spatial_map = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
