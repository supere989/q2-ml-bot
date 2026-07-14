"""Gym-like environment backed by a real Yamagi network client."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
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


@dataclass(frozen=True)
class ClientActionDispatch:
    """Identity and wire tick for one action sent to the embedded client."""

    client_id: str
    client_slot: int
    after_sequence: int
    action_tick: int
    map_name: str
    action: Action


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
        client_data_root: str | None = None,
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
        self.client_data_root = (
            Path(client_data_root).resolve()
            if client_data_root is not None
            else self.client_root / ".ml-clients"
        )
        self.harness_host = harness_host
        self.harness_port = int(harness_port)
        self.qport = int(qport if qport is not None else 10000 + harness_port % 50000)
        self.client_id = client_id or uuid.uuid4().hex
        try:
            encoded_client_id = self.client_id.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("client_id must contain only ASCII characters") from error
        if not encoded_client_id or len(encoded_client_id) >= 40:
            raise ValueError("client_id must contain between 1 and 39 ASCII bytes")
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.client_id) is None:
            raise ValueError(
                "client_id may contain only letters, digits, dot, underscore, and dash"
            )
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
        # WITH_XDG Yamagi clients otherwise race on generated-map downloads.
        # Give every routed client a private data/config/cache namespace.
        client_sandbox = self.client_data_root / self.client_id
        client_home = client_sandbox / "home"
        client_data_home = client_sandbox / "data"
        client_home.mkdir(parents=True, exist_ok=True)
        client_data_home.mkdir(parents=True, exist_ok=True)
        # Yamagi prefers an existing ~/.yq2 over XDG paths, so HOME must be
        # private as well; otherwise a legacy host install defeats isolation.
        env["HOME"] = str(client_home)
        env["XDG_DATA_HOME"] = str(client_data_home)
        args = [
            str(self.client_binary),
            "-datadir", str(self.client_root),
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
            # An unread PIPE eventually blocks a long-running client. Keep it
            # only for explicit diagnostics, where crash output is valuable.
            stdout=subprocess.PIPE if self.debug else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=self.debug,
        )
        return self._receive(after_sequence=None)

    def _receive(
        self,
        after_sequence: int | None,
        timeout: float | None = None,
    ) -> ClientTelemetry:
        if self._socket is None:
            raise RuntimeError("network client is not started")
        receive_timeout = self.timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + receive_timeout
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

    def dispatch_action(self, action: Action) -> ClientActionDispatch:
        """Send without waiting so a manager can dispatch a complete batch."""
        if self._socket is None or self._client_address is None or self._last is None:
            raise RuntimeError("call start() before step()")
        dispatch = ClientActionDispatch(
            client_id=self.client_id,
            client_slot=self._last.client_slot,
            after_sequence=self._last.sequence,
            action_tick=self._last.server_frame,
            map_name=self._last.map_name,
            action=action,
        )
        self._socket.sendto(
            pack_action(action, self._last.server_frame), self._client_address
        )
        return dispatch

    def receive_telemetry(
        self,
        *,
        after_sequence: int,
        timeout: float | None = None,
    ) -> ClientTelemetry:
        """Receive the next routed packet after an explicitly known sequence."""
        return self._receive(after_sequence=after_sequence, timeout=timeout)

    @staticmethod
    def _base_info(current: ClientTelemetry) -> dict:
        obs = current.observation
        return {
            "client_id": current.client_id,
            "client_slot": current.client_slot,
            "sequence": current.sequence,
            "server_frame": current.server_frame,
            "network_native": True,
            "map": current.map_name,
            "terminal_reason": int(obs.terminal_reason),
            "action_debug_tick": int(obs.action_debug[0]),
            "action_debug_accepted": int(obs.action_debug[1]),
            "action_debug_timeout_count": int(obs.action_debug[2]),
            "action_debug_weapon": int(obs.action_debug[3]),
            "action_debug_movement": [float(value) for value in obs.action_debug[4:8]],
        }

    def initial_result(
        self,
        current: ClientTelemetry,
        *,
        vector: bool = False,
    ):
        """Convert a packet returned by :meth:`start` into reset output."""
        info = self._base_info(current)
        if not vector:
            return current.observation, info
        self._spatial.reset(current.map_name, current.observation)
        self._spatial_map = current.map_name
        memory = self._spatial.memory_features(current.observation)
        return current.observation.to_vector(memory), info

    def transition_result(
        self,
        current: ClientTelemetry,
        *,
        vector: bool = False,
    ):
        """Convert an already echo-validated telemetry packet into a step result."""
        obs = current.observation
        reward = authoritative_reward(obs)
        info = self._base_info(current)
        info.update({
            "reward_base": float(reward),
            "damage_dealt": float(obs.reward_damage_dealt),
            "damage_taken": float(obs.reward_damage_taken),
            "kills": float(obs.reward_kill),
            "deaths": float(obs.reward_death),
            "items": float(obs.reward_item_pickup),
            "hook_reward": float(obs.reward_hook_traversal),
            "damage_prox": float(obs.reward_damage_taken_prox),
            "offense": float(obs.reward_offense),
            "survival": float(obs.reward_survival),
            "rune_held": float(obs.rune_flags.sum() > 0.0),
            "self_debug_flags": int(obs.self_debug[3]),
            "self_debug_control_source": int(obs.self_debug[2]),
            "spatial_bonus": 0.0,
        })
        if vector:
            if current.map_name != self._spatial_map:
                self._spatial.reset(current.map_name, obs)
                self._spatial_map = current.map_name
            spatial_bonus, spatial_info = self._spatial.update(obs)
            memory = self._spatial.memory_features(obs)
            info.update(spatial_info)
            info["spatial_bonus"] = float(spatial_bonus)
            obs_result = obs.to_vector(memory)
            reward += spatial_bonus
        else:
            obs_result = obs
        return obs_result, reward, obs.is_terminal, False, info

    def reset_episode_vector(self):
        """Reset only policy-side episode memory around the latest live frame."""
        if self._last is None:
            raise RuntimeError("call start() before resetting an episode")
        observation, _info = self.initial_result(self._last, vector=True)
        return observation

    def step(self, action: Action):
        dispatch = self.dispatch_action(action)
        current = self.receive_telemetry(after_sequence=dispatch.after_sequence)
        return self.transition_result(current)

    def reset(self):
        """Start the real client and return the existing policy input shape."""
        current = self.start()
        return self.initial_result(current, vector=True)

    def step_vector(self, action: Action):
        """Gym-like policy step with the same lattice-extended observation."""
        dispatch = self.dispatch_action(action)
        current = self.receive_telemetry(after_sequence=dispatch.after_sequence)
        return self.transition_result(current, vector=True)

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
