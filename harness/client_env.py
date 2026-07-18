"""Gym-like environment backed by a real Yamagi network client."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

import numpy as np

from .client_protocol import (
    ClientTelemetry,
    PublicTelemetryPrivilegeViolation,
    parse_client_telemetry,
)
from .multires_admission import (
    PRIVATE_CAUSAL_INFO_KEY,
    PRIVATE_SPATIAL_REWARD_INFO_KEY,
    SPATIAL_ATTESTATION_INFO_KEY,
    SpatialFeatureProvider,
    SpatialFeatureProviderFactory,
    SpatialProviderBinding,
    validate_spatial_provider_frame,
)
from .protocol import (
    Action,
    ActionDebugIndex,
    ML_FIRE_GATE_PROTECTED,
    ML_FIRE_GATE_TARGET,
    R_DAMAGE_DEALT,
    R_DAMAGE_TAKEN,
    R_DEATH,
    R_HOOK,
    R_ITEM,
    R_KILL,
    pack_action,
)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


TARGET_ACQUIRE_REWARD = _float_env("R_TARGET_ACQUIRE", 0.02)
TARGET_ACQUIRE_COOLDOWN_FRAMES = 20


@dataclass(frozen=True)
class ClientActionDispatch:
    """Identity and wire tick for one action sent to the embedded client."""

    client_id: str
    client_slot: int
    after_sequence: int
    action_tick: int
    map_name: str
    action: Action


@dataclass(frozen=True)
class ClientTelemetryDrain:
    """Newest valid telemetry found without waiting on the harness socket."""

    previous: ClientTelemetry
    latest: ClientTelemetry
    packet_count: int
    map_names: tuple[str, ...]

    @property
    def advanced(self) -> bool:
        return self.packet_count > 0

    @property
    def map_changed(self) -> bool:
        return any(name != self.previous.map_name for name in self.map_names)


@dataclass(frozen=True)
class PublicTelemetryAudit:
    """Exhaustive datagram accounting for one public harness socket."""

    datagrams_seen: int
    public_packets_decoded: int
    routed_packets_accepted: int
    malformed_packets_rejected: int
    foreign_client_packets_rejected: int
    stale_packets_rejected: int
    teacher_packets_detected: int

    def as_dict(self) -> dict[str, int]:
        return {
            "datagrams_seen": self.datagrams_seen,
            "public_packets_decoded": self.public_packets_decoded,
            "routed_packets_accepted": self.routed_packets_accepted,
            "malformed_packets_rejected": self.malformed_packets_rejected,
            "foreign_client_packets_rejected": self.foreign_client_packets_rejected,
            "stale_packets_rejected": self.stale_packets_rejected,
            "teacher_packets_detected": self.teacher_packets_detected,
        }


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
        harness_route_host: str | None = None,
        harness_route_port: int | None = None,
        qport: int | None = None,
        client_id: str | None = None,
        name: str | None = None,
        game: str = "lithium",
        timeout: float = 8.0,
        spatial_seed: int | None = None,
        multires_spatial_provider: SpatialFeatureProvider | None = None,
        multires_spatial_provider_factory: SpatialFeatureProviderFactory | None = None,
        expected_atlas_sha256: str | None = None,
        expected_runtime_manifest_sha256: str | None = None,
        debug: bool = False,
        deterministic_frame_barrier: bool = False,
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
        self.harness_route_host = (
            harness_host if harness_route_host is None else harness_route_host
        )
        self.harness_route_port = int(
            harness_port if harness_route_port is None else harness_route_port
        )
        if not (1 <= self.harness_port <= 65535):
            raise ValueError("harness_port must be between 1 and 65535")
        if not (1 <= self.harness_route_port <= 65535):
            raise ValueError("harness_route_port must be between 1 and 65535")
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
        self.deterministic_frame_barrier = bool(deterministic_frame_barrier)
        self._multires_spatial_provider = multires_spatial_provider
        self._multires_spatial_provider_factory = multires_spatial_provider_factory
        self._expected_atlas_sha256 = expected_atlas_sha256
        self._expected_runtime_manifest_sha256 = expected_runtime_manifest_sha256
        if (
            multires_spatial_provider is not None
            and multires_spatial_provider_factory is not None
        ):
            raise ValueError("select one multires provider or provider factory")
        if (
            multires_spatial_provider is not None
            or multires_spatial_provider_factory is not None
        ):
            if not expected_runtime_manifest_sha256:
                raise ValueError(
                    "multires spatial provider requires runtime-manifest digest"
                )
            if multires_spatial_provider is not None and not expected_atlas_sha256:
                raise ValueError("fixed multires provider requires Atlas digest")
            if multires_spatial_provider is not None and not callable(
                getattr(multires_spatial_provider, "sample", None)
            ):
                raise TypeError("multires spatial provider must implement sample()")
            if multires_spatial_provider_factory is not None and not callable(
                getattr(multires_spatial_provider_factory, "create", None)
            ):
                raise TypeError("multires provider factory must implement create()")
            self._spatial = None
        else:
            # Unvectorized transport/registration probes may use the network
            # client without a policy provider.  The retired dense voxel
            # reward is no longer imported or constructed as an implicit
            # fallback; any vector request below fails closed.
            if spatial_seed is not None:
                raise ValueError(
                    "spatial_seed belongs to the retired legacy reward path"
                )
            self._spatial = None
        self._spatial_map: str | None = None
        self._multires_map_name: str | None = None
        self._multires_map_epoch = 0
        self._last_policy_vector: np.ndarray | None = None
        self._boundary_projection_key: tuple[object, ...] | None = None
        self._boundary_projection_vector: np.ndarray | None = None
        self._boundary_projection_info: dict | None = None
        # Closing a network client is terminal for that object. A restart must
        # construct a fresh environment (and therefore a fresh provider
        # binding); this prevents a closed fixed provider from being sampled.
        self._closed = False
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._client_address: tuple[str, int] | None = None
        self._last: ClientTelemetry | None = None
        self._telemetry_datagrams_seen = 0
        self._public_packets_decoded = 0
        self._routed_packets_accepted = 0
        self._malformed_packets_rejected = 0
        self._foreign_client_packets_rejected = 0
        self._stale_packets_rejected = 0
        self._teacher_packets_detected = 0
        self._target_alignment_map: str | None = None
        self._target_was_aligned = False
        self._last_target_acquire_frame = -TARGET_ACQUIRE_COOLDOWN_FRAMES

    def start(self) -> ClientTelemetry:
        if self._closed:
            raise RuntimeError(
                "network client is closed; construct a fresh environment to restart"
            )
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
        # The conduit credential is inherited privately by the child and is
        # never placed in argv, a cvar, an archived config, or debug output.
        env["Q2_ML_CLIENT_TELEMETRY_TOKEN"] = self.telemetry_token
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
            *(
                ("+set", "ml_frame_barrier", "1")
                if self.deterministic_frame_barrier else ()
            ),
            "+set", "ml_telemetry_server", self.telemetry_server,
            "+set", "ml_harness_addr",
            f"{self.harness_route_host}:{self.harness_route_port}",
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

    @property
    def public_telemetry_audit(self) -> PublicTelemetryAudit:
        return PublicTelemetryAudit(
            datagrams_seen=self._telemetry_datagrams_seen,
            public_packets_decoded=self._public_packets_decoded,
            routed_packets_accepted=self._routed_packets_accepted,
            malformed_packets_rejected=self._malformed_packets_rejected,
            foreign_client_packets_rejected=self._foreign_client_packets_rejected,
            stale_packets_rejected=self._stale_packets_rejected,
            teacher_packets_detected=self._teacher_packets_detected,
        )

    def _parse_public_datagram(self, data: bytes) -> ClientTelemetry | None:
        self._telemetry_datagrams_seen += 1
        try:
            telemetry = parse_client_telemetry(data)
        except PublicTelemetryPrivilegeViolation:
            self._teacher_packets_detected += 1
            raise
        if telemetry is None:
            self._malformed_packets_rejected += 1
            return None
        self._public_packets_decoded += 1
        return telemetry

    def _route_public_telemetry(
        self, telemetry: ClientTelemetry, *, after_sequence: int | None
    ) -> bool:
        if telemetry.client_id != self.client_id:
            self._foreign_client_packets_rejected += 1
            return False
        if after_sequence is not None and telemetry.sequence <= after_sequence:
            self._stale_packets_rejected += 1
            return False
        self._routed_packets_accepted += 1
        return True

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
            telemetry = self._parse_public_datagram(data)
            if telemetry is None or not self._route_public_telemetry(
                telemetry, after_sequence=after_sequence
            ):
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

    def drain_latest_telemetry(self) -> ClientTelemetryDrain:
        """Drain queued routed packets without blocking and retain the newest."""
        if self._socket is None or self._last is None:
            raise RuntimeError("call start() before draining telemetry")
        if self._process is not None and self._process.poll() is not None:
            output = self._process.stdout.read() if self._process.stdout else ""
            raise RuntimeError(
                f"network client exited with {self._process.returncode}:\n"
                f"{output[-4000:]}"
            )
        previous = self._last
        latest = previous
        packet_count = 0
        map_names = []
        previous_timeout = self._socket.gettimeout()
        self._socket.setblocking(False)
        try:
            while True:
                try:
                    data, address = self._socket.recvfrom(65535)
                except (BlockingIOError, socket.timeout):
                    break
                telemetry = self._parse_public_datagram(data)
                if telemetry is None or not self._route_public_telemetry(
                    telemetry, after_sequence=latest.sequence
                ):
                    continue
                latest = telemetry
                packet_count += 1
                map_names.append(telemetry.map_name)
                self._client_address = address
        finally:
            self._socket.settimeout(previous_timeout)
        self._last = latest
        return ClientTelemetryDrain(
            previous=previous,
            latest=latest,
            packet_count=packet_count,
            map_names=tuple(map_names),
        )

    @staticmethod
    def _base_info(current: ClientTelemetry) -> dict:
        obs = current.observation
        velocity = np.asarray(obs.self_state[3:6], dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise RuntimeError("public own velocity is not finite xyz")
        pitch = float(obs.pitch)
        if not np.isfinite(pitch):
            raise RuntimeError("public true-view pitch is not finite")
        return {
            "client_id": current.client_id,
            "client_slot": current.client_slot,
            "sequence": current.sequence,
            "server_frame": current.server_frame,
            "map_epoch": current.map_epoch,
            "applied_action_tick": current.applied_action_tick,
            "barrier_version": current.barrier_version,
            "barrier_capabilities": current.barrier_capabilities,
            "network_native": True,
            "map": current.map_name,
            "terminal_reason": int(obs.terminal_reason),
            "action_debug_tick": int(obs.action_debug[ActionDebugIndex.TICK]),
            "action_debug_accepted": int(obs.action_debug[ActionDebugIndex.ACCEPTED]),
            "action_debug_timeout_count": int(
                obs.action_debug[ActionDebugIndex.TIMEOUT_COUNT]
            ),
            "action_debug_weapon": int(obs.action_debug[ActionDebugIndex.WEAPON]),
            "action_debug_movement": [
                float(obs.action_debug[index])
                for index in (
                    ActionDebugIndex.MOVE_FORWARD,
                    ActionDebugIndex.MOVE_RIGHT,
                    ActionDebugIndex.LOOK_YAW,
                    ActionDebugIndex.LOOK_PITCH,
                )
            ],
            "action_debug_vertical_intent": int(
                obs.action_debug[ActionDebugIndex.VERTICAL_INTENT]
            ),
            "action_debug_applied_upmove": int(
                obs.action_debug[ActionDebugIndex.APPLIED_UPMOVE]
            ),
            "action_debug_actual_ducked": bool(
                obs.action_debug[ActionDebugIndex.ACTUAL_DUCKED]
            ),
            "action_debug_water_vertical_mode": bool(
                obs.action_debug[ActionDebugIndex.WATER_VERTICAL_MODE]
            ),
            "action_debug_fire": bool(obs.action_debug[ActionDebugIndex.FIRE]),
            "action_debug_hook": int(obs.action_debug[ActionDebugIndex.HOOK]),
            "action_debug_flags": int(obs.action_debug[ActionDebugIndex.FLAGS]),
            "standing_blocked": bool(obs.standing_blocked),
            "damage_dealt": float(obs.reward_damage_dealt),
            # Both values are ordinary own-state facts already present in the
            # public ml_obs_t.  No causal/teacher field is consulted.
            "movement_speed": float(np.linalg.norm(velocity[:2])),
            "true_view_pitch_deg": pitch,
        }

    @staticmethod
    def _telemetry_projection_key(
        current: ClientTelemetry,
    ) -> tuple[object, ...]:
        """Identity of one immutable client telemetry datagram.

        Sequence is the primary conduit fence.  The remaining fields make a
        restart/map rollover unable to alias a cached boundary projection.
        """
        return (
            current.client_id,
            int(current.client_slot),
            int(current.sequence),
            current.map_name,
            int(current.server_frame),
            int(current.causal.client_life_epoch),
            int(current.causal.tick),
        )

    def _cache_boundary_projection(
        self,
        current: ClientTelemetry,
        vector: np.ndarray,
        info: dict,
    ) -> None:
        # Boundary consumers may see only factual/attestation information.
        # Private causal reward evidence belongs exclusively to the accepted
        # transition that produced it and must not be replayed by polling.
        boundary_info = self._base_info(current)
        attestation = info.get(SPATIAL_ATTESTATION_INFO_KEY)
        if attestation is not None:
            boundary_info[SPATIAL_ATTESTATION_INFO_KEY] = copy.deepcopy(
                attestation
            )
        self._boundary_projection_key = self._telemetry_projection_key(current)
        self._boundary_projection_vector = np.asarray(
            vector, dtype=np.float32
        ).copy()
        self._boundary_projection_info = boundary_info

    def _cached_boundary_projection(
        self, current: ClientTelemetry
    ) -> tuple[np.ndarray, dict] | None:
        if (
            self._boundary_projection_key
            != self._telemetry_projection_key(current)
            or self._boundary_projection_vector is None
            or self._boundary_projection_info is None
        ):
            return None
        return (
            self._boundary_projection_vector.copy(),
            copy.deepcopy(self._boundary_projection_info),
        )

    def _multires_result(
        self,
        current: ClientTelemetry,
        info: dict,
        *,
        episode_projection: bool,
    ):
        map_changed = bool(
            self._multires_map_name is not None
            and self._multires_map_name != current.map_name
        )
        next_epoch = self._multires_map_epoch + int(map_changed)
        if self._multires_spatial_provider_factory is not None and (
            self._multires_spatial_provider is None or map_changed
        ):
            # Construct and validate the replacement before closing the old
            # provider or mutating epoch/name state. A failed rotation can be
            # retried without leaving the environment half-rotated.
            binding = self._multires_spatial_provider_factory.create(
                current, map_epoch=next_epoch
            )
            if (
                not isinstance(binding, SpatialProviderBinding)
                or not callable(getattr(binding.provider, "sample", None))
                or not callable(getattr(binding.provider, "close", None))
            ):
                raise RuntimeError("multires provider factory returned no binding")
            old = self._multires_spatial_provider
            if old is not None:
                old.close()
            self._multires_spatial_provider = binding.provider
            self._expected_atlas_sha256 = binding.atlas_sha256
            self._multires_map_name = current.map_name
            self._multires_map_epoch = next_epoch
        elif self._multires_map_name is None:
            self._multires_map_name = current.map_name
        provider = self._multires_spatial_provider
        if provider is None:
            raise RuntimeError("multires result requested without a spatial provider")
        frame = provider.sample(
            current, episode_projection=episode_projection
        )
        spatial = validate_spatial_provider_frame(
            frame,
            current,
            expected_atlas_sha256=str(self._expected_atlas_sha256),
            expected_runtime_manifest_sha256=str(
                self._expected_runtime_manifest_sha256
            ),
            expected_map_epoch=self._multires_map_epoch,
            require_private_reward_evidence=not episode_projection,
        )
        info[SPATIAL_ATTESTATION_INFO_KEY] = {
            "schema": frame.schema,
            "feature_schema_sha256": frame.feature_schema_sha256,
            "atlas_sha256": frame.atlas_sha256,
            "runtime_manifest_sha256": frame.runtime_manifest_sha256,
            "map_epoch": frame.map_epoch,
        }
        if not episode_projection:
            info[PRIVATE_CAUSAL_INFO_KEY] = current.causal
            info[PRIVATE_SPATIAL_REWARD_INFO_KEY] = frame.private_reward_evidence
        return current.observation.to_vector(spatial)

    @property
    def uses_multires_spatial(self) -> bool:
        """Whether this client was configured for the sole production path."""
        return bool(
            self._multires_spatial_provider is not None
            or self._multires_spatial_provider_factory is not None
        )

    @staticmethod
    def _target_is_aligned(current: ClientTelemetry) -> bool:
        obs = current.observation
        flags = int(obs.action_debug[ActionDebugIndex.FLAGS])
        alive = float(obs.self_state[6]) > 0.0
        return bool(
            alive
            and not obs.is_terminal
            and flags & ML_FIRE_GATE_TARGET
            and not flags & ML_FIRE_GATE_PROTECTED
        )

    def _reset_target_alignment(self, current: ClientTelemetry) -> None:
        self._target_alignment_map = current.map_name
        self._target_was_aligned = self._target_is_aligned(current)
        self._last_target_acquire_frame = (
            int(current.server_frame) - TARGET_ACQUIRE_COOLDOWN_FRAMES
        )

    def _target_acquisition_bonus(
        self, current: ClientTelemetry
    ) -> tuple[float, bool, bool]:
        if current.map_name != self._target_alignment_map:
            self._target_alignment_map = current.map_name
            self._target_was_aligned = False
            self._last_target_acquire_frame = (
                int(current.server_frame) - TARGET_ACQUIRE_COOLDOWN_FRAMES
            )
        aligned = self._target_is_aligned(current)
        acquired = bool(
            aligned
            and not self._target_was_aligned
            and int(current.server_frame) - self._last_target_acquire_frame
            >= TARGET_ACQUIRE_COOLDOWN_FRAMES
        )
        bonus = TARGET_ACQUIRE_REWARD if acquired else 0.0
        if acquired:
            self._last_target_acquire_frame = int(current.server_frame)
        self._target_was_aligned = aligned
        if current.observation.is_terminal:
            self._target_was_aligned = False
        return float(bonus), acquired, aligned

    def initial_result(
        self,
        current: ClientTelemetry,
        *,
        vector: bool = False,
    ):
        """Convert a packet returned by :meth:`start` into reset output."""
        if self._closed:
            raise RuntimeError("network client is closed")
        if (
            self._multires_spatial_provider is not None
            or self._multires_spatial_provider_factory is not None
        ):
            if not vector:
                raise RuntimeError("multires client environments require vector=True")
            cached = self._cached_boundary_projection(current)
            if cached is not None:
                return cached
            self._reset_target_alignment(current)
            info = self._base_info(current)
            vector_result = self._multires_result(
                current, info, episode_projection=True
            )
            self._last_policy_vector = np.asarray(
                vector_result, dtype=np.float32
            ).copy()
            self._cache_boundary_projection(current, vector_result, info)
            return vector_result, info
        self._reset_target_alignment(current)
        info = self._base_info(current)
        if not vector:
            return current.observation, info
        raise RuntimeError(
            "vector policy input requires an attested multires provider; "
            "the legacy spatial fallback is retired"
        )

    def transition_result(
        self,
        current: ClientTelemetry,
        *,
        vector: bool = False,
    ):
        """Convert an already echo-validated telemetry packet into a step result."""
        if self._closed:
            raise RuntimeError("network client is closed")
        obs = current.observation
        if (
            self._multires_spatial_provider is not None
            or self._multires_spatial_provider_factory is not None
        ):
            if not vector:
                raise RuntimeError("multires client environments require vector=True")
            info = self._base_info(current)
            obs_result = self._multires_result(
                current, info, episode_projection=False
            )
            self._last_policy_vector = np.asarray(
                obs_result, dtype=np.float32
            ).copy()
            self._cache_boundary_projection(current, obs_result, info)
            # All reward is admitted later from the private causal conduit.
            # Network batching must never mix the legacy dense scalar here.
            return obs_result, 0.0, obs.is_terminal, False, info
        reward = authoritative_reward(obs)
        target_bonus, target_acquired, target_aligned = (
            self._target_acquisition_bonus(current)
        )
        reward += target_bonus
        info = self._base_info(current)
        info.update({
            "reward_base": float(reward),
            "target_alignment_bonus": target_bonus,
            "target_acquired": target_acquired,
            "target_aligned": target_aligned,
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
            raise RuntimeError(
                "vector policy input requires an attested multires provider; "
                "the legacy spatial fallback is retired"
            )
        else:
            obs_result = obs
        return obs_result, reward, obs.is_terminal, False, info

    def reset_episode_vector(self):
        """Reset only policy-side episode memory around the latest live frame."""
        if self._last is None:
            raise RuntimeError("call start() before resetting an episode")
        if self._last_policy_vector is None:
            raise RuntimeError("latest policy vector is unavailable")
        return self._last_policy_vector.copy()

    def drain_multires_runtime_snapshots(self) -> tuple[dict, ...]:
        """Drain public Atlas/Dyn telemetry from the active admitted provider."""
        provider = self._multires_spatial_provider
        drain = getattr(provider, "drain_runtime_snapshots", None)
        if provider is None or not callable(drain):
            raise RuntimeError("active multires provider lacks runtime telemetry")
        value = drain()
        if not isinstance(value, tuple) or any(
            not isinstance(item, dict) for item in value
        ):
            raise RuntimeError("multires provider runtime telemetry is malformed")
        return value

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
        if self._closed:
            return
        self._closed = True
        # Detach first. Even if provider shutdown reports an error, no later
        # call can observe or sample a stale closed provider reference.
        provider = self._multires_spatial_provider
        self._multires_spatial_provider = None
        process = self._process
        self._process = None
        sock = self._socket
        self._socket = None
        self._client_address = None
        self._last = None
        self._spatial_map = None
        self._multires_map_name = None
        self._multires_map_epoch = 0
        self._last_policy_vector = None
        self._boundary_projection_key = None
        self._boundary_projection_vector = None
        self._boundary_projection_info = None
        self._target_alignment_map = None
        self._target_was_aligned = False
        self._last_target_acquire_frame = -TARGET_ACQUIRE_COOLDOWN_FRAMES
        try:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        finally:
            try:
                if sock is not None:
                    sock.close()
            finally:
                try:
                    if provider is not None:
                        close = getattr(provider, "close", None)
                        if callable(close):
                            close()
                finally:
                    if self._multires_spatial_provider_factory is not None:
                        reset_session = getattr(
                            self._multires_spatial_provider_factory,
                            "reset_session",
                            None,
                        )
                        if callable(reset_session):
                            reset_session()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
