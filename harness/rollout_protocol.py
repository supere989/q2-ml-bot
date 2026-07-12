"""Versioned synchronous PPO rollout transport for trusted LAN workers.

The wire format is deterministic and pickle-free. A learner publishes one
policy version, accepts only batches produced by that exact artifact, waits for
a quorum, updates, then publishes the next version. HTTP is used only as a
small dependency-free framing layer; q2ded and lattice queries remain local to
each worker.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, Optional

import numpy as np

PROTOCOL_VERSION = 1
POLICY_MAGIC = b"Q2PL0001"
BATCH_MAGIC = b"Q2RB0001"
MAX_POLICY_BYTES = 128 * 1024 * 1024
MAX_BATCH_BYTES = 512 * 1024 * 1024
ALLOWED_DTYPES = {
    "|u1", "|i1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8"
}


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_envelope(magic: bytes, manifest: dict, payload: bytes) -> bytes:
    encoded = _canonical_json(manifest)
    return magic + struct.pack("!I", len(encoded)) + encoded + payload


def _decode_envelope(data: bytes, magic: bytes, maximum: int) -> tuple[dict, bytes]:
    if len(data) > maximum:
        raise ValueError(f"message exceeds {maximum} bytes")
    if len(data) < 12 or data[:8] != magic:
        raise ValueError("invalid message header")
    manifest_size = struct.unpack("!I", data[8:12])[0]
    end = 12 + manifest_size
    if end > len(data):
        raise ValueError("truncated message manifest")
    try:
        manifest = json.loads(data[12:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid message manifest") from error
    return manifest, data[end:]


@dataclass(frozen=True)
class PolicyArtifact:
    version: int
    payload: bytes
    sha256: str
    config_hash: str = ""

    @classmethod
    def create(cls, version: int, payload: bytes, config_hash: str = ""):
        if int(version) < 0:
            raise ValueError("policy version must be nonnegative")
        if len(payload) > MAX_POLICY_BYTES:
            raise ValueError("policy artifact is too large")
        return cls(int(version), bytes(payload), _sha256(payload), str(config_hash))

    def encode(self) -> bytes:
        return _encode_envelope(POLICY_MAGIC, {
            "protocol_version": PROTOCOL_VERSION,
            "policy_version": self.version,
            "policy_sha256": self.sha256,
            "config_hash": self.config_hash,
            "payload_bytes": len(self.payload),
        }, self.payload)

    @classmethod
    def decode(cls, data: bytes):
        manifest, payload = _decode_envelope(data, POLICY_MAGIC, MAX_POLICY_BYTES + 65536)
        if int(manifest.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise ValueError("unsupported policy protocol version")
        artifact = cls.create(
            int(manifest["policy_version"]), payload, manifest.get("config_hash", "")
        )
        if artifact.sha256 != manifest.get("policy_sha256"):
            raise ValueError("policy artifact checksum mismatch")
        if len(payload) != int(manifest.get("payload_bytes", -1)):
            raise ValueError("policy artifact length mismatch")
        return artifact


@dataclass
class RolloutBatch:
    metadata: dict
    arrays: Dict[str, np.ndarray]

    REQUIRED_METADATA = {
        "worker_id", "sequence", "policy_version", "policy_sha256",
        "config_hash", "seed", "game_seed", "rollout_index", "determinism_key",
    }

    def __post_init__(self):
        missing = self.REQUIRED_METADATA - set(self.metadata)
        if missing:
            raise ValueError(f"missing rollout metadata: {sorted(missing)}")
        normalized = {}
        for name, array in self.arrays.items():
            if not name or not name.replace("_", "").isalnum():
                raise ValueError(f"invalid array name: {name!r}")
            value = np.ascontiguousarray(array)
            dtype = value.dtype.str
            if dtype not in ALLOWED_DTYPES:
                raise ValueError(f"unsupported rollout dtype: {dtype}")
            normalized[name] = value
        if not normalized:
            raise ValueError("rollout batch must contain arrays")
        self.arrays = normalized

    def rollout_hash(self) -> str:
        digest = hashlib.sha256()
        stable = {
            key: self.metadata[key]
            for key in (
                "policy_version", "policy_sha256", "config_hash", "seed",
                "game_seed", "rollout_index", "determinism_key",
            )
        }
        digest.update(_canonical_json(stable))
        for name in sorted(self.arrays):
            array = self.arrays[name]
            digest.update(_canonical_json({
                "name": name, "dtype": array.dtype.str, "shape": list(array.shape)
            }))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def encode(self) -> bytes:
        payload_parts = []
        schemas = []
        offset = 0
        for name in sorted(self.arrays):
            array = self.arrays[name]
            raw = array.tobytes(order="C")
            schemas.append({
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            })
            payload_parts.append(raw)
            offset += len(raw)
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "metadata": self.metadata,
            "rollout_hash": self.rollout_hash(),
            "arrays": schemas,
            "payload_bytes": offset,
        }
        encoded = _encode_envelope(BATCH_MAGIC, manifest, b"".join(payload_parts))
        if len(encoded) > MAX_BATCH_BYTES:
            raise ValueError("rollout batch is too large")
        return encoded

    @classmethod
    def decode(cls, data: bytes):
        manifest, payload = _decode_envelope(data, BATCH_MAGIC, MAX_BATCH_BYTES)
        if int(manifest.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise ValueError("unsupported rollout protocol version")
        if len(payload) != int(manifest.get("payload_bytes", -1)):
            raise ValueError("rollout payload length mismatch")
        arrays = {}
        occupied = 0
        for schema in manifest.get("arrays", []):
            name = str(schema["name"])
            dtype = np.dtype(schema["dtype"])
            if dtype.str not in ALLOWED_DTYPES:
                raise ValueError(f"unsupported rollout dtype: {dtype.str}")
            shape = tuple(int(value) for value in schema["shape"])
            if any(value < 0 for value in shape):
                raise ValueError("negative rollout array dimension")
            offset = int(schema["offset"])
            nbytes = int(schema["nbytes"])
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if offset != occupied or nbytes != expected or offset + nbytes > len(payload):
                raise ValueError("invalid rollout array layout")
            arrays[name] = np.frombuffer(
                payload[offset:offset + nbytes], dtype=dtype
            ).reshape(shape).copy()
            occupied += nbytes
        if occupied != len(payload):
            raise ValueError("unclaimed rollout payload bytes")
        batch = cls(dict(manifest["metadata"]), arrays)
        if batch.rollout_hash() != manifest.get("rollout_hash"):
            raise ValueError("rollout hash mismatch")
        return batch

    def validate_ppo_schema(self) -> None:
        required = {
            "obs", "actions", "rewards", "dones", "values", "log_probs",
            "h_states", "c_states", "last_obs", "last_h", "last_c",
        }
        missing = required - set(self.arrays)
        if missing:
            raise ValueError(f"missing PPO arrays: {sorted(missing)}")
        obs = self.arrays["obs"]
        if obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[1] < 1:
            raise ValueError("obs must have shape (steps, envs, obs_dim)")
        steps, envs, obs_dim = obs.shape
        expected = {
            "actions": (steps, envs, 8),
            "rewards": (steps, envs),
            "dones": (steps, envs),
            "values": (steps, envs),
            "log_probs": (steps, envs),
            "h_states": (steps, envs, 256),
            "c_states": (steps, envs, 256),
            "last_obs": (envs, obs_dim),
            "last_h": (envs, 256),
            "last_c": (envs, 256),
        }
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        float_arrays = required - {"dones"}
        if any(self.arrays[name].dtype != np.dtype("<f4") for name in float_arrays):
            raise ValueError("PPO floating arrays must be float32")
        if self.arrays["dones"].dtype != np.dtype("uint8"):
            raise ValueError("PPO dones must be uint8")
        if int(self.metadata.get("n_envs", -1)) != envs:
            raise ValueError("PPO n_envs metadata mismatch")
        if self.metadata.get("producer") != "q2":
            raise ValueError("PPO producer must be q2")
        if self.metadata.get("lattice_mode") not in {
            "fresh_worker_session", "persistent", "versioned_snapshot"
        }:
            raise ValueError("invalid PPO lattice_mode")
        if not isinstance(self.metadata.get("deterministic_actions"), bool):
            raise ValueError("PPO deterministic_actions must be boolean")


@dataclass(frozen=True)
class BatchDecision:
    status: str
    accepted: bool
    detail: str
    policy_version: int
    quorum_count: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class RolloutCoordinator:
    """Thread-safe synchronous generation barrier for one learner."""

    def __init__(self, quorum: int, schema: str = "any"):
        if int(quorum) < 1:
            raise ValueError("quorum must be positive")
        self.quorum = int(quorum)
        if schema not in {"any", "ppo"}:
            raise ValueError("schema must be 'any' or 'ppo'")
        self.schema = schema
        self._policy: Optional[PolicyArtifact] = None
        self._batches: Dict[int, Dict[str, RolloutBatch]] = {}
        self._batch_ids: set[str] = set()
        self._determinism: Dict[str, str] = {}
        self._sealed_versions: set[int] = set()
        self._condition = threading.Condition()

    def publish(self, artifact: PolicyArtifact) -> None:
        with self._condition:
            if self._policy is not None and artifact.version <= self._policy.version:
                raise ValueError("policy versions must increase monotonically")
            self._policy = artifact
            self._batches.pop(artifact.version, None)
            self._condition.notify_all()

    def policy(self) -> Optional[PolicyArtifact]:
        with self._condition:
            return self._policy

    def submit(self, encoded: bytes) -> BatchDecision:
        try:
            batch = RolloutBatch.decode(encoded)
        except (KeyError, TypeError, ValueError) as error:
            return BatchDecision("invalid", False, str(error), -1, 0)
        with self._condition:
            if self._policy is None:
                return BatchDecision("no_policy", False, "learner has no policy", -1, 0)
            current = self._policy
            version = int(batch.metadata["policy_version"])
            if version != current.version or batch.metadata["policy_sha256"] != current.sha256:
                label = "stale" if version < current.version else "wrong_policy"
                return BatchDecision(label, False, "batch policy is not current", current.version,
                                     len(self._batches.get(current.version, {})))
            if batch.metadata.get("config_hash", "") != current.config_hash:
                return BatchDecision("wrong_config", False, "config hash mismatch",
                                     current.version, len(self._batches.get(current.version, {})))
            if current.version in self._sealed_versions:
                return BatchDecision("generation_closed", False,
                                     "learner already consumed this generation",
                                     current.version, 0)
            if self.schema == "ppo":
                try:
                    batch.validate_ppo_schema()
                except ValueError as error:
                    return BatchDecision("invalid_schema", False, str(error),
                                         current.version,
                                         len(self._batches.get(current.version, {})))
            batch_id = _sha256(encoded)
            if batch_id in self._batch_ids:
                return BatchDecision("duplicate", False, "batch already received",
                                     current.version, len(self._batches.get(current.version, {})))
            key = str(batch.metadata["determinism_key"])
            rollout_hash = batch.rollout_hash()
            previous = self._determinism.get(key)
            if previous is not None and previous != rollout_hash:
                return BatchDecision("determinism_mismatch", False,
                                     "same determinism key produced different rollout",
                                     current.version, len(self._batches.get(current.version, {})))
            self._determinism[key] = rollout_hash
            worker_key = f"{batch.metadata['worker_id']}:{int(batch.metadata['sequence'])}"
            generation = self._batches.setdefault(current.version, {})
            if worker_key in generation:
                return BatchDecision("duplicate_sequence", False,
                                     "worker sequence already received", current.version,
                                     len(generation))
            generation[worker_key] = batch
            self._batch_ids.add(batch_id)
            self._condition.notify_all()
            return BatchDecision("accepted", True, "batch accepted", current.version,
                                 len(generation))

    def wait_for_quorum(self, version: int, timeout: float) -> list[RolloutBatch]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._batches.get(int(version), {})) < self.quorum:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)
            generation = self._batches[int(version)]
            keys = sorted(generation)[:self.quorum]
            result = [generation.pop(key) for key in keys]
            self._sealed_versions.add(int(version))
            generation.clear()
            return result

    def status(self) -> dict:
        with self._condition:
            version = self._policy.version if self._policy else -1
            return {
                "protocol_version": PROTOCOL_VERSION,
                "policy_version": version,
                "policy_sha256": self._policy.sha256 if self._policy else "",
                "config_hash": self._policy.config_hash if self._policy else "",
                "quorum": self.quorum,
                "schema": self.schema,
                "generation_closed": version in self._sealed_versions,
                "accepted_for_current": len(self._batches.get(version, {})),
            }


class CoordinatorServer:
    def __init__(self, coordinator: RolloutCoordinator, host: str, port: int,
                 token: str = ""):
        self.coordinator = coordinator
        self.token = token
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _authorized(self):
                return not owner.token or self.headers.get("Authorization") == f"Bearer {owner.token}"

            def _json(self, status: int, payload: dict):
                encoded = _canonical_json(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                if self.path == "/v1/status":
                    self._json(200, owner.coordinator.status())
                elif self.path == "/v1/policy":
                    artifact = owner.coordinator.policy()
                    if artifact is None:
                        self._json(404, {"error": "no policy"})
                        return
                    encoded = artifact.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                if self.path != "/v1/batches":
                    self._json(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_BATCH_BYTES:
                    self._json(413, {"error": "invalid content length"})
                    return
                decision = owner.coordinator.submit(self.rfile.read(length))
                status = 202 if decision.accepted else (200 if decision.status == "duplicate" else 409)
                if decision.status == "invalid":
                    status = 422
                self._json(status, decision.as_dict())

            def log_message(self, _format, *_args):
                return

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def address(self):
        return self.httpd.server_address

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class CoordinatorClient:
    def __init__(self, base_url: str, token: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)

    def _request(self, path: str, data: Optional[bytes] = None):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def fetch_policy(self) -> PolicyArtifact:
        status, payload = self._request("/v1/policy")
        if status != 200:
            raise RuntimeError(f"policy fetch failed ({status}): {payload.decode(errors='replace')}")
        return PolicyArtifact.decode(payload)

    def submit(self, batch: RolloutBatch) -> BatchDecision:
        status, payload = self._request("/v1/batches", batch.encode())
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid coordinator response ({status})") from error
        return BatchDecision(**decoded)

    def status(self) -> dict:
        status, payload = self._request("/v1/status")
        if status != 200:
            raise RuntimeError(f"status fetch failed ({status})")
        return json.loads(payload)


def deterministic_synthetic_batch(
    artifact: PolicyArtifact,
    worker_id: str,
    sequence: int,
    seed: int,
    game_seed: int,
    rollout_index: int,
    steps: int = 32,
    obs_dim: int = 64,
    action_dim: int = 8,
) -> RolloutBatch:
    """Generate a portable deterministic batch for protocol/LAN validation."""
    rng = np.random.default_rng(int(seed) ^ (int(game_seed) << 1) ^ int(rollout_index))
    arrays = {
        "obs": rng.standard_normal((steps, obs_dim), dtype=np.float32),
        "actions": rng.standard_normal((steps, action_dim), dtype=np.float32),
        "rewards": rng.standard_normal(steps, dtype=np.float32),
        "dones": (rng.random(steps) < 0.05).astype(np.uint8),
        "values": rng.standard_normal(steps, dtype=np.float32),
        "log_probs": rng.standard_normal(steps, dtype=np.float32),
    }
    determinism_key = (
        f"v{artifact.version}:{artifact.sha256}:cfg={artifact.config_hash}:"
        f"seed={seed}:game={game_seed}:rollout={rollout_index}"
    )
    return RolloutBatch({
        "worker_id": str(worker_id),
        "sequence": int(sequence),
        "policy_version": artifact.version,
        "policy_sha256": artifact.sha256,
        "config_hash": artifact.config_hash,
        "seed": int(seed),
        "game_seed": int(game_seed),
        "rollout_index": int(rollout_index),
        "determinism_key": determinism_key,
    }, arrays)


def merge_ppo_batches(batches: Iterable[RolloutBatch]) -> dict[str, np.ndarray]:
    """Merge one synchronous quorum along the environment axis for PPO."""
    batches = list(batches)
    if not batches:
        raise ValueError("cannot merge an empty rollout quorum")
    for batch in batches:
        batch.validate_ppo_schema()
    reference = batches[0]
    generation = (
        int(reference.metadata["policy_version"]),
        str(reference.metadata["policy_sha256"]),
        str(reference.metadata.get("config_hash", "")),
        str(reference.metadata["lattice_mode"]),
        bool(reference.metadata["deterministic_actions"]),
    )
    step_shape = reference.arrays["obs"].shape[::2]
    for batch in batches[1:]:
        candidate = (
            int(batch.metadata["policy_version"]),
            str(batch.metadata["policy_sha256"]),
            str(batch.metadata.get("config_hash", "")),
            str(batch.metadata["lattice_mode"]),
            bool(batch.metadata["deterministic_actions"]),
        )
        if candidate != generation:
            raise ValueError("cannot merge rollout batches from different policies")
        if batch.arrays["obs"].shape[::2] != step_shape:
            raise ValueError("rollout step/observation dimensions do not match")
    time_major = (
        "obs", "actions", "rewards", "dones", "values", "log_probs",
        "h_states", "c_states",
    )
    final_state = ("last_obs", "last_h", "last_c")
    merged = {
        name: np.concatenate([batch.arrays[name] for batch in batches], axis=1)
        for name in time_major
    }
    merged.update({
        name: np.concatenate([batch.arrays[name] for batch in batches], axis=0)
        for name in final_state
    })
    return merged
