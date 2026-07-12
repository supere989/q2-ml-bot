"""Failure/recovery primitives for distributed rollout workers.

This module deliberately stays separate from :mod:`harness.rollout_protocol`.
It defines the control-plane contracts needed before the synchronous rollout
prototype can become a durable service, without changing the proven batch wire
format or the live learner yet.

The important ownership boundary is:

* an assignment belongs to a stable learner lane, not to a worker host;
* a lease fences one attempt to execute that assignment;
* a replacement receives the same assignment/seeds with a newer lease epoch;
* lattice payloads become authoritative only after the learner adopts them
  into its checksum- and parent-chained store.

Lease IDs are fencing tokens, not authentication secrets.  The transport must
continue to authenticate workers independently.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence


ASSIGNMENT_SCHEMA_VERSION = 1
LATTICE_ARTIFACT_SCHEMA_VERSION = 1
LATTICE_STATE_VERSION = 1
LATTICE_MAGIC = b"Q2LA0001"
MAX_LATTICE_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_LATTICE_JSON_BYTES = 256 * 1024 * 1024
MAX_IDENTITY_LENGTH = 256
MAX_SEED = 2**31 - 1

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSIGNMENT_ID = re.compile(r"^q2a1-[0-9a-f]{64}$")


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_int(name: str, value, minimum: int = 0, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bounds}")
    return value


def _require_identity(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, trimmed string")
    if len(value) > MAX_IDENTITY_LENGTH or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} contains invalid characters or is too long")
    return value


def _require_sha256(name: str, value: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return value
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_time(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_digest(domain: str, values: Mapping[str, object]) -> str:
    return _sha256(_canonical_json({"domain": domain, "values": dict(values)}))


def deterministic_lane_id(learner_id: str, config_hash: str, lane_index: int) -> str:
    """Return a stable lattice/rollout lane ID independent of worker hosts."""
    learner_id = _require_identity("learner_id", learner_id)
    config_hash = _require_sha256("config_hash", config_hash)
    lane_index = _require_int("lane_index", lane_index)
    digest = _stable_digest(
        "q2-rollout-lane-v1",
        {
            "learner_id": learner_id,
            "config_hash": config_hash,
            "lane_index": lane_index,
        },
    )
    return f"q2l1-{digest}"


def deterministic_seed(base_seed: int, domain: str, **identity) -> int:
    """Derive a portable signed-31-bit seed from assignment identity."""
    base_seed = _require_int("base_seed", base_seed, 0, MAX_SEED)
    domain = _require_identity("seed domain", domain)
    digest = _stable_digest(
        f"q2-seed-v1:{domain}", {"base_seed": base_seed, **identity}
    )
    return (base_seed + int(digest[:16], 16)) % (MAX_SEED + 1)


@dataclass(frozen=True)
class RolloutAssignment:
    """Immutable work identity which survives worker replacement."""

    learner_id: str
    config_hash: str
    policy_version: int
    policy_sha256: str
    lane_index: int
    rollout_index: int
    seed: int
    game_seed: int
    steps: int
    n_envs: int
    map_name: str
    lattice_artifact_sha256: str = ""
    lane_id: str = field(init=False)
    assignment_id: str = field(init=False)

    def __post_init__(self) -> None:
        learner_id = _require_identity("learner_id", self.learner_id)
        config_hash = _require_sha256("config_hash", self.config_hash)
        policy_version = _require_int("policy_version", self.policy_version)
        policy_sha256 = _require_sha256("policy_sha256", self.policy_sha256)
        lane_index = _require_int("lane_index", self.lane_index)
        rollout_index = _require_int("rollout_index", self.rollout_index)
        seed = _require_int("seed", self.seed, 0, MAX_SEED)
        game_seed = _require_int("game_seed", self.game_seed, 0, MAX_SEED)
        steps = _require_int("steps", self.steps, 1)
        n_envs = _require_int("n_envs", self.n_envs, 1)
        map_name = _require_identity("map_name", self.map_name)
        lattice_sha = _require_sha256(
            "lattice_artifact_sha256",
            self.lattice_artifact_sha256,
            optional=True,
        )
        lane_id = deterministic_lane_id(learner_id, config_hash, lane_index)
        manifest = {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "learner_id": learner_id,
            "config_hash": config_hash,
            "policy_version": policy_version,
            "policy_sha256": policy_sha256,
            "lane_index": lane_index,
            "lane_id": lane_id,
            "rollout_index": rollout_index,
            "seed": seed,
            "game_seed": game_seed,
            "steps": steps,
            "n_envs": n_envs,
            "map_name": map_name,
            "lattice_artifact_sha256": lattice_sha,
        }
        object.__setattr__(self, "lane_id", lane_id)
        object.__setattr__(
            self,
            "assignment_id",
            f"q2a1-{_sha256(_canonical_json(manifest))}",
        )

    @property
    def determinism_key(self) -> str:
        return f"q2-assignment:{self.assignment_id}"

    def batch_contract(self) -> dict:
        """Metadata fields a future leased rollout submission must match."""
        return {
            "assignment_id": self.assignment_id,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "game_seed": self.game_seed,
            "rollout_index": self.rollout_index,
            "determinism_key": self.determinism_key,
            "map_name": self.map_name,
            "n_envs": self.n_envs,
            "lattice_mode": "versioned_snapshot",
            "lattice_artifact_sha256": self.lattice_artifact_sha256,
        }

    def as_dict(self) -> dict:
        return {
            "assignment_schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "learner_id": self.learner_id,
            "config_hash": self.config_hash,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "lane_index": self.lane_index,
            "lane_id": self.lane_id,
            "rollout_index": self.rollout_index,
            "seed": self.seed,
            "game_seed": self.game_seed,
            "steps": self.steps,
            "n_envs": self.n_envs,
            "map_name": self.map_name,
            "lattice_artifact_sha256": self.lattice_artifact_sha256,
            "assignment_id": self.assignment_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping):
            raise ValueError("assignment must be an object")
        if value.get("assignment_schema_version") != ASSIGNMENT_SCHEMA_VERSION:
            raise ValueError("unsupported assignment schema")
        try:
            assignment = cls(
                learner_id=value["learner_id"],
                config_hash=value["config_hash"],
                policy_version=value["policy_version"],
                policy_sha256=value["policy_sha256"],
                lane_index=value["lane_index"],
                rollout_index=value["rollout_index"],
                seed=value["seed"],
                game_seed=value["game_seed"],
                steps=value["steps"],
                n_envs=value["n_envs"],
                map_name=value["map_name"],
                lattice_artifact_sha256=value.get(
                    "lattice_artifact_sha256", ""
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid assignment: {error}") from error
        if value.get("lane_id") != assignment.lane_id:
            raise ValueError("assignment lane ID mismatch")
        if value.get("assignment_id") != assignment.assignment_id:
            raise ValueError("assignment ID mismatch")
        return assignment


def build_generation_assignments(
    *,
    learner_id: str,
    config_hash: str,
    policy_version: int,
    policy_sha256: str,
    rollout_index: int,
    lanes: int,
    steps: int,
    n_envs: int,
    maps: Sequence[str],
    base_seed: int,
    base_game_seed: int,
    lattice_artifacts: Optional[Mapping[int, str]] = None,
) -> tuple[RolloutAssignment, ...]:
    """Build a deterministic, lane-ordered assignment set for one generation."""
    lanes = _require_int("lanes", lanes, 1)
    if isinstance(maps, (str, bytes)) or not maps:
        raise ValueError("maps must contain at least one map")
    normalized_maps = tuple(_require_identity("map_name", name) for name in maps)
    lattice_artifacts = dict(lattice_artifacts or {})
    for lane_index, digest in lattice_artifacts.items():
        _require_int("lattice artifact lane", lane_index)
        _require_sha256("lattice artifact digest", digest)
    unknown_lanes = set(lattice_artifacts) - set(range(lanes))
    if unknown_lanes:
        raise ValueError(f"lattice artifacts reference unknown lanes: {sorted(unknown_lanes)}")

    result = []
    for lane_index in range(lanes):
        seed_identity = {
            "learner_id": learner_id,
            "config_hash": config_hash,
            "lane_index": lane_index,
        }
        result.append(RolloutAssignment(
            learner_id=learner_id,
            config_hash=config_hash,
            policy_version=policy_version,
            policy_sha256=policy_sha256,
            lane_index=lane_index,
            rollout_index=rollout_index,
            seed=deterministic_seed(base_seed, "python", **seed_identity),
            game_seed=deterministic_seed(base_game_seed, "game", **seed_identity),
            steps=steps,
            n_envs=n_envs,
            map_name=normalized_maps[lane_index % len(normalized_maps)],
            lattice_artifact_sha256=lattice_artifacts.get(lane_index, ""),
        ))
    return tuple(result)


class LeaseError(RuntimeError):
    """Base class for assignment lease failures."""


class AssignmentUnavailableError(LeaseError):
    """The requested assignment cannot currently be leased."""


class StaleLeaseError(LeaseError):
    """A heartbeat/result used an expired or superseded fencing token."""


@dataclass(frozen=True)
class AssignmentLease:
    assignment: RolloutAssignment
    worker_id: str
    epoch: int
    lease_id: str
    issued_at: float
    heartbeat_at: float
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return _require_time("now", now) >= self.expires_at

    def as_dict(self) -> dict:
        return {
            "assignment": self.assignment.as_dict(),
            "worker_id": self.worker_id,
            "epoch": self.epoch,
            "lease_id": self.lease_id,
            "issued_at": self.issued_at,
            "heartbeat_at": self.heartbeat_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping):
            raise ValueError("lease must be an object")
        try:
            assignment = RolloutAssignment.from_dict(value["assignment"])
            worker_id = _require_identity("worker_id", value["worker_id"])
            epoch = _require_int("lease epoch", value["epoch"], 1)
            lease_id = _require_identity("lease_id", value["lease_id"])
            issued_at = _require_time("issued_at", value["issued_at"])
            heartbeat_at = _require_time("heartbeat_at", value["heartbeat_at"])
            expires_at = _require_time("expires_at", value["expires_at"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid assignment lease: {error}") from error
        if heartbeat_at < issued_at or expires_at <= heartbeat_at:
            raise ValueError("invalid assignment lease timestamps")
        return cls(
            assignment=assignment,
            worker_id=worker_id,
            epoch=epoch,
            lease_id=lease_id,
            issued_at=issued_at,
            heartbeat_at=heartbeat_at,
            expires_at=expires_at,
        )


@dataclass
class _LeaseRecord:
    assignment: RolloutAssignment
    state: str = "pending"
    attempts: int = 0
    active: Optional[AssignmentLease] = None
    last_failure: str = ""
    completed_lease_id: str = ""


class AssignmentLeaseBook:
    """Thread-safe in-memory lease registry with epoch-based fencing.

    Expiration and retry limits are evaluated against an injected/explicit
    monotonic timestamp, which makes recovery decisions reproducible in tests.
    Persistence and HTTP exposure intentionally remain integration work.
    """

    def __init__(
        self,
        lease_ttl: float = 30.0,
        max_attempts: int = 3,
        incarnation_id: Optional[str] = None,
    ):
        lease_ttl = float(lease_ttl)
        if not math.isfinite(lease_ttl) or lease_ttl <= 0:
            raise ValueError("lease_ttl must be finite and positive")
        self.lease_ttl = lease_ttl
        self.max_attempts = _require_int("max_attempts", max_attempts, 1)
        self.incarnation_id = _require_identity(
            "incarnation_id", incarnation_id or secrets.token_hex(16)
        )
        self._records: Dict[str, _LeaseRecord] = {}
        self._lease_to_assignment: Dict[str, str] = {}
        self._lock = threading.RLock()

    def add(self, assignments: Iterable[RolloutAssignment]) -> None:
        with self._lock:
            for assignment in assignments:
                if not isinstance(assignment, RolloutAssignment):
                    raise TypeError("lease book accepts RolloutAssignment values")
                existing = self._records.get(assignment.assignment_id)
                if existing is not None and existing.assignment != assignment:
                    raise ValueError("assignment ID collision")
                self._records.setdefault(
                    assignment.assignment_id, _LeaseRecord(assignment)
                )

    def _now(self, now: Optional[float]) -> float:
        return time.monotonic() if now is None else _require_time("now", now)

    def _expire_record(self, record: _LeaseRecord, now: float) -> bool:
        lease = record.active
        if record.state != "leased" or lease is None or now < lease.expires_at:
            return False
        record.active = None
        record.last_failure = "lease_expired"
        record.state = "failed" if record.attempts >= self.max_attempts else "pending"
        return True

    def expire(self, now: Optional[float] = None) -> tuple[str, ...]:
        current = self._now(now)
        with self._lock:
            expired = [
                assignment_id
                for assignment_id, record in self._records.items()
                if self._expire_record(record, current)
            ]
        return tuple(sorted(expired))

    def claim(
        self,
        worker_id: str,
        *,
        now: Optional[float] = None,
        assignment_id: Optional[str] = None,
    ) -> Optional[AssignmentLease]:
        worker_id = _require_identity("worker_id", worker_id)
        current = self._now(now)
        with self._lock:
            for record in self._records.values():
                self._expire_record(record, current)
            if assignment_id is not None:
                record = self._records.get(assignment_id)
                if record is None:
                    raise AssignmentUnavailableError("unknown assignment")
                candidates = [record]
            else:
                candidates = sorted(
                    self._records.values(),
                    key=lambda item: (
                        item.assignment.policy_version,
                        item.assignment.lane_index,
                        item.assignment.assignment_id,
                    ),
                )
            for record in candidates:
                if record.state != "pending":
                    continue
                if record.attempts >= self.max_attempts:
                    record.state = "failed"
                    continue
                epoch = record.attempts + 1
                lease_id = "q2lease1-" + _stable_digest(
                    "q2-assignment-lease-v1",
                    {
                        "assignment_id": record.assignment.assignment_id,
                        "worker_id": worker_id,
                        "epoch": epoch,
                        "incarnation_id": self.incarnation_id,
                    },
                )
                lease = AssignmentLease(
                    assignment=record.assignment,
                    worker_id=worker_id,
                    epoch=epoch,
                    lease_id=lease_id,
                    issued_at=current,
                    heartbeat_at=current,
                    expires_at=current + self.lease_ttl,
                )
                record.attempts = epoch
                record.state = "leased"
                record.active = lease
                record.last_failure = ""
                self._lease_to_assignment[lease_id] = record.assignment.assignment_id
                return lease
            if assignment_id is not None:
                raise AssignmentUnavailableError("assignment is not leaseable")
            return None

    def _active_record(self, lease_id: str, now: float) -> _LeaseRecord:
        assignment_id = self._lease_to_assignment.get(lease_id)
        record = self._records.get(assignment_id or "")
        if record is None:
            raise StaleLeaseError("unknown lease")
        self._expire_record(record, now)
        if record.state != "leased" or record.active is None:
            raise StaleLeaseError("lease is expired or no longer active")
        if record.active.lease_id != lease_id:
            raise StaleLeaseError("lease was superseded")
        if now < record.active.heartbeat_at:
            raise ValueError("lease clock moved backwards")
        return record

    def heartbeat(
        self,
        lease_id: str,
        worker_id: str,
        *,
        now: Optional[float] = None,
    ) -> AssignmentLease:
        worker_id = _require_identity("worker_id", worker_id)
        current = self._now(now)
        with self._lock:
            record = self._active_record(lease_id, current)
            active = record.active
            assert active is not None
            if active.worker_id != worker_id:
                raise StaleLeaseError("lease worker does not match")
            renewed = AssignmentLease(
                assignment=active.assignment,
                worker_id=active.worker_id,
                epoch=active.epoch,
                lease_id=active.lease_id,
                issued_at=active.issued_at,
                heartbeat_at=current,
                expires_at=current + self.lease_ttl,
            )
            record.active = renewed
            return renewed

    def active_lease(
        self, lease_id: str, *, now: Optional[float] = None
    ) -> AssignmentLease:
        current = self._now(now)
        with self._lock:
            record = self._active_record(lease_id, current)
            assert record.active is not None
            return record.active

    def complete(
        self,
        lease_id: str,
        worker_id: str,
        *,
        now: Optional[float] = None,
        before_complete: Optional[Callable[[], None]] = None,
    ) -> RolloutAssignment:
        worker_id = _require_identity("worker_id", worker_id)
        current = self._now(now)
        with self._lock:
            record = self._active_record(lease_id, current)
            assert record.active is not None
            if record.active.worker_id != worker_id:
                raise StaleLeaseError("lease worker does not match")
            if before_complete is not None:
                before_complete()
            assignment = record.assignment
            record.completed_lease_id = lease_id
            record.active = None
            record.state = "completed"
            return assignment

    def release(
        self,
        lease_id: str,
        worker_id: str,
        reason: str,
        *,
        retryable: bool = True,
        now: Optional[float] = None,
    ) -> str:
        worker_id = _require_identity("worker_id", worker_id)
        reason = _require_identity("reason", reason)
        current = self._now(now)
        with self._lock:
            record = self._active_record(lease_id, current)
            assert record.active is not None
            if record.active.worker_id != worker_id:
                raise StaleLeaseError("lease worker does not match")
            record.active = None
            record.last_failure = reason
            if retryable and record.attempts < self.max_attempts:
                record.state = "pending"
            else:
                record.state = "failed"
            return record.state

    def snapshot(self, assignment_id: str) -> dict:
        with self._lock:
            record = self._records.get(assignment_id)
            if record is None:
                raise AssignmentUnavailableError("unknown assignment")
            active = record.active
            return {
                "assignment_id": assignment_id,
                "state": record.state,
                "attempts": record.attempts,
                "worker_id": active.worker_id if active else "",
                "lease_id": active.lease_id if active else "",
                "lease_epoch": active.epoch if active else record.attempts,
                "expires_at": active.expires_at if active else None,
                "last_failure": record.last_failure,
                "completed_lease_id": record.completed_lease_id,
            }

    def restore_completed(
        self,
        assignment_id: str,
        *,
        attempts: int,
        completed_lease_id: str,
    ) -> None:
        """Restore a durably accepted assignment without reviving its lease."""
        attempts = _require_int("attempts", attempts, 1, self.max_attempts)
        completed_lease_id = _require_identity(
            "completed_lease_id", completed_lease_id
        )
        with self._lock:
            record = self._records.get(assignment_id)
            if record is None:
                raise AssignmentUnavailableError("unknown assignment")
            if record.state not in {"pending", "completed"}:
                raise StaleLeaseError("cannot restore over an active assignment")
            record.attempts = max(record.attempts, attempts)
            record.active = None
            record.state = "completed"
            record.completed_lease_id = completed_lease_id
            record.last_failure = ""


def validate_batch_lease(
    metadata: Mapping[str, object],
    assignment: RolloutAssignment,
    lease: AssignmentLease,
    *,
    now: Optional[float] = None,
) -> None:
    """Fail closed if rollout metadata does not satisfy its leased contract."""
    if lease.assignment.assignment_id != assignment.assignment_id:
        raise ValueError("lease belongs to a different assignment")
    if now is not None and lease.is_expired(now):
        raise ValueError("rollout lease has expired")
    expected = assignment.batch_contract()
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"rollout metadata mismatch: {name}")
    if metadata.get("worker_id") != lease.worker_id:
        raise ValueError("rollout metadata mismatch: worker_id")
    if metadata.get("lease_id") != lease.lease_id:
        raise ValueError("rollout metadata mismatch: lease_id")
    if metadata.get("lease_epoch") != lease.epoch:
        raise ValueError("rollout metadata mismatch: lease_epoch")


@dataclass(frozen=True)
class BackoffPolicy:
    """Bounded exponential retry timing with stable per-worker jitter."""

    initial_delay: float = 0.5
    maximum_delay: float = 30.0
    multiplier: float = 2.0
    jitter_fraction: float = 0.2
    max_attempts: int = 0

    def __post_init__(self) -> None:
        for name in ("initial_delay", "maximum_delay", "multiplier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        jitter_fraction = float(self.jitter_fraction)
        object.__setattr__(self, "jitter_fraction", jitter_fraction)
        if self.initial_delay > self.maximum_delay:
            raise ValueError("initial_delay must not exceed maximum_delay")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least one")
        if (
            not math.isfinite(jitter_fraction)
            or not 0 <= self.jitter_fraction <= 1
        ):
            raise ValueError("jitter_fraction must be in [0, 1]")
        _require_int("max_attempts", self.max_attempts)

    def delay(self, attempt: int, jitter_key: str = "default") -> float:
        attempt = _require_int("attempt", attempt)
        try:
            raw = min(
                self.maximum_delay,
                self.initial_delay * (self.multiplier ** min(attempt, 1024)),
            )
        except OverflowError:
            raw = self.maximum_delay
        if self.jitter_fraction == 0:
            return raw
        jitter_key = _require_identity("jitter_key", jitter_key)
        digest = _stable_digest(
            "q2-reconnect-jitter-v1",
            {"jitter_key": jitter_key, "attempt": attempt},
        )
        unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        factor = 1.0 + self.jitter_fraction * (2.0 * unit - 1.0)
        return min(self.maximum_delay, max(0.0, raw * factor))


class RetryExhaustedError(RuntimeError):
    pass


class ReconnectBackoff:
    """Mutable reconnect counter; call ``reset`` after any successful request."""

    def __init__(self, policy: Optional[BackoffPolicy] = None, jitter_key: str = "worker"):
        self.policy = policy or BackoffPolicy()
        self.jitter_key = _require_identity("jitter_key", jitter_key)
        self.attempts = 0

    def next_delay(self) -> float:
        if self.policy.max_attempts and self.attempts >= self.policy.max_attempts:
            raise RetryExhaustedError(
                f"reconnect retry budget exhausted after {self.attempts} attempts"
            )
        result = self.policy.delay(self.attempts, self.jitter_key)
        self.attempts += 1
        return result

    def reset(self) -> None:
        self.attempts = 0


RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_http_status(status: int) -> bool:
    return int(status) in RETRYABLE_HTTP_STATUSES


@dataclass(frozen=True)
class LatticePayloadInfo:
    state_version: int
    env_steps: int
    instance_count: int


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant {value}")


def validate_lattice_payload(payload: bytes) -> LatticePayloadInfo:
    """Validate the existing spatial.py lattice JSON/gzip checkpoint shape."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("lattice payload must be bytes")
    payload = bytes(payload)
    if not payload or len(payload) > MAX_LATTICE_PAYLOAD_BYTES:
        raise ValueError("lattice payload is empty or too large")
    if payload.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as handle:
                decoded = handle.read(MAX_LATTICE_JSON_BYTES + 1)
        except (EOFError, OSError) as error:
            raise ValueError("invalid gzip lattice payload") from error
        if len(decoded) > MAX_LATTICE_JSON_BYTES:
            raise ValueError("decompressed lattice payload is too large")
    else:
        decoded = payload
        if len(decoded) > MAX_LATTICE_JSON_BYTES:
            raise ValueError("lattice JSON payload is too large")
    try:
        document = json.loads(
            decoded.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid lattice JSON payload") from error
    if not isinstance(document, dict):
        raise ValueError("lattice payload root must be an object")
    state_version = _require_int("lattice state version", document.get("version", 0))
    if state_version != LATTICE_STATE_VERSION:
        raise ValueError(f"unsupported lattice state version {state_version}")
    env_steps = _require_int("lattice env_steps", document.get("env_steps", -1))
    instances = document.get("instances")
    if not isinstance(instances, list):
        raise ValueError("lattice instances must be a list")
    for instance in instances:
        if not isinstance(instance, dict) or not isinstance(instance.get("maps", {}), dict):
            raise ValueError("invalid lattice instance")
        for map_name, cells in instance.get("maps", {}).items():
            _require_identity("lattice map name", map_name)
            if not isinstance(cells, list):
                raise ValueError("lattice map cells must be a list")
            for cell in cells:
                if not isinstance(cell, dict):
                    raise ValueError("lattice cell must be an object")
                coordinate = cell.get("cell")
                if (
                    not isinstance(coordinate, list)
                    or len(coordinate) != 3
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in coordinate)
                ):
                    raise ValueError("lattice cell coordinate must contain three integers")
    return LatticePayloadInfo(state_version, env_steps, len(instances))


@dataclass(frozen=True)
class LatticeArtifact:
    """Learner-owned envelope around one accepted lane's lattice snapshot."""

    learner_id: str
    config_hash: str
    lane_index: int
    lane_id: str
    source_assignment_id: str
    policy_version: int
    parent_artifact_sha256: str
    env_steps: int
    instance_count: int
    payload: bytes = field(repr=False)
    payload_sha256: str

    def __post_init__(self) -> None:
        _require_identity("learner_id", self.learner_id)
        _require_sha256("config_hash", self.config_hash)
        _require_int("lane_index", self.lane_index)
        expected_lane = deterministic_lane_id(
            self.learner_id, self.config_hash, self.lane_index
        )
        if self.lane_id != expected_lane:
            raise ValueError("lattice lane identity mismatch")
        _require_identity("source_assignment_id", self.source_assignment_id)
        if not _ASSIGNMENT_ID.fullmatch(self.source_assignment_id):
            raise ValueError("invalid lattice source assignment ID")
        _require_int("policy_version", self.policy_version)
        _require_sha256(
            "parent_artifact_sha256", self.parent_artifact_sha256, optional=True
        )
        _require_int("env_steps", self.env_steps)
        _require_int("instance_count", self.instance_count, 1)
        _require_sha256("payload_sha256", self.payload_sha256)
        payload = bytes(self.payload)
        if _sha256(payload) != self.payload_sha256:
            raise ValueError("lattice payload checksum mismatch")
        info = validate_lattice_payload(payload)
        if info.env_steps != self.env_steps:
            raise ValueError("lattice env_steps metadata mismatch")
        if info.instance_count != self.instance_count:
            raise ValueError("lattice instance count metadata mismatch")
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_assignment(cls, assignment: RolloutAssignment, payload: bytes):
        info = validate_lattice_payload(payload)
        if info.instance_count != assignment.n_envs:
            raise ValueError(
                "lattice instance count does not match assignment environments"
            )
        return cls(
            learner_id=assignment.learner_id,
            config_hash=assignment.config_hash,
            lane_index=assignment.lane_index,
            lane_id=assignment.lane_id,
            source_assignment_id=assignment.assignment_id,
            policy_version=assignment.policy_version,
            parent_artifact_sha256=assignment.lattice_artifact_sha256,
            env_steps=info.env_steps,
            instance_count=info.instance_count,
            payload=bytes(payload),
            payload_sha256=_sha256(bytes(payload)),
        )

    def _manifest(self) -> dict:
        return {
            "artifact_schema_version": LATTICE_ARTIFACT_SCHEMA_VERSION,
            "lattice_state_version": LATTICE_STATE_VERSION,
            "learner_id": self.learner_id,
            "config_hash": self.config_hash,
            "lane_index": self.lane_index,
            "lane_id": self.lane_id,
            "source_assignment_id": self.source_assignment_id,
            "policy_version": self.policy_version,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "env_steps": self.env_steps,
            "instance_count": self.instance_count,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": len(self.payload),
        }

    def encode(self) -> bytes:
        manifest = _canonical_json(self._manifest())
        result = LATTICE_MAGIC + struct.pack("!I", len(manifest)) + manifest + self.payload
        if len(result) > MAX_LATTICE_PAYLOAD_BYTES + 65536:
            raise ValueError("encoded lattice artifact is too large")
        return result

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.encode())

    @classmethod
    def decode(cls, encoded: bytes):
        if not isinstance(encoded, (bytes, bytearray, memoryview)):
            raise TypeError("encoded lattice artifact must be bytes")
        encoded = bytes(encoded)
        if len(encoded) < 12 or encoded[:8] != LATTICE_MAGIC:
            raise ValueError("invalid lattice artifact header")
        manifest_bytes = struct.unpack("!I", encoded[8:12])[0]
        end = 12 + manifest_bytes
        if end > len(encoded) or manifest_bytes > 65536:
            raise ValueError("invalid lattice artifact manifest length")
        try:
            manifest = json.loads(
                encoded[12:end].decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid lattice artifact manifest") from error
        if not isinstance(manifest, dict):
            raise ValueError("lattice artifact manifest must be an object")
        try:
            artifact_schema = _require_int(
                "artifact_schema_version",
                manifest.get("artifact_schema_version", 0),
            )
            lattice_schema = _require_int(
                "lattice_state_version",
                manifest.get("lattice_state_version", 0),
            )
        except ValueError as error:
            raise ValueError(f"invalid lattice artifact: {error}") from error
        if artifact_schema != LATTICE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported lattice artifact schema")
        if lattice_schema != LATTICE_STATE_VERSION:
            raise ValueError("unsupported lattice state schema")
        payload = encoded[end:]
        if len(payload) != int(manifest.get("payload_bytes", -1)):
            raise ValueError("lattice artifact payload length mismatch")
        try:
            return cls(
                learner_id=manifest["learner_id"],
                config_hash=manifest["config_hash"],
                lane_index=manifest["lane_index"],
                lane_id=manifest["lane_id"],
                source_assignment_id=manifest["source_assignment_id"],
                policy_version=manifest["policy_version"],
                parent_artifact_sha256=manifest.get("parent_artifact_sha256", ""),
                env_steps=manifest["env_steps"],
                instance_count=manifest["instance_count"],
                payload=payload,
                payload_sha256=manifest["payload_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid lattice artifact: {error}") from error

    def validate_source_assignment(self, assignment: RolloutAssignment) -> None:
        expected = (
            assignment.learner_id,
            assignment.config_hash,
            assignment.lane_index,
            assignment.lane_id,
            assignment.assignment_id,
            assignment.policy_version,
            assignment.lattice_artifact_sha256,
            assignment.n_envs,
        )
        actual = (
            self.learner_id,
            self.config_hash,
            self.lane_index,
            self.lane_id,
            self.source_assignment_id,
            self.policy_version,
            self.parent_artifact_sha256,
            self.instance_count,
        )
        if actual != expected:
            raise ValueError("lattice artifact does not match its source assignment")

    def validate_recovery_for(self, assignment: RolloutAssignment) -> None:
        if not assignment.lattice_artifact_sha256:
            raise ValueError("assignment requests an empty lattice genesis")
        if self.artifact_sha256 != assignment.lattice_artifact_sha256:
            raise ValueError("assignment references a different lattice artifact")
        if (
            self.learner_id != assignment.learner_id
            or self.config_hash != assignment.config_hash
            or self.lane_id != assignment.lane_id
            or self.instance_count != assignment.n_envs
            or self.policy_version > assignment.policy_version
        ):
            raise ValueError("lattice artifact is incompatible with assignment recovery")


class LearnerLatticeStore:
    """Atomic, learner-authoritative store for per-lane lattice chains."""

    def __init__(self, root: Path, learner_id: str, config_hash: str):
        self.root = Path(root)
        self.learner_id = _require_identity("learner_id", learner_id)
        self.config_hash = _require_sha256("config_hash", config_hash)
        self._lock = threading.RLock()

    def _lane_dir(self, lane_index: int) -> Path:
        lane_id = deterministic_lane_id(self.learner_id, self.config_hash, lane_index)
        return self.root / f"lane_{lane_index:04d}_{lane_id[-12:]}"

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _validate_owner(self, artifact: LatticeArtifact) -> None:
        if (
            artifact.learner_id != self.learner_id
            or artifact.config_hash != self.config_hash
        ):
            raise ValueError("lattice artifact belongs to a different learner/config")

    def latest(self, lane_index: int) -> Optional[LatticeArtifact]:
        lane_index = _require_int("lane_index", lane_index)
        lane_dir = self._lane_dir(lane_index)
        pointer_path = lane_dir / "latest.json"
        if not pointer_path.is_file():
            return None
        try:
            pointer = json.loads(pointer_path.read_text())
            if not isinstance(pointer, dict):
                raise ValueError("lattice latest pointer must be an object")
            if (
                _require_int(
                    "lattice latest-pointer version", pointer.get("version", 0)
                )
                != 1
            ):
                raise ValueError("unsupported lattice latest-pointer version")
            filename = pointer["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("invalid lattice latest-pointer filename")
            artifact = LatticeArtifact.decode((lane_dir / filename).read_bytes())
            self._validate_owner(artifact)
            if artifact.lane_index != lane_index:
                raise ValueError("lattice latest pointer crosses lanes")
            if artifact.artifact_sha256 != pointer.get("artifact_sha256"):
                raise ValueError("lattice latest-pointer checksum mismatch")
            pointer_policy_version = _require_int(
                "lattice latest-pointer policy version",
                pointer.get("policy_version", -1),
            )
            if artifact.policy_version != pointer_policy_version:
                raise ValueError("lattice latest-pointer policy version mismatch")
            return artifact
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid lattice latest pointer") from error

    def get(self, lane_index: int, artifact_sha256: str) -> LatticeArtifact:
        lane_index = _require_int("lane_index", lane_index)
        artifact_sha256 = _require_sha256("artifact_sha256", artifact_sha256)
        matches = sorted(self._lane_dir(lane_index).glob(f"lattice_*_{artifact_sha256}.q2la"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one lattice artifact {artifact_sha256}, found {len(matches)}"
            )
        artifact = LatticeArtifact.decode(matches[0].read_bytes())
        self._validate_owner(artifact)
        if artifact.lane_index != lane_index or artifact.artifact_sha256 != artifact_sha256:
            raise ValueError("stored lattice artifact identity mismatch")
        return artifact

    def adopt(
        self, assignment: RolloutAssignment, payload: bytes
    ) -> tuple[LatticeArtifact, Path]:
        """Validate a worker payload, then publish it under learner ownership."""
        artifact = LatticeArtifact.from_assignment(assignment, payload)
        return artifact, self.publish(artifact, assignment)

    def publish(
        self, artifact: LatticeArtifact, assignment: RolloutAssignment
    ) -> Path:
        artifact.validate_source_assignment(assignment)
        self._validate_owner(artifact)
        lane_dir = self._lane_dir(artifact.lane_index)
        with self._lock:
            latest = self.latest(artifact.lane_index)
            if latest is None:
                if artifact.parent_artifact_sha256:
                    raise ValueError("first lattice artifact cannot name a parent")
            else:
                if artifact.policy_version < latest.policy_version:
                    raise ValueError("lattice artifact policy versions must increase")
                if artifact.policy_version == latest.policy_version:
                    if artifact.artifact_sha256 == latest.artifact_sha256:
                        matches = list(lane_dir.glob(
                            f"lattice_*_{artifact.artifact_sha256}.q2la"
                        ))
                        if len(matches) != 1:
                            raise ValueError("idempotent lattice artifact file is missing")
                        return matches[0]
                    raise ValueError("conflicting lattice artifact for policy version")
                if artifact.parent_artifact_sha256 != latest.artifact_sha256:
                    raise ValueError("lattice artifact parent is not the lane latest")
                if artifact.env_steps < latest.env_steps:
                    raise ValueError("lattice artifact env_steps cannot move backwards")

            filename = (
                f"lattice_{artifact.policy_version:020d}_"
                f"{artifact.artifact_sha256}.q2la"
            )
            target = lane_dir / filename
            if target.exists():
                stored = LatticeArtifact.decode(target.read_bytes())
                if stored.artifact_sha256 != artifact.artifact_sha256:
                    raise ValueError("lattice artifact filename collision")
            else:
                self._atomic_write(target, artifact.encode())
            pointer = _canonical_json({
                "version": 1,
                "filename": filename,
                "artifact_sha256": artifact.artifact_sha256,
                "policy_version": artifact.policy_version,
            }) + b"\n"
            self._atomic_write(lane_dir / "latest.json", pointer)
            return target

    def recover_for(self, assignment: RolloutAssignment) -> Optional[LatticeArtifact]:
        if not assignment.lattice_artifact_sha256:
            return None
        artifact = self.get(
            assignment.lane_index, assignment.lattice_artifact_sha256
        )
        artifact.validate_recovery_for(assignment)
        return artifact
