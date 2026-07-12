"""Conservative acceptance gate for concurrent real-q2 rollout probes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ThroughputGateError(ValueError):
    pass


@dataclass(frozen=True)
class WorkerThroughput:
    worker_id: str
    started_unix_ns: int
    finished_unix_ns: int
    elapsed_seconds: float
    transitions: int
    timeouts: int
    rollout_sps: float
    runtime_manifest_sha256: str
    policy_sha256: str
    config_hash: str
    producer: str


@dataclass(frozen=True)
class ThroughputGateResult:
    passed: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]
    workers: tuple[WorkerThroughput, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "workers": [worker.__dict__.copy() for worker in self.workers],
        }


def _collection(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("collection")
    if value is None:
        value = record.get("throughput")
    if not isinstance(value, Mapping):
        raise ThroughputGateError("record is missing collection timing metadata")
    return value


def parse_worker_throughput(record: Mapping[str, Any]) -> WorkerThroughput:
    collection = _collection(record)
    try:
        worker_id = str(record["worker_id"])
        started = int(collection["started_unix_ns"])
        finished = int(collection["finished_unix_ns"])
        elapsed = float(collection["elapsed_seconds"])
        transitions = int(collection["transitions"])
        timeouts = int(collection.get("timeouts", 0))
        runtime_digest = str(record["runtime_manifest_sha256"])
        policy_sha256 = str(record["policy_sha256"])
        config_hash = str(record.get("config_hash", ""))
        producer = str(record.get("producer", ""))
    except (KeyError, TypeError, ValueError) as error:
        raise ThroughputGateError(f"invalid collection record: {error}") from error
    if not worker_id:
        raise ThroughputGateError("worker_id cannot be empty")
    if started <= 0 or finished <= started:
        raise ThroughputGateError(f"{worker_id}: invalid collection interval")
    if elapsed <= 0 or transitions <= 0 or timeouts < 0:
        raise ThroughputGateError(f"{worker_id}: invalid elapsed/transitions/timeouts")
    epoch_elapsed = (finished - started) / 1_000_000_000.0
    tolerance = max(0.250, elapsed * 0.05)
    if abs(epoch_elapsed - elapsed) > tolerance:
        raise ThroughputGateError(
            f"{worker_id}: wall/monotonic duration mismatch "
            f"({epoch_elapsed:.3f}s vs {elapsed:.3f}s)"
        )
    return WorkerThroughput(
        worker_id=worker_id,
        started_unix_ns=started,
        finished_unix_ns=finished,
        elapsed_seconds=elapsed,
        transitions=transitions,
        timeouts=timeouts,
        rollout_sps=transitions / elapsed,
        runtime_manifest_sha256=runtime_digest,
        policy_sha256=policy_sha256,
        config_hash=config_hash,
        producer=producer,
    )


def evaluate_throughput_gate(
    records: Iterable[Mapping[str, Any]],
    *,
    min_workers: int = 2,
    min_aggregate_sps: float = 0.0,
    baseline_sps: float = 0.0,
    min_speedup: float = 1.0,
    min_overlap_ratio: float = 0.50,
    max_timeouts: int = 0,
    expected_runtime_manifest_sha256: Optional[str] = None,
) -> ThroughputGateResult:
    workers = tuple(parse_worker_throughput(record) for record in records)
    reasons: list[str] = []
    if len(workers) < int(min_workers):
        reasons.append(f"need at least {int(min_workers)} workers, got {len(workers)}")
    worker_ids = [worker.worker_id for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        reasons.append("worker IDs are not unique")
    if any(worker.producer != "q2" for worker in workers):
        reasons.append("all probes must be producer=q2")

    runtime_digests = {worker.runtime_manifest_sha256 for worker in workers}
    if any(not _SHA256.fullmatch(value) for value in runtime_digests):
        reasons.append("every worker must carry a valid runtime manifest SHA-256")
    if len(runtime_digests) != 1:
        reasons.append("workers attested to different runtime manifests")
    if (
        expected_runtime_manifest_sha256
        and runtime_digests != {expected_runtime_manifest_sha256}
    ):
        reasons.append("worker runtime manifest does not match the expected digest")

    policy_hashes = {worker.policy_sha256 for worker in workers}
    if len(policy_hashes) != 1:
        reasons.append("workers used different policy artifacts")
    config_hashes = {worker.config_hash for worker in workers}
    if len(config_hashes) != 1:
        reasons.append("workers used different learner configurations")
    total_timeouts = sum(worker.timeouts for worker in workers)
    if total_timeouts > int(max_timeouts):
        reasons.append(f"timeouts {total_timeouts} exceed allowed {int(max_timeouts)}")

    if workers:
        union_start = min(worker.started_unix_ns for worker in workers)
        union_finish = max(worker.finished_unix_ns for worker in workers)
        overlap_start = max(worker.started_unix_ns for worker in workers)
        overlap_finish = min(worker.finished_unix_ns for worker in workers)
        union_seconds = (union_finish - union_start) / 1_000_000_000.0
        overlap_seconds = max(0.0, (overlap_finish - overlap_start) / 1_000_000_000.0)
        shortest_seconds = min(worker.elapsed_seconds for worker in workers)
        # Epoch and monotonic clocks are sampled separately, so nanosecond-
        # scale sampling jitter can otherwise report 1.00000x overlap.
        overlap_ratio = min(1.0, overlap_seconds / shortest_seconds)
        total_transitions = sum(worker.transitions for worker in workers)
        aggregate_sps = total_transitions / union_seconds
    else:
        union_seconds = overlap_seconds = overlap_ratio = aggregate_sps = 0.0
        total_transitions = total_timeouts = 0
    if overlap_ratio < float(min_overlap_ratio):
        reasons.append(
            f"collection overlap {overlap_ratio:.3f} is below {float(min_overlap_ratio):.3f}"
        )
    required_sps = max(float(min_aggregate_sps), float(baseline_sps) * float(min_speedup))
    if aggregate_sps < required_sps:
        reasons.append(
            f"aggregate SPS {aggregate_sps:.3f} is below required {required_sps:.3f}"
        )

    metrics = {
        "worker_count": len(workers),
        "total_transitions": total_transitions,
        "total_timeouts": total_timeouts,
        "union_seconds": round(union_seconds, 6),
        "overlap_seconds": round(overlap_seconds, 6),
        "overlap_ratio": round(overlap_ratio, 6),
        "aggregate_sps": round(aggregate_sps, 6),
        "required_sps": round(required_sps, 6),
        "baseline_sps": float(baseline_sps),
        "speedup_vs_baseline": (
            round(aggregate_sps / float(baseline_sps), 6)
            if float(baseline_sps) > 0 else None
        ),
    }
    return ThroughputGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
        workers=workers,
    )
