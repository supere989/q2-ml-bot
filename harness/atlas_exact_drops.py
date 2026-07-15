"""Batched exact Atlas drop classification over persistent B1 authorities.

This module deliberately contains no collision or fall approximation.  It
admits an already-returned q2-pmove-oracle trajectory through
``atlas_drop_replay`` and obtains the matching fall result from one persistent
q2-fall-oracle process.  Unknown evidence is retained as Unknown and can never
create a traversal edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import subprocess
import time
from typing import Any, Mapping, Sequence

from .atlas_drop_replay import (
    evaluate_drop_evidence,
    pmove_requests_for_drop_manifest,
    prepare_drop_fall_request,
)


DROP_CLASSIFICATION_SCHEMA = "q2-atlas-exact-drop-classification-v1"
DROP_EVIDENCE_PMOVE = 2
DROP_EVIDENCE_FALL = 8
DROP_EVIDENCE_EXACT = DROP_EVIDENCE_PMOVE | DROP_EVIDENCE_FALL
DROP_VALIDATION_VERSION = 1


class ExactDropAnalysisError(RuntimeError):
    """The exact fall authority failed or changed during an Atlas build."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ExactDropAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _decode_json_object(raw: bytes) -> dict[str, Any]:
    """Decode one strict oracle record (RFC JSON, unique object keys)."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON token {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = item
        return result

    value = json.loads(
        raw, parse_constant=reject_constant, object_pairs_hook=unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("oracle response is not an object")
    return value


class FallOracleProcess:
    """One persistent, request-bounded q2-fall-oracle NDJSON process."""

    def __init__(
        self,
        binary: Path,
        *,
        max_requests: int,
        batch_size: int = 32,
        batch_timeout_seconds: float = 10.0,
        exit_timeout_seconds: float = 2.0,
    ) -> None:
        self.binary = binary.resolve()
        self.max_requests = max_requests
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.exit_timeout_seconds = exit_timeout_seconds
        self.requests = 0
        self._buffer = bytearray()
        self.process = subprocess.Popen(
            [str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self._kill()
            raise ExactDropAnalysisError("failed to open fall oracle pipes")
        self.identity = self.call([{
            "id": "atlas-fall-identity", "op": "identity",
            "fall_damagemod": 1.0, "deathmatch": True, "dmflags": 0,
        }])[0]
        self._validate_identity(self.identity)

    def _read_lines(self, count: int) -> list[dict[str, Any]]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + self.batch_timeout_seconds
        records: list[dict[str, Any]] = []
        while len(records) < count:
            while b"\n" in self._buffer and len(records) < count:
                raw, _, remainder = self._buffer.partition(b"\n")
                self._buffer[:] = remainder
                try:
                    value = _decode_json_object(raw)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise ExactDropAnalysisError(
                        "fall oracle emitted invalid JSON"
                    ) from error
                records.append(value)
            if len(records) == count:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise ExactDropAnalysisError("fall oracle batch timed out")
            ready, _, _ = select.select(
                [self.process.stdout.fileno()], [], [], remaining
            )
            if not ready:
                self._kill()
                raise ExactDropAnalysisError("fall oracle batch timed out")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                detail = b""
                if self.process.stderr is not None:
                    detail = self.process.stderr.read(4096)
                raise ExactDropAnalysisError(
                    "fall oracle exited early: "
                    + detail.decode(errors="replace")
                )
            self._buffer.extend(block)
        return records

    def call(
        self, requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        if len(requests) > self.batch_size:
            output: list[dict[str, Any]] = []
            for start in range(0, len(requests), self.batch_size):
                output.extend(self.call(requests[start:start + self.batch_size]))
            return output
        self.requests += len(requests)
        if self.requests > self.max_requests:
            raise ExactDropAnalysisError("fall oracle request budget exceeded")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(
                b"".join(_canonical(dict(request)) + b"\n" for request in requests)
            )
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise ExactDropAnalysisError("fall oracle pipe closed") from error
        records = self._read_lines(len(requests))
        for request, record in zip(requests, records):
            if record.get("id") != request.get("id"):
                raise ExactDropAnalysisError("fall oracle response ID mismatch")
            if record.get("ok") is not True:
                raise ExactDropAnalysisError(
                    f"fall oracle rejected {request.get('op')}: "
                    f"{record.get('error')}"
                )
            if hasattr(self, "identity"):
                for field in ("schema", "tool_identity", "physics_identity"):
                    if record.get(field) != self.identity.get(field):
                        raise ExactDropAnalysisError(
                            f"fall oracle {field} changed within process"
                        )
        return records

    @staticmethod
    def _validate_identity(record: Mapping[str, Any]) -> None:
        required = {
            "ok", "id", "op", "schema", "physics_identity", "tool_identity",
            "parameters", "constants", "source",
        }
        if set(record) != required:
            raise ExactDropAnalysisError("fall oracle identity fields differ")
        if (
            record.get("op") != "identity"
            or record.get("schema") != "q2-fall-oracle-v1"
        ):
            raise ExactDropAnalysisError("fall oracle identity schema differs")
        _require_sha256(record.get("tool_identity"), "fall tool identity")
        _require_sha256(record.get("physics_identity"), "fall physics identity")
        if record.get("parameters") != {
            "fall_damagemod": 1, "deathmatch": True, "dmflags": 0,
        }:
            raise ExactDropAnalysisError("fall oracle parameter block differs")

    def _kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=self.exit_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._kill()
            self.process.wait()
        if self.process.returncode:
            detail = b""
            if self.process.stderr is not None:
                detail = self.process.stderr.read(4096)
            raise ExactDropAnalysisError(
                f"fall oracle exited {self.process.returncode}: "
                + detail.decode(errors="replace")
            )

    def __enter__(self) -> "FallOracleProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None:
            self._kill()
            self.process.wait()
        else:
            self.close()


@dataclass(frozen=True)
class DropTrajectory:
    """One deterministic Pmove trajectory plus its Atlas boundary context."""

    identifier: str
    source_l1: tuple[int, int, int]
    direction: tuple[int, int]
    mode: str
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    dynamic_movers: bool = False


def _drop_manifest(
    trajectory: DropTrajectory,
    *,
    bsp: Path,
    pmove_process: Any,
    fall_process: FallOracleProcess,
) -> dict[str, Any]:
    request = trajectory.request
    commands = request.get("commands")
    if not isinstance(commands, list) or len(commands) < 2:
        raise ExactDropAnalysisError("drop trajectory has no bounded command horizon")
    cadence = commands[0].get("msec") if isinstance(commands[0], Mapping) else None
    if type(cadence) is not int or any(
        not isinstance(command, Mapping) or command.get("msec") != cadence
        for command in commands
    ):
        raise ExactDropAnalysisError("drop trajectory cadence is not uniform")
    parameters = pmove_process.identity["parameters"]
    pmove = {
        "origin": list(request.get("origin", ())),
        "velocity": list(request.get("velocity", [0.0, 0.0, 0.0])),
        "pm_type": request.get("pm_type", 0),
        "pm_flags": request.get("pm_flags", 0),
        "pm_time": request.get("pm_time", 0),
        "gravity": request.get("gravity", parameters["gravity"]),
        "airaccelerate": request.get(
            "airaccelerate", parameters["airaccelerate"]
        ),
        "delta_angles_short": list(
            request.get("delta_angles_short", [0, 0, 0])
        ),
        "snapinitial": request.get("snapinitial", False),
        "commands": [dict(command) for command in commands],
    }
    return {
        "schema": "q2-atlas-drop-replay-manifest-v1",
        "id": trajectory.identifier,
        "map_path": str(bsp.resolve()),
        "horizon_frames": len(commands),
        "cadence_msec": cadence,
        "dynamic_movers": trajectory.dynamic_movers,
        "pmove": pmove,
        "fall": {
            "fall_damagemod": 1.0,
            "deathmatch": True,
            "dmflags": 0,
            "health": 100,
        },
        "authorities": {
            "pmove": {
                "executable_sha256": _sha256_file(pmove_process.binary),
                "tool_identity": pmove_process.identity["tool_identity"],
                "physics_identity": pmove_process.identity["physics_identity"],
                "map_sha256": pmove_process.identity["map_sha256"],
            },
            "fall": {
                "executable_sha256": _sha256_file(fall_process.binary),
                "tool_identity": fall_process.identity["tool_identity"],
                "physics_identity": fall_process.identity["physics_identity"],
            },
        },
    }


def classify_drop_trajectories(
    trajectories: Sequence[DropTrajectory],
    *,
    bsp: Path,
    pmove_process: Any,
    fall_process: FallOracleProcess,
) -> list[dict[str, Any]]:
    """Classify trajectories without launching a process per candidate.

    Both oracle processes are persistent.  Identity and simulation/evaluation
    records are batched.  The result order exactly matches ``trajectories``.
    """

    if not trajectories:
        return []
    manifests = [
        _drop_manifest(
            trajectory, bsp=bsp, pmove_process=pmove_process,
            fall_process=fall_process,
        )
        for trajectory in trajectories
    ]
    pmove_stages = [pmove_requests_for_drop_manifest(item) for item in manifests]
    pmove_indices = [
        index for index, stage in enumerate(pmove_stages)
        if stage["classification"] == "NeedsPmoveAuthority"
    ]
    identity_requests = [
        pmove_stages[index]["identity_request"] for index in pmove_indices
    ]
    pmove_budget = getattr(pmove_process, "max_requests", None)
    if pmove_budget is None:
        pmove_budget = pmove_process.limits.max_oracle_requests
    if pmove_process.requests + len(identity_requests) > pmove_budget:
        raise ExactDropAnalysisError(
            "exact drop Pmove request preflight exceeds oracle budget"
        )
    # Every trajectory reaching Pmove admission may require a fall identity
    # plus an evaluation.  Reserve the worst case before either authority is
    # written so a cap can never produce a partially classified set.
    if fall_process.requests + 2 * len(pmove_indices) > fall_process.max_requests:
        raise ExactDropAnalysisError(
            "exact drop fall request preflight exceeds oracle budget"
        )
    pmove_identities = pmove_process.call(identity_requests)
    prepared: list[dict[str, Any]] = [dict(stage) for stage in pmove_stages]
    identities_by_index: dict[int, Mapping[str, Any]] = {}
    for index, identity in zip(pmove_indices, pmove_identities):
        identities_by_index[index] = identity
        prepared[index] = prepare_drop_fall_request(
            manifests[index],
            pmove_identity=identity,
            pmove_response=trajectories[index].response,
            pmove_executable_sha256=_sha256_file(pmove_process.binary),
            map_sha256=pmove_process.identity["map_sha256"],
        )
    exact_indices = [
        index for index, stage in enumerate(prepared)
        if stage["classification"] == "NeedsFallAuthority"
    ]
    fall_requests: list[Mapping[str, Any]] = []
    for index in exact_indices:
        fall_requests.extend([
            prepared[index]["fall_identity_request"],
            prepared[index]["fall_request"],
        ])
    fall_records = fall_process.call(fall_requests)
    record_cursor = 0
    results: list[dict[str, Any]] = []
    for index, (manifest, trajectory, stage) in enumerate(zip(
        manifests, trajectories, prepared
    )):
        if stage["classification"] == "Unknown":
            exact = stage
        else:
            identity = identities_by_index[index]
            fall_identity = fall_records[record_cursor]
            fall_response = fall_records[record_cursor + 1]
            record_cursor += 2
            exact = evaluate_drop_evidence(
                manifest,
                pmove_identity=identity,
                pmove_response=trajectory.response,
                fall_identity=fall_identity,
                fall_response=fall_response,
                pmove_executable_sha256=_sha256_file(pmove_process.binary),
                fall_executable_sha256=_sha256_file(fall_process.binary),
                map_sha256=pmove_process.identity["map_sha256"],
            )
        is_exact = exact.get("classification") == "Exact"
        pmove_validated = (
            index in identities_by_index
            and exact.get("reason") != "unsupported_dynamic_mover"
        )
        results.append({
            "schema": DROP_CLASSIFICATION_SCHEMA,
            "id": trajectory.identifier,
            "source_l1": list(trajectory.source_l1),
            "direction": list(trajectory.direction),
            "mode": trajectory.mode,
            "classification": exact,
            "evidence": (
                DROP_EVIDENCE_EXACT if is_exact
                else DROP_EVIDENCE_PMOVE if pmove_validated else 0
            ),
            "validation_version": DROP_VALIDATION_VERSION if pmove_validated else 0,
        })
    if record_cursor != len(fall_records):
        raise ExactDropAnalysisError("fall response accounting differs")
    return results


def summarize_drop_classifications(
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_safe = exact_lethal = unknown = 0
    severity: dict[str, int] = {}
    for item in classifications:
        result = item.get("classification")
        if not isinstance(result, Mapping) or result.get("classification") != "Exact":
            unknown += 1
            continue
        name = str(result.get("severity"))
        severity[name] = severity.get(name, 0) + 1
        if result.get("lethal") is True:
            exact_lethal += 1
        else:
            exact_safe += 1
    aggregate_evidence = DROP_EVIDENCE_PMOVE
    for item in classifications:
        evidence = item.get("evidence")
        if type(evidence) is int:
            aggregate_evidence |= evidence
    return {
        "classification_status": "oracle",
        "evidence": aggregate_evidence,
        "validation_version": DROP_VALIDATION_VERSION,
        "candidate_count": len(classifications),
        "exact_safe": exact_safe,
        "exact_lethal": exact_lethal,
        "unknown_omitted": unknown,
        "severity_counts": dict(sorted(severity.items())),
    }


__all__ = [
    "DROP_CLASSIFICATION_SCHEMA",
    "DROP_EVIDENCE_EXACT",
    "DROP_VALIDATION_VERSION",
    "DropTrajectory",
    "ExactDropAnalysisError",
    "FallOracleProcess",
    "classify_drop_trajectories",
    "summarize_drop_classifications",
]
