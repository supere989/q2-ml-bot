#!/usr/bin/env python3
"""Materialize six compiled, source-bound hook-v2 claims atomically."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_analyzer import (  # noqa: E402
    AnalyzerLimits,
    AtlasAnalysisError,
    FallOracleProcess,
    HookOracleProcess,
    OracleProcess,
    _admit_hook,
    _build_navigation,
    _directed_adjacency,
    _grid_index,
    _oracle_atlas_origin,
    _origin,
    _replay_hook_record_exact,
    canonical_json,
    sha256_file,
)
from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    admit_b1_runtime_authorities,
)
from harness.hook_claims_v2 import (  # noqa: E402
    HookClaimsV2Error,
    MATERIALIZATION_SCHEMA,
    RUNTIME_RECORD_COUNT,
    canonical_json as hook_canonical_json,
    load_candidates,
    load_materialization,
    render_runtime_sidecar,
    runtime_records_sha256,
    sha256_bytes,
    validate_materialization,
    validate_runtime_sidecar,
)
from harness.ibsp38 import parse_ibsp38  # noqa: E402


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _reachable_nodes(edges: list[dict], spawn_indices: dict) -> set[tuple[int, int, int]]:
    adjacency = _directed_adjacency(edges)
    reachable = set(spawn_indices.values())
    pending = list(reachable)
    while pending:
        current = pending.pop()
        for neighbor in adjacency.get(current, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)
    return reachable


def materialize(
    *,
    bsp: Path,
    meta: Path,
    runtime_sidecar: Path,
    output_attestation: Path,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_parity_attestation: Path,
    limits: AnalyzerLimits,
) -> dict:
    bsp = bsp.resolve()
    meta = meta.resolve()
    runtime_sidecar = runtime_sidecar.resolve()
    output_attestation = output_attestation.resolve()
    if not all(path.is_file() for path in (
        bsp, meta, runtime_sidecar, cm_oracle, pmove_oracle,
        hook_oracle, fall_oracle, hook_parity_attestation,
    )):
        raise AtlasAnalysisError("hook materialization input is missing")
    try:
        admit_b1_runtime_authorities(
            cm_oracle=cm_oracle, pmove_oracle=pmove_oracle,
            hook_oracle=hook_oracle, fall_oracle=fall_oracle,
            hook_parity_attestation=hook_parity_attestation,
            analysis_bsp=bsp, repo_root=ROOT,
        )
    except B1AuthorityError as error:
        raise AtlasAnalysisError(
            f"hook materialization B1 authority admission failed: {error}"
        ) from error
    metadata = parse_ibsp38(bsp)
    map_id = bsp.stem
    if meta.name != f"{map_id}.meta.json" or runtime_sidecar.name != f"{map_id}.json":
        raise AtlasAnalysisError("hook materialization filenames do not share the BSP map ID")
    candidates, candidates_sha256, meta_sha256 = load_candidates(meta)
    initial_bsp_sha256 = metadata.sha256
    initial_bsp_size = bsp.stat().st_size
    source_projection_sha256 = sha256_file(runtime_sidecar)
    source_projection = runtime_sidecar.read_bytes()
    hook_admission = _admit_hook(hook_oracle, hook_parity_attestation)
    if hook_admission.get("authority_admitted") is not True:
        raise AtlasAnalysisError("hook oracle/parity authority is absent")
    if b"# bundle_admissible: true" in source_projection:
        if not output_attestation.is_file():
            raise AtlasAnalysisError("admissible hook sidecar lacks its materialization")
        document, digest = load_materialization(output_attestation)
        if (
            document["map"] != map_id
            or document["bsp"] != {
                "sha256": initial_bsp_sha256, "size_bytes": initial_bsp_size,
            }
            or document["candidates"]["meta_sha256"] != meta_sha256
            or document["candidates"]["records_sha256"] != candidates_sha256
            or document["candidates"]["record_count"] != len(candidates["records"])
            or document["oracles"]["hook_parity_attestation_sha256"]
            != sha256_file(hook_parity_attestation)
        ):
            raise AtlasAnalysisError("existing hook materialization identities changed")
        validate_runtime_sidecar(
            source_projection, map_id=map_id, bsp_sha256=initial_bsp_sha256,
            materialization_sha256=digest, records=document["selected_records"],
        )
        expected_oracles = document["oracles"]
        for name, binary in (
            ("collision", cm_oracle), ("pmove", pmove_oracle), ("hook", hook_oracle),
        ):
            if sha256_file(binary) != expected_oracles[name]["executable_sha256"]:
                raise AtlasAnalysisError(
                    f"existing hook materialization {name} executable changed"
                )
        with OracleProcess(cm_oracle, bsp, "cm", limits) as current_cm:
            with OracleProcess(pmove_oracle, bsp, "pmove", limits) as current_pmove:
                with HookOracleProcess(
                    hook_oracle, str(hook_admission["physics_identity"]), limits,
                ) as current_hook:
                    for name, identity in (
                        ("collision", current_cm.identity),
                        ("pmove", current_pmove.identity),
                        ("hook", current_hook.identity),
                    ):
                        if any(
                            identity.get(field) != expected_oracles[name][field]
                            for field in ("tool_identity", "physics_identity")
                        ):
                            raise AtlasAnalysisError(
                                f"existing hook materialization {name} identity changed"
                            )
        return document
    if b"# bundle_admissible: false" not in source_projection:
        raise AtlasAnalysisError("source hook projection is not explicitly non-admissible")

    spawns = [
        (entity.index, tuple(axis + (9.0 if index == 2 else 0.0) for index, axis in enumerate(_origin(entity))))
        for entity in metadata.entities if entity.classname == "info_player_deathmatch"
    ]
    candidate_points = []
    for record in candidates["records"]:
        candidate_points.extend([
            tuple(value / 1000.0 for value in record["source_milliunits"]),
            tuple(value / 1000.0 for value in record["landing_milliunits"]),
        ])
    candidate_points = list(dict.fromkeys(candidate_points))

    selected: list[dict] = []
    selected_traces: list[dict] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    with OracleProcess(cm_oracle, bsp, "cm", limits) as cm:
        if cm.identity["map_sha256"] != initial_bsp_sha256:
            raise AtlasAnalysisError("collision oracle loaded different BSP bytes")
        origin = _oracle_atlas_origin(metadata, cm.identity)
        with OracleProcess(pmove_oracle, bsp, "pmove", limits) as pmove:
            if pmove.identity["map_sha256"] != initial_bsp_sha256:
                raise AtlasAnalysisError("Pmove oracle loaded different BSP bytes")
            with FallOracleProcess(
                fall_oracle,
                max_requests=limits.max_oracle_requests,
                batch_size=limits.oracle_batch,
                batch_timeout_seconds=limits.oracle_batch_timeout_seconds,
                exit_timeout_seconds=limits.process_exit_timeout_seconds,
            ) as fall, HookOracleProcess(
                hook_oracle, str(hook_admission["physics_identity"]), limits,
            ) as hook:
                nodes, edges, spawn_indices, _drop_classifications = _build_navigation(
                    cm, pmove, fall, bsp, spawns, origin, limits, candidate_points,
                )
                reachable = _reachable_nodes(edges, spawn_indices)
                geometries: set[tuple] = set()
                for index, candidate in enumerate(candidates["records"]):
                    source = tuple(
                        value / 1000.0 for value in candidate["source_milliunits"]
                    )
                    source_key = _grid_index(source, origin, 16)
                    if source_key not in reachable or source_key not in nodes:
                        rejection_counts["source_not_spawn_reachable"] += 1
                        continue
                    measured, reason, trace = _replay_hook_record_exact(
                        cm, pmove, hook, candidate,
                        request_prefix=f"preflight-hook:{index:04d}",
                        atlas_origin=origin,
                        expected_landing=False,
                    )
                    if measured is None:
                        rejection_counts[str(reason)] += 1
                        continue
                    landing = tuple(
                        value / 1000.0 for value in measured["landing_milliunits"]
                    )
                    target_key = _grid_index(landing, origin, 16)
                    target = nodes.get(target_key)
                    if target is None or not target.supported or not (
                        target.standing_clear or target.crouched_clear
                    ):
                        rejection_counts["measured_landing_l1_unavailable"] += 1
                        continue
                    geometry = (
                        tuple(measured["anchor_milliunits"]),
                        tuple(measured["landing_milliunits"]),
                        measured["flags"],
                    )
                    if geometry in geometries:
                        rejection_counts["duplicate_proven_geometry"] += 1
                        continue
                    geometries.add(geometry)
                    selected.append(measured)
                    assert trace is not None
                    selected_traces.append(trace)
                    if len(selected) == RUNTIME_RECORD_COUNT:
                        break
                oracle_records = {
                    "collision": {
                        "executable_sha256": sha256_file(cm.binary),
                        "tool_identity": cm.identity["tool_identity"],
                        "physics_identity": cm.identity["physics_identity"],
                        "requests": cm.requests,
                    },
                    "pmove": {
                        "executable_sha256": sha256_file(pmove.binary),
                        "tool_identity": pmove.identity["tool_identity"],
                        "physics_identity": pmove.identity["physics_identity"],
                        "requests": pmove.requests,
                    },
                    "hook": {
                        "executable_sha256": sha256_file(hook.binary),
                        "tool_identity": hook.identity["tool_identity"],
                        "physics_identity": hook.identity["physics_identity"],
                        "requests": hook.requests,
                    },
                    "hook_parity_attestation_sha256": sha256_file(
                        hook_parity_attestation
                    ),
                }
    if len(selected) != RUNTIME_RECORD_COUNT:
        raise AtlasAnalysisError(
            "compiled hook preflight proved "
            f"{len(selected)}/{RUNTIME_RECORD_COUNT} unique geometries; "
            f"rejections={dict(sorted(rejection_counts.items()))}"
        )
    # Refuse a time-of-check/time-of-use substitution before publishing either
    # admissible artifact. A stale attestation alone remains fail-closed.
    if (
        sha256_file(bsp) != initial_bsp_sha256
        or bsp.stat().st_size != initial_bsp_size
        or sha256_file(meta) != meta_sha256
        or sha256_file(runtime_sidecar) != source_projection_sha256
        or sha256_file(hook_parity_attestation)
        != oracle_records["hook_parity_attestation_sha256"]
    ):
        raise AtlasAnalysisError("hook materialization input changed during replay")
    document = {
        "schema": MATERIALIZATION_SCHEMA,
        "map": map_id,
        "passed": True,
        "bsp": {"sha256": initial_bsp_sha256, "size_bytes": initial_bsp_size},
        "candidates": {
            "meta_sha256": meta_sha256,
            "records_sha256": candidates_sha256,
            "record_count": len(candidates["records"]),
        },
        "source_projection_sha256": source_projection_sha256,
        "runtime_records_sha256": runtime_records_sha256(selected),
        "selected_records": selected,
        "validation_traces": selected_traces,
        "oracles": oracle_records,
        "replay": {
            "analyzer": "q2-hook-claim-materializer",
            "analyzer_version": "b2-c-v2",
            "verifier": "q2-atlas-analyzer-exact-hook-replay",
            "verifier_version": "b2-a-v2",
        },
        "request_count": sum(
            oracle_records[name]["requests"] for name in ("collision", "pmove", "hook")
        ),
    }
    document = validate_materialization(document)
    attestation_bytes = hook_canonical_json(document) + b"\n"
    attestation_sha256 = sha256_bytes(attestation_bytes)
    runtime_bytes = render_runtime_sidecar(
        map_id, initial_bsp_sha256, attestation_sha256, selected,
    )
    _atomic_write(output_attestation, attestation_bytes)
    _atomic_write(runtime_sidecar, runtime_bytes)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsp", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--runtime-sidecar", type=Path, required=True)
    parser.add_argument("--output-attestation", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-parity-attestation", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = materialize(
            bsp=args.bsp, meta=args.meta, runtime_sidecar=args.runtime_sidecar,
            output_attestation=args.output_attestation,
            cm_oracle=args.cm_oracle, pmove_oracle=args.pmove_oracle,
            hook_oracle=args.hook_oracle,
            fall_oracle=args.fall_oracle,
            hook_parity_attestation=args.hook_parity_attestation,
            limits=AnalyzerLimits(),
        )
    except (AtlasAnalysisError, HookClaimsV2Error, OSError, ValueError) as error:
        print(f"hook materialization failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "schema": "q2-hook-materialization-result-v2",
        "map": document["map"], "passed": True,
        "selected_count": len(document["selected_records"]),
        "attestation_sha256": sha256_file(args.output_attestation),
        "runtime_sidecar_sha256": sha256_file(args.runtime_sidecar),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
