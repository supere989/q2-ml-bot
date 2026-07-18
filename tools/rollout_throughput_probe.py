#!/usr/bin/env python3
"""Collect one instrumented real-q2 batch and submit it to a live coordinator.

This is a cold-start capacity probe, not a replacement for the persistent
worker.  It records setup time separately and defines collection time from the
first policy inference through the final q2 step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.runtime_attestation import load_runtime_manifest, verify_runtime_manifest
from harness.rollout_protocol import CoordinatorClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--attestation-key-env", default="")
    parser.add_argument("--record-out", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--game-seed", type=int, default=1)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--map-name", default="mltrain_00005208")
    parser.add_argument("--n-bots", type=int, default=4)
    parser.add_argument("--n-ml", type=int, default=4)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--timescale", type=float, default=10.0)
    parser.add_argument("--sv-port-base", type=int, required=True)
    parser.add_argument("--ml-port-base", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--lattice-dir", type=Path)
    args = parser.parse_args()

    manifest = load_runtime_manifest(args.runtime_manifest)
    if args.attestation_key_env and args.attestation_key_env not in os.environ:
        parser.error(
            "attestation key environment variable is unset: "
            + args.attestation_key_env
        )
    attestation_key = (
        os.environ[args.attestation_key_env].encode("utf-8")
        if args.attestation_key_env else None
    )
    verification = verify_runtime_manifest(
        manifest,
        hmac_key=attestation_key,
        require_signature=bool(args.attestation_key_env),
    )
    if not verification.valid:
        parser.error("invalid runtime manifest: " + "; ".join(verification.errors))

    # Lazy imports keep the simple gate/reporting path usable without Torch.
    from harness.env import Q2MultiEnv
    from models.policy import Q2BotPolicy
    from tools.rollout_worker import collect_q2_batch

    stats = {
        "overall_started": time.perf_counter(),
        "started_perf": None,
        "started_unix_ns": None,
        "finished_perf": None,
        "finished_unix_ns": None,
        "timeouts": 0,
    }
    original_act_batch = Q2BotPolicy.act_batch
    original_step_all = Q2MultiEnv.step_all

    def measured_act_batch(self, *call_args, **call_kwargs):
        if stats["started_perf"] is None:
            stats["started_unix_ns"] = time.time_ns()
            stats["started_perf"] = time.perf_counter()
        return original_act_batch(self, *call_args, **call_kwargs)

    def measured_step_all(self, *call_args, **call_kwargs):
        results = original_step_all(self, *call_args, **call_kwargs)
        stats["timeouts"] += sum(
            bool(result[4].get("timeout", False)) for result in results
        )
        stats["finished_perf"] = time.perf_counter()
        stats["finished_unix_ns"] = time.time_ns()
        return results

    Q2BotPolicy.act_batch = measured_act_batch
    Q2MultiEnv.step_all = measured_step_all
    artifact = CoordinatorClient(args.coordinator, token=args.token).fetch_policy()
    worker_args = SimpleNamespace(
        deterministic=bool(args.deterministic),
        seed=args.seed,
        device=args.device,
        sv_port_base=args.sv_port_base,
        ml_port_base=args.ml_port_base,
        map_name=args.map_name,
        game_seed=args.game_seed,
        n_bots=args.n_bots,
        n_ml=args.n_ml,
        max_ep_steps=args.max_ep_steps,
        timescale=args.timescale,
        steps=args.steps,
        lattice_dir=args.lattice_dir,
        worker_id=args.worker_id,
        sequence=args.sequence,
        rollout_index=args.rollout_index,
        deterministic_actions=args.deterministic_actions,
        runtime_manifest=args.runtime_manifest,
        attestation_key_env=args.attestation_key_env,
    )
    try:
        batch = collect_q2_batch(artifact, worker_args)
    finally:
        Q2BotPolicy.act_batch = original_act_batch
        Q2MultiEnv.step_all = original_step_all

    if stats["started_perf"] is None or stats["finished_perf"] is None:
        raise RuntimeError("rollout collector produced no measured q2 steps")
    elapsed = float(stats["finished_perf"] - stats["started_perf"])
    transitions = int(args.steps * batch.metadata["n_envs"])
    collection = {
        "started_unix_ns": int(stats["started_unix_ns"]),
        "finished_unix_ns": int(stats["finished_unix_ns"]),
        "elapsed_seconds": elapsed,
        "setup_seconds": float(stats["started_perf"] - stats["overall_started"]),
        "transitions": transitions,
        "timeouts": int(stats["timeouts"]),
        "rollout_sps": transitions / elapsed,
    }
    if batch.metadata.get("runtime_manifest_sha256") != verification.digest:
        raise RuntimeError(
            "collector runtime digest does not match the probe manifest"
        )
    batch.metadata["collection"] = collection
    client = CoordinatorClient(args.coordinator, token=args.token)
    decision = client.submit(batch)
    record = {
        **batch.metadata,
        "rollout_hash": batch.rollout_hash(),
        "submission": decision.as_dict(),
    }
    args.record_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.record_out.with_name(args.record_out.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.record_out)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if decision.accepted or decision.status == "duplicate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
