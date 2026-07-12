#!/usr/bin/env python3
"""Run the synchronous rollout coordinator for a learner generation."""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.rollout_protocol import CoordinatorServer, PolicyArtifact, RolloutCoordinator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=38888)
    parser.add_argument("--quorum", type=int, default=1)
    parser.add_argument("--schema", choices=("ppo", "any"), default="ppo")
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--attestation-key-env", default="")
    parser.add_argument("--require-attestation-signature", action="store_true")
    parser.add_argument("--token", default="")
    parser.add_argument("--spool", type=Path, default=Path("distributed_batches"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config_hash = args.config_hash
    if args.config:
        config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    runtime_digest = ""
    if args.runtime_manifest:
        from harness.runtime_attestation import (
            load_runtime_manifest,
            verify_runtime_manifest,
        )

        manifest = load_runtime_manifest(args.runtime_manifest)
        key = (
            os.environ[args.attestation_key_env].encode()
            if args.attestation_key_env else None
        )
        verification = verify_runtime_manifest(
            manifest,
            hmac_key=key,
            require_signature=args.require_attestation_signature,
        )
        if not verification.valid:
            parser.error("invalid runtime manifest: " + "; ".join(verification.errors))
        runtime_digest = verification.digest
    elif args.schema == "ppo":
        parser.error("--runtime-manifest is required for schema=ppo")
    artifact = PolicyArtifact.create(
        args.version, args.policy.read_bytes(), config_hash, runtime_digest
    )
    coordinator = RolloutCoordinator(
        args.quorum,
        schema=args.schema,
        expected_runtime_manifest_sha256=runtime_digest,
    )
    coordinator.publish(artifact)
    server = CoordinatorServer(
        coordinator, args.bind, args.port, token=args.token
    ).start()
    stopping = False

    def stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({
        "event": "coordinator_started",
        "bind": server.address[0],
        "port": server.address[1],
        **coordinator.status(),
    }, sort_keys=True), flush=True)
    try:
        while not stopping:
            batches = coordinator.wait_for_quorum(artifact.version, args.timeout)
            if not batches:
                print(json.dumps({
                    "event": "quorum_timeout", **coordinator.status()
                }, sort_keys=True), flush=True)
                if args.once:
                    return 2
                continue
            generation_dir = args.spool / f"policy_{artifact.version:08d}"
            generation_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for batch in batches:
                worker = str(batch.metadata["worker_id"]).replace("/", "_")
                sequence = int(batch.metadata["sequence"])
                digest = batch.rollout_hash()
                target = generation_dir / f"{worker}_{sequence:08d}_{digest[:12]}.q2rb"
                target.write_bytes(batch.encode())
                saved.append(str(target))
            print(json.dumps({
                "event": "quorum_ready",
                "policy_version": artifact.version,
                "batches": len(batches),
                "saved": saved,
            }, sort_keys=True), flush=True)
            if args.once:
                return 0
            # A real learner publishes N+1 only after consuming this quorum.
            # Hold N here rather than silently training on repeated generations.
            time.sleep(0.1)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
