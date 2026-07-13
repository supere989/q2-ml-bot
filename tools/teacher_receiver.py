#!/usr/bin/env python3
"""Receive passive live 3ZB2 demonstrations and write atomic training batches."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.spatial import VoxelSpatialReward
from harness.teacher_protocol import TEACHER_PACKET_SIZE, parse_teacher_sample


def _flush(output_dir: Path, batch_id: int, rows: dict[str, list]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"teacher_{batch_id:012d}.npz"
    temporary = output_dir / f".{final.name}.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            obs=np.stack(rows["obs"]).astype(np.float32),
            actions=np.stack(rows["actions"]).astype(np.float32),
            sequence=np.asarray(rows["sequence"], dtype=np.uint32),
            ticks=np.asarray(rows["ticks"], dtype=np.uint32),
            slots=np.asarray(rows["slots"], dtype=np.uint16),
            maps=np.asarray(rows["maps"], dtype="U32"),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="100.86.206.50")
    parser.add_argument("--port", type=int, default=32511)
    parser.add_argument("--source", default="100.101.57.114")
    parser.add_argument("--output_dir", default="data/live_3zb2")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--flush_seconds", type=float, default=120.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)
    print(f"teacher_receiver={args.bind}:{args.port} source={args.source} packet={TEACHER_PACKET_SIZE}", flush=True)

    rows = {key: [] for key in ("obs", "actions", "sequence", "ticks", "slots", "maps")}
    memories: dict[tuple[str, int], VoxelSpatialReward] = {}
    last_sequence = 0
    accepted = rejected = lost = 0
    batch_id = int(time.time() * 1000)
    last_flush = last_report = time.monotonic()
    while True:
        try:
            data, address = sock.recvfrom(TEACHER_PACKET_SIZE + 64)
        except socket.timeout:
            data = b""
            address = ("", 0)
        now = time.monotonic()
        if data:
            sample = parse_teacher_sample(data) if address[0] == args.source else None
            if sample is None:
                rejected += 1
            else:
                # q2ded owns the sender sequence, so a clean server restart
                # returns it to one. Treat only a small sequence after a
                # well-established stream as a new sender epoch; ordinary
                # reordering is still rejected.
                if sample.sequence <= last_sequence:
                    if sample.sequence <= 64 and last_sequence > 256:
                        print(f"sender restart: sequence {last_sequence} -> "
                              f"{sample.sequence}; resetting spatial memories", flush=True)
                        last_sequence = 0
                        memories.clear()
                    else:
                        rejected += 1
                        continue
                if last_sequence:
                    lost += max(0, sample.sequence - last_sequence - 1)
                last_sequence = sample.sequence
                key = (sample.map_name, sample.bot_slot)
                memory = memories.get(key)
                if memory is None:
                    memory = VoxelSpatialReward.from_env(seed=sample.bot_slot)
                    memory.reset(sample.map_name, sample.observation)
                    memories[key] = memory
                features = memory.memory_features(sample.observation)
                rows["obs"].append(sample.observation.to_vector(features))
                rows["actions"].append(sample.action)
                rows["sequence"].append(sample.sequence)
                rows["ticks"].append(sample.tick)
                rows["slots"].append(sample.bot_slot)
                rows["maps"].append(sample.map_name)
                memory.update(sample.observation)
                accepted += 1

        should_flush = len(rows["obs"]) >= args.batch_size or (
            rows["obs"] and now - last_flush >= args.flush_seconds
        )
        if should_flush:
            path = _flush(output_dir, batch_id, rows)
            print(f"batch={path} rows={len(rows['obs'])} accepted={accepted} rejected={rejected} lost={lost}", flush=True)
            batch_id += 1
            rows = {key: [] for key in rows}
            last_flush = now
        if now - last_report >= 10.0:
            print(f"live accepted={accepted} buffered={len(rows['obs'])} rejected={rejected} lost={lost}", flush=True)
            last_report = now


if __name__ == "__main__":
    raise SystemExit(main())
