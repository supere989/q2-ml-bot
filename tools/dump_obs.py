#!/usr/bin/env python3
"""Bare minimum: receive packets on 27957, write to a log file, send dummy action back."""
import socket, struct, time, sys

PORT = 27957
LOG  = "/tmp/dump_obs.log"

ML_OBS_MAGIC = 0x514D4C4F
ML_ACT_MAGIC = 0x514D4C41
ACT_FMT = "<IIffff4B"

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("127.0.0.1", PORT))
s.settimeout(1.0)

with open(LOG, "w") as f:
    f.write(f"dump_obs started, listening on {PORT}\n")
    f.flush()

    n = 0
    last_print = time.time()
    while True:
        try:
            data, addr = s.recvfrom(4096)
        except socket.timeout:
            continue

        n += 1

        # parse header: magic, tick, bot_slot, then yaw, pitch, then self pos[3]
        if len(data) >= 32:
            magic, tick, slot = struct.unpack("<III", data[:12])
            yaw, pitch         = struct.unpack("<ff",  data[12:20])
            px, py, pz         = struct.unpack("<fff", data[20:32])
        else:
            magic = tick = slot = 0; yaw = pitch = px = py = pz = 0

        # send back zero action
        act = struct.pack(ACT_FMT, ML_ACT_MAGIC, tick,
                          0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
        s.sendto(act, addr)

        if time.time() - last_print > 2.0 or n <= 5:
            f.write(f"[{n:6d}] addr={addr} len={len(data)} "
                    f"magic={magic:#x} tick={tick} slot={slot} "
                    f"pos=({px:.0f},{py:.0f},{pz:.0f})\n")
            f.flush()
            last_print = time.time()
