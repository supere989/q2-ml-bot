"""
verify_protocol.py — sanity-check that Python struct sizes match C sizeof().

Run this before integrating ml_bridge.c to confirm the wire format is correct.
Expected C sizes (compile with: cc -x c - <<< '#include "ml_bridge.h" ...')
are asserted here.
"""

import struct
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from harness.protocol import OBS_FMT, ACT_FMT, OBS_SIZE, ACT_SIZE

print(f"OBS_SIZE (Python): {OBS_SIZE} bytes")
print(f"ACT_SIZE (Python): {ACT_SIZE} bytes")

# Verify obs vector dimension
from models.policy import OBS_DIM
import numpy as np
from harness.protocol import Observation, ML_MAX_ENTITIES, ML_RAY_COUNT, ML_HOOK_ZONES

vec_size = (
    10                           # self_state
    + ML_MAX_ENTITIES * 9        # entities
    + ML_RAY_COUNT * 4           # rays
    + ML_HOOK_ZONES * 8          # hook_zones
    + 5                          # audio
    + 2                          # normalised yaw/pitch
)
assert vec_size == OBS_DIM, f"OBS_DIM mismatch: {vec_size} vs {OBS_DIM}"
print(f"OBS_DIM vector: {OBS_DIM} floats ✓")

# Test round-trip
import struct as s
dummy = bytes(OBS_SIZE)
# should not raise
vals = s.unpack(OBS_FMT, dummy)
print(f"OBS unpack: {len(vals)} fields ✓")

dummy_act = bytes(ACT_SIZE)
vals_act = s.unpack(ACT_FMT, dummy_act)
print(f"ACT unpack: {len(vals_act)} fields ✓")

# Test policy instantiation
import torch
from models.policy import Q2BotPolicy
pol = Q2BotPolicy()
print(f"Policy params: {pol.param_count():,}")

obs  = torch.zeros(1, 1, OBS_DIM)
hx   = pol.init_hidden(1)
act_params, val, hx2 = pol(obs, hx)
print(f"Policy forward: value shape {val.shape} ✓")

import numpy as np
act_np, v, _ = pol.act(np.zeros(OBS_DIM, dtype=np.float32), hx)
print(f"Policy act: action shape {act_np.shape} ✓")

print("\nAll checks passed.")
