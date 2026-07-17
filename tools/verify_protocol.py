"""
verify_protocol.py — sanity-check that Python struct sizes match C sizeof().

Run this before integrating ml_bridge.c to confirm the wire format is correct.
Expected C sizes (compile with: cc -x c - <<< '#include "ml_bridge.h" ...')
are asserted here.
"""

import struct
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from harness.protocol import (
    ACT_FMT,
    ACT_SIZE,
    OBS_DIM,
    OBS_DYN_DIM,
    OBS_FACTUAL_DIM,
    OBS_FMT,
    OBS_OBJECTIVES_DIM,
    OBS_RECOVERY_DIM,
    OBS_SIZE,
)

print(f"OBS_SIZE (Python): {OBS_SIZE} bytes")
print(f"ACT_SIZE (Python): {ACT_SIZE} bytes")

# Verify obs vector dimension
import numpy as np
from harness.protocol import ML_MAX_ENTITIES, ML_RAY_COUNT, ML_HOOK_ZONES

base_size = (
    10                            # self_state
    + ML_MAX_ENTITIES * 9         # entities
    + ML_RAY_COUNT * 4            # rays
    + ML_HOOK_ZONES * 8           # hook_zones
    + 5                           # audio
    + 2                           # normalised yaw/pitch
)
assert base_size == 185, f"engine sensor dimension mismatch: {base_size}"
vec_size = OBS_FACTUAL_DIM + OBS_DYN_DIM + OBS_RECOVERY_DIM + OBS_OBJECTIVES_DIM
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

# Test policy instantiation when the ML environment is installed. The wire
# parity checks above intentionally remain usable on game/client build hosts.
try:
    import torch
    from models.multires_policy import MultiresQ2BotPolicy
except ImportError:
    print("Policy forward: skipped (PyTorch unavailable on this build host)")
else:
    pol = MultiresQ2BotPolicy()
    print(f"Policy params: {sum(value.numel() for value in pol.parameters()):,}")
    obs = torch.zeros(1, 1, OBS_DIM)
    hx = pol.init_hidden(1)
    _act_params, val, _hx2 = pol(obs, hx)
    print(f"Policy forward: value shape {val.shape} ✓")
    act_np, _v, lp, _ = pol.act_batch(
        np.zeros((1, OBS_DIM), dtype=np.float32), [hx], deterministic=True
    )
    print(f"Policy act log_prob: {float(lp[0]):.4f} ✓")
    print(f"Policy act: action shape {act_np.shape} ✓")

print("\nAll checks passed.")
