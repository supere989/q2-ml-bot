from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT.parent / "q2-ml-client"


def test_real_c_barrier_core_fault_matrix_and_microproof(tmp_path: Path) -> None:
    cc = shutil.which("cc")
    assert cc is not None
    runner = tmp_path / "ml-frame-barrier-runner"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(CLIENT / "tests/ml_frame_barrier_fault_runner.c"),
            str(CLIENT / "src/server/sv_ml_frame_barrier_core.c"),
            "-o",
            str(runner),
        ],
        check=True,
    )
    cold_one = subprocess.check_output([runner], text=True)
    cold_two = subprocess.check_output([runner], text=True)
    assert cold_one == cold_two
    evidence = json.loads(cold_one)
    assert evidence["schema"] == "q2-network-client-frame-barrier-core-v1"
    assert evidence["clients"] == 4
    assert evidence["frames"] == 32
    assert all(
        evidence[name]
        for name in (
            "duplicate",
            "reorder",
            "missing",
            "map_reset",
            "death_reset",
            "future_fault",
            "disconnect_fault",
        )
    )
