from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from train.multires_live import MultiresLiveTrainer
from harness.multires_admission import SPATIAL_ATTESTATION_INFO_KEY


class _PPO:
    def __init__(self):
        self.updates = 0

    def update(self, _batch):
        self.updates += 1
        return {"optimizer_steps": 1}


def _trainer(*, snapshots, observer):
    trainer = object.__new__(MultiresLiveTrainer)
    infos = tuple(
        tuple({
            "client_id": f"client-{client}",
            "map": "q2dm1",
            "server_frame": 20 + round_index,
            SPATIAL_ATTESTATION_INFO_KEY: {"map_epoch": 4},
        } for client in range(4))
        for round_index in range(2)
    )
    rollout = SimpleNamespace(
        valid=np.ones((4, 2), dtype=np.bool_), infos=infos
    )
    trainer.collector = SimpleNamespace(
        collect=lambda **_kwargs: rollout,
    )
    trainer.drain_runtime_snapshots = lambda: snapshots
    trainer.runtime_snapshot_observer = observer
    trainer.ppo = _PPO()
    return trainer


def _runtime_snapshot(frame):
    return {
        "source_identity": {
            "client_id": tuple(f"client-{client}" for client in range(4)),
            "map_name": ("q2dm1",) * 4,
            "map_epoch": (4,) * 4,
            "server_frame": (frame,) * 4,
        }
    }


def test_runtime_snapshot_cardinality_fails_before_ppo_mutation():
    trainer = _trainer(snapshots=({},), observer=lambda **_value: None)
    with pytest.raises(RuntimeError, match="count differs before PPO"):
        trainer.train_update(policy_version=0)
    assert trainer.ppo.updates == 0


def test_runtime_snapshot_rejection_fails_before_ppo_mutation():
    def reject(**_value):
        raise RuntimeError("runtime telemetry rejected")

    trainer = _trainer(
        snapshots=(_runtime_snapshot(20), _runtime_snapshot(21)),
        observer=reject,
    )
    with pytest.raises(RuntimeError, match="runtime telemetry rejected"):
        trainer.train_update(policy_version=0)
    assert trainer.ppo.updates == 0


def test_runtime_snapshot_source_must_match_admitted_rollout_before_observer():
    observed = []
    rebound = _runtime_snapshot(21)
    rebound["source_identity"] = dict(rebound["source_identity"])
    rebound["source_identity"]["client_id"] = (
        "client-0", "client-1", "client-2", "wrong-client"
    )
    trainer = _trainer(
        snapshots=(_runtime_snapshot(20), rebound),
        observer=lambda **value: observed.append(value),
    )
    with pytest.raises(RuntimeError, match="source differs from admitted rollout"):
        trainer.train_update(policy_version=0)
    assert observed == []
    assert trainer.ppo.updates == 0
