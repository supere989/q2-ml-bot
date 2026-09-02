"""Authoritative fire-gate reconciliation for collected rollouts.

Torch-free so rollout workers and the trainer share one implementation
(train/ppo.py re-exports this as ``_reconcile_server_fire_suppressions``;
tools/rollout_worker.py uses it on the network-native collection path).
"""

from __future__ import annotations

import numpy as np


def reconcile_server_fire_suppressions(
    actions: np.ndarray,
    log_probs: np.ndarray,
    fire_allowed: np.ndarray,
    fire_metadata: dict | None,
    step_results,
    *,
    n_ml: int,
) -> int:
    """Replace server-suppressed fire with its exact hard-mask likelihood.

    The network server may invalidate a shot after collection-time inference
    because protection or target state changed. The applied action is no-fire,
    whose log-probability under the resulting closed gate is zero. Remove the
    sampled fire log-probability from the recorded joint likelihood and store
    the closed mask so every later PPO evaluation uses that same distribution.
    """
    suppressions = 0
    for server_index, results in step_results:
        base = server_index * n_ml
        for bot_index, (_obs, _reward, _term, _trunc, info) in enumerate(results):
            if not info.get("fire_gate_suppressed", False):
                continue
            vector_index = base + bot_index
            if (
                fire_metadata is None
                or actions[vector_index, 5] <= 0.5
                or not fire_allowed[vector_index]
            ):
                raise RuntimeError(
                    "server suppressed fire outside the recorded network "
                    "target-gate distribution"
                )
            fire_log_probability = float(
                fire_metadata["raw_fire_log_probability"][vector_index]
            )
            if not np.isfinite(fire_log_probability):
                raise RuntimeError(
                    "server-suppressed fire has a non-finite behavior "
                    "log-probability"
                )
            log_probs[vector_index] -= fire_log_probability
            actions[vector_index, 5] = 0.0
            fire_allowed[vector_index] = False
            suppressions += 1
    return suppressions
