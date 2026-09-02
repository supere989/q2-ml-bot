"""Trainer-side glue driving one rollout through the pipelined collector.

``Q2NetworkClientBatch.collect_rounds_pipelined`` owns the round cycle; this
module maps that driver onto the semantics ppo.py's serial collection block
has today:

- per-round inference context (values, log-probs, hidden stacks, fire-gate
  metadata) is stashed by round id at ``infer`` time and consumed when the
  matching BatchRound is emitted one iteration later;
- ``Q2NetworkClientMultiEnv.step_all``'s episode bookkeeping (max_ep_steps
  truncation, ``finalize_episode`` outcome bonuses, per-slot step counters)
  is replicated exactly, at the same logical point;
- done envs get their spatial/hidden reset and the reset observation vector
  BEFORE the next inference, matching the serial timing where ppo.py calls
  ``reset_slot`` right after the done round;
- boundary rounds (map epoch, telemetry gap, realtime catchup, action-state
  resync) reset per-slot episode state and are never buffered.

Torch-free by construction: the policy, the rollout buffer, and hidden-state
handling all enter through callables, so the whole flow is unit-testable on
CPU. train/ppo.py wires the real callables in when Q2_PIPELINED_COLLECT=1.
"""

from __future__ import annotations

import numpy as np

from .client_batch import decode_policy_action


class PipelinedNetworkRollout:
    """Collect one rollout (up to ``n_steps`` trainable rounds) pipelined.

    Callbacks:
      act(obs_np) -> ctx dict. Required keys: ``current_obs``, ``h_step``,
        ``c_step``, ``actions`` (per-env 8-dim action vectors), ``values``,
        ``log_probs``, ``fire_allowed``. Optional: ``fire_metadata``.
      buf_add(ctx, rewards, dones) -> None. Called at most n_steps times.
      reset_hidden(client_index) -> None. Recurrent-state reset for a done
        or boundary client, called before the next inference that uses it.
      on_boundary(infos) -> None. Serial parity: the trainer resets its
        per-venv episode accumulators and logs the barrier.
      on_accepted(ctx, results, rewards, dones) -> None (optional). Called
        before buf_add for trainer-side telemetry/reconciliation. ``results``
        is a list of (obs, reward, terminated, truncated, info) tuples with
        the adapter's outcome adjustments already merged into ``info``.
    """

    def __init__(self, multi_env, *, n_steps: int, policy_version: int):
        self.srv = multi_env
        self.batch = multi_env._batch
        self.n_steps = max(1, int(n_steps))
        self.policy_version = int(policy_version)
        self.accepted_steps = 0
        self._ctx: dict[int, dict] = {}
        self._adjusted: dict[int, tuple] = {}

    def collect(
        self,
        observations,
        *,
        act,
        buf_add,
        reset_hidden,
        on_boundary,
        on_accepted=None,
    ):
        self._act = act
        self._buf_add = buf_add
        self._reset_hidden = reset_hidden
        self._on_boundary = on_boundary
        self._on_accepted = on_accepted
        # The iteration budget is effectively unbounded: boundary rounds
        # consume iterations without producing trainable steps, and a serial
        # loop would also wait out a long map-epoch barrier. should_stop is
        # the only exit that matters.
        return self.batch.collect_rounds_pipelined(
            observations,
            rounds=1_000_000_000,
            infer=self._infer,
            policy_version=self.policy_version,
            on_round=self._on_round,
            should_stop=self._should_stop,
            after_collate=self._after_collate,
        )

    # ── driver hooks ────────────────────────────────────────────────────

    def _should_stop(self) -> bool:
        return self.accepted_steps >= self.n_steps

    def _infer(self, vectors, round_id: int):
        obs_in = np.asarray(vectors, dtype=np.float32)
        ctx = self._act(obs_in)
        self._ctx[round_id] = ctx
        # Boundary iterations leave undispatched contexts behind; keep the
        # stash bounded to recent round ids.
        for stale in [rid for rid in self._ctx if rid < round_id - 8]:
            self._ctx.pop(stale, None)
            self._adjusted.pop(stale, None)
        return [
            decode_policy_action(ctx["actions"][k])
            for k in range(self.srv.n_ml)
        ]

    def _after_collate(self, round_id: int, results):
        """Episode bookkeeping + done resets, before the next inference.

        Mirrors Q2NetworkClientMultiEnv.step_all: per-slot step counters,
        max_ep_steps truncation, finalize_episode outcome bonuses. Then, for
        each done env, the spatial/hidden reset and the reset observation
        vector — matching the serial timing where ppo.py calls reset_slot
        right after the done round's processing.
        """
        rewards = np.asarray(
            [float(result[1]) for result in results], dtype=np.float32
        )
        dones = np.zeros(len(results), dtype=np.float32)
        outcomes: dict[int, tuple] = {}
        replacement = None
        for bi, result in enumerate(results):
            info = result[4]
            terminated = bool(result[2])
            truncated = bool(result[3])
            self.srv._ep_steps[bi] += 1
            if (
                not terminated
                and self.srv._ep_steps[bi] >= self.srv.max_ep_steps
            ):
                truncated = True
            if not (terminated or truncated):
                continue
            outcome_bonus, outcome_info = self.srv._spatial_rewards[
                bi
            ].finalize_episode(
                terminal_reason=int(info.get("terminal_reason", 0)),
                truncated=truncated,
            )
            rewards[bi] += float(outcome_bonus)
            dones[bi] = 1.0
            # The emitted BatchRound keeps the pristine transition_result
            # info (serial parity); the adapter's outcome adjustments are
            # recorded separately and merged only into the on_accepted copy.
            outcomes[bi] = (float(outcome_bonus), outcome_info)
            reset_vector = self.batch.envs[bi].reset_episode_vector()
            if replacement is None:
                replacement = [r[0] for r in results]
            replacement[bi] = np.asarray(reset_vector, dtype=np.float32)
            self._reset_hidden(bi)
        self._adjusted[round_id] = (rewards, dones, outcomes)
        if replacement is None:
            return None
        return self.batch._collate_observations(replacement, self.batch.vector)

    def _on_round(self, round_result) -> None:
        infos = round_result.infos
        trainable_flags = [
            bool(info.get("trainable_transition", False)) for info in infos
        ]
        if not all(trainable_flags):
            if any(trainable_flags):
                raise RuntimeError(
                    "network collector returned a partially trainable "
                    "map-epoch round"
                )
            for bi in range(len(infos)):
                self.srv._ep_steps[bi] = 0
                self._reset_hidden(bi)
            self._on_boundary(infos)
            return

        round_id = round_result.round_id
        for info in infos:
            if int(info.get("policy_version", -1)) != self.policy_version:
                raise RuntimeError(
                    "network collector returned an unversioned or "
                    "non-trainable transition"
                )
        ctx = self._ctx.pop(round_id, None)
        adjusted = self._adjusted.pop(round_id, None)
        if ctx is None or adjusted is None:
            raise RuntimeError(
                f"pipelined round {round_id} arrived without inference "
                "context"
            )
        rewards, dones, outcomes = adjusted
        if self.accepted_steps >= self.n_steps:
            # Rollout buffer is full; the in-flight round is discarded the
            # same way a serial trainer simply stops collecting.
            return
        results = []
        for bi, info in enumerate(infos):
            merged = dict(info)
            if bi in outcomes:
                outcome_bonus, outcome_info = outcomes[bi]
                merged["spatial_bonus"] = (
                    float(merged.get("spatial_bonus", 0.0)) + outcome_bonus
                )
                merged.update(outcome_info)
            results.append((
                round_result.observations[bi],
                round_result.rewards[bi],
                round_result.terminated[bi],
                round_result.truncated[bi],
                merged,
            ))
        if self._on_accepted is not None:
            self._on_accepted(ctx, results, rewards, dones)
        self._buf_add(ctx, rewards, dones)
        self.accepted_steps += 1
