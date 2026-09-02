# Pipelined network collection — Q2_PIPELINED_COLLECT — 2026-07-24

Status: implemented, opt-in. Default off (serial `collect_round` unchanged).

## What it is

`Q2_PIPELINED_COLLECT=1` makes the network-native rollout loop in
`train/ppo.py` collect through
`Q2NetworkClientBatch.collect_rounds_pipelined` instead of calling
`collect_round` once per rollout step. The batch driver owns the round
cycle: validate echoes → minimal stateful collate → infer → dispatch the
next round → assemble and emit the PREVIOUS round's BatchRound while the
new echoes are in flight. The trainer's inference (`policy.act_batch`),
fire-gate reconciliation, posture/behavior telemetry, and rollout-buffer
writes are callbacks wired in by ppo.py (`harness/pipelined_trainer.py`,
torch-free and unit-tested on CPU).

## Semantics (what is preserved)

- Round content is identical to serial: same accepted rounds, rewards,
  feature vectors, infos, tags, and admission accounting
  (`failed_rounds`, `echo_timeouts`, `telemetry_gap_resyncs`,
  `realtime_catchup_resyncs`) for the same action sequence. Evidence:
  `tools/collect_equivalence.py` over a 300-round recorded live session
  (serial vs pipelined, byte-identical BatchRounds, run again under
  `Q2_NEAREST_VERIFY=1`), and `tests/test_pipelined_collect.py` /
  `tests/test_pipelined_trainer.py` (scripted serial-reference comparison
  including a death round and a realtime-catchup boundary).
- Fail-closed admission is unchanged: stale/mismatched echo rejection,
  map-epoch and telemetry-gap barriers, and policy-version monotonicity go
  through the exact serial code paths.
- No added policy latency: `infer` always receives the freshest collated
  vectors (post-death reset vectors included, via the driver's
  `after_collate` hook — the same vectors `reset_slot` produces serially).
- Determinism: inference happens in the same order with the same inputs as
  serial; only pure BatchRound assembly and trainer buffer writes move
  (they overlap the next echo wait). No RNG is consumed on the deferred
  path, so seeded runs are reproducible identically when the flag is on.
  With the flag off nothing changes at all.

## What intentionally differs

- Exactly one extra inference + dispatched round per rollout: the
  in-flight tail round is discarded when the rollout buffer fills
  (`should_stop`). Its spatial deposits come from real gameplay and its
  episode bookkeeping is absorbed by the synchronization boundary that
  opens the next rollout (server runs free during the PPO update →
  catchup boundary → `initial_result` resets episode state).
- Metrics for an accepted round post one iteration later than serial;
  totals match after the tail flush.

## When to enable

Enable on lanes where the serial cycle overruns the tick — 2x timescale
(50 ms tick) and above, where drain→dispatch→echo-wait→post-processing→
inference must fit inside one inter-tick interval. On the nobara ladder
(4 clients, 4096-cell lattice, stub inference) the pipelined driver
sustained ~78 accepted t/s at 2x and ~83 t/s at 4x (vs ~30 t/s at 1x),
zero echo timeouts. With 30 ms inference, 4x livelocks in BOTH modes —
the cap is inference latency vs the tick, which pipelining cannot hide.

Requires: single network-native server (the production topology),
`Q2_FAST_NEAREST` recommended (the post-processing path must stay well
under the tick).
