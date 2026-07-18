# B5 no-update pretraining validation and gate

This is the fail-closed boundary between the assembled multires runtime and a
first training update. It does not train, warm start, modify a checkpoint, or
authorize B6. It proves that the exact candidate runtime can execute all
pretraining evaluators deterministically and can pass the existing production
500-transition proof while model and optimizer bytes remain unchanged.

The producer is `tools/run_multires_pretraining_validation.py`. The independent
consumer is `tools/assemble_b5_gate.py`. A producer report alone is not a B5
gate.

## Required campaign matrix

The producer owns this exact order and runs two fresh same-seed replicates of
every row:

| Campaign | Required evidence |
|---|---|
| `guide_on` | Producer-derived non-zero guides and evaluator attempts |
| `guide_off` | Same scenario as guide-on, zero policy guide samples, explicit dropout coverage |
| `hazard_hook` | Hazard and hook fixtures, causal reward replay, no rate-reward or necessity-label violations |
| `posture_water_crouch` | Complete posture, water, and crouch fixtures with exact vertical/action echo |
| `aim_combat_holdout` | Visible/actionable targeting, permitted versus executed fire, hit chain/kill counters, yaw and pitch error |

The two replicates of a campaign must have identical trajectory digests and
identical result objects. Guide-on and guide-off must additionally bind the same
scenario identity. Performance counters are observations at this boundary, not
post-training thresholds; missing evaluator coverage or any safety violation is
still fatal.

Campaigns may run concurrently (`--jobs`, maximum 10). Concurrency changes only
wall-clock scheduling. The schedule, seeds, inputs, evidence membership, and
canonical result digests remain fixed.

## Campaign runner protocol

The producer defaults to the repository executable
`tools/run_multires_pretraining_campaign.py`. It accepts the following
producer-owned arguments:

```text
--protocol q2-multires-pretraining-campaign-v1
--campaign MODE
--replicate 0|1
--seed N --game-seed N --transition-count N
--repo-commit SHA1 --repo-tree SHA1
--runtime-manifest PATH --checkpoint PATH
--training-manifest PATH --bundle-manifest PATH --atlas-bin PATH
--q2ded PATH --client-binary PATH --runtime-root PATH --objectives PATH
--no-update --output PATH
```

The repository runner calls the exact `train.multires_one_run.preflight`,
loads the admitted random step-zero checkpoint, constructs the admitted Rust
Atlas/Dyn provider, and obtains every result from the real policy, action
decoder/fire gate, causal reward reducer, and `MultiresSeasonMetrics` report.
It does not accept result booleans, counters, or an alternate context builder.
It exclusively creates the
requested output as canonical JSON with schema
`q2-multires-pretraining-campaign-v1`. The validator rejects missing or extra
fields, malformed or placeholder digests, wrong source/runtime bindings,
non-zero instrumented optimizer-step or parameter-gradient counters,
partial/stale/resync admission,
non-finite metrics, a disconnected season/result pair, and a stale or forged
`result_sha256`.

Offline evaluator frames are constructed fixtures, not public-conduit
datagrams. Their generic season JSON therefore does not count as a teacher-byte
measurement. B5 instead requires the complete sibling document set of a green,
sealed `B4-EVIDENCE.json`, revalidates its real public-datagram accounting and
exact teacher-packet negative probe, and binds that upstream proof to the same
Atlas, runtime-manifest identity, and bot source commit/tree.
Each offline season report labels its privilege scope as
`not-measured-offline-no-public-conduit` and names that sealed B4 proof as the
required upstream evidence; it does not publish a synthetic zero violation
counter.

The optimizer has no standalone B5 artifact. Its authoritative state is the
optimizer envelope embedded in the attested checkpoint. Production admission
requires that loaded state to be fresh-empty at training step zero; each
campaign records canonical loaded policy and optimizer state digests before
and after evaluation, and the validator requires one unchanged identity across
all ten executions. During each evaluator, `optimizer.step` is wrapped and
every trainable policy parameter has a gradient hook; both measured counters
must remain zero. The suite-level no-update record is derived from the ten
validated campaign records rather than populated independently. Hashing an
unrelated optimizer file is not admissible.

`result_sha256` is the SHA-256 of canonical JSON for:

```json
{"domain":"q2-multires-pretraining-campaign-result-v1","evidence":EVIDENCE_WITHOUT_RESULT_SHA256}
```

with sorted keys, compact separators, ASCII encoding, no NaN/Infinity, and one
trailing newline.

The CLI pins this repository runner; there is no external-runner or fallback
selector at the B5 gate.

## Exact 500-transition proof

After all ten campaign executions pass, the producer invokes the repository
file `tools/run_multires_500_transition_proof.py` with:

```text
--mode production
--seed SEED
--game_seed GAME_SEED
--divergence_game_seed GAME_SEED_PLUS_ONE
--transition_count 500
--out WORK_DIR/proof-500.json
```

The suite owns the proof command completely. It passes the exact B5 artifact
paths, fixed policy/map epoch, seeds, transition count, work directory, and the
current repository `train/multires_one_run.py` through the current Python
interpreter. There is no public proof-argument or trainer-command injection
surface. Only the three explicit operational locations (`q2ded`, client binary,
and runtime root) are supplied by the operator; the one-run preflight binds
their bytes and runtime closure before collection.

The proof must be production-admissible, contain exactly four clients and 125
transitions per client, reproduce the same-seed trajectory, diverge with the
different game seed, have no partial/stale/resync admissions or teardown
orphans, and bind the same runtime manifest, Atlas, checkpoint, and training
manifest as the campaigns.

## Producer invocation

The repository must be clean. Every input must be a non-empty regular file and
not a symlink. `--b3-gate` names the exact active-authority predecessor and
`--b4-evidence` names the aggregate in a complete B4 output
directory; all four sibling evidence documents are mandatory. `--runtime-manifest`
names the admitted compact B4 runtime-evidence JSON:
the producer validates its complete multires protocol contract and records both
the exact file hash and its `runtime_manifest_sha256` semantic identity. The
work directory and report must not already exist.
This compact file is `B4/b4-wire-generation.json`; it is not the full sealed
runtime manifest. B6 carries both files independently and requires their
semantic runtime identities to agree.
Both producer outputs and the final gate output must be outside the source
worktree so evidence creation cannot invalidate the clean-source identity it
claims.

```bash
python tools/run_multires_pretraining_validation.py \
  --repo-root /absolute/q2-ml-bot \
  --b3-gate /absolute/B3-GATE.json \
  --b4-evidence /absolute/B4/B4-EVIDENCE.json \
  --runtime-manifest /absolute/B4/b4-wire-generation.json \
  --checkpoint /absolute/checkpoint.pt \
  --training-manifest /absolute/training-manifest.json \
  --bundle-manifest /absolute/bundle-v3.json \
  --objectives /absolute/objectives.json \
  --atlas /absolute/map.atlas.bin \
  --q2ded /absolute/q2ded \
  --client-binary /absolute/q2-client \
  --runtime-root /absolute/runtime \
  --seed 7142026 --game-seed 4242 \
  --campaign-transitions 500 --jobs 5 \
  --work-dir /new/evidence/b5-pretraining \
  --output /new/evidence/b5-pretraining-validation.json
```

On a command timeout, the producer kills the entire fresh subprocess group and
publishes no report. It re-hashes all eight direct immutable inputs and
revalidates/re-hashes the four B4 sibling documents after the campaign and
proof executions. Any changed byte fails the run.

## Gate assembly

The proof report is intentionally supplied independently of the producer
report. The assembler re-hashes it, re-runs the proof predicates, re-validates
all ten embedded campaign records, recalculates every canonical digest, and
re-hashes every current input and tool. It also requires the current clean Git
commit/tree and the authoritative design/plan digests to match the producer.

```bash
python tools/assemble_b5_gate.py \
  --repo-root /absolute/q2-ml-bot \
  --validation-report /absolute/b5-pretraining-validation.json \
  --proof-report /absolute/b5-pretraining/proof-500.json \
  --b3-gate /absolute/B3-GATE.json \
  --b4-evidence /absolute/B4/B4-EVIDENCE.json \
  --runtime-manifest /absolute/B4/b4-wire-generation.json \
  --checkpoint /absolute/checkpoint.pt \
  --training-manifest /absolute/training-manifest.json \
  --bundle-manifest /absolute/bundle-v3.json \
  --objectives /absolute/objectives.json \
  --atlas /absolute/map.atlas.bin \
  --output /new/evidence/B5-GATE.json
```

The assembler uses exclusive create and writes nothing on failure. Missing,
modified, copied-from-another-source, placeholder, or re-signed-but-semantic
invalid evidence cannot produce a green gate. The schemas are
`schemas/q2-multires-pretraining-validation-v1.schema.json` and
`schemas/q2-multires-b5-gate-v1.schema.json`.

## Predecessor-chain admission

B5 requires the exact B3 gate as a direct immutable input in addition to the
B4 aggregate. It independently validates B3 against the active B2 authority,
then requires the B4 `predecessor` record to equal those exact B3 file bytes,
gate seal, cohort/declaration, Atlas-set identity, and bot source identity.
B5 records the B3 file in its immutable bindings and carries the validated
chain into the final gate. Therefore B5 cannot start from an orphan B4 green
label, cannot substitute a different B3 after validation, and cannot produce a
green report while B2 has no active final cohort.

## Pre-B6 retirement validation (non-service artifact)

B5 cannot require the final `q2-multires-cold-start-v2` declaration: v2 binds
the final B6 aggregate and integration verification, neither of which exists
at the B5 boundary. Use the explicit evidence-only mode instead:

```sh
python3 tools/validate_multires_retirement.py \
  --mode pre-b6 \
  --manifest /absolute/M4-RUNTIME-RETIREMENT.json \
  --expected-manifest-sha256 <sha256> \
  --cold-start /absolute/pre-b6-cold-start.json \
  --operational-root /absolute/isolated-runtime \
  --service-selector /absolute/isolated-runtime/service-selector
```

That declaration has schema `q2-multires-pre-b6-cold-start-v1` and contains
only the fresh checkpoint/attestation, B4 runtime evidence, training manifest,
bundle manifest, and Dyn snapshots. Its report schema is
`q2-multires-pre-b6-retirement-validation-v1` and records `mode: pre-b6`.
It proves retirement and fresh step-zero selection for B5/B6 evidence; it is
never a live-service admission artifact.

Omitting `--mode` selects `service`, which accepts only
`q2-multires-cold-start-v2`. The v2 path additionally requires the B6/final
integration envelope and report, clean bot source identity, primary runtime,
and trainer checkpoint. `train.multires_service` invokes only that default
service path, so a pre-B6 declaration cannot enter the live selector.
