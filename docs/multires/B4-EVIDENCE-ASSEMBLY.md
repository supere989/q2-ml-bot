# B4 evidence assembly

`tools/assemble_b4_evidence.py` is the fail-closed completion gate for plan
batch B4. It does not treat source tests, a build log, or an endpoint version
claim as evidence. It consumes one finalized, passing, real-network
frame-barrier qualification and the exact sealed runtime manifest bound by that
qualification, then derives the four B4 inputs expected by
`tools/verify_multires_integration.py`.

The authoritative requirements remain
`docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md` and
`docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md`. This document defines
only the mechanical evidence closure.

## Admission boundary

The assembler requires all of the following at once:

- the exact canonical green B3 gate, validated against the currently active
  B2 final-cohort authority; if no B2 authority is active, B4 cannot be
  assembled or revalidated;
- exact agreement between that B3 gate and B4 on the bot commit/tree and
  authoritative design/plan digests, with the B3 gate seal, active
  cohort/declaration, and Atlas-set identity carried into B4;
- absolute, non-symlink paths to a new-generation qualification and runtime
  manifest;
- canonical JSON in the encoding produced by the owning writers, with
  duplicate keys and non-finite values rejected;
- `passed=true`, `test_mode=false`, and a complete real-network execution;
- a successful replay of every raw scenario through
  `_validate_execution_evidence`, including deterministic, fault, lifecycle,
  epoch-drain, and stale-telemetry cases;
- exact clean Git commit and tree identities for the bot, Yamagi client, and
  Lithium game repositories, equal to the qualification source closure;
- agreement among Python, client C, and game C on protocol generation 2,
  observation/action magics, POD sizes, client wire v8, frame-barrier ABI,
  teacher v4, three-way vertical intent, and the physically separate causal
  channel;
- a structural source proof that `ml_client_telemetry_t` contains only its
  public header, `ml_obs_t`, and private non-policy causal block, while the
  distinct `ml_teacher_sample_t` packing path alone carries the demonstration
  action tail; the public and teacher datagrams have different magics, sizes,
  send functions, and destination paths, and the Yamagi public router contains
  no teacher wire identity;
- a runtime manifest that binds the execution seal, q2ded, game module,
  Yamagi client binary, and the exact Atlas SHA-256;
- zero action dispatch during the epoch barrier, explicit rejection of replayed
  old telemetry without clock regression, and four same-map/same-epoch client
  records;
- causal reward frames derived only from authoritative, trainable action echoes
  in the accepted real-network trajectory; and
- an exhaustive per-client public-datagram audit in sealed
  `baseline-cold-1` evidence. For each socket, `seen = decoded + malformed +
  teacher` and `decoded = routed + foreign + stale`; the route must be
  nonempty and the measured teacher count must be zero; and
- a deterministic negative injection built from the exact source-attested
  teacher magic, version, and 1,224-byte POD. The public Python conduit must
  raise `PublicTelemetryPrivilegeViolation`; treating it as an ignorable
  malformed packet fails B4.

Synthetic/test-mode evidence is permanently inadmissible. A dirty source tree,
legacy endpoint, partial source deployment, changed binary, changed Atlas,
missing scenario, stale seal, noncanonical file, or pre-existing output path
causes the command to fail without publishing a partial result. There is no
legacy fallback.

The production assembler has no injectable execution validator. It always
replays the qualification with
`qualify_network_client_frame_barrier._validate_execution_evidence`; tests
replace that private dependency only through monkeypatching and cannot select
a weaker validator through the command or public Python API.

The real-network qualifier writes `ml_teacher_enabled=0` into every isolated
server configuration. Teacher collection is not enabled to manufacture this
proof: separation is established by exact source/runtime closure, nonempty
normal public traffic, and the explicit fatal negative injection.

## Required sequencing

B4 changes alter the bot source commit/tree recorded by qualification. Commit
the final B4 source first, build the exact client/game/q2ded runtime, and then
run the full network qualification against those clean commits. Do not reuse a
qualification created before the final B4 commit: source-closure equality will
reject it.

The finalized runtime manifest must contain these exact semantic runtime keys:

```text
network_barrier_execution_evidence_sha256
expected_atlas_sha256
client_binary_sha256
client_binary_size
```

The first key binds the manifest to the full-network execution. The Atlas key
prevents otherwise identical B4 evidence from being transplanted to a different
B3 map/Atlas artifact.

## Command

```bash
python3 tools/assemble_b4_evidence.py \
  --b3-gate /absolute/evidence/B3-GATE.json \
  --qualification /absolute/evidence/frame-barrier-qualification.json \
  --runtime-manifest /absolute/evidence/runtime-manifest.json \
  --bot-repo /absolute/q2-ml-bot \
  --client-repo /absolute/q2-ml-client \
  --game-repo /absolute/q2-lithium-3zb2 \
  --atlas-sha256 "$ATLAS_SHA256" \
  --output-dir /absolute/evidence/B4
```

The output directory is published atomically and contains:

```text
B4-EVIDENCE.json
feature-action-contract.json
runtime-epoch-fencing.json
b4-wire-generation.json
causal-reward-admission.json
```

`B4-EVIDENCE.json` conforms to
`schemas/q2-multires-b4-evidence-v1.schema.json`. It binds the exact source
commits/trees, binary records, runtime and qualification seals, normative
documents, raw scenario proofs, and the byte identity of each derived document.
Its `public_privilege_proof` summary cross-binds the baseline raw-evidence
seal, measured public datagram totals and zero teacher detections, plus the
negative-probe packet digest and fatal result. The full client audits, source
ABI proof, and negative-probe record live in `b4-wire-generation.json` and are
recomputed by the integration verifier.
The aggregate `predecessor` record binds the exact B3 file bytes and semantic
gate seal, the active B2 cohort/declaration, B3 Atlas-set digest, and the bot
source commit/tree. A copied B3 from another source, a replaced B3 file, or a
retired/no-active B2 authority is fatal.
The four smaller documents can be referenced directly from a multires
integration evidence envelope.

## What a green B4 means

A green aggregate concludes only that one exact generation-2 runtime has a
closed feature/action contract, atomically aligned endpoints, deterministic
real-network frame barrier, epoch/stale fencing, and causally admissible reward
facts. It does not authorize deployment, staging, training, checkpoint reuse,
or B5/B6 promotion by itself.
