# B2-C generated-map compiled-world claim contract

Status: offline prototype contract for B2 integration
Schemas: `q2-generator-claims-v2`,
`q2-generator-claim-validation-v1`, and
`q2-generator-claim-campaign-v1`

Generator v6 `.map`, `.meta.json`, `.lattice.json`, hook-zone,
`.hook-materialization.json`, and route sidecars are claims to challenge. They
are never collision, movement, hazard, lighting, or hook authority. The
existing source/static validator remains a necessary generated-v6 criterion,
but its success cannot fill a missing or unknown compiled Atlas fact.

## Canonical claim input

`tools/generator_claim_validator.py` normalizes the six source files into one
canonical compact/sorted JSON document with a trailing LF. Its SHA-256 binds:

- all eight source and lattice spawn origins;
- every lava and generated `trigger_hurt` volume;
- every materialized source, anchor, measured landing, release schedule,
  distance, and flag;
- every route segment endpoint and each route's normalized cost claim; and
- SHA-256 identities for all six source files.

The hook materialization is a separate canonical record. It binds the exact
BSP and candidate metadata, the original non-admissible projection, six
selected records and ordered Pmove traces, executable/tool/physics identities,
and the pinned parity attestation. A final runtime hook sidecar is admissible
only when its canonical header and eight-field rows match this record exactly.

The legacy room graph can report zero cost between different locations in one
room. The canonical cost claim therefore uses the greater of the declared
room-graph distance and the route node sequence's geometric length. This does
not authorize a path; it only avoids treating a legacy zero as an oracle fact.
The compiled analyzer still must prove every segment connected and emit its
Atlas cost.

The analyzer accepts this document as an immutable input and echoes its digest
as `generator_claims_sha256` in `q2-atlas-analysis-v1`. Generated promotion
requires exact one-to-one result coverage for hazard, hook, and route claim
IDs. A changed source file necessarily changes the claims digest and makes an
older analysis report inadmissible.

## Generated-v6 promotion

The generated path requires every criterion below:

1. B1 is green, collision admission is positive, the BSP identity matches,
   and analysis confidence is complete/high. The collision executable must
   equal the exact B1-admitted binary SHA-256 and the analyzer identity must
   reproduce its local admitted source-closure SHA-256. A self-declared
   `deterministic_rebuild` boolean is not rebuild evidence.
2. A single independent full-cold producer launch finishes in at most 300
   seconds while 10 ms process-tree sampling stays at or below 512 MiB RSS.
   Its primary and cold digests must be equal and cover exactly seven local
   artifacts: byte digests for Atlas, Atlas transport, navigation,
   visibility, design signature, and routes, plus the normalized Atlas
   manifest digest. The verifier result and identity must also match the
   primary analysis.
3. The local artifacts and Atlas header independently satisfy all frozen
   limits: at most 1200 L0 chunks, 16 MiB decompressed L0, 32 MiB decompressed
   and resident Atlas, and 512 MiB packer RSS. Analysis, Atlas-manifest,
   header, and verifier identities and counts must agree.
4. Existing source/static v6 validation passes against the same-name compiled
   BSP.
5. Compiled spawn origins exactly match all eight claims. Every spawn is
   supported, standing/crouched clear, has an oracle-swept 96-unit column, has
   an escape edge, and is mutually connected to all other spawns. Compiled XY
   separation remains at least 384 units.
6. Every lava/hurt claim has positive raw and hull-expanded Atlas cells and
   positive oracle containment evidence. Aggregate hazard types must include
   every claimed type.
7. Compiled lethal-edge count matches the safety contract, all lethal exterior
   edges are guarded, and no uncontained edge remains.
8. qrad lightdata is nonempty and at most 2 MiB, has a nonzero digest and
   lightmapped faces, the compiled spawn-region count matches the v6 lighting
   contract, and no dark spawn region remains.
9. Hook authority is admitted under both the B1 hook physics identity and the
   exact SHA-256 of the accepted B1 hook-parity attestation. The analysis and
   hook materialization must independently carry that same attestation digest;
   matching physics text alone is insufficient. Every claim has exactly one
   source-bound, spawn-reachable edge whose exact source, anchor, release ticks,
   first grounded 1/8-unit landing, and ordered Pmove trajectory reproduce the
   materialization independently. L0 hook corridors contain only those
   replayed trajectory frames.
10. Every route segment has positive oracle/validation evidence, exact claimed
   endpoints, finite cost, and compiled connectivity. A route fails cost
   consistency only when its total differs by more than 1024 units **and** by
   more than 2x; the joint threshold accounts for the legacy room-centre cost
   model while still rejecting gross/tampered claims.

Any missing, failed, `unknown`, non-oracle, unbound, or malformed fact rejects
promotion. Claim values are never consulted as fallback evidence.

## Stock analysis is separate

`--mode stock` applies authored-map analysis quality only, but it uses the same
full-cold artifact authority. A BSP is stock only when its canonical ID and
SHA-256 match `docs/multires/stock-q2dm1-q2dm8.provenance.json`. Its exact
deathmatch-spawn count and item-class multiset must match
`tests/fixtures/corpus/stock-q2dm1-q2dm8.json` and the compiled design
signature. Stock hazard classification must carry positive oracle status,
evidence, and a validation version; an unconditional pass is rejected. Stock
analysis does **not** require generator tags, eight spawns, 98% tagged
floor-light coverage, v6 hook claims, or v6 headroom/guard metadata.

## Current producer gap (fail closed)

The validator consumes `q2-atlas-full-cold-proof-v1`, but the analyzer at this
integration point does not yet emit the cold-launch artifact digest sets or
elapsed/timeout evidence. The analyzer must add these exact proof members:

- `cold_artifact_sha256`, with the six exact artifact suffixes;
- `cold_artifact_semantic_sha256`, with `.atlas.manifest.json`;
- `elapsed_milliseconds`; and
- `timeout_limit_milliseconds`, exactly `300000`.

For stock maps it must also add
`compiled_world.hazards.classification_status`, `evidence`, and
`validation_version`. (The analyzer already emits
`oracles.hook.attestation_sha256`; promotion now validates it against the exact
B1-accepted attestation.) Until the producing analyzer publishes the missing
facts from the independent process and oracle path, B2 promotion remains red.
No validator-side fallback or inferred success is permitted.

## Offline workflow

Generate and compile the BSP beside its source files first. Then materialize
the hook candidates under the pinned B1 authorities:

```sh
python tools/materialize_hook_claims.py \
  --bsp /isolated/B2/generated/b2claim_0000.bsp \
  --meta /isolated/B2/generated/b2claim_0000.meta.json \
  --runtime-sidecar /isolated/B2/generated/b2claim_0000.json \
  --output-attestation /isolated/B2/generated/b2claim_0000.hook-materialization.json \
  --cm-oracle /isolated/B1/q2-cm-oracle \
  --pmove-oracle /isolated/B1/q2-pmove-oracle \
  --hook-oracle /isolated/B1/q2-hook-oracle \
  --hook-parity-attestation /isolated/B1/hook-parity-pullspeed-1700.json
```

Materialization fails closed unless exactly six unique geometries replay. Once
each compiled map has its canonical attestation and admissible runtime
sidecar, prepare the v2 claims:

```sh
python tools/run_generator_claim_campaign.py \
  --generated-dir /isolated/B2/generated \
  --glob 'b2claim_*.map' --expected-count 20 --phase prepare \
  --output /isolated/B2/claims-prepare.json
```

Run the B2-A v2 analyzer with the matching `.generator-claims.json`; the normal
path remains `candidate`/pending until its independent full-cold rebuild and
verifier replay succeed. Do not use `maps/compile.sh` for this gate because it
copies BSPs into a runtime map directory. Then validate the compiled campaign:

```sh
python tools/run_generator_claim_campaign.py \
  --generated-dir /isolated/B2/generated \
  --glob 'b2claim_*.map' --expected-count 20 --phase validate \
  --output /isolated/B2/compiled-claim-campaign.json
```

Campaign reports contain no timestamps and remain canonical across identical
inputs. The full-cold proof does carry integer elapsed milliseconds because
the frozen 300-second limit is an admission fact; semantic Atlas-manifest
comparison excludes only the independently sampled peak-RSS field.
