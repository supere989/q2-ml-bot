# B2 generated hook candidates and materialization v3

Status: sole current hook candidate/materialization contract for B2 integration

V3 replaces V2 in one direction. Current generator metadata uses only the
top-level key `hook_claim_candidates_v3`, candidate schema
`q2-hook-claim-candidates-v3`, materialization schema
`q2-hook-claim-materialization-v3`, and runtime sidecar schema
`q2-hook-runtime-sidecar-v3`. V2 artifacts are historical evidence only and
are rejected by every current materializer, claim builder, analyzer, campaign,
bundle, and runtime path. There is no V2 parser fallback, conversion, dual
acceptance, or selector.

## Unproven generator proposal

The generator emits a bounded, deterministically ordered pool. The container
has exactly:

```text
schema: "q2-hook-claim-candidates-v3"
tick_msec: 100
status: "unproven"
bundle_admissible: false
records: [...]
```

Each proposal record has exactly:

```text
claim_id
source_milliunits
anchor_milliunits
landing_milliunits
release_after_ticks
distance_milliunits
flags
```

Only `source_milliunits`, `anchor_milliunits`, and `release_after_ticks`
declare the bounded replay to attempt. `landing_milliunits` is a
non-authoritative authored landing hint used to distribute proposals across
the map. It is not an expected endpoint, an exact-cell constraint, collision
evidence, or promotion evidence. A candidate may not be rejected merely
because compiled replay lands outside the hint's L1 cell. The materialized
record overwrites that field with the measured exact landing.

`source_milliunits` is the proposed exact standing Pmove origin at the first
eighth-unit origin above an authored floor-clear support plane. Exact authored
deathmatch spawns rank ahead of generic same-floor samples; remaining ranks are
only deterministic proposal heuristics. Compiled checks, not generator ranks,
decide support, hull clearance, grounding, and spawn reachability.

`anchor_milliunits` proposes an authored hookable face. It never authorizes an
attachment. The collision trace must reach a solid, non-sky face, and the
materializer replaces the proposal with the engine's measured collision
impact, including collision epsilon. `distance_milliunits` is the rounded
Euclidean distance from `source + (0,0,22)` to the proposed anchor. The
materializer recomputes it from the source and measured anchor.

`release_after_ticks` selects one exact 100 ms hook-pull schedule. Candidate
IDs, source/anchor/hint geometries, and schedules are uniquely and stably
ordered. The pool remains bounded at 512 records, two proposed sources per
hint geometry, and 256 rows in the unproven runtime projection. The generated
`<map>.json` remains an eight-field projection:

```text
anchor.xyz landing.xyz distance flags
```

That projection is source evidence only. It always starts non-admissible and
must never enter a bundle or install path.

## Compiled landing discovery and seal

After BSP compilation, `tools/materialize_hook_claims.py` processes only the
declared V3 records, in their stable order, through the pinned CM, Pmove, hook,
and fall authorities. It performs no broad existential search and does not
invent new source, anchor, release, or schedule values.

For each record, materialization fails that record closed unless all of the
following hold:

1. The source is the declared exact fixed-point Pmove origin, has exact support
   and hull clearance, remains grounded under the zero-input preflight, and its
   L1 node is reachable from a compiled deathmatch spawn.
2. The measured hook impact is exactly attachable, solid, and non-sky under the
   admitted hook physics identity.
3. Exact hook pull establishes an airborne state, and the post-release frames
   then contain a false-to-true grounded Pmove transition. The first grounded
   frame's 1/8-unit Pmove origin is the measured landing.
4. The measured landing's actual L1 target exists in the compiled navigation
   graph, is supported, and has standing or crouched hull clearance.
5. The actual source and measured-landing L1 cells differ.
6. The measured anchor, measured landing, and flags form a runtime geometry not
   already selected.

The landing hint is not consulted by items 3--6. The first six records that
pass all compiled checks become the selected set. Any total other than exactly
six rejects the complete map. Passing subsets are not publishable.

The canonical materialization binds the BSP bytes, full V3 metadata/candidate
pool, original non-admissible projection, exactly six selected records,
ordered Pmove fixed-origin frames and first-grounded indices, executable/tool/
physics identities, B1 runtime authority seal, and exact hook-parity
attestation. Its landing policy is `compiled-first-grounded-exact-v3`: selected
`landing_milliunits` contains the measured first-grounded origin, never the
generator hint.

The attestation and final six-row runtime sidecar are constructed and checked
in an isolated temporary stage. Only after both canonical bytes and their
cross-digests validate may the complete stage be atomically sealed. Existing
destinations, partial files, a sidecar without its attestation, or an
attestation without its matching sidecar reject publication. Only the sealed
V3 sidecar may say `bundle_admissible: true`.

## Independent exact replay

The B2 Atlas analyzer treats neither the materializer's pass bit nor its
landing-discovery result as replay authority. It accepts only the sealed V3
record, independently reconstructs all six declared source/anchor/release
schedules under the same pinned identities, and runs strict expected-landing
mode. For every record it must reproduce:

- the measured collision anchor;
- the exact first-grounded 1/8-unit landing;
- the ordered Pmove fixed-origin frames, first-grounded index, and trace digest;
- source spawn reachability and distinct actual source/target L1 cells; and
- actual target support and standing-or-crouched clearance.

L0 hook corridors contain only those independently replayed Pmove frames. Any
identity, byte, ordering, endpoint, target-legality, trace, or digest mismatch
rejects the map.

V3 discovery exists only while producing a new staged materialization. It is
never available to independent analysis, claim validation, bundle admission,
or runtime loading. Failed V2 cohorts and artifacts may not be rematerialized,
converted, repaired, or admitted through V3.
