# B2 generated hook candidates v2 (retired)

Status: retired historical contract; non-admissible

V2 is preserved only to explain historical artifacts and failed campaigns. It
is superseded in one direction by `B2-HOOK-CANDIDATES-V3.md`. No current
materializer, claim builder, analyzer, campaign, bundle, or runtime accepts V2,
and there is no parser fallback, conversion, or dual-acceptance path. The
present-tense descriptions below describe the retired V2 protocol only; they
do not authorize new generation or materialization.

The generator emits an overcomplete source-bound hook replay pool in the
top-level `.meta.json` key `hook_claim_candidates_v2`. This pool is a proposal,
not collision, hook, movement, Atlas, promotion, bundle, or runtime authority.

The container has exactly these fields:

```text
schema: "q2-hook-claim-candidates-v2"
tick_msec: 100
status: "unproven"
bundle_admissible: false
records: [...]
```

Each record has exactly:

```text
claim_id
source_milliunits
anchor_milliunits
landing_milliunits
release_after_ticks
distance_milliunits
flags
```

`source_milliunits` is a proposed exact standing Pmove origin selected from
authored floor-clear samples. The generator places it at the first eighth-unit
origin above the authored support plane. Compiled preflight requires the pinned
CM support endpoint to ceil to that exact fixed origin, requires a stationary
hull to be clear there, and executes a zero-input Pmove under the admitted
gravity/air-acceleration identity; that frame must remain at the same fixed
origin and report grounded. Candidate `landing_milliunits` is only a desired
landing-region representative, 24 units above its authored support rather than
56 units above it. It is not evidence that Pmove lands at that coordinate. The
materializer requires the measured first post-release false-to-true grounded
origin to occupy the same exact L1 cell, then replaces the proposal with that
measured 1/8-unit Pmove origin. `anchor_milliunits` lies on the first real
ceiling or ceiling-light-panel face above the desired landing. A platform or
roof below that face suppresses the proposal; it is never relabeled as a
ceiling.

The proposed `anchor_milliunits` names the authored face. CM traces terminate
at the engine's exact collision impact, including its collision epsilon. The
materializer replaces the proposal with that measured impact in integer
milliunits, recomputes the source-bound distance, and the independent analyzer
must reproduce the same measured anchor. It never treats authored face
coordinates as the collision endpoint.

`distance_milliunits` is the rounded Euclidean distance from
`source + (0,0,22)` to the anchor, in milliunits. `release_after_ticks` selects
one exact 100 ms hook-pull schedule. Candidate IDs, geometries, sources, and
release schedules are deterministically ordered. Source and landing may not
occupy the same L1 cell.

The bounded pool preserves the complete former 43-geometry coordinate class,
then adds deterministic farthest-point L1 coverage across the remaining floors
and XY cells. The resulting union is ordered from one desired landing near each
authored spawn through the farthest remaining cells. Sources on authored spawn
floors are considered first, followed by sources on the desired landing floor;
these are only proposal heuristics and never reachability evidence. Both source
ranks are retained because the first exact campaign proved records from each.
Each receives at least four deterministic release schedules: rank zero
concentrates on the 2--5 interior with one-in-eight 1--4 coverage, while rank
one covers 3--6.
If a retained geometry has only one eligible source, its vacant bounded slots
are filled by omitted schedules on earlier selected sources.
A full pool therefore covers 64 desired landing geometries, both source ranks,
and every release tick. The caps remain 512 records, two sources per desired
geometry, and 256 runtime projection rows.

The allocation is based only on aggregate, quarantined campaign classes. The
13 initially passing maps selected 78 records: 47 source-rank-zero and 31
source-rank-one; release ticks 1--6 contributed 5, 14, 20, 18, 11, and 10
records respectively; and winning geometry ordinals ranged from 0 through 42.
No measured anchor, measured landing, map identity, or per-map winning row is
an input to proposal generation.

The generated `<map>.json` remains an eight-field C-loader-compatible
projection:

```text
anchor.xyz landing.xyz distance flags
```

It contains unique proposed geometries rather than repeated source/release
schedules. This raw projection is also unproven and must never enter a bundle
or install path. Keeping the prefix only preserves the existing loader and
policy observation shape; it does not authorize a hook edge.

After BSP compilation, `tools/materialize_hook_claims.py` must replay each
exact source/release schedule through the pinned CM, hook-law, and Pmove
authorities and materialize the first six unique proven geometries in stable
order. Its separate canonical attestation binds the BSP, metadata/candidate
pool, original non-admissible projection, selected records, ordered Pmove
traces, all executable/tool/physics identities, and hook-parity attestation.
Only after that attestation is atomically published may the final eight-field
sidecar say `bundle_admissible: true`.

The B2-A v2 analyzer ignores the materializer's pass bit as replay authority.
It independently replays all six sealed source/release schedules, requires the
same first grounded fixed origin and ordered trajectory digest, and derives L0
hook corridors only from those Pmove frames. If six cannot be proved, any
identity changes, or either replay differs, the map is rejected. There is no
generator fallback, broad existential hook search, fabricated claim,
source-metadata acceptance, or relaxed verification path.
