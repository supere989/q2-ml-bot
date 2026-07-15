# B2-C generated-map compiled-world claim contract

Status: offline prototype contract for B2 integration
Schemas: `q2-generator-claims-v3`,
`q2-generator-claim-validation-v1`, and
`q2-generator-claim-campaign-v2`

Nested hook authority uses only `q2-hook-claim-candidates-v4`,
`q2-hook-claim-materialization-v4`, and `q2-hook-runtime-sidecar-v4` as defined
by `B2-HOOK-CANDIDATES-V4.md`. Hook V2/V3 is retired historical evidence. It is
not a fallback or alternate input to any current claim, analysis, campaign,
bundle, or runtime path.

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
- every V4-materialized source, trace target, measured anchor, measured
  landing, release schedule, distance, and flag;
- every route segment endpoint and each route's normalized cost claim; and
- SHA-256 identities for all six source files.

The V4 hook materialization is a separate canonical record. It binds the exact
BSP and V4 candidate metadata, the original non-admissible projection, six
selected records whose landing is the compiled first-grounded Pmove origin,
ordered Pmove traces, executable/tool/physics identities, and the pinned parity
attestation. The generator's landing hint is never an expected-cell constraint
or compiled-world claim. A final V4 runtime hook sidecar is admissible only
when its canonical header and eight-field rows match this record exactly.

The generator retains the deterministic 16-unit standing-hull grid used to
establish source components and publishes each route's shortest-path distance
through that grid. Final lane walls, cover, buildings, towers, and lava are
therefore present in route scoring and cost claims rather than being discarded
after a Boolean connectivity decision. At source freeze, that distance is a
clean-implementation-bound generator proposal: the validator does not
reconstruct the generator's standing graph from the emitted artifacts or
require independent shortest-path equality. It requires only that the proposal
cover the exact route endpoint sequence's geometric length, and claim
preparation rejects an undercut instead of silently normalizing it. The
published integer source distance is rounded upward, so it may never fall below
that endpoint-loop lower bound, even by a sub-unit rounding tolerance. Neither
value authorizes a path: the compiled analyzer still must prove every segment
connected and emit its independent Atlas cost under the joint absolute-and-ratio
threshold in criterion 10.

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
   every claimed type. Compiled hurt fields first intersect the ordinary
   reachable-surface chunk permit. If that intersection is empty, a distinct
   hazard permit may retain the exact chunk immediately below a bounded floor
   strip. For generated maps, only the normalized lethal-edge segments select
   those strips, using the same inward witness as the later compiled safety
   challenge; source metadata is candidate scope and never supplies a cell.
   Stock analysis may use evidenced reachable boundary floor columns. This
   one-chunk downward band exists for fixed-point grounded-origin alignment
   only: it never admits unrelated wall/ceiling boundaries, interior floor
   fill, horizontal dilation, a two-chunk gap, or the trigger's full AABB. Only
   the inclusive compiled AABB intersection inside an admitted chunk is
   stored, under the same prospective 1200-chunk/16-MiB L0 budgets. The later
   CM probes must still prove every proposed generated floor, void, guard, and
   lethal catch exactly.
7. Compiled lethal-edge count matches the safety contract, all lethal exterior
   edges are guarded, and no uncontained edge remains.
8. qrad lightdata is nonempty and at most 2 MiB, has a nonzero digest and
   lightmapped faces, the exact compiled authored floor-light region-ID set
   matches the v6 source metadata, and deterministic compiled traces prove
   that no individual spawn is dark. Spawn navigation-region identity is a
   separate connectivity domain and is not evidence of authored light-region
   identity.
9. Hook authority is admitted under the sole V4 contract, the B1 hook physics
   identity, and the exact SHA-256 of the accepted B1 hook-parity attestation.
   The analysis and
   hook materialization must independently carry that same attestation digest;
   matching physics text alone is insufficient. Every claim has exactly one
   source-bound, spawn-reachable edge whose exact source, preserved trace
   target, measured anchor, release ticks, first grounded 1/8-unit landing,
   and ordered Pmove trajectory reproduce the materialization independently.
   L0 hook corridors contain only those replayed trajectory frames.
10. Every route segment has positive oracle/validation evidence, exact claimed
   endpoints, finite cost, and compiled connectivity. A route fails cost
   consistency only when its total differs by more than 1024 units **and** by
   more than 2x; the joint threshold admits bounded differences between the
   conservative source standing grid and compiled Pmove navigation while still
   rejecting gross/tampered claims. Endpoint support begins
   one exact 1/8-unit Pmove quantum above the claimed player origin. Likewise,
   navigation flood and non-spawn seed discovery retain their bounded raised
   search first, then use a grounded 1/8-unit support trace when the raised
   standing hull is rejected by a legal low overhang. When multiple authored
   endpoints alias one L1 key, projection searches only the adjacent L1
   stencil within 16 units and requires exact bidirectional standing-hull CM
   connectors before selecting by distance and stable `(z, y, x)` order.
   Neither path invents headroom or connectivity: compiled support normals,
   exact standing-hull edge sweeps, and the resulting Atlas graph remain
   authoritative. Generator source components also treat exact horizontal or
   ceiling contact as solid while preserving legal contact with the floor;
   a player-width gap is not a source-only route witness.

Any missing, failed, `unknown`, non-oracle, unbound, or malformed fact rejects
promotion. Claim values are never consulted as fallback evidence.

The analyzer-to-packer transport is solely
`q2-atlas-build-plan-v2`. Each admitted L0 bitplane is exactly 512 bytes and
each scalar plane exactly 4096 bytes, encoded as canonical lowercase hex.
Build-plan v1 index arrays are retired and rejected. This compact transport is
an implementation representation only: the frozen Atlas planes, cell
ordinals, prospective budget accounting, and Rust verifier output are
unchanged. It prevents source-language object overhead from becoming the
limiting resource for a sparse but broadly intersecting hazard band. Exact
semantic-union accounting uses the same 4096-bit chunk representation and is
prospectively capped at 16 MiB of bitmap payload; exceeding that separate
scratch bound rejects analysis before allocating the next bitmap.

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

## Full-cold producer closure (fail closed)

The analyzer now emits the complete `q2-atlas-full-cold-proof-v1` contract from
one independently launched `tools/atlas_cold_worker.py` process. It publishes
matching primary and cold `artifact_sha256` maps for the six byte-stable
artifacts, matching `artifact_semantic_sha256` maps for both
`.atlas.manifest.json` and `.analysis.manifest.json`, measured process-tree
peak RSS, `elapsed_milliseconds`, and `timeout_limit_milliseconds` exactly
`300000`. The analyzer rejects a byte or semantic mismatch, missing artifact,
non-positive resource sample, more than 512 MiB peak RSS, or more than 300
seconds elapsed before publishing a passed manifest.

Stock analysis likewise publishes
`compiled_world.hazards.classification_status`, `evidence`, and
`validation_version` from the collision/drop oracle path. Promotion validates
the hook attestation against the exact B1-accepted authority and independently
checks the full-cold field set and digest membership. B2 remains red until a
fresh declared cohort supplies passing producer evidence; no validator-side
fallback, inferred success, or historical artifact is permitted.

## Exact cohort admission

Campaign v2 has no directory-discovery mode. Its only membership authority is
the exact, pre-generation declaration supplied with `--declaration`. The
current declaration is
`docs/multires/B2-GENERATED-COHORT-DECLARATION.json`: it names 28 maps in a
fixed ordinal order, four for each of the seven concrete styles. A stage is
admissible only when `tools/run_generator_cohort.py` finds every declared file
and no unexpected regular file or symlink. A missing member plus an undeclared
replacement still fails even when the raw file or map count remains 28.

The stages are separate, non-nested directory roots:

- `materialized` contains the five declared generator source files, the BSP,
  and the hook-materialization attestation for every map (196 exact files);
- `claims` contains an exact copy of that materialized set plus one canonical
  `.generator-claims.json` for every map (224 exact files); and
- `analysis` contains the eight compiled Atlas analysis artifacts for every
  map, including the explicit `.analysis.manifest.json` consumed by campaign
  validation (224 exact files).

The generator source and Atlas output both name a file `<map>.routes.json`, but
they have different authority. The generator file is a claim bound into the
claims stage. The analysis file is the independently derived Atlas route
artifact. They must remain in their respective roots; Atlas routes must never
replace, overwrite, or be copied into the generator claims stage.

Generator route sidecars now use version 2. Every node carries its explicit
`source_component` proposal, and every route names the exact `start_node_id`
and component used during selection. The source-freeze validator requires the
source graph to contain exactly eight deathmatch spawns, all assigned to one
shared non-null standing component. The start spawn and every selected item
endpoint must also share the route's declared component. If final source
geometry has no one component that can hold all eight clear, separated,
map-spanning starts, generation fails; it may not scatter starts across
components. Before publishing the source freeze, the validator independently
parses `info_player_deathmatch` entities from the emitted `.map`, requires
exactly eight unique canonical origins, and requires their sorted set to equal
the route sidecar's eight unique spawn origins. The report binds the parsed
origins and both canonical digests; a fabricated or stale route sidecar fails
the entire cohort. Source route validation does not claim that item or spawn
nodes have compiled floor support. Room connection rows remain serialized
diagnostics and risk priors; they are never reachability evidence. These
component labels are still generator claims, not compiled collision authority.
Independent Atlas route probes must challenge the exact endpoint loop and
directed spawn reachability before admission. Version 1 sidecars are retired
and rejected; there is no inferred start-spawn or room-edge fallback.

Preparation evaluates all 28 claim documents in declaration order before it
publishes anything. If any build fails, no claims root or passing subset is
published. After all 28 build, the tool copies the exact materialized stage and
writes the claims into a temporary sibling, verifies exact claims membership,
then atomically renames that directory into place. The destination must not
already exist. There is no salvage, replacement-map, adjacent-artifact, or
fallback path.

Both prepare and validate reports are canonical compact/sorted JSON with a
trailing LF and no timestamp. The report path must be outside all exact stage
roots and must not exist: publication uses exclusive creation and refuses to
overwrite an earlier result (`O_CREAT | O_EXCL`). The report binds `cohort_id`,
the canonical declaration SHA-256, exact stage-membership report SHA-256
values, all 28 ordinal rows, and aggregate failures. `compiled_validation`
additionally binds the B1 gate SHA-256. `pass_count` is diagnostic; only
top-level `passed: true` for the complete declaration is admission.

The retired campaign-v1 options `--generated-dir`, `--glob`,
`--expected-count`, and `--phase` are not accepted. Glob/count equality was
never proof that the declared cohort survived unchanged.

## Declaration status

The historical cohort `b2g26_final_71426` failed its first source-freeze
attempt and is permanently non-admissible. That failure is retained as
evidence; it did not authorize a source, materialized, claims, or analysis
stage. Do not repair it by substituting maps, selecting its passing members,
regenerating behind the same declaration, or feeding an older materialized
population into campaign v2.

The historical replacement cohort `b2g26_final_71427` passed its exact
28-member source freeze, compile membership, and static validation, then
failed the first hook materialization on ordinal 0,
`b2g26_open_71427000`. The retained process stdout is empty and the original
stderr and exit status were not durably captured. Operator commentary recorded
exit 1 and the diagnostic `compiled hook preflight proved 2/6 unique
geometries`, with rejection counts `anchor_not_exactly_attachable=28`,
`duplicate_proven_geometry=2`,
`measured_landing_outside_desired_l1=400`, and
`source_not_spawn_reachable=80`; that commentary is explicitly not a hashed
raw log. The durable exact-membership check is authoritative for admission: it
found 168 of 196 required materialized files, with all 28
`.hook-materialization.json` attestations absent and no unexpected files.

The canonical failure record is
`docs/multires/B2-GENERATED-COHORT-71427-FAILURE.json`. Cohort 71427 is
permanently non-admissible. Its source and compiled results remain evidence
only; it published no materialized, claims, or analysis stage. Do not rerun or
regenerate it behind the same declaration, repair or substitute a member,
select a passing subset, or feed its compiled files or an older materialized
population into campaign v2.

The next replacement, `b2g26_final_71428`, passed its source freeze, compile,
V3 hook materialization, and claims preparation, then failed exact compiled
Atlas analysis 0/28. The failure classes were five unsupported route
endpoints, eight routes without evidenced connectivity, eight dense L0 hurt
fills, six lava claims without compiled hits, and one non-idempotent V3 hook
replay. Its exact declaration and canonical failure record are archived as
`B2-GENERATED-COHORT-71428-DECLARATION.json` and
`B2-GENERATED-COHORT-71428-FAILURE.json`. Cohort 71428 is permanently
non-admissible; no member or V3 artifact may be repaired, converted, retried,
or salvaged.

The sparse-hurt, exact-lava, final-endpoint, standing-component route, and V4
hook fixes were committed before replacement cohort `b2g26_final_71429` was
declared. Its first source-freeze attempt produced complete, byte-identical
140-file primary and cold populations, then correctly published no freeze
report because the validator still interpreted diagnostic room edges as
reachability. The first refusal was ordinal 5,
`b2g26_towers_71429101`; a complete archived scan found the same stale-contract
conflict in seven maps. The exact declaration and canonical failure record are
archived as `B2-GENERATED-COHORT-71429-DECLARATION.json` and
`B2-GENERATED-COHORT-71429-FAILURE.json`. Cohort 71429 is permanently
non-admissible; its generated members may not be reused, regenerated, repaired,
or salvaged.

The version-2 route-sidecar/source-freeze fix was committed before cohort
`b2g26_final_71430` was declared. Its source freeze, compile, static
validation, V4 hook materialization, and claims preparation all passed for 28
members. Exact compiled Atlas analysis then failed 0/28 because real Pmove
grounded origins at Z 24.03125 put the reachable floor-surface permit exactly
one 64-unit L0 chunk above the generated kill plane. Strict three-dimensional
surface clipping consequently retained no raw or hull-expanded hurt evidence.
Its exact declaration and canonical failure record are archived as
`B2-GENERATED-COHORT-71430-DECLARATION.json` and
`B2-GENERATED-COHORT-71430-FAILURE.json`. Cohort 71430 is permanently
non-admissible; none of its source, BSP, V4 hook, claims, or analysis artifacts
may be reused, repaired, retried, or salvaged.

Two strictly diagnostic shadows then challenged those immutable inputs without
publishing an analysis subset. Commit `48f5caf` passed 22/28 and exposed six
independent late failures after the one-chunk hurt-boundary implementation:
two low-overhead route
probes, three over-broad hazard chunk scopes, and one L1 representative alias.
Commit `8ce1e75` passed 27/28 with deterministic full-cold parity for every
successful member. The sole remaining retired claim, arena-open seed 71430402,
correctly failed because the source standing-component model had treated exact
horizontal hull contact as clearance across a compiled wall. The corrected
source model deterministically keeps that endpoint in its real component and
selects a different route; it never bridges the Atlas graph or repairs the
retired sidecar. The canonical diagnostic record is
`B2-GENERATED-COHORT-71430-SHADOW-8CE1E75.json`.

All discovered implementation fixes and their fail-closed regressions were
committed before fresh cohort `b2g26_final_71431` was declared. Its 28 new
members use seeds 71431000 through 71431603 under the same all-or-nothing,
declared-before-generation policy. The current alias and immutable named copy
are byte-identical; no 71430 source, BSP, hook, claim, or analysis member is an
input to 71431.

The first and only 71431 source-freeze attempt produced complete,
byte-identical 140-file primary and cold populations, then correctly published
no freeze report. Ordinal 23, arena-vertical seed 71431503, contained an
objective-tower top 80 units below an overlapping room ceiling. The generator
had not registered objective or ordinary towers with its horizontal-surface
model; a second ordinary tower in the same source left a 62-unit cross-room
gap that the thin-brush source classifier could not see. The exact declaration
and canonical failure record are archived as
`B2-GENERATED-COHORT-71431-DECLARATION.json` and
`B2-GENERATED-COHORT-71431-FAILURE.json`. Cohort 71431 is permanently
non-admissible; its members may not be reused, regenerated, repaired, retried,
or salvaged. Both tower paths now reject candidates against every registered
horizontal surface, retain their minimum semantic height, and register every
accepted tower for the final global challenge. A replacement requires a new
committed declaration and entirely new seeds.

The tower-surface fix, exact 71431503 regression, and immutable 71431 failure
record were committed before replacement cohort `b2g26_final_71432` was
declared. Its 28 new members use seeds 71432000 through 71432603 under the same
all-or-nothing, declared-before-generation policy. The current alias and
immutable named copy are byte-identical; no 71431 source or later-stage member
is an input to 71432.

The first and only 71432 source-freeze attempt also produced complete,
byte-identical 140-file primary and cold populations and correctly published
no freeze report. Ordinal 5, towers seed 71432101, emitted four corner pockets
but retained only three enterable corner-light zones: the late objective tower
occupied the reserved interior of `corner_0` after its boundary walls had been
accepted. The exact declaration and canonical failure record are archived as
`B2-GENERATED-COHORT-71432-DECLARATION.json` and
`B2-GENERATED-COHORT-71432-FAILURE.json`. Cohort 71432 is permanently
non-admissible; its members may not be reused, regenerated, repaired, retried,
or salvaged. Ordinary structure clearance now protects the full reserved
traversable volume of every hollow building and corner pocket, so towers,
cover, lava, and objectives cannot silently erase a promised interior.

The reserved-interior fix, exact 71432101 regression, and immutable 71432
failure record were committed before replacement cohort
`b2g26_final_71433` was declared. Its 28 new members use seeds 71433000
through 71433603 under the same all-or-nothing, declared-before-generation
policy. The current alias and immutable named copy are byte-identical; no
71432 source or later-stage member is an input to 71433.

The first and only 71433 source-freeze attempt failed during primary
generation at ordinal 21, arena-vertical seed 71433501. All eight starts were
in one conservative standing component with six offense items but only one
survival/value item; fourteen qualifying survival/value items were segregated
in components without a spawn, so the mandatory two-distinct-endpoint
survival route correctly refused publication. The pre-fix writer left 21
complete predecessors and three partial failed-member files; the cold pass
never began and no freeze report was published. The exact declaration and
canonical failure record are archived as
`B2-GENERATED-COHORT-71433-DECLARATION.json` and
`B2-GENERATED-COHORT-71433-FAILURE.json`. Cohort 71433 is permanently
non-admissible; no partial or complete member may be reused, regenerated,
repaired, retried, or salvaged. The generator now reserves two true offense
and two true survival pickups in one exact spawn-bearing source component,
then solves the remaining heat economy around them. Route construction also
runs in memory before the first member file is written, and route starts are
keyed by both room and source component.

The spawn-component economy fix, exact 71433501 regression, atomic route
refusal regression, and immutable 71433 failure record were committed before
replacement cohort `b2g26_final_71434` was declared. Its 28 entirely new
members use seeds 71434000 through 71434603 under the same all-or-nothing,
declared-before-generation policy. The current alias and immutable named copy
are byte-identical; no 71433 source or later-stage member is an input to
71434.

The first and only 71434 source-freeze attempt was operator-terminated at
ordinal 20, arena-vertical seed 71434500, after approximately 30 CPU-minutes
inside ceiling-sandwich normalization. The primary population contains 20
complete predecessors and no partial failed member; the cold pass never began
and no freeze report was published. The exact declaration and canonical
failure record are archived as `B2-GENERATED-COHORT-71434-DECLARATION.json`
and `B2-GENERATED-COHORT-71434-FAILURE.json`. Cohort 71434 is permanently
non-admissible; no member may be reused, regenerated, repaired, retried, or
salvaged. The repair now moves only the upper room ceiling, requires a strict
integer increase on every iteration, rejects repeated states, and fails closed
at explicit height and iteration fences. The monotonic ceiling normalization
preserves the 96-unit safe-headroom and low/mid/high spatial-proportion
contracts.

The monotonic-normalization fix, exact six-state-cycle fixture, timeout-bound
71434500 regression, and immutable 71434 failure record were committed before
replacement cohort `b2g26_final_71435` was declared. Its 28 entirely new
members use seeds 71435000 through 71435603 under the same all-or-nothing,
declared-before-generation policy. The current alias and immutable named copy
are byte-identical; no 71434 source or later-stage member is an input to
71435.

Cohort 71435 completed byte-identical source freeze, compile/static
validation, V4 materialization, claims preparation, and full-cold Atlas
analysis for all 28 members. Its first and only compiled-promotion campaign
then failed closed on every member. The exact declaration and canonical
failure record are archived as `B2-GENERATED-COHORT-71435-DECLARATION.json`
and `B2-GENERATED-COHORT-71435-FAILURE.json`. Cohort 71435 is permanently
non-admissible; none of its source, compiled, materialized, claims, analysis,
or passing criterion evidence may be reused or salvaged.

The retained evidence exposed four independent defects. Lighting compared
authored floor-light tile count with an unrelated count of spawn navigation
components; nine maps also had true compiled-dark spawns because source
coverage used floor-level horizontal reach instead of exact spawn-eye 3D LOS.
Nine maps scattered starts across multiple standing components, and twelve
route summaries discarded wall-detour distance after source connectivity.
Compiled promotion remained strict. The replacement implementation now binds
the exact compiled floor-light region-ID set, records deterministic per-spawn
darkness diagnostics, proves spawn-eye source lighting, selects all eight
starts from one source standing component, and retains 16-unit standing-grid
geodesics in route scoring and claims.

Those fixes and the immutable 71435 failure record were committed before
replacement cohort `b2g26_final_71436` was declared. Its 28 entirely new
members use seeds 71436000 through 71436603 under the same all-or-nothing,
declared-before-generation policy. Its then-current alias and immutable named
copy were byte-identical; no 71435 source or later-stage member was an input
to 71436.

The first and only 71436 source-generation attempt failed at ordinal 13,
`b2g26_pits_71436301`, when the deterministic spawn placer could not find
eight clear, separated, map-spanning starts in one source standing component.
Only the fresh ordinal-0-through-12 prefix was written; the cold generation
never began and no source-freeze report or later stage was published. The
exact evidence is archived in
`B2-GENERATED-COHORT-71436-FAILURE.json`. Cohort 71436 is permanently retired
and its prefix cannot be retried, salvaged, reused, or substituted.

Implementation fix commit `57f4082` and the immutable 71436 named declaration
and failure archive all preceded the declaration of replacement cohort
`b2g26_final_71437`. Its 28 entirely new members use seeds 71437000 through
71437603 under the same all-or-nothing, declared-before-generation policy.
The alias and immutable 71437 named copy were byte-identical when declared and
remain so as a fail-closed historical boundary. No 71436 artifact was reusable
by 71437: all primary, cold, compiled, materialized, claims, analysis,
validation, Dyn, and report paths had to be fresh. In particular, generation
did not reuse archived root
`/home/raymondj/multires-artifacts/atlas-v1/B2/generated-final-71436-73d55811`;
it began in the fresh authority-bound root
`/home/raymondj/multires-artifacts/atlas-v1/B2/generated-final-71437-${ATLAS_AUTHORITY_SHA256:0:8}`
with new empty `source` and `source-cold` directories and reserved the fresh
exclusive sibling report path
`/home/raymondj/multires-artifacts/atlas-v1/B2/generated-final-71437-${ATLAS_AUTHORITY_SHA256:0:8}-report.json`.

The first and only 71437 source-generation attempt failed at ordinal 10,
`b2g26_canyon_71437202`. Final geometry had 487 locally legal candidates in
nine source standing components, but every component failed the required
1024-by-1024 map span. Deterministic forensics identified lane-wall static blockers
as the first stage to split the last map-spanning component;
later cover, corner, and objective blockers reduced the candidate pool but did
not cause the first loss of admission. Only the fresh ordinal-0-through-9
prefix was written. Cold generation never began, no source-freeze report or
later stage was published, and exact evidence is archived in
`B2-GENERATED-COHORT-71437-FAILURE.json`. Cohort 71437 is permanently retired.
Its prefix and every associated artifact and path cannot be retried, salvaged,
reused, or substituted. No replacement cohort has been declared.

Before any replacement may be declared, source construction reserves four
connected standing-volume bands around one deterministic arena perimeter and
certifies eight exact final spawn witnesses on that ring. Every later standing
solid is admitted through one protected-domain check; compound hallway,
lane-wall, and staircase assemblies are admitted and emitted all-or-none. The
final selector consumes the certified witnesses only after rechecking local
clearance and escape, separation, dual-axis span, and one exact source standing
component. Lane-wall endpoints begin at the ring's inner boundary, preserving
the two sight-blocking walls and central gaps that define lane arenas without
entering the reserved circulation volume. These implementation invariants do
not authorize generation; a separate clean commit must still declare an
entirely fresh cohort and artifact root.

## Offline workflow

The commands in this section use `COHORT_ID` from the committed authoritative
replacement declaration. It must never name retired cohort 71426, 71427,
71428, 71429, 71430, 71431, 71432, 71433, 71434, 71435, 71436, or 71437. There
is currently no declaration authorized for this workflow; the commands below
remain a template for use only after an implementation fix and a separately
committed fresh declaration with entirely new members and paths.

Generate and compile the BSP beside its source files first. Then materialize
only the V4 hook candidates under the pinned B1 authorities. Materialization
discovers the first grounded compiled Pmove landing without constraining it to
the generator hint; independent analysis later requires the sealed exact
landing and ordered trace to replay identically:

```sh
python tools/materialize_hook_claims.py \
  --bsp /isolated/B2/generated/b2claim_0000.bsp \
  --meta /isolated/B2/generated/b2claim_0000.meta.json \
  --runtime-sidecar /isolated/B2/generated/b2claim_0000.json \
  --output-attestation /isolated/B2/generated/b2claim_0000.hook-materialization.json \
  --cm-oracle /isolated/B1/q2-cm-oracle \
  --pmove-oracle /isolated/B1/q2-pmove-oracle \
  --hook-oracle /isolated/B1/q2-hook-oracle \
  --fall-oracle /isolated/B1/q2-fall-oracle \
  --hook-parity-attestation /isolated/B1/hook-parity-pullspeed-1700.json
```

Materialization fails closed unless exactly six unique measured geometries
replay, pass actual source/target L1 legality, and are atomically sealed with
their V4 attestation. V2/V3 candidates, attestations, and runtime sidecars are
rejected rather than converted or retried. The example paths below
deliberately keep every stage and report separate. Once
all 28 declared compiled maps have canonical attestations and admissible
runtime sidecars, prepare the v3 claims:

```sh
python tools/run_generator_claim_campaign.py prepare \
  --declaration docs/multires/B2-GENERATED-COHORT-DECLARATION.json \
  --materialized-dir "/isolated/B2/${COHORT_ID}/materialized" \
  --claims-dir "/isolated/B2/${COHORT_ID}/claims" \
  --output "/isolated/B2/reports/${COHORT_ID}-claims-prepare.json"
```

Run the B2 Atlas analyzer with the matching claim from the claims root and write
all eight Atlas artifacts to the separate analysis root. The normal path
remains `candidate`/pending until its independent full-cold rebuild and
verifier replay succeed. Do not use `maps/compile.sh` for this gate because it
copies BSPs into a runtime map directory. Then validate exact claims and
analysis membership before any per-map validator runs:

```sh
python tools/run_generator_claim_campaign.py validate \
  --declaration docs/multires/B2-GENERATED-COHORT-DECLARATION.json \
  --claims-dir "/isolated/B2/${COHORT_ID}/claims" \
  --analysis-dir "/isolated/B2/${COHORT_ID}/analysis" \
  --b1-gate docs/multires/B1-GATE.json \
  --output "/isolated/B2/reports/${COHORT_ID}-compiled-validation.json"
```

Any missing or unexpected file in either root rejects the complete campaign
before `validate_generated_map` is called. Analysis manifests are passed by
their explicit analysis-root paths; no adjacent lookup is performed. Campaign
reports remain canonical across identical inputs. The full-cold proof does
carry integer elapsed milliseconds because the frozen 300-second limit is an
admission fact; semantic Atlas-manifest comparison excludes only the
independently sampled peak-RSS field.
