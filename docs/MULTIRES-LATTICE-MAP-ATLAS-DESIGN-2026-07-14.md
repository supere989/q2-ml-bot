# Multi-Resolution Lattice and Map Atlas Specification

Status: Externally reviewed and accepted specification v1  
Date: 2026-07-14  
Target branch: `feature/multires-map-atlas-v1` from `feature/rust-lattice`  
Owners: q2-ml-bot, q2-lithium-3zb2, and q2-ml-client maintainers

Owner directive: this is a one-way replacement. The previous trainer/model
lineages are retired, no operational fallback or rollback package is created,
and no legacy policy, optimizer, Dyn snapshot, or runtime may be selected by a
resume or deployment path. Existing historical artifacts are evidence only;
they are not an operational fallback.

## 1. Purpose

This specification defines a map-static, multi-resolution spatial atlas and a
per-client dynamic Lattice for Quake II bot training and inference. It also
defines the map-analysis pipeline used to learn design distributions from
stock and licensed community maps, the metadata emitted by the procedural map
generator, advisory route guideposts, stance-aware navigation, hazard recovery,
and the breaking observation/action boundary required by the next fresh model.

The design must improve spatial accuracy without moving collision authority out
of the Quake II engine, leaking privileged state to public actors, disturbing
the production server's 10 Hz frame pacing, or turning Lattice hints into direct
action authority.

## 2. Baseline and invariants

The following existing contracts remain authoritative:

- The network-native client harness is the primary trainer.
- Public collection uses four protocol-34 clients and leaves two human slots.
- The public and teacher servers, map queues, credentials, and runtimes remain
  isolated.
- Fire authorization and exact aim geometry come from current-frame engine
  visibility and exposure, never from a Lattice cell.
- The 64-unit target thermal overlay is per-client, expires within five ticks,
  and is never checkpointed.
- Persistent dynamic Lattice state is local to the rollout client/worker. There
  is no per-frame LAN Lattice RPC.
- Static map artifacts are compiled and attested on WSL. The production VPS
  must not compile maps.
- Generated maps preserve the v6 lighting, safe-headroom, spawn escape,
  horizontal-sandwich rejection, and lethal-void guard contracts.
- Existing checkpoints with invalid pitch or backward-motion behavior are not
  resume candidates.
- Any observation or action layout change is atomic across the game module,
  network client, Python harness, teacher receiver, rollout schema, and policy.

## 3. Goals

1. Represent player-scale collision and hazards near playable surfaces.
2. Supply traversable hazard guardrails and recovery instructions.
3. Model standing and crouched navigation accurately.
4. Provide advisory routes to weapons, ammunition, health, armor, powerups,
   runes, safe regions, and evidenced enemies.
5. Analyze authored BSPs using the same spatial vocabulary used to validate
   generated BSPs.
6. Extract distributional design priors without copying authored layouts.
7. Keep hot-path feature generation below the live collection budget.
8. Serialize every static and dynamic artifact deterministically for
   cross-host attestation and rollout fencing.
9. Start the new observation/action contract as a fresh model lineage.

## 4. Non-goals

- Replacing engine BSP collision or `pmove` with voxel collision.
- Materializing a dense 4-unit volume over the map bounding box.
- Maintaining four independent live heat stores.
- Authorizing fire, selecting exact aim points, or exposing through-wall enemy
  state through route guideposts.
- Reconstructing original `.map` authoring brushes from arbitrary BSPs.
- Copying room graphs, coordinates, or portal layouts from community maps.
- Downloading or redistributing maps without provenance and license review.
- Running BSP analysis or map compilation on the public VPS.
- Resuming the new policy from a shape-compatible legacy checkpoint.

## 5. Coordinate system and hulls

All layers use Quake world coordinates and a shared map origin. Let
`origin[a] = 256 * floor(integer_model0_mins[a] / 256)` using mathematical
floor on every axis. Model bounds are integer-rounded from the pinned collision
oracle before snapping. The origin is part of the atlas content hash. Cell
sizes are powers of four so every parent relationship is exact:

```text
L0:   4 units
L1:  16 units
L2:  64 units
L3: 256 units
```

The normative transforms are:

```text
index(world, level)[a] = floor((world[a] - origin[a]) / cell_size[level])
center(index, level)[a] = origin[a] + (index[a] + 0.5) * cell_size[level]
parent(index)[a] = floor(index[a] / 4)
```

The child set of parent `p` is the closed 4x4x4 block whose minimum child
index is `4*p`. Indices are signed `i32`; serialized counts are `u64`. Maps
whose snapped index ranges exceed `i32` are rejected. Atlas, Dyn, thermal,
Python, and Rust use this same transform. Originless/world-zero grids and
truncating signed division are forbidden.

Quake II player hulls are:

| Stance | Mins | Maxs | Size | L0 span by volume |
|---|---|---|---|---|
| Standing | `(-16,-16,-24)` | `(16,16,32)` | 32x32x56 | 8x8x14 |
| Crouched | `(-16,-16,-24)` | `(16,16,4)` | 32x32x28 | 8x8x7 |

Continuous positions can intersect one additional cell on an axis at a cell
boundary. Atlas generation therefore uses engine box traces and conservative
intersection rather than assuming grid-aligned hulls.

## 6. Two-store architecture

The four resolutions are shared index spaces, not four equivalent live
overlays. Storage is explicitly namespaced:

- `Atlas[L0..L3]` is map-static, content-addressed, read-only at runtime, and
  shared by every client in a map epoch.
- `Dyn[L2..L3]` is per-client, local to the rollout process, and never mixed
  into atlas bytes.

Wire fields, checkpoint keys, and query APIs always name the store, such as
`atlas_l1_clearance` or `dyn_l2_threat`; a bare `lattice_l2` name is invalid.

### 6.1 Static map atlas

The static atlas is produced after BSP compilation and shared by every client
on that map.

| Level | Storage | Primary contents |
|---|---|---|
| L0 / 4u | Sparse chunked bitplanes | Exact surface/hazard narrow bands, raw contents, hull-expanded forbidden origins, hook geometry |
| L1 / 16u | Sparse navigation cells | Standing/crouched clearance, support, traversal edges, hazard cost-to-safety |
| L2 / 64u | Derived tactical map layer | Regions, exposure summaries, objective costs, recovery basins |
| L3 / 256u | Derived strategic map layer | Region graph, long routes, map priors, design summaries |

L0 is never allocated as a full map AABB or complete floor fill. Reachability
is seeded from oracle-clear deathmatch spawns, item/teleporter destinations,
and a stance-aware L1 flood. Runtime L0 retention is restricted to chunks that
intersect:

- solid/clip surface bands no thicker than 24 units
- hazardous contents expanded by the larger relevant player hull
- spawn columns
- validated hook corridors
- mover envelopes

An optional one-chunk boundary band may be retained around the reachable
flood. Unreachable skybox and exterior volumes are not voxelized. L1 origin
predicates and validated edges, not L0 free-space fill, are the navigation
authority.

### 6.2 Dynamic per-client Lattice

Dynamic gameplay state remains per-client and local to the collector:

- L2 holds short-range tactical state, target thermal data, evidenced enemy
  memory, local hazard events, and transient opportunity/readiness.
- `Dyn[L3]` is a derived mip of persistent `Dyn[L2]` encounter, death,
  objective, and route-experience channels. It has no independent v1 deposit
  path. Thermal, raw readiness plumes, and other ephemeral channels are
  excluded from the mip.
- Static atlas data is referenced by content hash and is not copied into every
  client checkpoint.
- Thermal data, raw L0 occupancy, and cached recovery vectors are never
  checkpointed.

Generator `.lattice.json` objective/danger priors are inputs to atlas creation
and Dyn preload. They are not a third runtime store.

### 6.3 Snapshot and LAN contract

- Per-frame transport contains zero Atlas or Dyn bytes.
- At each map epoch, workers load Atlas by manifest digest from the attested
  map bundle. A missing or mismatched digest fails closed.
- PPO lease/batch `lattice_payload` contains Dyn only, with magic, schema,
  origin, L2/L3 cell sizes, atlas SHA-256, map identity, environment steps, and
  client count.
- L0 chunks, atlas graphs, thermal tracks, and cached recovery directions are
  forbidden in Dyn payloads.
- Compressed Dyn payload has a 2 MiB soft and 8 MiB hard four-client limit.
- Checkpoints store Dyn and reference `atlas_sha256`; restore refuses a missing
  atlas rather than silently substituting unrelated priors.
- The authoritative multires snapshot uses a new Rust magic (`Q2LAT002` or
  later) and includes all mass/confidence required for exact feature rebuild.
  Legacy Python JSON/gzip is non-authoritative after cutover.

## 7. Static atlas representation

### 7.1 L0 chunks

The reference chunk is 16x16x16 L0 cells, covering a 64-unit cube; one L0
chunk AABB therefore equals one L2 cell AABB. Each chunk uses structure-of-
arrays bitplanes or compact integer planes.

World/clip contents from model 0 and collision brushes:

- solid and window
- playerclip
- monsterclip, diagnostic only and never a player blocker by itself
- water, slime, lava, and mist/nonsolid
- ladder
- current direction bitfield

Entity-expanded fields from the entity string and brush submodels:

- hurt
- push/gravity
- teleport trigger
- mover solid in its declared reference pose and swept envelope
- areaportal metadata

Surface-adjacent fields:

- sky
- slick
- warp
- nodraw, diagnostic only
- hookable surface

Derived fields:

- standing-forbidden origin
- crouched-forbidden origin
- confidence/unknown

Void and lethal drop are derived hazards, not BSP contents bits. Generator kill
planes are classified from their `trigger_hurt`, catch surface, and marker
metadata rather than being mislabeled as sky contents.

Binary fields use bitplanes. Contents and severity use compact integer planes.
Floating-point values are not stored per L0 cell.

### 7.2 L1 navigation cells

L1 cells represent possible player-origin regions, not occupancy of a
16-unit cube in isolation. `standing_origin_clear(c)` is true only when an
oracle-clear supported standing-origin sample exists in `c`; crouched clearance
uses the crouched hull. `safe_to_stand(c)` additionally requires safe support,
hazard-expanded clearance, and an allowed floor-normal class. The cell center
is tried first, followed by up to four deterministic stratified samples.

Each materialized L1 cell records:

- standing-clear
- crouched-clear
- safe-to-stand
- supported floor and floor normal class
- clearance height
- hazard type bitset and severity
- signed or unsigned hazard clearance
- scalar cost-to-safety
- region identifier
- confidence and evidence source

Traversal edges are typed:

- walk
- strafe-capable walk
- step
- jump
- controlled drop
- crouch enter/hold/exit
- water transition
- mover
- teleporter
- hook

Every edge includes required stance, cost, risk, evidence, and validation
version. Hook edges additionally reference an anchor, landing region, expected
physics identity, and validation trace. Absolute region IDs are map-ephemeral
and are never exposed directly to the policy.

Edge validation follows engine movement semantics:

- walk/strafe uses the stance hull, `MASK_PLAYERSOLID`, an allowed floor
  normal, and height delta no greater than `STEPSIZE=18`
- step uses the actual up-18, forward, down engine sequence rather than a
  diagonal trace through the riser
- jump and controlled drop use `q2-pmove-oracle`, which executes the exact
  q2-ml-client/Yamagi `Pmove` implementation against the pinned collision map
  with declared usercmd sequences, gravity, air acceleration, clip/slide,
  water, step, and landing semantics; generator `JUMP_H=50` is a prior, not
  the collision law
- ladder requires ladder contact
- water edges include waterlevel and current semantics
- slick/current edges retain control modifiers
- mover edges carry blocker identity and state confidence
- hook edges require a solid non-sky anchor and `q2-hook-oracle`, built from the
  exact q2-lithium hook attach/pull implementation and its full-velocity-
  overwrite behavior; the result is bound to the game-module physics identity

Edges not validated by the collision oracle are omitted or explicitly unknown;
they are never guessed from generator room adjacency.

### 7.3 Conservative aggregation

Static parent values are derived deterministically:

- hazard severity is the maximum child severity
- clearance is the minimum child clearance
- a parent is passable for a stance only when the required child corridor is
  passable for that stance
- a coarser stance-passable cell contains a finer clear origin that can reach
  the relevant cell boundary through stance-legal child edges; majority-clear
  aggregation is forbidden
- parent cost is the minimum child cost among stance-reachable children;
  blocked cost uses a fixed positive-infinity sentinel, never NaN
- static confidence is the minimum child confidence among children that
  participate in the stored aggregate; route confidence is computed later
- contents flags are a bitwise union

Hazards and blocked cells must never disappear through averaging.

## 8. Collision, visibility, and surface authority

The analyzer shall load every admitted BSP through the same pinned Yamagi
collision module used by the network client: map load, point contents, and box
trace. The reference implementation is a batch CLI backed by a reusable C
collision library from q2-ml-client/Yamagi; Rust may call that CLI/library
offline, while the runtime Rust crate never parses BSP collision. A separate
BSP parser may read entities, faces, lightmaps, visibility, and metadata only.
No Atlas cell or edge is marked solid, supported, passable, or hookable without
an oracle result.

Movement trajectories use a second entry point in the same offline tool,
`q2-pmove-oracle`, that executes the exact pinned `Pmove` source rather than an
independent ballistic approximation. Its physics identity hashes collision and
pmove source/build identity plus all movement cvars/constants that affect the
trajectory. Jump/drop edges are omitted if the trajectory cannot be reproduced
through this oracle.

Lithium hook trajectories use `q2-hook-oracle`, compiled from shared hook
attach/pull code in the pinned q2-lithium game-module revision. Its identity
hashes the source/build and hook parameter block, including pullspeed and the
full-velocity overwrite law. Golden trajectories must match an isolated q2ded
replay before hook edges are emitted. If the helper or parity proof is absent,
Atlas v1 emits zero hook edges; it never substitutes a CM-only estimate.

Stationary and swept player tests use the exact `MASK_PLAYERSOLID` equivalent
used by `pmove`. Support probes reproduce client ground checks. Atlas sightline
summaries use `MASK_SHOT`, but never feed live fire authorization. PVS and
areaportal state are coarse filters only; mover state invalidates cached
visibility summaries until re-traced.

Required analysis includes:

- model-0 bounds and BSP identity/version
- brush contents and clip contents
- standing and crouched stationary hull traces
- swept hull traces for traversal edges
- support and safe landing tests
- cluster PVS as a coarse potentially-visible filter
- explicit traces for actual sampled occlusion/exposure
- doors, lifts, trains, buttons, teleporters, hurt triggers, and brush submodels
- entity and surface lighting plus compiled lightdata diagnostics

PVS is never treated as line of sight. Dynamic movers produce stateful,
confidence-tagged edges rather than unconditional static corridors.

## 9. Hazards and recovery

### 9.1 Hazard sources

Static hazards include lava, slime, hurt volumes, void-facing ledges, lethal
drops, crushing movers, and map-specific lethal contents. Dynamic environmental
damage can increase confidence or add a bounded local event, but combat damage
does not populate the environmental hazard channel.

Two L0 masks are retained:

1. raw hazardous-space intersection
2. hazardous space expanded by the selected player hull

The second mask identifies forbidden player-origin positions.

### 9.2 Scalar recovery field

Recovery is based on scalar cost-to-safety, not a hand-authored vector. Safe
cells seed a deterministic multi-source graph solve over L1/L2 traversability.
The runtime derives the recovery direction from the best decreasing-cost edge.

The solver observes walls, stance, movers, drops, and hazards. It cannot diffuse
through solid geometry merely because two cells are close in Euclidean space.

Static cost-to-safety is computed offline over Atlas edges with no mover
dependency and stored at L1. Atlas L2 contains min-pooled coarse costs. Movers
and dynamic environmental hazards are handled by a bounded online repair over
the current two-L2-cell neighborhood, capped at 4096 L1 nodes. A full field is
never recomputed per frame. Best-edge ties resolve by lower cost, edge-type
enum, and then neighbor `(iz,iy,ix)`.

The public policy sees only the advisory recovery bundle defined below:

- hazard strength and type
- hull clearance
- time-to-impact when velocity makes it meaningful
- primary walk recovery direction in yaw-local coordinates
- alternate route direction when a real branch exists
- recovery cost and confidence

Solver-internal best-edge type, required stance, forced action, and
`hook_was_necessary` are teacher/reward tags, not public policy features.
Walking is preferred in the public solve, and public cost/directions exclude
hook edges. Hook necessity is a teacher/reward calculation using the frozen
`HOOK_RECOVERY_WALK_BUDGET_TICKS=15` at the 10 Hz game cadence; the constant and
tick rate are bound into the physics/runtime manifest. A hook is necessary only
when no walk path reaches safety inside that budget, a hook path does, and it
lowers repaired cost. The public policy receives no hook-solver candidate or
`must_hook` bit; it combines ordinary recovery fields with existing factual
hook zones. Every live hook attempt still requires a visible matching anchor
and current legal traces. The policy always chooses the button action.

## 10. Guideposts and objective fields

Guideposts are advisory observations derived from independent scalar objective
fields. A single blended vector is forbidden because it hides conflicts between
objectives.

Static or timed objective classes include:

- weapon classes
- weapon-specific ammunition
- health
- armor
- powerups
- runes
- control regions
- spawn egress

Safe recovery is represented only by the dedicated recovery block and is not
an objective-candidate class.

The public policy receives at most four candidates. Each provides one-hot
objective class, yaw-local direction, traversal cost, route risk, confidence,
and an availability **belief** based on the client's own observations and item
timing. Exact global item timers, required actions, required stances, and forced
hook decisions are forbidden public features. The policy retains final
authority.

Enemy guideposts are dynamic and per-client. Public actors may use only current
visibility, audio, or thermal-class remembered evidence with the same maximum
five-tick TTL as the thermal overlay. Exact privileged enemy paths are
teacher/critic labels only and are never packed into the public policy
observation.

Training uses seeded guide dropout and guide-off evaluation. After guide
curriculum begins, global guide-off probability is at least 0.20 and per-class
dropout is at least 0.30, seeded from map seed, policy version, client index,
and tick bucket. Guide fields never authorize firing, never enter exact aim
anchoring, and never contain solver answer bits.

## 11. Authored-map ingestion and design learning

### 11.1 Corpus admission

Every archive enters a quarantine area and receives:

- archive and BSP SHA-256 hashes
- source URL or manual origin
- author and license/redistribution record
- path traversal, symlink, case-collision, decompression-ratio, and executable
  content checks
- classic `IBSP` version 38 and structural validation; other BSP families use
  a separate future lane
- canonical ID, aliases, and exact/near-duplicate classification
- analysis-only versus redistributable eligibility

Admission also bounds total uncompressed bytes, member count, entity-string
size, nested archives, and duplicate case-folded names. Absolute paths, `..`,
symlinks, and executable/script payloads are rejected. Lump ranges must be
in-file, the entity string parseable, and model 0 present. Custom assets are
inventoried; missing assets reduce confidence and prohibit live admission.
No downloaded executable, game module, installer, or script is run. No map is
installed into a live runtime automatically.

The first corpus is the locally installed stock `q2dm1` through `q2dm8` set.
Community maps are out of the initial prototype. Default corpus policy is
analysis-only: BSP bytes and custom assets are not redistributed. Community
redistribution requires a written grant covering the BSP and every custom
asset; otherwise only aggregate priors, hashes, and the upstream URL leave
quarantine.

### 11.2 Extracted design priors

The corpus contributes anonymized distributions rather than coordinates or
complete graphs:

- playable area and vertical range
- ceiling/headroom bands
- corridor and chokepoint widths
- standing versus crouch-only traversal ratio
- path length, cycle, branching, and centrality distributions
- cover density and sightline length distributions
- spawn separation and spawn-to-resource costs
- item economy and timing relationships
- hazard adjacency and recovery distance
- lighting coverage and contrast
- hook opportunity and required-action frequency

The prior pack contains integer/fixed-point aggregate histograms, percentiles,
coarse counts, and source hashes, but no world coordinates, room/portal graphs,
edge endpoints, or per-map adjacency matrices. Near-copy detection compares
only binned descriptors such as degree histograms, item-class multisets, and
height-band mass with documented thresholds; it stores no reverse-mappable
portal signatures.

## 12. Generator integration

The existing generator remains the geometry author. Design priors may bias only
bounded style mixture and the declared generator knobs: `occupied_density`,
`corridor_prob`, `hallway_ratio`, `terrace_levels`, `tower_prob`, `lane_prob`,
`lava_prob`, `extra_arena_prob`, `arena_cover_range`, `corner_range`, and
`large_building_ratio`. They never inject source coordinates, rooms, entity
placements, or graph fragments. Stock/community routes derive from entity
origins plus Atlas L1; generator room-graph routes are only claims to revalidate.

For every generated map:

1. Generate `.map` source and current sidecars.
2. Run existing source-level geometry, lighting, spawn, and safety validation.
3. Compile BSP on the WSL map farm.
4. Run the authored-map analyzer on the compiled BSP.
5. Produce the static atlas, traversal graph, visibility summary, and design
   signature.
6. Compare compiled-world results with generator claims.
7. Reject or repair disconnected spawns, unsafe stance transitions, uncontained
   hazards, dark regions, invalid hooks, or excessive prior divergence.
8. Attest the accepted atlas and analysis artifacts in the map bundle.

Current bundle v2 consumers remain unchanged during offline prototyping. A new
bundle version is required before the atlas becomes a mandatory runtime
artifact.

Acceptance gates are split:

- **Analysis quality** for any BSP requires oracle load, parse, reachable spawn
  analysis, hazard classification, deterministic Atlas, and declared confidence.
- **Generator promotion** applies existing v6 source/light/headroom/spawn/
  lethal-guard contracts plus Atlas-versus-generator claim consistency.

Stock/community maps do not fail because they lack `_ml_*` tags, 98% tagged
floor-light coverage, or generator-specific headroom conventions when their
engine traversal is valid.

## 13. Artifact contract

Large micro-resolution data is not stored in JSON or git. An offline map
analysis bundle contains:

```text
<map>.analysis.manifest.json
<map>.atlas.bin.zst
<map>.navigation.bin.zst
<map>.visibility.bin.zst
<map>.design-signature.json
<map>.routes.json
```

The manifest includes:

- schema and byte-order version
- BSP SHA-256 and size
- analyzer, collision-engine, generator, and physics versions
- shared origin, bounds, resolutions, and chunk dimensions
- player hull definitions
- channel table and persistence class
- per-artifact SHA-256, uncompressed size, and cell/chunk counts
- limitations and confidence summary
- provenance/corpus reference when applicable

Canonical digests cover the uncompressed little-endian payload. Compression is
a zstd level-3 transport envelope with no content dictionary; compressed bytes
are not the authoritative cross-version digest. Writers reject NaN/Inf, sort
chunk/node keys by `(iz,iy,ix)`, and produce byte-identical uncompressed bytes
on supported hosts.

Bundle v2 remains the mandatory live artifact set during the prototype.
Analysis files travel separately or in a future bundle v3+ with explicit Atlas
keys. Current consumers ignore optional analysis artifacts. Atlas absence or
hash mismatch becomes fail-closed only after every consumer upgrades atomically;
it never restores VPS compilation as a fallback.

## 14. Observation, action, and privilege contract

The new policy is a breaking protocol generation. It uses a new observation
magic, action magic, public client wire version, teacher version, rollout
telemetry schema, model state schema, and runtime manifest. Exact integers are
assigned in the implementation batch, but every endpoint changes atomically
and rejects mixed versions.

### 14.1 Action

The logical rollout action retains eight slots:

```text
[forward, strafe, yaw, pitch, vertical_intent, fire, hook, weapon]
```

Forward, strafe, yaw, and pitch retain their continuous semantics. The current
jump boolean becomes a three-way categorical head:

```text
crouch_or_down | neutral | jump_or_up
```

The wire action POD stores this enum as one byte. The client projects it every
decision to `upmove=-320`, `0`, or `+320`. On ground/air the negative intent is
a held crouch request and the positive intent is a jump request. In water the
same signs mean swim down/up. Simultaneous crouch+jump is not representable.

Authoritative debug echo, outside the policy vector, includes requested enum,
applied upmove sign/magnitude class, actual `PMF_DUCKED`, and water-vertical
mode. Command admission compares requested enum with applied command sign;
actual stance is a resulting state and may legitimately lag. Policy-factual
state includes actual ducked, an engine standing-hull trace at the current
origin (`standing_blocked`), and water-vertical mode. Teacher demonstrations
must project real duck/jump/swim commands into this enum; legacy jump-only
samples are not trainable under the new schema.

### 14.2 Observation taxonomy

Every feature is named and belongs to exactly one class:

1. **Policy factual:** own state, current local sensors, visible/audible entities,
   attested map geometry, and bounded remembered evidence.
2. **Advisory ablatable:** objective guideposts, recovery summaries, and route
   suggestions that can be zeroed online.
3. **Teacher-only privileged:** oracle world state and labels unavailable to a
   public actor.

Teacher-only fields use a physically separate packing path and are asserted
zero/absent on the public conduit. They are not an optional tail on a public
packet. Public guide/recovery fields are limited to map-static facts or
per-client evidence with explicit confidence/TTL. Forbidden public fields
include exact through-wall enemy state, exact unobserved item timers, solver
best-action/required-stance bits, fire permission, and Lattice-derived aim.

### 14.3 Frozen policy-vector layout

The first multires policy vector is 298 floats:

| Block | Width | Contract |
|---|---:|---|
| Engine/client factual | 198 | Existing always-on 195 factual features plus `actual_ducked`, `standing_blocked`, and `water_vertical_mode` |
| Named Dyn memory | 24 | Current-cell scores/confidence; immediate thermal engagement; threat, opportunity, and self-fire directions/scores; survivability projection |
| Advisory recovery | 16 | Five hazard-type bits, strength, clearance, cost, confidence, primary yaw-local direction, alternate yaw-local direction, time-to-impact |
| Four objective candidates | 60 | Four fixed 15-float slots |

Every direction in the last three blocks uses yaw-local forward, Quake-right,
and world-up. The 24 Dyn fields are frozen as:

```text
0:5   current engagement, threat, opportunity, self-fire, confidence
5:9   immediate thermal direction xyz + heat
9:13  combat-threat direction xyz + score
13:17 opportunity direction xyz + score
17:21 self-fire direction xyz + score
21:24 win margin, effective-health norm, own-DPS share
```

Each objective candidate is:

```text
direction xyz, cost, risk, confidence, availability_belief,
class_onehot[8]
```

The eight classes are weapon, ammunition, health, armor, powerup, rune,
control, and spawn-egress. Empty/dropped slots are all zero. Enemy guidance
remains in the five-tick thermal/evidence path and is not a ninth static class.
An all-zero alternate recovery vector means no valid alternate branch.
Any feature-layout change after this freeze requires a new policy and protocol
generation.

### 14.4 Admission and resync

Collection remains synchronous four-client lockstep. A transition is trainable
only when every client produces a same-round authoritative echo with matching
action generation, forward/strafe, wrapped look, vertical intent/applied sign,
hook, weapon, and reconciled fire suppression. Existing map-epoch,
telemetry-gap, and action-state boundaries remain nontrainable whole-batch
resyncs. A one-client timeout while others remain live, movement/reliable
corruption, mixed schema versions, or teacher bytes on the public conduit is
fatal.

The map-epoch barrier dispatches no actions through BSP download/load. Actual
duck state is consumed as the following state; only a command-sign mismatch is
an action mismatch. Lifecycle changes that prevent a causal next state use the
existing whole-batch action-state resync.

### 14.5 Fresh lineage

Allowed initialization is either:

1. full random initialization of encoder, LSTM, actor heads, posture head,
   critic, auxiliaries, normalization, and optimizer; or
2. BC/distillation into a newly constructed 298-float/three-way-posture module
   graph using new-schema or explicitly projected demonstrations, followed by
   PPO with a reset optimizer.

Forbidden initialization includes any `torch.load` of a legacy policy or
optimizer, partial legacy backbone/LSTM/jump-head loading, a pre-schema resume
directory, rejected pitch/backward lineages, or dynamic Lattice state not
bound to the current Atlas digest. Process startup fails before the first
environment step unless observation/action dimensions, categorical
cardinalities, all wire/schema versions, and Atlas identity match one attested
runtime manifest.

## 15. Reward and curriculum contract

Rewards target irreversible outcomes and non-replayable progress rather than
actuator rates or dense oracle following.

A hazard episode opens only from environmental contents damage, Atlas hazard
entry, or evidenced void-edge proximity; combat damage cannot open it. The
episode ends at safe arrival, death, or timeout. It can pay one credit for each
new-best cost-to-safety milestone, with a fixed episode cap, and one safe-
arrival reward. It penalizes environmental damage/death and invalid hook
attempts. There is no per-tick gradient reward.

Reward eligibility is keyed by `(client_life_epoch, hazard_region_id)`. Timeout
closes eligibility without resetting its high-water mark. The same region
cannot open another rewarded episode until the client has remained beyond the
safe-clearance threshold for 30 consecutive ticks; a different hazard region
gets its own key. Death changes life epoch but retains the environmental-death
penalty. This makes timeout/re-entry and boundary oscillation structurally
unable to replay progress credit.

A hook-assisted bonus requires successful safe arrival and the debug-only
`hook_was_necessary` label. Crouch pays only on first validated crouch-edge
entry/completion or a refractory combat outcome; sustained ducking does not
pay. Unnecessary crouch is measured from actual `PMF_DUCKED` while standing is
clear and no crouch edge is active.

Hard bans:

- positive reward proportional to strafe, jump, crouch, or hook frequency
- open-ended reward for matching guide direction or scalar descent
- policy access to `hook_was_necessary` or solver best-action labels
- any fire/aim reward derived from Atlas, Dyn, or guide fields

Curriculum stages:

1. transport, posture, water vertical control, death-screen lifecycle,
   standing/crouched traversal, and action echo
2. hazard avoidance and walking recovery without enemies
3. hook-assisted recovery and controlled drops
4. pickups, objective guideposts, and guide dropout
5. target alignment and combat on generated maps
6. combat season on stock maps
7. full guide-off ablation
8. public-topology shadow canary

Combat remains the causal ladder: actionable exposure, alignment, permitted
fire, executed fire, hit, repeated hit, and kill.

## 16. Performance and storage budgets

Prototype hard budgets on representative 3584x3584x704 maps:

- no dense L0 allocation
- L0 has at most 1200 resident chunks and 16 MiB decompressed bytes
- total resident Atlas is at most 32 MiB per loaded map; build peak RSS is
  measured separately and capped at 512 MiB for the prototype
- four dynamic client lattices under 8 MiB combined, excluding model tensors
- each Dyn has at most 20,000 materialized cells
- after map-epoch preload, total Atlas+Dyn feature assembly for one accepted
  transition across all four clients has p99 under 0.5 ms on the WSL host;
  counters separate Dyn query, Atlas lookup, recovery, and guidepost time
- no per-frame atlas or Lattice payload on the LAN
- Atlas L0/L1/L2/L3 is loaded once per map epoch and shared read-only; disk
  install/decompression is measured outside the resident query budget
- full cost-field recomputation occurs only during offline Atlas construction;
  runtime uses the bounded local repair from section 9
- policy-facing queries use hierarchical coarse-to-fine lookup and never scan
  all microcells

## 17. Determinism, tests, and acceptance

Required tests include:

- negative-coordinate parent/child cases for nonmultiples at every level using
  mathematical floor
- golden collision-oracle fixtures for solid, window, playerclip, water,
  slime, lava, ladder, hurt submodel, sky/drop, and documented dilation
- standing/crouched passage, blocked-stand, and STEPSIZE 18 pass/fail fixtures
- jump/drop trajectories reproduce `q2-pmove-oracle` endpoints and landing
  state under the manifest physics cvars
- lava, slime, hurt, void, crush, current, and safe-drop fixtures
- 56--95-unit unsafe sandwich and 96-unit safe-headroom fixtures
- PVS-positive but MASK_SHOT-blocked occlusion, plus closed/open areaportal
- mover, teleporter, and hook-edge fixtures
- hook-oracle golden trajectories match isolated q2ded attach/pull/landing
  traces before any Atlas hook edge is admitted
- conservative aggregation property tests
- every finite non-safe recovery cell has a strictly lower-cost legal neighbor
  or an explicit mover-gated plateau classification
- exact Rust parity for graph/cost functions; 1e-6 Python/Rust parity only for
  pure f32 functions with identical operation order
- byte-identical atlas serialization across repeated runs and hosts
- observation/action size, magic, wire, echo, and generation admission tests
- same-seed 500-transition trajectory hash match in deterministic lockstep
- teacher-only fields absent on public packets
- guide-on/guide-off matched-seed evaluations
- prior/design-signature schema lint proving there are no source-map world
  coordinates, graphs, or edge endpoints
- bundle-v2 install succeeds with analysis artifacts present or absent

Before community ingestion, all eight stock deathmatch maps must load through
two independent collision-oracle process launches and produce pinned,
byte-identical Atlas hashes, reachable deathmatch-spawn counts, and item-class
multisets. Generated-map treatment batches must improve predeclared histogram
distances without lowering source/static validation pass rate, reducing style
entropy below its declared floor, or converging on near-identical layout hashes.

### 17.1 Analyzer admission

Each admitted BSP must:

1. pass quarantine, SHA-256, IBSP-38 lump, entity, and model-0 validation
2. load through the pinned collision oracle
3. contain at least two mutually reachable clear deathmatch spawns for analysis,
   and at least eight for six-player generated-map training admission
4. complete the stance-aware spawn-reachable flood without exceeding Atlas
   budgets
5. reproduce its uncompressed Atlas digest on a cold second build
6. emit a coordinate-free design signature

Generated-map revalidation additionally proves that oracle-clear spawns and
escape paths match generator claims, Atlas hazard coverage contains generator
danger volumes, lethal exterior remains guarded, and every published hook edge
has a current legal anchor and landing. These gates do not impose v6 authored
lighting/tag requirements on stock maps.

### 17.2 Required telemetry

TensorBoard and season JSON record:

- existing network-client admission, gap, map-epoch, stale, and echo counters
- requested crouch/neutral/jump-or-up rates, actual ducked rate,
  standing-blocked rate, water mode, command-echo match, and state resyncs
- true-view pitch/down-look, forward/backward command rates, and movement speed
- hazard evidence, bounded new-best credits, safe arrivals, environmental
  damage/deaths, and recovery time
- hook recovery success, invalid attempts, and raw hook rate as an audit only
- guide global/per-class dropout and matched guide-off task/combat metrics
- teacher-field violations on the public conduit, which must remain zero
- Atlas load/hash failures, resident/build RSS, cell/chunk counts, deserialize
  time, and the four resident-query timing components from section 16
- actionable exposure, alignment, fire permission, executed fire, hits,
  repeated hits, kills, hidden fire, and fixed-holdout aim errors
- Dyn cell counts, live/expired thermal tracks, snapshot bytes, and proof that
  thermal is absent from checkpoint manifests

Full command echo, oracle item timers, and `hook_was_necessary` remain debug or
teacher labels and are not policy inputs.

## 18. Deployment and one-way retirement

Deployment is staged and fail-closed:

1. Offline analyzer and atlas artifacts only.
2. Isolated WSL map-analysis and generator revalidation.
3. Optional map-farm analysis attachment without runtime consumption.
4. Atomic protocol/client/game/Python/teacher package in isolated runtimes.
5. Fresh-model curriculum in a new TensorBoard and checkpoint root.
6. Stock and generated quality seasons plus guide-off ablation.
7. Public-topology shadow with two human slots preserved.
8. Promotion only after every gate below passes.

Before stages 4 or 7, write a retirement manifest naming every legacy runtime,
policy, optimizer, Dyn snapshot, protocol schema, and map-bundle path that must
be absent from operational selection. The new package is installed atomically;
endpoints are never upgraded or downgraded independently. Cold restart must
reconstruct the new package from its attested source and artifacts, not select
an old package.

### 18.1 Promotion gates

**G0 identity and fresh-lineage retirement**

- one attested runtime manifest binds game module, client, Python, teacher,
  rollout, Atlas, bundle, observation, action, and policy schemas
- no legacy tensor/optimizer/Dyn load
- public packets contain no teacher-only bytes
- operational selectors contain no legacy runtime, policy, optimizer, or Dyn
  fallback

**G1 transport and echo** over at least 16,384 accepted transitions:

- failed rounds and echo timeouts are zero
- authoritative action echo acceptance is at least 97%
- vertical-intent command echo match is at least 99% outside declared resyncs
- water/land command-projection skew is zero
- map-epoch and whole-batch telemetry-gap recovery are exercised successfully;
  partial-client timeout remains fatal

**G2 posture and locomotion** on no-visible-target subsets where applicable:

- true-view down-look rate is at most 10%
- mean pitch is within +/-10 degrees while moving at least 96 units/s
- forward-command rate is not below backward-command rate, and backward rate
  is at most 40%
- unnecessary actual-crouch rate is below a threshold frozen before training
- jump, crouch, strafe, and hook do not collapse into constant/spam actions

**G3 Atlas, stance, and hazard:**

- stock-map determinism and all section-17 fixtures pass
- induced hazard scenarios improve safe-arrival and environmental-death rates
  over matched no-recovery baselines
- new-best credits remain bounded per episode with no boundary oscillation farm
- hook recoveries use engine-valid anchors; invalid attempts are not net-positive

**G4 guide dependence:**

- after curriculum stage 7, matched-seed guide-off task success degrades no
  more than 15% relative to guide-on for traversal/pickup/recovery tasks
- guide-off policy exceeds neutral/random baselines
- global guide dropout was never zero after stage 4 began

**G5 combat**, independently for generated and stock seasons of at least 16,384
accepted transitions each:

- actionable exposure, post-command alignment, permitted fire, executed fire,
  hit, repeated hit, and kill counts are all strictly positive in that order
- aligned-fire precision is at least 85%
- hidden fire is zero
- visible-contact yaw MAE is at most 12 degrees and pitch MAE at most 8 degrees,
  or a stricter frozen BC holdout bar applies
- guide-off evaluation does not reopen down-look/backward collapse or reduce
  combat events to zero

**G6 public shadow:**

- `maxclients=6` with four ML clients leaves two human slots
- no compile/analyze work runs on the VPS
- G1 remains green through human join/leave and map transitions
- a complete cold restart of the new runtime succeeds before promotion

Promotion is `G0 && G1 && G2 && G3 && G4 && G5_generated && G5_stock && G6`.
Lower loss, raw fire rate, or guide-on-only success is insufficient.

## 19. Rejection conditions

The prototype is rejected if any of the following is true:

- dense 4-unit map allocation or hot-path microcell scans
- static atlas state copied into every client checkpoint
- thermal or cached recovery vectors serialized as persistent memory
- fire/aim authorization derived from Lattice or guide fields
- public policy access to exact through-wall enemy state
- stance-ambiguous traversal or non-conservative hazard aggregation
- map downloads installed without quarantine and provenance
- generated maps accepted from source metadata without compiled-BSP reanalysis
- mixed observation/action/wire versions admitted
- new model initialized from a rejected or shape-compatible legacy checkpoint
- public deployment without guide-off, stock-map, cold-restart, and human-slot gates
- Atlas built without the pinned Yamagi collision oracle or with the wrong
  movement/shot mask
- stock maps rejected solely for missing generator-specific `_ml_*` contracts
- prior artifacts containing coordinates, portal graphs, or adjacency matrices
- dense per-tick cost-to-safety reward or public solver best-action labels
- vertical command projection/echo not admitted under section 14
- mandatory Atlas deployment that breaks bundle v2 before atomic bundle-v3
  consumer cutover

## 20. Frozen v1 implementation decisions

1. Collision and movement authority is a q2-ml-client/Yamagi reusable C library
   exposed through offline `q2-cm-oracle` and `q2-pmove-oracle` batch entry
   points. Hook authority is `q2-hook-oracle` built from pinned q2-lithium hook
   code and parity-checked against isolated q2ded. Rust orchestrates/builds
   Atlas; runtime Rust never parses BSP collision. Missing trajectory oracles
   cause jump/drop/hook edges to be omitted, never approximated.
2. L1 graph nodes are sorted `(iz,iy,ix)` and stored as deterministic CSR with
   `u32` offsets/targets and fixed edge records containing type, stance, flags,
   blocker, cost, risk, and confidence. Static costs use unsigned fixed point
   with 1/256-unit quantum and `0xffffffff` infinity.
3. Static cost-to-safety is stored at L1; Atlas L2 min-pools stance-reachable
   cost. Online repair is bounded as section 9 specifies.
4. Hook edges are excluded from public static recovery cost. The debug/teacher
   necessity test uses `HOOK_RECOVERY_WALK_BUDGET_TICKS=15` at 10 Hz.
5. Bundle v2 remains live during the offline prototype. Mandatory Atlas delivery
   is bundle v3 and cuts over all consumers atomically through the existing WSL
   farm/install attestation path.
6. The policy layout is the fixed 298-float, K=4 contract in section 14.
7. `Dyn[L3]` is a derived mip of persistent Dyn L2 channels only.
8. Rust `Q2LAT002+` is the authoritative Dyn snapshot. Thermal and other
   ephemeral channels remain outside it.
9. All Atlas levels are preloaded at map epoch under the 32 MiB/1200-L0-chunk
   gate. Maps exceeding it fail the prototype class rather than demand paging.
10. Community maps are analysis-only by default and remain outside prototype
   admission until stock/generated gates and quarantine automation pass.
11. Atlas artifacts pinned by a training checkpoint are retained until that
    lineage is archived; garbage collection cannot remove referenced hashes.
