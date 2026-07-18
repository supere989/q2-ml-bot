# Multi-Resolution Lattice and Map Atlas Execution Plan

Status: Active; B2 toolchain qualification is required before another final
generated cohort is declared
Date: 2026-07-14  
Methodology amendment: 2026-07-16
Normative specification:
`docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md`

Owner directive: this plan performs a one-way replacement. The previous
trainer/model lineages are retired, no operational fallback or rollback package
is created, and legacy artifacts remain historical evidence only. Any older
instruction to preserve or restore a runtime triple is superseded.

## 1. Outcome

Deliver a working, attested prototype that:

1. analyzes stock and generated IBSP-38 maps through exact collision, movement,
   and hook physics oracles
2. builds a deterministic static `Atlas[L0..L3]`
3. runs a per-client Rust `Dyn[L2..L3]`
4. feeds stance, hazard recovery, and four advisory objective candidates to a
   fresh 298-float recurrent policy
5. supports categorical crouch/down, neutral, and jump/up commands with causal
   engine echo
6. revalidates generated BSPs and learns coordinate-free map-design priors
7. deploys atomically through isolated WSL staging before any public shadow
8. starts a new model, optimizer, normalization, Atlas-bound Dyn state,
   checkpoint root, and TensorBoard root

Public promotion is not the definition of prototype completion. The prototype
is complete when the isolated WSL trainer passes Batches 0 through 7. Public
promotion requires Batches 8 and 9.

## 2. Execution rules

- Every agent reads `AGENTS.md` and the normative specification before editing.
- At most three construction agents run beside the integrating root.
- Agents receive exclusive repository/file ownership per batch.
- Use separate git worktrees/branches when two agents touch the same repository.
- Agents do not deploy, stop training, mutate services, or copy artifacts to
  another host. Only the root integration batch may do that after its gate.
- No batch starts until the previous gate report is written and green.
- B2 toolchain qualification is disposable, retryable, and non-admissible. It
  must pass before an immutable final generated cohort is declared.
- Once declared, a final cohort remains immutable and no-retry; qualification
  artifacts, maps, or passing subsets can never satisfy final-cohort evidence.
- This methodology amendment changes the normative design/plan digests. Every
  prior B1 seal and B2 admission artifact is historical for future gates; a
  fresh B1 seal bound to these exact documents is required before B2
  qualification. Reusing unchanged oracle binaries does not reuse their old
  seal or gate authority.
- Each batch produces commits, tests, artifact hashes, and a concise handoff.
- Existing user changes and unrelated dirty files are preserved.
- Secrets, complete client commands, and credential files never appear in logs,
  commits, or handoffs.
- Public and teacher queues, ports, prefixes, credentials, and runtimes remain
  isolated throughout.
- No compile or analysis job runs on the production VPS.

## 3. Branch and worktree layout

Root creates coordinated feature branches without changing the existing public
runtime:

| Repository | Base | Feature branch |
|---|---|---|
| `q2-ml-bot` | `feature/rust-lattice` | `feature/multires-map-atlas-v1` |
| `q2-ml-client` | `feature/ml-client-harness` | `feature/multires-map-atlas-v1` |
| `q2-lithium-3zb2` | `ml-wip-20260611` | `feature/multires-map-atlas-v1` |

Per-agent worktrees use batch-qualified branches such as
`agent/b1-atlas-core`. Root integrates with ordinary merges or cherry-picks and
deletes worktrees only after the batch gate passes.

## 4. Batch matrix

| Batch | Parallel work | Primary output | Runtime impact |
|---|---|---|---|
| B0 | Root only | Baseline, branches, frozen manifests | None |
| B1 | Three agents | Physics oracles, Atlas schema/core, quarantine/parser | None |
| B2 | Three agents | Stock/generated Atlas builder, Dyn snapshot, claim validator | None |
| B3 | Three agents | Recovery/guide fields, priors, optional bundle-v3 tooling | Offline only |
| B4 | Three agents | Atomic game/client/Python protocol generation | Isolated binaries only |
| B5 | Three agents | Policy, rewards, curriculum, metrics, deterministic suite | Isolated tests only |
| B6 | Root plus auditors | WSL staging deployment and transport proof | Isolated WSL |
| B7 | Root plus auditors | Fresh model initialization and curriculum start | Isolated/new run |
| B8 | Root plus evaluators | Generated, stock, and guide-off seasons | Public topology shadow only after gates |
| B9 | Root only | Rollback drill and public promotion decision | Public only if all gates pass |
| B10 | Three agents | Licensed community corpus expansion and generator priors | Post-prototype, offline |

## 5. B0 — Baseline, scope, and legacy retirement

Owner: root integrator only.

### Work

1. Confirm all three repository bases and dirty-state ownership.
2. Create coordinated feature branches and agent worktrees.
3. Record current commits, runtime manifests, wire versions, model dimensions,
   map bundle version, and active service topology without recording secrets.
4. Stop and retire the active legacy trainer, clients, and TensorBoard view.
5. Pin the accepted specification digest into the plan gate report.
6. Create an artifact namespace for offline Atlas builds and fixtures; keep
   large binaries outside git.
7. Define a retirement inventory for legacy game modules, network clients,
   Python trees, policies, configs, map bundles, and protocol schemas; none may
   remain selectable by the new runtime.

### Gate B0

- all three branches start from named bases
- worktrees are clean except known user-owned changes
- specification and plan digests recorded
- public/teacher services remain untouched until the atomic protocol cutover
- legacy trainer/model processes are stopped and the retirement inventory is complete

## 6. B1 — Parallel foundations

### Agent B1-A — Exact physics oracles

Repositories: `q2-ml-client`, then `q2-lithium-3zb2`; exclusive ownership of
new offline oracle code and shared physics helpers.

Deliver:

- `q2-cm-oracle`: batch map load, point contents, box traces, PVS/cluster query
- `q2-pmove-oracle`: exact pinned `Pmove` execution using the collision oracle
- `q2-hook-oracle`: hook attach/pull simulation from shared Lithium hook code
- machine-readable oracle request/response schema and physics identity
- golden q2ded parity harness for hook trajectories
- no runtime behavior change in this batch

Tests:

- solid/window/playerclip/water/slime/lava/ladder/current contents
- standing/crouched stationary and swept hulls
- STEPSIZE 18 sequence
- jump/drop endpoint and landing reproduction
- hook attach, full-velocity overwrite, pull, collision, and landing parity
- missing or failed trajectory oracle omits its edge type

### Agent B1-B — Rust Atlas schema and storage

Repository: `q2-ml-bot`; exclusive ownership of new Atlas Rust modules, binary
schema, and unit/property tests.

Deliver:

- shared origin/index implementation for 4/16/64/256 levels
- signed-floor negative-coordinate correctness
- L0 16-cubed sparse chunk bitplanes
- deterministic L1 node and CSR edge records
- canonical uncompressed serialization and zstd transport envelope
- manifest writer/reader with collision, pmove, hook, and specification digests
- strict cell/chunk/RSS/payload guards

Tests:

- parent/child property suite, including negative coordinates
- canonical key ordering and stable uncompressed SHA-256
- corrupt/oversized/mixed-schema rejection
- conservative hazard/clearance aggregation
- no dense L0 allocation path

### Agent B1-C — Corpus quarantine and BSP metadata

Repository: `q2-ml-bot`; exclusive ownership of quarantine, provenance, entity,
face/lightmap/PVS metadata parsing, and security tests. This agent does not
implement collision.

Deliver:

- IBSP-38 structural/lump/entity/model-0 validation
- archive quarantine with size/member/ratio/path/symlink/case/executable checks
- provenance and license records
- exact and near-duplicate classification
- entity/submodel catalog for spawns, items, movers, triggers, and teleporters
- stock-map fixture inventory for `q2dm1` through `q2dm8`

Tests include malicious archive fixtures, malformed lumps, missing model 0,
oversized entity strings, alias deduplication, and clean stock admission.

### Root B1 integration

- merge all three workstreams
- reconcile oracle/Atlas request formats
- run clean builds and full repository tests
- write `B1-GATE.json` containing commits, physics IDs, test counts, and failures

### Gate B1

- exact collision, pmove, and hook helpers build without changing live behavior
- Atlas empty/fixture artifacts serialize deterministically
- all eight stock maps pass structural quarantine
- no external/community downloads yet
- no service or cross-host deployment occurred

## 7. B2 — Atlas construction and compiled-world validation

### Agent B2-A — Atlas analyzer/orchestrator

Build the offline pipeline that calls the B1 oracles and produces:

- spawn-reachable, stance-aware L1 flood
- sparse L0 surface/hazard/spawn/hook/mover chunks
- standing/crouched origin predicates
- typed movement edges and confidence
- Atlas L2/L3 aggregates
- PVS plus MASK_SHOT trace summaries
- analysis manifest, Atlas, navigation, visibility, and design-signature files

### Agent B2-B — Rust Dyn and snapshot generation

Extend the current Rust Lattice without changing Atlas:

- explicit `Dyn[L2]` persistent channel store
- derived persistent `Dyn[L3]` mip
- thermal remains separate and ephemeral
- `Q2LAT002+` canonical Dyn snapshot with Atlas digest/origin
- batch payload soft/hard limits and map-epoch fencing
- named 24-float Dyn feature block in yaw-local coordinates

### Agent B2-C — Generator claim validator

Compare generated source/meta claims against compiled Atlas:

- implement the cheap BSP/CM invariant preflight independently of the
  generator and source/static validator
- spawn origin, column, separation, and escape
- lava/hurt coverage
- lethal exterior and guard containment
- lighting/lightdata diagnostics
- hook anchor and landing legality
- generator route claims versus Atlas cost/connectivity
- stock analysis quality separated from generated v6 promotion rules

### B2 toolchain qualification

Before declaring an immutable final cohort, run a disposable qualification
lane against the exact pinned generator, q2tool, CM/Pmove/hook authorities, and
validators. Qualification is retryable because none of its output is
admissible evidence. It must include:

1. a fresh B1 authority seal and gate bound to the amended design and plan;
   the prior `B1-authorities-909b1e46` seal is historical only
2. real q2tool-to-BSP-to-CM golden fixtures for engine-linked spawns, including
   the `+9` spawn-origin lift and room-ceiling boundaries where 104 and 105
   units fail the compiled 96-unit column criterion and 106 units passes
3. one disposable 28-map representative campaign through source validation,
   real compilation, the compiled-CM preflight, materialization/claims, full
   Atlas construction, and promotion validation
4. exact stage-membership, deterministic rebuild, timeout, and resource checks
5. a qualification report that is explicitly `non_admissible: true`

Qualification is green only when every golden and infrastructure preflight
passes and at least 20 of the 28 disposable maps complete the entire lifecycle
through promotion validation. That 20-map threshold is a generator/toolchain
signal, not final admission; it does not permit selection of a passing subset
or excuse a systemic stage failure.

### B2 final-cohort lifecycle

Only after qualification is green may root commit a fresh declaration and
invoke the immutable final lane. Its order is normative:

1. source generation and source/static validation
2. real q2tool compilation
3. cheap compiled-CM invariant preflight over every declared BSP, covering at
   least spawn identity, engine-linked stance, support, 96-unit column,
   separation, escape, basic hazard containment, and lightdata presence
4. hook materialization and claim preparation
5. full Atlas construction and deterministic cold rebuild
6. compiled promotion/claim validation
7. Dyn, performance, and assembled B2-gate evidence

The compiled-CM preflight must consume the BSP and pinned collision authority;
validating a copied `.map` source is not compiled-world validation.

Artifact states are distinct: **built** means staging bytes exist;
**published** means an exact stage was atomically made available but remains
non-admissible; **validated** means that stage passed its named independent
checks; **admitted** means every applicable final-cohort and B2 gate passed.
Reports and operators must not use these terms interchangeably.

### Gate B2

- fresh B1 seal/gate binds the exact amended design and plan digests
- `q2dm1` through `q2dm8` produce cold-rebuild-stable Atlas hashes
- pinned stock spawn counts and item-class multisets match
- B2 toolchain qualification is green and explicitly non-admissible
- all 28 maps in the immutable final declaration pass source/static,
  compiled-CM preflight, full Atlas, and compiled-promotion validation
- no generated claim can override oracle failure
- representative Atlas stays within 1200 L0 chunks, 16 MiB L0, and 32 MiB total
- resident four-client feature assembly p99 is below 0.5 ms
- four Dyn snapshots remain below 8 MiB combined and payload hard limits pass

Gate B2 proves offline artifact, oracle, determinism, and performance
integrity. It does not prove policy learning, targeting, locomotion, combat
quality, trainer cutover, or public readiness; those require B4 through B9.

## 8. B3 — Recovery, guideposts, priors, and bundle preparation

### Agent B3-A — Recovery and guide fields

- fixed-point L1 multi-source cost-to-safety
- L2 min-pooled costs
- capped 4096-node local mover/dynamic-hazard repair
- deterministic best/alternate descent vectors
- five hazard classes, TTI, confidence, and zero-vector alternate semantics
- hook necessity debug label using 15-tick walking budget
- four objective candidates with eight-class one-hot packing

### Agent B3-B — Design priors and generator feedback

- coordinate-free integer/fixed-point histograms from stock maps
- parameter-bias interface limited to the spec's generator knobs
- baseline/treatment generation with identical seeds
- histogram-distance and layout-diversity reports
- schema lint rejecting coordinates/graphs/endpoints
- no community input in the prototype

### Agent B3-C — Bundle and farm preparation

- optional analysis artifact delivery beside bundle v2
- proposed bundle-v3 manifest and installer tests
- v2 consumers remain unaffected
- missing/mismatched mandatory Atlas fails only in v3-capable isolated runtimes
- farm health reports analysis failure without restoring VPS compilation

### Gate B3

- every finite non-safe cell descends or has an explicit mover plateau
- recovery/guide packing matches the frozen 76-float spatial addition
- matched generated treatment improves predeclared prior metrics without lower
  static pass rate or diversity collapse
- v2 install works with analysis artifacts present and absent
- bundle v3 remains isolated and is not enabled publicly

## 9. B4 — Atomic protocol generation

This batch is developed in parallel but integrated as one indivisible package.
No endpoint is deployed independently.

### Agent B4-A — Game module and teacher telemetry

Repository: `q2-lithium-3zb2`.

- new observation/action/debug POD generation
- actual ducked, engine standing-blocked trace, water vertical mode
- requested vertical enum/applied upmove echo
- expanded generation-gated action debug
- teacher-only packing physically separate from public packing
- public assertion that teacher fields are absent
- existing fire-gate, death-screen, map lifecycle, and target geometry preserved

### Agent B4-B — Network client projection

Repository: `q2-ml-client`.

- new public wire/action version
- categorical vertical projection to signed upmove
- water semantics and command echo compatibility
- new obs/action sizes and reject-on-mismatch registration
- existing private generation bits and reliable hook/weapon requests preserved

### Agent B4-C — Python protocol and batch admission

Repository: `q2-ml-bot`.

- exact parser/packer parity with C and client
- frozen 198 factual fields
- 24 Dyn, 16 recovery, and 60 guide fields
- eight logical rollout actions with three-way posture head
- authoritative command echo admission and existing whole-batch resync behavior
- runtime attestation updated for all schema dimensions/cardinalities

### Gate B4

- clean C rebuild after shared-header changes
- client, server, and Python struct sizes/magics agree
- old/new mixed endpoints reject each other
- forward/strafe/look/vertical/fire/hook/weapon echoes reconcile causally
- partial-client timeout remains fatal
- map-epoch and telemetry-gap barriers remain nontrainable whole-batch boundaries
- public packet teacher-field violations equal zero

## 10. B5 — Policy, rewards, curriculum, and validation

### Agent B5-A — Policy and trainer

- new 298-input module graph
- three-way categorical vertical head and neutral-biased initialization
- named feature offsets; no anonymous last-24 indexing
- ONNX/export changes if export remains required
- fail-closed legacy checkpoint/resume rejection
- guide dropout seeded by map/policy/client/tick bucket

### Agent B5-B — Hazard/stance rewards and telemetry

- `(life_epoch,hazard_region_id)` reward episodes
- retained high-water mark across timeout
- 30-tick safe-clearance rearm
- bounded new-best credits and one arrival credit
- sparse crouch traversal/outcome credit
- hook necessity label debug-only
- all section-17 telemetry and season JSON fields

### Agent B5-C — Deterministic and quality suites

- same-seed 500-transition hash proof
- posture/water/crouch fixtures
- induced hazard and hook-recovery campaigns
- guide-on/off matched-seed evaluator
- fixed aim/combat holdout and combat-ladder report
- retirement-manifest and new-runtime cold-start validation

### Gate B5

- all repository and cross-repository tests pass
- no legacy tensor, optimizer, or pre-Atlas Dyn can load
- no rate-based positive reward exists for strafe/jump/crouch/hook
- reward timeout/re-entry cannot replay credit
- deterministic lockstep hashes match
- guide-on/off evaluator and season report are runnable before training starts

## 11. B6 — Isolated WSL deployment

Owner: root. Construction agents become read-only auditors.

### Preconditions

- B0–B5 green
- retirement inventory complete
- no unrelated active public process is stopped
- staging checkout and runtime are separate from the WSL operational mirror and
  public/teacher VPS runtimes

### Deployment

1. Copy only the attested feature branches/artifacts to an isolated WSL staging
   tree; verify checksums.
2. Clean-build the game module and client.
3. Install into isolated runtime paths and unused ports.
4. Analyze stock maps and a generated test batch on WSL.
5. Load bundle-v3 Atlas only in staging.
6. Start four network clients with the new protocol against the staging server.
7. Run deterministic transport, map-epoch, death/respawn, crouch, water, hazard,
   hook, and human-slot smoke tests.
8. Record latency/RSS/payload/echo metrics and stop staging cleanly.

### Gate B6

- G0 identity/fresh-lineage retirement and G1 transport pass
- Atlas hashes and oracles match local results
- query p99/RSS/payload budgets pass on WSL
- no public/teacher service, queue, port, map, or credential changed
- staging teardown leaves no orphan client/server process

## 12. B7 — Fresh model reset and new training path

This batch creates a new lineage; the previous lineage is retired and cannot be
mutated or resumed.

### Initialization

- run name: `public_network_multires_atlas_fresh_v1` unless already occupied
- new checkpoint, TensorBoard, resume, rollout, and season-report roots
- new random model graph, recurrent state, critic, all heads, optimizer, and
  normalization statistics
- fresh `Q2LAT002+` Dyn bound to the current Atlas hashes
- no thermal carryover
- no `Q2_RESUME_DIR` pointing at a legacy directory
- optional BC/distillation starts from random new-schema weights and new-schema
  demonstrations only; it never partially loads old modules

### Curriculum start

1. transport/posture/water/death-screen echo
2. standing/crouched traversal
3. hazard avoidance and walking recovery
4. hook-assisted recovery and controlled drops
5. pickups and guide dropout
6. generated-map combat

Each stage writes a fixed gate report before the next begins. Reward weights do
not change inside a stage and are fully recorded at launch.

### Gate B7

- startup manifest proves fresh lineage and rejects all legacy resume paths
- G1 and G2 pass over at least 16,384 accepted transitions
- G3 induced hazard/recovery tests improve over matched baseline
- TensorBoard contains the complete required telemetry and only this new run in
  its new current-run view
- no promotion claim is made from loss convergence alone

## 13. B8 — Quality seasons and public-topology shadow

1. Complete generated-map combat season and G5-generated.
2. Complete stock-map combat season and G5-stock.
3. Complete matched-seed guide-on/off season and G4.
4. Archive immutable season reports, policy, runtime manifest, Atlas hashes,
   Dyn schema, and reward configuration.
5. Only after all above pass, run the policy on public topology in shadow while
   preserving `maxclients=6`, four ML clients, and two human slots.
6. Keep map compilation/analysis on WSL and preserve interlaced stock/generated
   rotation and queue prefix isolation.

Gate B8 is the full specification predicate through G6 except the final
new-runtime cold-restart drill.

## 14. B9 — Cold-restart drill and one-way promotion decision

Owner: root only.

1. Stop the isolated shadow cleanly.
2. Reconstruct the new runtime from its attested source and artifacts.
3. Prove the new clients/server/policy reconnect and map rotation works.
4. Verify that no operational selector can load a legacy runtime or model.
5. Compare every G0–G6 report.
6. Promote publicly only when the complete predicate is true.
7. If any gate fails, hold and revise the new implementation; do not restore a
   retired model lineage.

After a successful promotion, commit/push all three repositories, refresh only
the documented operational mirrors with checksum verification, update AGENTS
and the handoff document, and remove legacy model/runtime artifacts from
operational roots according to the retirement manifest.

## 15. B10 — Community-map corpus expansion

This post-prototype batch implements the requested large authored-map surface
without weakening licensing or runtime safety.

### Parallel work

- acquisition agent: source/license ledger and intact upstream downloads
- quarantine agent: archive safety, asset inventory, hashes, deduplication
- analysis agent: Atlas/design signatures and aggregate prior updates

### Rollout

1. Admit a manually reviewed pilot set of 10 maps.
2. Validate analysis-only storage and coordinate-free prior output.
3. Expand to 50 diverse deathmatch maps.
4. Run baseline/treatment generator batches and anti-copy/diversity gates.
5. Expand further only when generator quality improves without static,
   lighting, traversal, or layout-diversity regression.

No third-party BSP or custom asset is rehosted or installed publicly without a
written redistribution grant and the existing frozen-policy canary process.

## 16. Required batch handoff format

Every agent returns:

```text
Batch / agent:
Repositories and commits:
Files owned:
Artifacts and SHA-256:
Tests run and results:
Performance measurements:
Known limitations:
Spec deviations: none | exact section and justification
Deployment performed: no
Recommended integration order:
```

Root does not integrate a handoff that omits tests, hashes, limitations, or
declares an undocumented spec deviation.

## 17. Stop conditions

Stop the active batch and do not advance if:

- a required oracle cannot reproduce engine behavior
- Atlas or Dyn exceeds its hard memory/payload budget
- deterministic hashes diverge across cold runs/hosts
- public and teacher feature paths cannot be proven separate
- action echo cannot causally verify vertical intent
- guide-off evaluation collapses behavior
- posture/down-look/backward behavior breaches G2
- combat ladder does not reach hits, repeated hits, and kills on both seasons
- the new runtime cannot complete a clean attested cold restart
- an external map lacks acceptable provenance/license status

These conditions require revision of the batch or specification; they are not
permission to weaken gates or deploy partially.
