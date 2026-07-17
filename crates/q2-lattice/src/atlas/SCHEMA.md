# Static Atlas binary schema v1

The authoritative identity is SHA-256 over the canonical uncompressed
little-endian payload. The zstd wrapper is transport only; its bytes are not a
cross-version identity.

## Canonical payload

Header (`136` bytes):

| Field | Encoding |
|---|---|
| magic | 8 bytes, `Q2ATL001` |
| schema, byte order | `u16`, `u16` (`1`, `0x454c`) |
| header length | `u32` (`136`) |
| snapped origin xyz | 3 × `i64` |
| cell sizes | 4 × `u32` (`4,16,64,256`) |
| L0 chunk, L1 node, L1 edge, L2 cell, L3 cell counts | 5 × `u64` |
| corresponding section byte lengths | 5 × `u64` |

All keys are strictly increasing `(iz,iy,ix)`. Counts are `u64` on disk and
must pass configured `usize`, cardinality, payload, and resident-size guards
before allocation.

Sections are:

1. Sparse L0 chunks. Each record is xyz `i32`, a `u64` bitplane mask, a `u8`
   scalar-plane mask, then selected 4096-bit planes and selected 4096-byte
   compact planes in enum order. Empty planes and chunks are noncanonical.
2. Fixed 40-byte L1 navigation nodes.
3. CSR: `u64` offset count, `u32` offsets, then fixed 28-byte edges. Nodes are
   sorted `(iz,iy,ix)`; each adjacency sorts by edge enum then target
   `(iz,iy,ix)` and remaining fixed fields.
4. Fixed 28-byte L2 aggregate cells.
5. Fixed 28-byte L3 aggregate cells.

Static costs and clearances use integers/fixed point. `0xffffffff` is the only
blocked/infinite cost sentinel. The schema contains no per-cell float, so NaN
or infinity has no representation.

## Zstd envelope

The 64-byte `Q2AZS001` header carries schema, compression ID, fixed level 3,
reserved zero, uncompressed/compressed `u64` lengths, and the 32-byte SHA-256
of the uncompressed payload. It is followed by one dictionary-free zstd frame.
Decoding caps the zstd window and output before allocation, then verifies exact
length and digest.

## Sparse-only invariant

L0 exposes only `SparseL0::new` plus insertion of a nonempty `L0Chunk`. There
is no map-AABB or dense-cell constructor. One 16-cubed L0 chunk covers the same
64-unit cube as one L2 cell.

## Manifest-coupled oracle admission

The binary graph is structurally decoded before its separate canonical
analysis manifest is available. Admission is complete only after pairing the
payload with that manifest and calling `verify_atlas_artifact`.

`q2-atlas-verify CANONICAL_ATLAS_MANIFEST.json RAW_ATLAS.bin` is the narrow
process boundary for consumers that cannot link the Rust crate. It requires one
and only one `application/vnd.q2.atlas-v1` artifact identity, applies default
budgets and full oracle admission, and rechecks the raw digest, counts, graph,
and origin. Success is a single canonical `q2-atlas-verification-v1` JSON line;
failure writes no summary and exits with status 65.

The manifest requires a semantic `q2-cm-oracle` contract bound to the compiled
BSP SHA-256 and its provenance digest. `q2-pmove-oracle` and `q2-hook-oracle`
contracts are optional. Jump and controlled-drop edges require an admitted
Pmove contract. Hook edges require admitted Pmove and hook contracts plus the
canonical eight-case isolated-q2ded parity attestation. Every materialized edge
has nonzero evidence and validation version. Missing or mismatched trajectory
authority omits its edge classes; collision remains mandatory and is never
approximated from BSP metadata or generator claims.

Parity collision/Pmove physics identities are explicitly fixture-scoped because
those identities include the synthetic parity BSP digest. The attestation binds
their exact executable digests to the map-specific admitted tools and binds the
map-independent hook physics identity directly.

## Objective authority and runtime guide derivation

The sole objective artifact is canonical `<map>.objectives.json`, schema
`q2-atlas-objectives-v1`, with media type
`application/vnd.q2.atlas-objectives-v1`. It binds the canonical map ID, BSP
SHA-256, raw Atlas SHA-256, shared origin, stable entity IDs, objective class,
world milliunits, and admitted L1 target. A runtime manifest must attest exactly
one such artifact. Missing, duplicate, stale, or mixed objective identity fails
closed, including on maps with zero objective records.

Generator `<map>.routes.json` remains a source/item-timing claim and is never an
Atlas analysis objective artifact. Production guide candidates are derived
inside `AtlasRuntime` from the installed Atlas, objective artifact, and
per-objective availability beliefs. External geometry, path cost, risk, class,
or confidence candidates have no production API.

## Live Q2LAT002 Dyn runtime

`Q2LAT002` is the canonical per-client persistent L2 state plus its derived L3
mip. Loading a snapshot initializes storage only; a query is rejected until a
transactional live batch establishes a nonzero client-life epoch and strictly
increasing server frame. Every batch, including an empty batch, compares the
previous environment-step value, advances to a monotonic new value, and is
fenced by Atlas digest, map digest, snapped origin, map epoch, numeric client
identity, life epoch, and server frame.

The only deposit events are `engagement`, `threat`, `opportunity`,
`self_fire`, and `death` under schema `q2-dyn-named-event-v1`. Kind codes are
one through five in that order and each nonzero ID must equal
`(server_frame << 3) | kind_code`. Every event has a finite factual world
point. It deposits one unit into exactly one named channel and one sample; callers cannot provide
weights, confidence, cells, feature values, or Dyn24 blocks. Events are sorted
by ID, duplicates and stale IDs are rejected, and deposits are count-merged by
canonical L2 index before one derived-L3 rebuild. Accepted factual deposits set
the touched cell confidence to one. Opportunity deposits are discrete
actionability/L2-transition facts, threat requires positive observed damage,
and self-fire is an echoed shot edge; unchanged presence or held input never
repeats a persistent deposit.
All five events come only from the client's public factual observation and
accepted public action echo: positive damage dealt plus a current or
maximum-five-tick visible target point, positive damage taken at the own point,
a newly actionable visible target, an accepted echoed shot edge, and public
death reward/health terminal at the own point. Physically separate private
causal and reward-admission evidence never seeds Dyn24. An empty transaction
still advances the live frame/decay cursor when private causal evidence is
absent or incomplete.

Persistent channels, sample mass, and confidence decay by an exact factor of
one half whenever the absolute environment-step counter crosses a 1024-step
interval. Power-of-two decay makes a direct N-interval advance bit-identical
to any partition into smaller advances. Cells whose persistent mass reaches
zero are removed. Thermal evidence remains a maximum-five-tick query input and
never enters either snapshot or checkpoint bytes.

The default admission budgets are five events per transaction (one per named
kind) and 10,000,000 events per client life, in addition to the existing cell, resident-memory, and
snapshot-byte ceilings. `Q2DRT001` is the canonical live checkpoint wrapper: it
contains the current Q2LAT002 snapshot plus the life/frame/event cursor and is
restored only with its mandatory wrapper SHA-256 and all expected identity
fences. A bare Q2LAT002 export is suitable for Dyn transport and inspection;
only Q2DRT001 preserves duplicate/stale rejection across trainer restart.

## Private recovery evidence

The same admitted L1 query that emits normalized public Recovery16 also emits
private raw `RecoveryEvidence`: L1 index, Q8 cost-to-safety (where `u32::MAX` is
the valid unreachable sentinel), signed Q8 safety-boundary clearance, raw
hazard type/severity, confidence, and `atlas_region_id`.

`hazard_component_id` is an additional static reward-space identity. Hazardous
L1 nodes connected by admitted static adjacency form components. Components
receive 1-based IDs in minimum `(iz,iy,ix)` seed order. A deterministic
multi-source graph distance then assigns every reachable safe recovery node to
the nearest component; equal-distance ties choose the lower component ID and
zero means no reachable static hazard. This association deliberately persists
past the first safe node into positive-clearance/rearm space. It is immutable
for one Atlas digest/map epoch, never generated from movement history, and
cannot manufacture a fresh ID during boundary oscillation. The open reward
episode retains its admitted entry component until close/rearm even if the
client later crosses a different fixed basin.

Signed clearance is packer-derived, never analyzer-supplied. The deterministic
multi-source solve uses the admitted static L1 graph as an undirected geometric
adjacency, excludes hook/mover/blocker edges, seeds both sides of every
hazard/safe boundary at half an L1 cell (`8 * 256` Q8), and propagates only
within the same hazard class using edge Q8 distance. Safe values are positive;
hazard values are negative. `i32::MAX` means a safe component has no reachable
hazard boundary and `i32::MIN` means a hazardous component has no reachable
safe boundary. Runtime admission recomputes and verifies the complete field.

`atlas_region_id` is only the Atlas L1 SCC/traversability identity. It is not
the QM3C causal `hazard_region_id`, must never be substituted for it, and does
not key reward episodes. Likewise `hazard_component_id` is neither a traversal
SCC nor a server damage-source ID. The provider envelope additionally fences Atlas
SHA-256, map epoch, client ID, nonzero client life epoch, and server frame;
QM3C owns the independent causal hazard ID and verifies that its tick equals
the enclosing server frame.
