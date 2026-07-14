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
