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
