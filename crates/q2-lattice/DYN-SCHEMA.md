# Per-client Dyn binary schema v2

`DynState` is per-client persistent tactical evidence at L2 (64 units) plus a
deterministic L3 (256 units) mip. It never owns or encodes Atlas cells, thermal
tracks, readiness plumes, L0 occupancy, recovery vectors, or guideposts.

The authoritative snapshot begins with `Q2LAT002`. `Q2LAT001` is retired and
is rejected without conversion. The fixed 208-byte little-endian header binds
schema 2, dictionary-free zstd level 3, compressed and uncompressed lengths,
the uncompressed payload SHA-256, Atlas SHA-256, map SHA-256, snapped origin,
the frozen L2/L3 cell sizes (`64`, `256`), map epoch, environment steps, client
ID/count, and L2/L3 counts. Reserved bits must be zero. Earlier 200-byte
`Q2LAT002` prototypes did not carry the cell-size fence and are rejected rather
than inferred or upgraded in place.

The compressed body contains ordered L2 records followed by ordered L3
records. A record is 40 bytes: signed xyz `i32`, engagement/threat/opportunity/
self-fire/death `f32`, sample-mass `f32`, and confidence `f32`. Records are
strictly ordered `(iz,iy,ix)`. All floats are finite, scores and mass are
nonnegative, and confidence is in `[0,1]`. L3 is recomputed in fixed child
`z,y,x` order by summing scores/mass and max-pooling confidence, and it must
bit-match the payload; no independent L3 deposit exists.

Default admission limits are 20,000 combined L2+L3 materialized cells and just
under 2 MiB resident plus 2 MiB uncompressed/compressed bytes per client. A
synchronous batch is reported above 2 MiB compressed and rejected at more
than 8 MiB compressed or at 8 MiB resident. Batch client IDs must be exactly
`0..client_count`, with one Atlas/map/origin/epoch fence and one
environment-step value.

The fixed policy block is the named 24-float layout in the multires design.
Direction triples are yaw-local forward, Quake-right, and world-up. Thermal is
passed ephemerally to feature assembly; persistent engagement is the fallback
when no live thermal signal exists.
