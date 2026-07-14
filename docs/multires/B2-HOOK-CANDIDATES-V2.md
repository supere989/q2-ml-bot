# B2 generated hook candidates v2

Status: offline, unproven generator output

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
authored floor-clear samples. `landing_milliunits` is a desired,
support-aligned landing proposal and is therefore 24 units above its authored
support, not 56 units above it. It is not final trajectory evidence.
`anchor_milliunits` lies on the first real ceiling or ceiling-light-panel face
above the landing. A platform or roof below that face suppresses the proposal;
it is never relabeled as a ceiling.

`distance_milliunits` is the rounded Euclidean distance from
`source + (0,0,22)` to the anchor, in milliunits. `release_after_ticks` selects
one exact 100 ms hook-pull schedule. Candidate IDs, geometries, sources, and
release schedules are deterministically ordered. Source and landing may not
occupy the same L1 cell.

The generated `<map>.json` remains an eight-field C-loader-compatible
projection:

```text
anchor.xyz landing.xyz distance flags
```

It contains unique proposed geometries rather than repeated source/release
schedules. This raw projection is also unproven and must never enter a bundle
or install path. Keeping the prefix only preserves the existing loader and
policy observation shape; it does not authorize a hook edge.

After BSP compilation, exact preflight must replay candidates through the
pinned CM, hook-law, and Pmove authorities, take the first post-release
grounded Pmove frame, and replace the proposed landing with that frame's exact
fixed-point origin. It then groups results by runtime geometry and materializes
exactly the first six proven geometries in stable order. If six cannot be
proved, the map is rejected. Only those measured final records and that
materialized sidecar are eligible for the final source/static pass, canonical
claim hash binding, cold independent analyzer replay, and eventual bundle
admission. There is no generator fallback, fabricated claim, source-metadata
acceptance, or relaxed verification path.
