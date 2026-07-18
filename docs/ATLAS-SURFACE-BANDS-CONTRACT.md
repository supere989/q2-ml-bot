# Atlas L0 Surface-Band Contract

`harness/atlas_surface_bands.py` is the exact CM boundary for sparse L0
surface-band discovery. It is deliberately independent of BSP parsing,
reachability generation, and Atlas serialization.

## Admission

The caller must provide explicit reachable chunk keys and may provide one
additional boundary chunk. The module sorts those keys by `(z,y,x)`, removes
duplicates, never accepts a map AABB, and never infers a dense scan range.

Before creating any collision request, admission prospectively reserves the
occupancy bitplane and all four possible surface-material bitplanes for every
authorized chunk through `L0BudgetState`. A cumulative chunk/byte rejection is
an Exact rejected result with zero oracle calls and zero materialization.
Successful execution accounts only chunks and planes actually retained.

### Candidate-scoped path

`discover_scoped_surface_bands` is the performance path. It accepts global L0
cell coordinates in `SurfaceCandidateGroup` values. Every cell must belong to
its group chunk (`cell_axis // 16 == chunk_axis`), and every group chunk must be
in the caller-authorized reachable/boundary set. All groups and cells are
validated and canonicalized before the first oracle call. Duplicate group
records and overlapping cells are deduplicated.

For each group, discovery first probes only its candidate cells and the minimal
six-face, one-cell exposure halo. Verified surface seeds then drive targeted
occupancy expansion through occupied cells in authorized chunks, one layer at
a time, through depth five. It never issues an exhaustive 16-cubed chunk or
28-cubed chunk-plus-halo scan. The occupancy cache is discarded after each
candidate group; the only cross-group structure contains cells that will be
materialized. This prevents a map-wide multi-million-cell occupancy set.

The B2 analyzer supplies one exact hull-bottom floor candidate per admitted L1
node. Adjacent L1 samples are four L0 cells apart, so their depth-five occupied
bands overlap. At a stance-corridor boundary it supplies wall and ceiling
candidates from the exact admitted `NavNode.position` plus standing/crouched
hull extents, spaced no farther than 16 world units over each face. Only chunks
that own an explicit corridor candidate are authorized; an inward band crosses
a chunk boundary only when the adjacent chunk independently owns a candidate.
This is the complete surface scope for the admitted sampled corridor; it is not
a claim that an entire coarse L1 cube is free space.

The analyzer executes one canonical owner-chunk group at a time without a
map-wide response cache. Before the first request it conservatively sums each
group's Manhattan-depth-five occupancy union plus six possible surface traces
per candidate. A separate 16,000,000-request planning cap rejects pathological
scope while the unchanged 2,000,000-request collision-oracle cap fails closed
on actual requests. Logical and physical counts are both recorded and are equal
in the no-cross-group-cache implementation.

`SurfaceBandResult.request_counts` records the exact occupancy and surface
trace request counts. These counts include exact surface traces that return no
hit but never include requests suppressed by candidate or expansion overlap.

`discover_surface_bands` remains the exhaustive compatibility path. It probes
an authorized chunk plus its bounded 24-unit witness halo and should not be
selected for large cold-map builds when deterministic candidates are already
available.

## Collision evidence

- Model 0 uses q2-cm-oracle-v1 `box_trace`.
- A real inline model at one fixed pose uses `transformed_box_trace` with its
  oracle-owned `headnode`, `origin`, and normalized `angles`. `model_index` is
  provenance and is not an oracle request field.
- A mover sweep is a non-surface envelope and returns Unknown. Callers must
  query explicit fixed poses instead.

Occupancy is `startsolid || allsolid` from a stationary 4-unit cube trace
(`mins=-2`, `maxs=2`) centered on each L0 cell. Requests are emitted in bounded
batches and deterministic `(z,y,x)` order.

An occupied cell is a surface seed only when a face neighbor is CM-clear and a
sweep of the same 4-unit cube from that clear canonical neighbor to the occupied
canonical center returns a hit. This preserves the occupancy volume at
boundary-aligned, sloped, and corner contacts where a center-point trace may
remain clear. Plane normal and surface metadata come only from that hit trace;
occupancy responses never confer `sky`, `slick`, `warp`, or `nodraw`.

Classification uses the exact hit-plane normal:

- floor: `normal.z >= 0.7`
- ceiling: `normal.z <= -0.7`
- wall: otherwise

Every retained witness preserves the exact CM surface name, flags, value, and
normal. Band cells combine nearest-witness flags with bitwise OR without
discarding the individual evidence.

## Retention and authority

Discovery performs a bounded inward flood from each exposed occupied seed.
Depth zero is the seed and depth five is the final retained cell: exactly six
4-unit cells (24 units) at most. Depth six, the seventh cell, is never retained.
Only cells in caller-authorized chunks are materialized; halo cells are
collision witnesses only.

Malformed, missing, rejected, or mismatched oracle evidence makes the entire
operation Unknown. A successfully evidenced result, including an empty result,
is Exact. Candidate-scoped Exact means exact collision evidence within the
provided scope; it is not a claim that the caller supplied every map surface.
Output chunks and cells are canonical `(z,y,x)` order.
