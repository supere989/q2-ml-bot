# B2 Atlas surface and entity integration

The B2 analyzer now treats `harness/atlas_surface_bands.py` and
`harness/atlas_entity_semantics.py` as its only authorities for the facts they
own.

## Reachable surfaces

- Candidate cells come from exact reachable L1 node positions and Quake player
  hull extents. There is no model-bounds or whole-chunk surface scan.
- Every reachable node contributes its hull-bottom floor point. Missing or
  stance-incompatible L1 neighbors contribute bounded wall/ceiling face points.
- q2-cm-oracle stationary 4-unit cube results establish occupancy. A separate
  sweep of that same cube from a clear canonical neighbor to the occupied
  canonical center establishes the plane and surface flags, including boundary,
  slope, and corner contacts that a center-point trace can miss.
- Model 0 uses `box_trace`. An inline model uses
  `transformed_box_trace(headnode, origin, angles)` only at an exactly declared
  fixed pose.
- Candidate count, request upper bound, logical occupancy/hit requests, physical
  CM requests, materialized cells, and authorized chunks are emitted in the
  analysis manifest. Groups use bounded local caches only; there is no map-wide
  response cache.

## Entity semantics

- Ordered `angle`/`angles`, `G_SetMovedir`, sliding endpoints, rotating axes,
  train target topology, and `trigger_hurt` linked AABBs come from the pure
  entity-semantics module.
- Non-toggle `trigger_hurt` state is exact: initially active volumes become
  hurt/forbidden planes and permanently START_OFF volumes emit none. Toggleable
  volumes retain only potential `unknown` coverage.
- Sliding and rotating movers may emit a collision-backed reference surface at
  their declared current pose. A train does so only when its initial target is
  uniquely determined by the first-eight-edict `G_PickTarget` law.
- Dynamic mover envelopes are conservative `mover_swept_envelope + unknown`
  cells. They do not receive collision or traversal authority and do not create
  mover edges.
- `func_train` marks every eligible target pose independently. It does not fill
  a union AABB between target positions, because topology does not prove
  continuous occupancy there and a path corner may teleport.

## Budgets and separation

Every first `(chunk, plane)` allocation passes cumulative prospective
`L0BudgetState` admission before the chunk or plane is created. Repeated cell
writes use an admitted-plane cache, so accounting remains constant-time per
cell. Dynamic envelopes pre-admit the two affected planes for every clipped
retained chunk, then stream local indices without a global cell-tuple list. The
final chunk/plane set must exactly equal the budget ledger.

The pre-existing drop/void classification block is intentionally unchanged in
this integration. Its independent Pmove/fall-oracle replacement owns that
authority and must be integrated separately.
