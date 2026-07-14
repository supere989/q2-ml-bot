# Atlas entity-semantics contract

`harness/atlas_entity_semantics.py` is the pure ordered-entity layer between
IBSP metadata and the collision-backed Atlas analyzer. It reproduces the pinned
Lithium/Yamagi entity laws without importing the analyzer, invoking an oracle,
or promoting metadata into collision authority.

## Authority rule

Every public operation returns `AuthorityResult[T]`:

- `Exact` means the returned entity-string law or budget result is fully
  determined.
- `Unknown` may include a conservative value, but that value is not eligible
  for deterministic traversal or collision admission.

Mover geometry claims separately mark their declared transform as Exact and
their collision authority as Unknown. The analyzer must attach transformed-CM
evidence before treating a pose as solid or passable.

## Inputs

Entity properties remain ordered `(key, value)` pairs. Do not convert them to a
mapping. Quake parses keys in source order, case-insensitively; `angle` and
`angles` share `s.angles`, so the last occurrence of either wins.

- Sliding mover bounds are the inline **cmodel** bounds used by `gi.setmodel`;
  their collision-loader one-unit expansion is already present.
- `trigger_hurt` bounds are raw IBSP dmodel bounds. The helper applies the
  cmodel expansion, trigger-link expansion, and player-link expansion.
- `train_topology` receives all spawned candidates in strict edict order and
  the train's cmodel mins.

## Frozen behaviors

- `set_movedir`: full `angles`, `angle=-1` up, `angle=-2` down, then cleared
  runtime angles.
- Sliding movers: `sum(abs(movedir[i]) * size[i]) - lip`; negative distance is
  preserved. START_OPEN swaps declared endpoints.
- Trains: `G_PickTarget` candidates are the first eight matching edicts.
  Duplicate groups return Unknown selected-route authority while retaining the
  exact candidate set. TELEPORT and connected-TELEPORT stops, unresolved
  targets, and open terminals remain explicit.
- Rotators: class-specific X/Y flag precedence, REVERSE, integer/default
  rotating-door distance, START_OPEN endpoint swap, and entity-origin pivot.
  Swept bounds are potential envelopes only.
- `trigger_hurt`: runtime touch is inclusive linked-AABB overlap, not convex
  brush contact. Toggleable state has zero static runtime confidence; a
  non-toggle state is exact.
- L0 prospective accounting charges 21 bytes per nonempty chunk, 512 bytes per
  first bitplane, and 4096 bytes per first scalar plane. It rejects before the
  1201st chunk or the byte overrun and never mutates the prior state on reject.

Drop classification and wall/ceiling discovery are intentionally outside this
module because they require collision/Pmove oracle evidence.
