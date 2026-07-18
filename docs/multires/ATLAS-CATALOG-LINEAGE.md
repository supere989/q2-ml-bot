# Multi-map Atlas catalog lineage

The primary trainer admits one immutable `q2-multires-atlas-catalog-v1` before
launching `q2ded`. The catalog is the stable spatial-data identity for the
entire policy lineage. A run may move between admitted maps without changing
its policy lineage.

## Two identities with different lifetimes

- `atlas_catalog_sha256` is stable across checkpoints, curriculum stages,
  lifecycle evidence, and promotion evidence. It seals the complete admitted
  map set and the exact Rust extension used to create Dyn state.
- `atlas_sha256` is the active map identity. It appears in frames, rollouts,
  update evidence, season reports, and metrics, and changes when the server
  changes map.
- `qualification_atlas_sha256` remains the Atlas used by the B4-B6 runtime
  qualification. It is never rewritten to impersonate the active map.

A join that compares an active Atlas digest to the stable catalog digest is a
contract error. A checkpoint carrying only a per-map Atlas digest is legacy and
is rejected; checkpoint format v1 has no fallback path.

## Portable deployment tree

Every catalog artifact path is relative to the catalog directory, is confined
to that directory tree, and is sealed by SHA-256. The catalog contains, for
each map, the BSP, uncompressed Atlas, Atlas manifest, bundle manifest, and
objective artifact. It also seals the exact `q2_lattice_rs` extension artifact.
Copying the complete tree to another host preserves the semantic catalog
identity; rebinding any referenced bytes does not.

Catalog admission reconstructs each map tuple from the referenced documents.
Unknown maps, missing files, duplicate map names, path escapes, noncanonical
catalog bytes, semantic mismatches, or extension substitution fail before the
game server is launched.

## Dyn lifecycle

The catalog authorizes `rust-empty-per-map-epoch-v1`. For every client and each
new observed map epoch, `DynRuntime.empty` creates a new Q2LAT002 state with
zero thermal cells, events, cursors, and environment steps. No Dyn snapshot or
thermal checkpoint is carried through the catalog.

Therefore a generated-map to stock-map to generated-map sequence creates three
independent Dyn instances. Returning to a map does not restore its previous
thermal state; only the stable Atlas artifact is reused. Map epochs must remain
strictly increasing.

## Authoring and validation

Use `tools/manage_atlas_catalog.py author` with absolute input paths rooted
under the intended portable catalog directory and the exact built Rust
extension. Distribute the whole directory, not the catalog JSON alone. Use the
tool's `validate` command on each destination host before selecting it for a
primary run.

The trainer, service proof, lifecycle selector, and retirement gate must all
bind both the catalog file artifact record and its semantic
`atlas_catalog_sha256`. A catalog change intentionally starts a new policy
lineage; it is not an in-place map update.
