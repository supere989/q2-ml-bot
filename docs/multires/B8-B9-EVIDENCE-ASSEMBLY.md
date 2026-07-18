# B8/B9 Evidence Assembly

`tools/assemble_b8_b9_evidence.py` is the executable, non-deploying boundary
for plan batches B8 and B9. It does not launch a server, trainer, client,
TensorBoard, shadow, restart, or public mutation.

## Archive first

Create a new archive with the `archive` command. The output directory must not
exist. Supply every role printed by `--help`/a missing-role failure as an
absolute `ROLE=PATH` argument. The integration evidence roles are discovered
from `integration_envelope`; its nine paths must be relative and self-contained.

```sh
python3 tools/assemble_b8_b9_evidence.py archive \
  --kind season --season-kind generated \
  --archive-id generated-quality-001 \
  --artifact current_season=/absolute/current-season.json \
  --artifact evaluation=/absolute/generated-evaluation.json \
  --artifact b7_gate=/absolute/B7-GATE.json \
  --artifact b7_stage_evaluation=/absolute/B7-stage-evaluation.json \
  --artifact integration_envelope=/absolute/integration/envelope.json \
  --artifact integration_report=/absolute/integration/report.json \
  --artifact runtime_manifest=/absolute/runtime-manifest.json \
  --artifact policy=/absolute/policy.pt \
  --artifact atlas=/absolute/generated-atlas-or-inventory.bin \
  --artifact atlas_catalog=/absolute/atlas-catalog.json \
  --artifact dyn_schema=/absolute/dyn-schema.json \
  --artifact reward_configuration=/absolute/reward-config.json \
  --artifact source_identity=/absolute/source-identity.json \
  --artifact retirement_manifest=/absolute/retirement.json \
  --artifact legacy_selector_absence=/absolute/legacy-absence.json \
  --out-dir /absolute/archives/generated-quality-001
```

The author copies exact bytes to an incoming directory, discovers and copies
all nine integration files, reruns the integration verifier, validates B6
source identity and runtime retirement binding, writes the content manifest
last, changes every member and directory to read-only, then atomically renames
the archive into place. Archives reject symlinks, undeclared members, and files
with a hard-link count other than one. Generated and stock archives use the same policy/runtime/
Dyn/reward/source identity, but may carry different Atlas inventories.

## Cold-restart reconstruction authority

A restart archive must include
`reconstruction_manifest=/absolute/reconstruction-manifest.json`. This is an
actual immutable input, not a digest-shaped assertion in the cold-restart
evidence. Its schema is
`q2-multires-b9-reconstruction-manifest-v1`; it binds the selected B8 gate,
runtime manifest, Atlas catalog, lineage root, and the exact byte count and
SHA-256 of every identity artifact reconstructed for the restart. It must set
`complete=true` and carry a valid `manifest_sha256` self-seal.

The cold-restart evidence field `reconstruction_manifest_sha256` is the SHA-256
of that exact manifest file. B9 independently opens the archived manifest,
checks its self-seal and all artifact bindings against the restart archive and
B8 shadow identity, then records both the file digest and evidence seal in the
promotion manifest. A missing, symlinked, hard-linked, modified, rebound, or
wrong-digest reconstruction manifest fails before an eligible B9 decision can
be emitted.

## Gate sequence

```sh
python3 tools/assemble_b8_b9_evidence.py season \
  --archive-manifest /absolute/archives/generated-quality-001/archive-manifest.json \
  --out /absolute/evidence/B8-GENERATED.json
python3 tools/assemble_b8_b9_evidence.py season \
  --archive-manifest /absolute/archives/stock-quality-001/archive-manifest.json \
  --out /absolute/evidence/B8-STOCK.json
python3 tools/assemble_b8_b9_evidence.py b8 \
  --generated-gate /absolute/evidence/B8-GENERATED.json \
  --stock-gate /absolute/evidence/B8-STOCK.json \
  --shadow-archive-manifest /absolute/archives/public-shadow-001/archive-manifest.json \
  --out /absolute/evidence/B8-GATE.json
python3 tools/assemble_b8_b9_evidence.py b9 \
  --b8-gate /absolute/evidence/B8-GATE.json \
  --restart-archive-manifest /absolute/archives/cold-restart-001/archive-manifest.json \
  --out /absolute/evidence/B9-DECISION.json
```

Generated and stock gates independently evaluate G0 through G5. They must be
separately archived seasons, not two labels over one observation window. Each
season gate therefore binds its season ID, archive ID and both manifest seals,
current-season/evaluator file and evidence seals, causal/network window seals,
and active Atlas identity. B8 requires every one of those season-specific
bindings to differ while requiring the policy, runtime, source, Dyn schema,
reward configuration, Atlas catalog, and lineage root to remain shared. A
duplicate archive, copied current/evaluation pair, reused window, or generated/
stock relabel fails before a B8 gate can be emitted.

B8 additionally requires matched guide-on/off evidence and a public-topology shadow with
`maxclients=6`, four ML clients, two human slots, exercised join/leave and map
changes, and zero VPS compile/analyze work. B9 adds the attested cold restart
and compares the complete G0-G6 predicate.

A green B9 result says only `eligible-for-root-manual-promotion`. Every emitted
document fixes `automatic_promotion=false`, `public_mutation_authorized=false`,
and `public_mutation_performed=false`. Root performs any later public promotion
as a separate action.
