# B2 gate assembly

`tools/assemble_b2_gate.py` is the only writer for a green
`q2-multires-b2-gate-v1` document. It has no discovery, count-only,
replacement, salvage, partial, overwrite, or red-report mode. A failed input
prints one refusal to stderr and writes no gate.

Current status: cohort 71438 passed its sole 28-member source freeze, then
failed closed on the first WSL compile command during RAD with
`unable to load pics/colormap.pcx`. Compilation stopped with no complete
compiled member and no compiled report, later-stage report, Atlas run, Dyn run,
or gate. Cohort 71438 is permanently retired. Its source populations, partial
WSL residuals, source report, and parallel release builds cannot be retried,
reused, copied, run, salvaged, substituted, or admitted. Exact evidence is in
`B2-GENERATED-COHORT-71438-FAILURE.json`. No replacement cohort is authorized
or declared; the alias and assembler remain pinned to 71438 solely as a
fail-closed historical identity.

The assembler requires the repository to be clean before it reads the current
commit, tree, generator, route generator, and complete Atlas analyzer closure.
The assembler would require a source freeze, generated Atlas build report,
test report, every analysis manifest, and real Dyn proof that all carry the same
implementation authority. The exact 71437 partial prefix remains archived and
retired. The successful 71438 source freeze and failed WSL compile root are now
also historical evidence only. They cannot satisfy a future population, and
no 71438 producer may be invoked again under the no-retry declaration.

## Required producer reports

- `tools/run_generator_cohort.py generate` historically wrote the exact 71438
  source freeze from two distinct fresh directories. That successful source
  evidence is retired with the failed cohort and cannot be retried, copied,
  reused, or promoted.
- `tools/run_generator_cohort.py verify-stage` writes canonical compiled and
  materialized membership reports. `tools/run_compiled_static_campaign.py`
  writes canonical `q2-generator-v6-compiled-static-campaign-v1` evidence with
  all 28 independently recomputed `static_validate` rows.
- `tools/run_generator_claim_campaign.py prepare` writes the exact claims
  stage and prepare report.
- `tools/run_generated_atlas_campaign.py` atomically writes the exact 224-file
  analysis root and `q2-generated-atlas-build-campaign-v1` report.
- `tools/run_generator_claim_campaign.py validate` writes the 28-map compiled
  validation report.
- Each stock map has one canonical
  `<map>.stock-validation.json` from
  `tools/generator_claim_validator.py --mode stock`. The stock BSP, analysis,
  and validation roots contain exactly the eight pinned maps and no other
  file or symlink. Independent cold evidence is embedded in and revalidated
  from each analysis manifest.
- `tools/q2-dyn-evidence` runs on `DESKTOP-RTX2080` WSL and atomically writes
  its report plus four real `Q2LAT002` snapshots. Its selected map must be a
  member of the admitted 71438 population. The assembler binds it to the exact
  local Atlas manifest, raw Atlas, BSP, analyzer authority, crate commit, WSL
  identity, snapshot bytes, negative fences, and p99 measurement. Build the
  producer once from the clean committed root, retain that exact executable,
  and copy those same executable bytes to WSL. The gate independently decodes
  every header and zstd payload, validates the payload digest/cells/derived L3,
  and requires a byte-identical semantic and canonical-zstd re-encode. A magic
  prefix alone is never evidence.
- `tools/run_b2_test_suite.py --output NEW_DIRECTORY` executes the fixed Python,
  Rust, and standalone Dyn-helper suites and atomically writes
  `b2-test-report.json` plus seven hashed raw logs. The directory must be
  outside the repository so its creation does not invalidate the clean Git
  binding.

## Assembly template

All values are exact paths. `OUT` must not exist and must be outside the
implementation repository so publishing the gate cannot invalidate its own
clean-tree authority.

```sh
python tools/assemble_b2_gate.py \
  --design docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md \
  --plan docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md \
  --repo-root "$PWD" \
  --b1-gate docs/multires/B1-GATE.json \
  --cm-oracle "$CM_ORACLE" \
  --pmove-oracle "$PMOVE_ORACLE" \
  --hook-oracle "$HOOK_ORACLE" \
  --fall-oracle "$FALL_ORACLE" \
  --hook-attestation "$HOOK_ATTESTATION" \
  --atlas-verifier "$ATLAS_VERIFIER" \
  --declaration docs/multires/B2-GENERATED-COHORT-DECLARATION.json \
  --source-dir "$SOURCE" \
  --source-cold-dir "$SOURCE_COLD" \
  --source-freeze-report "$SOURCE_REPORT" \
  --compiled-dir "$COMPILED" \
  --compiled-membership-report "$COMPILED_MEMBERSHIP" \
  --compiled-static-report "$COMPILED_STATIC" \
  --materialized-dir "$MATERIALIZED" \
  --materialized-membership-report "$MATERIALIZED_MEMBERSHIP" \
  --claims-dir "$CLAIMS" \
  --claims-prepare-report "$CLAIMS_PREPARE" \
  --analysis-dir "$ANALYSIS" \
  --generated-build-report "$GENERATED_BUILD" \
  --generated-validation-report "$GENERATED_VALIDATION" \
  --stock-provenance docs/multires/stock-q2dm1-q2dm8.provenance.json \
  --stock-inventory tests/fixtures/corpus/stock-q2dm1-q2dm8.json \
  --stock-bsp-dir "$STOCK_BSPS" \
  --stock-analysis-dir "$STOCK_ANALYSIS" \
  --stock-validation-dir "$STOCK_VALIDATION" \
  --dyn-evidence-executable "$DYN_EVIDENCE_EXECUTABLE" \
  --dyn-evidence-report "$DYN_EVIDENCE/b2-dyn-evidence.json" \
  --test-report "$TEST_EVIDENCE/b2-test-report.json" \
  --output "$OUT"
```

The assembler independently reruns exact membership, all 28 generated claim
validations, and all eight stock validations. It derives representative Atlas
limits from the admitted analysis manifest selected by the Dyn report, rather
than accepting a second budget assertion. The output is canonical compact,
sorted JSON with one trailing newline and is created with exclusive-create
semantics only after every B2 predicate is green.
