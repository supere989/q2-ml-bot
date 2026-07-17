# B2 gate assembly

`tools/assemble_b2_gate.py` is the only writer for a green
`q2-multires-b2-gate-v1` document. It has no discovery, count-only,
replacement, salvage, partial, overwrite, or red-report mode. A failed input
prints one refusal to stderr and writes no gate.

Methodology amendment: 2026-07-16. The amended design and plan have new
normative digests. Therefore every prior B1 seal and B2 gate/admission artifact
is historical for future assembly. Before B2 qualification, create a fresh B1
gate and authority seal that bind the amended documents and exact oracle
binaries. The prior `B1-authorities-909b1e46` seal cannot authorize a new
qualification or final cohort, even if its binaries are later verified as
byte-identical.

B2 assembly distinguishes artifact states. `built` means staging bytes exist;
`published` means exact stage membership was atomically exposed but remains
non-admissible; `validated` means that stage passed its named independent
checks; `admitted` means all 28 final declared maps and every B2 gate passed.
The assembler accepts only validated inputs and emits admission only after the
complete predicate is green.

A green B2 gate proves offline artifact, oracle, determinism, and performance
integrity only. It does not prove policy learning, targeting, locomotion,
reward quality, trainer cutover, or public readiness.

Historical cohort `b2g26_final_71439` passed its sole 28-member source
freeze and sole WSL compile, publishing an exact 168-file compiled stage. Its sole
materialization then failed closed on the first map,
`b2g26_open_71439000`, before any hook materialization. CPython 3.10.12 raised
an unterminated-string `SyntaxError` at `harness/atlas_analyzer.py:5404`.
The wrapper stopped after one attempt, left only a byte-identical 168-file
staging residual, and published no materialized stage. The source, compile,
and materialization report SHA-256 values are respectively
`fbcbca7c134c2d2595ab98cfe939f615b226cab4a5e28e836f824d41e4f76255`,
`fc6435e81ac1d10f8a32602169df68cc34103c4b64a2cdbcf96be55260a3733d`,
and `b171b2ee4ab02f8b960684544e49471dcfc5e11cdef105687a77938e1dcafe69`.
Exact evidence is in `B2-GENERATED-COHORT-71439-FAILURE.json`.

Cohort 71439 is permanently retired. Its source and cold populations, WSL
source and compiled publication, materialization residual, reports, logs, and
producer snapshot cannot be retried, resumed, reused, copied, run for cohort
purposes, salvaged, substituted, or admitted. Its immutable named declaration
remains pinned to its exact 28 map/seed rows, and the retirement registry still
rejects it before any evidence can be assembled.

Replacement cohort `b2g26_final_71440` was explicitly authorized. Its canonical
named declaration and then-current alias were byte-identical with SHA-256
`d71b86a109bb359f927457d3904cef3116d83c59104cc85b3a87dd43ddc791b2`.
They declare exactly 28 new rows in ordinal order: four each for `open`,
`towers`, `canyon`, `pits`, `arena_open`, `arena_vertical`, and `arena_lanes`,
using disjoint seed blocks 71440000..71440003 through 71440600..71440603.

Cohort 71440 passed its sole source freeze, WSL compile, WSL materialization,
compiled membership/static validation, materialized membership, and claims
preparation. Their report SHA-256 values are respectively
`2abbb7c9de511fd4b497111317d61be439f37c96702441d6d7190e9afb5cf19c`,
`94681d77f53b0514a2795865d593b6007d58bef9e9bbf1be0a7ef2f16d7e46b1`,
`11689967027196a77443d02628da1ee72df33bfa71475a1967634e268f47afc4`,
`620ec6d827a42feb99603bc15de3f825d51335144d18b5c5d225af8650648a90`,
`56ce7ddb048a04b21beb230d7382859b040d000355bd3effbeae192da61f448a`,
`a8490259cc955cd02427cf9bf7be95f72fb3e66830300f5c4cc850ee65a52eda`,
and `d30a578fbbf4ff03542809536e3f90d7314fc802fe9f15d9e659de2b330e6546`.
Generated Atlas analysis had not begun.

Fresh stock validation exposed that the validator required every stock spawn
to reach another, stronger than the design's requirement for at least one
mutually reachable clear pair. q2dm6 already has multiple qualifying pairs;
spawn 126 is a clear, supported sink because dynamic rotating-mover traversal
is intentionally Unknown. Commit `8ceb5b7` aligns stock admission with the
design and leaves Atlas authority plus generated all-to-all admission intact.

The declaration binds repository commit and tree. Therefore the correction
invalidates 71440's `9327683` implementation binding even though its artifacts
are sound. `B2-GENERATED-COHORT-71440-FAILURE.json` archives the exact evidence.
Cohort 71440 is permanently retired, and none of its source, compiled,
materialized, claims, stock, report, log, or producer-snapshot bytes may be
retried, resumed, reused, copied, salvaged, substituted, or admitted. Its
immutable named declaration remains rejected by the retirement registry.

Fresh replacement cohort `b2g26_final_71441` was explicitly authorized. Its
canonical named declaration and then-current alias were byte-identical with
SHA-256
`5929532e0edae77b48073abccf4a4f3afdbacfb6905d1eadfb7f18d1dc5ba151`.
They declare 28 fresh rows in ordinal order, four per concrete style, using
seed blocks 71441000..71441003 through 71441600..71441603. The sole source
freeze passed 28/28 from clean commit `89a2726` and tree `82e5581`; its report
SHA-256 is
`c241b81b458eb525334a720e9059902dabef30347195ba1200d63b530133f3e3`.

The sole WSL compiler invocation then failed in preflight before q2tool or map
ordinal 0 because the nested log-root parent was absent. Its canonical failure
report SHA-256 is
`292e0e483c66596bfba58972bdf0e58ed36d938b3412c8868a3b2c10ba510aa3`.
No compiled staging/publication, log leaf, materialization, analysis, Dyn, gate,
deployment, or training action occurred. Exact evidence is archived in
`B2-GENERATED-COHORT-71441-FAILURE.json`.

Cohort 71441 is permanently retired. Its source, WSL, report, and
producer-snapshot bytes cannot be retried, repaired, resumed, reused, copied,
salvaged, substituted, or admitted. Its immutable named declaration remains
rejected by the retirement registry.

Historical declaration statement: fresh replacement cohort
`b2g26_final_71442` was authorized by its then-current alias. Its canonical
named declaration and alias were byte-identical with SHA-256
`fe4b86bbb0ab331dca4f7fd1418106c69ba4d4ea34b36774cb7e9259d27502bc`.
They declare 28 fresh rows in ordinal order, four per concrete style, using
seed blocks 71442000..71442003 through 71442600..71442603. Its sole source
freeze, compilation, source/static campaign, materialization, claims
preparation, and 28-map Atlas construction completed. Compiled promotion then
passed 25/28 and rejected four spawn rows across three maps because compiled
CM evidence measured only 92 units of the required 96-unit spawn column.
`B2-GENERATED-COHORT-71442-FAILURE.json` is the exact terminal authority.

Cohort 71442 is permanently retired. None of its source, compiled,
materialized, claims, analysis, report, test, or WSL bytes may be retried,
repaired, resumed, reused, copied forward, salvaged, substituted, or admitted.
No Dyn, assembled gate, deployment, or training action ran. There is no active
final cohort; a replacement declaration is forbidden until the fresh B1 seal
and disposable qualification lane are green.

The exact clean immediate-predecessor implementation snapshot at commit
`8d89df4a787e261f8a4fb935908191f8df7634b2` and tree
`0a0f48f7686c860cc7c5afc6d3b3252ef0952681` passed the mandatory no-write
pre-declaration language floor under WSL CPython 3.10. Its git-archive SHA-256
was `4ea65f725f7ea9e2b08b8da60a6ace7b785a704ef036495ace3d0ce5c66b7fdb`
and its tracked-content manifest SHA-256 was
`2e050906f6b3710573a6050b96ccdb901f0772cea1ba05960c5212846c10cd18`.
That snapshot did not contain the 71440 declaration and is not a 71440
producer snapshot. Every future declaration-bearing commit must repeat
`python3.10 -B tools/check_python_syntax_floor.py --root SNAPSHOT` and the
materializer import/CLI preflights under the actual pinned runtime before any
source generation or WSL cohort bootstrap. The pinned runtime is
`/home/raymond/miniconda3/bin/python3.11` (CPython 3.11.4, executable
SHA-256 `b25abf001748dc7ebb4b25013b2572d4e6913246b4c3b8e8b726b3da45494ff4`).
That runtime supplies zstandard 0.19.0 through `__init__.py` SHA-256
`8a65cd4ab44112e1433a097daee7ce8600047995f3289f13d758bb001c06a553`
and the active C backend SHA-256
`40ece7fa91097e53ee4785cef01baae3f220f8dc891e20d94d4e07a1d77c9120`.
The neighboring `/home/raymond/miniconda3/bin/python` convenience symlink
resolves to the same executable but is not an admissible input; the gate binds
the regular `python3.11` file directly.
The system Python 3.10 interpreter is syntax authority only because it lacks
zstandard. These repeated checks must finish before source generation or WSL
cohort bootstrap; a local newer-Python parse is not sufficient because Python 3.14
accepted the PEP-701 construct that terminated 71439 on Python 3.10.
Infrastructure evidence records these as two distinct authorities: the
`python310-syntax-floor` check binds `/usr/bin/python3.10` (CPython 3.10.12,
SHA-256 `7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86`),
while the infrastructure producer and materializer remain bound to the
separate CPython 3.11.4 executable and zstandard module digests above. The
syntax report must never be required to identify the execution runtime.

The assembler must reject declarations for retired cohorts 71426 through
71442 before reading campaign evidence. A future final declaration is eligible
only after the fresh B1 seal and non-admissible qualification report pass.
Existing clean-repository, source-freeze, Atlas, test, manifest, and Dyn
requirements remain part of the frozen gate contract; all earlier B1/B2
admission evidence is historical only.

## Frozen producer-report contract

There is no authorized final-cohort producer report. First produce a canonical
disposable qualification report with `non_admissible: true`. It must bind the
fresh amended-document B1 seal, real q2tool/BSP/CM golden fixtures (`+9` spawn
lift; 104 and 105 fail; 106 passes), and one 28-map qualification campaign.
Every golden and infrastructure preflight must pass, and at least 20 maps must
complete the full lifecycle through promotion validation. Qualification bytes
or passing subsets can never satisfy the final gate.

The disposable campaign is orchestrated by
`tools/run_b2_qualification.py`. Driver-plan schema v2 hash-binds every
external file before source generation, including both the explicitly supplied
B1 oracle paths and the canonical Atlas placements beneath
`CLIENT_ROOT/release` and `LITHIUM_ROOT/tools`. The paired CM, Pmove, and hook
files must be byte-identical. A missing canonical build artifact, a symlink, or
input drift therefore fails the dry run instead of producing a late 28-map
Atlas rejection. The base `pak0.pak`, toolchain, syntax report, normative
documents, boundary proof, runtime modules, packer, and verifier are covered by
the same pre-generation binding.
During retained-evidence replay, a hook-materialization attestation's
`source_projection_sha256` is checked against the original compiled `.json`.
The materialized `.json` is the intentionally upgraded runtime projection and
must never be substituted as its own source; BSP and every other immutable
compiled member still require byte identity across the two roots.

After qualification is green, a separately committed fresh declaration may
authorize one immutable/no-retry final producer attempt. Its required order is
source/source-static, real compilation, compiled-CM preflight,
materialization/claims, full Atlas/cold rebuild, compiled promotion, then Dyn,
tests, and assembly. Every item below comes from fresh final-cohort roots.

- `tools/run_generator_cohort.py generate` is the only authorized producer for
  the exact final source freeze from two distinct fresh directories.
- `tools/compile_generated_cohort.py` is the only authorized final
  compiled-stage producer. Its canonical report
  retains per-map terminal logs and exit status, and publishes only the exact
  168-file declaration with atomic no-replace semantics.
- The compiled-CM preflight consumes every real BSP with the fresh-sealed CM
  authority and binds declaration, BSP, oracle, and implementation digests. It
  must pass exact 28/28 spawn identity, engine-linked stance/support/96-unit
  column, separation, escape, basic hazard containment, and lightdata checks
  before materialization. Copied `.map` validation cannot fill this report.
- `tools/materialize_generated_cohort.py` is the only authorized final
  cohort-level V4 stage producer. Its canonical report
  publishes only the exact 196-file declaration with atomic no-replace
  semantics. Its B1 authorities are explicit immutable inputs, never files
  discovered beside a cohort.
- `tools/run_generator_cohort.py verify-stage` writes canonical compiled and
  materialized membership reports. `tools/run_compiled_static_campaign.py`
  writes historical-schema `q2-generator-v6-compiled-static-campaign-v1`
  evidence with all 28 independently recomputed `static_validate` rows. This
  is source/static evidence despite its schema name; it is not compiled-CM
  evidence.
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
  its report plus four real `Q2LAT002` snapshots. Its selected map would have to
  be a member of an admitted population. The assembler binds it to the exact
  local Atlas manifest, raw Atlas, BSP, analyzer authority, crate commit, WSL
  identity, snapshot bytes, negative fences, and p99 measurement. Build the
  producer once from the clean committed root, retain that exact executable,
  and copy those same executable bytes to WSL. The gate independently decodes
  every header and zstd payload, validates the payload digest/cells/derived L3,
  and requires a byte-identical semantic and canonical-zstd re-encode. A magic
  prefix alone is never evidence.
- `tools/run_b2_test_suite.py --output NEW_DIRECTORY` executes the fixed Python
  syntax-floor, pytest, Rust, and standalone Dyn-helper suites and atomically
  writes `b2-test-report.json` plus eight hashed raw logs. The syntax-floor log
  binds the exact interpreter used by that suite; it complements rather than
  replaces the mandatory WSL Python 3.10 pre-declaration check. The directory
  must be outside the repository so its creation does not invalidate the clean
  Git binding.

## Retired cohort 71441 producer transcript

Cohort 71441 is retired and no replacement is authorized. The following
commands preserve the producer contract and the shape of its terminal attempt;
they are non-executable and must not be rerun against any 71441 source,
staging, log, report, WSL, producer-snapshot, or release-build path.

```sh
python tools/compile_generated_cohort.py \
  --declaration docs/multires/B2-GENERATED-COHORT-71441-DECLARATION.json \
  --source-root "$FUTURE_ROOT/source" \
  --staging-root "$FUTURE_ROOT/compiled-staging" \
  --publish-root "$FUTURE_ROOT/compiled" \
  --log-root "$FUTURE_ROOT/compile-logs" \
  --report "$FUTURE_ROOT/compile-report.json" \
  --q2tool /home/raymond/q2-rollout/q2-ml-bot/maps/q2tools/bin/q2tool \
  --basedir "$FUTURE_ROOT/assets/baseq2" \
  --timeout-seconds 3600
```

`--basedir` must name the `baseq2` directory that directly contains
`pak0.pak`, not its parent `assets` directory. The producer parses the PAK
directory and hash-binds the case-insensitive `pics/colormap.pcx` member. It
never searches a parent or sibling for assets. Maps are invoked in declaration
order with exactly `-bsp -vis -fast -rad -bounce 0 -threads 1 -basedir`;
lexical glob order is forbidden. The source root must have the exact 140-file
declaration, and successful postcompile membership must be exactly 168 files.
The staging, publication, log, and report leaves must all be absent at start.
Only an all-green population is published with
`renameat2(RENAME_NOREPLACE)`.

```sh
MATERIALIZER_PY=/home/raymond/miniconda3/bin/python3.11
B1_AUTHORITIES=/home/raymond/q2-multires-isolated/B1-authorities-909b1e46
"$MATERIALIZER_PY" -B tools/materialize_generated_cohort.py \
  --declaration docs/multires/B2-GENERATED-COHORT-71441-DECLARATION.json \
  --compiled-dir "$FUTURE_ROOT/compiled" \
  --stage-dir "$FUTURE_ROOT/materialized-staging" \
  --materialized-dir "$FUTURE_ROOT/materialized" \
  --log-dir "$FUTURE_ROOT/materialize-logs" \
  --report "$FUTURE_ROOT/materialize-report.json" \
  --cm-oracle "$B1_AUTHORITIES/q2-cm-oracle" \
  --pmove-oracle "$B1_AUTHORITIES/q2-pmove-oracle" \
  --hook-oracle "$B1_AUTHORITIES/q2-hook-oracle" \
  --fall-oracle "$B1_AUTHORITIES/q2-fall-oracle" \
  --hook-parity-attestation "$B1_AUTHORITIES/hook-parity-pullspeed-1700.json" \
  --timeout-seconds 900
```

Materialization also follows declaration order, requires the exact 168-file
compiled input, and publishes exactly 196 files only through
`renameat2(RENAME_NOREPLACE)`. Both producers fail fast. Their failed staging,
logs, and report roots are terminal, non-admissible, non-reusable evidence:
no retry, resume, subset, copy-forward, or future-cohort reuse is permitted.

The historical WSL authority directory
`/home/raymond/q2-multires-isolated/B1-authorities-909b1e46` is an independent
B1 input bundle for its original normative digests, not a cohort artifact. Its
canonical
`CONTENT-MANIFEST.json` SHA-256 is
`8d163d87a6919fc5d7f3761b17aa1aeaae7e71a5c505b80392a315802e11a92f`.
Its exact seven filenames are `B1-GATE.json`, `CONTENT-MANIFEST.json`,
`hook-parity-pullspeed-1700.json`, `q2-cm-oracle`, `q2-fall-oracle`,
`q2-hook-oracle`, and `q2-pmove-oracle`. Those immutable B1 bytes may be
referenced independently by the terminal 71439 attempt, but the directory must
remain outside all cohort roots and cannot prove cohort membership or progress.
The binary bytes may be inputs to a fresh B1 verification, but the old seal is
historical and cannot authorize future qualification or assembly. A fresh B1
seal must bind the amended design/plan digests. Neither operation authorizes
reuse of any retired cohort byte.

## Assembly template

All values are exact paths. `OUT` must not exist and must be outside the
implementation repository so publishing the gate cannot invalidate its own
clean-tree authority. No current alias authorizes execution. This template
becomes executable only after the fresh B1 seal, green non-admissible
qualification report, and separately committed final declaration exist.

```sh
python tools/assemble_b2_gate.py \
  --design docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md \
  --plan docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md \
  --repo-root "$PWD" \
  --b1-gate "$FRESH_B1_GATE" \
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
  --compiled-cm-preflight-report "$COMPILED_CM_PREFLIGHT" \
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
  --qualification-report "$QUALIFICATION_REPORT" \
  --output "$OUT"
```

The assembler independently verifies that qualification is green and
non-admissible, the fresh B1 seal binds the exact design/plan digests, the CM
preflight passes all 28 final BSPs, exact membership holds, all 28 generated
claim validations pass, and all eight stock validations pass. It derives
representative Atlas limits from the admitted analysis manifest selected by
the Dyn report rather than accepting a second budget assertion. The output is
canonical compact, sorted JSON with one trailing newline and is created with
exclusive-create semantics only after every B2 predicate is green.
