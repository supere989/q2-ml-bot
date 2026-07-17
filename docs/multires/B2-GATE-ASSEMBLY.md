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
No Dyn, assembled gate, deployment, or training action ran. At that historical
boundary there was no active final cohort, and replacement remained forbidden
until the fresh B1 seal and disposable qualification lane became green.

The fresh disposable qualification `b2q26_7005800_71618000` is green 28/28.
Its canonical report SHA-256 is
`99c13db93a8dacb9fe24f181126b8c30203f4005fdd5e96fb0b9a165ba2168f9`;
it remains explicitly non-admissible and cannot contribute population bytes or
a passing subset to the final gate.

Historical declaration statement: fresh replacement cohort
`b2g26_final_71443` was explicitly authorized. Its canonical named declaration
and then-current alias were byte-identical with SHA-256
`d890e151cbc3446622a8c0f5fdd2bd23352583c6431e1484262587c3c7246713`.
They declare 28 fresh rows in ordinal order, four per concrete style, using
seed blocks 71443000..71443003 through 71443600..71443603. The sole source
freeze and declaration-aware q2tool compile passed all 28 maps; compilation
atomically published the exact 168-file stage.

The test publisher was started out of order in parallel with compilation,
before compiled-CM preflight, promotion, Dyn, and the required final test
position. It selected syntax-only `/usr/bin/python3.10`, where pytest 9.1.1 was
present but `zstandard` was absent. Pytest exited 2 with six collection errors
and zero executed tests, and the publisher emitted neither its atomic output
root nor a test report. Compilation later completed, but no compiled-CM
preflight, materialization, claims, analysis, Dyn, gate, deployment, or
training action ran. `B2-GENERATED-COHORT-71443-FAILURE.json` is the exact
terminal authority, canonical SHA-256
`da89be636079b0cc38583281113002f0578d2608c5a31af052fca8c03d05f723`.
Its terminal phase is `out-of-order-test-runtime-preflight`.
It binds source-freeze report SHA-256
`6e748dd45bfd013cfd9c57f2ec60289b9abf40da946e511d771efe096d02a456`,
compiled report SHA-256
`c0c7f8c857e8ef60f0f74b959fef6b34f458fc69223146d7245ce2e79de76d84`,
and diagnostic log SHA-256
`196d25d0de40e4333dda9fe4c946e84ae571133554cb72e5ffa1c835bef1bb2d`.

Cohort 71443 is permanently retired. None of its source, compiled, log,
diagnostic, report, test, WSL, or producer-snapshot bytes may be retried,
repaired, resumed, reused, copied forward, salvaged, substituted, or admitted.
Its alias remains historical only and is rejected by the retirement registry
before campaign evidence is read. The replacement authorization below does not
alter that terminal history or permit any 71443 byte to cross into a successor.

## Retired 71444 final attempt

The test-runtime and publisher-ordering defects are corrected at the
methodology boundary: all execution-runtime dependency probes are pre-source,
and the sole final test publisher is ordered after generated promotion and
Dyn. Fresh B1 gate SHA-256
`eb99e08e5934d281556b0b6584ab23fe236adb8fce81f1cc7045229b368b9a25`
and disposable qualification `b2q26_275d4fa_71623700` bind implementation
commit `275d4fa646ccf2c64ba8628cd4aa8b21644fa90d`, tree
`7bd808b2194a44b80dc64fb88c700209d4657e9a`, and unchanged generator,
routes, and Atlas-analyzer authorities. The qualification is green 28/28 with
zero failures; its canonical report SHA-256 is
`09bd298d87739515d468f432219eefcad01e8586a87a71339f5121900a6f57c5`.
It remains explicitly non-admissible and supplies no map, seed, stage,
artifact, Dyn evidence, or passing subset to the final lane.

Exactly one immutable/no-retry final producer attempt is authorized for
`b2g26_final_71444`. The current canonical alias and immutable named
`B2-GENERATED-COHORT-71444-DECLARATION.json` are byte-identical with SHA-256
`da27e96b3fe8c3719a7ff1593e37b4ac768f53a36f38c877566af495a6b551bf`.
They declare exactly 28 fresh rows: four per concrete style, in declaration
order, using seed blocks 71444000..71444003 through 71444600..71444603. No
71444 producer may run until this declaration-bearing commit is clean and
every pre-source check below is green.

The first source-generation invocation consumes the sole authorization.
Any source or later-stage failure permanently retires 71444; there is no
resume, repair, repeated invocation, replacement member, passing subset,
salvage, or reuse under that declaration.

That terminal condition occurred at the final materialization authority preflight.
Source generation, q2tool compilation, compiled membership,
compiled-CM, and compiled/static validation passed all 28 maps. The sole
materializer invocation then rejected before any map subprocess because its
final-wrapper constant still named historical B1 gate SHA-256 `909b1e46...`
while the repository correctly contained fresh B1 gate SHA-256
`eb99e08e5934d281556b0b6584ab23fe236adb8fce81f1cc7045229b368b9a25`.
No materialized staging or publication exists; the materialization log root is
empty. `B2-GENERATED-COHORT-71444-FAILURE.json` is the exact terminal
authority, canonical SHA-256
`b709b038772e349583de4eea549ec16d6180ac820ea9ff1a4e382a0ec14ccf01`.
It binds source report `0986e0c70e04c7d1a70427c0218e079b885f2bbe269b3280a81a4245c2c7c098`,
compile report `2a93eb8782c488768eb1c81bade03872eced3e64ad65de16eec948d614986e33`,
compiled-CM report `a465649db8a9dc34da0e6513ef93710416bb849049608808cdaa256e9adaf4ff`,
and materialization failure report
`75c4d8fd2d38d9cc7ad4fdf32b612d4d761ff9ea3b46fdf66d3ec0a367cc1962`.

Cohort 71444 is permanently retired. None of its source, compiled, CM,
static, materialization-report, log, WSL, or producer-snapshot bytes may be
retried, resumed, reused, copied forward, salvaged, substituted, or admitted.
Its immutable named declaration is historical only and the retirement registry
rejects it before evidence. The final-wrapper authority fix, its exact no-write
preflight, the fresh B1 reseal, and a new disposable qualification are now
committed and green; they authorize only the distinct 71445 declaration below.

## Retired 71445 final attempt

Fresh B1 gate SHA-256
`58295d227ddd3694a0ddae5af46e2bbc98cc60dbe6b6751b4e42df01c06b1cd6`
and disposable qualification `b2q26_3b17223_71625100` bind implementation
commit `3b17223ab32e20152aead1eb32a79e239d6f4d8a`, tree
`fa2b106d19dbb115e6acd4c344b3820b3013464a`, and unchanged generator,
routes, and Atlas-analyzer authorities. The qualification is green 28/28 at
every population stage with zero failures; its canonical assembled report
SHA-256 is
`351baccaabf405e0ef240c1def18e4ede796ff417e73230524e9f0f9b0c0491b`.
It remains explicitly non-admissible and supplies no map, seed, stage,
artifact, Dyn evidence, or passing subset to the final lane.

Exactly one immutable/no-retry final producer attempt is authorized for
`b2g26_final_71445`. The current canonical alias and immutable named
`B2-GENERATED-COHORT-71445-DECLARATION.json` are byte-identical with SHA-256
`ffa5b9ccfee0340f1bad533a23fedd103a08d14d125149d1516a2326fb8a091b`.
They declare exactly 28 fresh rows: four per concrete style, in declaration
order, using seed blocks 71445000..71445003 through 71445600..71445603. No
71445 producer may run until this declaration-bearing commit is clean and
every pre-source check below is green.

The first source-generation invocation consumes the sole authorization.
Any source or later-stage failure permanently retires 71445; there is no
resume, repair, repeated invocation, replacement member, passing subset,
salvage, or reuse under that declaration. Qualification artifacts and every
retired 71426..71444 byte remain forbidden inputs.

That terminal condition occurred during the sole primary source generation.
The first 12 declarations were emitted as complete five-file forensic members;
ordinal 12, `b2g26_pits_71445300`, then raised
`could not place a unique lava-rim reward`. The cold population remained
empty, the source-freeze report was never published, and no compile or later
stage ran. Optional lava-pool
placement admitted and emitted its in-memory geometry before proving that the
paired mega-health reward had a final-geometry floor origin, so an exhausted
reward search could not transactionally decline the optional pool or try a new
candidate. Qualification's different seeds did not cover this rejection path.
`B2-GENERATED-COHORT-71445-FAILURE.json` is the exact terminal authority,
canonical SHA-256
`d134ddd35bb6e93f1fffa71d2b6176d402ba70c2d4242b2f55b6be40efd651af`.
It binds primary/cold membership reports
`cf87d90e7f7d40a9baae7e5bf54c27491f26d4a28531830f4a5cc79e4add1db7` /
`2167bfdef17cf247e329e5761dc7e44d3c22d34f5a3181faea5b8c2f737ee8a3`.

Cohort 71445 is permanently retired. None of its source, report, WSL, or
producer-snapshot bytes may be retried, resumed, reused, copied forward,
salvaged, substituted, or admitted. A successor requires the transactional
lava/reward fix, deterministic rejection-path coverage, and a fresh green
non-admissible qualification before a new immutable declaration.

## Authorized 71446 final attempt

The transactional lava/reward fix and its deterministic rejection-path test
are committed at `a05ddb7037774c1b246a6b13972b228570acb8ef`, tree
`01c27fc60da4ae6f2aedd6138c50dabfcd866525`. Fresh B1 gate SHA-256
`58295d227ddd3694a0ddae5af46e2bbc98cc60dbe6b6751b4e42df01c06b1cd6`
and disposable qualification `b2q26_a05ddb7_71626100` bind that exact
implementation, generator SHA-256
`fb71a121d05dc02ad4d634f537abb331ed7d4ea29da0e5c3199afe8c0b442001`,
unchanged routes authority, and unchanged Atlas-analyzer authority. The
qualification is green 28/28 at source, compile, compiled-CM,
materialization, claims, full independent cold Atlas build, and promotion,
with infrastructure 6/6 and zero failures. Its canonical assembled report
SHA-256 is
`69e2b1979feae22c706839dc24f8923b60e34d5b623c8f03b0e5ebb51181a549`.
It remains explicitly non-admissible and supplies no artifact, map, seed,
Dyn evidence, or passing subset to the final lane.

Exactly one immutable/no-retry final producer attempt is authorized for
`b2g26_final_71446`. The current canonical alias and immutable named
`B2-GENERATED-COHORT-71446-DECLARATION.json` are byte-identical with SHA-256
`58d52bd958249a70bf8115ab1c442fb6888a6d69b290a636303986f69acb658f`.
They declare exactly 28 fresh rows, four per concrete style in declaration
order, using seed blocks 71446000..71446003 through 71446600..71446603.
No 71446 producer may run until the declaration-bearing commit is clean and
every pre-source authority check below is green.

The first source-generation invocation consumes the sole 71446
authorization. Any source or later-stage failure permanently retires 71446;
there is no resume, retry, repair, repeated invocation, replacement member,
passing subset, salvage, or reuse under this declaration. Qualification
artifacts and every retired 71426..71445 byte remain forbidden inputs.

The declaration-bearing 71446 producer commit must be a strict successor of
qualified commit `a05ddb7037774c1b246a6b13972b228570acb8ef`. Gate replay
must prove that ancestry, the exact qualified commit/tree and report binding,
byte-identical generator/routes/Atlas-analyzer authority fields, and a complete
Git delta equal to the frozen 71446 declaration/gate/schema/direct-test
authorization path set. Any producer, analyzer, validator, physics,
toolchain, or unrelated committed change rejects the qualification.

The declaration-bearing 71445 producer commit was required to be a strict successor of
qualified commit `3b17223ab32e20152aead1eb32a79e239d6f4d8a`. Gate replay
must prove that ancestry, the exact qualified commit/tree and report binding,
byte-identical generator/routes/Atlas-analyzer authority fields, and a complete
Git delta equal to the frozen 71445 declaration/gate/schema/direct-test
authorization path set. That relation is historical and cannot authorize a
future cohort.

The declaration-bearing 71444 producer commit was required to be a strict successor of
qualified commit `275d4fa646ccf2c64ba8628cd4aa8b21644fa90d`. Gate replay
must prove that ancestry, the exact qualified commit/tree and report binding,
byte-identical generator/routes/Atlas-analyzer authority fields, and a complete
Git delta equal to the frozen 71444 declaration/gate/schema/direct-test
authorization path set. That relation is historical and cannot authorize a
future cohort.

Before source generation, the exact clean declaration-bearing snapshot must
pass the no-write `/usr/bin/python3.10 -B` syntax-floor scan, the pinned
`/home/raymond/miniconda3/bin/python3.11` materializer import/CLI preflights,
a no-write `tools/preflight_b2_materialization_authorities.py` invocation over
the exact five final oracle/attestation paths,
a same-process `import pytest, zstandard` probe, and
`python -m pytest --version`. These checks may run concurrently only when they
are read-only and must all finish green before the one source invocation. The
final producer lane itself is strictly sequential:
source/source-static, real compilation, compiled-CM preflight,
materialization/claims, full Atlas/cold rebuild, generated promotion, Dyn,
the sole atomic test suite, then assembly. Tests never overlap compilation or
any other final stage.

For historical 71443 gate replay, qualification necessarily preceded the
separately committed final declaration, so its repository commit could not
equal the final producer commit. That retired relation required the qualified
commit to be an ancestor, identical generator/routes/Atlas-analyzer authority
fields, and a complete Git delta equal to the frozen 71443
declaration/gate/test authorization path set. It cannot authorize 71444 or any
71443 byte; the later 71444 successor relation is also now historical and
cannot authorize a future cohort.

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
Before any future final-cohort source generation, the exact interpreter path
reserved for `tools/run_b2_test_suite.py` must also pass a same-process
`import pytest, zstandard` probe and `python -m pytest --version`. A missing
dependency retires no cohort because this is a pre-source infrastructure
check; after source publication the canonical test suite remains a single
no-retry action and runs only after generated promotion and Dyn, never in
parallel with compilation or another final stage.
Infrastructure evidence records these as two distinct authorities: the
`python310-syntax-floor` check binds `/usr/bin/python3.10` (CPython 3.10.12,
SHA-256 `7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86`),
while the infrastructure producer and materializer remain bound to the
separate CPython 3.11.4 executable and zstandard module digests above. The
syntax report must never be required to identify the execution runtime.

The assembler must reject declarations for retired cohorts 71426 through
71445 before reading campaign evidence. Only the exact fresh 71446 declaration
is eligible for the one final attempt authorized above.
Existing clean-repository, source-freeze, Atlas, test, manifest, and Dyn
requirements remain part of the frozen gate contract; all earlier B1/B2
admission evidence is historical only.

## Frozen producer-report contract

The 71443 test publisher produced no atomic test report. Qualification report
`99c13db93a8dacb9fe24f181126b8c30203f4005fdd5e96fb0b9a165ba2168f9`
satisfied the disposable prerequisite but remains `non_admissible: true` and
cannot override 71443 retirement. Qualification or retired bytes and passing
subsets can never satisfy a future final gate.

The retired 71444 prerequisite was the independent green 28/28
qualification `b2q26_275d4fa_71623700`, report SHA-256
`09bd298d87739515d468f432219eefcad01e8586a87a71339f5121900a6f57c5`,
bound to fresh B1 gate SHA-256
`eb99e08e5934d281556b0b6584ab23fe236adb8fce81f1cc7045229b368b9a25`
and qualified implementation commit/tree
`275d4fa646ccf2c64ba8628cd4aa8b21644fa90d` /
`7bd808b2194a44b80dc64fb88c700209d4657e9a`. Its one authorization was
consumed by 71444 and it remains non-admissible historical evidence rather
than final population input or successor authority.

The retired 71445 prerequisite was the independent green 28/28 qualification
`b2q26_3b17223_71625100`, report SHA-256
`351baccaabf405e0ef240c1def18e4ede796ff417e73230524e9f0f9b0c0491b`,
bound to fresh B1 gate SHA-256
`58295d227ddd3694a0ddae5af46e2bbc98cc60dbe6b6751b4e42df01c06b1cd6`
and qualified implementation commit/tree
`3b17223ab32e20152aead1eb32a79e239d6f4d8a` /
`fa2b106d19dbb115e6acd4c344b3820b3013464a`. It is non-admissible proof of
the fixed toolchain only; its authorization was consumed by the terminal
71445 source attempt and cannot authorize a successor.

The active 71446 prerequisite is the independent green 28/28 qualification
`b2q26_a05ddb7_71626100`, report SHA-256
`69e2b1979feae22c706839dc24f8923b60e34d5b623c8f03b0e5ebb51181a549`,
bound to fresh B1 gate SHA-256
`58295d227ddd3694a0ddae5af46e2bbc98cc60dbe6b6751b4e42df01c06b1cd6`
and qualified implementation commit/tree
`a05ddb7037774c1b246a6b13972b228570acb8ef` /
`01c27fc60da4ae6f2aedd6138c50dabfcd866525`. It is non-admissible proof
only; the separate immutable 71446 declaration consumes its sole successor
authorization.

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
clean-tree authority. The current alias names fresh cohort 71446 and can
execute this template. The command shape below becomes executable only after a
fresh final population completes every preceding stage and supplies the exact
71446 evidence paths.

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
