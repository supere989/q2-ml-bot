# B3 Gate Assembly

`tools/assemble_b3_gate.py` assembles the offline B3 milestone only. It performs
no build, cross-host copy, service change, trainer launch, TensorBoard launch,
runtime install, or bundle-v3 enablement.

The assembler requires four canonical evidence documents:

1. the exact green `q2-multires-b2-gate-v1` predecessor bound to the current
   authoritative design and plan and to the explicitly activated, non-retired
   B2 final-cohort authority;
2. a green `q2-b3-design-prior-campaign-v1` measured campaign;
3. `q2-b3-recovery-guide-evidence-v1` measured across an Atlas set; and
4. `q2-b3-bundle-evidence-v1` from offline installer/farm tests.

The component documents conform to
`schemas/q2-b3-recovery-guide-evidence-v1.schema.json` and
`schemas/q2-b3-bundle-evidence-v1.schema.json` respectively.

The recovery/guide document must bind the current repository commit/tree and
the assembler-recomputed recovery/guide source closure. It also records an
exact regular `q2_lattice_rs` extension file, its bytes/SHA-256, repository
tree, source closure, and qualification-command identity. Ambient
`PYTHONPATH` imports are forbidden. It records the finite
non-safe cell count, strict-descending count, explicit mover-plateau count,
zero unresolved cells, maximum local repair size no greater than 4096, the five
frozen hazard classes, 15-tick walking budget, 10 Hz cadence, and 300u/s Q8
speed for the hook-necessity debug label, 16 recovery floats, and four 15-float
guide candidates using the frozen eight objective classes. The producer uses
the Rust evaluator on every current hook-source cell and requires a nonzero,
complete edge count. At least one objective-bearing fixture must be nonzero.
Hook necessity remains private teacher/debug evidence and never enters public
Recovery16 or Guide60.

`tools/run_b3_component_evidence.py recovery-guide` runs the frozen Rust
recovery, guide, hook-necessity, and extension-backed Python qualification
commands itself. `bundle` runs the frozen map-farm suite itself. Every child is
placed in an isolated process group, has a finite 600-second ceiling, and is
terminated as a group on timeout. Evidence contains every exact command, exit
status, pass count, and stdout/stderr byte/hash record. The gate recomputes the
command-list and report digests; arbitrary nonzero hashes are rejected.

Bundle evidence is claim-addressed rather than inferred from an aggregate
pytest exit status. `BUNDLE_CLAIM_NODE_IDS` predeclares the exact
`tests/test_map_farm.py::test_*` node IDs supporting each bundle-v2,
bundle-v3, and farm boolean. Each node runs as its own command and must appear
exactly once with exit zero and exactly one passed test. `claim_tests` records
the required IDs and `passed` outcome for every claim, including the negative
derived values `public_enabled: false` and
`vps_compilation_fallback_enabled: false`. The producer derives the three
boolean sections from this mapping; the gate independently reconstructs both
the mapping and booleans from the test-run records. A generic passing
`tests/test_map_farm.py` invocation, missing node, renamed node, duplicate,
non-single outcome, or hand-stamped boolean is rejected. The bundle schema
also pins each claim's ordered `required_node_ids` and `node_outcomes` to these
exact nodes; generic claim-shaped JSON is insufficient.

The production `produce_bundle()` API accepts only an output path and
repository root. It always reads the real clean Git identity and executes the
frozen node commands itself; there is no repository/test injection argument.
The private in-memory document builder exists only for validation tests and
cannot publish an evidence file.

The production `assemble_b3_gate()` API likewise accepts only its path bundle.
It reads the real clean Git identity itself before publication. Its private
report builder can validate deterministic fixtures in memory but performs no
write, so an injected repository identity cannot be used to publish a gate.

The farm health fixture exercises the concrete `report_analysis_failure()`
path and requires `analysis_status: "failed"`, the exact surfaced error, and
the immutable `vps_compilation_fallback_enabled: false` status. This prepares
the isolated analysis lane without enabling bundle v3 or compilation on the
public VPS.

The bundle document likewise binds the current commit/tree and recomputed
bundle source closure. It proves bundle-v2 consumers install with analysis
artifacts both present and absent, bundle v3 is isolated and public-disabled,
missing or mismatched mandatory Atlas artifacts fail closed, atomic failure
restores the prior isolated generation, farm health exposes analysis failure,
and VPS compilation fallback remains disabled.

There is deliberately no active B2 final-cohort authority after the terminal
retirement of `b2g26_final_71446`. While
`tools.assemble_b2_gate.ACTIVE_FINAL_AUTHORITY` is `None`, B3 assembly and B3
validation both fail before accepting a predecessor. A future B2 qualification
must add and explicitly activate a fresh immutable successor declaration. B3
then calls the production B2 `validate_gate()` implementation, rejects 71446
even if a process attempts to nominate it as authority, and requires both the
cohort ID and declaration SHA-256 to equal that active authority. The sealed B3
predecessor section carries those two identities plus the exact B2 gate-file
digest; a B3 gate from an earlier active authority is invalid after authority
rotation. A minimal object that merely claims B2/green status is not an
admissible predecessor.

Every evidence document must name a measured evidence kind and set
`synthetic_claims` to false. All-zero hashes, placeholder markers, duplicate or
noncanonical JSON, source-closure drift, repository drift, normative-document
drift, a red predecessor, and pre-existing output are hard refusals.

The assembled gate carries `gate_sha256`, computed over canonical JSON for
`{"domain":"q2-multires-b3-gate-v1","gate":<gate without gate_sha256>}`.
`validate_b3_gate()` requires the exact top-level and nested B3 structure,
revalidates all fixed green/offline predicates and cross-bindings, and
recomputes this domain-separated seal. Downstream milestones must call this
validator or reproduce the same computation; accepting a minimal object merely
because it says `schema`, `green`, or `status` is forbidden.

Concrete component production precedes assembly:

```text
python tools/run_b3_component_evidence.py recovery-guide \
  --declaration /artifact/declaration.json \
  --claims-dir /artifact/claims --analysis-dir /artifact/analysis \
  --extension-file "$PWD/target/debug/libq2_lattice_rs.so" \
  --output /artifact/b3-recovery-guide.json

python tools/run_b3_component_evidence.py bundle \
  --output /artifact/b3-bundle.json
```

Example:

```text
python tools/assemble_b3_gate.py \
  --b2-gate /artifact/B2-GATE.json \
  --prior-campaign /artifact/b3-prior-campaign.json \
  --recovery-guide-evidence /artifact/b3-recovery-guide.json \
  --bundle-evidence /artifact/b3-bundle.json \
  --output /artifact/B3-GATE.json
```

The output conforms to `schemas/q2-multires-b3-gate-v1.schema.json`. Green
means only that B3 offline recovery, guide, prior, and bundle-preparation gates
closed. Bundle v3 remains isolated; B4 protocol work, B5 policy validation, WSL
staging, training, and public deployment remain unauthorized.
