# B3 Design-Prior Campaign

`tools/run_b3_design_prior_campaign.py` is the authoritative offline producer
for the B3-B generator-feedback proof. It does not compile, install, stage, or
launch anything. The two-phase protocol prevents metric selection after seeing
generated results.

## Frozen methodology

`prepare` reads exactly the stock `q2dm1` through `q2dm8` compiled-Atlas
`*.design-signature.json` files, rejects every field outside the coordinate-free
signature contract, and emits integer/fixed-point aggregate priors. Coordinates,
room or portal graphs, adjacency, edge endpoints, and community-map input have
no representation in the accepted schema.

The command freezes 28 unique seeds, exactly four maps in each of the seven
styles in both lanes, knob deltas, source/analyzer identities, the authoritative
design/plan hashes, and every metric threshold. Baseline and treatment use the
same deterministic style for each seed. Treatment may alter only allowlisted
knobs; changing style mixture is rejected as a confounder.

The only accepted treatment knobs are:

`occupied_density`, `corridor_prob`, `hallway_ratio`, `terrace_levels`,
`tower_prob`, `lane_prob`, `lava_prob`, `extra_arena_prob`,
`arena_cover_range`, `corner_range`, and `large_building_ratio`.

Probability deltas are integer ppm encoded as `delta + 200000`, so the accepted
range 0 through 400000 represents -200000 through +200000 ppm. The terrace
delta is encoded as `delta + 1`; two-element range deltas encode each element as
`delta + 2`. Effective probabilities must remain in `[0,1000000]`, terraces in
`[1,4]`, and ordered range endpoints in `[0,12]`. The bias document must repeat
the exact frozen balanced weights; they are an identity check, not a treatment
degree of freedom.

`evaluate` consumes two canonical lane manifests and the exact compiled-Atlas
design-signature directories. Every manifest row must match the frozen seed,
style, effective knobs, source/static pass result, layout digest, BSP digest,
and signature digest. All 28 rows are mandatory; there is no subset, retry,
replacement, or missing-signature mode. The signature BSP identity must match
the lane record. Repository, generator, analyzer, stock, and normative-document
identity drift aborts without publishing a report.

The frozen primary distances are integer total-variation ppm for topology
(degree and typed-edge mass), economy (item-class mass), and environment
(edges/node, items/spawn, and lightmapped-face ratio bands). Treatment must
strictly improve all three and their aggregate. Both lanes must retain a 100%
source/static pass rate. Exact layout uniqueness must remain 100%; descriptor
uniqueness and style Simpson-diversity have declared floors, and treatment
style diversity may regress by at most 50000 ppm.

## Input documents

The seed document is canonical compact/sorted JSON plus LF:

```json
{"schema":"q2-b3-design-prior-seeds-v1","seeds":[28 unique integers]}
```

The bias document has schema `q2-b3-generator-bias-v1`, exact seven-style
`style_weights_ppm`, and a nonempty allowlisted `knob_delta` object.

A lane manifest has schema `q2-b3-design-prior-lane-v1`, the exact plan digest,
`evidence_kind: "measured-compiled-atlas"`, `synthetic_claims: false`, the
frozen implementation object, no failures, `passed: true`, and 28 rows. A row
contains `ordinal`, `map`, `seed`, `style`, exact `generator_knobs`,
`source_static_passed`, `layout_sha256`, `bsp_sha256`, and a `{bytes,sha256}`
record for the design signature.

`tools/run_b3_design_prior_lanes.py` is the concrete local-only lane producer.
It preflights every authority, timeout, job bound, parent directory, and path
overlap before creating its work root. The compiled-CM timeout is finite in
`(0,60]`; the rejected historical value `3600` cannot reach the filesystem. It
then runs source freeze/static validation, q2tool, compiled-CM, materialization,
claims, Atlas, and promotion validation in that order. The output binds all
eight stage reports and all nine executable or evidence authorities. The Atlas
packer and verifier are required, preflighted non-symlink executable files,
supplied by absolute path, passed explicitly into Atlas construction, and
recorded separately in the lane manifest. Relative paths are rejected before
the lane work root exists so the child snapshot working directory cannot
rebind an authority. An ambient Cargo build inside the immutable implementation snapshot
is not an admitted substitute. There is no remote, resume, subset, replacement,
or runtime-install mode.

Example workflow:

```text
python tools/run_b3_design_prior_campaign.py prepare \
  --campaign-id b3_prior_01 --stock-analysis-dir /artifact/stock-analysis \
  --bias /artifact/bias.json --seeds /artifact/seeds.json \
  --output /artifact/b3-prior-plan.json

# Run each declared generator -> static -> q2tool -> Atlas lane offline with
# tools/run_b3_design_prior_lanes.py, exact B1 authority paths, and binaries
# built from the same clean repository identity, for example:
cargo build --locked --release -p q2-lattice --bins \
  --target-dir /artifact/b3-toolbuild

python tools/run_b3_design_prior_lanes.py \
  ... \
  --packer /artifact/b3-toolbuild/release/q2-atlas-pack \
  --verifier /artifact/b3-toolbuild/release/q2-atlas-verify

python tools/run_b3_design_prior_campaign.py evaluate \
  --plan /artifact/b3-prior-plan.json \
  --stock-analysis-dir /artifact/stock-analysis \
  --baseline-lane /artifact/baseline-lane.json \
  --baseline-analysis-dir /artifact/baseline-analysis \
  --treatment-lane /artifact/treatment-lane.json \
  --treatment-analysis-dir /artifact/treatment-analysis \
  --output /artifact/b3-prior-campaign.json
```

The final report conforms to
`schemas/q2-b3-design-prior-campaign-v1.schema.json`. A red regression decision
is durable evidence of failure, never permission to retune the frozen report.
A new treatment requires a new campaign ID and a new plan prepared before its
lanes exist.
