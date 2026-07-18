# B7 fresh-training lifecycle

`tools/manage_b7_lifecycle.py` is the executable boundary between the green
B5/B6 no-update evidence chain and the new B7 lineage. It never starts a
trainer, changes a public host, or promotes a policy. `train.multires_service`
remains the only trainer/TensorBoard selector.

## Fresh stage 1

`author-stage1` requires the exact green B5 gate and a green B6 aggregate that
binds those B5 bytes and the step-zero checkpoint. It refuses an occupied run
tag or service selector. It creates the one allowed run tag,
`public_network_multires_atlas_fresh_v1`, copies only the B6-bound random
step-zero checkpoint into its otherwise-empty checkpoint root, and authors
digest-named immutable primary-runtime and cold-start-v2 documents. The
service-v2 selector is written last.

The three templates contain the already-attested B6/integration/runtime
bindings and operational settings. The tool replaces only the lifecycle-owned
stage, checkpoint, run-root, and selector records. A template is not evidence
and cannot substitute for the B5/B6 binding check.

```sh
python3 tools/manage_b7_lifecycle.py author-stage1 \
  --b5-gate /absolute/B5-GATE.json \
  --b6-gate /absolute/B6-GATE.json \
  --stage-configuration /absolute/stage-01.json \
  --primary-template /absolute/primary-template.json \
  --cold-start-template /absolute/cold-start-v2-template.json \
  --service-template /absolute/service-v2-template.json \
  --runtime-root /absolute/new-b7-runtime
```

Every stage configuration freezes `minimum_accepted_transitions` and a
nonempty `gate_predicates` list. A predicate has exactly `name`, `path`,
`operator`, and `threshold`; operators are `eq`, `ge`, `gt`, `le`, and `lt`.
The evaluator reads those paths from the sealed current-season report. Missing,
non-comparable, or non-finite values fail closed.

## Stage evaluation and advancement

Run a stage with a finite update count, stop it cleanly, and call `advance`.
The tool verifies the sealed `season/current.json`, derives the evaluator
decision from the frozen predicates, hard-links an immutable archive of that
exact report, seals a predecessor gate, and creates the next primary runtime
with checkpoint mode `same-lineage-stage-advance`. It creates a new immutable
cold-start-v2 declaration whose trainer checkpoint and primary-runtime records
select the same lineage. The existing service configuration is compared with
the bytes read at transaction start and atomically replaced last.

```sh
python3 tools/manage_b7_lifecycle.py advance \
  --runtime-root /absolute/new-b7-runtime \
  --next-stage-configuration /absolute/stage-02.json
```

The prior `season/current.json` intentionally remains in place until the first
successful update of the next stage replaces it. Both primary admission and
the training core accept it only when its raw bytes, counters, checkpoint,
lineage, stage identity, and path equal the sealed predecessor-gate bindings.
This removes the former old-report/new-stage mismatch without deleting evidence
or scanning for a latest checkpoint.

Each gate contains `automatic_promotion: false`. A failed evaluator writes its
immutable result but writes no gate and changes no selector. Stage 7 is
terminal for this tool; B8/B9 own later season, shadow, restart, and promotion
decisions.

## Stage 7 guide-off requirement

Stage 7 is `full-guide-off-ablation`. Its immutable configuration must select
guide mode `off`, bind a sealed matched-seed guide-on reference by absolute path
and SHA-256, name the guide-off task-success and global-dropout metric paths,
freeze a neutral/random baseline, and cap allowed guide-off degradation at
`0.15` or less. The evaluator requires:

- the matched seed to equal the guide-on reference;
- positive global guide dropout;
- guide-off task success above the neutral baseline; and
- relative guide-off degradation no greater than the frozen cap.

The ordinary frozen gate predicates are evaluated in addition to these
stage-specific checks. Passing stage 7 still makes no promotion claim.

## Primary telemetry admission

The primary B7 measurement seam is not synthetic. Horizontal movement speed
and true view pitch are derived from the ordinary client's own public
`ml_obs_t`; guide class/drop decisions come from the exact seeded advisory
transform that produced the action's policy vector. The collector binds that
transform audit to client and source frame, rejects extra fields (including a
teacher action), and publishes it only after the complete rollout admits.

Resident Atlas/Dyn samples come from the active Rust provider. Four clients'
Dyn, Atlas lookup, recovery, and guide timings are summed per lockstep frame,
while Atlas identity/count/RSS and public thermal/Dyn counters remain
separately auditable. Every accepted lockstep round has real public
speed/pitch/guide audit and one four-client Rust Atlas/Dyn runtime sample. The
sample is admitted through `observe_runtime_snapshot` before PPO mutation.
Missing, partial, mixed-client, private, or malformed telemetry fails the
primary update closed.
