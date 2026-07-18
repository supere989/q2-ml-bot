# Primary Trainer and TensorBoard Admission

`train.multires_service` is the only primary-trainer/TensorBoard selector. Its
configuration schema is `q2-multires-service-v2`; its retirement declaration
schema is `q2-multires-cold-start-v2`. Version 1 service/cold-start documents
are permanently non-admissible because they can describe only B4-era inputs.

## Freeze point

Freeze and commit the bot repository before B3. Use that exact clean commit and
tree through B3, B4, B5, B6, final integration verification, service preflight,
and launch. The B6 `source_repositories.bot` record and the live clean checkout
running `train.multires_service` must both equal the configured source pair.
A later documentation-only commit is still a different source and requires a
new downstream chain.

After the final B6 aggregate exists, create the integration envelope and its
precomputed canonical report:

```sh
python3 tools/verify_multires_integration.py \
  --evidence /absolute/evidence/integration-envelope.json \
  --out /absolute/evidence/integration-report.json
git rev-parse --verify 'HEAD^{commit}'
git rev-parse --verify 'HEAD^{tree}'
git status --porcelain=v1 --untracked-files=normal
sha256sum /absolute/evidence/integration-envelope.json \
  /absolute/evidence/integration-report.json
```

The status output must be empty. Do not edit, reformat, copy over, or re-seal
either JSON after recording its raw-file SHA-256.

## Service configuration

The `proof` object must include both `atlas_catalog` and
`expected_atlas_catalog_sha256`. The cold-start declaration repeats the
catalog as `inputs.atlas_catalog` and binds the semantic digest at top level as
`atlas_catalog_sha256`. Checkpoints bind that stable catalog digest; the
proof's `expected_atlas_sha256` remains the selected active map. See
[`ATLAS-CATALOG-LINEAGE.md`](ATLAS-CATALOG-LINEAGE.md) for the portable catalog
and per-map Dyn lifecycle.

`multires-service.json` has the existing exact fields plus this required
record. Paths and digests are examples, not selectors to copy literally:

```json
{
  "schema": "q2-multires-service-v2",
  "integration_admission": {
    "envelope": {
      "path": "/absolute/evidence/integration-envelope.json",
      "sha256": "<raw-envelope-file-sha256>"
    },
    "report": {
      "path": "/absolute/evidence/integration-report.json",
      "sha256": "<raw-report-file-sha256>"
    },
    "bot_source": {
      "commit": "<exact-bot-git-commit>",
      "tree": "<exact-bot-git-tree>"
    }
  }
}
```

The `inputs` object in the exact `q2-multires-cold-start-v2` declaration must
repeat the same immutable records and source identity:

```json
{
  "integration_envelope": {
    "path": "/absolute/evidence/integration-envelope.json",
    "sha256": "<raw-envelope-file-sha256>"
  },
  "integration_report": {
    "path": "/absolute/evidence/integration-report.json",
    "sha256": "<raw-report-file-sha256>"
  },
  "integration_bot_source": {
    "commit": "<exact-bot-git-commit>",
    "tree": "<exact-bot-git-tree>"
  }
}
```

Both integration files must be regular non-symlink files under an admitted
operational root. Every evidence path in the envelope must resolve to a
regular non-symlink file, and the envelope must contain exactly the nine
`verify_multires_integration` gate slots.

## Mechanical launch gate

Preflight reruns `verify_multires_integration` against the immutable envelope
and requires byte-for-byte equality with the precomputed canonical report,
including its `report_sha256`, raw envelope hash, gate order, per-gate evidence
hashes, empty failure set, and overall pass. It then cross-checks the service's
runtime identity, stable Atlas catalog, active-map Atlas identity, retirement
manifest, exact runtime manifest,
runtime evidence, checkpoint, training manifest, objectives, bundle manifest,
Atlas file, and bot source against the B6 aggregate.

For `start`, this happens once before the selector lease or interrupted-attempt
reconciliation and again before trainer process creation. The wrapper transfers
the private selector lease to that child and waits up to 30 seconds for the
child's sealed `multires-primary-attempt.json`, which is written only after the
child independently repeats `service_preflight`. The wrapper validates that
attempt against the exact runtime, training configuration, checkpoint, run
roots, process identity, and selector-token digest, then reruns integration
admission once more. TensorBoard cannot be spawned before those checks pass and
the admitted trainer is still alive. Any changed byte, path rebound, dirty
checkout, source drift, stale report, missing B6 binding, failed gate, malformed
child attempt, or early trainer exit stops both processes. Run:

```sh
python3 -m train.multires_service --runtime_root /absolute/runtime preflight
python3 -m train.multires_service --runtime_root /absolute/runtime start
```

Only the first command's successful `integration_admission.verification` and a
successful second in-process verification permit launch. A green B6 aggregate
alone remains non-admissible.

## Curriculum selector changes

Fresh stage 1 and later finite-stage changes are authored by
`tools/manage_b7_lifecycle.py`; see
[`B7-FRESH-TRAINING-LIFECYCLE.md`](B7-FRESH-TRAINING-LIFECYCLE.md). Stage 1
uses `fresh-step-zero`. A passed predecessor gate advances the same explicit
lineage with `same-lineage-stage-advance`; this mode requires stage greater
than one, the exact gate-bound completed report at `season/current.json`, and
the checkpoint named and hashed by that report. It never scans a directory for
the latest checkpoint. The service-v2 selector is the final atomic write in an
advance transaction, and the complete integration admission is re-run before
the changed selector can create a process.

The service module must run from the exact clean Git worktree root whose
commit/tree is frozen in B6. A copied operational mirror without its own Git
worktree is deliberately non-admissible, even if its Python files happen to
match; deploy and launch from the staged exact-source checkout.
