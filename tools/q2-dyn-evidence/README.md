# B2 Dyn evidence helper

`q2-dyn-evidence` is the fail-closed B2 runtime evidence command for one real,
currently admitted Atlas/BSP identity. It is a standalone crate so the evidence
driver itself does not enter the Atlas analyzer authority closure. Its build
script embeds deterministic helper and `q2-lattice` source-closure hashes; the
runtime recomputes both closures from `--repo-root` and rejects any mismatch.

The command requires explicit expected map, origin, analyzer-authority, crate
commit, map epoch, and environment-step bindings. The crate commit is also
embedded at build time; a stale or relabelled binary is rejected. The supplied
canonical Atlas manifest admits the raw Atlas, and the supplied BSP bytes must
match that manifest before any snapshot is written.

Build from a clean committed tree:

```sh
Q2_LATTICE_CRATE_COMMIT=$(git rev-parse HEAD) \
  cargo build --release --locked \
  --manifest-path tools/q2-dyn-evidence/Cargo.toml
```

Run on `DESKTOP-RTX2080` WSL with paths from the admitted current-authority
campaign (the output directory must not already exist):

```sh
tools/q2-dyn-evidence/target/release/q2-dyn-evidence \
  --repo-root "$PWD" \
  --atlas "$ATLAS" \
  --manifest "$ATLAS_MANIFEST" \
  --bsp "$BSP" \
  --expected-map-id "$MAP_ID" \
  --expected-origin "$ORIGIN_X,$ORIGIN_Y,$ORIGIN_Z" \
  --expected-analyzer-authority "$ANALYZER_AUTHORITY" \
  --expected-crate-commit "$(git rev-parse HEAD)" \
  --map-epoch "$MAP_EPOCH" \
  --environment-steps "$ENVIRONMENT_STEPS" \
  --samples 4000 \
  --output "$NEW_EVIDENCE_DIRECTORY"
```

The command atomically publishes exactly these files. Publication uses Linux
`renameat2(RENAME_NOREPLACE)`; an existing or racing destination wins without
being modified, and the losing staging directory is removed:

- `b2-dyn-evidence.json`, compact recursively key-sorted canonical JSON with
  one trailing newline
- `client0.q2lat002`
- `client1.q2lat002`
- `client2.q2lat002`
- `client3.q2lat002`

The report schema is `q2-b2-dyn-evidence-v1`. Its top-level fields are
`schema`, `passed`, `authority`, `provenance`, `host`, `atlas`, `dyn_state`,
`negative_fences_and_limits`, and `performance`. A gate may admit the report
only when all of the following are true:

- `passed` is `true`
- authority map/commit/analyzer/epoch values equal the campaign declaration
- `provenance.executable` matches the exact helper binary, both source closures
  recompute from their ordered input lists, equal their build-time embedded
  hashes, and are bound to the declared embedded repository commit
- all four snapshot hashes match the sibling files and all four round trips
  are byte-identical
- client IDs are exactly `0..3` at one common environment step
- every boolean in `negative_fences_and_limits` is `true`: stale Atlas, map,
  origin, map epoch, and environment step; wrong and duplicate client identity;
  retired and mixed schema; corrupted payload digest; mixed cell sizes; soft
  compressed notification; and hard compressed, resident, and materialized-cell
  rejection
- both combined byte counts are strictly below `8388608`
- `performance.resident_samples >= 2000`
- `performance.total.p99_ns < performance.total_p99_limit_ns == 500000`
- the host and kernel are the declared WSL evidence host, and
  `host.machine_identity_sha256` is SHA-256 of the canonical
  `/etc/machine-id` value: exactly 32 lowercase hexadecimal bytes after
  excluding at most one trailing LF; all other whitespace is rejected

The timed scope is one accepted resident transition across all four clients:
an origin-indexed exact L2 aggregate lookup in the admitted Atlas followed by
the 24-float Dyn feature assembly. Atlas load/decompression and snapshot I/O
are intentionally outside the resident hot-path measurement.

The helper source closure contains this crate's `Cargo.toml`, `Cargo.lock`,
`README.md`, `build.rs`, and recursively sorted `src/**/*.rs`. The lattice
source closure contains `crates/q2-lattice/Cargo.toml` and recursively sorted
`crates/q2-lattice/src/**/*.rs`. Its hash is SHA-256 over compact JSON records
`[{"path":RELATIVE_PATH,"sha256":FILE_SHA256},...]` in path order. File sizes
are reported but are not part of that closure digest. Each closure repeats the
embedded repository commit and reports a commit-bound SHA-256 over compact JSON
`{"repo_commit":COMMIT,"source_closure_sha256":CLOSURE}`. The B2 gate independently
rehashes the executable, both source closures, and fully decodes all snapshots;
the helper's own `passed` value is necessary but not sufficient admission.

Repository checks:

```sh
cargo fmt --manifest-path tools/q2-dyn-evidence/Cargo.toml -- --check
cargo test --manifest-path tools/q2-dyn-evidence/Cargo.toml
cargo clippy --manifest-path tools/q2-dyn-evidence/Cargo.toml \
  --all-targets -- -D warnings
```

Historical- or superseded-authority runs are diagnostics only and cannot be
copied into a current B2 campaign.
