# Multires isolated WSL source staging

`tools/stage_multires_wsl.py` is the B6 source-triple transport boundary. It
does not build binaries, start a server, start a trainer, alter a service, or
touch the public/teacher runtimes. It only publishes a new, isolated source
stage on the WSL host whose hostname is exactly `DESKTOP-RTX2080`.

The tool has two explicit modes:

- `preflight` performs all local source and remote destination/toolchain checks
  without creating a remote file or directory.
- `stage` repeats those checks, transfers committed Git objects, reconstructs
  and verifies the three repositories, then atomically publishes a previously
  nonexistent destination.

## Required bindings

The caller supplies the expected branch, commit, and tree for each repository.
The defaults bind all three branches to `feature/multires-map-atlas-v1`; commit
and tree values have no default. Every repository must be clean, including
untracked files, and must contain no tracked symlink or submodule. The tool
also binds the exact SHA-256 and size of the authoritative design and plan in
the bot repository.

The WSL Rust toolchain is fixed at:

```text
/home/raymond/q2-multires-isolated/tooling/rust-1.96.1-x86_64-unknown-linux-gnu
```

Pass the independently verified SHA-256 values for its regular, non-symlinked
`bin/rustc` and `bin/cargo`. The tool recomputes both hashes remotely in both
modes. It does not trust a path or version string alone.

The declared destination must be an absolute path below an existing absolute
isolated root. Its parent must exist, neither the root nor the parent may
contain a symlink component, and the destination may not already exist in any
form. Use a new leaf for every cutover attempt; old stages are never overlaid
or deleted.

## Operator sequence

After the three repositories have been committed and are clean, capture each
identity:

```bash
git -C /path/to/repo symbolic-ref --short HEAD
git -C /path/to/repo rev-parse HEAD
git -C /path/to/repo rev-parse 'HEAD^{tree}'
```

Run the no-write check first. The complete invocation intentionally names all
attested source identities:

```bash
python3 tools/stage_multires_wsl.py preflight \
  --bot-repo /path/to/q2-ml-bot \
  --bot-commit BOT_COMMIT --bot-tree BOT_TREE \
  --client-repo /path/to/q2-ml-client \
  --client-commit CLIENT_COMMIT --client-tree CLIENT_TREE \
  --game-repo /path/to/q2-lithium-3zb2 \
  --game-commit GAME_COMMIT --game-tree GAME_TREE \
  --host wsl-box \
  --isolated-root /home/raymond/q2-multires-isolated/B6 \
  --destination /home/raymond/q2-multires-isolated/B6/NEW_STAGE \
  --rustc-sha256 RUSTC_SHA256 \
  --cargo-sha256 CARGO_SHA256
```

Only a green preflight should be repeated with `stage` in place of
`preflight`. `stage` re-inspects the source triple after creating the bundles,
so a branch, commit, tree, index, or untracked-file race fails before SSH
transfer.

## Publication and failure semantics

Transport is a fixed-membership tar containing one canonical transfer manifest
and three Git bundles. Remote extraction rejects links, non-regular members,
unsafe paths, duplicates, missing members, oversized data, and bundle
hash/size mismatches. Each bundle is fetched into a new repository and checked
for exact branch, commit, tree, clean status, strict Git object validity,
tracked-file count, a canonical SHA-256 of the complete recursive Git tree
closure, and a second canonical closure containing each path, size, and the
independent SHA-256 of its blob bytes. The staged copies of the normative
documents are hashed again.

Work occurs beneath a nonce-qualified temporary sibling created by that
invocation. Any error removes only that temporary sibling. Publication uses
Linux `renameat2(RENAME_NOREPLACE)`, so a destination that appears during the
transfer cannot be replaced, even if it is empty. A successful destination
contains only:

```text
repositories/q2-ml-bot/
repositories/q2-ml-client/
repositories/q2-lithium-3zb2/
staging-manifest.json
```

`staging-manifest.json` is canonical JSON written atomically with mode `0600`.
Its `semantic` payload binds host, source commits/trees/file closures,
normative hashes, toolchain paths/hashes, isolated root, destination, and
explicit declarations that no public runtime, service, or trainer changed.
The timestamp is isolated under `informational`; it does not affect
`semantic_sha256`.

This stage is source provenance for later clean builds and isolated B6 tests.
It is not a B6 gate, a trainer admission, or permission to mutate a live
runtime.
