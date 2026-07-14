# B1 Runtime Authority Seal

`harness/atlas_b1_authority.py` is the fail-closed bridge from the accepted
Batch B1 evidence to an Atlas analysis process. It does not discover tools,
trust filenames, or accept a caller-provided gate. It loads only the repository
`docs/multires/B1-GATE.json` and requires:

- the exact B1 schema, batch, green status/predicate, one-way retirement
  directive, admission invariants, and no recorded failures
- the accepted design and plan SHA-256 values, recomputed from the current
  repository files
- the supplied canonical hook parity artifact's byte SHA-256 to equal the gate
  digest, independent of its location
- exact parity schema, eight case IDs, check set, pullspeed, hook tool/physics,
  collision/Pmove fixture identities, fixture BSP digest, and vector digest
- supplied CM, Pmove, hook, and fall executable bytes to equal their sealed
  digests
- live identity responses from each executable with strict fields and
  self-consistent physics preimages
- CM/Pmove tool closure and exact source identities to equal B1, while their
  map and physics identities bind the actual BSP being analyzed
- hook runtime parameters/source/tool/physics to equal the parity attestation
- fall source/tool and default-parameter physics identity to equal B1

The absolute parity-artifact path recorded in B1-GATE is informational. The
admission code never opens it, so relocating the same bytes to another host is
valid and replacing a file at the old path is not.

## API

For an analyzer/build-tool integration, call the all-or-nothing entry point:

```python
from harness.atlas_b1_authority import admit_b1_runtime_authorities

seal = admit_b1_runtime_authorities(
    cm_oracle=cm_path,
    pmove_oracle=pmove_path,
    hook_oracle=hook_path,
    fall_oracle=fall_path,
    hook_parity_attestation=attestation_path,
    analysis_bsp=bsp_path,
)
```

It returns `B1RuntimeAuthoritySeal`; `seal.as_dict()` is deterministic and
manifest-ready. Any missing, malformed, stale, self-declared, noncanonical, or
mismatched input raises `B1AuthorityError`. There is no partial seal and no
fallback authority. The isolated `hookprobe-v1` BSP does not need to remain on
the runtime host because the accepted attestation bytes and B1 gate already
seal its digest, CM/Pmove fixture identities, vector transcript, q2ded checks,
and tool identities.

`load_b1_authority_gate()` and `admit_hook_parity_attestation()` are available
for diagnostics and preflight, but they do not replace the complete admission
call before Atlas construction.
