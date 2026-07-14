# Exact controlled-drop replay contract

`harness/atlas_drop_replay.py` is the offline bridge between the pinned
`q2-pmove-oracle` trajectory and Lithium's pinned `q2-fall-oracle` impact law.
It does not approximate gravity, collision, landing, water mitigation, or fall
damage, and it does not modify or depend on `atlas_analyzer.py`.

## Manifest

The strict `q2-atlas-drop-replay-manifest-v1` object contains exactly:

```text
schema, id, map_path, horizon_frames, cadence_msec, dynamic_movers,
pmove, fall, authorities
```

`horizon_frames` is 2 through 4096 and must equal the ordered command count.
Every command must use the declared `cadence_msec`; mixed or invalid cadence is
Unknown. `pmove` carries the complete initial state, gravity, air acceleration,
angle delta, snap mode, and commands. `fall` binds effective
`fall_damagemod`, deathmatch mode, `dmflags`, and positive starting health.
Float parameters are reduced to their effective IEEE-754 f32 values before
identity comparison.

`authorities.pmove` pins the executable, tool, physics, and BSP SHA-256 values.
`authorities.fall` pins the executable, tool, and parameter-bound physics
SHA-256 values. The map file digest must equal the pmove identity's map digest.
All digests are lowercase SHA-256 strings.

Dynamic brush-mover state is not represented by `q2-pmove-oracle-v1`.
Therefore `dynamic_movers` must be `false`; a mover-dependent candidate is
Unknown and its controlled-drop edge is omitted.

## Landing and fall request

Pmove frames must be a sealed, contiguous sequence with `command_index` equal
to array position, `command_count` equal to the manifest horizon, and `final`
byte-for-JSON equal to the last frame. Every world origin and velocity must
equal its signed fixed-point value multiplied by 1/8. Water level is restricted
to 0 through 3.

The landing is the first adjacent frame pair where `grounded` changes from
`false` to `true`. Frame zero can never prove a landing because it has no
preceding oracle frame. Thus an initially grounded/same-frame result is
Unknown unless a later airborne-to-ground transition occurs.

The fall evaluation uses:

- `old_velocity_z` from the preceding airborne pmove frame
- `velocity_z`, `waterlevel`, and exact fixed landing origin from the landing frame
- manifest-bound fall parameters and health
- player model 255, `MOVETYPE_WALK` 4, and `grounded=true`
- no Orange hook, no CTF grapple, grapple state FLY, and a one-second
  no-recent-release sentinel

The exact request is retained in the result. The trajectory SHA-256 covers a
domain tag, map and pmove physics identities, exact simulate request, every
ordered frame, and final state using canonical sorted-key JSON.

## APIs and process model

For persistent or batched oracle integration:

```python
pmove_requests = pmove_requests_for_drop_manifest(manifest)
# Batch pmove_requests["identity_request"] and ["simulate_request"].
prepared = prepare_drop_fall_request(
    manifest,
    pmove_identity=pmove_identity,
    pmove_response=pmove_simulation,
    pmove_executable_sha256=pmove_binary_digest,
    map_sha256=bsp_digest,
)
# Batch prepared["fall_identity_request"] and prepared["fall_request"].
result = evaluate_drop_evidence(
    manifest,
    pmove_identity=pmove_identity,
    pmove_response=pmove_simulation,
    fall_identity=fall_identity,
    fall_response=fall_evaluation,
    pmove_executable_sha256=pmove_binary_digest,
    fall_executable_sha256=fall_binary_digest,
    map_sha256=bsp_digest,
)
```

Both functions are pure: they perform no filesystem access and launch no
processes. `replay_drop(...)` is a convenience adapter that hashes local
artifacts and accepts either an injected reusable runner/session or the bundled
one-shot subprocess runner. The CLI is:

```sh
python3 -m harness.atlas_drop_replay \
  --manifest drop.json --pmove-oracle /attested/q2-pmove-oracle \
  --fall-oracle /attested/q2-fall-oracle
```

## Result and omission rule

Only fully admitted evidence returns `classification: "Exact"` with
`safe`, `lethal`, and `severity`. Exact evidence also includes the fixed and
world landing origin, impact velocities, water level, trajectory digest, both
request payloads, full oracle identities, pmove response binding, and complete
fall response. `safe` is the inverse of the fall oracle's unmitigated lethal
projection; mitigation beyond the raw `T_Damage` request remains outside this
contract.

Every failure returns `classification: "Unknown"` and
`omit_controlled_drop_edge: true`, without `safe`, `lethal`, or `severity`.
This includes a missing authority, identity or executable mismatch, malformed,
nonfinite, tampered, duplicated, or out-of-order evidence, invalid water or
cadence, no landing within the bounded horizon, and dynamic-mover dependence.
There is no geometry-only, ballistic, or default-damage fallback.
