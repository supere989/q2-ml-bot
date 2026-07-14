# Target/thermal-lattice prototype — 2026-07-14

## Design conclusion

Targeting is a two-resolution perception system:

1. A high-resolution target solution supplies the exact point to aim at.
2. A per-client voxel overlay supplies short-lived hostile heat for pursuit,
   continuity, and movement.
3. The existing persistent lattice remains historical map knowledge.

The voxel lattice is the tactical spatial substrate, but a 64- or 256-unit
cell is never used as a crosshair target. Current-frame actionable geometry is
the only source of fire permission. Cooling heat may guide movement around a
corner for five ticks; it may not authorize a shot through that corner.

Claude, Agy, Grok, and three focused internal reviews independently found the
same core defects: origin-to-origin aim geometry, world-space absolute target
velocity beside a local target vector, a C/Python actionability mismatch, and
bot-local target vectors being deposited directly into world lattice cells.
The reviews differed mainly on whether to expand the wire immediately. The
prototype uses the lower-risk, shape-compatible path.

## Implemented target contract

The nine-float entity row keeps its size but changes semantics:

```text
rel_pos[3]  = shooter eye -> first eye-and-muzzle-clear damageable body point
              in full client->v_angle forward / Quake-right / up coordinates
vel[3]      = target velocity - shooter velocity in the same basis
health      = target health
is_enemy    = shared C hostility predicate
visible     = signed exact exposure, clear probes / six
              positive: fire-actionable; negative: shooter-protected, track only
```

The probe order is torso/chest, head, hips, legs, left torso, right torso. Each
probe must pass `MASK_SHOT` traces from both the shooter eye and the common
weapon muzzle. Observation packing plus the 14/16-degree server fire shield
consume the same selected point. Dead, target-protected, non-damageable,
spectator, friendly, and occluded clients are omitted instead of leaking health
in zero-position rows. Shooter spawn protection retains a negative exposure:
aim and thermal tracking use its magnitude, while both fire gates require a
positive value. One clear probe is valid and is no longer discarded by the
former `> 0.5` compatibility test.

Distance remains exactly derivable from `norm(rel_pos)`. Weapon-specific
intercept is deliberately deferred until the hitscan geometry is proven.

Policy input stays 219 floats with `Q2_EXT_OBS=1`; the packet size and model
tensor shapes do not change. The observation magic is now `QMLP`, the public
client conduit requires version 4, teacher envelopes require version 2, and rollout telemetry
requires `ppo-telemetry-v8`. Old game, client, teacher, or worker endpoints are
rejected instead of silently parsing the new semantics.

## Implemented thermal overlay

Each `Q2NetworkClientEnv` already owns its own `VoxelSpatialReward`, which is
the isolation boundary. The new overlay lives inside that instance:

- exact target points are transformed from the full Quake local view basis
  back into world coordinates before voxelization;
- 64-unit combat voxels retain an exact subvoxel point in the track record;
- current heat is weighted by exact exposure;
- the target's world velocity is reconstructed from local relative velocity
  plus shooter velocity;
- an occluded track extrapolates and cools deterministically for at most five
  server ticks;
- map reset, self death, kill evidence, tick rollback, or expiry clears heat;
- live/cooling heat owns the explicitly named immediate-engagement readout in
  lattice slots 5--8; persistent engagement knowledge resumes at expiry;
- target identity combines edict slot with a connection/life epoch, so a
  respawn or slot reuse cannot inherit another life's heat;
- transient tracks are not written to lattice checkpoints, Rust snapshots, or
  observed heat exports.

Persistent enemy deposits are also exposure-weighted. The previous code gave a
one-probe corner peek the same heat as a fully exposed target.

The causal order is fixed:

```text
validate obs_t -> cool tracks to t -> deposit evidence visible at t
-> freeze state vector s_t -> choose action a_t
-> receive obs_t+1 -> admit the authoritative result of a_t
```

No future observation changes an already-created policy vector.

## Locomotion/posture corrections

The audit found two trainer-side causes unrelated to missing target data:

- The lattice direction loss explicitly taught negative-forward movement when
  a visible target was behind the bot. It now withholds movement supervision
  until the look anchor turns the target into the forward hemisphere.
- The no-target posture anchor now applies only while actual horizontal speed
  is at least 96 units/s. It teaches the recurrent pitch head to return to
  level, while visible targets retain unrestricted vertical aim.

Backward movement cost is raised from 0.010 to 0.020. The invalid inherited
segment reached 70.5% backward commands and 100% down-look near +86 degrees;
the step-4,063,488 checkpoint is therefore not a resume candidate.

## Action provenance correction

Continuous actions were formerly sampled from an unbounded Normal, clipped by
the client, and stored in PPO as the unclipped value. The prototype uses the
exact censored-Normal mass/density at movement/look limits and proactively
restricts pitch to the remaining `[-89, 89]` view range. The aim teacher uses
that same executable pitch command. PPO replay scores the censored action with
stable tail probabilities; ordinary Normal entropy is omitted because it is
not the entropy of a censored distribution. Authoritative admission compares
echoed yaw, pitch, movement, jump, fire, hook, and weapon. Hook/weapon remain
ordinary reliable Quake commands, with their exact policy request carried in
the otherwise-unused protocol-34 `usercmd.impulse` byte for tick attribution.
Five otherwise-unused button bits join that byte to carry a modulo-192 action
generation; the server strips the private bits before gameplay. A same-server-frame
echo is admitted only when that generation matches, while multiple
`ClientThink` calls record the latest intended angle relative to the current
decision generation's initial view within that frame, so a previous decision
in the same server frame and duplicate held usercmds can neither contaminate,
multiply, nor erase the look decision. If live death/respawn timing changes pitch
after inference and Quake clips the causal command at its hard 89-degree view
bound, the whole four-client round becomes a nontrainable resynchronization
boundary; PPO never stores the requested action as though it executed. Yaw
echoes use wrapped angular deltas.

## Training migration

Use only the immutable policy checkpoint at step 4,055,296, not the rejected
4,063,488 descendant. The observation shape is unchanged, so policy weights
strict-load. Start a new lineage with `--reset_optimizer 1 --reset_lattice 1`:
six entity features changed meaning, old Adam moments are not trustworthy, and
the old dynamic lattice contains deposits made with the incorrect local/world
transform. Attested map sidecar priors rebuild the persistent lattice;
transient target heat always starts empty.

Keep the 512-sample recurrent minibatch, two epochs, and conservative 0.02 aim
anchor. A new run must use its own checkpoint and TensorBoard directory.

## Prototype gates

- C clean build and all Python tests pass in the WSL PyTorch environment.
- Local/world transform round trip is within 1e-3 units, including pitched
  views and Quake's right-vector sign.
- A 1/6 exposure target remains a valid aim/fire candidate.
- Thermal heat is in the correct world voxel, cools monotonically, and expires.
- PPO sampled/replayed log probability matches at static and global pitch
  bounds.
- Look, hook, or weapon echoes that disagree with the dispatched action are
  rejected.
- Live transport has zero failed rounds, stale admissions, and echo timeouts.
- Backward-command and down-look rates fall from the rejected baseline.
- The full ladder becomes nonzero in order: actionable exposure, alignment,
  fire permission, executed fire, hit, repeated hit, kill.
- Promotion requires real damage and kills on both a generated map and a stock
  map; lower loss or higher raw fire rate is not sufficient.

## Deferred milestone

After the shape-compatible prototype passes, add point-kind/exposure-mask
telemetry, target-specific hit attribution, and weapon-specific projectile
intercept. If Rust ownership is warranted, implement a separate
non-serializing `ThermalTrackIndex`; do not add immortal live-target values to
the persistent `LatticeIndex` channels.
