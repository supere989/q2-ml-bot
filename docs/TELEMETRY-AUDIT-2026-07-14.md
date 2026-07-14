# Network-client telemetry audit — 2026-07-14

## Superseding prototype result

The follow-up review and implementation are recorded in
`TARGET-THERMAL-LATTICE-PROTOTYPE-2026-07-14.md`. The original verdict below
correctly established that targets were present, but the richer audit found
four quality blockers: origin-to-origin target geometry, absolute world target
velocity, incorrect local-to-world lattice deposits, and incomplete
continuous-action provenance. The shape-compatible prototype corrects them.
The `engagement_anchor_v3` step-4,063,488 checkpoint was stopped and rejected
after reaching 100% down-look and 70.5% backward commands. A subsequent
target/thermal canary loaded the immutable step-4,055,296 policy with a new
optimizer and reset lattice. It received actionable target data on up to 54.5%
of accepted transitions and maintained 97.5--99.2% authoritative action
acceptance with zero failed rounds or echo timeouts, but still returned to
100% down-look and about 66% backward commands on a generated map. Claude,
Grok, and the local code audit attributed this to an inherited
action/recurrent prior plus weak corrective leverage. Agy independently raised
a possible pitch double-rotation, but the Quake-basis algebra and executable
tests reject it: a world-horizontal target observed at +10 degrees pitch has
local `(cos(p), 0, sin(p))` coordinates and correctly produces a -10-degree
command, the full local/world basis round-trips within 1e-3, and the live fresh
run subsequently reached 8.1/4.2-degree conditional yaw/pitch error plus an
aligned hit. The conduit and thermal/fire role separation are not the fault.

That warm-start is also rejected and archived. The subsequent fully fresh
`public_network_thermal_fresh_v1` run proved that posture and movement could
recover, but its step-49,152 deterministic holdout remained at 39.6-degree yaw
and 16.9-degree pitch error. The active `public_network_thermal_bc_live_v2`
canary is a distilled-backbone clone of that step with its matching lattice.
Its holdout measures 2.23/1.69-degree yaw/pitch error, 30.0% post-command
alignment, 92.7% aligned-fire precision, zero hidden fire, and 0.0041 movement
drift. In the first 16,384 live generated-map transitions it produced 1,539
aligned frames, 1,117 aligned shots, 49 hit events, 15 repeat-hit events, and
two kills with 4.2% down-look and no transport errors. This is direct evidence
that the telemetry is sufficient to train target engagement. The generated
season passed; a stock-map season is still required before promotion.

## Original verdict

The 219-feature network-client observation is sufficient for the current
prototype to learn ordinary target acquisition, alignment, firing, locomotion,
damage exchange, and lattice-directed navigation. Live `engagement_anchor_v3`
packets contain multiple hostile clients, visible contact, finite aim labels,
fire-gate permission, and damage events. No missing target channel explains the
current inherited policy's backward/downward behavior.

The representation is not complete for high-skill combat. Stable target
identity remains debug-only and the 16 navigation rays are horizontal. Local
relative velocity and a best actionable damage point are now implemented;
Connection/life-epoch identity is now implemented for thermal tracks;
projectile leading and vertical clearance remain follow-up improvements.

## Audited channels

| Channel | Policy input | Reward/audit | Finding |
| --- | --- | --- | --- |
| self position/velocity, health, armor, weapon, ammo | yes | yes | populated and normalized |
| true yaw/pitch | yes | explicit posture tags | fixed in C `f49caee`; never use the one-third rendered model pitch |
| up to eight client entities | yes | count tags | live mean was about 4.8 hostile entities/frame |
| relative position, health, hostile, visible | yes | visibility and conditional aim tags | live visible-contact rate reached 54--66% on `mllive_44987431` |
| entity velocity | yes | no dedicated tag | target minus shooter velocity in the same full-view local basis as the aim point |
| geometry rays | yes | indirect | 16 horizontal distance rays; no vertical clearance rays |
| hook anchors/landings/flags | yes | detailed hook tags | sidecar-attested and active on generated maps |
| audio direction/age/alert | yes | fire/audio shaping | one global sound source, not source-identified |
| rune and inbound-damage direction | yes with `Q2_EXT_OBS=1` | extended tags | present in the active 219-vector run |
| damage dealt/taken, kills/deaths, items, hook traversal | reward, not next-state feature | explicit step and episode tags | live damage events observed; outcomes are trainable |
| per-client slot/control identity | no | debug only | correct routed client isolation, not a policy feature |
| applied action echo | no | admission and posture audit | movement/buttons checked; look response remains an audit-hardening opportunity |
| session/lattice memory | yes, 24 features | memory/prior/route tags | loaded and map-scoped |

## Live evidence and interpretation

The first true-view audit update reported mean entity/enemy count 4.793,
visible contact on 32.42% of accepted frames, nonzero fire permission/execution,
and 0.03125 damage dealt per step. On the v6 generated map, subsequent segments
reported 54--66% visible-contact frames. This proves the target conduit and
hostility/visibility classification are functioning.

The inherited 4,055,296-step policy nevertheless began with roughly 64--71%
backward commands and often drove the true view toward the +89-degree down
clamp. Python `f197fba`/`fb85aa7` therefore add signed locomotion costs, a dense
forward-only level-posture reward, a 15-degree no-target pitch penalty, and
visibility-conditional engagement metrics. The active run also enables the
recurrent geometric look/fire anchor at coefficient 0.02 with a complete
512-sample minibatch. These are new training objectives; visual behavior is not
expected to change in the first few PPO updates.

## Remaining audit gates

Before promoting a checkpoint, require a seasonal window with all of:

- backward command rate and true down-look rate decreasing from the recorded
  baseline;
- conditional yaw/pitch error and anchor MAE decreasing without loss of
  movement speed;
- fire-gate permission followed by damage and kills, not merely raw firing;
- zero failed network rounds, echo timeouts, or stale-policy admission;
- a successful stock/generated transition and a played v6 map with its lethal
  guard and lighting contract loaded.
