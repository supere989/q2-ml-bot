# Network-client telemetry audit — 2026-07-14

## Verdict

The 219-feature network-client observation is sufficient for the current
prototype to learn ordinary target acquisition, alignment, firing, locomotion,
damage exchange, and lattice-directed navigation. Live `engagement_anchor_v3`
packets contain multiple hostile clients, visible contact, finite aim labels,
fire-gate permission, and damage events. No missing target channel explains the
current inherited policy's backward/downward behavior.

The representation is not complete for high-skill combat. Stable target
identity remains debug-only, entity velocity is world-space while relative
position is view-local, and the 16 navigation rays are horizontal. Those are
follow-up improvements for target continuity, projectile leading, and vertical
clearance; they do not block the basic engagement prototype.

## Audited channels

| Channel | Policy input | Reward/audit | Finding |
| --- | --- | --- | --- |
| self position/velocity, health, armor, weapon, ammo | yes | yes | populated and normalized |
| true yaw/pitch | yes | explicit posture tags | fixed in C `f49caee`; never use the one-third rendered model pitch |
| up to eight client entities | yes | count tags | live mean was about 4.8 hostile entities/frame |
| relative position, health, hostile, visible | yes | visibility and conditional aim tags | live visible-contact rate reached 54--66% on `mllive_44987431` |
| entity velocity | yes | no dedicated tag | populated, but still world-space rather than view-local |
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

