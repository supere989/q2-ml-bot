# Network-client frame barrier methodology

This prototype moves the logical-frame clock into the Yamagi server engine.
It is isolated-only and default-off. With `sv_ml_frame_barrier=0`, the packet,
command, and frame paths remain the ordinary Yamagi paths.

The isolated mode requires `sv_ml_frame_barrier=1`, an exact
`sv_ml_frame_barrier_clients` roster equal to `maxclients`, `ml_async=0`, and
clients advertising barrier version 1/capability 1. The numeric suffix of each
`ml_client_id` must equal its engine slot. Humans, mixed versions, extra slots,
and ordinary gameplay string commands are rejected.

The bootstrap transaction has two proofs. First, every spawned client emits
its telemetry-registration datagram and then reliably announces
`ml_barrier_bootstrap_ready 1 <client_id>`. Second, the engine captures one
genuine action-free neutral `usercmd`; it does not synthesize one. Only after
the complete roster has both does the engine run the sole bootstrap frame.
That frame polls the game conduit, accepts the routes, and emits the initial
ACK and telemetry. Policy actions cannot become ready before the client has
validated accepted v8 telemetry.

Lithium's observer-first settings are not allowed to leak into this isolated
methodology. The sealed qualification config sets both `use_startobserver=0`
and `use_startchasecam=0`. Before the batch is marked started, Python derives
an action-free role certificate from each initial telemetry packet's
engine-owned `self_debug`: edict and slot identity must agree, control source
must be an ordinary human/network client, health and life epoch must be
positive, and the client/visible bits must be present while dead, bot, ML,
observer, `SOLID_NOT`, noclip, no-client, spectator, `PM_SPECTATOR`, and
`PM_FREEZE` bits are absent. The compact debug life epoch must agree with the
causal life epoch. Any mismatch fails before the first trajectory action.
`deadflag == DEAD_NO` or positive health alone is never accepted as proof of a
playable role.

Wire v8 carries private causal v2 without changing the 80-byte tail. Bit 20,
`ROLE_PLAYING`, is game-owned and remains true for an ordinary participant
through normal play, death/corpse cameras, and stock teleport settling. Bit
21, `ROLE_PUBLIC_PM_NORMAL`, is true exactly when that participant's public
protocol playerstate is `PM_NORMAL` **and** the game entity is not
`SOLID_NOT`; it implies `ROLE_PLAYING`. A PM-normal/solid anomaly therefore
fails closed instead of entering the action or settle path. Every routed
packet must carry `ROLE_PLAYING`, and every trainable transition plus every
bootstrap certificate must also carry `ROLE_PUBLIC_PM_NORMAL`. The client
fences these private facts against the exact same-frame public playerstate;
death/GIB lifecycle packets and intermission `PM_FREEZE` remain permitted only
as nontrainable boundaries. A non-normal intermission boundary is consumed
without arming the teleport-settle/view-rebase state machine; only an exact
`ROLE_PUBLIC_PM_NORMAL` causal packet may arm that same-life settling path.
Missing role bits, a v7/v1 sender, or mid-run role loss is fatal, with no
compatibility parser.

For each later logical frame the engine continues polling packets while world
time is frozen. Each slot must provide both `ml_barrier_ready 1 <full tick>`
and a staged ordinary `usercmd` whose modulo-192 identity matches that tick.
Identical duplicates are idempotent. Stale recovery candidates may be found in
the normal command triple. Conflicting duplicates, future readiness,
disconnects, roster violations, unexpected gameplay commands, and timeout are
sticky faults. A complete transaction applies one sanitized 100 ms command in
stable slot order and then exactly one `SV_RunGameFrame`.

The exact sealed fault vocabulary is
`baseline,duplicate,stale,future,brief-drop,sustained-drop,conflict,death,`
`same-life-hold,epoch-drain,drain-sigkill,load-delay,old-telemetry`.
Drain-time disconnect is a distinct `drain-sigkill` fault, not an alias for
ordinary `epoch-drain`. Its injected intermission must report
`drain_hold_ms=3000`; ordinary epoch drain must report `drain_hold_ms=0`.
The transaction that triggers intermission has already started when drain is
entered, so its one `action_commit` witness is logged immediately after
`epoch_drain_enter`. From `intermission_injected` through the drain boundary,
it must be the only commit: this rejects a forged witness before drain entry as
well as one later in the drain. Its action tick and server frame must exactly
match `intermission_injected`, its server frame must also match the enter event,
and its map epoch must match the active pre-drain map. Any missing, duplicate,
or mismatched commit fails; apply and telemetry events remain forbidden
throughout drain.
Admissible disconnect evidence must prove, in one uninterrupted drain window,
`epoch_drain_enter` < at least two strictly increasing `epoch_drain_clock`
events < the single global slot-0 `disconnect reason=liveness` < the single
global `fatal fault=disconnect`. Any map reset, startup clock, bootstrap, or
epoch change before that disconnect fails closed. A disconnect observed only
after the next map starts is explicitly a false green, not drain-time
evidence.

Qualification uses production q2ded/game/client processes and ordinary network
clients. The only qualification-only surface is startup-sealed, dedicated-only
fault control. All four `sv_ml_frame_barrier_test_*` cvars become immutable at
the first map. Qualification requires `test_mode=1`; a training process must be
a fresh process with mode 0, empty fault, and tick 0. The final artifact records
both configurations and is itself `non_admissible_for_training=true`: it is
proof used to admit a runtime, never rollout data.

Raw scenario JSON also has a `test_mode` field. That field means only that a
Python synthetic fixture executor produced the raw schema; it is separate
from the real dedicated server's sealed `sv_ml_frame_barrier_test_mode=1`.
Synthetic `test_mode=true` evidence is permanently non-admissible even when
all of its structural assertions pass.

This role seal was added after a frozen-stock death run exposed the missing
precondition: four clients inherited observer startup, telemetry called them
alive because only `deadflag` was consulted, and the client correctly stopped
when a settling snapshot revealed `PM_SPECTATOR`. That run is negative audit
evidence, not a timing failure; increasing timeouts or permitting spectator
rebases is forbidden.

Independent work is fanned out in parallel without weakening the final gate:

1. Build q2ded, game, and client independently and run the C fault/core matrix.
2. Run Python wire/parser, collector, and admission tests against wire v8,
   1248-byte telemetry, and the 100-byte ACK.
3. Cold-launch the exact hashed binaries for every full-network scenario, with
   up to four isolated loopback scenarios running concurrently.
4. Revalidate every raw file, script, digest, binary, source tree, and runtime
   semantic in a separate finalization pass.

For `jobs>1`, every probed server, telemetry, harness, and qport identity is
retained in a process-local lease until its scenario tears down all processes.
The probe sockets close only so the real endpoints can bind; a leased port
cannot be recycled into another in-flight scenario. Success and failure both
release the lease through the runner's unconditional cleanup path.

The 19-scenario full-network matrix uses a fixed 32-frame/four-client action table, never a
policy. Two cold launches with one seed must be byte-identical and a second seed
must diverge. It covers exact duplicate idempotence; stale reject/no mutation
and recovery; fatal future/conflict; command-triple-only brief-drop recovery;
sustained timeout; active and drain-time disconnect; fifth, human, and mixed
capability rejection; a real server-injected death with exactly one slot-0
death terminal followed by synchronized respawn/alive telemetry; a stock
same-life teleporter hold; and
intermission into a distinct map. The death case must continue through the
ordinary network clients and game lifecycle; a core-only reset is not
substitute evidence.

Death has terminal precedence and a one-way collector state machine. The death
terminal is a zero-reward, whole-batch nontrainable boundary, followed by zero
or more corpse boundaries, exactly one new-life command-space rebase, one or
more stock `PMF_TIME_TELEPORT` settling boundaries, and exactly one final
same-generation actionable prime. Only the following exact synchronized
transition is admissible. The pending death identity is not removed and no
reward is admitted during rebase or settling. After the first new-life packet,
rollback to the old life is fatal; a rapid re-death on the exact next life
replaces the pending death identity and requires the subsequent exact life.
The public causal life epoch advances without changing client ID, slot, route,
map, or epoch.

The same rule covers stock holds that do not change life identity. Every
same-life `ECHO_VALID=1`, `FACTS_COMPLETE=1`,
`TRANSITION_TRAINABLE=0` command holds the whole batch nontrainable. The first
later exact actionable command is discarded once as a prime and only its
successor may train. Qualification records the accepted-trajectory ordinal
explicitly because the sealed absolute fault tick is independent of how many
actions startup admitted. A deterministic nonzero yaw/pitch probe makes the
stock pitch clamp observable. The last timer-clearing command must have a
structured entry-latched witness showing live PMF cleared while the transition
remains `E=1/F=1/T=0`; its boundary must show level pitch, exact yaw, movement,
generation, and causal identity. The exact fault vocabulary is
`baseline,duplicate,stale,future,brief-drop,sustained-drop,conflict,death,same-life-hold,epoch-drain,drain-sigkill,load-delay,old-telemetry`.

Drain must advance
the engine clock while producing no ML telemetry or action application, then
perform a fresh bootstrap. A pre-roster load held longer than the action timeout
must not start that timeout. Replayed old telemetry must be rejected by the
client without regressing its accepted clock. Hook and weapon controls must
arrive through the reliable deferred path in hook-then-weapon order.

Rich collector diagnostics retain causal flags, lifecycle phases, and private
reward evidence only for admission and audit. Every policy observation
transform instead receives a new immutable allowlist projection containing
only client identity, map, server frame, and the exact five-field spatial
attestation. Both mapping levels are read-only; missing or extra attestation
keys, private causal/spatial objects, and future diagnostic fields fail closed
or remain unreachable. This projection is used for both action inference and
bootstrap value inference.

Evidence publication is intentionally acyclic. `run` publishes canonical
`q2-network-client-frame-barrier-execution-v1` evidence and raw scenario files;
its digest is independent of a runtime manifest. The runtime manifest binds
that execution digest in
`runtime_config.network_barrier_execution_evidence_sha256`. `finalize` then
revalidates the raw evidence and binds the exact manifest digest plus execution
digest into a `q2-network-client-frame-barrier-qualification-v1` envelope. No
manifest field binds the whole envelope and no compatibility fallback exists.
The one-run trainer gate recomputes this closure and also requires the recorded
bot, client, and game commits/trees to still be clean and current.
