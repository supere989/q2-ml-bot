# Network-Native Client Harness Prototype

Status: public cutover candidate with batched PPO integration.

## Conclusion

The training policy can now occupy a real Quake II player connection without
giving up the authoritative observation and reward state used by the existing
219-feature policy. Movement, view, jump, fire, grapple, and weapon selection
travel through ordinary protocol-34 client commands. Privileged state travels
through a separate authenticated, per-client UDP conduit owned by `game.so`.

The first end-to-end proof used a headless Yamagi client against an isolated
Lithium q2ded. It completed the normal challenge, connect, configstring,
baseline, and begin lifecycle before the server registered telemetry. A second
proof connected two independent clients to the same server and required both
routes to echo different strafe commands and produce physical displacement:

```text
PASS independent_routes=2 slots=[0, 1] frames=[102, 107] displacements=[350.9, 476.6] steps_each=20
```

The single-client proof now fails unless ordinary usercmds move the player and
the server echoes the requested movement and fire action. With `Q2_EXT_OBS=1`,
the client-backed environment reconstructed the current lattice-extended input
without changing its shape:

```text
client_id=... slot=0 vector_dim=219 displacement=694.5
```

A 70-step soak crossed periodic conduit re-registration while preserving a
monotonic sequence, uninterrupted action echo, and continued movement.

## Data path

```text
policy / Vector Lattice
        |  local action packet
        v
headless Yamagi client -- normal usercmd/netchan --> q2ded
        ^                                      |
        | private telemetry for this client_id |
        +---- game.so authenticated conduit <--+
```

Each harness instance generates an opaque `ml_client_id`. Yamagi publishes it
in normal userinfo and includes it in conduit registration. `game.so` accepts a
route only when all of these match:

- the shared, non-serverinfo telemetry token;
- an active non-bot client whose `ml_client_id` matches exactly;
- the registration datagram's source IPv4 address and UDP port and the normal
  connection's engine-supplied `ip` userinfo value.

The route is indexed by the resolved client slot and unicasts only that
player's `ml_obs_t`. Every envelope repeats `client_id`, client slot, server
frame, map name, and a monotonic route sequence. Python rejects malformed
envelopes, identity mismatches, observation/slot mismatches, and stale
sequences. Periodic re-registration refreshes NAT state without resetting the
sequence.

## Components

### Lithium / 3ZB2 game module

Repository: `q2-lithium-3zb2`, branch `ml-wip-20260611`.

- `ml_client_wire.h`: fixed-layout registration, acknowledgement, and
  telemetry envelopes with compile-time size assertions.
- `ml_client_telemetry.c/.h`: authenticated route registry, command echo,
  authoritative observation emission, and shutdown cleanup.
- `g_combat.c`, `g_items.c`, `p_client.c`: collect the same damage, kill,
  death, and pickup channels for registered network clients as for in-process
  ML bots.
- `g_main.c`: emits telemetry after simulation and playerstate finalization.

Server cvars are default-off:

```text
ml_client_telemetry 1
ml_client_telemetry_port 28201
ml_client_telemetry_token <secret>
```

Use a unique telemetry port for each q2ded process on a rollout host. Many
clients on that server share the one port and are separated by `client_id`.

### Yamagi client

Repository: `supere989/yquake2`, branch `feature/ml-client-harness`; local
checkout `/home/raymondj/q2-ml-client`. It is based on upstream Yamagi commit
`c4b5fe33a8a3c250cb35227209d4a2d469f8fd97`.

- `cl_ml_harness.c`: registration, private telemetry forwarding, latest-action
  cache, and action-to-usercmd/reliable-command projection.
- `ml_headless 1`: skips renderer, audio, menu, and physical input startup but
  retains the complete network, download, snapshot, prediction, and client
  command lifecycle.
- Every process needs a distinct `qport` when several clients share one host.

The server secret is present in the client process cvar for this prototype.
Production WAN deployment should replace the static token with a challenge and
HMAC or place the conduit exclusively on the trusted LAN/Tailscale fabric.

### Python harness

- `harness/client_protocol.py`: validates and decodes the client envelope.
- `harness/client_env.py`: launches a real client, exchanges policy actions,
  computes authoritative reward, and exposes `reset()` / `step_vector()` with
  the existing Vector Lattice tail.
- `tools/network_client_smoke.py`: single-client raw or 219-vector proof; it
  asserts physical displacement and authoritative movement/fire echo.
- `tools/network_client_multi_smoke.py`: multi-client route isolation proof;
  it asserts distinct slots, IDs, opposite per-route action echoes, and
  displacement for both players.

## What changes at cutover

The existing in-process bridge is lockstep and can stop q2ded while waiting for
the learner. A regular network client cannot safely do that: q2ded must keep
servicing every connection. The network-native backend is therefore real-time
and latest-action-wins. That is correct for live behavior but changes PPO's
collection timing.

`harness.client_batch` now owns N independent clients, dispatches a complete
same-policy round before collecting, and admits only usercmd echoes matching
the requested movement/buttons. Policy versions are monotonic, stale or
mismatched experience is rejected, and a map change produces an explicitly
nontrainable synchronization boundary. `train/ppo.py` uses this adapter when
`Q2_NETWORK_CLIENTS` is nonzero and records its admission metrics in
TensorBoard.

The public cutover layout keeps four full clients on WSL, two open human slots,
normal game traffic on UDP 28000, and the per-client conduit on UDP 28049 over
Tailscale. The conduit port must be firewalled to the rollout host. The public
q2ded remains at `timescale 1`; its map lifecycle alternates shuffled stock
maps with generated `mllive_*` maps supplied by the WSL map farm.

The remaining release gate is operational rather than architectural: pass the
cross-host four-client canary, archive the prior trainer at a checkpoint,
install the coordinated secret/firewall/module, and verify advancing accepted
transitions plus TensorBoard metrics before declaring the new lane primary.
