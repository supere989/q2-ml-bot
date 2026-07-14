# Network-client batch collector

`harness.client_batch` is the synchronous collection layer for the
network-native Yamagi harness. It owns multiple ordinary client connections,
dispatches a policy action to every client before waiting for any one client,
and returns a round only after every action has a matching authoritative
`game.so` action echo.

## Integration API

The compatibility adapter has the surface used by `train/ppo.py`:

```python
from harness.client_batch import build_network_client_multi_env

server = build_network_client_multi_env(
    n_clients=4,
    server="PUBLIC_TAILSCALE_IP:28000",
    telemetry_server="PUBLIC_TAILSCALE_IP:28049",
    telemetry_token=token,
    client_binary="/opt/q2-ml-client/quake2",
    client_root="/opt/q2-ml-client/runtime",
    harness_port_base=39000,
    qport_base=49000,
    client_id_prefix="wsl-live",
    initial_policy_version=checkpoint_steps,
)

observations = server.reset_all()
server.set_policy_version(checkpoint_steps)
results = server.step_all(actions)
# Or tag a rollout update directly:
results = server.step_all(actions, policy_version=checkpoint_steps)
observation = server.reset_slot(0)
metrics = server.metrics.as_dict()
server.close()
```

The adapter exposes `n_ml`, `active_map`, and `_spatial_rewards` as expected by
the current trainer. `reset_slot()` resets lattice/episode memory around the
latest live frame; it does not restart or rotate the public q2ded process.

For code that needs provenance as a first-class result, construct
`Q2NetworkClientBatch` and call `collect_round(actions,
policy_version=version)`. It returns `BatchRound.tags`, including `round_id`,
`policy_version`, per-client `action_tick`, client ID, and client slot.

## Admission rule

For each client, the collector sends the existing 28-byte action with
`action_tick` equal to the latest routed server frame. It then rejects packets
until all of these conditions hold:

1. `action_debug.accepted` is set.
2. The engine's echoed execution tick is newer than `action_tick`.
3. The echoed execution tick is not ahead of the observation frame.
4. Forward/right movement and jump/fire buttons match the policy action.

Look is a one-shot delta, while hook and weapon selection are reliable client
commands, so those values cannot be compared to every held `usercmd_t`. Their
behavior remains a separate parity gate. A rejected or timed-out round raises
`AuthoritativeEchoError`; no mismatched transition is returned to PPO as
trainable data.

Reward deltas from packets rejected while waiting are summed into the admitted
transition, and a terminal flag seen on any of those packets is preserved.
Echo validation therefore does not silently discard combat reward or a brief
death/intermission boundary.

Accepted info dictionaries contain:

- `batch_round_id`, `policy_version`, and `action_tick`
- `authoritative_echo_tick` and `authoritative_echo_valid=True`
- `trainable_transition=True`
- stale and mismatched echo rejection counts before admission

Policy versions are monotonic within a collector. Attempting to dispatch a
lower version raises `StalePolicyVersionError` before any action is sent.
Policy version is rollout provenance enforced by Python; the current Quake
wire packet carries the action tick but does not carry policy version.

## Metrics

`metrics.as_dict()` produces stable TensorBoard-ready names under
`network_client/*`, including accepted/failed rounds, dispatched/accepted
transitions, stale-policy rejections, stale/mismatched authoritative echoes,
timeouts, authoritative-echo acceptance rate, and maximum within-round server
frame span.

## Client filesystem and process isolation

Every client gets a distinct qport, harness UDP port, client ID, and
HOME/XDG namespace. The default writable tree is
`<client_root>/.ml-clients/<client_id>/{home,data}`; `client_data_root` can
place it elsewhere. Setting both variables avoids Yamagi falling back to a
legacy shared `~/.yq2` directory. Yamagi is also launched with
`-datadir <client_root>`, so common
read-only PAKs and assets remain shared while HTTP-downloaded generated maps,
configuration, and cache writes cannot race between clients.

Production clients use `stdout=DEVNULL` so an unread pipe cannot stall a
long-running process. `debug=True` deliberately retains a pipe and includes
the last client output if the process exits during collection.
