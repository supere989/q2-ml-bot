# Multiplayer navigation demo — 2026-07-23

Status: working local demo. N heuristic agents join one lithium q2ded as
ordinary protocol-34 network clients and navigate q2dm1 without any
learning. Server stays human-joinable; multiplayer is the default, not a
special mode.

## What this is

`tools/navigate_demo.py` drives N independent headless Yamagi clients
(`harness/client_env.py`) against a local or remote q2ded with the
`ml_client_telemetry` conduit enabled. Each agent:

- steers with the 16 horizontal depth rays (open ray nearest the target
  heading; widest ray + deterministic sway when roaming),
- pursues waypoints extracted live from the map BSP inside the pak
  (`maps/<map>.bsp` entity lump; item/ammo/weapon/player-spawn origins),
- only chases waypoints on its own level (|dz| <= 96) and within a 400-unit
  horizon — a ray-only heuristic cannot pathfind through doorways, so
  distant targets are ignored until exploration brings them close,
- throttles to half speed inside 160 units of the target to avoid orbiting
  the pickup radius,
- recovers from stuck states (reverse + fixed turn burst), writes off
  unreachable waypoints, and ignores respawn teleports in its distance
  accounting.

Actions are dispatched to all agents before waiting on any client's next
telemetry frame (same pattern as `harness/client_batch.py`), so the control
rate stays at the full 10 Hz conduit rate as agent count grows.

## Reproducing (this machine)

Runtime root `~/q2-ml-client/release` was assembled as:

- fresh binaries redeployed (the deployed tree lagged three commits):
  `build/release/quake2` → `release/quake2`,
  `q2-lithium-3zb2/lithium/gamex86_64.so` → `release/lithium/game.so`;
- `release/baseq2/pak{0,1}.pak` and `release/lithium/pak{0,1}.pak` symlinked
  from `~/.local/share/YamagiQ2/`;
- `lithium.cfg`, `maps.lst`, `admin.lst` copied from
  `q2-lithium-3zb2/lithium/`.

Server (token lives in a file, never on the Python command line):

```sh
cd ~/q2-ml-client/release
stdbuf -oL -eL ./q2ded +set game lithium +set dedicated 1 +set port 28200 \
  +set deathmatch 1 +set cheats 1 +set maxclients 8 \
  +set timelimit 0 +set fraglimit 0 +set autospawn 0 \
  +set ml_client_telemetry 1 +set ml_client_telemetry_port 28201 \
  +set ml_client_telemetry_token "$TOKEN" +map q2dm1
```

Demo (reads `Q2_ML_CLIENT_TELEMETRY_TOKEN` from the environment):

```sh
cd ~/q2-ml-bot
Q2_ML_CLIENT_TELEMETRY_TOKEN=$TOKEN python3 tools/navigate_demo.py \
  --server 127.0.0.1:28200 --telemetry_server 127.0.0.1:28201 \
  --client_binary ~/q2-ml-client/release/quake2 \
  --client_root ~/q2-ml-client/release \
  --agents 4 --seconds 90
```

Human join: any normal Quake 2 client, `connect 127.0.0.1:28200`. Slots not
used by agents are open (maxclients 8).

## Results (local, q2dm1)

- `tools/network_client_multi_smoke.py`: PASS, 2 routes,
  displacements [364.6, 476.4] — rerun after demo development, no
  regression.
- `navigate_demo.py --agents 2 --seconds 60`: PASS, both agents,
  ~10.3k/10.6k units, 15 waypoints captured by one agent.
- `navigate_demo.py --agents 4 --seconds 90`: PASS, 4 clients on slots
  0-3, per-agent distance 18.7k-21.1k units over ~800 frames, swarm total
  10 waypoint captures, stuck recovery exercised on every agent.

PASS criteria: every agent exceeds `--min_distance` traveled (default 500),
and the swarm aggregates at least `--min_waypoints * agents` captures
(default 1 each). Aggregate capture count is used because per-agent
captures are spawn luck under a ray-only heuristic — the low agent rotates
between runs while the swarm total stays well above the bar.

## Known gaps / not done

- No real pathfinding. Agents cannot route to a known-far target through
  corridors; they explore until targets come within the horizon. This is
  the honest ceiling of ray-only steering and the strongest argument for
  the learned policy or a map-graph navigator next.
- Water/lava movement is untuned (no pitch control for swimming); stuck
  recovery eventually escapes, but clumsily.
- No combat: fire/hook/weapon actions are unused here.
- The local token is a static shared secret in client cvars — fine on
  loopback, already documented as not WAN-grade in
  `docs/NETWORK-CLIENT-HARNESS.md`.

## Live review session (added 2026-07-23, second pass)

Combat is enabled and verified: agents aim via the telemetry entity block
(view-basis rel_pos) and fire through the server fire gate. Two server-side
fixes were required before clients could see or hurt each other:

- **`use_startobserver 1` in `lithium.cfg` made every harness client a
  Lithium observer** (SOLID_NOT, invisible to `ML_TargetSolution`). Set
  `use_startobserver 0` in the server launch. This was the root cause of
  "zero entity sightings"; it did not affect movement, so it looked like a
  sensing bug for a long time. Symptom to check first if entities are
  ever empty again: `ent->lithium_flags & LITHIUM_OBSERVER`.
- `start_weapon 9` (railgun) in `release/lithium/lithium.cfg` — instant-hit
  makes heuristic aim actually land; the default blaster projectile is too
  slow.

Aim sign convention: the entity `rel_pos` basis is left-handed, so the yaw
correction is `-atan2(right, forward)`; pitch correction is
`-atan2(up, hypot(forward, right))`.

Recorded `.dm2` demos are **not reviewable** in this fork: client-side demo
playback is absent (no `demo` command / `CL_ReadDemoMessage`), and `demomap`
runs the server in attract loop, which rejects client connections ("Remote
connect in attract loop. Ignored."). Client-side review is therefore **live
spectating**, which is also the better demo:

```sh
# bots (from ~/q2-ml-bot, server on 127.0.0.1:28200):
Q2_ML_CLIENT_TELEMETRY_TOKEN=$TOKEN python3 tools/navigate_demo.py \
  --client_binary ~/q2-ml-client/release/quake2 \
  --client_root ~/q2-ml-client/release \
  --map mltrain_00005200 --arena 700 --agents 4 --seconds 7200 --min_waypoints 0

# human reviewer (graphical client, any machine that can reach the host):
cd ~/q2-ml-client/release
./quake2 +set game lithium +connect 127.0.0.1:28200
```

`--arena 700` restricts waypoints to the map-centroid cluster so agents
converge and firefights happen. The human spawns as a player
(`use_startobserver 0` applies to everyone) and can fight the bots or use
the Lithium menu (`cmd menu`) to become an observer with chasecam.
Verified 2026-07-23: 4 agents, 120s session — 3 railgun kills by one agent,
deaths on all agents (map hazards), swarm navigation PASS.

Known quirk: `+set` groups on the q2ded command line are capped by
`MAX_NUM_ARGVS` (50) — at ~16 `+set` triples the server aborts with
`argc > MAX_NUM_ARGVS`; move extra cvars into a cfg instead.

## Next phases (roadmap, not started)

1. **RL on top of this lane.** Resume PPO per
   `docs/HANDOFF-2026-07-13-NETWORK.md`: the active lineage is
   `public_network_thermal_bc_live_v2` (LR 1e-5, anchor 0.02, 398-cell
   lattice). Older anchors (step-4,063,488; the
   `public_network_thermal_target_v1` warm-start) are quality-invalid —
   never resume them. Training runs on the WSL box, not here.
2. **Valheim.** Equivalent architecture, different substrate: closed-source
   Unity game, so the q2 conduit becomes a BepInEx plugin on a dedicated
   server exposing per-client state/action over a socket (mirror the q2
   design: auth token, per-client routes keyed by player id,
   latest-action-wins). Headless clients connect as ordinary players —
   same "multiplayer by default" contract. The project's public VPS is
   named `valheim-server`; inventory what is already deployed there before
   designing the plugin. Valheim has no deterministic 10 Hz lockstep, so
   expect real-time collection only.
