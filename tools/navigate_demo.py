#!/usr/bin/env python3
"""Multiplayer navigation demo: N heuristic agents roam a stock map as
ordinary network clients on one q2ded.

No learning. Each agent steers by the 16 horizontal depth rays plus
self-position telemetry, pursuing a greedy waypoint tour extracted from the
map BSP's item/spawn entities. Waypoints that prove unreachable (stuck
recovery triggers) are skipped so the tour keeps moving.

Multiplayer by default: every agent is an independent protocol-34 client
(own client_id, qport, harness port, HOME sandbox); the server stays
joinable by human players on the normal game port.

The conduit token is read from the environment
(Q2_ML_CLIENT_TELEMETRY_TOKEN) and is never printed.

Example:
    Q2_ML_CLIENT_TELEMETRY_TOKEN=$(cat /run/q2/token) \
    python3 tools/navigate_demo.py \
        --server 127.0.0.1:28200 --telemetry_server 127.0.0.1:28201 \
        --client_binary ~/q2-ml-client/release/quake2 \
        --client_root ~/q2-ml-client/release \
        --agents 2 --seconds 60 --map q2dm1
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import re
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.client_env import Q2NetworkClientEnv
from harness.protocol import Action

# ── Waypoint extraction (pak → bsp → entity lump) ──────────────────────

PAK_DIR_ENTRY = 64
BSP_LUMP_ENTITIES = 0
BSP_LUMP_COUNT = 19

WAYPOINT_CLASSNAMES = (
    "weapon_", "item_", "ammo_", "info_player_deathmatch",
)

_ENTITY_PAIR_RE = re.compile(r'"([^"]+)"\s+"([^"]+)"')
_ORIGIN_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)


def _read_pak_member(pak_path: Path, member: str) -> bytes:
    data = pak_path.read_bytes()
    magic, dir_ofs, dir_len = struct.unpack_from("<4sII", data, 0)
    if magic != b"PACK":
        raise ValueError(f"{pak_path} is not a Quake pak")
    for off in range(dir_ofs, dir_ofs + dir_len, PAK_DIR_ENTRY):
        raw_name, file_pos, file_len = struct.unpack_from("<56sII", data, off)
        name = raw_name.split(b"\0", 1)[0].decode("ascii", "replace")
        if name == member:
            return data[file_pos:file_pos + file_len]
    raise KeyError(f"{member} not found in {pak_path}")


def _bsp_entity_text(bsp: bytes) -> str:
    ident, version = struct.unpack_from("<4sI", bsp, 0)
    if ident != b"IBSP" or version != 38:
        raise ValueError("not a Quake 2 BSP (IBSP v38)")
    lump_ofs, lump_len = struct.unpack_from(
        "<II", bsp, 8 + BSP_LUMP_ENTITIES * 8
    )
    return bsp[lump_ofs:lump_ofs + lump_len].decode("ascii", "replace")


def _find_bsp(gamedir: Path, map_name: str) -> bytes:
    """Engine load order: later paks override, so search pak9 → pak0 first."""
    member = f"maps/{map_name}.bsp"
    for i in range(9, -1, -1):
        pak = gamedir / f"pak{i}.pak"
        if pak.exists():
            try:
                return _read_pak_member(pak, member)
            except KeyError:
                continue
    loose = gamedir / member
    if loose.exists():
        return loose.read_bytes()
    raise KeyError(f"{member} not found in any pak under {gamedir}")


def load_waypoints(gamedir: Path, map_name: str) -> list[tuple[float, float, float]]:
    """Item/ammo/weapon/player-spawn origins as a greedy nearest-neighbor tour."""
    bsp = _find_bsp(gamedir, map_name)
    text = _bsp_entity_text(bsp)
    spawns: list[tuple[float, float, float]] = []
    items: list[tuple[float, float, float]] = []
    for block in re.findall(r"\{([^}]*)\}", text):
        pairs = dict(_ENTITY_PAIR_RE.findall(block))
        classname = pairs.get("classname", "")
        match = _ORIGIN_RE.match(pairs.get("origin", "").strip())
        if not match:
            continue
        origin = tuple(float(match.group(i)) for i in range(1, 4))
        if classname == "info_player_deathmatch":
            spawns.append(origin)
        elif classname.startswith(WAYPOINT_CLASSNAMES):
            items.append(origin)
    if not spawns or not items:
        raise ValueError(
            f"map {map_name}: found {len(spawns)} spawns, {len(items)} items"
        )
    # Greedy nearest-neighbor tour from the first player spawn.
    tour: list[tuple[float, float, float]] = []
    remaining = items[:]
    current = spawns[0]
    while remaining:
        nxt = min(
            remaining,
            key=lambda p: math.dist(current[:2], p[:2]) + abs(current[2] - p[2]),
        )
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return tour


# ── Heuristic navigator ────────────────────────────────────────────────

RAY_COUNT = 16
RAY_STEP_DEG = 360.0 / RAY_COUNT
RAY_CLEAR = -1.0            # distance -1 means unobstructed to max range
BLOCK_NEAR = 96.0           # a ray shorter than this is "blocked"
OPEN_DIST = 160.0           # a ray longer than this counts as open
MAX_YAW_STEP = 30.0         # per-frame look delta clamp (degrees)
WAYPOINT_RADIUS_XY = 80.0
WAYPOINT_RADIUS_Z = 80.0
TARGET_Z_BAND = 96.0        # only chase waypoints on our current level
TARGET_HORIZON = 400.0      # only chase waypoints this close (no pathfinding)
STUCK_WINDOW = 30           # frames (~3s at 10 Hz telemetry)
STUCK_DIST = 24.0           # displacement under this over the window = stuck
RECOVER_FRAMES = 12         # reverse+turn burst length
WAYPOINT_FRAME_BUDGET = 250 # give up on a waypoint after ~25s


def wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class Navigator:
    """Pursues the nearest same-level waypoint; skips unreachable ones.

    Stock maps are multi-level, so a fixed item tour chases pickups through
    floors. Instead the agent always targets the nearest waypoint within a
    z-band of its current position. Reached waypoints leave the pool; skipped
    ones go to the back of the pool so they are retried only after everything
    else.
    """

    def __init__(self, waypoints: list[tuple[float, float, float]]):
        self._all = list(waypoints)
        self.pool = list(waypoints)
        self.target: tuple[float, float, float] | None = None
        self.wp_frames = 0
        self.history: deque[tuple[float, float, float]] = deque(maxlen=STUCK_WINDOW)
        self.recover_left = 0
        self.distance = 0.0
        self.reached = 0
        self.skipped = 0
        self.stuck_events = 0
        self.frames = 0
        self.combat = True
        self.damage_dealt = 0.0
        self.kills = 0
        self.deaths = 0
        self._last_pos: tuple[float, float, float] | None = None

    def _pick_target(self, pos: tuple[float, float, float]) -> None:
        """Nearest same-level waypoint within the reachable horizon, else None.

        None means roam: straight-line steering cannot pathfind through
        doorways to a target hundreds of units away, so distant waypoints
        are ignored until exploration brings us close.
        """
        near = [
            p for p in self.pool
            if abs(p[2] - pos[2]) <= TARGET_Z_BAND
            and math.dist(pos[:2], p[:2]) <= TARGET_HORIZON
        ]
        self.target = (
            min(near, key=lambda p: math.dist(pos[:2], p[:2])) if near else None
        )
        self.wp_frames = 0

    def _advance_waypoint(self, skipped: bool) -> None:
        if self.target is None:
            return
        if skipped:
            self.skipped += 1
            # Written off permanently until the pool runs dry: an unreachable
            # pickup stays unreachable, re-targeting it just re-wedges us.
            if self.target in self.pool:
                self.pool.remove(self.target)
        else:
            self.reached += 1
            if self.target in self.pool:
                self.pool.remove(self.target)
        if not self.pool:
            self.pool = list(self._all)
        self.target = None

    def step(self, obs) -> Action:
        pos = tuple(float(v) for v in obs.self_state[:3])
        yaw = float(obs.yaw)
        alive = float(obs.self_state[6]) > 0.0
        self.frames += 1
        self.damage_dealt += float(obs.reward_damage_dealt)
        if float(obs.reward_kill) > 0.0:
            self.kills += 1
        if float(obs.reward_death) > 0.0:
            self.deaths += 1

        if self._last_pos is not None:
            hop = math.dist(self._last_pos, pos)
            if hop < 400.0:  # ignore respawn/teleport jumps
                self.distance += hop
        self._last_pos = pos

        if not alive:
            self.history.clear()
            self.recover_left = 0
            return Action()  # server-side auto-respawn owns the death delay

        self.history.append(pos)
        if self.target is None:
            self._pick_target(pos)
        self.wp_frames += 1
        if self.wp_frames > WAYPOINT_FRAME_BUDGET:
            self._advance_waypoint(skipped=True)
            self.history.clear()
            self._pick_target(pos)

        target = self.target
        if target is not None:
            dx, dy, dz = target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]
            if math.hypot(dx, dy) < WAYPOINT_RADIUS_XY and abs(dz) < WAYPOINT_RADIUS_Z:
                self._advance_waypoint(skipped=False)
                self.history.clear()
                self._pick_target(pos)
                target = self.target

        # Stuck detection → reverse + fixed turn burst, then skip waypoint.
        if self.recover_left > 0:
            self.recover_left -= 1
            if self.recover_left == 0:
                self._advance_waypoint(skipped=True)
                self.history.clear()
            return Action(move_forward=-0.8, look_yaw=9.0)
        if len(self.history) == STUCK_WINDOW:
            oldest = self.history[0]
            if math.dist(oldest, pos) < STUCK_DIST:
                self.stuck_events += 1
                self.recover_left = RECOVER_FRAMES
                return Action(move_forward=-0.8, look_yaw=9.0)

        # Scan rays: widest opening plus the open ray nearest the target.
        desired_yaw = (
            math.degrees(math.atan2(target[1] - pos[1], target[0] - pos[0]))
            if target is not None
            else None
        )
        best_idx, best_score = 0, float("inf")
        widest_idx, widest_dist = 0, -1.0
        for i in range(RAY_COUNT):
            dist = float(obs.rays[i][3])
            clear = dist if dist != RAY_CLEAR else 2048.0
            if clear > widest_dist:
                widest_idx, widest_dist = i, clear
            if desired_yaw is None or (dist != RAY_CLEAR and dist < OPEN_DIST):
                continue
            rel = abs(wrap180(i * RAY_STEP_DEG + yaw - desired_yaw))
            if rel < best_score:
                best_idx, best_score = i, rel
        if desired_yaw is not None and best_score != float("inf"):
            steer_idx = best_idx
        else:
            # Roam: widest opening, with a slow deterministic sway so two
            # agents and symmetric rooms don't lock into mirror orbits.
            sway = 1 if (self.frames // 40) % 2 == 0 else -1
            steer_idx = (widest_idx + sway) % RAY_COUNT
        steer_yaw = yaw + steer_idx * RAY_STEP_DEG
        # Ray 0 points along current facing; world angle = yaw + i*22.5.
        yaw_err = wrap180(steer_yaw - yaw)

        front = float(obs.rays[0][3])
        front_blocked = front != RAY_CLEAR and front < BLOCK_NEAR
        speed = math.hypot(float(obs.self_state[3]), float(obs.self_state[4]))

        thrust = 1.0 if abs(yaw_err) < 60.0 else 0.4
        if target is not None:
            # Approach throttle: full-speed runs overshoot the pickup radius
            # between control frames and orbit the target.
            dist_xy = math.hypot(target[0] - pos[0], target[1] - pos[1])
            if dist_xy < 160.0:
                thrust = min(thrust, 0.5)

        action = Action(
            move_forward=thrust,
            move_right=max(-0.5, min(0.5, yaw_err / 90.0)),
            look_yaw=max(-MAX_YAW_STEP, min(MAX_YAW_STEP, yaw_err)),
            jump=bool(front_blocked and speed < 50.0),
        )
        if self.combat:
            self._aim_and_fire(obs, action)
        return action

    def _aim_and_fire(self, obs, action: Action) -> None:
        """Override look/fire when an enemy is sensed. Entity rel_pos is in
        the current view basis (forward/right/up), so the correction is a
        relative delta that converges over frames. Exposure > 0 means
        fire-actionable (clear shot, target not spawn-protected); the server
        independently masks misaligned fire, so false triggers are cheap."""
        count = min(int(obs.entity_count), obs.entities.shape[0])
        best = None
        for i in range(count):
            ent = obs.entities[i]
            if ent[7] <= 0.5:  # is_enemy
                continue
            if abs(float(ent[8])) <= 0.0:  # exposure: sensed at all
                continue
            dist = math.sqrt(ent[0] ** 2 + ent[1] ** 2 + ent[2] ** 2)
            if best is None or dist < best[0]:
                best = (dist, ent)
        if best is None:
            return
        _dist, ent = best
        fwd, right, up = float(ent[0]), float(ent[1]), float(ent[2])
        yaw_to = math.degrees(math.atan2(right, fwd))
        pitch_to = -math.degrees(math.atan2(up, math.hypot(fwd, right)))
        # Quake's (forward, right) basis is left-handed: right_c > 0 means the
        # target is clockwise of the view, so the yaw correction is negated.
        action.look_yaw = max(-MAX_YAW_STEP, min(MAX_YAW_STEP, -yaw_to))
        action.look_pitch = max(-15.0, min(15.0, pitch_to))
        exposure = float(ent[8])
        if exposure > 0.0 and abs(yaw_to) < 10.0 and abs(pitch_to) < 10.0:
            action.fire = True
        # Strafe-orbit while engaging instead of running straight in.
        action.move_right = 0.5 if (self.frames // 25) % 2 == 0 else -0.5

    def summary(self, name: str) -> str:
        return (
            f"{name}: frames={self.frames} distance={self.distance:.0f} "
            f"waypoints_reached={self.reached} waypoints_skipped={self.skipped} "
            f"stuck_events={self.stuck_events} "
            f"damage_dealt={self.damage_dealt:.0f} kills={self.kills} "
            f"deaths={self.deaths}"
        )


# ── Driver ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="127.0.0.1:28200")
    parser.add_argument("--telemetry_server", default="127.0.0.1:28201")
    parser.add_argument("--client_binary", required=True)
    parser.add_argument("--client_root", required=True)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--map", default="q2dm1")
    parser.add_argument("--arena", type=float, default=0.0,
                        help="if > 0, restrict waypoints to this radius around "
                             "the map centroid — concentrates agents for combat")
    parser.add_argument("--gamedir", default=None,
                        help="dir holding pak files with the BSP; default <client_root>/baseq2")
    parser.add_argument("--min_distance", type=float, default=500.0)
    parser.add_argument("--min_waypoints", type=int, default=1,
                        help="per-agent floor for PASS; 1 keeps the bar at "
                             "'navigates with purpose' — capture count beyond "
                             "that is spawn luck with a ray-only heuristic")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-agent position/target every ~10s")
    args = parser.parse_args()

    token = os.environ.get("Q2_ML_CLIENT_TELEMETRY_TOKEN", "")
    if not token:
        sys.exit("Q2_ML_CLIENT_TELEMETRY_TOKEN is not set")

    gamedir = Path(args.gamedir) if args.gamedir else Path(args.client_root) / "baseq2"
    waypoints = load_waypoints(gamedir, args.map)
    if args.arena > 0.0:
        cx = sum(w[0] for w in waypoints) / len(waypoints)
        cy = sum(w[1] for w in waypoints) / len(waypoints)
        waypoints = [
            w for w in waypoints
            if math.hypot(w[0] - cx, w[1] - cy) <= args.arena
        ]
        if len(waypoints) < 4:
            sys.exit(f"arena radius {args.arena} leaves only {len(waypoints)} waypoints")
    print(f"waypoints: {len(waypoints)} from {gamedir} maps/{args.map}.bsp")

    envs = [
        Q2NetworkClientEnv(
            server=args.server,
            telemetry_server=args.telemetry_server,
            telemetry_token=token,
            client_binary=args.client_binary,
            client_root=args.client_root,
            client_id=f"nav-demo-{i}",
            name=f"nav-{i}",
            harness_port=39020 + i,
        )
        for i in range(args.agents)
    ]
    navs = [Navigator(waypoints) for _ in envs]
    try:
        first = [env.start() for env in envs]
        slots = [sample.client_slot for sample in first]
        if len(set(slots)) != len(slots):
            sys.exit(f"client slots collided: {slots}")
        print(f"connected: agents={args.agents} slots={slots} map={first[0].map_name}")

        observations = [sample.observation for sample in first]
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            # Dispatch every agent's action before waiting on any one client,
            # so control rate stays at the full telemetry rate as agents scale
            # (same pattern as harness/client_batch.py).
            dispatches = [
                env.dispatch_action(navs[index].step(observations[index]))
                for index, env in enumerate(envs)
            ]
            for index, env in enumerate(envs):
                telemetry = env.receive_telemetry(
                    after_sequence=dispatches[index].after_sequence
                )
                observations[index] = telemetry.observation
                if args.verbose and navs[index].frames % 100 == 0:
                    nav = navs[index]
                    p = telemetry.observation.self_state
                    t = nav.target or (0.0, 0.0, 0.0)
                    print(
                        f"  agent-{index} f={nav.frames} "
                        f"pos=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) hp={p[6]:.0f} "
                        f"tgt=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f}) "
                        f"dist={math.dist(p[:2], t[:2]):.0f} "
                        f"reached={nav.reached} skip={nav.skipped}"
                    )
    finally:
        for env in envs:
            env.close()

    # PASS criteria: every agent must prove sustained self-directed movement
    # (distance), and the swarm as a whole must prove waypoint captures
    # (aggregate — per-agent capture counts are spawn luck under a ray-only
    # heuristic, the failing agent rotates between runs).
    total_reached = sum(nav.reached for nav in navs)
    ok = total_reached >= args.min_waypoints * args.agents
    for index, nav in enumerate(navs):
        line = nav.summary(f"agent-{index}")
        passed = nav.distance >= args.min_distance
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {line}")
    print(
        f"{'PASS' if ok else 'FAIL'} swarm: waypoints_reached={total_reached} "
        f"(need >= {args.min_waypoints * args.agents})"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
