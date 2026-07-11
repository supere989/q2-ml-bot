"""
env.py — Q2 training environments.

Q2MultiEnv: one q2ded server → N bot slots, each as a virtual env.
Q2BotEnv:   legacy single-slot wrapper (kept for testing).

All bots on a server get ML bridges (game.so behaviour), so we bind one
UDP socket per slot and collect observations from every bot simultaneously.
This turns unused-slot timeout overhead into useful training signal.
"""

import dataclasses
import socket
import select
import subprocess
import time
import os
import signal
import random
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _Env:
        metadata = {}

        def reset(self, *args, **kwargs):
            return None

    class _Gym:
        Env = _Env

    class _Box:
        def __init__(self, low, high, shape=None, dtype=None):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Box = _Box

    gym = _Gym()
    spaces = _Spaces()
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from .protocol import (
    Observation, Action, parse_obs, pack_action,
    ML_BASE_PORT, OBS_SIZE, Observation as Obs,
)
from .spatial import VoxelSpatialReward

Q2_ROOT          = Path(os.environ.get("Q2_ROOT", "/home/raymond/q2_lithium_merge"))
Q2DED            = Q2_ROOT / "q2ded"
# Lockstep timing: the engine blocks up to ML_STEP_TIMEOUT_MS per frame for
# the policy's action — it must cover a full trainer step (GPU forward +
# bookkeeping across all venvs, ~0.6s at 48 venvs), or every frame silently
# times out and coasts on the stale action (pipelined behavior in disguise).
STEP_TIMEOUT_MS  = int(os.environ.get("Q2_HARNESS_STEP_TIMEOUT_MS", "1500"))
ML_STEP_TIMEOUT_MS = int(os.environ.get("Q2_ML_STEP_TIMEOUT_MS", "1000"))
OBS_DIM          = Obs.OBS_DIM
ACTION_DIM       = 8
ML_PORT_STRIDE   = int(os.environ.get("Q2_ML_PORT_STRIDE", "32"))

_BOTLISTS = {1: "1v1sk1", 2: "ml2sk1", 3: "2v2sk1", 4: "ml4sk1"}


def discover_map_pool(
    map_name: str = "q2dm1",
    map_glob: str = "",
    map_dir: Optional[str] = None,
) -> List[str]:
    """Return map names available to q2ded, stripping .bsp suffixes."""
    if not map_glob:
        return [map_name]

    roots = [Path(map_dir)] if map_dir else [
        Q2_ROOT / "baseq2" / "maps",
        Q2_ROOT / "lithium" / "maps",
    ]
    names = set()
    patterns = [p.strip() for p in map_glob.split(",") if p.strip()]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    names.add(path.stem if path.suffix == ".bsp" else path.name)

    out = sorted(names)
    if not out:
        searched = ", ".join(str(r) for r in roots)
        raise FileNotFoundError(f"no maps matched {map_glob!r} under {searched}")
    return out


def _decode_action(raw: np.ndarray) -> Action:
    return Action(
        move_forward = float(np.clip(raw[0], -1, 1)),
        move_right   = float(np.clip(raw[1], -1, 1)),
        look_yaw     = float(np.clip(raw[2], -45, 45)),
        look_pitch   = float(np.clip(raw[3], -30, 30)),
        jump         = bool(raw[4] > 0.5),
        fire         = bool(raw[5] > 0.5),
        hook         = int(np.clip(raw[6], 0, 3)),
        weapon       = int(np.clip(raw[7], 0, 9)),
    )


class Q2MultiEnv:
    """
    One q2ded server providing training data for all N ML-enabled bot slots.

    Since game.so enables ML bridges on every autospawned bot, we bind a
    UDP socket per ML slot. Only the TOP bot slot is ML-controlled (training);
    the rest are 3ZB2 AI opponents so the bot actually has someone to fight.

    Slot assignment: bots fill from maxclients-1 downward.
    Each server contributes ONE training env (the top slot is ML).
        server 0: maxclients= 8, ML slot 7, AI slots 6,5  → UDP 27957
        server 1: maxclients=16, ML slot 15, AI slots 14,13 → UDP 27965
    """

    def __init__(
        self,
        server_id:    int = 0,
        map_name:     str = "q2dm1",
        map_pool:     Optional[List[str]] = None,
        map_seed:     int = 0,
        map_change_episodes: int = 0,
        n_bots:       int = 4,
        port_offset:  int = 0,
        maxclients:   Optional[int] = None,
        ml_slot:      Optional[int] = None,
        num_ml_bots:  int = 1,
        max_ep_steps: int = 1000,
        timedemo:     int = 1,
        timescale:    float = 1.0,
        ml_async:     Optional[bool] = None,
        timelimit:    float = 0,
        fraglimit:    int = 0,
        intermission_time: Optional[float] = None,
        intermission_maxtime: Optional[float] = None,
        harness_step_timeout_ms: Optional[int] = None,
        ml_step_timeout_ms: Optional[int] = None,
        console_pipe: bool = False,
        start_observer: bool = False,
        spectator_only: bool = False,
    ):
        self.server_id    = server_id
        self.map_name     = map_name
        self.map_pool     = list(map_pool or [map_name])
        self.active_map   = self.map_pool[0]
        self._rng         = random.Random(map_seed + server_id * 1009)
        self.map_change_episodes = max(0, int(map_change_episodes))
        self.n_bots       = n_bots        # total bots (1 ML + n_bots-1 AI)
        # Port bases are env-overridable so multiple trainers can run on one
        # box without colliding (parallel ablation runs). Each run picks a
        # disjoint (Q2_SV_PORT_BASE, Q2_ML_PORT_BASE) slab; defaults preserve
        # single-run behaviour.
        self.sv_port      = int(os.environ.get("Q2_SV_PORT_BASE", "27910")) + port_offset
        self.max_ep_steps = max_ep_steps
        self.timedemo     = int(timedemo)
        # Wall-clock compression: the engine honours `timescale` on q2ded, so
        # frames fire N× faster while game-time physics stay identical. The
        # ML bridge's blocking action wait turns this into natural lockstep —
        # the server runs as fast as the policy answers, up to 10*N Hz.
        self.timescale    = max(0.1, float(timescale))
        self.timelimit    = float(timelimit)
        self.fraglimit    = int(fraglimit)
        self.intermission_time = intermission_time
        self.intermission_maxtime = intermission_maxtime
        self.step_timeout_ms = max(
            1,
            int(harness_step_timeout_ms)
            if harness_step_timeout_ms is not None else STEP_TIMEOUT_MS,
        )
        self.ml_step_timeout_ms = max(
            0,
            int(ml_step_timeout_ms)
            if ml_step_timeout_ms is not None else ML_STEP_TIMEOUT_MS,
        )
        self.console_pipe = bool(console_pipe)
        self.start_observer = bool(start_observer)
        self.spectator_only = bool(spectator_only)

        # Each q2ded gets its own UDP port block, so all servers can use normal
        # low client slots instead of consuming high slots to avoid port clashes.
        ml_base = int(os.environ.get("Q2_ML_PORT_BASE", str(ML_BASE_PORT)))
        self.ml_port_base = ml_base + server_id * max(8, ML_PORT_STRIDE)
        self._maxclients = (
            int(maxclients) if maxclients is not None
            else max(8, int(n_bots), int(num_ml_bots))
        )
        self._num_ml     = max(1, min(int(num_ml_bots), int(n_bots), self._maxclients))
        # By default only the top bot slot is ML. Eval can lower this threshold
        # to control multiple spawned bots on maps where 3ZB2 has no route file.
        self._ml_slot    = int(ml_slot) if ml_slot is not None else self._maxclients - self._num_ml
        ml_stop          = min(self._maxclients, self._ml_slot + self._num_ml)
        self.bot_slots   = list(range(self._ml_slot, ml_stop))
        self.ml_ports    = [self.ml_port_base + slot for slot in self.bot_slots]
        self.n_ml        = len(self.bot_slots)
        # Action protocol. Lockstep (default): the engine's G_RunFrame
        # pre-pass sends every ML bot's obs before any bot blocks for its
        # action (two-phase), so multi-bot lockstep no longer deadlocks and
        # every decision spans exactly one game frame. Pipelined (ml_async,
        # opt-in via Q2_ML_ASYNC=1): the game free-runs and coasts on cached
        # actions — only sound when the trainer outpaces the game; when the
        # trainer is the bottleneck it holds stale actions for multi-second
        # game windows and the bots are effectively open-loop.
        if ml_async is None:
            ml_async = os.environ.get("Q2_ML_ASYNC", "0") == "1"
        self.ml_async    = bool(ml_async)

        self._proc:       Optional[subprocess.Popen] = None
        self._socks:      List[Optional[socket.socket]] = [None] * self.n_ml
        self._game_addrs: List[Optional[tuple]]         = [None] * self.n_ml
        self._last_obs:   List[Optional[Observation]]   = [None] * self.n_ml
        self._ep_steps:   List[int]                     = [0] * self.n_ml
        self._episodes_done = 0
        self._spatial_rewards = [VoxelSpatialReward.from_env() for _ in range(self.n_ml)]

    def _obs_vector(self, slot_idx: int, obs: Observation) -> np.ndarray:
        memory = self._spatial_rewards[slot_idx].memory_features(obs)
        return obs.to_vector(memory)

    # ── server lifecycle ─────────────────────────────────────────────

    def _choose_map(self) -> str:
        if len(self.map_pool) == 1:
            return self.map_pool[0]
        return self._rng.choice(self.map_pool)

    def _start_server(self):
        if self._proc and self._proc.poll() is None:
            return
        self.active_map = self._choose_map()
        botlist = _BOTLISTS.get(self.n_bots, "4v4sk1")
        cfg_path = Q2_ROOT / "lithium" / f"ml_server_{self.sv_port}.cfg"
        cfg_lines = [
            "set dedicated 1",
            "set deathmatch 1",
            "set cheats 1",
            f"set timelimit {self.timelimit:g}",
            f"set fraglimit {self.fraglimit}",
            "set use_mapqueue 0",
            "set mapqueue \"\"",
            "set map_random 0",
            # Runes: the mod's DYNAMIC random spawner is intentional entropy —
            # the bot must read-and-react to runes it didn't place. Keep it
            # (do NOT static-place runes here — that breaks the entropy).
            # Rune *usage* is taught in a separate controlled gym, not here.
            "set use_runes 1",
            f"set use_startobserver {1 if self.start_observer else 0}",
            "set use_startchasecam 0",
            "set autospawn 1",
            f"set botlist {botlist}",
            "set ml_enabled 1",
            f"set ml_bot_slot {self._ml_slot}",
            f"set ml_port_base {self.ml_port_base}",
            f"set ml_spectators_only {1 if self.spectator_only else 0}",
            f"set ml_step_timeout {self.ml_step_timeout_ms}",
            f"set ml_async {1 if self.ml_async else 0}",
            "set use_hook 1",
            "set hook_speed 1900",
            "set hook_pullspeed 1700",
            "set hook_pullspeed_max 2000",
            "set hook_pullscale 0.25",
            "set hook_gravity_comp 1.0",
            "set hook_min_lift 180",
            "set hook_maxtime 15.0",
            "set hook_damage 1",
            "set hook_initdamage 10",
            "set hook_maxdamage 20",
            "set hook_delay 0.2",
            "set rocket_speed_start 650",
            "set rocket_speed_max 2000",
            "set rocket_accel_time 0.75",
            "set rocket_accel_curve 12",
            "set rocket_haste_refire 0.36",
            "set energy_light_speed 1",
            "set ml_training_intermission 0",
            "set ml_training_epoch 0",
            "set ml_training_epochs 0",
            "set ml_training_progress 0",
            "set ml_training_status \"\"",
            f"set maxclients {self._maxclients}",
            f"set timedemo {self.timedemo}",
            f"set timescale {self.timescale:g}",
        ]
        if self.intermission_time is not None:
            cfg_lines.append(f"set intermission_time {float(self.intermission_time):g}")
        if self.intermission_maxtime is not None:
            cfg_lines.append(f"set intermission_maxtime {float(self.intermission_maxtime):g}")
        cfg_lines.extend([
            f"map {self.active_map}",
            "",
        ])
        cfg_path.write_text("\n".join(cfg_lines))
        cmd = [
            str(Q2DED),
            "+set", "game",            "lithium",
            "+set", "ip",              "127.0.0.1",
            "+set", "port",            str(self.sv_port),
            "+exec", cfg_path.name,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if self.console_pipe else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Q2_ROOT),
            preexec_fn=os.setsid,
        )
        # Map-load grace before reset_all() starts polling for obs. The old
        # fixed 8s dominated wall-clock at high timescale with frequent map
        # rotation (reset_all retries handle a server that needs longer).
        time.sleep(float(os.environ.get("Q2_SERVER_BOOT_WAIT", "3")))

    def _stop_server(self):
        if self._proc:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=5)
            self._proc = None

    def server_command(self, command: str) -> bool:
        """Send one q2ded console command when this env owns a stdin pipe."""
        if not self._proc or self._proc.poll() is not None or not self._proc.stdin:
            return False
        line = command.rstrip() + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
            self._proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def set_cvar(self, name: str, value: object) -> bool:
        text = str(value).replace('"', "'")
        return self.server_command(f'set {name} "{text}"')

    def set_training_progress(
        self,
        active: bool,
        epoch: int = 0,
        epochs: int = 0,
        progress: float = 0.0,
        status: str = "",
    ) -> None:
        pct = max(0.0, min(100.0, float(progress)))
        self.set_cvar("ml_training_intermission", 1 if active else 0)
        self.set_cvar("ml_training_epoch", int(epoch))
        self.set_cvar("ml_training_epochs", int(epochs))
        self.set_cvar("ml_training_progress", f"{pct:.1f}")
        self.set_cvar("ml_training_status", status[:64])

    def trigger_next_round(self, map_name: Optional[str] = None) -> List[np.ndarray]:
        target = map_name or self.active_map
        self.active_map = target
        self.set_training_progress(False)
        if not self.server_command(f'gamemap "{target}"'):
            self._stop_server()
        time.sleep(8.0)
        return self.reset_all()

    def _open_sockets(self):
        for i, port in enumerate(self.ml_ports):
            if self._socks[i]:
                self._socks[i].close()
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", port))
            s.settimeout(5.0)
            self._socks[i] = s

    def _send_action(self, slot_idx: int, raw_action: np.ndarray, tick: int):
        sock = self._socks[slot_idx]
        addr = self._game_addrs[slot_idx]
        if sock is None or addr is None:
            return
        sock.sendto(pack_action(_decode_action(raw_action), tick), addr)

    def close(self):
        self._stop_server()
        for s in self._socks:
            if s:
                s.close()
        self._socks = [None] * self.n_ml

    # ── public interface ─────────────────────────────────────────────

    def _rotate_map_live(self) -> List[np.ndarray]:
        """Rotate to the next map WITHOUT restarting q2ded.

        Sends `gamemap` down the server's stdin console (keeps game.so and
        our UDP sockets alive). Process restart cost ~40s of wall time per
        rotation at high timescale; this costs the map load (~2s). Falls
        back to a full restart if the pipe or the reload fails.
        """
        old_ticks = [obs.tick if obs is not None else 0 for obs in self._last_obs]
        self.active_map = self._choose_map()
        try:
            # `sv ml_rotate` = Bot_LevelChange() + gamemap. A bare gamemap
            # bypasses ExitLevel, so 3zb2 never re-reserves its bots and the
            # new level comes up empty (no obs -> 25s stall -> full restart).
            self._proc.stdin.write(f"sv ml_rotate {self.active_map}\n".encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, AttributeError, OSError):
            self._stop_server()
            return self.reset_all()
        print(f"[Server {self.server_id}] rotating live → {self.active_map}", flush=True)

        self._game_addrs = [None] * self.n_ml
        self._last_obs   = [None] * self.n_ml
        self._ep_steps   = [0] * self.n_ml

        obs_list = []
        for i, slot in enumerate(self.bot_slots):
            sock = self._socks[i]
            if sock:
                sock.settimeout(0.0)
                while True:
                    try:
                        sock.recvfrom(4096)
                    except (BlockingIOError, socket.timeout, OSError):
                        break
            # level.framenum resets on map load, so a fresh-map obs has a
            # smaller tick than anything from the old map. Skip stragglers.
            obs = None
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline:
                cand = self._recv_one(i, timeout=max(0.1, deadline - time.monotonic()))
                if cand is None:
                    break
                if old_ticks[i] and cand.tick >= old_ticks[i]:
                    continue  # old-map straggler
                obs = cand
                break
            if obs is None:
                print(f"[Server {self.server_id}] live rotation failed on slot {slot} "
                      f"— falling back to restart", flush=True)
                self._stop_server()
                return self.reset_all()
            self._last_obs[i] = obs
            self._spatial_rewards[i].reset(self.active_map, obs)
            self._send_action(i, np.zeros(ACTION_DIM, np.float32), obs.tick)
            obs_list.append(self._obs_vector(i, obs))
        return obs_list

    def reset_all(self, _retries: int = 0) -> List[np.ndarray]:
        """Start (or restart) the server. Returns initial obs for all slots."""
        if _retries >= 5:
            raise RuntimeError(f"[Server {self.server_id}] failed after 5 restarts")

        self._start_server()
        print(f"[Server {self.server_id}] map {self.active_map}  port {self.sv_port}  "
              f"slots {self.bot_slots}  UDP {self.ml_ports}  (attempt {_retries+1})")
        self._open_sockets()
        self._game_addrs = [None] * self.n_ml
        self._last_obs   = [None] * self.n_ml
        self._ep_steps   = [0] * self.n_ml

        obs_list = []
        for i, slot in enumerate(self.bot_slots):
            print(f"[Server {self.server_id}] waiting for obs on slot {slot}...")
            obs = self._recv_one(i, timeout=25.0)
            if obs is None:
                print(f"[Server {self.server_id}] slot {slot}: no obs — restarting")
                self._stop_server()
                time.sleep(2)
                return self.reset_all(_retries + 1)
            self._last_obs[i] = obs
            self._ep_steps[i] = 0
            self._spatial_rewards[i].reset(self.active_map, obs)
            self._send_action(i, np.zeros(ACTION_DIM, np.float32), obs.tick)
            obs_list.append(self._obs_vector(i, obs))
            print(f"[Server {self.server_id}] slot {slot} ready, tick={obs.tick}")

        return obs_list

    def step_all(self, actions: List[np.ndarray]) -> List[Tuple]:
        """
        Send actions to all bots, receive next observations.

        Returns list of (obs, reward, terminated, truncated, info) per slot.
        Slots that time out return (last_obs, -1.0, True, False, {timeout: True}).
        """
        results = [None] * self.n_ml

        # Send every bot's action before blocking on any observation.  The game
        # services ML slots sequentially, so per-slot send/recv would starve
        # later slots under strict tick validation.
        for i in range(self.n_ml):
            last = self._last_obs[i]
            if last is None:
                results[i] = (np.zeros(OBS_DIM, np.float32), -1.0, True, False,
                              {"map": self.active_map})
                continue

            self._send_action(i, actions[i], last.tick)

        # Slots without a socket fall through to the timeout result below;
        # they must not sit in `pending`, where they would block the
        # whole-frame finalize for every other slot.
        pending = {i for i in range(self.n_ml)
                   if results[i] is None and self._socks[i] is not None}
        obs_by_slot: Dict[int, Observation] = {}
        deadline = time.monotonic() + self.step_timeout_ms / 1000.0
        sock_to_slot = {self._socks[i]: i for i in pending}
        # Per-slot aggregation persists across select wake-ups: a slot is
        # only done when it yields an obs from a NEW frame (tick changed).
        # Finalizing on the first drained packet raced the engine's per-bot
        # obs sends within a frame — a slot drained microseconds early kept
        # returning the PREVIOUS frame's obs, so its echoed action tick was
        # permanently stale and lockstep degraded to a timeout per frame.
        agg0 = {"reward_damage_dealt": 0.0, "reward_damage_taken": 0.0,
                "reward_kill": 0.0, "reward_death": 0.0,
                "reward_item_pickup": 0.0, "reward_hook_traversal": 0.0,
                "reward_damage_taken_prox": 0.0, "reward_offense": 0.0,
                "reward_survival": 0.0}
        slot_agg  = {i: dict(agg0) for i in pending}
        slot_term = {i: (False, 0) for i in pending}
        slot_new  = {i: None for i in pending}
        while pending and sock_to_slot:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                readable, _, _ = select.select(list(sock_to_slot.keys()), [], [], remaining)
            except OSError:
                break
            if not readable:
                break
            for sock in readable:
                i = sock_to_slot.get(sock)
                if i is None or i not in pending:
                    continue
                # Drain the slot's queue. The game sends an obs per frame
                # (pre-pass in lockstep, free-run in pipelined mode), and
                # each packet carries reward deltas the engine clears after
                # sending. Act on the NEWEST obs, SUM rewards across all
                # drained packets, never lose a terminal flag.
                sock.setblocking(False)
                try:
                    while True:
                        try:
                            data, src_addr = sock.recvfrom(4096)
                        except (BlockingIOError, OSError):
                            break
                        obs = parse_obs(data)
                        if obs is None:
                            print(
                                f"[Server {self.server_id}] slot {self.bot_slots[i]}: "
                                f"invalid obs packet len={len(data)} expected={OBS_SIZE} from={src_addr}",
                                flush=True,
                            )
                            continue
                        self._game_addrs[i] = src_addr
                        for k in slot_agg[i]:
                            slot_agg[i][k] += float(getattr(obs, k))
                        if obs.is_terminal:
                            slot_term[i] = (True, int(obs.terminal_reason))
                        cur = slot_new[i]
                        if cur is None or obs.tick >= cur.tick:
                            slot_new[i] = obs
                finally:
                    sock.setblocking(True)
            # Finalize all pending slots TOGETHER, and only once every one
            # of them has an obs from the SAME new frame. Frames are global
            # per server (one pre-pass burst per frame), but the burst's
            # packets land microseconds apart — finalizing a slot on its
            # first new obs let mixed frames through, so one bot's echoed
            # action tick was permanently stale and its RecvAction burned a
            # full timeout every frame.
            ready = {}
            for i in pending:
                newest = slot_new[i]
                if newest is None:
                    break
                last = self._last_obs[i]
                if last is not None and newest.tick == last.tick:
                    break      # stale re-read of the frame we already have
                ready[i] = newest
            else:
                if ready:
                    target = max(o.tick for o in ready.values())
                    if all(o.tick == target for o in ready.values()):
                        for i, newest in ready.items():
                            term_flag, term_reason = slot_term[i]
                            obs_by_slot[i] = dataclasses.replace(
                                newest,
                                is_terminal=term_flag or bool(newest.is_terminal),
                                terminal_reason=term_reason or int(newest.terminal_reason),
                                **slot_agg[i],
                            )
                        pending -= set(ready)

        for i in range(self.n_ml):
            if results[i] is not None:
                continue

            last = self._last_obs[i]
            obs = obs_by_slot.get(i)
            if obs is None:
                results[i] = (self._obs_vector(i, last), -1.0, True, False,
                              {"timeout": True, "map": self.active_map})
                continue

            base_reward = obs.reward
            spatial_bonus, spatial_info = self._spatial_rewards[i].update(obs)
            terminated = obs.is_terminal
            self._ep_steps[i] += 1
            truncated  = (not terminated) and (self._ep_steps[i] >= self.max_ep_steps)
            if terminated or truncated:
                outcome_bonus, outcome_info = self._spatial_rewards[i].finalize_episode(
                    terminal_reason=obs.terminal_reason,
                    truncated=truncated,
                )
                spatial_bonus += outcome_bonus
                spatial_info.update(outcome_info)
            reward     = base_reward + spatial_bonus
            self._last_obs[i] = obs
            info = {
                "map": self.active_map,
                "reward_base": float(base_reward),
                "damage_dealt": float(obs.reward_damage_dealt),
                "damage_taken": float(obs.reward_damage_taken),
                "kills": float(obs.reward_kill),
                "deaths": float(obs.reward_death),
                "items": float(obs.reward_item_pickup),
                "hook_reward": float(obs.reward_hook_traversal),
                "damage_prox": float(obs.reward_damage_taken_prox),
                "offense": float(obs.reward_offense),
                "survival": float(obs.reward_survival),
                "rune_held": float(obs.rune_flags.sum() > 0.0),
                **spatial_info,
            }
            results[i] = (self._obs_vector(i, obs), reward, terminated, truncated, info)

        return results

    def reset_slot(self, slot_idx: int) -> np.ndarray:
        """
        Reset one virtual env after episode end.

        The server keeps running; we just drain stale packets and wait for
        the next fresh obs on this slot's socket (bot will respawn in-game).
        """
        self._ep_steps[slot_idx] = 0
        self._game_addrs[slot_idx] = None
        self._episodes_done += 1

        if (self.map_change_episodes > 0 and len(self.map_pool) > 1 and
                self._episodes_done % self.map_change_episodes == 0):
            if (self.console_pipe and self._proc is not None
                    and self._proc.poll() is None):
                return self._rotate_map_live()[slot_idx]
            self._stop_server()
            return self.reset_all()[slot_idx]

        # Do NOT touch the socket. Consuming a fresh obs here put this slot
        # one frame ahead of its server-mates, and the whole-frame finalize
        # in step_all could never re-equalize faster than the engine's
        # timeout cascade — one death desynced the server permanently. The
        # episode boundary is bookkeeping only: reuse the terminal frame as
        # the new episode's initial obs (the next real frame arrives via
        # the normal step flow 100ms of game time later, frame-synced).
        obs = self._last_obs[slot_idx]
        if obs is None:
            return np.zeros(OBS_DIM, np.float32)
        self._spatial_rewards[slot_idx].reset(self.active_map, obs)
        return self._obs_vector(slot_idx, obs)

    # ── internals ────────────────────────────────────────────────────

    def _recv_one(self, slot_idx: int, timeout: float) -> Optional[Observation]:
        sock = self._socks[slot_idx]
        if sock is None:
            return None
        sock.settimeout(timeout)
        try:
            data, src_addr = sock.recvfrom(4096)
            self._game_addrs[slot_idx] = src_addr
            obs = parse_obs(data)
            if obs is None:
                print(
                    f"[Server {self.server_id}] slot {self.bot_slots[slot_idx]}: "
                    f"invalid obs packet len={len(data)} expected={OBS_SIZE} from={src_addr}",
                    flush=True,
                )
            return obs
        except (socket.timeout, OSError):
            return None


# ── Legacy single-slot env (kept for standalone testing) ────────────

class Q2BotEnv(gym.Env):
    """Single-bot wrapper around Q2MultiEnv.  Use Q2MultiEnv for training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id:       int = 0,
        map_name:     str = "q2dm1",
        bot_slot:     int = 1,
        port_offset:  int = 0,
        timelimit:    int = 0,
        n_bots:       int = 1,
        max_ep_steps: int = 1000,
    ):
        super().__init__()
        self._multi = Q2MultiEnv(
            server_id    = env_id,
            map_name     = map_name,
            n_bots       = n_bots,
            port_offset  = port_offset,
            max_ep_steps = max_ep_steps,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low  = np.array([-1,-1,-45,-30, 0, 0, 0, 0], dtype=np.float32),
            high = np.array([ 1, 1,  45, 30, 1, 1, 3, 9], dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._multi.reset_all()[0], {}

    def step(self, action):
        results = self._multi.step_all([action])
        obs, r, term, trunc, info = results[0]
        if term or trunc:
            obs = self._multi.reset_slot(0)
        return obs, r, term, trunc, info

    def close(self):
        self._multi.close()
