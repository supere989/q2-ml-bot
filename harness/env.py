"""
env.py — Gymnasium environment wrapping a single Q2 dedicated server instance.

Each Q2BotEnv manages:
  - one q2ded subprocess (headless, loopback-only)
  - one UDP socket listening for observations from game.so
  - one bot slot
"""

import socket
import subprocess
import time
import os
import signal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .protocol import (
    Observation, Action, parse_obs, pack_action,
    ML_BASE_PORT, Observation as Obs,
)

Q2_ROOT     = Path(os.environ.get("Q2_ROOT", "/home/raymond/q2_lithium_merge"))
Q2DED       = Q2_ROOT / "q2ded"
Q2_MAPS     = ["q2dm1", "q2dm2", "q2dm3", "q2dm4", "q2dm5", "q2dm6", "q2dm7", "q2dm8"]
STEP_TIMEOUT_MS = 80   # ms to wait for action reply before reusing last
OBS_DIM     = Obs.OBS_DIM


class Q2BotEnv(gym.Env):
    """Single-bot Quake 2 training environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id:     int   = 0,
        map_name:   str   = "q2dm1",
        bot_slot:   int   = 1,
        port_offset: int  = 0,
        timelimit:  int   = 5,       # minutes per episode
        n_bots:     int   = 3,       # opponent bots (using old 3ZB2 AI)
    ):
        super().__init__()
        self.env_id    = env_id
        self.map_name  = map_name
        self.bot_slot  = bot_slot
        self.sv_port   = 27910 + port_offset
        self.ml_port   = ML_BASE_PORT + bot_slot
        self.timelimit = timelimit
        self.n_bots    = n_bots

        self._proc:   Optional[subprocess.Popen] = None
        self._sock:   Optional[socket.socket]    = None
        self._last_obs: Optional[Observation]    = None

        # observation: flat float32 vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32,
        )

        # action: [move_fwd, move_right, look_yaw, look_pitch,
        #          jump, fire, hook, weapon]
        self.action_space = spaces.Box(
            low  = np.array([-1,-1,-45,-30, 0, 0, 0, 0], dtype=np.float32),
            high = np.array([ 1, 1,  45, 30, 1, 1, 3, 9], dtype=np.float32),
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def _start_server(self):
        if self._proc and self._proc.poll() is None:
            return  # already running

        cmd = [
            str(Q2DED),
            "+set", "game",       "lithium",
            "+set", "dedicated",  "1",
            "+set", "ip",         "127.0.0.1",
            "+set", "port",       str(self.sv_port),
            "+set", "deathmatch", "1",
            "+set", "timelimit",  str(self.timelimit),
            "+set", "autospawn",  "0",
            "+set", "ml_enabled", "1",        # triggers ML bridge in game.so
            "+set", "ml_bot_slot", str(self.bot_slot),
            "+map", self.map_name,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        time.sleep(3.0)   # wait for map load

        # spawn opponent bots via rcon-equivalent
        for _ in range(self.n_bots):
            self._send_cmd(f"sv addbot 1\n")
            time.sleep(0.2)

    def _stop_server(self):
        if self._proc:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._proc = None

    def _open_socket(self):
        if self._sock:
            self._sock.close()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", self.ml_port))
        self._sock.settimeout(5.0)

    def _send_cmd(self, cmd: str):
        """Send a server command via a second UDP socket (rcon placeholder)."""
        # TODO: implement rcon or use a pipe; for now we rely on autospawn
        pass

    # ── gymnasium interface ───────────────────────────────────────────

    def reset(
        self,
        seed=None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        map_name = (options or {}).get("map", self.map_name)
        if map_name != self.map_name:
            self.map_name = map_name
            self._stop_server()

        self._start_server()
        self._open_socket()
        self._last_obs = None

        # wait for first observation from the ML bot
        obs = self._recv_obs(timeout=10.0)
        if obs is None:
            # server not ready — restart
            self._stop_server()
            time.sleep(1)
            return self.reset(seed=seed, options=options)

        self._last_obs = obs
        return obs.to_vector(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._last_obs is not None, "call reset() first"

        act = self._decode_action(action)
        tick = self._last_obs.tick

        # send action
        pkt = pack_action(act, tick)
        self._sock.sendto(pkt, ("127.0.0.1", self.ml_port - ML_BASE_PORT + ML_BASE_PORT))

        # wait for next observation
        obs = self._recv_obs(timeout=STEP_TIMEOUT_MS / 1000.0)
        if obs is None:
            # server died or stalled
            return (
                self._last_obs.to_vector(),
                -1.0,   # penalty for lost step
                True,   # terminate
                False,
                {"timeout": True},
            )

        reward    = obs.reward
        terminated = obs.is_terminal
        truncated  = False
        self._last_obs = obs

        return obs.to_vector(), reward, terminated, truncated, {}

    def close(self):
        self._stop_server()
        if self._sock:
            self._sock.close()
            self._sock = None

    # ── helpers ──────────────────────────────────────────────────────

    def _recv_obs(self, timeout: float) -> Optional[Observation]:
        if self._sock is None:
            return None
        self._sock.settimeout(timeout)
        try:
            data, _ = self._sock.recvfrom(4096)
            return parse_obs(data)
        except (socket.timeout, OSError):
            return None

    @staticmethod
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
