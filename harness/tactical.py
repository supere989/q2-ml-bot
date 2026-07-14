"""
tactical.py - optional Ollama-backed tactical intent sidecar.

This module keeps language-model reasoning outside the game DLL and outside the
fixed policy observation shape. It converts a compact observation summary into
a low-frequency intent packet that can be logged, used for replay critique, or
optionally applied as a conservative action modifier during evaluation.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .protocol import Observation


LIVE_MODEL = "llama3.2:3b"
QUALITY_MODEL = "qwen3-8b-gemini-3-pro-preview-high-reasoning-distill-q4_k_m:custom"
OFFLINE_MODEL = "qwen2.5-coder:7b"

TACTICS = {
    "engage",
    "evade_to_health",
    "take_cover",
    "hold_angle",
    "chase_sound",
    "rotate_item",
    "explore",
}

_TACTIC_ALIASES = {
    "evade": "evade_to_health",
    "escape": "evade_to_health",
    "defend": "take_cover",
    "cover": "take_cover",
    "hold": "hold_angle",
    "search": "chase_sound",
    "attack": "engage",
}


def _clamp01(value: object, default: float) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


@dataclass
class TacticalIntent:
    tactic: str = "explore"
    aggression: float = 0.5
    caution: float = 0.5
    cover_bias: float = 0.5
    sound_weight: float = 0.5
    prediction_weight: float = 0.5
    reason: str = ""
    valid: bool = False
    source: str = ""
    latency_s: float = 0.0

    @classmethod
    def neutral(cls, source: str = "") -> "TacticalIntent":
        return cls(source=source)

    @classmethod
    def from_payload(
        cls,
        payload: object,
        source: str = "",
        latency_s: float = 0.0,
    ) -> "TacticalIntent":
        if not isinstance(payload, dict):
            return cls.neutral(source=source)

        raw_tactic = str(payload.get("t", payload.get("tactic", "explore"))).lower()
        tactic = _TACTIC_ALIASES.get(raw_tactic, raw_tactic)
        if tactic not in TACTICS:
            tactic = "explore"

        return cls(
            tactic=tactic,
            aggression=_clamp01(payload.get("a", payload.get("aggression")), 0.5),
            caution=_clamp01(payload.get("c", payload.get("caution")), 0.5),
            cover_bias=_clamp01(payload.get("cb", payload.get("cover_bias")), 0.5),
            sound_weight=_clamp01(payload.get("sw", payload.get("sound_weight")), 0.5),
            prediction_weight=_clamp01(
                payload.get("pw", payload.get("prediction_weight")), 0.5
            ),
            reason=str(payload.get("reason", ""))[:160],
            valid=True,
            source=source,
            latency_s=latency_s,
        )

    def compact(self) -> Dict[str, object]:
        return {
            "tactic": self.tactic,
            "aggression": round(self.aggression, 3),
            "caution": round(self.caution, 3),
            "cover_bias": round(self.cover_bias, 3),
            "sound_weight": round(self.sound_weight, 3),
            "prediction_weight": round(self.prediction_weight, 3),
            "valid": self.valid,
            "source": self.source,
            "latency_s": round(self.latency_s, 3),
        }


def _visible_enemy_stats(obs: Observation) -> Tuple[int, float, float, float]:
    count = max(0, min(int(obs.entity_count), obs.entities.shape[0]))
    visible = 0
    nearest = float("inf")
    best_exposure = 0.0
    nearest_health = 0.0
    for ent in obs.entities[:count]:
        exposure = abs(float(ent[8]))
        if not (ent[7] > 0.5 and exposure > 0.0):
            continue
        dist = float(np.linalg.norm(ent[:3]))
        visible += 1
        if dist < nearest:
            nearest = dist
            nearest_health = float(ent[6])
        best_exposure = max(best_exposure, exposure)
    if visible == 0:
        nearest = 0.0
    return visible, nearest, best_exposure, nearest_health


def _audio_summary(obs: Observation) -> Tuple[str, float, float]:
    x, y, _z, age, alert = [float(v) for v in obs.audio[:5]]
    if alert <= 0.0 or (abs(x) + abs(y)) < 1e-3:
        return "none", float(age), float(alert)

    angle = math.degrees(math.atan2(y, max(x, 1e-3)))
    if abs(angle) < 20:
        direction = "front"
    elif angle >= 20:
        direction = "left_front" if angle < 100 else "left"
    else:
        direction = "right_front" if angle > -100 else "right"
    return direction, float(age), float(alert)


def summarize_observation(obs: Observation, info: Optional[Dict[str, object]] = None) -> str:
    """Return a compact text state for a tactical LLM prompt."""
    info = info or {}
    visible_count, nearest_enemy, exposure, enemy_health = _visible_enemy_stats(obs)
    sound_dir, sound_age, alert = _audio_summary(obs)
    health = int(round(float(obs.self_state[6])))
    armor = int(round(float(obs.self_state[7])))
    weapon = int(round(float(obs.self_state[8])))
    ammo = int(round(float(obs.self_state[9])))
    damage_taken = float(info.get("damage_taken", 0.0) or 0.0)
    damage_dealt = float(info.get("damage_dealt", 0.0) or 0.0)
    hook_near = bool(float(info.get("hook_required_near", 0.0) or 0.0) > 0.0)

    return (
        f"health {health}, armor {armor}, weapon_id {weapon}, ammo {ammo}, "
        f"enemy_visible {visible_count > 0}, visible_enemies {visible_count}, "
        f"nearest_enemy {nearest_enemy:.0f}, exposure {exposure:.2f}, "
        f"nearest_enemy_health {enemy_health:.0f}, opponent_skill unknown, "
        f"sound {sound_dir}, sound_age {sound_age:.2f}, alert {alert:.2f}, "
        f"damage_taken_recent {damage_taken:.1f}, damage_dealt_recent {damage_dealt:.1f}, "
        f"required_hook_near {hook_near}"
    )


class OllamaTacticalReasoner:
    """Low-frequency Ollama client for tactical intent packets."""

    def __init__(
        self,
        model: str = LIVE_MODEL,
        host: str = "http://127.0.0.1:11434",
        interval_steps: int = 20,
        timeout_s: float = 3.0,
        keep_alive: str = "10m",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.interval_steps = max(1, int(interval_steps))
        self.timeout_s = max(0.1, float(timeout_s))
        self.keep_alive = keep_alive
        self.last_step = -self.interval_steps
        self.intent = TacticalIntent.neutral(source=model)
        self.update_count = 0
        self.error_count = 0
        self.last_error = ""

    @classmethod
    def from_env(cls, fallback_model: str = LIVE_MODEL) -> "OllamaTacticalReasoner":
        return cls(
            model=os.environ.get("Q2_TACTICAL_MODEL", fallback_model),
            host=os.environ.get("Q2_OLLAMA_HOST", "http://127.0.0.1:11434"),
            interval_steps=int(os.environ.get("Q2_TACTICAL_INTERVAL", "20")),
            timeout_s=float(os.environ.get("Q2_TACTICAL_TIMEOUT", "10.0")),
            keep_alive=os.environ.get("Q2_TACTICAL_KEEP_ALIVE", "10m"),
        )

    def reset(self) -> None:
        self.last_step = -self.interval_steps
        self.intent = TacticalIntent.neutral(source=self.model)
        self.update_count = 0
        self.error_count = 0
        self.last_error = ""

    def maybe_update(
        self,
        obs: Observation,
        info: Optional[Dict[str, object]],
        step: int,
    ) -> Tuple[TacticalIntent, bool]:
        if step - self.last_step < self.interval_steps:
            return self.intent, False
        self.last_step = step
        try:
            self.intent = self.query(obs, info)
            self.update_count += 1
            return self.intent, True
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.error_count += 1
            self.last_error = str(exc)[:200]
            return self.intent, False

    def query(self, obs: Observation, info: Optional[Dict[str, object]]) -> TacticalIntent:
        prompt = (
            "Return exactly one minified JSON object with no markdown and no extra keys. "
            "Schema: {\"t\":\"take_cover\",\"a\":0.2,\"c\":0.8,\"cb\":0.9,\"sw\":0.5,\"pw\":0.6}. "
            "t is one of: engage,evade,take_cover,hold,chase_sound,rotate_item,explore. "
            "a,c,cb,sw,pw are floats 0..1 for "
            "aggression,caution,cover_bias,sound_weight,prediction_weight. "
            "Low health increases risk but does not forbid engage; weigh damage race, "
            "opponent threat, cover, health access, and contact confidence. "
            "Quake2 state: "
            + summarize_observation(obs, info)
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": 512,
                "num_predict": 96,
            },
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        latency = time.time() - start
        response = obj.get("response", "").strip()
        parsed = json.loads(response)
        return TacticalIntent.from_payload(parsed, source=self.model, latency_s=latency)


def apply_intent_to_action(
    action: np.ndarray,
    intent: TacticalIntent,
    obs: Optional[Observation],
) -> np.ndarray:
    """Conservatively bias an existing policy action with a tactical intent."""
    if not intent.valid:
        return action

    out = np.array(action, dtype=np.float32, copy=True)
    visible_count = 0
    if obs is not None:
        visible_count, _nearest, _exposure, _enemy_health = _visible_enemy_stats(obs)

    if visible_count == 0 and intent.tactic in {"evade_to_health", "take_cover", "rotate_item"}:
        out[5] = 0.0  # do not fire blindly while taking a survival/resource intent

    if intent.tactic in {"evade_to_health", "take_cover"}:
        out[0] = float(np.clip(out[0] * (1.0 - 0.25 * intent.caution), -1.0, 1.0))
        if abs(out[1]) < 0.2:
            out[1] = 0.35 if intent.sound_weight < 0.5 else -0.35

    if obs is not None and intent.tactic == "chase_sound":
        x, y, _z, age, alert = [float(v) for v in obs.audio[:5]]
        if alert > 0.0 and age < 1.5 and (abs(x) + abs(y)) > 1e-3:
            yaw = math.degrees(math.atan2(y, max(x, 1e-3)))
            out[2] = float(np.clip(out[2] + yaw * 0.25 * intent.sound_weight, -45, 45))

    return out
