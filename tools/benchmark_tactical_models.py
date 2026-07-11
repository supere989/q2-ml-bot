#!/usr/bin/env python3
"""Benchmark Ollama models as Quake II tactical intent sidecars."""

import argparse
import json
import sys
import time
import urllib.request
from typing import Dict


DEFAULT_MODELS = [
    "llama3.2:3b",
    "qwen3-8b-gemini-3-pro-preview-high-reasoning-distill-q4_k_m:custom",
    "qwen2.5-coder:7b",
    "qwen-2.5-coder-3b-instruct-q4.0:latest",
]

PROMPT = (
    "Return exactly one minified JSON object with no markdown and no extra keys. "
    "Schema: {\"t\":\"take_cover\",\"a\":0.2,\"c\":0.8,\"cb\":0.9,\"sw\":0.5,\"pw\":0.6}. "
    "t is one of: engage,evade,take_cover,hold,chase_sound,rotate_item,explore. "
    "a,c,cb,sw,pw are floats 0..1 for "
    "aggression,caution,cover_bias,sound_weight,prediction_weight. "
    "Quake2 state: health 37, armor 0, shotgun ammo 5, enemy not visible, "
    "sound left_front age .4, damage from right, enemy likely has rocket, "
    "health nearby, cover nearby, opponent skill unknown. "
    "Low health increases risk but does not forbid engage; weigh damage race, "
    "opponent threat, cover, health access, and contact confidence."
)


def score(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    out = 0
    if payload.get("t") in {"evade", "take_cover", "hold", "chase_sound"}:
        out += 2
    elif payload.get("t") == "engage":
        out += 1
    try:
        aggression = float(payload.get("a", 1.0))
        caution = float(payload.get("c", 0.0))
        cover_bias = float(payload.get("cb", 0.0))
        sound_weight = float(payload.get("sw", 0.0))
        if payload.get("t") == "engage" and aggression >= 0.55:
            out += 1
        elif aggression <= 0.55:
            out += 1
        if caution >= 0.65:
            out += 1
        if cover_bias >= 0.55:
            out += 1
        if sound_weight >= 0.40:
            out += 1
    except (TypeError, ValueError):
        pass
    return out


def call_model(host: str, model: str, timeout: float, keep_alive: str) -> Dict[str, object]:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "format": "json",
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "num_ctx": 512,
            "num_predict": 96,
        },
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    wall = time.time() - start

    text = obj.get("response", "").strip()
    try:
        parsed = json.loads(text)
        valid = True
    except json.JSONDecodeError:
        parsed = text[:240]
        valid = False

    return {
        "model": model,
        "valid_json": valid,
        "score": score(parsed),
        "wall_seconds": round(wall, 3),
        "load_seconds": round((obj.get("load_duration") or 0) / 1e9, 3),
        "eval_seconds": round((obj.get("eval_duration") or 0) / 1e9, 3),
        "eval_count": obj.get("eval_count"),
        "response": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--keep_alive", default="10m")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        try:
            row = call_model(args.host, model, args.timeout, args.keep_alive)
        except Exception as exc:
            row = {"model": model, "error": repr(exc)}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    print("SUMMARY")
    for row in sorted(
        rows,
        key=lambda r: (int(r.get("score", -1)), -float(r.get("wall_seconds", 9999.0))),
        reverse=True,
    ):
        print(
            json.dumps(
                {
                    "model": row.get("model"),
                    "score": row.get("score"),
                    "wall_seconds": row.get("wall_seconds"),
                    "valid_json": row.get("valid_json"),
                    "error": row.get("error"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
