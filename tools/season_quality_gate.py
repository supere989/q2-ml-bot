#!/usr/bin/env python3
"""Evaluate one or more completed distributed-training seasons for promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.season_quality_gate import SeasonGateConfig, evaluate_promotion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--min-successful-seasons", type=int, default=3)
    parser.add_argument("--min-generations", type=int, default=100)
    parser.add_argument("--min-env-steps", type=int, default=1_000_000)
    parser.add_argument("--min-map-episodes", type=int, default=100)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--max-approx-kl-p95", type=float, default=0.03)
    parser.add_argument("--max-clip-fraction-p95", type=float, default=0.20)
    args = parser.parse_args()
    config = SeasonGateConfig(
        min_successful_seasons=args.min_successful_seasons,
        min_generations=args.min_generations,
        min_env_steps=args.min_env_steps,
        min_map_episodes=args.min_map_episodes,
        min_speedup=args.min_speedup,
        max_approx_kl_p95=args.max_approx_kl_p95,
        max_clip_fraction_p95=args.max_clip_fraction_p95,
    )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = evaluate_promotion(reports, config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
