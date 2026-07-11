#!/usr/bin/env bash
set -euo pipefail

Q2ML_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_DIR="$(cd "$Q2ML_DIR/.." && pwd)"

export Q2_ROOT="${Q2_ROOT:-$WORKSPACE_DIR/q2_lithium_merge}"
export Q2_POLICY_STATEFUL="${Q2_POLICY_STATEFUL:-0}"

cd "$Q2ML_DIR"
exec python3 -u tools/evaluate_1v1.py "$@"
