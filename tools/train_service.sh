#!/bin/sh
# Thin operator wrapper with separate proof and primary-trainer selectors.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RUNTIME_ROOT=${Q2_MULTIRES_RUNTIME_ROOT:-}

if [ -z "$RUNTIME_ROOT" ]; then
    echo "Q2_MULTIRES_RUNTIME_ROOT must name the sealed runtime directory" >&2
    exit 2
fi

case "${1:-}" in
    preflight|prove|start|stop|status)
        cd "$ROOT"
        exec python3 -u -m train.multires_service \
            --runtime_root "$RUNTIME_ROOT" "$1"
        ;;
    logs)
        LOG_PATH="$RUNTIME_ROOT/multires-service.log"
        if [ ! -f "$LOG_PATH" ]; then
            echo "No multires service log at $LOG_PATH" >&2
            exit 1
        fi
        exec tail -n 80 "$LOG_PATH"
        ;;
    *)
        echo "Usage: Q2_MULTIRES_RUNTIME_ROOT=/abs/runtime $0 {preflight|prove|start|stop|status|logs}" >&2
        exit 2
        ;;
esac
