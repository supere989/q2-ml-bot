#!/usr/bin/env bash
# phaseb_worker.sh — Phase B network-native rollout worker (2026-07-25).
#
# Usage: phaseb_worker.sh w1|w2
#   w1: lane 28310/28311, harness base 39100, qport base 49100
#   w2: lane 28320/28321, harness base 39200, qport base 49200
#
# Starts a dedicated q2ded on its lane (maxclients 12, 2 3ZB2 bots ml2sk1,
# mllive_44987431, timelimit 0, 1x pacing) then runs tools/rollout_worker.py
# --mode network --continuous (NON-leased: no lease fencing/recovery) with
# 8 ML clients.  Secrets are sourced from mode-600 env files and never
# printed.  Server lifecycle is exact-PID only.
set -euo pipefail

LANE="${1:?usage: phaseb_worker.sh w1|w2}"
case "$LANE" in
    w1) GAME_PORT=28310; TELEMETRY_PORT=28311; HARNESS_BASE=39100; QPORT_BASE=49100;
        WORKER_ID=phaseb-w1; SEED=7150001 ;;
    w2) GAME_PORT=28320; TELEMETRY_PORT=28321; HARNESS_BASE=39200; QPORT_BASE=49200;
        WORKER_ID=phaseb-w2; SEED=7150002 ;;
    *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac

TREE=/home/raymond/q2-rollout/q2-ml-bot
STAGING=/home/raymond/q2-network-client-staging-20260713
SERVER_ROOT="$STAGING/server-runtime"
CLIENT_ROOT="$STAGING/runtime"
SECRET_DIR=/home/raymond/q2-rollout
MANIFEST="$TREE/runtime-manifest-phaseb.json"
MAP=mllive_44987431
SERVER_CFG="ml_network_local_${GAME_PORT}.cfg"
LOG_DIR="$TREE/logs/phaseb"
mkdir -p "$LOG_DIR"

for f in "$SECRET_DIR/local-pilot-telemetry.env" "$SECRET_DIR/phaseb-rollout-token.env" "$SECRET_DIR/phaseb-attestation-key.env"; do
    [ "$(stat -c '%a' "$f")" = "600" ] || { echo "refusing: $f not mode 0600" >&2; exit 1; }
    source "$f"
done
: "${Q2_ML_CLIENT_TELEMETRY_TOKEN:?missing}"
: "${Q2_ROLLOUT_TOKEN:?missing}"
: "${Q2_ROLLOUT_ATTESTATION_KEY:?missing}"
export Q2_ML_CLIENT_TELEMETRY_TOKEN Q2_ROLLOUT_TOKEN Q2_ROLLOUT_ATTESTATION_KEY

# --- server cfg (regenerated per launch; token-bearing, mode 0600) ------------
old_umask=$(umask); umask 077
cat > "$SERVER_ROOT/lithium/$SERVER_CFG" <<EOF
set dedicated 1
set deathmatch 1
set cheats 1
set timelimit 0
set fraglimit 0
set use_mapqueue 0
set mapqueue ""
set map_random 0
set autospawn 1
set botlist ml2sk1
set allow_client_bot_controls 0
set maxclients 12
set ml_enabled 0
set ml_bot_slot 99
set ml_teacher_enabled 0
set ml_client_telemetry 1
set ml_client_telemetry_port $TELEMETRY_PORT
set ml_client_telemetry_token "$Q2_ML_CLIENT_TELEMETRY_TOKEN"
set timedemo 0
set timescale 1
set use_runes 1
set use_startobserver 0
set use_startchasecam 0
set use_hook 1
set hook_speed 1900
set hook_pullspeed 1700
set hook_maxtime 15.0
set hook_damage 1
set hook_initdamage 10
set hook_maxdamage 20
set hook_delay 0.2
set rocket_speed_start 650
set rocket_speed_max 2000
set rocket_accel_time 0.75
set rocket_accel_curve 12
set rocket_haste_refire 0.36
set energy_light_speed 1
map $MAP
EOF
umask "$old_umask"
chmod 600 "$SERVER_ROOT/lithium/$SERVER_CFG"

# --- q2ded (exact PID) ---------------------------------------------------------
SRVLOG="$LOG_DIR/${WORKER_ID}_server.log"
(cd "$SERVER_ROOT" && exec stdbuf -oL -eL ./q2ded \
    +set game lithium +set dedicated 1 +set ip 127.0.0.1 \
    +set port "$GAME_PORT" +exec "$SERVER_CFG") > "$SRVLOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
waited=0
while [ "$waited" -lt 30 ]; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "q2ded died; see $SRVLOG" >&2; exit 1; }
    grep -q "ML client telemetry: listening on UDP $TELEMETRY_PORT" "$SRVLOG" 2>/dev/null && break
    sleep 1; waited=$((waited + 1))
done
[ "$waited" -lt 30 ] || { echo "q2ded not ready in 30s" >&2; exit 1; }

# --- coordinator readiness (learner must be listening before first fetch) -----
cwait=0
until (exec 3<>/dev/tcp/127.0.0.1/38888) 2>/dev/null; do
    cwait=$((cwait + 1))
    [ "$cwait" -lt 60 ] || { echo "coordinator not ready in 60s" >&2; exit 1; }
    sleep 1
done

# --- worker environment (must match the attested manifest env) -----------------
export Q2_ROOT="$SERVER_ROOT"
export Q2_EXT_OBS=1
export Q2_RUST_LATTICE=1
export Q2_ML_ASYNC=0
export Q2_POLICY_STATEFUL=1
export Q2_RUST_EXTENSION_PATH=/home/raymond/q2-rollout/python/q2_lattice_rs.so
export PYTHONPATH=/home/raymond/q2-rollout/python${PYTHONPATH:+:$PYTHONPATH}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export Q2_SOURCE_REVISION=q2-ml-bot-v5-phaseb-2026-07-25
export LD_LIBRARY_PATH="$CLIENT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export Q2_NETWORK_SERVER="127.0.0.1:$GAME_PORT"
export Q2_NETWORK_TELEMETRY_SERVER="127.0.0.1:$TELEMETRY_PORT"
export Q2_NETWORK_CLIENT_BINARY="$CLIENT_ROOT/quake2"
export Q2_NETWORK_CLIENT_ROOT="$CLIENT_ROOT"
export Q2_NETWORK_CLIENT_DATA_ROOT="$STAGING/client-data/phaseb_$LANE"
export Q2_NETWORK_HARNESS_PORT_BASE="$HARNESS_BASE"
export Q2_NETWORK_QPORT_BASE="$QPORT_BASE"
export Q2_NETWORK_CLIENT_TIMEOUT=60
export Q2_NETWORK_ROUND_TIMEOUT=3
export Q2_NETWORK_MAX_REJECTED_ECHOES=16
export Q2_NETWORK_CLIENT_ID_PREFIX="$WORKER_ID"

LOG="$LOG_DIR/${WORKER_ID}.log"
{
    echo "=== phaseb worker $WORKER_ID ==="
    echo "date: $(date -Is)"
    echo "lane: game 127.0.0.1:$GAME_PORT telemetry 127.0.0.1:$TELEMETRY_PORT harness $HARNESS_BASE+ qport $QPORT_BASE+ map=$MAP"
    echo "mode: network, continuous, NON-leased (v1); 8 ML clients + 2 3ZB2 bots; steps=128/rollout"
    echo "manifest: $MANIFEST"
    echo "--- environment (secrets redacted) ---"
    printenv | grep -E '^(Q2_|LD_LIBRARY_PATH=|PYTHONPATH=|CUBLAS|PYTHONHASHSEED)' \
        | grep -vE 'TOKEN|KEY' | sort
    echo "--- worker output ---"
} >> "$LOG"

cd "$TREE"
exec python3 -u tools/rollout_worker.py \
    --mode network --continuous \
    --coordinator http://127.0.0.1:38888 \
    --token "$Q2_ROLLOUT_TOKEN" \
    --worker-id "$WORKER_ID" \
    --sequence 1 --seed "$SEED" --game-seed "$SEED" --rollout-index 0 \
    --steps 128 --map-name "$MAP" --n-bots 8 --n-ml 8 \
    --max-ep-steps 1000 --timescale 1.0 \
    --deterministic 0 \
    --runtime-manifest "$MANIFEST" \
    --attestation-key-env Q2_ROLLOUT_ATTESTATION_KEY \
    --lattice-dir "$TREE/training-data/checkpoints/phaseb_zero_v1/worker_state/$WORKER_ID" \
    >> "$LOG" 2>&1
