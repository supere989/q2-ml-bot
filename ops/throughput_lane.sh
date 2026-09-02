#!/usr/bin/env bash
# throughput_lane.sh — Phase A max-throughput lane (2026-07-24).
#
# Single-config production run (not an ablation matrix): the maximum clean
# rate verified for this lane, combining every throughput lever:
#   - Q2_NETWORK_CLIENTS env (default 8) ML clients (idle-sleep patched) + 2 3ZB2 bots
#     (ml2sk1), maxclients 12 (2 spare engine slots per the 3ZB2 rule)
#   - TIMESCALE=2 (timedemo 1 + cl_maxfps 20 server-side; cl_maxfps 60 client)
#   - Q2_PIPELINED_COLLECT=1 (pipelined collector wired into the trainer)
#   - fast-nearest lattice, wire v5, fight decision bias ON (defaults)
#   - lattice_direction_coef=0 (v5-era default)
# Baseline: 33 accepted t/s (4 clients, 1x, serial collector).
# 2026-07-24: 8 clients at 2x pipelined LIVELOCKED (per-iteration compute >
# 50ms tick); TIMESCALE is now an env (default 1). aim_anchor_fire_weight
# pinned to 0 (bc_live_v2 lineage value; the new ppo default is 1).
# Target: ~140-160 accepted t/s; verify via the log's sps column and the
# admission counters (failed_rounds / echo_timeouts / telemetry_gap must
# stay zero; realtime_catchup_resyncs are designed nontrainable barriers).
#
# Same BC pin/seed/determinism as the ablation matrices for comparability.
#
# Usage:
#   throughput_lane.sh            # run once (~200k transitions)
#   TOTAL_STEPS=500000 throughput_lane.sh
#
# The server cfg is regenerated per run from the sourced token (mode 0600);
# the token is NEVER printed.  The run log records the full Q2_* environment
# minus *TOKEN* lines.
set -euo pipefail

TREE=/home/raymond/q2-rollout/q2-ml-bot
STAGING=/home/raymond/q2-network-client-staging-20260713
SERVER_ROOT="$STAGING/server-runtime"
CLIENT_ROOT="$STAGING/runtime"
SERVER_CFG=ml_network_local_28300.cfg
GAME_PORT=28300
TELEMETRY_PORT=28301
MAP=mllive_44987431
SEED=7142026
TOTAL_STEPS="${TOTAL_STEPS:-249152}"
BOT_COUNT="${BOT_COUNT:-2}"
INIT="${INIT:-bc}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-throughput}"
TOKEN_ENV=/home/raymond/q2-rollout/local-pilot-telemetry.env
RESUME_PIN="$TREE/training-data/resume/ablation_pilot_bc_v2_00049152"
LOG_DIR="$TREE/logs/throughput"
mkdir -p "$LOG_DIR"

ALL_ARMS=(throughput_v1)
if [ "$#" -gt 0 ]; then
    ARMS=("$@")
else
    ARMS=("${ALL_ARMS[@]}")
fi
for arm in "${ARMS[@]}"; do
    case " ${ALL_ARMS[*]} " in
        *" $arm "*) ;;
        *) echo "unknown arm: $arm (valid: ${ALL_ARMS[*]})" >&2; exit 2 ;;
    esac
done

# --- opponents ----------------------------------------------------------------
case "$BOT_COUNT" in
    0) AUTOSPAWN=0; BOTLIST='""';  BOT_DESC="none (pure ML self-play)" ;;
    1) AUTOSPAWN=1; BOTLIST=1v1sk1; BOT_DESC="1 3ZB2 bot (1v1sk1)" ;;
    2) AUTOSPAWN=1; BOTLIST=ml2sk1; BOT_DESC="2 3ZB2 bots (ml2sk1)" ;;
    3) AUTOSPAWN=1; BOTLIST=2v2sk1; BOT_DESC="3 3ZB2 bots (2v2sk1)" ;;
    4) AUTOSPAWN=1; BOTLIST=ml4sk1; BOT_DESC="4 3ZB2 bots (ml4sk1)" ;;
    *) echo "BOT_COUNT must be 0-4 (got $BOT_COUNT)" >&2; exit 2 ;;
esac

# --- token (mode-checked, never echoed) ---------------------------------------
if [ "$(stat -c '%a' "$TOKEN_ENV")" != "600" ]; then
    echo "refusing to source $TOKEN_ENV: not mode 0600" >&2
    exit 1
fi
source "$TOKEN_ENV"
: "${Q2_ML_CLIENT_TELEMETRY_TOKEN:?token env file missing Q2_ML_CLIENT_TELEMETRY_TOKEN}"
export Q2_ML_CLIENT_TELEMETRY_TOKEN

# --- server cfg (regenerated per arm; token-bearing, mode 0600) ---------------
# $1 = arm timescale (integer; >1 adds timedemo 1 + cl_maxfps 10*N pacing)
write_server_cfg() {
    local ts="$1"
    local sim_hz=$((10 * ts))
    local path="$SERVER_ROOT/lithium/$SERVER_CFG"
    local old_umask
    old_umask=$(umask)
    umask 077
    cat > "$path" <<EOF
set dedicated 1
set deathmatch 1
set cheats 1
set timelimit 0
set fraglimit 0
set use_mapqueue 0
set mapqueue ""
set map_random 0
set autospawn $AUTOSPAWN
set botlist $BOTLIST
set allow_client_bot_controls 0
set maxclients 12
set ml_enabled 0
set ml_bot_slot 99
set ml_teacher_enabled 0
set ml_client_telemetry 1
set ml_client_telemetry_port $TELEMETRY_PORT
set ml_client_telemetry_token "$Q2_ML_CLIENT_TELEMETRY_TOKEN"
set timedemo $([ "$ts" -gt 1 ] && echo 1 || echo 0)
$([ "$ts" -gt 1 ] && echo "set cl_maxfps $sim_hz")
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
    chmod 600 "$path"
}

# --- server lifecycle (exact PIDs only; no pattern kills) ----------------------
SERVER_PID=""
start_server() {
    local log="$1"
    (cd "$SERVER_ROOT" && exec stdbuf -oL -eL ./q2ded \
        +set game lithium \
        +set dedicated 1 \
        +set ip 127.0.0.1 \
        +set port "$GAME_PORT" \
        +exec "$SERVER_CFG") > "$log" 2>&1 &
    SERVER_PID=$!
    local waited=0
    while [ "$waited" -lt 30 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "q2ded exited during startup; see $log" >&2
            return 1
        fi
        if grep -q "ML client telemetry: listening on UDP $TELEMETRY_PORT" "$log" 2>/dev/null; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "q2ded did not report telemetry readiness in 30s; see $log" >&2
    return 1
}
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

# --- common trainer environment ------------------------------------------------
export Q2_NETWORK_CLIENTS="${Q2_NETWORK_CLIENTS:-8}"
export Q2_NETWORK_SERVER="127.0.0.1:$GAME_PORT"
export Q2_NETWORK_TELEMETRY_SERVER="127.0.0.1:$TELEMETRY_PORT"
export Q2_NETWORK_CLIENT_BINARY="$CLIENT_ROOT/quake2"
export Q2_NETWORK_CLIENT_ROOT="$CLIENT_ROOT"
export Q2_NETWORK_CLIENT_DATA_ROOT="$STAGING/client-data/throughput"
export Q2_NETWORK_HARNESS_PORT_BASE=39000
export Q2_NETWORK_QPORT_BASE=49000
export Q2_NETWORK_CLIENT_TIMEOUT=60
export Q2_NETWORK_ROUND_TIMEOUT=3
export Q2_NETWORK_MAX_REJECTED_ECHOES=16
export Q2_NETWORK_CLIENT_ID_PREFIX=ablation-v5
export Q2_EXT_OBS=1
if [ "$INIT" = "zero" ]; then
    RESUME_ARGS=(--reset_lattice 1)
else
    export Q2_RESUME_DIR="$RESUME_PIN"
    RESUME_ARGS=(--reset_optimizer 1 --resume)
fi
export Q2_RUNS_DIR="$TREE/training-data/runs"
export Q2_LATTICE_DIR=/home/raymond/q2-rollout/runtime/baseq2/maps
export Q2_RUST_LATTICE=1
export Q2_RUST_EXTENSION_PATH=/home/raymond/q2-rollout/python/q2_lattice_rs.so
export Q2_POLICY_STATEFUL=1
export Q2_ML_ASYNC=0
export LD_LIBRARY_PATH="$CLIENT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH=/home/raymond/q2-rollout/python${PYTHONPATH:+:$PYTHONPATH}

run_arm() {
    local arm="$1"
    local run_tag="${RUN_TAG_PREFIX}_$arm"
    local ckpt_dir="$TREE/training-data/checkpoints/$run_tag"
    local arm_log="$LOG_DIR/$run_tag.log"
    local server_log="$LOG_DIR/${run_tag}_server.log"
    mkdir -p "$ckpt_dir"

    # Per-arm switches.  lattice_direction_coef=0 for ALL arms (v5 era).
    local extra_env=()
    local arm_ts=1
    local arm_desc="fight decision bias ON (default weights)"
    case "$arm" in
        throughput_v1)
            arm_ts=${TIMESCALE:-1}
            extra_env+=(Q2_PIPELINED_COLLECT=1)
            arm_desc="Phase A max-throughput: 8 clients, 2x, pipelined collector, bias ON"
            ;;
    esac
    local sim_hz=$((10 * arm_ts))
    local client_fps=$((15 * arm_ts)); [ "$client_fps" -lt 60 ] && client_fps=60

    # Per-arm server cfg (pacing depends on arm timescale), then fresh server.
    write_server_cfg "$arm_ts"
    stop_server
    start_server "$server_log"

    local client_extra=()
    if [ "$arm_ts" -gt 1 ]; then
        client_extra=(Q2_NETWORK_CLIENT_EXTRA_ARGS="+set cl_maxfps $client_fps")
    fi

    {
        echo "=== $run_tag ==="
        echo "date: $(date -Is)"
        echo "arm: $arm  ($arm_desc)"
        echo "seed: $SEED  warm_start: $RESUME_PIN (step 49152, reset_optimizer 1)"
        echo "horizon: total_steps=$TOTAL_STEPS (~200k collected transitions past warm start)"
        echo "server: 127.0.0.1:$GAME_PORT telemetry=127.0.0.1:$TELEMETRY_PORT map=$MAP cfg=$SERVER_CFG timelimit=0 fraglimit=0"
        echo "lane rate: ${arm_ts}x (sim ${sim_hz} Hz; timedemo+cl_maxfps server-side, cl_maxfps $client_fps client-side)"
        echo "lane: wire v5 (hazard MOD channels, hit attribution, self_exposure, score frame) + fast-nearest + PIPELINED collector;"
        echo "  opponents=$BOT_DESC; lattice_direction_coef=0; clients=8 (maxclients 12); idle-sleep patched"
        echo "--- environment (secrets redacted) ---"
        printenv | grep -E '^(Q2_|LD_LIBRARY_PATH=|PYTHONPATH=)' | grep -v 'TOKEN' | sort
        for kv in "${extra_env[@]:-}"; do [ -n "$kv" ] && echo "$kv"; done
        for kv in "${client_extra[@]:-}"; do [ -n "$kv" ] && echo "$kv"; done
        echo "BOT_COUNT=$BOT_COUNT"
        echo "--- command ---"
        echo "python3 -u -m train.ppo --n_servers 1 --n_bots_per_server "$Q2_NETWORK_CLIENTS" --n_ml_bots "$Q2_NETWORK_CLIENTS"" \
             "--n_steps "${N_STEPS:-128}" --n_epochs 2 --batch_size 1024 --chunk_len 16" \
             "--total_steps $TOTAL_STEPS --lr 1e-5 --clip_eps 0.2 --vf_coef 0.1" \
             "--ent_coef 0.005 --max_grad_norm 0.5 --aux_coef 0.01" \
             "--aim_anchor_coef 0.02 --aim_anchor_fire_weight 0 --lattice_direction_coef 0" \
             "--map_name $MAP --map_change_episodes 0 --max_ep_steps 1000" \
             "--timelimit 0 --fraglimit 0 --timescale $arm_ts --save_every 8192" \
             "--seed $SEED --game_seed $SEED --deterministic 1" \
             "${RESUME_ARGS[*]}"
        echo "--- trainer output ---"
    } >> "$arm_log"

    (
        cd "$TREE"
        env "${extra_env[@]}" "${client_extra[@]}" \
            Q2_RUN_TAG="$run_tag" \
            Q2_CKPT_DIR="$ckpt_dir" \
            python3 -u -m train.ppo \
                --n_servers 1 --n_bots_per_server "$Q2_NETWORK_CLIENTS" --n_ml_bots "$Q2_NETWORK_CLIENTS" \
                --n_steps "${N_STEPS:-128}" --n_epochs 2 --batch_size 1024 --chunk_len 16 \
                --total_steps "$TOTAL_STEPS" --lr 1e-5 --clip_eps 0.2 \
                --vf_coef 0.1 --ent_coef 0.005 --max_grad_norm 0.5 \
                --aux_coef 0.01 --aim_anchor_coef 0.02 --aim_anchor_fire_weight 0 \
                --lattice_direction_coef 0 \
                --map_name "$MAP" --map_change_episodes 0 --max_ep_steps 1000 \
                --timelimit 0 --fraglimit 0 --timescale "$arm_ts" --save_every 8192 \
                --seed "$SEED" --game_seed "$SEED" --deterministic 1 \
                "${RESUME_ARGS[@]}"
    ) >> "$arm_log" 2>&1
    echo "=== $run_tag finished: $(date -Is) ===" >> "$arm_log"
}

for arm in "${ARMS[@]}"; do
    echo "[throughput_lane] starting arm: $arm"
    run_arm "$arm"
    echo "[throughput_lane] arm complete: $arm"
done
stop_server
trap - EXIT INT TERM
echo "[throughput_lane] all requested arms complete"
