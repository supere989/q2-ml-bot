#!/usr/bin/env bash
# ablation_v5_matrix.sh — v5-era ablation matrix (2026-07-24).
#
# Lane: WIRE V5 (survival pack: hazard-typed last_damage_mod/last_death_mod,
# hit attribution, self_exposure, score frame) + fight decision bias +
# pipelined collector (synced, NOT wired into the trainer loop — ppo.py still
# drives serial collect_round via the multi-env adapter) + fast-nearest.
#
# Arms (sequential; same BC pin/seed/determinism; lattice_direction_coef=0
# for ALL arms):
#   v5_bias_on     all roles on, fight decision bias active (defaults), 1x
#   v5_bias_off    bias neutralized by zeroing all five Q2_BIAS_W_* weights
#                  (verified in harness/spatial.py: all weights zero =>
#                  fight_bias = clip(0) = 0 always; reward thresholds -0.2/0.3
#                  and deposit conditions never fire; bias_active is only an
#                  alive-check, no master switch exists), 1x
#   v5_bias_on_2x  same as v5_bias_on but TIMESCALE=2 (banks the collector
#                  work; 4-client 2x livelocked pre-v5 — supervise: if
#                  admission failures/echo timeouts/telemetry-gap resyncs
#                  appear, restart this arm at TIMESCALE=1)
#
# Every arm: fresh server restart, BC pin resume
# training-data/resume/ablation_pilot_bc_v2_00049152 (policy + 398-cell
# lattice at step 49,152, --reset_optimizer 1), seed 7142026
# (--seed N --game_seed N --deterministic 1, Q2_ML_ASYNC=0),
# --total_steps 249152 (~200k collected transitions past warm start),
# 4 ML clients + 2 3ZB2 bots (ml2sk1, maxclients 8 = 2 spare slots),
# fixed map mllive_44987431, timelimit 0 / fraglimit 0 (no map rotation).
#
# Usage:
#   ablation_v5_matrix.sh                  # all three arms, sequential
#   ablation_v5_matrix.sh v5_bias_on_2x    # one arm (any subset)
#
# The server cfg is (re)generated per arm from the sourced token (mode 0600);
# the token is NEVER printed.  Per-arm run logs record the full Q2_*
# environment minus *TOKEN* lines.
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
TOTAL_STEPS=249152
BOT_COUNT="${BOT_COUNT:-2}"
TOKEN_ENV=/home/raymond/q2-rollout/local-pilot-telemetry.env
RESUME_PIN="$TREE/training-data/resume/ablation_pilot_bc_v2_00049152"
LOG_DIR="$TREE/logs/ablation_v5_matrix"
mkdir -p "$LOG_DIR"

ALL_ARMS=(v5_bias_on v5_bias_off v5_bias_on_2x)
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
set maxclients 8
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
export Q2_NETWORK_CLIENTS=4
export Q2_NETWORK_SERVER="127.0.0.1:$GAME_PORT"
export Q2_NETWORK_TELEMETRY_SERVER="127.0.0.1:$TELEMETRY_PORT"
export Q2_NETWORK_CLIENT_BINARY="$CLIENT_ROOT/quake2"
export Q2_NETWORK_CLIENT_ROOT="$CLIENT_ROOT"
export Q2_NETWORK_CLIENT_DATA_ROOT="$STAGING/client-data/ablation_v5_matrix"
export Q2_NETWORK_HARNESS_PORT_BASE=39000
export Q2_NETWORK_QPORT_BASE=49000
export Q2_NETWORK_CLIENT_TIMEOUT=60
export Q2_NETWORK_ROUND_TIMEOUT=3
export Q2_NETWORK_MAX_REJECTED_ECHOES=16
export Q2_NETWORK_CLIENT_ID_PREFIX=ablation-v5
export Q2_EXT_OBS=1
export Q2_RESUME_DIR="$RESUME_PIN"
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
    local run_tag="ablation_v5_$arm"
    local ckpt_dir="$TREE/training-data/checkpoints/$run_tag"
    local arm_log="$LOG_DIR/$run_tag.log"
    local server_log="$LOG_DIR/${run_tag}_server.log"
    mkdir -p "$ckpt_dir"

    # Per-arm switches.  lattice_direction_coef=0 for ALL arms (v5 era).
    local extra_env=()
    local arm_ts=1
    local arm_desc="fight decision bias ON (default weights)"
    case "$arm" in
        v5_bias_on)    ;;
        v5_bias_off)
            extra_env+=(Q2_BIAS_W_MARGIN=0 Q2_BIAS_W_EXPOSURE=0
                        Q2_BIAS_W_HAZARD=0 Q2_BIAS_W_OUTNUMBERED=0
                        Q2_BIAS_W_ESCAPE=0)
            arm_desc="fight decision bias OFF (all five Q2_BIAS_W_* weights zeroed => fight_bias==0)"
            ;;
        v5_bias_on_2x)
            arm_ts=2
            arm_desc="fight decision bias ON (default weights), TIMESCALE=2"
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
        echo "lane: wire v5 (hazard MOD channels, hit attribution, self_exposure, score frame) + fast-nearest;"
        echo "  opponents=$BOT_DESC; lattice_direction_coef=0 (all arms); pipelined collector NOT wired (serial collect_round)"
        echo "--- environment (secrets redacted) ---"
        printenv | grep -E '^(Q2_|LD_LIBRARY_PATH=|PYTHONPATH=)' | grep -v 'TOKEN' | sort
        for kv in "${extra_env[@]:-}"; do [ -n "$kv" ] && echo "$kv"; done
        for kv in "${client_extra[@]:-}"; do [ -n "$kv" ] && echo "$kv"; done
        echo "BOT_COUNT=$BOT_COUNT"
        echo "--- command ---"
        echo "python3 -u -m train.ppo --n_servers 1 --n_bots_per_server 4 --n_ml_bots 4" \
             "--n_steps 128 --n_epochs 2 --batch_size 512 --chunk_len 16" \
             "--total_steps $TOTAL_STEPS --lr 1e-5 --clip_eps 0.2 --vf_coef 0.1" \
             "--ent_coef 0.005 --max_grad_norm 0.5 --aux_coef 0.01" \
             "--aim_anchor_coef 0.02 --lattice_direction_coef 0" \
             "--map_name $MAP --map_change_episodes 0 --max_ep_steps 1000" \
             "--timelimit 0 --fraglimit 0 --timescale $arm_ts --save_every 8192" \
             "--seed $SEED --game_seed $SEED --deterministic 1" \
             "--reset_optimizer 1 --resume"
        echo "--- trainer output ---"
    } >> "$arm_log"

    (
        cd "$TREE"
        env "${extra_env[@]}" "${client_extra[@]}" \
            Q2_RUN_TAG="$run_tag" \
            Q2_CKPT_DIR="$ckpt_dir" \
            python3 -u -m train.ppo \
                --n_servers 1 --n_bots_per_server 4 --n_ml_bots 4 \
                --n_steps 128 --n_epochs 2 --batch_size 512 --chunk_len 16 \
                --total_steps "$TOTAL_STEPS" --lr 1e-5 --clip_eps 0.2 \
                --vf_coef 0.1 --ent_coef 0.005 --max_grad_norm 0.5 \
                --aux_coef 0.01 --aim_anchor_coef 0.02 \
                --lattice_direction_coef 0 \
                --map_name "$MAP" --map_change_episodes 0 --max_ep_steps 1000 \
                --timelimit 0 --fraglimit 0 --timescale "$arm_ts" --save_every 8192 \
                --seed "$SEED" --game_seed "$SEED" --deterministic 1 \
                --reset_optimizer 1 --resume
    ) >> "$arm_log" 2>&1
    echo "=== $run_tag finished: $(date -Is) ===" >> "$arm_log"
}

for arm in "${ARMS[@]}"; do
    echo "[ablation_v5_matrix] starting arm: $arm"
    run_arm "$arm"
    echo "[ablation_v5_matrix] arm complete: $arm"
done
stop_server
trap - EXIT INT TERM
echo "[ablation_v5_matrix] all requested arms complete"
