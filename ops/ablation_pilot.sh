#!/usr/bin/env bash
# ablation_pilot.sh — lattice role-gate ablation pilot (2026-07-23 refactor).
#
# Runs the 4-arm pilot SEQUENTIALLY against ONE WSL-local lithium q2ded
# (loopback game port 28300, telemetry 28301).  Arms:
#   all_on        defaults (all lattice channels live)
#   reward_off    Q2_LATTICE_REWARD=0
#   obs_off       Q2_LATTICE_OBS=0
#   direction_off --lattice_direction_coef 0
#
# Every arm warm-starts from the immutable BC pin
# training-data/resume/ablation_pilot_bc_v2_00049152 (policy + 398-cell
# lattice at step 49,152, optimizer reset) with matched seed 7142026 and the
# deterministic controls required by AGENTS.md
# (--seed N --game_seed N --deterministic 1, Q2_ML_ASYNC=0).
# Horizon: --total_steps 100000, i.e. ~50k accepted transitions past the
# 49,152-step warm start, per arm.
#
# Usage:
#   ablation_pilot.sh             # all four arms, sequential
#   ablation_pilot.sh all_on      # just one arm (any subset of names)
#
# The telemetry token is read from the mode-0600 env file and NEVER printed;
# the per-arm run log records the full Q2_* environment minus *TOKEN* lines.
set -euo pipefail

TREE=/home/raymond/q2-rollout/q2-ml-bot
STAGING=/home/raymond/q2-network-client-staging-20260713
SERVER_ROOT="$STAGING/server-runtime"
CLIENT_ROOT="$STAGING/runtime"
SERVER_CFG=ml_network_local_28300.cfg
GAME_PORT=28300
TELEMETRY_PORT=28301
MAP=mltrain_00005200
SEED=7142026
TOTAL_STEPS=100000
TOKEN_ENV=/home/raymond/q2-rollout/local-pilot-telemetry.env
RESUME_PIN="$TREE/training-data/resume/ablation_pilot_bc_v2_00049152"
LOG_DIR="$TREE/logs/ablation_pilot"
mkdir -p "$LOG_DIR"

ALL_ARMS=(all_on reward_off obs_off direction_off)
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

# --- token (mode-checked, never echoed) -------------------------------------
if [ "$(stat -c '%a' "$TOKEN_ENV")" != "600" ]; then
    echo "refusing to source $TOKEN_ENV: not mode 0600" >&2
    exit 1
fi
source "$TOKEN_ENV"
: "${Q2_ML_CLIENT_TELEMETRY_TOKEN:?token env file missing Q2_ML_CLIENT_TELEMETRY_TOKEN}"
export Q2_ML_CLIENT_TELEMETRY_TOKEN

# --- server lifecycle (exact PIDs only; no pattern kills) --------------------
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

# --- common trainer environment ----------------------------------------------
export Q2_NETWORK_CLIENTS=4
export Q2_NETWORK_SERVER="127.0.0.1:$GAME_PORT"
export Q2_NETWORK_TELEMETRY_SERVER="127.0.0.1:$TELEMETRY_PORT"
export Q2_NETWORK_CLIENT_BINARY="$CLIENT_ROOT/quake2"
export Q2_NETWORK_CLIENT_ROOT="$CLIENT_ROOT"
export Q2_NETWORK_CLIENT_DATA_ROOT="$STAGING/client-data/ablation_pilot"
export Q2_NETWORK_HARNESS_PORT_BASE=39000
export Q2_NETWORK_QPORT_BASE=49000
export Q2_NETWORK_CLIENT_TIMEOUT=60
export Q2_NETWORK_ROUND_TIMEOUT=3
export Q2_NETWORK_MAX_REJECTED_ECHOES=16
export Q2_NETWORK_CLIENT_ID_PREFIX=ablation-pilot
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
    local run_tag="ablation_pilot_$arm"
    local ckpt_dir="$TREE/training-data/checkpoints/$run_tag"
    local arm_log="$LOG_DIR/$run_tag.log"
    local server_log="$LOG_DIR/${run_tag}_server.log"
    mkdir -p "$ckpt_dir"

    # Per-arm ablation switches.
    local extra_env=()
    local direction_coef=0.02
    case "$arm" in
        all_on)        ;;
        reward_off)    extra_env+=(Q2_LATTICE_REWARD=0) ;;
        obs_off)       extra_env+=(Q2_LATTICE_OBS=0) ;;
        direction_off) direction_coef=0 ;;
    esac

    # Fresh server per arm: identical initial map state for every arm.
    stop_server
    start_server "$server_log"

    {
        echo "=== $run_tag ==="
        echo "date: $(date -Is)"
        echo "arm: $arm  seed: $SEED  warm_start: $RESUME_PIN (step 49152, reset_optimizer 1)"
        echo "horizon: total_steps=$TOTAL_STEPS (~50k accepted transitions past warm start)"
        echo "server: 127.0.0.1:$GAME_PORT telemetry=127.0.0.1:$TELEMETRY_PORT map=$MAP cfg=$SERVER_CFG"
        echo "--- environment (secrets redacted) ---"
        printenv | grep -E '^(Q2_|LD_LIBRARY_PATH=|PYTHONPATH=)' | grep -v 'TOKEN' | sort
        echo "--- command ---"
        echo "python3 -u -m train.ppo --n_servers 1 --n_bots_per_server 4 --n_ml_bots 4" \
             "--n_steps 128 --n_epochs 2 --batch_size 512 --chunk_len 16" \
             "--total_steps $TOTAL_STEPS --lr 1e-5 --clip_eps 0.2 --vf_coef 0.1" \
             "--ent_coef 0.005 --max_grad_norm 0.5 --aux_coef 0.01" \
             "--aim_anchor_coef 0.02 --lattice_direction_coef $direction_coef" \
             "--map_name $MAP --map_change_episodes 0 --max_ep_steps 1000" \
             "--timelimit 15 --fraglimit 20 --timescale 1 --save_every 8192" \
             "--seed $SEED --game_seed $SEED --deterministic 1" \
             "--reset_optimizer 1 --resume"
        echo "--- trainer output ---"
    } >> "$arm_log"

    (
        cd "$TREE"
        env "${extra_env[@]}" \
            Q2_RUN_TAG="$run_tag" \
            Q2_CKPT_DIR="$ckpt_dir" \
            python3 -u -m train.ppo \
                --n_servers 1 --n_bots_per_server 4 --n_ml_bots 4 \
                --n_steps 128 --n_epochs 2 --batch_size 512 --chunk_len 16 \
                --total_steps "$TOTAL_STEPS" --lr 1e-5 --clip_eps 0.2 \
                --vf_coef 0.1 --ent_coef 0.005 --max_grad_norm 0.5 \
                --aux_coef 0.01 --aim_anchor_coef 0.02 \
                --lattice_direction_coef "$direction_coef" \
                --map_name "$MAP" --map_change_episodes 0 --max_ep_steps 1000 \
                --timelimit 15 --fraglimit 20 --timescale 1 --save_every 8192 \
                --seed "$SEED" --game_seed "$SEED" --deterministic 1 \
                --reset_optimizer 1 --resume
    ) >> "$arm_log" 2>&1
    echo "=== $run_tag finished: $(date -Is) ===" >> "$arm_log"
}

for arm in "${ARMS[@]}"; do
    echo "[ablation_pilot] starting arm: $arm"
    run_arm "$arm"
    echo "[ablation_pilot] arm complete: $arm"
done
stop_server
trap - EXIT INT TERM
echo "[ablation_pilot] all requested arms complete"
