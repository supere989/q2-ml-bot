#!/bin/bash
# Q2 ML Training Service — fully automated, portable across AMD ROCm and NVIDIA CUDA hosts
PIDFILE=/tmp/q2_train.pids
LOGFILE=/tmp/q2_train.log
Q2ML_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="$Q2ML_DIR/runs"
TB_PORT=6006
TENSORBOARD_BIN="${TENSORBOARD_BIN:-/home/raymond/miniconda3/bin/tensorboard}"

# Detect GPU platform
HAS_AMD_DPM=0
[ -e /sys/class/drm/card1/device/power_dpm_force_performance_level ] && HAS_AMD_DPM=1
HAS_NVIDIA=0
command -v nvidia-smi >/dev/null 2>&1 || [ -e /usr/lib/wsl/lib/nvidia-smi ] && HAS_NVIDIA=1

GPU_DPM=/sys/class/drm/card1/device
SUDOPW="$(cat /home/raymond/sudo_auth.key 2>/dev/null)"

_gpu_cap() {
    if [ "$HAS_AMD_DPM" = "1" ] && [ -n "$SUDOPW" ]; then
        echo "$SUDOPW" | sudo -S bash -c "
            echo manual > $GPU_DPM/power_dpm_force_performance_level
            echo 1      > $GPU_DPM/pp_dpm_sclk
        " 2>/dev/null && echo "  GPU: AMD clock capped to DPM level 1"
    fi
}

_gpu_restore() {
    if [ "$HAS_AMD_DPM" = "1" ] && [ -n "$SUDOPW" ]; then
        echo "$SUDOPW" | sudo -S bash -c "
            echo auto > $GPU_DPM/power_dpm_force_performance_level
        " 2>/dev/null && echo "  GPU: AMD clock restored to auto"
    fi
}

# NVIDIA: ensure libcuda is findable when launched via SSH non-login session
if [ "$HAS_NVIDIA" = "1" ] && [ -d /usr/lib/wsl/lib ]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
fi

# Only set HSA override for AMD ROCm; harmless if present but explicit
TRAIN_ENV=""
if [ "$HAS_AMD_DPM" = "1" ]; then
    TRAIN_ENV="HSA_OVERRIDE_GFX_VERSION=9.0.0"
fi

# Spatial reward shaping is computed from existing observations, so it does not
# change checkpoint shape. Override these before invoking the script to tune it.
export Q2_SPATIAL_REWARD="${Q2_SPATIAL_REWARD:-1}"
export Q2_VOXEL_SIZE="${Q2_VOXEL_SIZE:-256}"
export R_VOXEL_NEW_CELL="${R_VOXEL_NEW_CELL:-0.02}"
export R_TACTICAL_ENGAGEMENT="${R_TACTICAL_ENGAGEMENT:-0.01}"
export R_AIM_ALIGNMENT="${R_AIM_ALIGNMENT:-0.05}"
export R_DAMAGE_DEALT="${R_DAMAGE_DEALT:-0.01}"
export R_KILL="${R_KILL:-5.0}"
export R_DAMAGE_TAKEN="${R_DAMAGE_TAKEN:-0.005}"
export R_DEATH="${R_DEATH:-3.0}"
export R_ITEM="${R_ITEM:-0.2}"
export Q2_POLICY_STATEFUL="${Q2_POLICY_STATEFUL:-0}"

cd "$Q2ML_DIR" || exit 1

case "$1" in
  start)
    echo "=== Starting Q2 ML training stack ===" | tee -a "$LOGFILE"
    tmux kill-session -t q2_ppo 2>/dev/null || true
    tmux kill-session -t q2_tb  2>/dev/null || true
    pkill -f "q2ded|start_train_env.sh|monitor_stats.py|train\.ppo|tensorboard" 2>/dev/null || true
    sleep 1

    _gpu_cap
    mkdir -p "$RUNS_DIR"

    # PPO trainer in its own tmux session — survives shell exits and WSL idle
    tmux new-session -d -s q2_ppo -c "$Q2ML_DIR" \
        "env $TRAIN_ENV python3 -u -m train.ppo \
            --n_servers 4 --n_bots_per_server 4 --total_steps 20000000 --resume \
            --lr 1e-5 --ent_coef 0.002 \
            --map_glob 'mltrain_*.bsp' --map_change_episodes 1 \
            2>&1 | tee $LOGFILE.ppo"

    # TensorBoard in its own tmux session
    tmux new-session -d -s q2_tb -c "$Q2ML_DIR" \
        "$TENSORBOARD_BIN --logdir $RUNS_DIR --port $TB_PORT --bind_all \
            2>&1 | tee $LOGFILE.tensorboard"

    # Capture the trainer python PID for status checks (tmux pid is the shell)
    sleep 2
    PPO_PID=$(pgrep -f "python3 -u -m train\.ppo" | head -1)
    TB_PID=$(pgrep -f "$TENSORBOARD_BIN --logdir" | head -1)
    echo -e "${PPO_PID:-?}\n${TB_PID:-?}" > "$PIDFILE"
    echo "Stack started in tmux." | tee -a "$LOGFILE"
    echo "  PPO log:     tail -f $LOGFILE.ppo   (or: tmux attach -t q2_ppo)"
    echo "  TensorBoard: http://localhost:$TB_PORT          (or: tmux attach -t q2_tb)"
    ;;

  stop)
    tmux kill-session -t q2_ppo 2>/dev/null || true
    tmux kill-session -t q2_tb  2>/dev/null || true
    pkill -f "q2ded|train\.ppo|tensorboard" 2>/dev/null || true
    rm -f "$PIDFILE"
    _gpu_restore
    echo "Stopped."
    ;;

  status)
    if [ -f "$PIDFILE" ]; then
        echo "=== Process status ==="
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                ps -p "$pid" -o pid,%cpu,%mem,cmd --no-headers
            else
                echo "PID $pid — not running"
            fi
        done < "$PIDFILE"
        echo ""
        echo "=== Recent PPO output ==="
        tail -10 "$LOGFILE.ppo" 2>/dev/null || echo "No PPO log yet"
    else
        echo "Not running (no pidfile)"
    fi
    ;;

  logs)
    tail -40 "$LOGFILE.ppo" 2>/dev/null || echo "No PPO log yet"
    ;;

  *)
    echo "Usage: $0 {start|stop|status|logs}"
    ;;
esac
