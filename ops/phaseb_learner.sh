#!/usr/bin/env bash
# phaseb_learner.sh — Phase B embedded PPO learner/coordinator (2026-07-25).
#
# From-zero infrastructure validation: no resume pin (publishes policy
# version 0), quorum=2 network-native rollout workers x 8 envs each
# (total_venvs=16), n_steps=128/worker, batch_size 2048.
# See docs/PHASE-B-MULTILANE-2026-07-25.md.
set -euo pipefail

TREE=/home/raymond/q2-rollout/q2-ml-bot
SECRET_DIR=/home/raymond/q2-rollout
MANIFEST="$TREE/runtime-manifest-phaseb.json"
RUN_TAG=phaseb_zero_v1
LOG_DIR="$TREE/logs/phaseb"
mkdir -p "$LOG_DIR" "$TREE/training-data/checkpoints/$RUN_TAG"

for f in "$SECRET_DIR/phaseb-rollout-token.env" "$SECRET_DIR/phaseb-attestation-key.env"; do
    [ "$(stat -c '%a' "$f")" = "600" ] || { echo "refusing: $f not mode 0600" >&2; exit 1; }
    source "$f"
done
export Q2_ROLLOUT_TOKEN Q2_ROLLOUT_ATTESTATION_KEY
: "${Q2_ROLLOUT_TOKEN:?missing}"; : "${Q2_ROLLOUT_ATTESTATION_KEY:?missing}"

export Q2_DISTRIBUTED_LEARNER=1
export Q2_EXT_OBS=1
export Q2_RUST_LATTICE=1
export Q2_POLICY_STATEFUL=1
export Q2_ML_ASYNC=0
export Q2_RUST_EXTENSION_PATH=/home/raymond/q2-rollout/python/q2_lattice_rs.so
export PYTHONPATH=/home/raymond/q2-rollout/python${PYTHONPATH:+:$PYTHONPATH}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export Q2_SOURCE_REVISION=q2-ml-bot-v5-phaseb-2026-07-25

export Q2_ROLLOUT_BIND=127.0.0.1
export Q2_ROLLOUT_PORT=38888
export Q2_ROLLOUT_RUNTIME_MANIFEST="$MANIFEST"
export Q2_ROLLOUT_ATTESTATION_KEY_ENV=Q2_ROLLOUT_ATTESTATION_KEY
export Q2_ROLLOUT_QUORUM=2
export Q2_ROLLOUT_ENVS_PER_WORKER=8

export Q2_RUN_TAG="$RUN_TAG"
export Q2_CKPT_DIR="$TREE/training-data/checkpoints/$RUN_TAG"
export Q2_RUNS_DIR="$TREE/training-data/runs"
export Q2_LATTICE_DIR=/home/raymond/q2-rollout/runtime/baseq2/maps

{
    echo "=== phaseb learner ($RUN_TAG) ==="
    echo "date: $(date -Is)"
    echo "init: from zero (no resume; publishes version 0)"
    echo "quorum: 2 workers x 8 envs = 16; n_steps=128/worker; batch_size 2048 (min for recurrent anchor)"
    echo "manifest: $MANIFEST"
    echo "--- environment (secrets redacted) ---"
    printenv | grep -E '^(Q2_|LD_LIBRARY_PATH=|PYTHONPATH=|CUBLAS|PYTHONHASHSEED)' \
        | grep -vE 'TOKEN|KEY' | sort
    echo "--- trainer output ---"
} >> "$LOG_DIR/${RUN_TAG}.log"

cd "$TREE"
exec python3 -u -m train.ppo \
    --n_steps 128 --n_epochs 2 --batch_size 2048 --chunk_len 16 \
    --total_steps 2000000 --lr 1e-4 --clip_eps 0.2 \
    --vf_coef 0.1 --ent_coef 0.005 --max_grad_norm 0.5 \
    --aux_coef 0.01 --aim_anchor_coef 0.02 \
    --lattice_direction_coef 0 \
    --map_name mllive_44987431 --map_change_episodes 0 --max_ep_steps 1000 \
    --timelimit 0 --fraglimit 0 --timescale 1 --save_every 8192 \
    --seed 7142026 --deterministic 1 \
    >> "$LOG_DIR/${RUN_TAG}.log" 2>&1
