#!/usr/bin/env bash
# launch_matrix.sh — start the parallel ablation run matrix on the WSL box.
#
# Each run is a self-contained trainer in its own tmux session with a disjoint
# port slab (Q2_SV_PORT_BASE / Q2_ML_PORT_BASE), checkpoint dir (Q2_CKPT_DIR /
# Q2_RUN_TAG), and TensorBoard run name. The split-path Q2_EXT_OBS flag picks
# the policy input width: off → 206-dim (resumes the 1.4M checkpoint), on →
# 216-dim fresh policy that can see rune state + the inbound-damage vector.
#
# Run ON the WSL box (paths are box-local). Usage:
#   bash tools/launch_matrix.sh            # launch all runs in $RUNS
#   bash tools/launch_matrix.sh A B        # launch only the named runs
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Shared reward env (kept identical across runs so ablations are clean).
read -r -d '' SHARED_ENV <<'EOF'
Q2_POLICY_STATEFUL=1 Q2_SERVER_BOOT_WAIT=3
R_DAMAGE_DEALT=0.006 R_KILL=2.0 R_DEATH=0.75 R_DAMAGE_TAKEN=0.0015
R_SESSION_MEMORY_ENGAGEMENT=0.006 R_SESSION_MEMORY_OPPORTUNITY=0.010
R_SESSION_MEMORY_THREAT=0.008 R_SESSION_MEMORY_DEATH_AVERSION=0.012
R_SESSION_MEMORY_SELF_FIRE=0.018 R_THREAT_ENGAGEMENT=0.035 R_THREAT_AIM=0.050
R_THREAT_FIRE=0.045 R_THREAT_DAMAGE=0.012 R_THREAT_KILL=2.500 R_THREAT_IGNORE=0.045
R_SURVIVE_THREAT=0.002 R_SURVIVE_LOW_HEALTH=0.003 R_EPISODE_WIN=2.000
R_EPISODE_SURVIVAL=0.650 R_EPISODE_LOSS=1.250 R_EPISODE_IDLE=0.500
R_DAMAGE_PROX_AVERSION=0.004 R_OFFENSE_RUNE=0.004 R_SURVIVAL_RUNE=0.004
EOF
SHARED_ENV=$(echo "$SHARED_ENV" | tr '\n' ' ')

COMMON_ARGS="--n_bots_per_server 8 --n_ml_bots 4 --n_steps 64 \
--total_steps 40000000 --lr 1e-5 --ent_coef 0.002 \
--map_glob mltrain_*.bsp --map_change_episodes 256 --fraglimit 30 --timescale 8"

# Run table: TAG | EXT_OBS | N_SERVERS | SV_BASE | ML_BASE | AGGRESSION | EXTRA
# All fresh on the redesigned (damage-as-lens) reward — the resume checkpoint
# was regressing and is 206-dim anyway. Sweeps: ext-obs (B vs C) and
# temperament (A aggressive / B balanced / D patient). Disjoint port slabs.
declare -A RUN
RUN[A]="1 6 27910 27950 1.0 --map_seed 0"   # ext-on, AGGRESSIVE temperament
RUN[B]="1 6 28400 28450 0.6 --map_seed 0"   # ext-on, balanced (the main config)
RUN[C]="0 6 28900 28950 0.6 --map_seed 0"   # ext-OFF, balanced (isolates ext-obs vs B)
RUN[D]="1 6 29400 29450 0.3 --map_seed 0"   # ext-on, PATIENT temperament
RUNS=${RUNS:-"A B C D"}
[ "$#" -gt 0 ] && RUNS="$*"

for tag in $RUNS; do
    spec="${RUN[$tag]:-}"
    [ -z "$spec" ] && { echo "unknown run '$tag'"; continue; }
    read -r ext nsrv svb mlb agg extra <<<"$spec"
    sess="q2_ppo_${tag}"
    # WARM=1 resumes each run from its own checkpoints/<tag> (preserve progress
    # while adding the new resource); otherwise fresh.
    resume=""
    if [ "${WARM:-0}" = "1" ] && ls "checkpoints/${tag}"/policy_[0-9]*.pt >/dev/null 2>&1; then
        resume="--resume"
    fi
    tmux kill-session -t "$sess" 2>/dev/null
    tmux new-session -d -s "$sess" -c "$PWD" \
"env $SHARED_ENV Q2_EXT_OBS=$ext Q2_RUN_TAG=$tag Q2_EXCHANGE_AGGRESSION=$agg \
Q2_SV_PORT_BASE=$svb Q2_ML_PORT_BASE=$mlb Q2_CKPT_DIR=checkpoints/${tag} \
python3 -u -m train.ppo --n_servers $nsrv $COMMON_ARGS $extra $resume \
2>&1 | tee /tmp/q2_train.${tag}.log"
    echo "launched $sess  ext_obs=$ext aggression=$agg sv=$svb ml=$mlb resume=${resume:-fresh}  ($extra)"
done
echo "watch:  tmux ls ; tail -f /tmp/q2_train.<TAG>.log ; TensorBoard runs/ppo_<TAG>_*"
