#!/usr/bin/env bash
# launch_cover_ab.sh — the cover learnability A/B (#24).
#
# Trains one arm on the v3 COVER maps (mltrainv3_*) with the SAME config as
# the matrix's D arm (ext-obs on, patient) which trains on the v2 NO-COVER
# maps (mltrain_*). Same seeds, so cover geometry is the only difference.
# Compare V3 vs D learnability (K/D climb, survival, map cards) to grade
# cover by survivability — the judge already said it's economy-neutral, so
# this is the decisive test. Own port slab + checkpoint dir; does not touch
# the baking matrix. Usage (on the WSL box): bash tools/launch_cover_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

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
R_RUNE_SWITCH=0.05
EOF
SHARED_ENV=$(echo "$SHARED_ENV" | tr '\n' ' ')

# Mirror D exactly (ext-on, aggression 0.3) — only the map pool differs.
tmux kill-session -t q2_ppo_V3 2>/dev/null
tmux new-session -d -s q2_ppo_V3 -c "$PWD" \
"env $SHARED_ENV Q2_EXT_OBS=1 Q2_RUN_TAG=V3 Q2_EXCHANGE_AGGRESSION=0.3 \
Q2_SV_PORT_BASE=30400 Q2_ML_PORT_BASE=30450 Q2_CKPT_DIR=checkpoints/V3 \
python3 -u -m train.ppo --n_servers 6 --n_bots_per_server 8 --n_ml_bots 4 \
--n_steps 64 --total_steps 40000000 --lr 1e-5 --ent_coef 0.002 \
--map_glob 'mltrainv3_*.bsp' --map_change_episodes 256 --fraglimit 30 --timescale 8 \
2>&1 | tee /tmp/q2_train.V3.log"
echo "launched q2_ppo_V3 (cover A/B: ext-on patient on v3-cover maps; compare to matrix D on v2)"
