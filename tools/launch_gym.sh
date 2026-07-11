#!/usr/bin/env bash
# launch_gym.sh — the rune gym: a controlled run that drills rune usage.
#
# Trains on the runegym_* maps (clustered "rune rack" of all 5 types + the
# mod's combat for natural health swings), so the bot learns the rune game —
# pick/switch the rune that maximizes win_margin (low→regen, healthy→strength,
# the chicken-tie vampire bridge) — in isolation from the main map entropy.
# Ext-obs ON (it must perceive rune flags + win_margin) and PATIENT temperament
# (rune mastery is about deliberate gear choices, not reckless pressing).
#
# Runs alongside the main matrix on its own port slab + checkpoint dir.
# Usage (on the WSL box):  bash tools/launch_gym.sh
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
R_RUNE_SWITCH=0.08
EOF
SHARED_ENV=$(echo "$SHARED_ENV" | tr '\n' ' ')

# R_RUNE_SWITCH bumped (0.05→0.08) in the gym — the switch decision is the
# whole point here, so weight it harder than in the main env.
tmux kill-session -t q2_gym 2>/dev/null
tmux new-session -d -s q2_gym -c "$PWD" \
"env $SHARED_ENV Q2_EXT_OBS=1 Q2_RUN_TAG=GYM Q2_EXCHANGE_AGGRESSION=0.3 \
Q2_SV_PORT_BASE=29900 Q2_ML_PORT_BASE=29950 Q2_CKPT_DIR=checkpoints/GYM \
python3 -u -m train.ppo --n_servers 4 --n_bots_per_server 8 --n_ml_bots 4 \
--n_steps 64 --total_steps 40000000 --lr 1e-5 --ent_coef 0.002 \
--map_glob 'runegym_*.bsp' --map_change_episodes 256 --fraglimit 30 --timescale 8 \
2>&1 | tee /tmp/q2_train.GYM.log"
echo "launched q2_gym  (rune gym: ext-obs on, patient, runegym_* maps, R_RUNE_SWITCH=0.08)"
echo "watch: tail -f /tmp/q2_train.GYM.log ; TB runs/ppo_GYM_*  (ext/rune_switch_mean, ext/rune_held_rate)"
